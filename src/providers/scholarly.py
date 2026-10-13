"""Open-access full text for a DOI or arXiv link.

The ``scholarly`` read provider (``ScholarlyRead``), a ``UrlSpecificReader``: it
accepts a ``doi.org`` link or an ``arxiv.org`` ``/abs/`` or ``/pdf/`` link, and
the read pipeline offers it those before the probe. A plain ``doi.org`` link
fails on every generic reader — the publisher's landing page is a bot wall —
while a legal open-access copy of the paper usually lives in a repository. This
reader finds that copy and returns its text:

- an arXiv id (an ``arxiv.org`` link or an arXiv DOI ``10.48550/arXiv.<id>``,
  which Unpaywall does not index) → the paper's PDF from ``arxiv.org/pdf/<id>``;
- any other DOI → Unpaywall's OA locations, tried repository-first: a PubMed
  Central location becomes Europe PMC's JATS full-text XML (abstract + body),
  any other location's ``url_for_pdf`` becomes that PDF. The publisher landing
  ``url`` is never fetched — it is the wall.

Unpaywall wants an identifying email, so without ``unpaywall_email`` the reader
does not accept a (non-arXiv) DOI at all and the url goes to the generic chain.

Unlike the API-based readers, every GET here is a third-party url fetched from
OUR process, so the pipeline hands this reader the SSRF-guarded client (see
``fetches_target_locally``). A DOI with no OA copy raises ``ProviderError`` and
falls through to the generic chain, so the normal readers can still try the
landing page.
"""

from __future__ import annotations

import re
from urllib.parse import quote, unquote, urlsplit
import xml.etree.ElementTree as ET

import httpx

from src import failure_reason
from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.pdf import NO_TEXT_LAYER_NOTICE, extract_pdf_text
from src.providers.registry import register

_DOI_HOSTS = frozenset({"doi.org", "dx.doi.org", "www.doi.org"})

_UNPAYWALL_API = "https://api.unpaywall.org/v2/"
_EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
_ARXIV_PDF = "https://arxiv.org/pdf/"

# Provider name for request_with_retry's error messages.
_PROVIDER = "scholarly"

# arXiv's own DOI prefix (compared lowercased): `10.48550/arXiv.<id>`.
_ARXIV_DOI_PREFIX = "10.48550/arxiv."

# A PubMed Central / Europe PMC article url. Unpaywall writes its PMC locations
# as `.../pmc/articles/<digits>` WITHOUT the `PMC` prefix, so it is optional,
# and its Europe PMC ones both with and without a `/pmc` segment.
_PMC_RE = re.compile(
    r"^https?://(?:www\.ncbi\.nlm\.nih\.gov/pmc|pmc\.ncbi\.nlm\.nih\.gov|europepmc\.org(?:/pmc)?)"
    r"/articles/(?:pmc)?(\d+)",
    re.IGNORECASE,
)

# The two shapes a candidate OA location resolves to (see `_fetch_target`).
_PDF = "pdf"
_XML = "xml"

# JATS block elements publishers nest inside a `<p>`. Their text is not part of
# that paragraph: a table would be glued into it, and the `<p>`s inside a figure
# caption or a list are reached by the paragraph walk on their own.
_NESTED_BLOCKS = frozenset(
    {
        "fig",
        "fig-group",
        "table-wrap",
        "table-wrap-group",
        "list",
        "disp-quote",
        "boxed-text",
        "disp-formula",
        "supplementary-material",
    }
)


def doi_from_url(url: str) -> str | None:
    """The DOI of a ``doi.org`` url — the url-decoded path, no leading slash — or ``None``.

    Recognises ``doi.org`` / ``dx.doi.org`` / ``www.doi.org`` with a path that
    is a DOI (starts with ``10.``); anything else — a non-DOI doi.org page, a
    publisher url, a malformed url — is ``None``; this never raises.
    """
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    if host not in _DOI_HOSTS:
        return None
    doi = unquote(parts.path).lstrip("/").strip()
    return doi if doi.startswith("10.") else None


def arxiv_id_from_url(url: str) -> str | None:
    """The arXiv id of an arXiv url, or ``None``.

    Recognises an ``arxiv.org`` ``/abs/`` or ``/pdf/`` url (the id is the path
    after the marker, a trailing ``.pdf`` dropped) and a ``doi.org`` url of an
    arXiv DOI ``10.48550/arXiv.<id>``. Anything else is ``None``; this never
    raises.
    """
    doi = doi_from_url(url)
    if doi is not None:
        if doi.lower().startswith(_ARXIV_DOI_PREFIX):
            return doi[len(_ARXIV_DOI_PREFIX) :] or None
        return None
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    if not host.endswith("arxiv.org"):
        return None
    for marker in ("/abs/", "/pdf/"):
        if marker in parts.path:
            ident = parts.path.split(marker, 1)[1].strip("/")
            if ident.lower().endswith(".pdf"):
                ident = ident[:-4]
            return ident or None
    return None


def _ordered_locations(payload: dict) -> list[dict]:
    """The OA locations to try, repository ones first (stable otherwise).

    ``oa_locations`` plus ``best_oa_location`` when it is not already among them;
    then every ``host_type == "repository"`` location moved to the front, the
    rest kept in their original order.
    """
    locations = [loc for loc in payload.get("oa_locations") or [] if isinstance(loc, dict)]
    best = payload.get("best_oa_location")
    if isinstance(best, dict) and best not in locations:
        locations.append(best)
    repositories = [loc for loc in locations if loc.get("host_type") == "repository"]
    rest = [loc for loc in locations if loc.get("host_type") != "repository"]
    return repositories + rest


def _fetch_target(location: dict) -> tuple[str, str] | None:
    """The one fetch target a location resolves to, or ``None`` to skip it.

    A PubMed Central location → Europe PMC's JATS full-text XML; otherwise a
    location with a ``url_for_pdf`` → that PDF. The landing ``url`` itself is
    never fetched — on a DOI it is usually the publisher's bot wall.
    """
    match = _PMC_RE.match(location.get("url") or "")
    if match:
        return _XML, f"{_EUROPEPMC_API}PMC{match.group(1)}/fullTextXML"
    pdf = location.get("url_for_pdf")
    if pdf:
        return _PDF, pdf
    return None


def _local(tag: str) -> str:
    """An element tag without its XML namespace (``{ns}p`` → ``p``)."""
    return tag.rsplit("}", 1)[-1]


def _own_text(para: ET.Element) -> str:
    """A ``<p>``'s own text: inline markup kept as words, nested blocks left out."""
    parts = [para.text or ""]
    for child in para:
        if _local(child.tag) not in _NESTED_BLOCKS:
            parts.append("".join(child.itertext()))
        parts.append(child.tail or "")
    return " ".join("".join(parts).split())


def _jats_text(body: bytes) -> str:
    """Abstract + body paragraph text of a JATS full-text XML document.

    Gathers every ``<p>`` under an ``<abstract>`` or ``<body>`` element, each
    from its own text only (see ``_own_text``), whitespace collapsed. ``""``
    when the bytes do not parse or carry no such text.
    """
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return ""
    paragraphs: list[str] = []
    for element in root.iter():
        if _local(element.tag) not in ("abstract", "body"):
            continue
        for para in element.iter():
            if _local(para.tag) == "p":
                text = _own_text(para)
                if text:
                    paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _extract(kind: str, content: bytes) -> str:
    """Text of a fetched candidate (``""`` when it yields none).

    A PDF without a text layer or an XML with no abstract/body counts as none.
    ``extract_pdf_text`` raises ``ProviderError`` on bytes that are not a PDF;
    the caller treats that like a transport error and moves to the next target.
    """
    if kind == _PDF:
        text = extract_pdf_text(content)
        return "" if text == NO_TEXT_LAYER_NOTICE else text.strip()
    return _jats_text(content)


def _doi_header(title: str | None, doi: str, location: dict, fetched_url: str) -> str:
    """A short header: the title (if any), the DOI, and where the text came from.

    The DOI names the published paper, but the text may be a preprint or an
    accepted manuscript, so the source (the location's ``url``, else the url
    fetched) is stated with Unpaywall's ``version`` when the location has one
    (submittedVersion / acceptedVersion / publishedVersion).
    """
    lines = [f"# {title}"] if title else []
    lines.append(f"DOI: {doi}")
    source = location.get("url") or fetched_url
    version = location.get("version")
    lines.append(f"Source: {source} ({version})" if version else f"Source: {source}")
    return "\n".join(lines) + "\n\n"


@register("scholarly")
class ScholarlyRead:
    """Read a DOI or arXiv url as the paper's legal open-access full text (no key needed)."""

    # Unlike the API-based readers, this one GETs third-party URLs from our own
    # process, so the pipeline must hand it the SSRF-guarded client.
    fetches_target_locally = True

    # `falls_through` left at its default True: a DOI with no OA copy should fall
    # through to the generic chain so the normal readers can still try the page.

    def __init__(self, config: ProviderConfig) -> None:
        self.name = config.name
        self.proxy = config.proxy
        self._config = config
        self._email = config.options.get("unpaywall_email")

    def accepts(self, url: str) -> bool:
        if arxiv_id_from_url(url) is not None:
            return True
        # Without the email Unpaywall cannot be asked, so a DOI is left to the
        # generic chain instead of costing a guaranteed failure here.
        return bool(self._email) and doi_from_url(url) is not None

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        arxiv_id = arxiv_id_from_url(url)
        if arxiv_id is not None:
            return await self._read_arxiv(client, arxiv_id)
        return await self._read_doi(client, doi_from_url(url))

    async def _read_arxiv(self, client: httpx.AsyncClient, arxiv_id: str) -> str:
        try:
            response = await request_with_retry(
                client,
                "GET",
                f"{_ARXIV_PDF}{arxiv_id}",
                retries=self._config.retries,
                provider=_PROVIDER,
            )
        except ProviderError as exc:
            if exc.status is None:
                raise
            # This url IS the paper, so its status is the target's own.
            raise ProviderError(
                f"scholarly: arXiv returned HTTP {exc.status}",
                reason=failure_reason.for_target_status(exc.status),
            ) from exc
        text = extract_pdf_text(response.content)
        if text == NO_TEXT_LAYER_NOTICE:
            raise ProviderError(
                f"scholarly: arXiv PDF has no text layer (arXiv:{arxiv_id})",
                reason=failure_reason.EMPTY,
            )
        return f"# arXiv:{arxiv_id}\n\n" + text

    async def _read_doi(self, client: httpx.AsyncClient, doi: str) -> str:
        try:
            response = await request_with_retry(
                client,
                "GET",
                # Old DOIs may hold `#`, `?` or `<>`; escape all but the `/`.
                f"{_UNPAYWALL_API}{quote(doi, safe='/')}",
                retries=self._config.retries,
                provider=_PROVIDER,
                params={"email": self._email},
            )
        except ProviderError as exc:
            if exc.status == 404:
                # A 404 from Unpaywall's own API means "not in its index", which
                # says nothing about the paper's page: EMPTY, not NOT_FOUND.
                raise ProviderError(
                    f"scholarly: Unpaywall has no OA record for {doi}",
                    reason=failure_reason.EMPTY,
                ) from exc
            if exc.status == 422:
                # Unpaywall refuses a placeholder address (any @example.com,
                # like the .env.example one) with "Please use your own email".
                raise ProviderError(
                    "scholarly: Unpaywall rejected UNPAYWALL_EMAIL (HTTP 422)",
                    reason=failure_reason.OTHER,
                ) from exc
            raise
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"scholarly: invalid Unpaywall response: {exc}", reason=failure_reason.OTHER
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderError(
                "scholarly: invalid Unpaywall response: not a JSON object",
                reason=failure_reason.OTHER,
            )

        title = payload.get("title")
        # Two locations may resolve to one target (the same PMC id on ncbi and on
        # Europe PMC): a target that already failed is not fetched again.
        seen: set[str] = set()
        for location in _ordered_locations(payload):
            target = _fetch_target(location)
            if target is None or target[1] in seen:
                continue
            kind, target_url = target
            seen.add(target_url)
            try:
                # A broken candidate (transport error, blocked url, bytes that
                # are not a PDF) is skipped, not fatal — the next target may work.
                resp = await client.get(target_url)
                text = _extract(kind, resp.content)
            except (httpx.HTTPError, ProviderError):
                continue
            if text:
                return _doi_header(title, doi, location, target_url) + text
        raise ProviderError(
            "scholarly: no open-access full text found", reason=failure_reason.EMPTY
        )
