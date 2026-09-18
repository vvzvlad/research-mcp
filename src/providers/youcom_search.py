"""You.com Search API provider (ydc-index.io web index).

API (verified against the official docs 2026-09-18,
https://you.com/docs/api-reference/search — the same page as clean Markdown at
https://you.com/docs/api-reference/search.md): POST
``https://ydc-index.io/v1/search``, ``Content-Type: application/json``, header
``X-API-Key``; JSON body ``{"query", "count", "offset", "country", "language",
...}`` → ``{"results": {"web": [...], "news": [...]}, "metadata": {...}}``.
Every ``results.web[]`` item carries ``url`` / ``title`` / ``description`` /
``snippets`` (plus thumbnail_url, page_age, contents, ...).

The snippet lives in ``description`` here ("A brief description of the content
of the search result") — NOT ``content`` as in searxng, nor ``snippet`` as in
serper. Beside it sits ``snippets``, an ARRAY of "short, keyword-centered text
fragments ... built for skimming"; it is used only when ``description`` comes
back empty. ``results`` and each section inside it are marked optional in the
schema, so a response without ``web`` is a normal empty answer, not a failure.
``news`` is ignored: this is the web-search half of the pipeline.

Paging: ``offset`` is a 0-based PAGE index — "The ``offset`` is calculated in
multiples of ``count``" — with the documented range ``0 <= offset <= 9``. Same
shape as brave's ``offset``, so page 10 is the deepest page served.

``count`` ("the maximum number of search results to return per section") has no
numeric bound in the API reference; the pricing section of
https://you.com/docs/quickstart.md bills the Web Search API at "$5.00 per 1,000
calls (up to 100 results per call)", which is where the cap below comes from.

Rate limit: 10 requests/second for self-serve accounts
(https://you.com/docs/rate-limits.md). That is loose enough that this provider
needs no local throttle — unlike brave, whose free plan allows 1 req/s.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

YOUCOM_ENDPOINT = "https://ydc-index.io/v1/search"

# Most results one call can return, from the pricing line quoted above ("up to
# 100 results per call"). The parameter reference states no bound of its own.
YOUCOM_COUNT_MAX = 100

# Deepest value `offset` accepts: the reference says "Range 0 <= offset <= 9",
# i.e. page 10 is the last page of a given result set. The tool's `page`
# argument has no upper bound of its own, so this provider refuses deeper pages
# itself — see `search` for why it refuses rather than clamping.
YOUCOM_OFFSET_MAX = 9

# Every value the `language` enum accepts, verbatim from the parameter reference
# (read 2026-09-18). BCP 47 in the vendor's own uppercase spelling — note it is
# not plain ISO 639-1: Chinese exists only as script variants (ZH-HANS /
# ZH-HANT), Portuguese only as regional ones (PT-BR / PT-PT), and English has
# both EN and EN-GB.
_YOUCOM_LANGS = frozenset(
    {
        "AR", "EU", "BN", "BG", "CA", "ZH-HANS", "ZH-HANT", "HR", "CS", "DA",
        "NL", "EN", "EN-GB", "ET", "FI", "FR", "GL", "DE", "EL", "GU",
        "HE", "HI", "HU", "IS", "IT", "JA", "KN", "KO", "LV", "LT",
        "MS", "ML", "MR", "NB", "PL", "PT-BR", "PT-PT", "PA", "RO", "RU",
        "SR", "SK", "SL", "ES", "SV", "TA", "TE", "TH", "TR", "UK",
        "VI",
    }
)  # fmt: skip


def _youcom_lang(value: str) -> str | None:
    """Map a caller language tag onto a code the `language` enum accepts.

    ``language`` is whatever the LLM passed — ``ru-RU``, ``en_US``, ``EN``,
    ``" ru "`` — while You.com validates the parameter against the closed list
    above (a value outside it is a 422). So normalise separator and case, try
    the full tag first (that keeps regional targeting where the enum has it:
    ``EN-GB``, ``PT-BR``, ``ZH-HANS``), then fall back to the bare language
    subtag.

    Returns None when nothing matches: omitting ``language`` lets You.com apply
    its own default (``EN``), which beats a guaranteed 422 that would drop us
    from the merge. Tags whose bare subtag is not in the enum — ``zh``, ``pt`` —
    land here too; no alias table is guessed on top of the documented list.
    """
    code = value.strip().replace("_", "-").upper()
    if code in _YOUCOM_LANGS:
        return code
    short = code.split("-")[0]
    return short if short in _YOUCOM_LANGS else None


@register("youcom_search")
class YouComSearch:
    """Web search via the You.com Search API (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("youcom_search requires an api_key")
        self.name = config.name
        self.proxy = config.proxy
        self._config = config

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int,
        page: int,
        language: str | None,
    ) -> list[SearchResult]:
        # Deeper than You.com can serve → refuse, do not clamp to the last page.
        # Clamping would spend a paid call just to hand back page 10 again, and
        # pages 11, 12, ... would each re-inject those same hits into the merge
        # (dedup runs within a single search() call, never across calls).
        if page > YOUCOM_OFFSET_MAX + 1:
            raise ProviderError(
                f"{self.name}: page {page} is beyond you.com's depth "
                f"(max {YOUCOM_OFFSET_MAX + 1})"
            )

        body: dict[str, Any] = {
            "query": query,
            "count": max(1, min(num_results, YOUCOM_COUNT_MAX)),
            # `offset` is a 0-based PAGE index, not a result offset. max(0, ...)
            # covers page <= 0: nothing upstream clamps `page` (src/server.py
            # clamps only num_results), and a negative offset is out of range.
            "offset": max(0, page - 1),
        }
        if language:
            # Unlike serper's Google-compatible `hl`, which takes a regional tag
            # as-is, `language` is checked against a closed enum — see
            # _youcom_lang. An unmappable tag means no `language` at all (the
            # API default), never a guessed value.
            lang = _youcom_lang(language)
            if lang:
                body["language"] = lang
        # `country` is deliberately NOT sent, even though the enum is documented
        # and includes RU: the tool contract has no country input, and deriving
        # one from `language` would be wrong for languages spoken in many
        # countries (en, es, ru, ...). Same call as in brave.py.
        response = await request_with_retry(
            client,
            "POST",
            YOUCOM_ENDPOINT,
            json=body,
            headers={
                "X-API-Key": self._config.api_key,
                "Content-Type": "application/json",
            },
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        # Neither `results` nor `results.web` is guaranteed by the schema: a
        # query that matched nothing (or only the news section) is a normal
        # empty answer.
        results = data.get("results")
        web = results.get("web") if isinstance(results, dict) else None
        if not isinstance(web, list):
            return []
        out: list[SearchResult] = []
        for item in web:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            if not url:
                continue
            # You.com calls the snippet "description" (one string). `snippets`
            # holds the same page's keyword fragments as a list and serves as a
            # fallback when the description is missing.
            snippet = (item.get("description") or "").strip()
            if not snippet:
                fragments = item.get("snippets")
                if isinstance(fragments, list):
                    snippet = " ".join(
                        part.strip()
                        for part in fragments
                        if isinstance(part, str) and part.strip()
                    )
            out.append(
                SearchResult(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=snippet,
                    source=self.name,
                )
            )
        return out
