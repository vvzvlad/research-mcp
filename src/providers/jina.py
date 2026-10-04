"""Jina AI Reader provider.

API (verified 2026-08-18 from https://docs.jina.ai/; the JSON answer measured
live 2026-10-04): GET ``https://r.jina.ai/{url}``. Headers used here:

- ``Accept: application/json`` switches the answer to a JSON envelope,
  ``{"code": 200, "status": 20000, "data": {...}, "meta": {...}}``. ``data``
  holds ``title``, ``url``, ``content`` (the Markdown body, WITHOUT the
  "Title:/URL Source:/Markdown Content:" header block of the plain-text
  answer), ``httpStatus`` (an int: the TARGET site's status) and an optional
  ``warning`` string.
- ``X-Return-Format: markdown`` asks for Markdown explicitly.
- ``Authorization: Bearer {key}`` is OPTIONAL — keyless works at a lower rate
  limit — so this instance is always enabled.
- ``X-Token-Budget: <int>`` caps the tokens one request may spend; EXCEEDING
  the budget FAILS the request, which in our pipeline just moves on to the
  next read provider — an acceptable trade for a bounded bill.
- ``X-Respond-With: readerlm-v2`` routes the conversion through the
  ReaderLM-v2 small LM: much better Markdown on complex/cluttered pages, at
  3x the token cost.
- ``X-Engine: browser`` picks the highest-quality (slowest) fetch engine.
- ``x-proxy: auto`` fetches through jina's residential address pool instead of
  its datacenter one, at 5x the token cost.
- ``X-Respond-With: jina-ocr-v1`` converts by OCR instead of by LM, at 40x —
  used for .pdf urls only.

jina answers HTTP 200 even when the site refused it, and converts the refusal
page into ``content`` as if it were the article; only ``data.httpStatus`` and
``data.warning`` tell (see ``_refusal``). Measured 2026-10-04: example.com →
httpStatus 200, no warning; a missing GitHub repo → 404 with "Target URL
returned error 404: Not Found"; an ozon.ru category → 403; a wildberries.ru
search → 498; a scispace.com CAPTCHA wall → 405 with a warning naming the
CAPTCHA. The readerlm-v2 + browser tier and ``x-proxy: auto`` answer in the
same envelope; jina-ocr-v1 was not measured (40x price) and is assumed to.

The last four headers form the keyed-only ESCALATION LADDER in ``read``: the
cheap plain conversion goes first, and only a thin/empty answer climbs the
ladder, one step at a time, stopping as soon as a step returns enough text. See
``_ESCALATIONS`` for what each step buys and what it costs. A missing page or a
CAPTCHA wall fails at once; any other refusal of the plain answer goes straight
to the steps on the residential exit; a refusal or a failure on any step ends
the ladder.
"""

from __future__ import annotations

from typing import NamedTuple
from urllib.parse import urlsplit

import httpx
from loguru import logger

from src import failure_reason
from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.registry import register

JINA_READER_BASE = "https://r.jina.ai/"

# The conversion tier every escalation shares: the ReaderLM-v2 small LM on the
# highest-quality fetch engine.
_HEAVY_TIER = {"X-Respond-With": "readerlm-v2", "X-Engine": "browser"}

# The keyed escalation ladder, tried in this order, each step only if everything
# before it STILL came back thinner than fallback_min_chars. An entry is
# (log label, extra headers, pdf_only). After a refused plain answer only the
# steps that carry ``x-proxy`` run: a parsing step would meet the same refusal.
# The rules shared by every step — one unretried request, longest answer wins,
# a refusal or a failure ends the ladder, log line only after jina's JSON came
# back — live in ``read``.
_ESCALATIONS: tuple[tuple[str, dict[str, str], bool], ...] = (
    # Step 1 — ReaderLM-v2 conversion on the browser engine. Fixes PARSING:
    # cluttered or JS-heavy pages the plain converter turns into noise. 3x
    # tokens.
    ("readerlm-v2 escalation", _HEAVY_TIER, False),
    # Step 2 — the same tier plus ``x-proxy: auto``, which moves the fetch onto
    # jina's residential address pool. That cures blocks by IP REPUTATION: sites
    # that refuse jina's datacenter ranges but serve an ordinary-looking
    # consumer address. It does NOT break a Cloudflare challenge — verified
    # live, so do not expect more of it than a different exit IP. Price: exactly
    # 5x the plain tokens (measured).
    ("residential-proxy escalation", {**_HEAVY_TIER, "x-proxy": "auto"}, False),
    # Step 3, .pdf urls ONLY — swap the conversion model for jina-ocr-v1, the
    # rest as in step 2. This is for scans (a PDF with no text layer) and for
    # PDFs served behind redirects, where no amount of HTML conversion helps.
    # Reaching a scan at all depends on Pipeline.read: a text-free PDF used to
    # return pypdf's "no text layer" notice as a success and never entered the
    # read chain. That branch now falls through to us and keeps the notice only
    # as a last resort — do not turn it back into an early return. Note the gate
    # is the URL PATH, so this step sees a scan only when the url ends in .pdf;
    # one recognised by Content-Type or %PDF magic alone reaches the chain but
    # not this step.
    # Price: 40x tokens, the most expensive step there is — hence last, and
    # hence never on a non-pdf url. The arithmetic, at X-Token-Budget=100000:
    # worst case 100000 * 40 = ~4M tokens, i.e. ~$0.20 for one page on the $50
    # per 1 billion tokens pack.
    (
        "jina-ocr-v1 escalation",
        {**_HEAVY_TIER, "X-Respond-With": "jina-ocr-v1", "x-proxy": "auto"},
        True,
    ),
)


class _Refusal(NamedTuple):
    """The site refused jina: the error message, its category, and ``final``."""

    message: str
    # A ``src.failure_reason`` constant.
    category: str
    # True when no escalation step can cure it: a page that does not exist, or
    # a CAPTCHA wall (the residential exit does not break challenges, see
    # ``_ESCALATIONS``).
    final: bool


class _Answer(NamedTuple):
    """One jina answer: the stripped ``data.content`` and the site's refusal."""

    text: str
    refusal: _Refusal | None


def _refusal(data: dict) -> _Refusal | None:
    """Why the answer's ``data`` is the site's refusal rather than the page, or None.

    Read from the fields only — ``httpStatus`` and ``warning`` — never from the
    Markdown, so an article that merely quotes an error is not a refusal.
    """
    status = data.get("httpStatus")
    warning = data.get("warning")
    if status in (404, 410):
        return _Refusal(f"target page returned HTTP {status}", failure_reason.NOT_FOUND, True)
    if isinstance(warning, str) and "CAPTCHA" in warning:
        return _Refusal("bot protection (CAPTCHA wall)", failure_reason.BOT_PROTECTION, True)
    if isinstance(status, int) and status >= 400:
        return _Refusal(
            f"target page returned HTTP {status}",
            failure_reason.for_target_status(status),
            False,
        )
    return None


def _is_pdf_url(url: str) -> bool:
    """True when the url's PATH ends in ``.pdf``, case-insensitively.

    The PATH only: a query or fragment must not decide this, so
    ``/page?file=a.pdf`` is not a pdf while ``/doc.PDF?v=2`` is.
    """
    return urlsplit(url).path.lower().endswith(".pdf")


@register("jina")
class JinaRead:
    """Read a page as Markdown via Jina Reader (api_key optional)."""

    def __init__(self, config: ProviderConfig) -> None:
        self.name = config.name
        self.proxy = config.proxy
        self._config = config

    async def _get(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        *,
        retries: int,
    ) -> _Answer:
        """One reader request, parsed from jina's JSON envelope."""
        response = await request_with_retry(
            client,
            "GET",
            f"{JINA_READER_BASE}{url}",
            retries=retries,
            provider=self.name,
            headers=headers,
        )
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            raise ProviderError(f"{self.name}: invalid JSON response")
        content = data.get("content")
        text = content.strip() if isinstance(content, str) else ""
        if text:
            # The plain-text answer led with the page's title, source url and
            # publication date; JSON mode moves them into fields, so put them
            # back in front of the body. Only in front of a body: a header
            # alone must not turn an empty answer into a non-empty one.
            header = [
                f"{label}: {value}"
                for label, value in (
                    ("Title", data.get("title")),
                    ("URL Source", data.get("url")),
                    ("Published Time", data.get("publishedTime")),
                )
                if isinstance(value, str) and value.strip()
            ]
            if header:
                text = "\n\n".join([*header, f"Markdown Content:\n{text}"])
        return _Answer(text, _refusal(data))

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        headers = {"Accept": "application/json", "X-Return-Format": "markdown"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
            # The budget only matters in keyed mode: keyless requests are not
            # billed, so the header would add a failure mode there for free.
            budget = int(self._config.options.get("token_budget", "0"))
            if budget > 0:
                headers["X-Token-Budget"] = str(budget)
        min_chars = self._config.fallback_min_chars
        # First attempt: the plain (cheap) markdown conversion. An HTTP-level
        # failure of jina itself here propagates as ProviderError WITHOUT the
        # escalations below: the heavy tiers fix parsing, not jina's own
        # errors. A site's refusal comes back inside a 200 — see _refusal.
        plain = await self._get(client, url, headers, retries=self._config.retries)
        refusal = plain.refusal
        if refusal and refusal.final:
            raise ProviderError(f"{self.name}: {refusal.message}", reason=refusal.category)
        # Any other refusal (403, 498, ...) is no text at all, however long the
        # refusal page: it climbs the ladder like an empty answer, where the
        # residential step may cure a block by IP reputation.
        text = "" if refusal else plain.text
        # Good enough → done, no second (3x-priced) request. The explicit
        # `text and` guard matters when fallback_min_chars=0: a length check
        # alone would accept "" as success and bypass the final empty-guard —
        # an empty text must never be returned as a successful read.
        if text and len(text) >= min_chars:
            return text
        is_pdf = _is_pdf_url(url)
        steps = [
            (label, extra_headers)
            for label, extra_headers, pdf_only in _ESCALATIONS
            if (is_pdf or not pdf_only)
            # After a refusal only a step on a different exit (x-proxy) can get
            # past it; a parsing-only step would just be refused again.
            and (refusal is None or "x-proxy" in extra_headers)
        ]
        # Thin or empty first answer. In keyed mode, climb the escalation ladder
        # one step at a time. Keyless mode gets no escalation at all — every
        # step is a paid feature and a keyless instance is not billed.
        #
        # A hopeless .pdf url therefore costs up to FOUR sequential requests.
        # That is a deliberate trade, not an oversight: it happens on roughly a
        # hundred problem urls a month out of three thousand, and the pages it
        # recovers are worth the latency on those.
        step_error: ProviderError | None = None
        if self._config.api_key:
            for label, extra_headers in steps:
                try:
                    answer = await self._get(
                        client,
                        url,
                        {**headers, **extra_headers},
                        # retries=0 on purpose, NOT self._config.retries: the
                        # fallback — the best text so far — is already in hand,
                        # so a retry of these slow, heavily-priced tiers can
                        # only add latency (up to ~2x request_timeout) and cost
                        # before returning what we would return anyway.
                        retries=0,
                    )
                except ProviderError as exc:
                    # A step that could not be served is a reason to hand the
                    # url to the next provider, not to buy the next (dearer)
                    # step. It must not mask a usable answer either: a
                    # non-empty thin text still feeds the pipeline's best_thin
                    # fallback, and a refusal met earlier is reported over it.
                    step_error = exc
                    break
                # One successful read() can mean SEVERAL billed 200s (each of
                # these above plain price) while the pipeline accounts a single
                # provider name. Each step logs its own line, and only after its
                # request returned jina's JSON (an HTTP-level rejection is not
                # billed), so counting these lines counts the extra billed
                # calls — modulo a client-side timeout on a request the server
                # already processed and billed (see the billed.append comment
                # in Pipeline.read). One caveat: a line is also emitted when the
                # read() still fails afterwards — every tier was empty or
                # refused — and then jina contributes no entry to the
                # pipeline's `billed` at all, so paid_calls + these log lines
                # undercounts that case. A final refusal on the plain attempt
                # raises before any line, leaving its one billed call uncounted
                # too.
                logger.info("{}: {} for url={}", self.name, label, url)
                if answer.refusal:
                    # A refused step brought no text, however long its page,
                    # and nothing further down the ladder gets past it.
                    refusal = answer.refusal
                    break
                # Longest wins: any tier can come back thin (or empty), so keep
                # whichever extracted more.
                if len(answer.text) > len(text):
                    text = answer.text
                # Enough text now → stop paying for dearer steps.
                if text and len(text) >= min_chars:
                    break
        if text:
            return text
        if refusal:
            raise ProviderError(f"{self.name}: {refusal.message}", reason=refusal.category)
        if step_error:
            raise step_error
        raise ProviderError(f"{self.name}: empty response", reason=failure_reason.EMPTY)
