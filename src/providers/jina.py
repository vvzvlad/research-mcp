"""Jina AI Reader provider.

API (verified 2026-08-18 from https://docs.jina.ai/): GET
``https://r.jina.ai/{url}`` returning Markdown. Headers used here:

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

The last four form the keyed-only ESCALATION LADDER in ``read``: the cheap plain
conversion goes first, and only a thin/empty answer climbs the ladder, one step
at a time, stopping as soon as a step returns enough text. See ``_ESCALATIONS``
for what each step buys and what it costs.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx
from loguru import logger

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.registry import register

JINA_READER_BASE = "https://r.jina.ai/"

# The conversion tier every escalation shares: the ReaderLM-v2 small LM on the
# highest-quality fetch engine.
_HEAVY_TIER = {"X-Respond-With": "readerlm-v2", "X-Engine": "browser"}

# The keyed escalation ladder, tried in this order, each step only if everything
# before it STILL came back thinner than fallback_min_chars. An entry is
# (log label, extra headers, pdf_only); the rules shared by every step — one
# unretried request, longest answer wins, log line only after a 200 — live in
# ``read``.
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

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        headers = {"X-Return-Format": "markdown"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
            # The budget only matters in keyed mode: keyless requests are not
            # billed, so the header would add a failure mode there for free.
            budget = int(self._config.options.get("token_budget", "0"))
            if budget > 0:
                headers["X-Token-Budget"] = str(budget)
        # First attempt: the plain (cheap) markdown conversion. An HTTP-level
        # failure here propagates as ProviderError WITHOUT the escalations
        # below: an upstream 4xx means the site refused jina's fetch, and the
        # heavy tiers fix parsing, not access.
        response = await request_with_retry(
            client,
            "GET",
            f"{JINA_READER_BASE}{url}",
            retries=self._config.retries,
            provider=self.name,
            headers=headers,
        )
        text = response.text.strip()
        # Good enough → done, no second (3x-priced) request. The explicit
        # `text and` guard matters when fallback_min_chars=0: a length check
        # alone would accept "" as success and bypass the final empty-guard —
        # an empty text must never be returned as a successful read.
        if text and len(text) >= self._config.fallback_min_chars:
            return text
        # Thin or empty first answer. In keyed mode, climb the escalation ladder
        # one step at a time. Keyless mode gets no escalation at all — every
        # step is a paid feature and a keyless instance is not billed.
        #
        # A hopeless .pdf url therefore costs up to FOUR sequential requests.
        # That is a deliberate trade, not an oversight: it happens on roughly a
        # hundred problem urls a month out of three thousand, and the pages it
        # recovers are worth the latency on those.
        if self._config.api_key:
            for label, extra_headers, pdf_only in _ESCALATIONS:
                # Some earlier step already produced enough text → stop paying.
                if text and len(text) >= self._config.fallback_min_chars:
                    break
                if pdf_only and not _is_pdf_url(url):
                    continue
                try:
                    retry_response = await request_with_retry(
                        client,
                        "GET",
                        f"{JINA_READER_BASE}{url}",
                        # retries=0 on purpose, NOT self._config.retries: the
                        # fallback — the best text so far — is already in hand,
                        # so a retry of these slow, heavily-priced tiers can
                        # only add latency (up to ~2x request_timeout) and cost
                        # before returning what we would return anyway.
                        retries=0,
                        provider=self.name,
                        headers={**headers, **extra_headers},
                    )
                except ProviderError:
                    # A step failing must not mask a usable answer: a non-empty
                    # thin text still feeds the pipeline's best_thin fallback,
                    # which beats surfacing nothing at all. The ladder stops
                    # here either way — a step that could not be served is a
                    # reason to hand the url to the next provider, not to buy
                    # the next (dearer) step.
                    if text:
                        return text
                    raise
                # One successful read() can mean SEVERAL billed 200s (each of
                # these above plain price) while the pipeline accounts a single
                # provider name. Each step logs its own line, and only after its
                # request returned (a failed step returns the thin text or
                # re-raises above; an HTTP-level rejection is not billed), so
                # counting these lines counts the extra billed calls — modulo a
                # client-side timeout on a request the server already processed
                # and billed (see the billed.append comment in Pipeline.read).
                # One caveat: a line is also emitted when the read() still fails
                # afterwards — the step returned 200 but every tier was empty,
                # so the empty-guard below raises ProviderError — and then jina
                # contributes no entry to the pipeline's `billed` at all, so
                # reconstructing spend as paid_calls + these log lines
                # undercounts that rare case.
                logger.info("{}: {} for url={}", self.name, label, url)
                retry_text = retry_response.text.strip()
                # Longest wins: any tier can come back thin (or empty), so keep
                # whichever extracted more.
                if len(retry_text) > len(text):
                    text = retry_text
        if not text:
            raise ProviderError(f"{self.name}: empty response")
        return text
