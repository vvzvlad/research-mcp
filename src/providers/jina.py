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

The last two form the keyed-only RETRY tier in ``read``: the cheap plain
conversion goes first, and only a thin/empty answer escalates to them.
"""

from __future__ import annotations

import httpx
from loguru import logger

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.registry import register

JINA_READER_BASE = "https://r.jina.ai/"


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
        # failure here propagates as ProviderError WITHOUT the readerlm retry
        # below: an upstream 4xx means the site refused jina's fetch, and the
        # LM tier fixes parsing, not access.
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
        # Thin or empty first answer. In keyed mode, escalate ONCE to the
        # heavy tier: ReaderLM-v2 conversion on the browser engine (the
        # existing format and budget headers are kept). Keyless mode gets no
        # such retry — the heavy tier is a paid feature and a keyless instance
        # is not billed.
        if self._config.api_key:
            retry_headers = {
                **headers,
                "X-Respond-With": "readerlm-v2",
                "X-Engine": "browser",
            }
            try:
                retry_response = await request_with_retry(
                    client,
                    "GET",
                    f"{JINA_READER_BASE}{url}",
                    # retries=0 on purpose, NOT self._config.retries: the
                    # fallback — the first attempt's text — is already in
                    # hand, so a retry of the slowest, 3x-priced tier can only
                    # add latency (up to ~2x request_timeout) and cost before
                    # returning what we would return anyway.
                    retries=0,
                    provider=self.name,
                    headers=retry_headers,
                )
            except ProviderError:
                # The retry failing must not mask a usable first answer: a
                # non-empty thin text still feeds the pipeline's best_thin
                # fallback, which beats surfacing nothing at all.
                if text:
                    return text
                raise
            # One successful read() can mean TWO billed 200s (this one at 3x
            # price) while the pipeline accounts a single provider name. Logged
            # only after the escalation request returned (a failed escalation
            # returns the thin text or re-raises above; an HTTP-level rejection
            # is not billed), so counting these lines counts the extra billed
            # heavy-tier calls — modulo a client-side timeout on a request the
            # server already processed and billed (see the billed.append
            # comment in Pipeline.read). One
            # caveat: this line is also emitted when the read() still fails
            # afterwards — the escalation returned 200 but both tiers were
            # empty, so the empty-guard below raises ProviderError — and then
            # jina contributes no entry to the pipeline's `billed` at all, so
            # reconstructing spend as paid_calls + these log lines undercounts
            # that rare case.
            logger.info("{}: readerlm-v2 escalation for url={}", self.name, url)
            retry_text = retry_response.text.strip()
            # Longer wins: either tier can come back thin (or empty), so keep
            # whichever extracted more.
            if len(retry_text) > len(text):
                text = retry_text
        if not text:
            raise ProviderError(f"{self.name}: empty response")
        return text
