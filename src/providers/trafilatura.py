"""Trafilatura read provider: direct HTTP fetch + main-content extraction.

The cheapest read path: GET the page with a browser-like UA, then run
``trafilatura.extract`` for Markdown. Requires no configuration, so it is always
enabled and goes first in the read pipeline.
"""

from __future__ import annotations

import re

import httpx
import trafilatura as _trafilatura

from src import failure_reason
from src.providers._http import request_with_retry
from src.providers.base import (
    BROWSER_USER_AGENT,
    ProviderConfig,
    ProviderError,
)
from src.providers.registry import register


# Share of the structural pick's words that must reappear in the full cascade's
# answer for the two to count as the same text (see extract_markdown). Measured
# on 199 pages from the prod read log: a cascade answer that extends the
# structural pick keeps 0.74-1.00 of its words, one that swaps it for another
# block of the page keeps 0.00-0.27.
_MIN_SHARED_WORDS = 0.5
_WORD = re.compile(r"\w+")


def _extract(html: str, *, fast: bool) -> str:
    # favor_precision keeps clean main content and drops nav/boilerplate (menus,
    # footers, logos). The trade-off is under-extraction: when precision returns
    # too little, the pipeline's fallback chain escalates to a heavier reader, so
    # the downside is bounded.
    return (
        _trafilatura.extract(
            html,
            output_format="markdown",
            include_comments=False,
            favor_precision=True,
            fast=fast,
        )
        or ""
    ).strip()


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text)}


def extract_markdown(html: str) -> str:
    """Extract main-content Markdown from an HTML string (pure, no I/O).

    Returns ``""`` when nothing extractable is found, or when trafilatura's own
    extractors disagree on which block of the page is the main content. Exposed
    so the pipeline can reuse an already-downloaded page body (the PDF-probe
    fetch) without GETting the same url a second time.
    """
    full = _extract(html, fast=False)
    if not full:
        return ""
    # trafilatura's cascade lets readability replace its own structural pick
    # (<article>, <main>) when its text is twice as long, and justext replace
    # the result when its text is at least a quarter as long: it compares
    # length, not position. That rescue is what reads forum threads and docs
    # pages whose article tag holds only part of the text, but it also crowns a
    # long block the page keeps hidden in the DOM — a terms-of-sale modal on
    # every product card wins over a short product description. fast=True is the
    # structural pick alone (~3 ms more on a 100 KB page). When the cascade's
    # answer holds less than half of the pick's words, it is another block of
    # the page, neither can be trusted, and the page goes to the next reader in
    # the chain, which returns the whole page.
    own = _extract(html, fast=True)
    own_words = _words(own)
    if own_words and len(own_words & _words(full)) < _MIN_SHARED_WORDS * len(own_words):
        return ""
    return full


@register("trafilatura")
class TrafilaturaRead:
    """Fetch + extract main content locally (no external service, no key)."""

    def __init__(self, config: ProviderConfig) -> None:
        self.name = config.name
        self.proxy = config.proxy
        self._config = config

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        try:
            response = await request_with_retry(
                client,
                "GET",
                url,
                retries=self._config.retries,
                provider=self.name,
                headers={"User-Agent": BROWSER_USER_AGENT},
            )
        except ProviderError as exc:
            # This request IS the page, so its 404/410 is the page saying it
            # does not exist — worded like the other providers' target errors.
            if exc.status in (404, 410):
                raise ProviderError(
                    f"{self.name}: target page returned HTTP {exc.status}",
                    reason=failure_reason.for_target_status(exc.status),
                ) from exc
            raise
        extracted = extract_markdown(response.text)
        if not extracted:
            raise ProviderError(
                f"{self.name}: no main content extracted", reason=failure_reason.EMPTY
            )
        return extracted
