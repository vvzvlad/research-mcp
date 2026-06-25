"""Trafilatura read provider: direct HTTP fetch + main-content extraction.

The cheapest read path: GET the page with a browser-like UA, then run
``trafilatura.extract`` for Markdown. Requires no configuration, so it is always
enabled and goes first in the read pipeline.
"""

from __future__ import annotations

import httpx
import trafilatura as _trafilatura

from src.providers._http import request_with_retry
from src.providers.base import (
    BROWSER_USER_AGENT,
    ProviderConfig,
    ProviderError,
)
from src.providers.registry import register


def extract_markdown(html: str) -> str:
    """Extract main-content Markdown from an HTML string (pure, no I/O).

    Returns ``""`` when nothing extractable is found. Exposed so the pipeline can
    reuse an already-downloaded page body (the PDF-probe fetch) without GETting
    the same url a second time.
    """
    return (
        _trafilatura.extract(
            html,
            output_format="markdown",
            include_comments=False,
            favor_recall=True,
        )
        or ""
    ).strip()


@register("trafilatura")
class TrafilaturaRead:
    """Fetch + extract main content locally (no external service, no key)."""

    def __init__(self, config: ProviderConfig) -> None:
        self.name = config.name
        self._config = config

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        response = await request_with_retry(
            client,
            "GET",
            url,
            retries=self._config.retries,
            provider=self.name,
            headers={"User-Agent": BROWSER_USER_AGENT},
        )
        extracted = extract_markdown(response.text)
        if not extracted:
            raise ProviderError(f"{self.name}: no main content extracted")
        return extracted
