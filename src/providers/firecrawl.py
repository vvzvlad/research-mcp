"""Firecrawl scrape read provider.

API (verified 2026-06, v2): POST ``https://api.firecrawl.dev/v2/scrape`` with
header ``Authorization: Bearer {key}`` and body
``{"url": url, "formats": ["markdown"], "proxy": "auto"}`` →
``{"success": true, "data": {"markdown": "..."}}``.
"""

from __future__ import annotations

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.registry import register

FIRECRAWL_SCRAPE_ENDPOINT = "https://api.firecrawl.dev/v2/scrape"


@register("firecrawl")
class FirecrawlRead:
    """Read a page as Markdown via Firecrawl v2 scrape (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("firecrawl requires an api_key")
        self.name = config.name
        self._config = config

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        body = {"url": url, "formats": ["markdown"], "proxy": "auto"}
        response = await request_with_retry(
            client,
            "POST",
            FIRECRAWL_SCRAPE_ENDPOINT,
            json=body,
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        payload = data.get("data") if isinstance(data, dict) else None
        markdown = payload.get("markdown") if isinstance(payload, dict) else None
        if not isinstance(markdown, str) or not markdown.strip():
            raise ProviderError(f"{self.name}: empty markdown")
        return markdown.strip()
