"""Crawl4AI read provider (self-hosted headless-browser crawler).

API: POST ``{url}/md`` with header ``Authorization: Bearer {token}`` and body
``{"url": url, "f": "fit"}`` → ``{"markdown": "...", "success": true}``.
"""

from __future__ import annotations

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.registry import register


@register("crawl4ai")
class Crawl4aiRead:
    """Headless-browser read via self-hosted Crawl4AI (requires url + token)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.url:
            raise ValueError("crawl4ai requires a url")
        if not config.token:
            raise ValueError("crawl4ai requires a token")
        self.name = config.name
        self.proxy = config.proxy
        self._base = config.url.rstrip("/")
        self._config = config

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        response = await request_with_retry(
            client,
            "POST",
            f"{self._base}/md",
            json={"url": url, "f": "fit"},
            headers={"Authorization": f"Bearer {self._config.token}"},
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        markdown = data.get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            raise ProviderError(f"{self.name}: empty markdown (bot protection?)")
        return markdown.strip()
