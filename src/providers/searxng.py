"""SearXNG search provider (self-hosted metasearch).

API: GET ``{url}/search?q=&format=json&pageno=&language=`` →
``{"results": [{"url", "title", "content", ...}], ...}``.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register


@register("searxng")
class SearxngSearch:
    """Search via a self-hosted SearXNG instance (requires ``url``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.url:
            raise ValueError("searxng requires a url")
        self.name = config.name
        self.proxy = config.proxy
        self._base = config.url.rstrip("/")
        self._config = config

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int,
        page: int,
        language: str | None,
    ) -> list[SearchResult]:
        params: dict[str, Any] = {"q": query, "format": "json", "pageno": page}
        if language:
            params["language"] = language
        response = await request_with_retry(
            client,
            "GET",
            f"{self._base}/search",
            params=params,
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON (is format=json enabled?)") from exc
        results = data.get("results")
        if not isinstance(results, list):
            return []
        out: list[SearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            if not url:
                continue
            out.append(
                SearchResult(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("content") or "").strip(),
                    source=self.name,
                )
            )
        return out
