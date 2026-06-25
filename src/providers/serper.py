"""Serper.dev Google Search API provider.

API (verified 2026-06): POST ``https://google.serper.dev/search`` with header
``X-API-KEY`` and JSON body ``{"q", "num", "page"}`` →
``{"organic": [{"title", "link", "snippet"}], ...}``.
"""

from __future__ import annotations

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

SERPER_ENDPOINT = "https://google.serper.dev/search"


@register("serper")
class SerperSearch:
    """Google search via Serper (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("serper requires an api_key")
        self.name = config.name
        self._config = config

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int,
        page: int,
        language: str | None,
    ) -> list[SearchResult]:
        body: dict[str, object] = {"q": query, "num": num_results, "page": page}
        if language:
            body["hl"] = language
        response = await request_with_retry(
            client,
            "POST",
            SERPER_ENDPOINT,
            json=body,
            headers={"X-API-KEY": self._config.api_key, "Content-Type": "application/json"},
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        organic = data.get("organic")
        if not isinstance(organic, list):
            return []
        out: list[SearchResult] = []
        for item in organic:
            if not isinstance(item, dict):
                continue
            url = (item.get("link") or "").strip()
            if not url:
                continue
            out.append(
                SearchResult(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("snippet") or "").strip(),
                    source=self.name,
                )
            )
        return out
