"""Exa search API provider.

API (verified 2026-06): POST ``https://api.exa.ai/search`` with header
``x-api-key`` and JSON body ``{"query", "numResults", "type": "auto"}`` →
``{"results": [{"url", "title", ...}], ...}``. Search only — no snippet field by
default, so the snippet is left empty.
"""

from __future__ import annotations

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

EXA_ENDPOINT = "https://api.exa.ai/search"

# Exa accepts numResults 1..100; clamp to keep the facade's contract (max 50)
# and avoid a 4xx on oversized values.
EXA_NUM_RESULTS_MAX = 50


@register("exa")
class ExaSearch:
    """Neural/keyword search via Exa (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("exa requires an api_key")
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
        capped = max(1, min(num_results, EXA_NUM_RESULTS_MAX))
        body = {"query": query, "numResults": capped, "type": "auto"}
        response = await request_with_retry(
            client,
            "POST",
            EXA_ENDPOINT,
            json=body,
            headers={"x-api-key": self._config.api_key, "Content-Type": "application/json"},
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
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
                    snippet=(item.get("text") or item.get("summary") or "").strip(),
                    source=self.name,
                )
            )
        return out
