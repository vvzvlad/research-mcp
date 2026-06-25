"""Tavily Extract read provider.

API (verified 2026-06): POST ``https://api.tavily.com/extract`` with header
``Authorization: Bearer {key}`` and body
``{"urls": [url], "extract_depth": "advanced", "format": "markdown"}`` →
``{"results": [{"url", "raw_content"}], "failed_results": [{"url", "error"}]}``.
A url in ``failed_results`` (or empty ``raw_content``) → ``ProviderError``.
"""

from __future__ import annotations

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.registry import register

TAVILY_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"


@register("tavily")
class TavilyRead:
    """Read a page as Markdown via Tavily Extract (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("tavily requires an api_key")
        self.name = config.name
        self._config = config

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        body = {"urls": [url], "extract_depth": "advanced", "format": "markdown"}
        response = await request_with_retry(
            client,
            "POST",
            TAVILY_EXTRACT_ENDPOINT,
            json=body,
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc

        failed = data.get("failed_results")
        if isinstance(failed, list):
            for item in failed:
                if isinstance(item, dict) and item.get("url") == url:
                    error = item.get("error") or "extraction failed"
                    raise ProviderError(f"{self.name}: {error}")

        results = data.get("results")
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                content = (item.get("raw_content") or "").strip()
                if content:
                    return content
        raise ProviderError(f"{self.name}: empty extraction")
