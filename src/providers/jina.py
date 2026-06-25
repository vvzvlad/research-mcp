"""Jina AI Reader provider.

API (verified 2026-06): GET ``https://r.jina.ai/{url}`` returning Markdown. The
``X-Return-Format: markdown`` header asks for Markdown explicitly. An API key is
OPTIONAL — keyless works at a lower rate limit — so this instance is always
enabled; when a key is present it is sent as ``Authorization: Bearer {key}``.
"""

from __future__ import annotations

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.registry import register

JINA_READER_BASE = "https://r.jina.ai/"


@register("jina")
class JinaRead:
    """Read a page as Markdown via Jina Reader (api_key optional)."""

    def __init__(self, config: ProviderConfig) -> None:
        self.name = config.name
        self._config = config

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        headers = {"X-Return-Format": "markdown"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        response = await request_with_retry(
            client,
            "GET",
            f"{JINA_READER_BASE}{url}",
            retries=self._config.retries,
            provider=self.name,
            headers=headers,
        )
        text = response.text.strip()
        if not text:
            raise ProviderError(f"{self.name}: empty response")
        return text
