"""Firecrawl v2 search provider.

API (contract read 2026-09-18 from docs.firecrawl.dev/features/search and its
Search API reference): POST ``https://api.firecrawl.dev/v2/search`` with headers
``Authorization: Bearer {key}`` and ``Content-Type: application/json``, JSON body
— ``query`` (required, at most 500 chars), ``limit`` (default 10, documented
range 1..100, applied PER source), ``sources`` (defaults to ``["web"]``; the
other values are ``"news"`` and ``"images"``), ... → ``{"success": true, "data":
{"web": [{"title", "description", "url", "position", ...}], "images": [...],
"news": [...]}, "warning", "id", "creditsUsed"}``.

The snippet lives in ``description`` here — NOT ``content`` as in tavily, and
``snippet`` is the field name only inside the ``news`` group, which this provider
never asks for. A response whose ``data`` carries no ``web`` block is a normal
empty answer, not a failure.

Highlights are on by default (``highlights: true``), so ``description`` holds a
query-relevant passage lifted from the page — Markdown and all — rather than the
site's own meta description; the field name is unchanged either way, and
Firecrawl falls back to the plain description when it cannot build a highlight.

Billing: 2 credits per 10 results, rounded up (1–10 results = 2 credits, 11–20 =
4, ...). This instance has 1000 credits a month, so ``limit`` is exactly what the
caller asked for and never a round number above it.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

FIRECRAWL_SEARCH_ENDPOINT = "https://api.firecrawl.dev/v2/search"

# Firecrawl caps `limit` at 100.
FIRECRAWL_LIMIT_MAX = 100


@register("firecrawl_search")
class FirecrawlSearch:
    """Web search via the Firecrawl v2 search API (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("firecrawl_search requires an api_key")
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
        # Firecrawl's search body has no paging parameter — no `offset`, `page`
        # or `start` — so page 2 would come back as page 1 verbatim. Refuse
        # rather than silently serving the first page again: that would spend
        # two more of the 1000 monthly credits on links we already have, and
        # pages 2, 3, ... would each re-inject those same hits into the merge
        # (dedup runs within a single search() call, never across calls). Same
        # reasoning as brave's refusal past its deepest servable page.
        if page > 1:
            raise ProviderError(
                f"{self.name}: page {page} is beyond firecrawl's depth (no paging)"
            )

        body: dict[str, Any] = {
            "query": query,
            # `limit` is charged in blocks of 10 rounded up, so never ask for
            # more than the caller wants.
            "limit": max(1, min(num_results, FIRECRAWL_LIMIT_MAX)),
            # Plain web hits only. `limit` is applied per source, so adding
            # "news" or "images" here would bill for their results as well.
            "sources": ["web"],
        }
        response = await request_with_retry(
            client,
            "POST",
            FIRECRAWL_SEARCH_ENDPOINT,
            json=body,
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            # Unlike brave, this provider does use the shared retry budget:
            # there is no per-second plan limit and no local throttle here, so a
            # retry 0.3s later is a plain second attempt, not a request racing
            # its own rate-limit window.
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        payload = data.get("data") if isinstance(data, dict) else None
        web = payload.get("web") if isinstance(payload, dict) else None
        if not isinstance(web, list):
            return []
        out: list[SearchResult] = []
        for item in web:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            if not url:
                continue
            out.append(
                SearchResult(
                    title=(item.get("title") or "").strip(),
                    # Firecrawl calls the snippet "description".
                    snippet=(item.get("description") or "").strip(),
                    url=url,
                    source=self.name,
                )
            )
        return out
