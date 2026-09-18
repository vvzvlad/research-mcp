"""Tavily Search API provider (paid, LLM-oriented search index).

API (contract read 2026-09-18 from docs.tavily.com → API Reference → Search):
POST ``https://api.tavily.com/search`` with headers ``Authorization: Bearer
{key}`` and ``Content-Type: application/json``, JSON body — ``query``
(required), ``max_results`` (default 10, documented range 0..20), ``search_depth``
(``basic`` | ``advanced`` | ``fast`` | ``ultra-fast``, default ``basic``),
``topic`` (``general`` | ``news`` | ``finance``, default ``general``), ``country``
(lowercase English country names, ``"russia"`` among them), ... → ``{"query",
"answer", "images", "results": [{"title", "url", "content", "score",
"raw_content", "published_date", ...}], "response_time", "request_id"}``.

The snippet lives in ``content`` here — NOT ``description`` as in brave, nor
``snippet`` as in serper. An empty ``results`` list is a normal empty answer, not
a failure.

Billing: ``basic`` / ``fast`` / ``ultra-fast`` cost 1 API credit per search,
``advanced`` costs 2. This instance has 1000 credits a month, so it always asks
for ``basic`` — the balanced 1-credit depth.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"

# Tavily caps `max_results` at 20.
TAVILY_MAX_RESULTS_MAX = 20

# Value sent in `country` for cyrillic queries — see `search` for the measurement
# behind it. Tavily's `country` enum is spelled out in lowercase English country
# names, not ISO codes, so this really is the literal string "russia".
TAVILY_CYRILLIC_COUNTRY = "russia"

# Basic Cyrillic block U+0400..U+04FF. It covers Russian as well as Ukrainian
# (і, ї, є, ґ), Belarusian, Bulgarian and Serbian letters, so "is this query
# written in cyrillic" is one range test.
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


@register("tavily_search")
class TavilySearch:
    """Web search via the Tavily Search API (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("tavily_search requires an api_key")
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
        # Tavily has no paging at all: the request body carries no `offset`,
        # `page` or `start`, so page 2 would come back as page 1 verbatim.
        # Refuse rather than silently serving the first page again — that would
        # spend one of the 1000 monthly credits on links we already have, and
        # pages 2, 3, ... would each re-inject those same hits into the merge
        # (dedup runs within a single search() call, never across calls). Same
        # reasoning as brave's refusal past its deepest servable page.
        if page > 1:
            raise ProviderError(f"{self.name}: page {page} is beyond tavily's depth (no paging)")

        body: dict[str, Any] = {
            "query": query,
            "max_results": max(1, min(num_results, TAVILY_MAX_RESULTS_MAX)),
            # `basic` = 1 credit, `advanced` = 2. On a 1000-credit monthly plan
            # the extra credit is not worth the relevance bump.
            "search_depth": "basic",
        }
        if _CYRILLIC_RE.search(query):
            # `country` goes out ONLY for cyrillic queries, and only as "russia".
            #
            # This is a measured heuristic, not a hunch: «АИРЕ63 однофазный
            # двигатель» without `country` returns foreign parts aggregators,
            # while the same query with country="russia" returns specialised
            # Russian shops — 6 of 6 hits.
            #
            # Tavily honours `country` only when `topic` is "general". This body
            # never sends `topic`, and Tavily's default for it is "general", so
            # the boost does apply.
            #
            # The downside, accepted knowingly: a cyrillic query about
            # Kazakhstan or Ukraine gets skewed towards the RF too, because the
            # script — not the subject — is what this test can see.
            body["country"] = TAVILY_CYRILLIC_COUNTRY

        response = await request_with_retry(
            client,
            "POST",
            TAVILY_SEARCH_ENDPOINT,
            json=body,
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            # Unlike brave, this provider does use the shared retry budget:
            # Tavily has no per-second plan limit and there is no local throttle
            # here, so a retry 0.3s later is a plain second attempt, not a
            # request racing its own rate-limit window.
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
                    # Tavily calls the snippet "content".
                    snippet=(item.get("content") or "").strip(),
                    url=url,
                    source=self.name,
                )
            )
        return out
