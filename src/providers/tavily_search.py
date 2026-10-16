"""Tavily Search API provider (paid, LLM-oriented search index).

API (per docs.tavily.com → API Reference → Search):
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
    """Web search via the Tavily Search API (requires ``api_key``).

    Keys: ``api_key`` first, then the optional ``options["api_key_2"]`` and
    ``options["api_key_3"]`` (other Tavily accounts). Unlike the readers, which
    are separate instances tried one after another, every search instance runs on
    each query concurrently — a second search instance would spend a credit on
    every key for every query. So the failover between keys lives here: a key the
    vendor reports as spent or throttled (402/429/432) hands the query to the next
    key, any other failure ends the search as before.
    """

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("tavily_search requires an api_key")
        self.name = config.name
        self.proxy = config.proxy
        self._config = config
        self._keys = [config.api_key] + [
            key for name in ("api_key_2", "api_key_3") if (key := config.options.get(name))
        ]

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int,
        language: str | None,
    ) -> list[SearchResult]:
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
            # Measured: «АИРЕ63 однофазный двигатель» without `country` returns
            # foreign parts aggregators, with country="russia" Russian shops.
            #
            # Tavily honours `country` only when `topic` is "general". This body
            # never sends `topic`, and Tavily's default for it is "general", so
            # the boost does apply.
            #
            # The downside, accepted knowingly: a cyrillic query about
            # Kazakhstan or Ukraine gets skewed towards the RF too, because the
            # script — not the subject — is what this test can see.
            body["country"] = TAVILY_CYRILLIC_COUNTRY

        for index, key in enumerate(self._keys):
            try:
                response = await request_with_retry(
                    client,
                    "POST",
                    TAVILY_SEARCH_ENDPOINT,
                    json=body,
                    headers={"Authorization": f"Bearer {key}"},
                    # Unlike brave, this provider does use the shared retry
                    # budget: Tavily has no per-second plan limit and there is no
                    # local throttle here, so a retry 0.3s later is a plain second
                    # attempt, not a request racing its own rate-limit window.
                    retries=self._config.retries,
                    provider=self.name,
                    # Tavily signals a spent monthly plan with the non-standard
                    # HTTP 432.
                    out_of_credits_statuses=frozenset({432}),
                )
                break
            except ProviderError as exc:
                # Only a spent or throttled key moves on to the next one; the
                # last key's error is the search's error. A spent key costs no
                # credit, so trying it again first on every query is cheap.
                if not exc.exhausts_quota or index == len(self._keys) - 1:
                    raise
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
