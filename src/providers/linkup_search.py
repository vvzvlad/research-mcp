"""Linkup Search API provider (own index + optional agentic retrieval).

API (contract read 2026-09-18 from the official reference,
https://docs.linkup.so/pages/documentation/api-reference/endpoint/post-search,
with depth semantics and prices from
https://docs.linkup.so/pages/documentation/get-started/concepts — NOT verified
against a live call, we hold no key yet): POST
``https://api.linkup.so/v1/search`` with ``Authorization: Bearer <key>``
(security scheme ``bearer``) and a JSON body whose REQUIRED fields are ``q``,
``depth`` and ``outputType``.

- ``q`` — "The natural language question for which you want to retrieve context."
- ``depth`` — ``flash`` / ``fast`` / ``standard`` / ``deep``. ``flash`` is
  "Lowest latency. Ranked sources and snippets from our index, no query
  reinterpretation, no LLM" (<200ms); ``fast`` is one-shot retrieval (~1s);
  ``standard`` is single-iteration agentic search (1-3s); ``deep`` chains
  several search-and-scrape iterations (5-30s).
- ``outputType`` — ``searchResults`` / ``sourcedAnswer`` / ``structured``. Only
  ``searchResults`` returns raw sources; the other two invoke an LLM.
- optional: ``maxResults`` (number, minimum 1), ``includeImages`` (default
  false), ``includeDomains`` / ``excludeDomains``, ``fromDate`` / ``toDate``,
  plus ``structuredOutputSchema`` / ``includeSources`` /
  ``includeInlineCitations`` which apply only to the other output types.

Response for ``searchResults``: ``{"results": [...]}`` where a text item carries
``name`` / ``url`` / ``content`` / ``favicon`` / ``type: "text"`` and an image
item only ``name`` / ``url`` / ``type: "image"``. So the TITLE is ``name`` (not
``title``) and the snippet is ``content`` — unlike brave's ``description`` or
serper's ``snippet``. An empty ``results`` list is a normal empty answer.

Pricing (2026-09-18): ``searchResults`` costs $0.005 per call at ``flash``,
``fast`` and ``standard`` alike, and $0.05 at ``deep`` — ten times more. Hence
the default depth below is the cheapest tier, taking ``flash`` within it because
it is also the fastest and the pipeline awaits every search provider.

Documented statuses: 400 invalid parameters, 401 invalid/missing key, 402
(payment details returned in the ``payment-required`` header), 429 "Rate limit
exceeded or insufficient credits" — 402/429 are already a hard failure in
``_http.py``. There is no pagination parameter and no language parameter.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

LINKUP_SEARCH_ENDPOINT = "https://api.linkup.so/v1/search"

# Cheapest depth tier (see the module docstring): flash/fast/standard all cost
# $0.005 per searchResults call, deep costs $0.05.
LINKUP_DEPTH = "flash"

# Raw sources, not an LLM-written answer: this is a search provider, and the
# pipeline does its own merging/reranking downstream.
LINKUP_OUTPUT_TYPE = "searchResults"

# `maxResults` has a documented minimum of 1 and no documented maximum; clamp to
# the tool's own num_results cap so an unvalidated value never goes upstream.
LINKUP_MAX_RESULTS_CAP = 50


@register("linkup_search")
class LinkupSearch:
    """Web search via the Linkup API (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("linkup_search requires an api_key")
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
        # No pagination in the API: page 2+ would repeat page 1's hits at full
        # price, so refuse rather than silently re-serve them (same rule as
        # brave beyond its depth limit).
        if page > 1:
            raise ProviderError(
                f"{self.name}: page {page} is unavailable (linkup has no pagination)"
            )

        body: dict[str, Any] = {
            "q": query,
            "depth": LINKUP_DEPTH,
            "outputType": LINKUP_OUTPUT_TYPE,
            "maxResults": max(1, min(num_results, LINKUP_MAX_RESULTS_CAP)),
        }
        # `language` is intentionally dropped: the request schema has no language
        # or locale field at all, and inventing one would be a 400 on a body the
        # API validates. `includeImages` is left at its default (false) — this
        # provider feeds a text pipeline.
        response = await request_with_retry(
            client,
            "POST",
            LINKUP_SEARCH_ENDPOINT,
            json=body,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
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
            # Image items carry no `content` and are not web pages the read
            # pipeline can use. `includeImages` is never sent (default false), so
            # this should not trigger — it just keeps a server-side default flip
            # from injecting contentless hits.
            if item.get("type") == "image":
                continue
            url = (item.get("url") or "").strip()
            if not url:
                continue
            out.append(
                SearchResult(
                    # Linkup calls the title "name" and the snippet "content".
                    title=(item.get("name") or "").strip(),
                    url=url,
                    snippet=(item.get("content") or "").strip(),
                    source=self.name,
                )
            )
        return out
