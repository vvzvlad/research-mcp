"""Parallel Search API provider (web search tuned for LLM consumption).

API (contract read 2026-09-18 from the official reference,
https://docs.parallel.ai/api-reference/search-api/search, cross-checked against
https://docs.parallel.ai/search-api/search-quickstart — NOT verified against a
live call, we hold no key yet): POST ``https://api.parallel.ai/v1/search`` with
headers ``x-api-key`` and ``Content-Type: application/json``. ``/v1beta/search``
is the legacy path, kept only "when maintaining an existing integration".

Request body (``V1SearchRequest``, ``additionalProperties: false``, so an
unknown key is a hard 422 — do not add fields on a hunch):

- ``search_queries`` — REQUIRED, array of strings, "Concise keyword search
  queries, 3-6 words each".
- ``objective`` — optional natural-language goal behind the search.
- ``mode`` — ``turbo`` / ``fast`` / ``basic`` / ``advanced``, ``advanced`` when
  omitted (~3s, highest quality).
- ``max_chars_total``, ``session_id``, ``client_model`` — optional.
- ``advanced_settings`` — where ``max_results`` lives ("Defaults to 10 if not
  provided"), next to ``location`` (an ISO 3166-1 alpha-2 COUNTRY code for geo
  targeting), ``source_policy``, ``fetch_policy`` and ``excerpt_settings``.

Response ``V1SearchResponse``: ``{"search_id", "results", "warnings", "usage",
"session_id"}``, where every ``results[]`` item has ``url`` and ``excerpts``
(both required) plus nullable ``title`` / ``publish_date``. The snippet here is
``excerpts`` — a LIST of markdown strings, not one string like brave's
``description`` or serper's ``snippet``; the strings are long by design (Parallel
returns compressed page extracts, sometimes ending in "... (content
truncated)"), so they are joined into a single snippet below. An empty
``results`` list is a normal empty answer, not a failure.

No pagination and no language parameter exist in the schema (checked 2026-09-18):
there is no page/offset/cursor anywhere, and ``advanced_settings.location`` is a
country code, not a language tag.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

PARALLEL_SEARCH_ENDPOINT = "https://api.parallel.ai/v1/search"

# The schema documents no ceiling for advanced_settings.max_results (only the
# default of 10), so clamp to the tool's own num_results cap — same reasoning as
# jina_search/exa: never push an unvalidated value upstream.
PARALLEL_MAX_RESULTS_CAP = 50

# Excerpts arrive as separate strings; SearchResult.snippet is one string. Blank
# line between them so the merged snippet stays readable as markdown.
_EXCERPT_SEPARATOR = "\n\n"


@register("parallel_search")
class ParallelSearch:
    """Web search via the Parallel Search API (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("parallel_search requires an api_key")
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
        # Parallel has no pagination at all, so page 2+ can only be served by
        # re-running the same search and handing back the same hits — a paid
        # request for results the caller already has (dedup runs inside one
        # search() call, never across calls). Refuse instead, exactly like brave
        # does past its depth limit.
        if page > 1:
            raise ProviderError(
                f"{self.name}: page {page} is unavailable (parallel search has no pagination)"
            )

        body: dict[str, Any] = {
            # One caller query → one search query. `objective` is deliberately
            # NOT sent: the facade receives a keyword query, not a separate
            # statement of intent, and repeating the query there would be a
            # guess about what the LLM meant rather than information Parallel
            # does not already have.
            "search_queries": [query],
            # max_results is nested under advanced_settings — it is NOT a
            # top-level field, and a top-level one would be rejected outright
            # (additionalProperties: false).
            "advanced_settings": {
                "max_results": max(1, min(num_results, PARALLEL_MAX_RESULTS_CAP))
            },
        }
        # `language` is intentionally dropped: the schema has no language field,
        # and `advanced_settings.location` is a COUNTRY code (us/gb/de/jp), so
        # feeding a language tag into it would both fail validation and mean
        # something else entirely. `mode` is left out too — Parallel then applies
        # its documented default (advanced).
        response = await request_with_retry(
            client,
            "POST",
            PARALLEL_SEARCH_ENDPOINT,
            json=body,
            headers={
                "x-api-key": self._config.api_key,
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
            url = (item.get("url") or "").strip()
            if not url:
                continue
            excerpts = item.get("excerpts")
            # Parallel calls the snippet "excerpts" and returns a list of them.
            parts = (
                [part.strip() for part in excerpts if isinstance(part, str) and part.strip()]
                if isinstance(excerpts, list)
                else []
            )
            out.append(
                SearchResult(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=_EXCERPT_SEPARATOR.join(parts),
                    source=self.name,
                )
            )
        return out
