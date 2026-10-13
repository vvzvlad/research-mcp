"""OpenAlex works search provider (scholarly papers in the web search merge).

API (https://docs.openalex.org/api-entities/works/search-works, work object
https://docs.openalex.org/api-entities/works/work-object; the envelope below is
measured live): GET ``https://api.openalex.org/works`` with ``search=<query>``
(stemmed full-text search over title, abstract, fulltext and keywords),
``per_page`` (1..100) and ``select`` limited to the fields mapped here. The key
goes in ``Authorization: Bearer <key>``: the docs make it equivalent to the
``api_key`` query parameter, and the header keeps the key out of urls.

Response: ``{"meta": {...}, "results": [Work]}``. A work's title is
``display_name``; ``doi`` is already a full ``https://doi.org/…`` url (or
null); ``primary_location.landing_page_url`` is the publisher page; ``id`` is
the work's openalex.org url. The abstract comes only as
``abstract_inverted_index`` (word → list of its positions in the text).

Cost: a search spends 10 credits of the key's daily budget (the free key gives
~1000 searches a day); a spent budget is a 429, which ``_http.py`` already
treats as a hard failure.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"

# `per_page` is documented as 1..100.
OPENALEX_PER_PAGE_MAX = 100

# Only what the mapping below reads: keeps the answer small.
OPENALEX_SELECT = "id,doi,display_name,publication_year,abstract_inverted_index,primary_location"

# An abstract runs to a few thousand chars; the hit only needs enough to judge it.
_SNIPPET_MAX_CHARS = 500


def _abstract(inverted_index: Any) -> str:
    """Rebuild the abstract text from OpenAlex's word → positions index."""
    if not isinstance(inverted_index, dict):
        return ""
    words: dict[int, str] = {}
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                words[position] = word
    return " ".join(words[position] for position in sorted(words))


@register("openalex_search")
class OpenAlexSearch:
    """Scholarly works search via the OpenAlex API (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("openalex_search requires an api_key")
        self.name = config.name
        self.proxy = config.proxy
        self._config = config

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int,
        language: str | None,
    ) -> list[SearchResult]:
        # `language` is ignored: filtering on the works' metadata language would
        # drop most of the literature, which is in English whatever the query.
        response = await request_with_retry(
            client,
            "GET",
            OPENALEX_WORKS_ENDPOINT,
            params={
                # `?` and `*` are wildcards to OpenAlex, rejected with a 400 by the
                # default (stemmed) search — and a question from the model ends in `?`.
                "search": query.replace("?", " ").replace("*", " "),
                "per_page": max(1, min(num_results, OPENALEX_PER_PAGE_MAX)),
                "select": OPENALEX_SELECT,
            },
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        results = data.get("results") if isinstance(data, dict) else None
        # The envelope always carries `results`, empty when nothing matched;
        # without it the search did not run, which is not an empty answer.
        if not isinstance(results, list):
            raise ProviderError(f"{self.name}: unexpected response (no results list)")
        out: list[SearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            location = item.get("primary_location")
            landing = location.get("landing_page_url") if isinstance(location, dict) else None
            # The DOI first: the scholarly reader turns a doi.org link into the
            # paper's open-access full text.
            url = (item.get("doi") or landing or item.get("id") or "").strip()
            if not url:
                continue
            year = item.get("publication_year")
            parts = [str(year) if year else "", _abstract(item.get("abstract_inverted_index"))]
            out.append(
                SearchResult(
                    title=(item.get("display_name") or "").strip(),
                    url=url,
                    snippet=". ".join(part for part in parts if part)[:_SNIPPET_MAX_CHARS],
                    source=self.name,
                )
            )
        return out
