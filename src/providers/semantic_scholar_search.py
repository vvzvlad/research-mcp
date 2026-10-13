"""Semantic Scholar paper search provider (scholarly papers in the web search merge).

API (per the official Graph API reference,
https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_relevance_search
— NOT verified against a live call): GET
``https://api.semanticscholar.org/graph/v1/paper/search`` with ``query``,
``limit`` (at most 100) and ``fields``; the key goes in ``x-api-key``. The docs
warn that hyphenated query terms yield no matches and say to replace the hyphen
with a space, so the query is sent that way. ``tldr`` is not requested.

Response: ``{"total", "offset", "next"?, "data": [FullPaper]}``. For zero hits
the swagger shows ``data: []``; a bare ``{"total": 0}`` without ``data`` is
accepted as an empty answer too. A paper carries ``title``,
``url`` (its semanticscholar.org page), ``abstract`` (may be null), ``year`` and
``externalIds`` (``DOI``, ``ArXiv``, ...).

The key allows 1 request per second; a 429 is already a hard failure in
``_http.py``.
"""

from __future__ import annotations

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

SEMANTIC_SCHOLAR_SEARCH_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"

# `limit` is documented as "must be <= 100".
SEMANTIC_SCHOLAR_LIMIT_MAX = 100

SEMANTIC_SCHOLAR_FIELDS = "title,url,abstract,year,externalIds"

# An abstract runs to a few thousand chars; the hit only needs enough to judge it.
_SNIPPET_MAX_CHARS = 500


@register("semantic_scholar_search")
class SemanticScholarSearch:
    """Paper search via the Semantic Scholar Graph API (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("semantic_scholar_search requires an api_key")
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
        # `language` is ignored: filtering on the papers' metadata language would
        # drop most of the literature, which is in English whatever the query.
        response = await request_with_retry(
            client,
            "GET",
            SEMANTIC_SCHOLAR_SEARCH_ENDPOINT,
            params={
                "query": query.replace("-", " "),
                "limit": max(1, min(num_results, SEMANTIC_SCHOLAR_LIMIT_MAX)),
                "fields": SEMANTIC_SCHOLAR_FIELDS,
            },
            headers={"x-api-key": self._config.api_key},
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        papers = data.get("data") if isinstance(data, dict) else None
        if papers is None and isinstance(data, dict) and data.get("total") in (0, "0"):
            return []
        # Anything else (no `data` with a non-zero or missing `total`, a `data`
        # that is not a list) means the search did not run as documented.
        if not isinstance(papers, list):
            raise ProviderError(f"{self.name}: unexpected response (no data list)")
        out: list[SearchResult] = []
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            ids = paper.get("externalIds")
            ids = ids if isinstance(ids, dict) else {}
            # A DOI or arXiv link first: the scholarly reader turns either into
            # the paper's open-access full text.
            if ids.get("DOI"):
                # DOIs are case-insensitive and OpenAlex lowercases them; the same
                # case here lets the merge dedup one paper found by both engines.
                url = f"https://doi.org/{ids['DOI'].lower()}"
            elif ids.get("ArXiv"):
                url = f"https://arxiv.org/abs/{ids['ArXiv']}"
            else:
                url = (paper.get("url") or "").strip()
            if not url:
                continue
            year = paper.get("year")
            parts = [str(year) if year else "", (paper.get("abstract") or "").strip()]
            out.append(
                SearchResult(
                    title=(paper.get("title") or "").strip(),
                    url=url,
                    snippet=". ".join(part for part in parts if part)[:_SNIPPET_MAX_CHARS],
                    source=self.name,
                )
            )
        return out
