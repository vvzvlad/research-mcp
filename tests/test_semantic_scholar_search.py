"""Semantic Scholar paper search: request shape, paper mapping, failures.

The payload follows the ``PaperRelevanceSearchBatch`` schema of the official
Graph API swagger (``total`` / ``offset`` / ``next`` / ``data``). Network I/O is
mocked with respx.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.semantic_scholar_search import (
    SEMANTIC_SCHOLAR_FIELDS,
    SEMANTIC_SCHOLAR_SEARCH_ENDPOINT,
    SemanticScholarSearch,
)

S2_PAGE = "https://www.semanticscholar.org/paper/abc"


def _paper(external_ids, **fields) -> dict:
    return {
        "paperId": "abc",
        "title": "A paper",
        "url": S2_PAGE,
        "abstract": "Deep nets learn.",
        "year": 2021,
        "externalIds": external_ids,
        **fields,
    }


async def _search(make_config, response: httpx.Response, query="q", num_results=5):
    route = respx.get(SEMANTIC_SCHOLAR_SEARCH_ENDPOINT).mock(return_value=response)
    provider = SemanticScholarSearch(make_config("semantic-scholar", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, query, num_results, "ru")
    return route, results


def _papers(*papers) -> httpx.Response:
    return httpx.Response(200, json={"total": len(papers), "offset": 0, "data": list(papers)})


@respx.mock
async def test_request_carries_the_key_and_unhyphenated_query(make_config):
    route, _ = await _search(make_config, _papers(), query="state-space models", num_results=0)
    request = route.calls.last.request
    assert request.headers["x-api-key"] == "k"
    # Hyphenated terms match nothing upstream; language is not sent.
    assert dict(request.url.params) == {
        "query": "state space models",
        "limit": "1",
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }


@respx.mock
async def test_url_prefers_doi_then_arxiv_then_the_s2_page(make_config):
    # S2 keeps the DOI's original case; it is lowercased to match OpenAlex's url
    # so the merge dedups a paper both engines found.
    papers = (
        _paper({"DOI": "10.1/ABC", "ArXiv": "2101.00001"}),
        _paper({"ArXiv": "2101.00001"}, abstract=None),
        _paper(None, year=None),
    )
    _, results = await _search(make_config, _papers(*papers))
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        ("A paper", "https://doi.org/10.1/abc", "2021. Deep nets learn.", "semantic-scholar"),
        ("A paper", "https://arxiv.org/abs/2101.00001", "2021", "semantic-scholar"),
        ("A paper", S2_PAGE, "Deep nets learn.", "semantic-scholar"),
    ]


@respx.mock
async def test_snippet_is_capped(make_config):
    _, results = await _search(make_config, _papers(_paper({}, abstract="x" * 2000)))
    assert len(results[0].snippet) == 500


@respx.mock
@pytest.mark.parametrize(
    "payload", [{"total": 0, "offset": 0}, {"total": "0"}, {"total": 0, "offset": 0, "data": []}]
)
async def test_nothing_found_is_an_empty_answer(make_config, payload):
    _, results = await _search(make_config, httpx.Response(200, json=payload))
    assert results == []


@respx.mock
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"total": 5, "offset": 0}),
        httpx.Response(200, json={"message": "Internal Server Error"}),
        httpx.Response(200, json={"total": 1, "data": "nope"}),
        httpx.Response(200, text="<html>not json</html>"),
    ],
)
async def test_an_answer_without_a_data_list_is_a_provider_error(make_config, response):
    # Never `[]`: that would read as "searched, found nothing".
    with pytest.raises(ProviderError):
        await _search(make_config, response)


def test_requires_an_api_key(make_config):
    with pytest.raises(ValueError):
        SemanticScholarSearch(make_config("semantic-scholar"))
