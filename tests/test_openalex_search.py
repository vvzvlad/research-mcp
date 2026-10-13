"""OpenAlex works search: request shape, work mapping, failures.

The payload mirrors the live ``/works`` envelope (``meta`` + ``results``) with
the selected work fields. Network I/O is mocked with respx.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.openalex_search import (
    OPENALEX_SELECT,
    OPENALEX_WORKS_ENDPOINT,
    OpenAlexSearch,
)

WORK = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1/abc",
    "display_name": "A paper",
    "publication_year": 2021,
    # "Deep nets learn. Nets learn." — a word may hold several positions.
    "abstract_inverted_index": {"Deep": [0], "nets": [1], "learn.": [2, 4], "Nets": [3]},
    "primary_location": {"landing_page_url": "https://publisher.test/abc"},
}


async def _search(make_config, response: httpx.Response, num_results=5, language=None):
    route = respx.get(OPENALEX_WORKS_ENDPOINT).mock(return_value=response)
    provider = OpenAlexSearch(make_config("openalex", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "deep nets", num_results, language)
    return route, results


def _works(*works) -> httpx.Response:
    return httpx.Response(200, json={"meta": {"count": len(works)}, "results": list(works)})


@respx.mock
async def test_request_carries_the_bearer_key_and_the_documented_params(make_config):
    route, _ = await _search(make_config, _works(), num_results=500, language="ru")
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer k"
    # The key travels in the header, never in the url; language is not sent.
    assert dict(request.url.params) == {
        "search": "deep nets",
        "per_page": "100",
        "select": OPENALEX_SELECT,
    }


@respx.mock
async def test_wildcard_characters_are_dropped_from_the_query(make_config):
    # OpenAlex answers 400 to `?`/`*` in a stemmed search; a question ends in `?`.
    route = respx.get(OPENALEX_WORKS_ENDPOINT).mock(return_value=_works())
    provider = OpenAlexSearch(make_config("openalex", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "what are deep* nets?", 5, None)
    assert route.calls.last.request.url.params["search"] == "what are deep  nets "


@respx.mock
async def test_maps_doi_title_and_year_prefixed_rebuilt_abstract(make_config):
    _, results = await _search(make_config, _works(WORK))
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        ("A paper", "https://doi.org/10.1/abc", "2021. Deep nets learn. Nets learn.", "openalex")
    ]


@respx.mock
async def test_url_falls_back_to_landing_page_then_openalex_id(make_config):
    no_doi = {**WORK, "doi": None}
    bare = {"id": "https://openalex.org/W2", "display_name": "Bare", "primary_location": None}
    no_url = {"display_name": "Nowhere"}
    _, results = await _search(make_config, _works(no_doi, bare, no_url))
    assert [(r.url, r.snippet) for r in results] == [
        ("https://publisher.test/abc", "2021. Deep nets learn. Nets learn."),
        ("https://openalex.org/W2", ""),
    ]


@respx.mock
async def test_snippet_is_capped(make_config):
    long_work = {**WORK, "abstract_inverted_index": {"word": list(range(400))}}
    _, results = await _search(make_config, _works(long_work))
    assert len(results[0].snippet) == 500
    assert results[0].snippet.startswith("2021. word word")


@respx.mock
async def test_empty_results_is_an_empty_answer(make_config):
    _, results = await _search(make_config, _works())
    assert results == []


@respx.mock
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"meta": {}}),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, text="<html>not json</html>"),
    ],
)
async def test_an_answer_without_a_results_list_is_a_provider_error(make_config, response):
    # Never `[]`: that would read as "searched, found nothing".
    with pytest.raises(ProviderError):
        await _search(make_config, response)


def test_requires_an_api_key(make_config):
    with pytest.raises(ValueError):
        OpenAlexSearch(make_config("openalex"))
