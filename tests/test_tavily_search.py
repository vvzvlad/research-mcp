"""Tavily search provider: response parsing, request body, country heuristic.

The payload mirrors the shape documented for POST https://api.tavily.com/search
(top-level ``query``/``answer``/``images``/``results``; the snippet lives in
``results[].content``). Network I/O is mocked with respx, like the rest of the
suite — there is no live key on this machine.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.tavily_search import (
    TAVILY_CYRILLIC_COUNTRY,
    TAVILY_MAX_RESULTS_MAX,
    TAVILY_SEARCH_ENDPOINT,
    TavilySearch,
)

TAVILY_PAYLOAD = {
    "query": "q",
    "images": [],
    "results": [
        {
            "title": "First hit",
            "url": "https://tavily.test/1",
            "content": "snippet one",
            "score": 0.81,
            "raw_content": None,
        },
        {
            "title": "Second hit",
            "url": "https://tavily.test/2",
            "content": "snippet two",
            "score": 0.55,
        },
    ],
    "response_time": "1.67",
}


def _body(route) -> dict:
    """The JSON body of the last request the route saw."""
    return json.loads(route.calls.last.request.content)


# -- response parsing ------------------------------------------------------


@respx.mock
async def test_parses_results_with_content_as_snippet(make_config):
    # Tavily's snippet field is `content` (not `description`/`snippet`).
    respx.post(TAVILY_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=TAVILY_PAYLOAD)
    )
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        ("First hit", "https://tavily.test/1", "snippet one", "tavily_search"),
        ("Second hit", "https://tavily.test/2", "snippet two", "tavily_search"),
    ]


@respx.mock
async def test_empty_results_list_is_an_empty_answer(make_config):
    # Nothing found is a normal empty answer, not a failure.
    respx.post(TAVILY_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"query": "q", "results": [], "images": []})
    )
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_missing_results_key_is_an_empty_answer(make_config):
    respx.post(TAVILY_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"query": "q"})
    )
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_items_without_url_are_skipped(make_config):
    respx.post(TAVILY_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"title": "No url", "url": "", "content": "x"},
                    "not-a-dict",
                    {"title": "Good", "url": "https://tavily.test/ok", "content": "y"},
                ]
            },
        )
    )
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://tavily.test/ok"]


# -- request shape ---------------------------------------------------------


@respx.mock
async def test_request_url_auth_header_and_body(make_config):
    route = respx.post(TAVILY_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=TAVILY_PAYLOAD)
    )
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "brushless motor", 5, 1, None)
    request = route.calls.last.request
    assert str(request.url) == TAVILY_SEARCH_ENDPOINT
    assert request.headers["Authorization"] == "Bearer k"
    assert _body(route) == {
        "query": "brushless motor",
        "max_results": 5,
        # basic = 1 credit, advanced = 2; the plan is 1000 credits a month.
        "search_depth": "basic",
    }


@respx.mock
async def test_max_results_is_capped_at_twenty(make_config):
    # Tavily documents max_results in the range 0..20.
    route = respx.post(TAVILY_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=TAVILY_PAYLOAD)
    )
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 50, 1, None)
    assert _body(route)["max_results"] == TAVILY_MAX_RESULTS_MAX


# -- country heuristic -----------------------------------------------------


@respx.mock
@pytest.mark.parametrize(
    "query",
    [
        "АИРЕ63 однофазный двигатель",  # the measured case
        "ГОСТ 12.2.007",  # cyrillic even when most of the query is digits
        "купить relay 24V",  # mixed script still counts as cyrillic
    ],
)
async def test_cyrillic_query_sends_country_russia(make_config, query):
    # Measured: «АИРЕ63 однофазный двигатель» returns foreign aggregators with no
    # country and specialised Russian shops with country="russia" (6 of 6).
    route = respx.post(TAVILY_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=TAVILY_PAYLOAD)
    )
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, query, 5, 1, None)
    body = _body(route)
    assert body["country"] == TAVILY_CYRILLIC_COUNTRY == "russia"
    # `country` only applies when topic is "general"; we rely on Tavily's own
    # default for that, so `topic` must stay out of the body.
    assert "topic" not in body


@respx.mock
@pytest.mark.parametrize("query", ["brushless single phase motor", "iec 60034-1", "AIRE63"])
async def test_latin_query_sends_no_country(make_config, query):
    # The boost must not leak into non-cyrillic queries — Tavily's own default
    # (no country) is the better guess there.
    route = respx.post(TAVILY_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=TAVILY_PAYLOAD)
    )
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, query, 5, 1, None)
    assert "country" not in _body(route)


# -- paging ----------------------------------------------------------------


@respx.mock
@pytest.mark.parametrize("page", [2, 7])
async def test_page_beyond_the_first_is_refused_without_a_request(make_config, page):
    # Tavily has no paging parameter, so page 2 would be page 1 again: refuse
    # instead of spending a credit on links already in the merge.
    route = respx.post(TAVILY_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=TAVILY_PAYLOAD)
    )
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, page, None)
    assert "no paging" in str(excinfo.value)
    assert route.call_count == 0  # never reached the network, so never billed


# -- failures --------------------------------------------------------------


@respx.mock
async def test_out_of_credits_is_a_provider_error_without_retry(make_config):
    route = respx.post(TAVILY_SEARCH_ENDPOINT).mock(return_value=httpx.Response(402))
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "out of credits" in str(excinfo.value)
    assert route.call_count == 1  # 402 fails over, it does not retry


@respx.mock
async def test_invalid_json_is_a_provider_error(make_config):
    respx.post(TAVILY_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, text="not json at all")
    )
    provider = TavilySearch(make_config("tavily_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "invalid JSON" in str(excinfo.value)


@respx.mock
async def test_server_error_uses_the_shared_retry_budget(make_config):
    # Unlike brave (retries=0 because of its 1 req/s plan limit), this provider
    # passes config.retries through: there is no throttle for a retry to fight.
    route = respx.post(TAVILY_SEARCH_ENDPOINT).mock(return_value=httpx.Response(500))
    config = make_config("tavily_search", api_key="k")
    assert config.retries == 1
    provider = TavilySearch(config)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await provider.search(client, "q", 5, 1, None)
    assert route.call_count == config.retries + 1


def test_requires_an_api_key(make_config):
    with pytest.raises(ValueError):
        TavilySearch(make_config("tavily_search"))
