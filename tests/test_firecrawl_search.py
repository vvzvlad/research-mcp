"""Firecrawl v2 search provider: response parsing, request body, paging refusal.

The payload mirrors the shape documented for POST
https://api.firecrawl.dev/v2/search — ``{"success": true, "data": {"web": [...]}}``
with the snippet in ``data.web[].description``. Network I/O is mocked with respx,
like the rest of the suite — there is no live key on this machine.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.firecrawl_search import (
    FIRECRAWL_LIMIT_MAX,
    FIRECRAWL_SEARCH_ENDPOINT,
    FirecrawlSearch,
)

FIRECRAWL_PAYLOAD = {
    "success": True,
    "data": {
        "web": [
            {
                "title": "First hit",
                "description": "snippet one",
                "url": "https://firecrawl.test/1",
                "position": 1,
            },
            {
                "title": "Second hit",
                "description": "snippet two",
                "url": "https://firecrawl.test/2",
                "position": 2,
            },
        ]
    },
    "id": "job-1",
    "creditsUsed": 2,
}


def _body(route) -> dict:
    """The JSON body of the last request the route saw."""
    return json.loads(route.calls.last.request.content)


# -- response parsing ------------------------------------------------------


@respx.mock
async def test_parses_web_results_with_description_as_snippet(make_config):
    # Firecrawl's snippet field is `description` (not `content` as in tavily).
    respx.post(FIRECRAWL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=FIRECRAWL_PAYLOAD)
    )
    provider = FirecrawlSearch(make_config("firecrawl_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        ("First hit", "https://firecrawl.test/1", "snippet one", "firecrawl_search"),
        ("Second hit", "https://firecrawl.test/2", "snippet two", "firecrawl_search"),
    ]


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"success": True, "data": {"web": []}},  # nothing found
        {"success": True, "data": {}},  # no web block at all
        {"success": True, "data": {"images": [{"url": "https://x.test/i.png"}]}},
        {"success": True},  # no data block at all
    ],
)
async def test_missing_or_empty_web_block_is_an_empty_answer(make_config, payload):
    # A response carrying no `web` group is a normal empty answer, not a crash.
    respx.post(FIRECRAWL_SEARCH_ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    provider = FirecrawlSearch(make_config("firecrawl_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_items_without_url_are_skipped(make_config):
    respx.post(FIRECRAWL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "web": [
                        {"title": "No url", "url": "", "description": "x"},
                        "not-a-dict",
                        {
                            "title": "Good",
                            "url": "https://firecrawl.test/ok",
                            "description": "y",
                        },
                    ]
                },
            },
        )
    )
    provider = FirecrawlSearch(make_config("firecrawl_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://firecrawl.test/ok"]


# -- request shape ---------------------------------------------------------


@respx.mock
async def test_request_url_auth_header_and_body(make_config):
    route = respx.post(FIRECRAWL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=FIRECRAWL_PAYLOAD)
    )
    provider = FirecrawlSearch(make_config("firecrawl_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "brushless motor", 8, 1, None)
    request = route.calls.last.request
    assert str(request.url) == FIRECRAWL_SEARCH_ENDPOINT
    assert request.headers["Authorization"] == "Bearer k"
    # Minimal body: `limit` is applied per source and billed in blocks of 10, so
    # only "web" is requested and only as many results as the caller asked for.
    assert _body(route) == {"query": "brushless motor", "limit": 8, "sources": ["web"]}


@respx.mock
async def test_limit_is_capped_at_the_documented_ceiling(make_config):
    route = respx.post(FIRECRAWL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=FIRECRAWL_PAYLOAD)
    )
    provider = FirecrawlSearch(make_config("firecrawl_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 500, 1, None)
    assert _body(route)["limit"] == FIRECRAWL_LIMIT_MAX


# -- paging ----------------------------------------------------------------


@respx.mock
@pytest.mark.parametrize("page", [2, 7])
async def test_page_beyond_the_first_is_refused_without_a_request(make_config, page):
    # Firecrawl has no paging parameter, so page 2 would be page 1 again: refuse
    # instead of spending 2 more credits on links already in the merge.
    route = respx.post(FIRECRAWL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=FIRECRAWL_PAYLOAD)
    )
    provider = FirecrawlSearch(make_config("firecrawl_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, page, None)
    assert "no paging" in str(excinfo.value)
    assert route.call_count == 0  # never reached the network, so never billed


# -- failures --------------------------------------------------------------


@respx.mock
async def test_out_of_credits_is_a_provider_error_without_retry(make_config):
    route = respx.post(FIRECRAWL_SEARCH_ENDPOINT).mock(return_value=httpx.Response(402))
    provider = FirecrawlSearch(make_config("firecrawl_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "out of credits" in str(excinfo.value)
    assert route.call_count == 1  # 402 fails over, it does not retry


@respx.mock
async def test_invalid_json_is_a_provider_error(make_config):
    respx.post(FIRECRAWL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, text="not json at all")
    )
    provider = FirecrawlSearch(make_config("firecrawl_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "invalid JSON" in str(excinfo.value)


@respx.mock
async def test_server_error_uses_the_shared_retry_budget(make_config):
    # Unlike brave (retries=0 because of its 1 req/s plan limit), this provider
    # passes config.retries through: there is no throttle for a retry to fight.
    route = respx.post(FIRECRAWL_SEARCH_ENDPOINT).mock(return_value=httpx.Response(500))
    config = make_config("firecrawl_search", api_key="k")
    assert config.retries == 1
    provider = FirecrawlSearch(config)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await provider.search(client, "q", 5, 1, None)
    assert route.call_count == config.retries + 1


def test_requires_an_api_key(make_config):
    with pytest.raises(ValueError):
        FirecrawlSearch(make_config("firecrawl_search"))
