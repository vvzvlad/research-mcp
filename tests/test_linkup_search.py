"""Linkup search provider: response parsing, request body, paging, failures.

The payload mirrors the ``SearchResultsOutput`` shape documented on 2026-09-18
at https://docs.linkup.so/pages/documentation/api-reference/endpoint/post-search
(``results[]`` items of ``type: "text"`` carrying ``name`` / ``url`` /
``content`` / ``favicon``). Network I/O is mocked with respx — no live call is
ever made, we hold no Linkup key.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.linkup_search import (
    LINKUP_DEPTH,
    LINKUP_MAX_RESULTS_CAP,
    LINKUP_OUTPUT_TYPE,
    LINKUP_SEARCH_ENDPOINT,
    LinkupSearch,
)

LINKUP_PAYLOAD = {
    "results": [
        {
            "type": "text",
            "name": "First hit",
            "url": "https://linkup.test/1",
            "content": "snippet one",
            "favicon": "https://linkup.test/favicon.ico",
        },
        {
            "type": "text",
            "name": "Second hit",
            "url": "https://linkup.test/2",
            "content": "snippet two",
            "favicon": "",
        },
    ]
}


def _body(route) -> dict:
    return json.loads(route.calls.last.request.content)


# -- response parsing ------------------------------------------------------


@respx.mock
async def test_parses_name_as_title_and_content_as_snippet(make_config):
    # Linkup names the title `name` and the snippet `content` — neither matches
    # brave's `title`/`description`.
    respx.post(LINKUP_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=LINKUP_PAYLOAD)
    )
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        ("First hit", "https://linkup.test/1", "snippet one", "linkup_search"),
        ("Second hit", "https://linkup.test/2", "snippet two", "linkup_search"),
    ]


@respx.mock
async def test_empty_results_list_is_a_normal_empty_answer(make_config):
    respx.post(LINKUP_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_missing_results_key_is_an_empty_result_list(make_config):
    respx.post(LINKUP_SEARCH_ENDPOINT).mock(return_value=httpx.Response(200, json={}))
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_image_items_and_items_without_url_are_skipped(make_config):
    # An image item has no `content` at all; includeImages is never requested,
    # so this only guards against a server-side default flip.
    respx.post(LINKUP_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"type": "image", "name": "A picture", "url": "https://linkup.test/i.png"},
                    {"type": "text", "name": "No url", "url": "", "content": "x"},
                    "not-a-dict",
                    {
                        "type": "text",
                        "name": "Good",
                        "url": "https://linkup.test/ok",
                        "content": "y",
                    },
                ]
            },
        )
    )
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://linkup.test/ok"]


# -- request shape ---------------------------------------------------------


@respx.mock
async def test_request_carries_the_bearer_key_and_the_documented_body(make_config):
    route = respx.post(LINKUP_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=LINKUP_PAYLOAD)
    )
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "what is linkup", 6, 1, None)
    request = route.calls.last.request
    assert str(request.url) == LINKUP_SEARCH_ENDPOINT
    assert request.method == "POST"
    assert request.headers["Authorization"] == "Bearer k"
    assert request.headers["Content-Type"] == "application/json"
    # q, depth and outputType are the three required fields.
    assert _body(route) == {
        "q": "what is linkup",
        "depth": LINKUP_DEPTH,
        "outputType": LINKUP_OUTPUT_TYPE,
        "maxResults": 6,
    }


def test_default_depth_is_the_cheapest_tier_and_output_is_raw_sources():
    # flash/fast/standard all cost $0.005 per searchResults call, deep costs
    # $0.05 — the default must never be deep. sourcedAnswer/structured would
    # invoke an LLM instead of returning sources.
    assert LINKUP_DEPTH == "flash"
    assert LINKUP_OUTPUT_TYPE == "searchResults"


@respx.mock
async def test_language_is_never_sent(make_config):
    # The request schema has no language or locale field, so a caller tag has
    # nowhere to go — sending an invented key would be a 400.
    route = respx.post(LINKUP_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=LINKUP_PAYLOAD)
    )
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, "ru-RU")
    body = _body(route)
    assert "language" not in body
    assert "includeImages" not in body  # default (false) is what we want


@respx.mock
@pytest.mark.parametrize(
    ("num_results", "expected"),
    [(0, 1), (-3, 1), (500, LINKUP_MAX_RESULTS_CAP)],
)
async def test_max_results_is_clamped(make_config, num_results, expected):
    # maxResults has a documented minimum of 1; the upper clamp keeps an
    # unvalidated value from going upstream.
    route = respx.post(LINKUP_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=LINKUP_PAYLOAD)
    )
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", num_results, 1, None)
    assert _body(route)["maxResults"] == expected


# -- paging ----------------------------------------------------------------


@respx.mock
async def test_second_page_is_refused_without_a_request(make_config):
    route = respx.post(LINKUP_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=LINKUP_PAYLOAD)
    )
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 3, None)
    assert "no pagination" in str(excinfo.value)
    assert route.call_count == 0


@respx.mock
@pytest.mark.parametrize("page", [0, -5, 1])
async def test_first_or_non_positive_page_is_served(make_config, page):
    route = respx.post(LINKUP_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=LINKUP_PAYLOAD)
    )
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert len(await provider.search(client, "q", 5, page, None)) == 2
    assert route.call_count == 1


# -- failures --------------------------------------------------------------


@respx.mock
async def test_payment_required_is_a_provider_error(make_config):
    # Linkup documents 402 for a missing payment method (details come back in a
    # `payment-required` header); _http.py turns it into a hard failure.
    route = respx.post(LINKUP_SEARCH_ENDPOINT).mock(return_value=httpx.Response(402))
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "out of credits" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_invalid_json_is_a_provider_error(make_config):
    respx.post(LINKUP_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, text="<html>not json</html>")
    )
    provider = LinkupSearch(make_config("linkup_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "invalid JSON" in str(excinfo.value)


def test_requires_an_api_key(make_config):
    with pytest.raises(ValueError):
        LinkupSearch(make_config("linkup_search"))
