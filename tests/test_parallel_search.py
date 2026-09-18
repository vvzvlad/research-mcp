"""Parallel Search provider: response parsing, request body, paging, failures.

The payload mirrors the ``V1SearchResponse`` shape documented on 2026-09-18 at
https://docs.parallel.ai/api-reference/search-api/search (``search_id`` /
``results`` / ``warnings`` / ``usage`` / ``session_id``; the snippet lives in
``results[].excerpts`` as a LIST of strings). Network I/O is mocked with respx —
no live call is ever made, we hold no Parallel key.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.parallel_search import (
    PARALLEL_MAX_RESULTS_CAP,
    PARALLEL_SEARCH_ENDPOINT,
    _SNIPPET_MAX_CHARS,
    ParallelSearch,
)

PARALLEL_PAYLOAD = {
    "search_id": "search_8a911eb27c7a4afaa20d0d9dc98d07c0",
    "session_id": "search_8a911eb27c7a4afaa20d0d9dc98d07c0",
    "warnings": None,
    "usage": [{"name": "sku_search", "count": 1}],
    "results": [
        {
            "url": "https://parallel.test/1",
            "title": "First hit",
            "publish_date": "2025-11-19",
            "excerpts": ["excerpt one", "excerpt two ... (content truncated)"],
        },
        {
            "url": "https://parallel.test/2",
            "title": None,
            "publish_date": None,
            "excerpts": ["only excerpt"],
        },
    ],
}


def _body(route) -> dict:
    return json.loads(route.calls.last.request.content)


# -- response parsing ------------------------------------------------------


@respx.mock
async def test_parses_results_and_joins_excerpts_into_one_snippet(make_config):
    # Parallel's snippet field is `excerpts` — a list, unlike brave's single
    # `description` string, so the parts are joined into one snippet.
    respx.post(PARALLEL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=PARALLEL_PAYLOAD)
    )
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        (
            "First hit",
            "https://parallel.test/1",
            "excerpt one\n\nexcerpt two ... (content truncated)",
            "parallel_search",
        ),
        ("", "https://parallel.test/2", "only excerpt", "parallel_search"),
    ]


@respx.mock
async def test_empty_results_list_is_a_normal_empty_answer(make_config):
    respx.post(PARALLEL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"search_id": "s", "session_id": "s", "results": []}
        )
    )
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_missing_results_key_is_an_empty_result_list(make_config):
    respx.post(PARALLEL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"search_id": "s", "session_id": "s"})
    )
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_items_without_url_or_excerpts_are_handled(make_config):
    # `url` and `excerpts` are the two required per-result fields, but a hit
    # without a url is unusable and a malformed `excerpts` must not crash.
    respx.post(PARALLEL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"url": "", "title": "No url", "excerpts": ["x"]},
                    "not-a-dict",
                    {"url": "https://parallel.test/ok", "title": "Ok", "excerpts": None},
                    {
                        "url": "https://parallel.test/mixed",
                        "title": "Mixed",
                        "excerpts": ["  kept  ", "", 7],
                    },
                ]
            },
        )
    )
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [(r.url, r.snippet) for r in results] == [
        ("https://parallel.test/ok", ""),
        ("https://parallel.test/mixed", "kept"),
    ]


# -- request shape ---------------------------------------------------------


@respx.mock
async def test_request_carries_the_api_key_header_and_the_documented_body(make_config):
    route = respx.post(PARALLEL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=PARALLEL_PAYLOAD)
    )
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "python asyncio timeouts", 7, 1, None)
    request = route.calls.last.request
    assert str(request.url) == PARALLEL_SEARCH_ENDPOINT
    assert request.method == "POST"
    assert request.headers["x-api-key"] == "k"
    assert request.headers["Content-Type"] == "application/json"
    # search_queries is the only required field; max_results is nested under
    # advanced_settings (a top-level max_results would be a 422 —
    # additionalProperties: false).
    assert _body(route) == {
        "search_queries": ["python asyncio timeouts"],
        "advanced_settings": {"max_results": 7},
    }


@respx.mock
async def test_language_is_never_sent(make_config):
    # The schema has no language field at all; `location` is a COUNTRY code, so
    # a language tag has nowhere to go.
    route = respx.post(PARALLEL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=PARALLEL_PAYLOAD)
    )
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, "ru-RU")
    body = _body(route)
    assert "location" not in body["advanced_settings"]
    assert "language" not in body
    assert "objective" not in body  # not guessed from the query either
    assert "mode" not in body  # Parallel applies its documented default


@respx.mock
@pytest.mark.parametrize(
    ("num_results", "expected"),
    [(0, 1), (-3, 1), (200, PARALLEL_MAX_RESULTS_CAP)],
)
async def test_max_results_is_clamped(make_config, num_results, expected):
    route = respx.post(PARALLEL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=PARALLEL_PAYLOAD)
    )
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", num_results, 1, None)
    assert _body(route)["advanced_settings"]["max_results"] == expected


# -- paging ----------------------------------------------------------------


@respx.mock
async def test_second_page_is_refused_without_a_request(make_config):
    # Parallel has no pagination: page 2 would repeat page 1's hits at full
    # price, so it must fail instead of silently re-serving them.
    route = respx.post(PARALLEL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=PARALLEL_PAYLOAD)
    )
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 2, None)
    assert "no pagination" in str(excinfo.value)
    assert route.call_count == 0  # never reached the network


@respx.mock
@pytest.mark.parametrize("page", [0, -5, 1])
async def test_first_or_non_positive_page_is_served(make_config, page):
    # Nothing upstream bounds `page` from below, and there is no offset to send,
    # so page 0 / a negative page is just the one page this API has.
    route = respx.post(PARALLEL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=PARALLEL_PAYLOAD)
    )
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert len(await provider.search(client, "q", 5, page, None)) == 2
    assert route.call_count == 1


# -- failures --------------------------------------------------------------


@respx.mock
async def test_payment_required_is_a_provider_error(make_config):
    route = respx.post(PARALLEL_SEARCH_ENDPOINT).mock(return_value=httpx.Response(402))
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "out of credits" in str(excinfo.value)
    assert route.call_count == 1  # 402 is never retried


@respx.mock
async def test_invalid_json_is_a_provider_error(make_config):
    respx.post(PARALLEL_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, text="<html>not json</html>")
    )
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "invalid JSON" in str(excinfo.value)


def test_requires_an_api_key(make_config):
    with pytest.raises(ValueError):
        ParallelSearch(make_config("parallel_search"))


@respx.mock
async def test_long_excerpts_are_truncated_to_the_snippet_cap(make_config):
    # Parallel returns compressed page extracts, not one-line summaries: several
    # per hit, each potentially long. Untrimmed, one web_search answer could
    # carry tens of kilobytes — every other provider's snippet is a sentence or
    # two. The cap matches the one rerank.py already applies to a result's text.
    payload = {
        "results": [
            {
                "url": "https://parallel.test/long",
                "title": "Long one",
                "excerpts": ["x" * 900, "y" * 900],
            }
        ]
    }
    respx.post(PARALLEL_SEARCH_ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert len(results[0].snippet) == _SNIPPET_MAX_CHARS


@respx.mock
async def test_a_snippet_exactly_at_the_cap_is_not_touched(make_config):
    # The boundary, not a short snippet: a snippet of exactly the cap length
    # must come back whole. (A plain short-snippet test would only repeat what
    # the parsing test above already asserts, and would pass with no cap at all.)
    payload = {
        "results": [
            {
                "url": "https://parallel.test/exact",
                "title": "Exact",
                "excerpts": ["z" * _SNIPPET_MAX_CHARS],
            }
        ]
    }
    respx.post(PARALLEL_SEARCH_ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    provider = ParallelSearch(make_config("parallel_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert results[0].snippet == "z" * _SNIPPET_MAX_CHARS
