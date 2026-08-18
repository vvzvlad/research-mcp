"""Jina Search (s.jina.ai) provider: envelope parsing and request shape.

The payload mirrors the reader-style envelope documented 2026-08-18:
``{"code": 200, "status": ..., "data": [{"title", "url", "description"}]}``.
Network I/O is mocked with respx, like the rest of the suite.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.jina_search import JINA_SEARCH_ENDPOINT, JinaSearch

JINA_PAYLOAD = {
    "code": 200,
    "status": 20000,
    "data": [
        {
            "title": "First hit",
            "url": "https://jina.test/1",
            "description": "snippet one",
        },
        {
            "title": "Second hit",
            "url": "https://jina.test/2",
            "description": "snippet two",
        },
    ],
}


def _sent_body(route) -> dict:
    """Decode the JSON body of the last captured request on ``route``."""
    return json.loads(route.calls.last.request.content)


# -- response parsing ------------------------------------------------------


@respx.mock
async def test_parses_the_reader_style_envelope(make_config):
    route = respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=JINA_PAYLOAD)
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        ("First hit", "https://jina.test/1", "snippet one", "jina-search"),
        ("Second hit", "https://jina.test/2", "snippet two", "jina-search"),
    ]
    # The SERP-only contract: Bearer auth plus X-Respond-With: no-content
    # (without it s.jina.ai would fetch every hit's page content).
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer k"
    assert request.headers["X-Respond-With"] == "no-content"
    assert request.headers["Accept"] == "application/json"


@respx.mock
async def test_snippet_falls_back_to_content(make_config):
    # Reader-style payloads sometimes carry the snippet in `content` instead of
    # `description` — the provider takes whichever is present.
    respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 200,
                "data": [{"title": "T", "url": "https://jina.test/c", "content": "from content"}],
            },
        )
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert results[0].snippet == "from content"


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"code": 200, "data": []},  # empty data list
        {"code": 200},  # data missing entirely
        {"code": 200, "data": "oops"},  # data of a wrong shape
    ],
)
async def test_missing_or_empty_data_is_an_empty_result_list(make_config, payload):
    # Nothing found (or a degenerate envelope) is a normal empty answer, not a
    # crash — same policy as brave's missing `web` block.
    respx.post(JINA_SEARCH_ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_items_without_url_are_skipped(make_config):
    respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 200,
                "data": [
                    {"title": "No url", "url": "", "description": "x"},
                    "not-a-dict",
                    {"title": "Good", "url": "https://jina.test/ok", "description": "y"},
                ],
            },
        )
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://jina.test/ok"]


@respx.mock
async def test_error_envelope_with_http_200_raises_provider_error(make_config):
    # HTTP 200 does not mean success for this API: the reader-style envelope
    # can carry an application error (`code` != 200). Without the guard the
    # instance would be logged and billed as having worked.
    respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"code": 422, "message": "invalid parameter"}
        )
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "code 422" in str(excinfo.value)


@respx.mock
async def test_string_code_200_envelope_parses_as_success(make_config):
    # Some reader-style envelopes serialize the code as a STRING ("200"); the
    # guard compares it str-normalized, so this must parse as a normal success
    # instead of misfiring as an API error.
    respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": "200",
                "data": [
                    {"title": "T", "url": "https://jina.test/s", "description": "d"}
                ],
            },
        )
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [(r.title, r.url, r.snippet) for r in results] == [
        ("T", "https://jina.test/s", "d")
    ]


@respx.mock
async def test_invalid_json_raises_provider_error(make_config):
    respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, text="<html>not json</html>")
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await provider.search(client, "q", 5, 1, None)


# -- request shape ---------------------------------------------------------


@respx.mock
@pytest.mark.parametrize("num_results", [1, 5, 10])
async def test_num_is_omitted_at_default_serp_depth(make_config, num_results):
    # The docs advise omitting `num` ("may cause latency and exclude
    # specialized result types") — the default depth already covers <= 10.
    route = respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=JINA_PAYLOAD)
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", num_results, 1, None)
    body = _sent_body(route)
    assert body["q"] == "q"
    assert "num" not in body


@respx.mock
async def test_num_is_sent_when_more_than_the_default_depth(make_config):
    route = respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=JINA_PAYLOAD)
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 15, 1, None)
    assert _sent_body(route)["num"] == 15


@respx.mock
async def test_num_is_clamped_to_the_documented_cap(make_config):
    # No documented ceiling for `num`, so it is clamped to the tool's own
    # num_results cap (symmetry with brave/exa) — a future server-side change
    # must not push an unvalidated value upstream.
    route = respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=JINA_PAYLOAD)
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 80, 1, None)
    assert _sent_body(route)["num"] == 50  # clamped to JINA_SEARCH_NUM_MAX


@respx.mock
@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("ru-RU", "ru"),  # primary subtag only
        ("en_US", "en"),  # underscore is a separator too
        ("EN", "en"),  # case folded
        ("  ru  ", "ru"),  # surrounding whitespace stripped
    ],
)
async def test_language_reduces_to_the_two_letter_hl(make_config, language, expected):
    route = respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=JINA_PAYLOAD)
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, language)
    assert _sent_body(route)["hl"] == expected


@respx.mock
@pytest.mark.parametrize("language", ["klingon", "x", "12", "", "   ", None])
async def test_non_two_letter_language_omits_hl(make_config, language):
    # jina validates hl as a two-letter code; anything that does not reduce to
    # one is omitted so its own default applies instead of a validation error.
    route = respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=JINA_PAYLOAD)
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, language)
    assert "hl" not in _sent_body(route)


@respx.mock
async def test_page_is_omitted_for_the_first_page_and_sent_beyond(make_config):
    route = respx.post(JINA_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=JINA_PAYLOAD)
    )
    provider = JinaSearch(make_config("jina-search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)
        first = _sent_body(route)
        await provider.search(client, "q", 5, 3, None)
        third = _sent_body(route)
    assert "page" not in first
    assert third["page"] == 3
    # `gl` is never sent — the tool contract has no country input.
    assert "gl" not in first
    assert "gl" not in third


def test_requires_an_api_key(make_config):
    # s.jina.ai blocks keyless access, so a keyless instance must fail to
    # build (and get skipped by the loader) instead of erroring at runtime.
    with pytest.raises(ValueError):
        JinaSearch(make_config("jina-search"))
