"""Octen search provider: response parsing, request body, language, failures.

The payload mirrors the Web Search response shape documented on 2026-09-18 at
https://docs.octen.ai/api-reference/search.md (envelope ``code`` / ``msg`` /
``request_id`` / ``data`` / ``meta``, hits under ``data.results`` with the
snippet in ``highlight``). Network I/O is mocked with respx — no live call is
ever made, we hold no Octen key.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.octen_search import (
    OCTEN_COUNT_MAX,
    OCTEN_SEARCH_ENDPOINT,
    _OCTEN_LANGS,
    OctenSearch,
)

OCTEN_PAYLOAD = {
    "code": 0,
    "msg": "success",
    "request_id": "20260626113629956CJ6G3AG7C2",
    "data": {
        "query": "q",
        "results": [
            {
                "title": "First hit",
                "url": "https://octen.test/1",
                "highlight": "snippet one",
                "authors": "octen.test",
                "time_published": "2026-06-18T00:00:00Z",
                "time_last_crawled": "2026-06-18T14:18:27Z",
                "favicon": "",
            },
            {
                "title": "Second hit",
                "url": "https://octen.test/2",
                "highlight": "snippet two",
            },
        ],
    },
    "meta": {
        "usage": {"num_search_queries": 1, "full_content_extra_count": 0},
        "latency": 1649,
        "warning": None,
    },
}


def _body(route) -> dict:
    return json.loads(route.calls.last.request.content)


# -- response parsing ------------------------------------------------------


@respx.mock
async def test_parses_results_with_highlight_as_snippet(make_config):
    # Octen's snippet field is `highlight`, and the hits sit one level down
    # under `data`.
    respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=OCTEN_PAYLOAD)
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        ("First hit", "https://octen.test/1", "snippet one", "octen_search"),
        ("Second hit", "https://octen.test/2", "snippet two", "octen_search"),
    ]


@respx.mock
async def test_empty_results_list_is_a_normal_empty_answer(make_config):
    respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"code": 0, "msg": "success", "data": {"query": "q", "results": []}}
        )
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"code": 0, "msg": "success"},  # no `data` envelope at all
        {"code": 0, "msg": "success", "data": {"query": "q"}},  # no `results` key
    ],
)
async def test_missing_data_or_results_is_an_empty_result_list(make_config, payload):
    respx.post(OCTEN_SEARCH_ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_items_without_url_are_skipped(make_config):
    respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "results": [
                        {"title": "No url", "url": "", "highlight": "x"},
                        "not-a-dict",
                        {"title": "Good", "url": "https://octen.test/ok", "highlight": "y"},
                    ]
                }
            },
        )
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://octen.test/ok"]


# -- request shape ---------------------------------------------------------


@respx.mock
async def test_request_carries_the_api_key_header_and_the_documented_body(make_config):
    route = respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=OCTEN_PAYLOAD)
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "погода в москве", 9, 1, None)
    request = route.calls.last.request
    assert str(request.url) == OCTEN_SEARCH_ENDPOINT
    assert request.method == "POST"
    assert request.headers["x-api-key"] == "k"
    assert request.headers["Content-Type"] == "application/json"
    # `query` and `count` only: highlight (the snippet), full_content, format and
    # safesearch are all left at their documented defaults.
    assert _body(route) == {"query": "погода в москве", "count": 9}


@respx.mock
@pytest.mark.parametrize(
    ("num_results", "expected"),
    [(0, 1), (-3, 1), (500, OCTEN_COUNT_MAX)],
)
async def test_count_is_clamped_to_the_documented_range(make_config, num_results, expected):
    # `count` is documented as 1..100; outside that it is a 400.
    route = respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=OCTEN_PAYLOAD)
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", num_results, 1, None)
    assert _body(route)["count"] == expected


# -- language --------------------------------------------------------------


@respx.mock
@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("ru", "ru"),
        ("ru-RU", "ru"),  # region dropped, the enum holds bare subtags only
        ("EN", "en"),  # case folded
        ("en_US", "en"),  # underscore is a separator too
        ("zh-CN", "zh"),
        ("pt-BR", "pt"),
        ("  ru  ", "ru"),  # surrounding whitespace stripped
    ],
)
async def test_language_is_sent_as_a_one_element_list(make_config, language, expected):
    # Octen's `language` is an ARRAY of ISO 639-1 codes, not a scalar.
    route = respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=OCTEN_PAYLOAD)
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, language)
    assert _body(route)["language"] == [expected]


@respx.mock
@pytest.mark.parametrize("language", ["xx-YY", "klingon", "uk", "", "   ", None])
async def test_unmappable_language_omits_the_field_entirely(make_config, language):
    # A code outside the closed enum (e.g. Ukrainian, which Octen does not list)
    # is a guaranteed 400 — sending nothing means "no language filter" instead.
    route = respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=OCTEN_PAYLOAD)
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, language)
    assert "language" not in _body(route)


def test_language_enum_matches_the_documented_set():
    # Verbatim from the request schema (18 codes, read 2026-09-18). Note `uk`,
    # `cs`, `sv` and friends are absent — do not add them on a hunch.
    assert _OCTEN_LANGS == {
        "ar", "de", "en", "es", "fr", "hi", "id", "it", "ja",
        "ko", "nl", "pl", "pt", "ru", "th", "tr", "vi", "zh",
    }  # fmt: skip


# -- paging ----------------------------------------------------------------


@respx.mock
async def test_second_page_is_refused_without_a_request(make_config):
    # No page/offset/cursor exists in the schema, so page 2 could only re-run
    # and re-bill the same search.
    route = respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=OCTEN_PAYLOAD)
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 2, None)
    assert "no pagination" in str(excinfo.value)
    assert route.call_count == 0


@respx.mock
@pytest.mark.parametrize("page", [0, -5, 1])
async def test_first_or_non_positive_page_is_served(make_config, page):
    route = respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, json=OCTEN_PAYLOAD)
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert len(await provider.search(client, "q", 5, page, None)) == 2
    assert route.call_count == 1


# -- failures --------------------------------------------------------------


@respx.mock
async def test_payment_required_is_a_provider_error(make_config):
    route = respx.post(OCTEN_SEARCH_ENDPOINT).mock(return_value=httpx.Response(402))
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "out of credits" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_insufficient_balance_403_is_reported_as_out_of_credits(make_config):
    # Octen answers a depleted account with 403 "Insufficient balance in
    # account" (not 402). This module adds no special case — the shared rule in
    # _http.py matches that wording against _CREDIT_MARKERS, so the operator
    # sees a billing state rather than a generic client error.
    route = respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            403, json={"code": 403, "msg": "Insufficient balance in account", "request_id": "r"}
        )
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "out of credits (HTTP 403)" in str(excinfo.value)
    assert "client error" not in str(excinfo.value)
    assert route.call_count == 1  # 4xx is never retried


@respx.mock
async def test_invalid_json_is_a_provider_error(make_config):
    respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(200, text="<html>not json</html>")
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "invalid JSON" in str(excinfo.value)


def test_requires_an_api_key(make_config):
    with pytest.raises(ValueError):
        OctenSearch(make_config("octen_search"))


@respx.mock
async def test_error_envelope_inside_http_200_raises(make_config):
    # s.jina.ai answers application failures with a 200 and a non-success code
    # in the envelope, and jina_search.py guards against exactly that. Octen has
    # the same envelope shape, so an error code here must not be recorded as a
    # successful (and billed) empty answer.
    route = respx.post(OCTEN_SEARCH_ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"code": 1001, "msg": "internal error", "request_id": "r", "data": None}
        )
    )
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "API error" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize(
    ("code", "is_success"),
    [
        (0, True),  # documented success value
        ("0", True),  # a stricter-typing day must not read as a failure
        (None, True),  # absent code: nothing to object to
        (1001, False),
        ("500", False),
    ],
)
async def test_envelope_code_decides_success(make_config, code, is_success):
    # One test for both directions, so it pins the guard itself and not just the
    # string/int normalisation: drop the guard and the two error rows fail.
    payload = {**OCTEN_PAYLOAD}
    if code is None:
        payload.pop("code")
    else:
        payload["code"] = code
    respx.post(OCTEN_SEARCH_ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        if is_success:
            assert await provider.search(client, "q", 5, 1, None)
        else:
            with pytest.raises(ProviderError) as excinfo:
                await provider.search(client, "q", 5, 1, None)
            assert "API error" in str(excinfo.value)


@respx.mock
async def test_highlight_as_a_list_is_joined_not_crashed(make_config):
    # "highlight snippets" is plural in the vendor's own wording and we have no
    # key to settle the shape by observation. A list reaching `.strip()` would
    # raise AttributeError, which Pipeline._one logs as a crashed provider on
    # every query — the instance would never return anything at all.
    payload = {
        "code": 0,
        "msg": "success",
        "request_id": "r",
        "data": {
            "query": "q",
            "results": [
                {
                    "title": "Listed highlights",
                    "url": "https://octen.test/1",
                    "highlight": ["first fragment", "  second fragment  ", "", 7],
                }
            ],
        },
    }
    respx.post(OCTEN_SEARCH_ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    provider = OctenSearch(make_config("octen_search", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.search(client, "q", 5, 1, None)
    assert len(out) == 1
    # Non-strings dropped, blanks dropped, the rest joined and trimmed.
    assert out[0].snippet == "first fragment second fragment"
