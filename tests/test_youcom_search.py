"""You.com search provider: response parsing, request shape, paging, failures.

The payload mirrors the response documented on 2026-09-18 at
https://you.com/docs/api-reference/search (``results.web[]`` with ``url`` /
``title`` / ``description`` / ``snippets``). Network I/O is mocked with respx,
like the rest of the suite.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.youcom_search import (
    YOUCOM_COUNT_MAX,
    YOUCOM_ENDPOINT,
    YOUCOM_OFFSET_MAX,
    _YOUCOM_LANGS,
    YouComSearch,
)

YOUCOM_PAYLOAD = {
    "results": {
        "web": [
            {
                "url": "https://youcom.test/1",
                "title": "First hit",
                "description": "snippet one",
                "snippets": ["fragment a", "fragment b"],
                "page_age": "2026-09-01",
            },
            {
                "url": "https://youcom.test/2",
                "title": "Second hit",
                "description": "snippet two",
                "snippets": ["fragment c"],
            },
        ],
        "news": [{"url": "https://youcom.test/news", "title": "News", "description": "n"}],
    },
    "metadata": {"search_uuid": "uuid", "query": "q", "latency": 0.4},
}


def _sent_body(route) -> dict:
    """Decode the JSON body of the last captured request on ``route``."""
    return json.loads(route.calls.last.request.content)


# -- response parsing ------------------------------------------------------


@respx.mock
async def test_parses_web_results_with_description_as_snippet(make_config):
    # The snippet field is `description` (a single string), not the `snippets`
    # array next to it; the `news` section is ignored.
    respx.post(YOUCOM_ENDPOINT).mock(return_value=httpx.Response(200, json=YOUCOM_PAYLOAD))
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        ("First hit", "https://youcom.test/1", "snippet one", "youcom"),
        ("Second hit", "https://youcom.test/2", "snippet two", "youcom"),
    ]


@respx.mock
async def test_snippets_array_is_used_when_description_is_missing(make_config):
    respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": {
                    "web": [
                        {
                            "url": "https://youcom.test/1",
                            "title": "No description",
                            "snippets": ["fragment a", "  ", "fragment b"],
                        }
                    ]
                }
            },
        )
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.snippet for r in results] == ["fragment a fragment b"]


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"results": {"news": []}, "metadata": {}},  # nothing in the web section
        {"metadata": {}},  # no `results` object at all
        {"results": {"web": []}},  # present but empty
    ],
)
async def test_response_without_web_results_is_a_normal_empty_answer(make_config, payload):
    respx.post(YOUCOM_ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_items_without_url_are_skipped(make_config):
    respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": {
                    "web": [
                        {"title": "No url", "url": "", "description": "x"},
                        "not-a-dict",
                        {"title": "Good", "url": "https://youcom.test/ok", "description": "y"},
                    ]
                }
            },
        )
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://youcom.test/ok"]


# -- request shape ---------------------------------------------------------


@respx.mock
async def test_posts_json_body_with_api_key_header(make_config):
    route = respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(200, json=YOUCOM_PAYLOAD)
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "тест", 5, 1, None)
    request = route.calls.last.request
    assert request.method == "POST"
    assert str(request.url) == YOUCOM_ENDPOINT
    assert request.headers["X-API-Key"] == "k"
    assert request.headers["Content-Type"] == "application/json"
    assert _sent_body(route) == {"query": "тест", "count": 5, "offset": 0}
    # No language given → the key is absent; country is never sent at all.
    assert "language" not in _sent_body(route)
    assert "country" not in _sent_body(route)


@respx.mock
async def test_count_is_capped_at_the_documented_hundred(make_config):
    route = respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(200, json=YOUCOM_PAYLOAD)
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 500, 1, None)
    assert _sent_body(route)["count"] == YOUCOM_COUNT_MAX


@respx.mock
async def test_zero_num_results_is_raised_to_one(make_config):
    route = respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(200, json=YOUCOM_PAYLOAD)
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 0, 1, None)
    assert _sent_body(route)["count"] == 1


@respx.mock
async def test_offset_is_the_zero_based_page_index(make_config):
    route = respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(200, json=YOUCOM_PAYLOAD)
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)  # page 1 → offset 0
        assert _sent_body(route)["offset"] == 0
        await provider.search(client, "q", 5, 3, None)  # page 3 → offset 2
        assert _sent_body(route)["offset"] == 2
        # Page 10 is the deepest one the API serves and goes through unchanged.
        await provider.search(client, "q", 5, YOUCOM_OFFSET_MAX + 1, None)
    assert _sent_body(route)["offset"] == YOUCOM_OFFSET_MAX


@respx.mock
@pytest.mark.parametrize("page", [0, -5])
async def test_non_positive_page_is_clamped_to_the_first_offset(make_config, page):
    route = respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(200, json=YOUCOM_PAYLOAD)
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, page, None)
    assert _sent_body(route)["offset"] == 0


@respx.mock
async def test_page_beyond_the_documented_depth_is_refused(make_config):
    # `offset` is capped at 9 upstream, so page 11+ is refused locally instead
    # of being clamped to page 10 (which would re-serve page 10's hits and bill
    # another call).
    route = respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(200, json=YOUCOM_PAYLOAD)
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, YOUCOM_OFFSET_MAX + 2, None)
    assert "beyond you.com's depth" in str(excinfo.value)
    assert route.call_count == 0  # never reached the network


# -- language normalisation ------------------------------------------------


@respx.mock
@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("ru-RU", "RU"),  # region dropped, the enum has no RU-RU
        ("ru", "RU"),  # case folded to the enum's spelling
        ("en_US", "EN"),  # underscore is a separator too
        ("en-GB", "EN-GB"),  # regional targeting KEPT where the enum has it
        ("pt-br", "PT-BR"),
        ("zh-hant", "ZH-HANT"),  # Chinese exists only as script variants
        ("  ja  ", "JA"),  # surrounding whitespace stripped
    ],
)
async def test_language_is_mapped_onto_an_accepted_enum_value(
    make_config, language, expected
):
    route = respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(200, json=YOUCOM_PAYLOAD)
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, language)
    assert _sent_body(route)["language"] == expected


@respx.mock
@pytest.mark.parametrize("language", ["xx-YY", "klingon", "zh", "pt", "", "   ", None])
async def test_unmappable_language_omits_the_field_entirely(make_config, language):
    # A value outside the enum is a guaranteed 422, which would drop you.com out
    # of the merge; sending nothing lets the API apply its default (EN). `zh` and
    # `pt` land here because the enum carries only their script/regional forms.
    route = respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(200, json=YOUCOM_PAYLOAD)
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, language)
    assert "language" not in _sent_body(route)


def test_the_documented_language_enum_is_upper_case():
    # The provider upper-cases the caller's tag before the lookup, so a
    # lower-case entry in the table would be unreachable.
    assert all(code == code.upper() for code in _YOUCOM_LANGS)


# -- failures --------------------------------------------------------------


@respx.mock
async def test_payment_required_is_a_provider_error(make_config):
    # 402 carries {"error", "message", "upgrade_url"}; _http turns it into a
    # ProviderError without retrying, so the pipeline fails over.
    route = respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(
            402,
            json={
                "error": "payment_required",
                "message": "The account cannot make paid API requests.",
                "upgrade_url": "https://you.com/platform",
            },
        )
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "out of credits" in str(excinfo.value)
    assert route.call_count == 1  # 402 is not retried


@respx.mock
async def test_non_json_body_is_a_provider_error(make_config):
    respx.post(YOUCOM_ENDPOINT).mock(
        return_value=httpx.Response(200, text="<html>gateway</html>")
    )
    provider = YouComSearch(make_config("youcom", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "invalid JSON" in str(excinfo.value)


def test_requires_an_api_key(make_config):
    with pytest.raises(ValueError):
        YouComSearch(make_config("youcom"))
