"""Brave search provider: response parsing, empty answers, paging, throttle.

The payload mirrors the shape captured from a live call on 2026-08-09 (top-level
``type``/``query``/``mixed``/``web``; the snippet lives in
``web.results[].description``).

Time is faked by swapping the ``time`` module reference inside the provider
module (monkeypatch restores it): patching the real ``time.monotonic`` would also
move the event loop's clock, and the tests must not sleep. Network I/O is mocked
with respx, like the rest of the suite.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from src.providers import brave as brave_module
from src.providers.base import ProviderError
from src.providers.brave import (
    BRAVE_COUNT_MAX,
    BRAVE_ENDPOINT,
    BRAVE_OFFSET_MAX,
    _BRAVE_LANGS,
    _LANG_ALIASES,
    _MIN_INTERVAL_SECONDS,
    BraveSearch,
)

BRAVE_PAYLOAD = {
    "type": "search",
    "query": {"original": "q"},
    "mixed": {"type": "mixed", "main": []},
    "web": {
        "type": "search",
        "family_friendly": True,
        "results": [
            {
                "type": "search_result",
                "title": "First hit",
                "url": "https://brave.test/1",
                "description": "snippet one",
                "language": "en",
                "meta_url": {"hostname": "brave.test"},
            },
            {
                "type": "search_result",
                "title": "Second hit",
                "url": "https://brave.test/2",
                "description": "snippet two",
            },
        ],
    },
}


class _FakeClock:
    """Stand-in for the ``time`` module inside the provider (only ``monotonic``).

    The default start is deliberately SMALLER than ``_MIN_INTERVAL_SECONDS``, so a
    first call only passes with the real ``-inf`` initializer for ``_last_call``
    (with a large start, ``0.0`` would look fine too).
    """

    def __init__(self, now: float = 0.5) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(brave_module, "time", fake)
    return fake


# -- response parsing ------------------------------------------------------


@respx.mock
async def test_parses_results_with_description_as_snippet(make_config, clock):
    # Brave's snippet field is `description` (not `content`/`snippet`).
    assert clock.now < _MIN_INTERVAL_SECONDS  # first call passes only thanks to -inf
    respx.get(BRAVE_ENDPOINT).mock(return_value=httpx.Response(200, json=BRAVE_PAYLOAD))
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        ("First hit", "https://brave.test/1", "snippet one", "brave"),
        ("Second hit", "https://brave.test/2", "snippet two", "brave"),
    ]


@respx.mock
async def test_missing_web_key_is_an_empty_result_list(make_config, clock):
    # A response with no `web` block at all (nothing found) is a normal empty
    # answer, not a crash.
    respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"type": "search", "query": {"original": "q"}})
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 5, 1, None) == []


@respx.mock
async def test_items_without_url_are_skipped(make_config, clock):
    respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {"title": "No url", "url": "", "description": "x"},
                        "not-a-dict",
                        {"title": "Good", "url": "https://brave.test/ok", "description": "y"},
                    ]
                }
            },
        )
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://brave.test/ok"]


# -- request shape ---------------------------------------------------------


@respx.mock
async def test_count_is_capped_at_twenty(make_config, clock):
    # Brave rejects count > 20, so num_results is clamped.
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 50, 1, None)
    request = route.calls.last.request
    assert request.url.params["count"] == str(BRAVE_COUNT_MAX)
    assert request.headers["X-Subscription-Token"] == "k"
    assert request.headers["Accept"] == "application/json"
    assert "search_lang" not in request.url.params  # no language → not sent
    assert "country" not in request.url.params  # never forced, see the provider


@respx.mock
async def test_offset_is_the_zero_based_page_index(make_config, clock):
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, "ru")  # page 1 → offset 0
        first = route.calls.last.request
        clock.advance(_MIN_INTERVAL_SECONDS)
        await provider.search(client, "q", 5, 3, None)  # page 3 → offset 2
        third = route.calls.last.request
    assert first.url.params["offset"] == "0"
    assert first.url.params["search_lang"] == "ru"  # language → search_lang
    assert third.url.params["offset"] == "2"


@respx.mock
async def test_last_servable_page_is_sent_unchanged(make_config, clock):
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, BRAVE_OFFSET_MAX + 1, None)  # page 10
    assert route.calls.last.request.url.params["offset"] == str(BRAVE_OFFSET_MAX)


@respx.mock
async def test_page_beyond_braves_depth_is_refused_without_spending_a_slot(
    make_config, clock
):
    # Page 11+ is a 422 upstream. The provider refuses it locally instead of
    # clamping to page 10 (which would re-serve page 10's hits and bill a
    # query), and the refusal must happen before the throttle is stamped — the
    # very next call, inside the same window, still has to go through.
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, BRAVE_OFFSET_MAX + 2, None)
        assert route.call_count == 0  # never reached the network
        results = await provider.search(client, "q", 5, 1, None)  # same window
    assert "beyond brave's depth" in str(excinfo.value)
    assert "throttled" not in str(excinfo.value)
    assert len(results) == 2
    assert route.call_count == 1  # the slot was still free


@respx.mock
@pytest.mark.parametrize("page", [0, -5])
async def test_non_positive_page_is_clamped_to_the_first_offset(make_config, clock, page):
    # Nothing upstream bounds `page` from below (server.py clamps only
    # num_results), and offset=-1 is a 422 — so page 0 or a negative page must
    # come out as offset 0, not as a rejected request.
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, page, None)
    assert route.calls.last.request.url.params["offset"] == "0"


@respx.mock
async def test_zero_num_results_is_raised_to_one(make_config, clock):
    # Brave rejects count=0; the lower clamp keeps the request valid.
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 0, 1, None)
    assert route.calls.last.request.url.params["count"] == "1"


# -- search_lang normalisation ---------------------------------------------


@respx.mock
@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("ru-RU", "ru"),  # region dropped, Brave has no ru-ru
        ("EN", "en"),  # case folded
        ("en_US", "en"),  # underscore is a separator too
        ("en-GB", "en-gb"),  # regional targeting KEPT where Brave has the code
        ("ja", "jp"),  # Brave spells Japanese "jp"
        ("zh-CN", "zh-hans"),  # Chinese exists only as script variants
        ("zh-TW", "zh-hant"),
        ("pt-BR", "pt-br"),  # Portuguese exists only as regional codes
        ("pt", "pt-br"),
        ("  ru  ", "ru"),  # surrounding whitespace stripped
    ],
)
async def test_language_is_mapped_onto_a_code_brave_accepts(
    make_config, clock, language, expected
):
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, language)
    assert route.calls.last.request.url.params["search_lang"] == expected


@respx.mock
@pytest.mark.parametrize("language", ["xx-YY", "klingon", "", "   ", None])
async def test_unmappable_language_omits_search_lang_entirely(
    make_config, clock, language
):
    # A code Brave does not know is a guaranteed 422, which would drop brave out
    # of the merge. Sending nothing lets Brave apply its own default instead.
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, language)
    assert "search_lang" not in route.calls.last.request.url.params


def test_every_alias_target_is_a_real_brave_code():
    # An alias pointing at a code outside the official enum would send a value
    # that is a guaranteed 422 — worse than sending nothing at all.
    assert set(_LANG_ALIASES.values()) <= _BRAVE_LANGS


# -- throttle --------------------------------------------------------------


@respx.mock
async def test_second_call_within_interval_is_throttled(make_config, clock):
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)
        clock.advance(_MIN_INTERVAL_SECONDS - 0.1)  # still inside the window
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "throttled" in str(excinfo.value)
    assert route.call_count == 1  # the skipped query never hit the network


@respx.mock
async def test_call_passes_exactly_at_the_interval(make_config, clock):
    # The guard is `< interval`, so a gap of exactly _MIN_INTERVAL_SECONDS passes.
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)
        clock.advance(_MIN_INTERVAL_SECONDS)
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://brave.test/1", "https://brave.test/2"]
    assert route.call_count == 2


@respx.mock
async def test_concurrent_calls_let_exactly_one_through(make_config, clock):
    # The check+assignment pair is atomic under asyncio (no await between them),
    # so two searches started together yield one hit and one throttle.
    route = respx.get(BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json=BRAVE_PAYLOAD)
    )
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        outcomes = await asyncio.gather(
            provider.search(client, "q", 5, 1, None),
            provider.search(client, "q", 5, 1, None),
            return_exceptions=True,
        )
    assert len([o for o in outcomes if isinstance(o, list)]) == 1
    assert len([o for o in outcomes if isinstance(o, ProviderError)]) == 1
    assert route.call_count == 1


@respx.mock
async def test_retry_is_disabled_so_one_slot_is_one_upstream_query(make_config, clock):
    # config.retries is 1 (the production default), but a rate-limited provider
    # must not retry: the second attempt lands in the same 1s window.
    route = respx.get(BRAVE_ENDPOINT).mock(return_value=httpx.Response(500))
    config = make_config("brave", api_key="k")
    assert config.retries == 1
    provider = BraveSearch(config)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await provider.search(client, "q", 5, 1, None)
    assert route.call_count == 1  # one attempt, no retry


@respx.mock
async def test_failed_query_still_spends_the_slot(make_config, clock):
    # _last_call is stamped before the request (fail-closed): a failed attempt
    # consumes the window too.
    route = respx.get(BRAVE_ENDPOINT).mock(return_value=httpx.Response(500))
    provider = BraveSearch(make_config("brave", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as first:
            await provider.search(client, "q", 5, 1, None)
        assert "throttled" not in str(first.value)  # it really was the HTTP failure
        clock.advance(_MIN_INTERVAL_SECONDS - 0.1)
        with pytest.raises(ProviderError) as second:
            await provider.search(client, "q", 5, 1, None)
    assert "throttled" in str(second.value)
    assert route.call_count == 1


def test_requires_an_api_key(make_config):
    with pytest.raises(ValueError):
        BraveSearch(make_config("brave"))
