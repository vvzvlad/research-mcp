"""SearXNG local throttle: one query per interval, enforced by SKIPPING.

Why it exists (prod measurement 2026-08-09): the only engine left alive in our
SearXNG instance is DuckDuckGo, which blocks for 7-8 minutes after ~4 rapid
queries but answers 8 of 8 when they are 45s apart.

The throttle must never sleep — ``Pipeline.search`` gathers all search providers
concurrently and awaits all of them, so a waiting searxng would stall the whole
``web_search``. It must also raise ``ProviderError`` rather than return ``[]``,
so the pipeline drops it from the ``providers=[...]`` log field.

Time is faked by swapping the ``time`` module reference inside the provider
module (monkeypatch restores it): patching the real ``time.monotonic`` would also
move the event loop's clock. Network I/O is mocked with respx, like the rest of
the suite.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from src.pipeline import Pipeline
from src.providers import searxng as searxng_module
from src.providers.base import ProviderError
from src.providers.searxng import _MIN_INTERVAL_SECONDS, SearxngSearch
from tests.conftest import _clear_provider_env

SEARXNG_URL = "http://searxng.test"
SEARXNG_PAYLOAD = {"results": [{"url": "https://sx.test/1", "title": "Sx", "content": "a"}]}
EXA_PAYLOAD = {"results": [{"url": "https://e.test/1", "title": "Exa"}]}


class _FakeClock:
    """Stand-in for the ``time`` module inside the provider (only ``monotonic``).

    The default start is deliberately SMALLER than ``_MIN_INTERVAL_SECONDS``: with
    a large start (say 1000.0) the first-call test would stay green even if
    ``_last_call`` were initialized to ``0.0``, because ``1000 - 0 >= 45``. From
    1.0 only the real ``-inf`` initializer lets the first query through.
    """

    def __init__(self, now: float = 1.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(searxng_module, "time", fake)
    return fake


@respx.mock
async def test_first_call_passes(make_config, clock):
    # Nothing has run yet (_last_call is -inf) → the very first query goes out.
    # The clock starts at 1.0 (< _MIN_INTERVAL_SECONDS) on purpose, so this fails
    # if _last_call is ever initialized to 0.0 instead of -inf.
    assert clock.now < _MIN_INTERVAL_SECONDS
    respx.get(f"{SEARXNG_URL}/search").mock(
        return_value=httpx.Response(200, json=SEARXNG_PAYLOAD)
    )
    provider = SearxngSearch(make_config("searxng", url=SEARXNG_URL))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://sx.test/1"]


@respx.mock
async def test_second_call_within_interval_is_throttled(make_config, clock):
    route = respx.get(f"{SEARXNG_URL}/search").mock(
        return_value=httpx.Response(200, json=SEARXNG_PAYLOAD)
    )
    provider = SearxngSearch(make_config("searxng", url=SEARXNG_URL))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)
        clock.advance(_MIN_INTERVAL_SECONDS - 1)  # still inside the window
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "throttled" in str(excinfo.value)
    assert route.call_count == 1  # the skipped query never hit the network


@respx.mock
async def test_call_passes_again_after_the_interval(make_config, clock):
    route = respx.get(f"{SEARXNG_URL}/search").mock(
        return_value=httpx.Response(200, json=SEARXNG_PAYLOAD)
    )
    provider = SearxngSearch(make_config("searxng", url=SEARXNG_URL))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)
        clock.advance(_MIN_INTERVAL_SECONDS + 1)
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://sx.test/1"]
    assert route.call_count == 2


@respx.mock
async def test_call_passes_exactly_at_the_interval(make_config, clock):
    # The boundary itself is allowed: the guard is `< interval`, so a gap of
    # exactly _MIN_INTERVAL_SECONDS must pass (pins `<` against `<=`).
    route = respx.get(f"{SEARXNG_URL}/search").mock(
        return_value=httpx.Response(200, json=SEARXNG_PAYLOAD)
    )
    provider = SearxngSearch(make_config("searxng", url=SEARXNG_URL))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)
        clock.advance(_MIN_INTERVAL_SECONDS)
        results = await provider.search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://sx.test/1"]
    assert route.call_count == 2


@respx.mock
async def test_concurrent_calls_let_exactly_one_through(make_config, clock):
    # The comment in the provider claims the check+assignment pair is atomic
    # under asyncio and needs no lock. Pin it: two searches started together (as
    # Pipeline.search would, via gather) must produce one hit and one throttle,
    # with a single upstream request.
    route = respx.get(f"{SEARXNG_URL}/search").mock(
        return_value=httpx.Response(200, json=SEARXNG_PAYLOAD)
    )
    provider = SearxngSearch(make_config("searxng", url=SEARXNG_URL))
    async with httpx.AsyncClient() as client:
        outcomes = await asyncio.gather(
            provider.search(client, "q", 5, 1, None),
            provider.search(client, "q", 5, 1, None),
            return_exceptions=True,
        )
    passed = [o for o in outcomes if isinstance(o, list)]
    throttled = [o for o in outcomes if isinstance(o, ProviderError)]
    assert len(passed) == 1
    assert [r.url for r in passed[0]] == ["https://sx.test/1"]
    assert len(throttled) == 1
    assert "throttled" in str(throttled[0])
    assert route.call_count == 1


@respx.mock
async def test_retry_is_disabled_so_one_slot_is_one_upstream_query(make_config, clock):
    # config.retries is 1 (the production default), but searxng must not use it:
    # a retry 0.3s later would put a second DDG query inside the same slot — the
    # very burst pattern the throttle exists to prevent.
    route = respx.get(f"{SEARXNG_URL}/search").mock(return_value=httpx.Response(500))
    config = make_config("searxng", url=SEARXNG_URL)
    assert config.retries == 1
    provider = SearxngSearch(config)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await provider.search(client, "q", 5, 1, None)
    assert route.call_count == 1  # one attempt, no retry


@respx.mock
async def test_failed_query_still_spends_the_slot(make_config, clock):
    # _last_call is stamped before the request on purpose (fail-closed): a failed
    # attempt still consumes the window, so the next query is throttled.
    route = respx.get(f"{SEARXNG_URL}/search").mock(return_value=httpx.Response(500))
    provider = SearxngSearch(make_config("searxng", url=SEARXNG_URL))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as first:
            await provider.search(client, "q", 5, 1, None)
        assert "throttled" not in str(first.value)  # it really was the HTTP failure
        clock.advance(_MIN_INTERVAL_SECONDS - 1)
        with pytest.raises(ProviderError) as second:
            await provider.search(client, "q", 5, 1, None)
    assert "throttled" in str(second.value)
    assert route.call_count == 1


@respx.mock
async def test_throttled_searxng_does_not_break_the_search(
    monkeypatch, settings, capture_logs, clock
):
    # The skip must cost nothing: the second search still returns the other
    # provider's hits, and searxng drops out of the providers=[...] log field
    # instead of pretending it took part with an empty result list.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", SEARXNG_URL)
    monkeypatch.setenv("EXA_API_KEY", "k")
    searxng_route = respx.get(f"{SEARXNG_URL}/search").mock(
        return_value=httpx.Response(200, json=SEARXNG_PAYLOAD)
    )
    respx.post("https://api.exa.ai/search").mock(
        return_value=httpx.Response(200, json=EXA_PAYLOAD)
    )

    pipe = Pipeline.build(settings)
    try:
        first = await pipe.search("q", num_results=10, page=1, language=None)
        clock.advance(2.0)  # our median gap between searches
        second = await pipe.search("q", num_results=10, page=1, language=None)
    finally:
        await pipe.aclose()

    assert {r.url for r in first} == {"https://sx.test/1", "https://e.test/1"}
    assert [r.url for r in second] == ["https://e.test/1"]  # exa still served it
    assert searxng_route.call_count == 1  # the second search never reached searxng

    lines = [m for m in capture_logs if m.startswith("search query=")]
    assert len(lines) == 2
    assert "'searxng'" in lines[0]
    assert "'searxng'" not in lines[1]  # excluded, not silently counted as used
    assert "'exa'" in lines[1]
