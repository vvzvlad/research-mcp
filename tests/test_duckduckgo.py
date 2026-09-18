"""DuckDuckGo (keyless): SERP parsing, uddg unwrapping, request body, failures,
the local throttle, and the zero-config build.

The HTML below is a trimmed copy of a real ``html.duckduckgo.com/html/``
response: ``.result`` rows whose ``a.result__a`` carries title + wrapped href and
whose ``.result__snippet`` carries the snippet, plus the "more results" row that
holds the next-page form and must NOT be mistaken for a hit.

Two rules this provider lives by, both pinned here:

- it must never return ``[]`` for a block, a captcha, a redesign or a page
  where not one row's href could be read — only for a search that ran and
  left nothing:
  DuckDuckGo's own "no results" page, or a page carrying nothing but ads
  (there is no status code to tell these apart, so the markup decides);
- its throttle skips, never sleeps (same design as searxng/brave — ``Pipeline.search``
  awaits every provider, so a waiting one stalls the whole ``web_search``).

Time is faked by swapping the ``time`` module reference inside the provider
module (monkeypatch restores it): patching the real ``time.monotonic`` would also
move the event loop's clock. Network I/O is mocked with respx — no live call is
ever made to DuckDuckGo.
"""

from __future__ import annotations

import asyncio
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from src.pipeline import Pipeline
from src.providers import duckduckgo as duckduckgo_module
from src.providers.base import ProviderError
from src.providers.duckduckgo import (
    DDG_ENDPOINT,
    DDG_PAGE_SIZE,
    _MIN_INTERVAL_SECONDS,
    DuckDuckGoSearch,
)
from tests.conftest import _clear_provider_env

# Two hits. The first href is protocol-relative and wraps a plain url; the second
# wraps a url that carries its own %28/%29 escapes (they stay escaped: the
# wrapper is decoded exactly once, never twice — see _unwrap_href).
SERP_HTML = """<!DOCTYPE html>
<html><body>
<div class="serp__results"><div id="links" class="results">

<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html&amp;rut=1a2b">
         asyncio &mdash; Asynchronous I/O</a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=1a2b">
      <b>asyncio</b> is a library to write concurrent code using async/await.</a>
    <div class="clear"></div>
  </div>
</div>

<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FPython_%2528programming_language%2529&amp;rut=3c4d">
         Python (programming language)</a>
    </h2>
    <a class="result__snippet" href="#">Python is a high-level programming language.</a>
  </div>
</div>

<div class="result result--more">
  <div class="result__a">More results</div>
  <form action="/html/" method="post"><input type="hidden" name="s" value="30"></form>
</div>

</div></div>
</body></html>
"""

# DuckDuckGo's own empty answer: no rows, but its "no results" marker is present.
NO_RESULTS_HTML = """<html><body>
<div class="serp__results"><div id="links" class="results">
<div class="no-results">No results.</div>
</div></div>
</body></html>
"""

# A block/captcha interstitial: no rows AND no marker. Indistinguishable from a
# markup change from here — which is exactly why both must raise.
BLOCKED_HTML = """<html><body>
<div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div>
<div class="anomaly-modal__description">Please complete the challenge below.</div>
</body></html>
"""


class _FakeClock:
    """Stand-in for the ``time`` module inside the provider (only ``monotonic``).

    The default start is deliberately SMALLER than ``_MIN_INTERVAL_SECONDS``: with
    a large start (say 1000.0) the first-call test would stay green even if
    ``_last_call`` were initialized to ``0.0``, because the gap would clear the
    interval on its own. From 1.0 only the real ``-inf`` initializer lets the
    first query through.
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
    monkeypatch.setattr(duckduckgo_module, "time", fake)
    return fake


def _provider(make_config) -> DuckDuckGoSearch:
    """The provider as the loader builds it: a name and nothing else."""
    return DuckDuckGoSearch(make_config("duckduckgo"))


def _serp(html: str = SERP_HTML, status: int = 200):
    return respx.post(DDG_ENDPOINT).mock(return_value=httpx.Response(status, text=html))


def _sent(route) -> dict[str, list[str]]:
    """The form-encoded body of the last request, parsed."""
    return parse_qs(route.calls.last.request.content.decode())


# -- SERP parsing ----------------------------------------------------------


@respx.mock
async def test_parses_title_url_and_snippet(make_config, clock):
    _serp()
    async with httpx.AsyncClient() as client:
        results = await _provider(make_config).search(client, "asyncio", 5, 1, None)
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        (
            "asyncio — Asynchronous I/O",
            "https://docs.python.org/3/library/asyncio.html",
            "asyncio is a library to write concurrent code using async/await.",
            "duckduckgo",
        ),
        (
            "Python (programming language)",
            "https://en.wikipedia.org/wiki/Python_%28programming_language%29",
            "Python is a high-level programming language.",
            "duckduckgo",
        ),
    ]


@respx.mock
async def test_the_more_results_row_is_not_a_hit(make_config, clock):
    # `.result--more` carries the next-page form and shares the `result` class
    # token with real hits; it has no `a.result__a`, which is how it is dropped.
    _serp()
    async with httpx.AsyncClient() as client:
        results = await _provider(make_config).search(client, "q", 5, 1, None)
    assert len(results) == 2
    assert all("More results" not in r.title for r in results)


@respx.mock
async def test_unwraps_the_uddg_redirector(make_config, clock):
    # Every href on the page is wrapped in /l/?uddg=<url-encoded url>. Handing
    # that wrapper to the model (or to read_page) would be useless, so the real
    # target must come back out, with the target's own escapes left alone.
    _serp()
    async with httpx.AsyncClient() as client:
        results = await _provider(make_config).search(client, "q", 5, 1, None)
    assert [r.url for r in results] == [
        "https://docs.python.org/3/library/asyncio.html",
        "https://en.wikipedia.org/wiki/Python_%28programming_language%29",
    ]
    assert not any("uddg" in r.url for r in results)


@respx.mock
async def test_target_escapes_survive_the_unwrapping(make_config, clock):
    # The wrapper is decoded exactly ONCE. A second decode would eat the
    # target's own escapes and point at a different resource: %23 would become
    # a fragment, %26 an extra parameter, %2F a path separator.
    _serp(
        '<html><body><div class="result"><a class="result__a" '
        'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.test%2Fs%3Fid%3Dx%2523frag%26'
        'p%3Da%2526b&amp;rut=9f">Tricky</a></div></body></html>'
    )
    async with httpx.AsyncClient() as client:
        results = await _provider(make_config).search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://ex.test/s?id=x%23frag&p=a%26b"]


@respx.mock
async def test_ad_rows_are_not_returned_as_results(make_config, clock):
    # An ad row is shaped like an organic hit (`result result--ad`) but its href
    # is DDG's own click wrapper: no uddg, no scheme. Handing it to the model
    # would pass advertising off as a search result, and read_page would then
    # refuse the scheme-less url with an unhelpful error.
    _serp(
        '<html><body>'
        '<div class="result result--ad"><a class="result__a" '
        'href="//duckduckgo.com/y.js?ad_domain=shop.test&amp;u3=https%3A%2F%2Fbing.test">'
        "Buy laptops cheap</a></div>"
        '<div class="result"><a class="result__a" href="https://organic.test/page">'
        "Organic</a></div>"
        "</body></html>"
    )
    async with httpx.AsyncClient() as client:
        results = await _provider(make_config).search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://organic.test/page"]


@respx.mock
async def test_unreadable_hrefs_raise_instead_of_looking_empty(make_config, clock):
    # Organic rows are there, but the redirector parameter has been renamed —
    # the shape a DuckDuckGo markup change takes. Dropping those rows quietly
    # would report a broken parser as "nothing found", and on a keyless
    # deployment duckduckgo is the ONLY search provider, so nobody would notice.
    _serp(
        '<html><body><div class="result"><a class="result__a" '
        'href="//duckduckgo.com/l/?uddgx=https%3A%2F%2Fex.test">Hit</a></div>'
        '<div class="result"><a class="result__a" href="/relative/path">Hit 2</a>'
        "</div></body></html>"
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError, match="unrecognised href"):
            await _provider(make_config).search(client, "q", 5, 1, None)


@respx.mock
async def test_one_unreadable_href_does_not_discard_the_readable_ones(make_config, clock):
    # The boundary between "we got results" and "the format changed": if DDG
    # rewrites only part of the page, what we could read still counts. Raising
    # here would turn a partial change into a total outage of the only search
    # provider a keyless deployment has.
    _serp(
        '<html><body><div class="result"><a class="result__a" '
        'href="https://ok.test/a">Readable</a></div>'
        '<div class="result"><a class="result__a" '
        'href="//duckduckgo.com/l/?uddgx=https%3A%2F%2Fex.test">Renamed</a>'
        "</div></body></html>"
    )
    async with httpx.AsyncClient() as client:
        results = await _provider(make_config).search(client, "q", 5, 1, None)
    assert [r.url for r in results] == ["https://ok.test/a"]


@respx.mock
async def test_a_page_of_ads_only_is_empty_not_blocked(make_config, clock):
    # The page answered — it just had nothing organic on it. Raising here would
    # report a working search as a block.
    _serp(
        '<html><body><div class="result result--ad"><a class="result__a" '
        'href="//duckduckgo.com/y.js?ad_domain=shop.test">Ad</a></div></body></html>'
    )
    async with httpx.AsyncClient() as client:
        assert await _provider(make_config).search(client, "q", 5, 1, None) == []


@respx.mock
async def test_href_without_uddg_is_used_as_is(make_config, clock):
    # DuckDuckGo serves some hrefs unwrapped; those must survive untouched
    # rather than be dropped for lacking the parameter.
    _serp(
        '<html><body><div class="result"><a class="result__a" '
        'href="https://plain.test/page">Plain</a>'
        '<a class="result__snippet">snippet</a></div></body></html>'
    )
    async with httpx.AsyncClient() as client:
        results = await _provider(make_config).search(client, "q", 5, 1, None)
    assert [(r.url, r.snippet) for r in results] == [("https://plain.test/page", "snippet")]


@respx.mock
async def test_row_without_a_snippet_yields_an_empty_snippet(make_config, clock):
    _serp(
        '<html><body><div class="result"><a class="result__a" '
        'href="https://plain.test/page">Plain</a></div></body></html>'
    )
    async with httpx.AsyncClient() as client:
        results = await _provider(make_config).search(client, "q", 5, 1, None)
    assert [(r.url, r.snippet) for r in results] == [("https://plain.test/page", "")]


# -- failure honesty: never [] for a failure -------------------------------


@respx.mock
async def test_no_results_page_returns_an_empty_list(make_config, clock):
    # The ONE case where [] is the honest answer: DuckDuckGo says so itself.
    _serp(NO_RESULTS_HTML)
    async with httpx.AsyncClient() as client:
        assert await _provider(make_config).search(client, "q", 5, 1, None) == []


@respx.mock
async def test_blocked_page_raises_instead_of_returning_empty(make_config, clock):
    # No rows and no "no results" marker → we were blocked (or the markup
    # changed). Returning [] would log a broken provider as a successful search.
    _serp(BLOCKED_HTML)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await _provider(make_config).search(client, "q", 5, 1, None)
    assert "duckduckgo" in str(excinfo.value)
    assert "markup" in str(excinfo.value)


@respx.mock
async def test_unknown_markup_raises(make_config, clock):
    # Same rule for a redesigned SERP: valid HTML, recognisable to nobody.
    _serp("<html><body><div class='hit'><a href='https://x.test'>X</a></div></body></html>")
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await _provider(make_config).search(client, "q", 5, 1, None)


@respx.mock
async def test_empty_body_raises(make_config, clock):
    # An empty body is not parseable HTML at all — still a failure, not [].
    _serp("")
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await _provider(make_config).search(client, "q", 5, 1, None)
    assert "unparseable" in str(excinfo.value)


@respx.mock
async def test_http_202_is_a_rate_limit_block_not_a_success(make_config, clock):
    # DuckDuckGo answers a burst with 202 + a "Ratelimit" body. request_with_retry
    # passes any 2xx through as success, so the provider itself has to catch this
    # — otherwise the block would be reported as "the markup changed".
    _serp("Ratelimit", status=202)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await _provider(make_config).search(client, "q", 5, 1, None)
    assert "rate limited (HTTP 202)" in str(excinfo.value)


@respx.mock
async def test_http_403_is_a_provider_error(make_config, clock):
    # The shared policy already covers the hard statuses; pinned so a future
    # "return [] on 4xx" cannot sneak in.
    _serp(BLOCKED_HTML, status=403)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await _provider(make_config).search(client, "q", 5, 1, None)


# -- request body: q / kl / s ----------------------------------------------


@respx.mock
async def test_query_is_form_encoded_and_default_region_is_wt_wt(make_config, clock):
    route = _serp()
    async with httpx.AsyncClient() as client:
        await _provider(make_config).search(client, "погода в москве", 5, 1, None)
    sent = _sent(route)
    assert sent["q"] == ["погода в москве"]
    assert sent["kl"] == ["wt-wt"]  # no language → no regional bias
    assert "s" not in sent  # page 1 omits the offset entirely


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("en", "us-en"),
        ("ru", "ru-ru"),
        ("de", "de-de"),
        ("fr", "fr-fr"),
        ("es", "es-es"),
        ("it", "it-it"),
        # The LLM's spellings, normalised exactly like _brave_lang does.
        ("ru-RU", "ru-ru"),
        ("EN", "us-en"),
        ("en_US", "us-en"),
        ("  de  ", "de-de"),
        # No mapping → wt-wt, never a guessed region.
        ("ja", "wt-wt"),
        ("zh-CN", "wt-wt"),
        ("", "wt-wt"),
    ],
)
@respx.mock
async def test_language_maps_to_a_kl_region(make_config, clock, language, expected):
    route = _serp()
    async with httpx.AsyncClient() as client:
        await _provider(make_config).search(client, "q", 5, 1, language)
    assert _sent(route)["kl"] == [expected]


@pytest.mark.parametrize(("page", "offset"), [(2, 30), (3, 60), (5, 120)])
@respx.mock
async def test_deeper_pages_send_the_result_offset(make_config, clock, page, offset):
    # The endpoint's own next-page form posts `s` as a RESULT offset.
    route = _serp()
    async with httpx.AsyncClient() as client:
        await _provider(make_config).search(client, "q", 5, page, None)
    assert _sent(route)["s"] == [str(offset)]
    assert offset == (page - 1) * DDG_PAGE_SIZE


@pytest.mark.parametrize("page", [0, -1, 1])
@respx.mock
async def test_page_one_or_below_sends_no_offset(make_config, clock, page):
    # Nothing upstream clamps `page`; a negative offset would be a bad request.
    route = _serp()
    async with httpx.AsyncClient() as client:
        await _provider(make_config).search(client, "q", 5, page, None)
    assert "s" not in _sent(route)


@respx.mock
async def test_browser_user_agent_is_sent(make_config, clock):
    # Without it the endpoint serves the block page instead of a SERP.
    route = _serp()
    async with httpx.AsyncClient() as client:
        await _provider(make_config).search(client, "q", 5, 1, None)
    assert "Mozilla/5.0" in route.calls.last.request.headers["User-Agent"]


# -- local throttle: SKIP, never sleep -------------------------------------


@respx.mock
async def test_first_call_passes(make_config, clock):
    # Nothing has run yet (_last_call is -inf) → the very first query goes out.
    # The clock starts at 1.0 (< _MIN_INTERVAL_SECONDS) on purpose, so this fails
    # if _last_call is ever initialized to 0.0 instead of -inf.
    assert clock.now < _MIN_INTERVAL_SECONDS
    _serp()
    async with httpx.AsyncClient() as client:
        results = await _provider(make_config).search(client, "q", 5, 1, None)
    assert len(results) == 2


@respx.mock
async def test_second_call_within_interval_is_throttled(make_config, clock):
    route = _serp()
    provider = _provider(make_config)
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)
        clock.advance(_MIN_INTERVAL_SECONDS - 1)  # still inside the window
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 5, 1, None)
    assert "throttled" in str(excinfo.value)
    assert route.call_count == 1  # the skipped query never hit the network


@respx.mock
async def test_call_passes_again_after_the_interval(make_config, clock):
    route = _serp()
    provider = _provider(make_config)
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)
        clock.advance(_MIN_INTERVAL_SECONDS + 1)
        await provider.search(client, "q", 5, 1, None)
    assert route.call_count == 2


@respx.mock
async def test_call_passes_exactly_at_the_interval(make_config, clock):
    # The boundary itself is allowed: the guard is `< interval`, so a gap of
    # exactly _MIN_INTERVAL_SECONDS must pass (pins `<` against `<=`).
    route = _serp()
    provider = _provider(make_config)
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)
        clock.advance(_MIN_INTERVAL_SECONDS)
        await provider.search(client, "q", 5, 1, None)
    assert route.call_count == 2


@respx.mock
async def test_concurrent_calls_let_exactly_one_through(make_config, clock):
    # The comment in the provider claims the check+assignment pair is atomic
    # under asyncio and needs no lock. Pin it: two searches started together (as
    # Pipeline.search would, via gather) must produce one hit and one throttle,
    # with a single upstream request.
    route = _serp()
    provider = _provider(make_config)
    async with httpx.AsyncClient() as client:
        outcomes = await asyncio.gather(
            provider.search(client, "q", 5, 1, None),
            provider.search(client, "q", 5, 1, None),
            return_exceptions=True,
        )
    passed = [o for o in outcomes if isinstance(o, list)]
    throttled = [o for o in outcomes if isinstance(o, ProviderError)]
    assert len(passed) == 1
    assert len(passed[0]) == 2
    assert len(throttled) == 1
    assert "throttled" in str(throttled[0])
    assert route.call_count == 1


@respx.mock
async def test_throttle_skips_instead_of_sleeping(make_config, clock):
    # The whole point of SKIP semantics: a taken slot costs no wall time, because
    # Pipeline.search awaits every provider. The fake clock never advances by
    # itself, so a sleeping implementation would hang here rather than raise.
    route = _serp()
    provider = _provider(make_config)
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 5, 1, None)
        # A real sleep would need _MIN_INTERVAL_SECONDS of wall time; this whole
        # call has to finish inside a fraction of that.
        async with asyncio.timeout(1.0):
            with pytest.raises(ProviderError):
                await provider.search(client, "q", 5, 1, None)
    assert route.call_count == 1


@respx.mock
async def test_retry_is_disabled_so_one_slot_is_one_upstream_query(make_config, clock):
    # config.retries is 1 (the production default), but duckduckgo must not use
    # it: a retry 0.3s later would put a second query inside the same slot — the
    # very burst pattern the throttle exists to prevent.
    route = _serp("", status=500)
    config = make_config("duckduckgo")
    assert config.retries == 1
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await DuckDuckGoSearch(config).search(client, "q", 5, 1, None)
    assert route.call_count == 1  # one attempt, no retry


@respx.mock
async def test_failed_query_still_spends_the_slot(make_config, clock):
    # _last_call is stamped before the request on purpose (fail-closed): a failed
    # attempt still consumes the window, so the next query is throttled.
    route = _serp("", status=500)
    provider = _provider(make_config)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as first:
            await provider.search(client, "q", 5, 1, None)
        assert "throttled" not in str(first.value)  # it really was the HTTP failure
        clock.advance(_MIN_INTERVAL_SECONDS - 1)
        with pytest.raises(ProviderError) as second:
            await provider.search(client, "q", 5, 1, None)
    assert "throttled" in str(second.value)
    assert route.call_count == 1


# -- zero-config end to end ------------------------------------------------


@respx.mock
async def test_search_works_with_no_env_at_all(monkeypatch, settings, capture_logs, clock):
    # The point of the whole provider: an empty environment still searches. No
    # key, no SearXNG, no variable of any kind — build the pipeline and get hits.
    _clear_provider_env(monkeypatch)
    _serp()

    pipe = Pipeline.build(settings)
    try:
        assert pipe.search_names == ["duckduckgo"]
        results = await pipe.search("asyncio", num_results=10, page=1, language=None)
    finally:
        await pipe.aclose()

    assert [r.url for r in results] == [
        "https://docs.python.org/3/library/asyncio.html",
        "https://en.wikipedia.org/wiki/Python_%28programming_language%29",
    ]
    line = next(m for m in capture_logs if m.startswith("search query="))
    assert "providers=['duckduckgo']" in line
    assert "paid_calls=0" in line  # keyless → never billed
