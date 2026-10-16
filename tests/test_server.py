"""Server wiring: the 4 tools are registered, descriptions are the verbatim
English texts (plus the cross-references), and each tool delegates to the
pipeline without ever raising.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from src.pipeline import ReadFailed, ReadItem, ReadOutcome, SearchOutcome, SearchReadOutcome
from src.providers.base import SearchResult
from src.server import _heartbeat, build_server


class FakePipeline:
    """Stand-in for Pipeline so server tests need no network or ENV.

    ``markdown`` overrides the page body every read returns — the hook the
    truncation tests use to hand the tools a page longer than the budget.
    """

    def __init__(self, markdown: str | None = None, thin: bool = False, teaser: bool = False):
        self.closed = False
        self._markdown = markdown
        self._thin = thin
        self._teaser = teaser
        # Arguments of the last search / search_and_read call (the caps and the
        # over-fetch are computed in the tool, so this is where a test sees them).
        self.search_args: tuple | None = None
        self.search_and_read_args: tuple | None = None

    async def aclose(self):
        self.closed = True

    def _body(self, url: str) -> str:
        return self._markdown if self._markdown is not None else f"# Markdown of {url}"

    async def search(self, query, num_results, language):
        self.search_args = (query, num_results, language)
        hit = SearchResult(title="Hit", url="https://x.test", snippet="snip", source="searxng")
        return SearchOutcome(
            results=[hit],
            attempted=["searxng", "serper"],
            answered=["searxng"],
            empty=[],
            failed=["serper"],
            failed_reasons=["timeout"],
            hits_before_dedup=1,
            reranked=False,
            elapsed_ms=3000,
        )

    async def read(self, url):
        if "boom" in url:
            # Two of the three failures are bot protection → that is the reason
            # the batch and the per-url entry must report.
            raise ReadFailed(
                "страница недоступна всеми способами (тест)",
                tried=["trafilatura", "jina", "crawl4ai"],
                failures=[
                    ("trafilatura", "network"),
                    ("jina", "bot-protection"),
                    ("crawl4ai", "bot-protection"),
                ],
            )
        return ReadOutcome(
            markdown=self._body(url),
            provider="jina",
            tried=["trafilatura", "jina"],
            failures=[("trafilatura", "empty")],
            thin=self._thin,
            elapsed_ms=1500,
            teaser=self._teaser,
        )

    async def search_and_read(self, query, num_results, language, candidates):
        # Canned outcome: the wave/over-fetch logic itself is the pipeline's and
        # is covered against respx in tests/test_pipeline.py.
        self.search_and_read_args = (query, num_results, language, candidates)
        search = await self.search(query, candidates, language)
        items = [
            ReadItem(
                title="Hit",
                url="https://x.test",
                snippet="snip",
                ok=True,
                markdown=self._body("https://x.test"),
                thin=self._thin,
                teaser=self._teaser,
            ),
            ReadItem(
                title="Boom",
                url="https://boom.test",
                snippet="nope",
                ok=False,
                error="страница недоступна всеми способами (тест)",
                reason="bot-protection",
            ),
        ][:num_results]
        return SearchReadOutcome(
            items=items,
            search=search,
            candidates=candidates,
            read_attempts=len(items) + 1,  # one failure was topped up by a wave
        )


class EmptySearchPipeline(FakePipeline):
    """A pipeline whose search brings back nothing, either way it can happen.

    ``dead=True`` means every instance raised (a broken search), ``dead=False``
    that they all answered with zero hits (an honestly empty result).
    """

    def __init__(self, dead: bool):
        super().__init__()
        self._dead = dead

    async def search(self, query, num_results, language):
        self.search_args = (query, num_results, language)
        return SearchOutcome(
            results=[],
            attempted=["searxng", "serper"],
            answered=[],
            empty=[] if self._dead else ["searxng", "serper"],
            failed=["searxng", "serper"] if self._dead else [],
            failed_reasons=["timeout", "rate-limit"] if self._dead else [],
            hits_before_dedup=0,
            reranked=False,
            elapsed_ms=900,
        )

    async def search_and_read(self, query, num_results, language, candidates):
        self.search_and_read_args = (query, num_results, language, candidates)
        search = await self.search(query, candidates, language)
        return SearchReadOutcome(items=[], search=search, candidates=0, read_attempts=0)


@pytest.fixture
def server(settings):
    return build_server(settings, pipeline=FakePipeline())


async def test_four_tools_registered(server):
    tools = {t.name for t in await server.list_tools()}
    assert tools == {"web_search", "read_page", "read_pages", "search_and_read"}


def test_server_instructions_advertise_youtube(server):
    # With deferred tool schemas the client shows only the server instructions,
    # so they are the one place that tells the model read_page opens YouTube.
    assert "YouTube" in server.instructions
    assert "read_page" in server.instructions


async def test_search_tools_take_no_page_argument(server):
    # No paging: a deeper result list is what num_results is for.
    by_name = {t.name: t for t in await server.list_tools()}
    for name in ("web_search", "search_and_read"):
        assert "page" not in by_name[name].inputSchema["properties"], name


async def test_descriptions_are_verbatim_english(server):
    # The texts are LLM-facing product, not documentation: they are pinned here
    # so a rewrite is a deliberate act. Russian survives only in the status
    # lines rendered by src/formatting.py, never in a description.
    by_name = {t.name: t for t in await server.list_tools()}
    ws = by_name["web_search"].description
    assert ws.startswith("Web search. Aggregates several sources")
    # The source list must not name a stale roster: duckduckgo is the keyless
    # default and the paid vendors come and go with their keys.
    assert "DuckDuckGo out of the box" in ws
    assert "This is ONLY search, it does NOT read pages." in ws
    rp = by_name["read_page"].description
    assert rp.startswith("Download ONE web page or PDF by url")
    # The description must not promise that scans come back empty or that no OCR
    # happens: since the scan fall-through, a .pdf url on a keyed deployment does
    # buy the jina OCR tier, and a scan never returns "" — it returns either text
    # or NO_TEXT_LAYER_NOTICE.
    assert "no OCR" not in rp
    assert "come back empty" not in rp
    assert "recognition" in rp
    assert "For several urls in one call — read_pages." in rp
    rps = by_name["read_pages"].description
    assert rps.startswith("Download SEVERAL pages or PDFs in one call (up to 20)")
    assert "{url, ok, markdown|error, reason}" in rps
    assert "summary is one status line for the batch" in rps
    sar = by_name["search_and_read"].description
    assert sar.startswith("Web search + the content of the top results in ONE call.")
    # No description may quote a runtime string that is still Russian: the
    # truncation marker is described, never reproduced.
    for description in (ws, rp, rps, sar):
        assert "содержимое обрезано" not in description


async def test_descriptions_cross_reference_each_other(server):
    # The four texts must form a routing graph, not four independent blurbs:
    # each says when to take IT and which of the others to take instead.
    by_name = {t.name: t.description for t in await server.list_tools()}
    for name, description in by_name.items():
        assert "When to take this one" in description, name

    ws, rp, rps, sar = (
        by_name["web_search"],
        by_name["read_page"],
        by_name["read_pages"],
        by_name["search_and_read"],
    )
    # web_search: only links/snippets; content wanted → the combined tool.
    assert "links and snippets only" in ws
    assert "search_and_read" in ws
    # read_page: one known url; the alternatives for the other two cases.
    assert "exactly one url" in rp
    assert "read_pages" in rp and "search_and_read" in rp
    # read_pages: urls already known; unknown urls → the combined tool.
    assert "the urls are already known" in rps
    assert "search_and_read" in rps and "read_page" in rps
    # search_and_read: the default for research, and when NOT to take it.
    assert "web_search" in sar and "read_pages" in sar and "read_page" in sar


async def _call(server, name: str, args: dict[str, Any]):
    result = await server.call_tool(name, args)
    if isinstance(result, tuple):
        return result[1] if len(result) > 1 else result[0]
    return result


def _as_list(structured: Any) -> list:
    """Normalize a list-returning tool's structured output to a plain list.

    FastMCP may wrap a list result as ``{"result": [...]}``; unwrap that.
    """
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    assert isinstance(structured, list)
    return structured


def _text(structured: Any) -> str:
    """The plain string a string-returning tool produced (FastMCP wraps it)."""
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    return str(structured)


def _pages(structured: Any) -> list:
    """The ``pages`` list out of read_pages' ``{summary, pages}`` answer."""
    if isinstance(structured, dict) and "result" in structured:
        structured = structured["result"]
    assert isinstance(structured, dict)
    return _as_list(structured["pages"])


def _results(structured: Any) -> list:
    """The ``results`` list out of search_and_read's ``{summary, results}``."""
    if isinstance(structured, dict) and "result" in structured:
        structured = structured["result"]
    assert isinstance(structured, dict)
    return _as_list(structured["results"])


def _summary(structured: Any) -> str:
    if isinstance(structured, dict) and "result" in structured:
        structured = structured["result"]
    return structured["summary"]


async def test_web_search_formats_results(server):
    out = str(await _call(server, "web_search", {"query": "hello"}))
    assert "Hit" in out
    assert "https://x.test" in out


async def test_web_search_appends_the_status_line(server):
    out = str(await _call(server, "web_search", {"query": "hello"}))
    assert (
        "Статус поиска: ответили 1 из 2 (searxng); хитов 1 → результатов 1; "
        "пусто: 0; ошибок: 1 (таймаут); 3.0 с"
    ) in out


async def test_read_page_returns_markdown(server):
    out = str(await _call(server, "read_page", {"url": "https://a.test"}))
    assert "Markdown of https://a.test" in out


async def test_read_page_appends_the_status_line_after_a_separator(server):
    # The winning provider and the number of providers tried must reach the
    # model, visually separated from the page text.
    out = _text(await _call(server, "read_page", {"url": "https://a.test"}))
    assert out == (
        "# Markdown of https://a.test\n\n---\n"
        "Статус чтения: jina (провайдеров испробовано: 2); 1.5 с"
    )


async def test_read_page_error_is_string(server):
    out = str(await _call(server, "read_page", {"url": "https://boom.test"}))
    assert "недоступна" in out


async def test_read_page_failure_carries_the_reason_category(server):
    # A failed read must not degrade to the raw aggregated exception text: the
    # category is what tells the model whether another attempt could ever help.
    out = _text(await _call(server, "read_page", {"url": "https://boom.test"}))
    assert out == (
        "страница недоступна всеми способами (тест)\n\n---\n"
        "Статус чтения: не прочитано (бот-защита); провайдеров испробовано: 3"
    )


async def test_read_pages_batch_mixed(server):
    out = await _call(
        server, "read_pages", {"urls": ["https://a.test", "https://boom.test"]}
    )
    text = str(out)
    assert "a.test" in text
    assert "boom.test" in text
    # The good url has markdown; the boom url is an error entry.
    assert "Markdown of https://a.test" in text
    assert "недоступна" in text


async def test_read_pages_summary_and_per_page_reason(server):
    out = await _call(
        server, "read_pages", {"urls": ["https://a.test", "https://boom.test"]}
    )
    assert _summary(out) == "Статус чтения: прочитано 1 из 2; ошибок: 1 (бот-защита)"
    pages = _pages(out)
    good = next(p for p in pages if p["url"] == "https://a.test")
    bad = next(p for p in pages if p["url"] == "https://boom.test")
    assert good["ok"] is True
    assert "reason" not in good  # a page that opened has nothing to explain
    assert bad["ok"] is False
    assert "недоступна" in bad["error"]  # the raw message stays
    assert bad["reason"] == "бот-защита"  # ...plus the category next to it


async def test_read_pages_respects_hard_limit(settings):
    # The cap is a hard constant (READ_PAGES_MAX=20), NOT a setting, so the
    # tool's "up to 20" promise stays true regardless of env overrides.
    from src.server import READ_PAGES_MAX

    assert READ_PAGES_MAX == 20
    srv = build_server(settings, pipeline=FakePipeline())
    urls = [f"https://a.test/{i}" for i in range(READ_PAGES_MAX + 2)]
    out = await _call(srv, "read_pages", {"urls": urls})
    items = _pages(out)
    assert len(items) == READ_PAGES_MAX
    processed = {item["url"] for item in items}
    assert "https://a.test/0" in processed
    assert f"https://a.test/{READ_PAGES_MAX - 1}" in processed
    # The 21st and 22nd urls are dropped.
    assert f"https://a.test/{READ_PAGES_MAX}" not in processed


async def test_read_pages_emits_summary_log(settings, capture_logs):
    srv = build_server(settings, pipeline=FakePipeline())
    await _call(srv, "read_pages", {"urls": ["https://a.test", "https://boom.test"]})
    line = next((m for m in capture_logs if m.startswith("read_pages count=")), None)
    assert line is not None
    assert "count=2" in line
    assert "ok=1" in line  # one good url, one boom (error)


# -- the per-page content budget -------------------------------------------


def _capped_settings(settings, max_chars: int):
    """The same settings with a small per-page budget for the batch tools."""
    return settings.model_copy(update={"read_batch_max_chars": max_chars})


LONG_PAGE = "A" * 200


async def test_read_pages_truncates_with_an_explicit_marker(settings):
    # 200 chars, budget 50 → 150 dropped, and the model is told exactly that.
    srv = build_server(_capped_settings(settings, 50), pipeline=FakePipeline(markdown=LONG_PAGE))
    pages = _pages(await _call(srv, "read_pages", {"urls": ["https://a.test"]}))
    markdown = pages[0]["markdown"]
    assert markdown == "A" * 50 + "\n\n[содержимое обрезано на 150 символах]"


async def test_read_page_is_never_truncated(settings):
    # Asking for ONE url is asking for all of it: the batch budget must not
    # leak into read_page, marker or not.
    srv = build_server(_capped_settings(settings, 50), pipeline=FakePipeline(markdown=LONG_PAGE))
    out = _text(await _call(srv, "read_page", {"url": "https://a.test"}))
    assert out.startswith(LONG_PAGE)
    assert "обрезано" not in out


async def test_search_and_read_truncates_page_content(settings):
    srv = build_server(_capped_settings(settings, 50), pipeline=FakePipeline(markdown=LONG_PAGE))
    out = await _call(srv, "search_and_read", {"query": "q", "num_results": 1})
    entry = _results(out)[0]
    assert entry["markdown"] == "A" * 50 + "\n\n[содержимое обрезано на 150 символах]"


THIN_MARKER = "\n\n[текст короткий (7 симв.) — возможно, это не вся страница]"


async def test_read_pages_marks_a_thin_fallback(settings):
    srv = build_server(settings, pipeline=FakePipeline(markdown="# short", thin=True))
    pages = _pages(await _call(srv, "read_pages", {"urls": ["https://a.test"]}))
    assert pages[0]["markdown"] == "# short" + THIN_MARKER


async def test_search_and_read_marks_a_thin_fallback(settings):
    srv = build_server(settings, pipeline=FakePipeline(markdown="# short", thin=True))
    out = await _call(srv, "search_and_read", {"query": "q", "num_results": 1})
    assert _results(out)[0]["markdown"] == "# short" + THIN_MARKER


# A paywall teaser is flagged as one in the batch tools too, not as "short".
TEASER_MARKER = "\n\n[похоже на анонс за пейволлом — это только начало статьи, не вся она]"


async def test_read_pages_flags_a_paywall_teaser(settings):
    srv = build_server(
        settings, pipeline=FakePipeline(markdown="# teaser", thin=True, teaser=True)
    )
    pages = _pages(await _call(srv, "read_pages", {"urls": ["https://a.test"]}))
    assert pages[0]["markdown"] == "# teaser" + TEASER_MARKER


async def test_search_and_read_flags_a_paywall_teaser(settings):
    srv = build_server(
        settings, pipeline=FakePipeline(markdown="# teaser", thin=True, teaser=True)
    )
    out = await _call(srv, "search_and_read", {"query": "q", "num_results": 1})
    assert _results(out)[0]["markdown"] == "# teaser" + TEASER_MARKER


# -- the combined tool ------------------------------------------------------


async def test_search_and_read_returns_results_with_content(server):
    out = await _call(server, "search_and_read", {"query": "hello"})
    entries = _results(out)
    good, bad = entries[0], entries[1]
    # A search hit and its content in ONE entry — the point of the tool.
    assert good == {
        "title": "Hit",
        "url": "https://x.test",
        "snippet": "snip",
        "ok": True,
        "markdown": "# Markdown of https://x.test",
    }
    assert bad["ok"] is False
    assert "недоступна" in bad["error"]
    assert bad["reason"] == "бот-защита"  # the category, like in read_pages
    assert "markdown" not in bad


async def test_search_and_read_summary_carries_both_status_lines(server):
    out = await _call(server, "search_and_read", {"query": "hello"})
    search_line, read_line = _summary(out).split("\n")
    assert search_line == (
        "Статус поиска: ответили 1 из 2 (searxng); хитов 1 → результатов 1; "
        "пусто: 0; ошибок: 1 (таймаут); 3.0 с"
    )
    # 1 page of 3 attempted reads, out of 12 over-fetched candidates.
    assert read_line == (
        "Статус чтения: прочитано 1 из 3 (кандидатов: 12); ошибок: 2 (бот-защита)"
    )


async def test_search_and_read_says_in_words_when_the_search_broke(settings):
    # An empty `results` list reads as "the topic does not exist" whichever way
    # the search died, so the combined tool must carry the same prose web_search
    # gives — otherwise the model reports "nothing found" on a total outage.
    srv = build_server(settings, pipeline=EmptySearchPipeline(dead=True))
    out = await _call(srv, "search_and_read", {"query": "hello"})
    summary = _summary(out)
    assert _results(out) == []
    assert "не выполнен: все поисковые провайдеры вернули ошибку" in summary
    assert "повторить запрос позже" in summary
    # The status lines still ride under the prose.
    assert "Статус поиска:" in summary and "Статус чтения:" in summary


async def test_search_and_read_says_in_words_when_nothing_matched(settings):
    # The other half of the same fork: providers answered, the topic is empty.
    srv = build_server(settings, pipeline=EmptySearchPipeline(dead=False))
    summary = _summary(await _call(srv, "search_and_read", {"query": "hello"}))
    assert "ничего не найдено" in summary
    assert "все поисковые провайдеры вернули ошибку" not in summary


async def test_search_and_read_over_fetches_candidates(settings):
    # The tool asks the search for more candidates than the pages it owes,
    # because some urls will not open: min(n * 2 + 2, SEARCH_RESULTS_MAX).
    from src.server import SEARCH_RESULTS_MAX

    pipe = FakePipeline()
    srv = build_server(settings, pipeline=pipe)
    await _call(srv, "search_and_read", {"query": "q", "num_results": 3})
    assert pipe.search_and_read_args[1] == 3  # pages requested
    assert pipe.search_and_read_args[3] == 8  # 3 * 2 + 2 candidates
    # ...and the over-fetch never exceeds what one search may return.
    await _call(srv, "search_and_read", {"query": "q", "num_results": 40})
    assert pipe.search_and_read_args[1] == 20  # capped at READ_PAGES_MAX
    assert pipe.search_and_read_args[3] == min(42, SEARCH_RESULTS_MAX)


async def test_numeric_string_arguments_are_accepted(settings):
    # Weak models send numbers as strings; that must not be a validation error.
    pipe = FakePipeline()
    srv = build_server(settings, pipeline=pipe)

    await _call(srv, "search_and_read", {"query": "q", "num_results": "8"})
    assert pipe.search_and_read_args[1] == 8

    await _call(srv, "web_search", {"query": "q", "num_results": "8"})
    assert pipe.search_args == ("q", 8, None)

    # The tolerance is only for numbers: a non-numeric string is still a clean
    # validation error, not something silently coerced.
    with pytest.raises(ToolError):
        await _call(srv, "web_search", {"query": "q", "num_results": "восемь"})


class _FakeCtx:
    """Duck-typed stand-in for the SDK ``Context`` in the heartbeat tests.

    ``token`` is what the fake request's ``progressToken`` holds — ``None``
    means the client never opted into progress, and ``broken=True`` makes the
    fake session fail every send the way a closed stream would. ``raising=True``
    mimics the real Context outside a request: its ``request_context`` property
    raises ``ValueError``, which the heartbeat must treat like no token. The
    fake session records every send; ``two_beats`` is set on the second one.
    """

    def __init__(self, token: int | None, *, raising: bool = False, broken: bool = False):
        self.beats: list[float] = []
        self.calls: list[dict] = []
        self._raising = raising
        self._broken = broken
        self.two_beats = asyncio.Event()
        self._request_context = SimpleNamespace(
            meta=SimpleNamespace(progressToken=token),
            session=self,
            request_id=42,
        )

    @property
    def request_context(self):
        if self._raising:
            raise ValueError("Context is not available outside of a request")
        return self._request_context

    @property
    def request_id(self):
        return str(self._request_context.request_id)

    async def send_progress_notification(
        self, *, progress_token, progress, related_request_id=None, **kwargs
    ):
        self.calls.append(
            {
                "progress_token": progress_token,
                "progress": progress,
                "related_request_id": related_request_id,
            }
        )
        self.beats.append(progress)
        if self._broken:
            raise RuntimeError("session stream closed")
        if len(self.beats) >= 2:
            self.two_beats.set()


async def test_heartbeat_beats_until_the_body_returns(monkeypatch):
    monkeypatch.setattr("src.server.HEARTBEAT_SECONDS", 0.01)
    ctx = _FakeCtx(token=7)
    async with _heartbeat(ctx):
        # No blind sleep: the body waits for the second beat to arrive, so a
        # slow event loop delays the test instead of failing it.
        await asyncio.wait_for(ctx.two_beats.wait(), 5)
    assert ctx.beats == sorted(ctx.beats)
    after_last_beat = len(ctx.beats)
    await asyncio.sleep(0.02)
    # No beat fires once the body has returned: the beat task is cancelled in
    # the finally block, not left running in the background.
    assert len(ctx.beats) == after_last_beat


async def test_heartbeat_ties_the_beat_to_the_tool_call(monkeypatch):
    # The notification must carry related_request_id: without it the transport
    # cannot attach it to this call's POST response — it goes to the standalone
    # GET stream, which a session with exhausted reconnects does not have, and
    # the beat is silently lost there.
    monkeypatch.setattr("src.server.HEARTBEAT_SECONDS", 0.01)
    ctx = _FakeCtx(token=7)
    async with _heartbeat(ctx):
        await asyncio.wait_for(ctx.two_beats.wait(), 5)
    assert all(call["related_request_id"] == ctx.request_id for call in ctx.calls)
    assert all(call["progress_token"] == 7 for call in ctx.calls)


async def test_heartbeat_without_a_progress_token_starts_nothing(monkeypatch):
    monkeypatch.setattr("src.server.HEARTBEAT_SECONDS", 0.01)
    ctx = _FakeCtx(token=None)
    async with _heartbeat(ctx):
        await asyncio.sleep(0.03)
    assert ctx.beats == []


async def test_heartbeat_outside_a_request_starts_nothing(monkeypatch):
    # The SDK Context raises on request_context outside a request; the heartbeat
    # must take that as "no progress subscription", not crash the tool.
    monkeypatch.setattr("src.server.HEARTBEAT_SECONDS", 0.01)
    ctx = _FakeCtx(token=7, raising=True)
    async with _heartbeat(ctx):
        await asyncio.sleep(0.03)
    assert ctx.beats == []


async def test_heartbeat_beat_failures_stop_the_beat(monkeypatch):
    # A send that raises (session stream closed under the call) must neither
    # fail the body nor keep beating: exactly one failed send, then silence.
    monkeypatch.setattr("src.server.HEARTBEAT_SECONDS", 0.01)
    ctx = _FakeCtx(token=7, broken=True)
    with pytest.raises(TimeoutError):
        async with _heartbeat(ctx):
            await asyncio.wait_for(ctx.two_beats.wait(), 0.3)
    assert len(ctx.calls) == 1


async def test_context_parameter_is_hidden_from_the_tool_schema(server):
    # The ctx argument is wiring, not model-facing API: FastMCP must keep it out
    # of the input schema of every tool that uses the heartbeat.
    by_name = {t.name: t for t in await server.list_tools()}
    for name in ("read_page", "read_pages", "search_and_read"):
        assert "ctx" not in by_name[name].inputSchema["properties"], name


class _SlowReadPipeline(FakePipeline):
    """FakePipeline whose every long call runs until the client has been notified.

    The ``done`` event is set by the test's progress callback once two beats
    have arrived — so the test has no timing assumptions at all: if the
    notifications never reach the client, the call blocks until the outer
    ``asyncio.wait_for`` fails the test. Both await paths wait: ``read`` is
    what read_page/read_pages spend their time in, while the canned
    ``search_and_read`` never reaches ``read``.
    """

    def __init__(self, done: asyncio.Event):
        super().__init__()
        self._done = done

    async def read(self, url):
        await self._done.wait()
        return await super().read(url)

    async def search_and_read(self, query, num_results, language, candidates):
        await self._done.wait()
        return await super().search_and_read(query, num_results, language, candidates)


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("read_page", {"url": "https://slow.test"}),
        ("read_pages", {"urls": ["https://slow.test"]}),
        ("search_and_read", {"query": "q"}),
    ],
)
async def test_long_calls_notify_progress_over_a_real_client(settings, monkeypatch, tool, args):
    # The user's failure, end to end: the client passes a progressToken (the way
    # Claude Code does when it tracks a tool call), the call runs long — and the
    # heartbeats must reach the client as notifications/progress, which resets
    # its idle timer instead of leaving the call silent until it aborts.
    monkeypatch.setattr("src.server.HEARTBEAT_SECONDS", 0.02)
    from mcp.shared.memory import create_connected_server_and_client_session

    beats: list[tuple[float, float | None, str | None]] = []
    got_two = asyncio.Event()

    def on_progress(progress: float, total: float | None, message: str | None):
        beats.append((progress, total, message))
        if len(beats) >= 2:
            got_two.set()

    async with create_connected_server_and_client_session(
        build_server(settings, pipeline=_SlowReadPipeline(got_two))
    ) as session:
        await asyncio.wait_for(
            session.call_tool(tool, args, progress_callback=on_progress), 10
        )
    assert len(beats) >= 2, "fewer than two progress notifications reached the client"
    assert all(total is None for _, total, _ in beats)
    assert [p for p, _, _ in beats] == sorted(p for p, _, _ in beats)
