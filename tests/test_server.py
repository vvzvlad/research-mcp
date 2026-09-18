"""Server wiring: the 3 tools are registered, descriptions are the verbatim
Russian texts, and each tool delegates to the pipeline without ever raising.
"""

from typing import Any

import pytest

from src.pipeline import ReadFailed, ReadOutcome, SearchOutcome
from src.providers.base import SearchResult
from src.server import build_server


class FakePipeline:
    """Stand-in for Pipeline so server tests need no network or ENV."""

    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True

    async def search(self, query, num_results, page, language):
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
            markdown=f"# Markdown of {url}",
            provider="jina",
            tried=["trafilatura", "jina"],
            failures=[("trafilatura", "empty")],
            thin=False,
            elapsed_ms=1500,
        )


@pytest.fixture
def server(settings):
    return build_server(settings, pipeline=FakePipeline())


async def test_three_tools_registered(server):
    tools = {t.name for t in await server.list_tools()}
    assert tools == {"web_search", "read_page", "read_pages"}


async def test_descriptions_are_verbatim_russian(server):
    by_name = {t.name: t for t in await server.list_tools()}
    ws = by_name["web_search"].description
    assert ws.startswith("Поиск в вебе. Агрегирует несколько источников")
    assert "SearXNG-метапоиск + при наличии Brave/Serper/Exa" in ws
    rp = by_name["read_page"].description
    assert rp.startswith("Скачать ОДНУ веб-страницу или PDF по url")
    assert "OCR нет" in rp
    rps = by_name["read_pages"].description
    assert rps.startswith("Скачать НЕСКОЛЬКО страниц или PDF за один вызов (до 20)")
    # Only the sentence describing the return shape changed with the summary.
    assert "{url, ok, markdown|error, reason}" in rps
    assert "summary — строка состояния по батчу" in rps


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
