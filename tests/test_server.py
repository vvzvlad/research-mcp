"""Server wiring: the 3 tools are registered, descriptions are the verbatim
Russian texts, and each tool delegates to the pipeline without ever raising.
"""

from typing import Any

import pytest

from src.providers.base import ProviderError, SearchResult
from src.server import build_server


class FakePipeline:
    """Stand-in for Pipeline so server tests need no network or ENV."""

    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True

    async def search(self, query, num_results, page, language):
        return [SearchResult(title="Hit", url="https://x.test", snippet="snip", source="searxng")]

    async def read(self, url):
        if "boom" in url:
            raise ProviderError("страница недоступна всеми способами (тест)")
        return f"# Markdown of {url}"


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
    assert "SearXNG-метапоиск + при наличии Serper/Exa" in ws
    rp = by_name["read_page"].description
    assert rp.startswith("Скачать ОДНУ веб-страницу или PDF по url")
    assert "OCR нет" in rp
    rps = by_name["read_pages"].description
    assert rps.startswith("Скачать НЕСКОЛЬКО страниц или PDF за один вызов (до 20)")
    assert "{url, ok, markdown|error}" in rps


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


async def test_web_search_formats_results(server):
    out = str(await _call(server, "web_search", {"query": "hello"}))
    assert "Hit" in out
    assert "https://x.test" in out


async def test_read_page_returns_markdown(server):
    out = str(await _call(server, "read_page", {"url": "https://a.test"}))
    assert "Markdown of https://a.test" in out


async def test_read_page_error_is_string(server):
    out = str(await _call(server, "read_page", {"url": "https://boom.test"}))
    assert "недоступна" in out


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


async def test_read_pages_respects_hard_limit(settings):
    # The cap is a hard constant (READ_PAGES_MAX=20), NOT a setting, so the
    # tool's "up to 20" promise stays true regardless of env overrides.
    from src.server import READ_PAGES_MAX

    assert READ_PAGES_MAX == 20
    srv = build_server(settings, pipeline=FakePipeline())
    urls = [f"https://a.test/{i}" for i in range(READ_PAGES_MAX + 2)]
    out = await _call(srv, "read_pages", {"urls": urls})
    items = _as_list(out)
    assert len(items) == READ_PAGES_MAX
    processed = {item["url"] for item in items}
    assert "https://a.test/0" in processed
    assert f"https://a.test/{READ_PAGES_MAX - 1}" in processed
    # The 21st and 22nd urls are dropped.
    assert f"https://a.test/{READ_PAGES_MAX}" not in processed
