"""Pipeline behaviour: search merge/dedup, read fallback, failover, PDF, retry.

All network I/O is mocked with respx. Pipelines are built via ``Pipeline.build``
with monkeypatched provider ENV so we control exactly which instances are on.
"""

from pathlib import Path

import httpx
import pytest
import respx

from src.pipeline import Pipeline
from src.providers.base import ProviderError

SAMPLE_PDF = (Path(__file__).parent / "fixtures" / "sample.pdf").read_bytes()

ARTICLE_HTML = (
    "<html><head><title>Test Article</title></head><body>"
    "<nav>menu menu menu</nav>"
    "<article><h1>Main Heading</h1>"
    + "<p>This is a substantial paragraph of the main article body. </p>" * 12
    + "</article>"
    "<footer>footer junk</footer>"
    "</body></html>"
)

THIN_HTML = "<html><body><div id='app'></div></body></html>"


def _clear_provider_env(monkeypatch):
    for var in (
        "SEARXNG_URL",
        "SERPER_API_KEY",
        "EXA_API_KEY",
        "JINA_API_KEY",
        "CRAWL4AI_URL",
        "CRAWL4AI_TOKEN",
        "TAVILY_1_API_KEY",
        "TAVILY_2_API_KEY",
        "FIRECRAWL_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# -- search merge + dedup --------------------------------------------------


@respx.mock
async def test_search_merges_and_dedups(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    monkeypatch.setenv("SERPER_API_KEY", "k")

    respx.get("http://searxng.test/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://dup.test/", "title": "Sx Dup", "content": "a"},
                    {"url": "https://only-sx.test", "title": "Sx Only", "content": "b"},
                ]
            },
        )
    )
    respx.post("https://google.serper.dev/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "organic": [
                    # Same url as searxng (trailing slash diff) → deduped out.
                    {"link": "https://dup.test", "title": "Serper Dup", "snippet": "c"},
                    {"link": "https://only-serper.test", "title": "Serper Only", "snippet": "d"},
                ]
            },
        )
    )

    pipe = Pipeline.build(settings)
    try:
        results = await pipe.search("q", num_results=10, page=1, language=None)
    finally:
        await pipe.aclose()

    urls = [r.url for r in results]
    assert "https://only-sx.test" in urls
    assert "https://only-serper.test" in urls
    # The duplicate appears once, and searxng (earlier in pipeline) wins.
    dup = [r for r in results if "dup.test" in r.url]
    assert len(dup) == 1
    assert dup[0].source == "searxng"


@respx.mock
async def test_search_trims_to_num_results(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    results_payload = [
        {"url": f"https://x.test/{i}", "title": f"t{i}", "content": "s"} for i in range(10)
    ]
    respx.get("http://searxng.test/search").mock(
        return_value=httpx.Response(200, json={"results": results_payload})
    )
    pipe = Pipeline.build(settings)
    try:
        results = await pipe.search("q", num_results=3, page=1, language=None)
    finally:
        await pipe.aclose()
    assert len(results) == 3


@respx.mock
async def test_search_clamps_non_positive_num_results(monkeypatch, settings):
    # The public pipeline method must not silently return [] for num<=0.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    respx.get("http://searxng.test/search").mock(
        return_value=httpx.Response(
            200, json={"results": [{"url": "https://x.test", "title": "t", "content": "s"}]}
        )
    )
    pipe = Pipeline.build(settings)
    try:
        results = await pipe.search("q", num_results=0, page=1, language=None)
    finally:
        await pipe.aclose()
    assert len(results) == 1  # clamped up to 1, not empty


@respx.mock
async def test_exa_clamps_num_results(monkeypatch, settings):
    # Exa must not receive an oversized numResults (would risk a 4xx).
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EXA_API_KEY", "k")
    route = respx.post("https://api.exa.ai/search").mock(
        return_value=httpx.Response(200, json={"results": [{"url": "https://e.test", "title": "E"}]})
    )
    pipe = Pipeline.build(settings)
    try:
        await pipe.search("q", num_results=999, page=1, language=None)
    finally:
        await pipe.aclose()
    sent_body = route.calls.last.request.content
    import json as _json

    assert _json.loads(sent_body)["numResults"] == 50  # clamped to EXA_NUM_RESULTS_MAX


# -- read happy path + single-GET reuse ------------------------------------


@respx.mock
async def test_read_html_uses_single_get(monkeypatch, settings):
    # The probe GET downloads the page; trafilatura reuses that body instead of
    # GETting the same url again (read_page is a hot path).
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    url = "https://good.test/article"
    route = respx.get(url).mock(return_value=httpx.Response(200, text=ARTICLE_HTML))
    pipe = Pipeline.build(settings)
    try:
        out = await pipe.read(url)
    finally:
        await pipe.aclose()
    assert "main article body" in out
    assert "footer junk" not in out  # trafilatura stripped the chrome
    assert route.call_count == 1  # NOT fetched twice


# -- read fallback chain ---------------------------------------------------


@respx.mock
async def test_read_fallback_trafilatura_thin_jina_error_crawl4ai_ok(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")  # enable a search provider
    monkeypatch.setenv("CRAWL4AI_URL", "http://crawl4ai.test")
    monkeypatch.setenv("CRAWL4AI_TOKEN", "tok")
    # jina is always enabled (keyless); make it error.

    url = "https://spa.test/page"
    # One probe GET; trafilatura reuses that body → thin HTML → too thin.
    respx.get(url).mock(return_value=httpx.Response(200, text=THIN_HTML))
    respx.get(f"https://r.jina.ai/{url}").mock(return_value=httpx.Response(500))
    crawl_md = "# Crawl4AI result\n\n" + ("Real content. " * 50)
    respx.post("http://crawl4ai.test/md").mock(
        return_value=httpx.Response(200, json={"markdown": crawl_md, "success": True})
    )

    pipe = Pipeline.build(settings)
    try:
        out = await pipe.read(url)
    finally:
        await pipe.aclose()
    assert "Crawl4AI result" in out


@respx.mock
async def test_read_tavily_1_429_fails_over_to_tavily_2(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    monkeypatch.setenv("TAVILY_1_API_KEY", "t1")
    monkeypatch.setenv("TAVILY_2_API_KEY", "t2")

    url = "https://doc.test/page"
    # trafilatura returns thin, jina errors → reach tavily-1 then tavily-2.
    respx.get(url).mock(return_value=httpx.Response(200, text=THIN_HTML))
    respx.get(f"https://r.jina.ai/{url}").mock(return_value=httpx.Response(500))

    good = "Tavily-2 extracted content. " * 40
    route = respx.post("https://api.tavily.com/extract")
    route.side_effect = [
        httpx.Response(429),  # tavily-1: rate limited → ProviderError, fail over
        httpx.Response(
            200, json={"results": [{"url": url, "raw_content": good}], "failed_results": []}
        ),  # tavily-2: ok
    ]

    pipe = Pipeline.build(settings)
    try:
        out = await pipe.read(url)
    finally:
        await pipe.aclose()
    assert "Tavily-2 extracted content" in out
    assert route.call_count == 2


@respx.mock
async def test_read_all_fail_raises(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")

    url = "https://dead.test/x"
    respx.get(url).mock(side_effect=httpx.ConnectError("nope"))
    respx.get(f"https://r.jina.ai/{url}").mock(side_effect=httpx.ConnectError("nope"))

    pipe = Pipeline.build(settings)
    try:
        with pytest.raises(ProviderError):
            await pipe.read(url)
    finally:
        await pipe.aclose()


# -- PDF detection ---------------------------------------------------------


@respx.mock
async def test_read_pdf_by_suffix(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    url = "https://files.test/doc.pdf"
    respx.get(url).mock(
        return_value=httpx.Response(
            200, content=SAMPLE_PDF, headers={"Content-Type": "application/pdf"}
        )
    )
    pipe = Pipeline.build(settings)
    try:
        out = await pipe.read(url)
    finally:
        await pipe.aclose()
    assert "Hello PDF research-mcp" in out


@respx.mock
async def test_read_pdf_by_magic_bytes(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    # url has no .pdf suffix and a generic content-type → detected by %PDF magic.
    url = "https://files.test/download"
    respx.get(url).mock(
        return_value=httpx.Response(
            200, content=SAMPLE_PDF, headers={"Content-Type": "application/octet-stream"}
        )
    )
    pipe = Pipeline.build(settings)
    try:
        out = await pipe.read(url)
    finally:
        await pipe.aclose()
    assert "Hello PDF research-mcp" in out


# -- transient retry -------------------------------------------------------


@respx.mock
async def test_transient_retry_then_success(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    route = respx.get("http://searxng.test/search")
    route.side_effect = [
        httpx.ConnectError("blip"),  # transient → retried
        httpx.Response(
            200, json={"results": [{"url": "https://ok.test", "title": "OK", "content": "s"}]}
        ),
    ]
    pipe = Pipeline.build(settings)
    try:
        results = await pipe.search("q", num_results=5, page=1, language=None)
    finally:
        await pipe.aclose()
    assert any(r.url == "https://ok.test" for r in results)
    assert route.call_count == 2
