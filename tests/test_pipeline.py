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
from src.providers.trafilatura import extract_markdown

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
        "SERPER_PROXY",
        "EXA_PROXY",
        "JINA_PROXY",
        "TAVILY_1_PROXY",
        "TAVILY_2_PROXY",
        "FIRECRAWL_PROXY",
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


# -- lazy self-healing client (regression: "client has been closed") -------


@respx.mock
async def test_read_recreates_client_after_aclose(monkeypatch, settings):
    # Regression for the prod bug: a premature aclose() (e.g. a streamable-http
    # lifespan shutdown) must NOT poison later requests with
    # "Cannot send a request, as the client has been closed.". We do NOT inject a
    # client here so the pipeline owns it and we exercise lazy recreation. respx
    # patches the httpx transport globally, so the recreated real client is also
    # intercepted.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    url = "https://good.test/article"
    respx.get(url).mock(return_value=httpx.Response(200, text=ARTICLE_HTML))

    pipe = Pipeline.build(settings)  # no injected client → pipeline creates it

    first = await pipe.read(url)
    assert "main article body" in first

    # Simulate the premature lifespan shutdown that closed the shared client.
    await pipe.aclose()

    # Same call again must succeed via a recreated client, not raise the
    # "client has been closed" error.
    second = await pipe.read(url)
    assert "main article body" in second

    await pipe.aclose()


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


# -- probe defers to provider chain (fix A) --------------------------------


@respx.mock
async def test_read_pdf_probe_403_falls_through_to_provider(monkeypatch, settings):
    # A .pdf url whose DIRECT probe GET 403s must NOT hard-fail: the probe defers
    # to the read chain, and jina (server-side, keyless) can still retrieve it.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")  # enable a search provider

    url = "https://files.test/doc.pdf"
    respx.get(url).mock(return_value=httpx.Response(403))
    jina_md = "# PDF via jina\n\n" + ("Server-side fetched content. " * 50)
    respx.get(f"https://r.jina.ai/{url}").mock(
        return_value=httpx.Response(200, text=jina_md)
    )

    pipe = Pipeline.build(settings)
    try:
        out = await pipe.read(url)  # must NOT raise ProviderError
    finally:
        await pipe.aclose()
    assert "PDF via jina" in out


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


# -- TLS verification retry (fix B) ----------------------------------------


@respx.mock
async def test_read_tls_verify_error_retries_insecure(monkeypatch, settings, capture_logs):
    # A TLS certificate-verification failure on the probe GET triggers ONE retry
    # with verification disabled. respx patches the transport globally, so the
    # throwaway insecure client is intercepted too: the second call returns the
    # PDF, which is extracted as the result.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")

    url = "https://broken-cert.test/doc.pdf"
    route = respx.get(url)
    route.side_effect = [
        httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"),
        httpx.Response(
            200, content=SAMPLE_PDF, headers={"Content-Type": "application/pdf"}
        ),
    ]

    pipe = Pipeline.build(settings)
    try:
        out = await pipe.read(url)
    finally:
        await pipe.aclose()
    assert "Hello PDF research-mcp" in out
    assert route.call_count == 2  # original + the insecure retry
    assert any("TLS verification failed" in m for m in capture_logs)


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


# -- per-request logging ---------------------------------------------------


@respx.mock
async def test_search_emits_per_request_log(monkeypatch, settings, capture_logs):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    respx.get("http://searxng.test/search").mock(
        return_value=httpx.Response(
            200, json={"results": [{"url": "https://ok.test", "title": "OK", "content": "s"}]}
        )
    )
    pipe = Pipeline.build(settings)
    try:
        await pipe.search("hello world", num_results=5, page=1, language=None)
    finally:
        await pipe.aclose()
    line = next((m for m in capture_logs if m.startswith("search query=")), None)
    assert line is not None
    assert "'searxng'" in line  # the provider that really ran
    assert "results=1" in line
    assert "elapsed_ms=" in line


@respx.mock
async def test_read_emits_per_request_log(monkeypatch, settings, capture_logs):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    url = "https://good.test/article"
    respx.get(url).mock(return_value=httpx.Response(200, text=ARTICLE_HTML))
    pipe = Pipeline.build(settings)
    try:
        await pipe.read(url)
    finally:
        await pipe.aclose()
    line = next((m for m in capture_logs if m.startswith("read url=")), None)
    assert line is not None
    assert "provider=trafilatura" in line
    assert "ok=true" in line
    assert "elapsed_ms=" in line


@respx.mock
async def test_search_log_counts_paid_calls(monkeypatch, settings, capture_logs):
    # searxng (free) + serper (paid) both return → 1 paid of 2 billed = 50.0%.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    monkeypatch.setenv("SERPER_API_KEY", "k")
    respx.get("http://searxng.test/search").mock(
        return_value=httpx.Response(
            200, json={"results": [{"url": "https://sx.test", "title": "Sx", "content": "a"}]}
        )
    )
    respx.post("https://google.serper.dev/search").mock(
        return_value=httpx.Response(
            200, json={"organic": [{"link": "https://sp.test", "title": "Sp", "snippet": "b"}]}
        )
    )
    pipe = Pipeline.build(settings)
    try:
        await pipe.search("q", num_results=10, page=1, language=None)
    finally:
        await pipe.aclose()
    line = next((m for m in capture_logs if m.startswith("search query=")), None)
    assert line is not None
    assert "paid_calls=1" in line  # serper is the one paid provider
    assert "paid_pct=50.0%" in line  # one paid of two billed calls


@respx.mock
async def test_read_log_counts_paid_calls(monkeypatch, settings, capture_logs):
    # trafilatura thin + jina 500 + tavily-1 ok → winning paid provider logged.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    monkeypatch.setenv("TAVILY_1_API_KEY", "t1")

    url = "https://doc.test/page"
    respx.get(url).mock(return_value=httpx.Response(200, text=THIN_HTML))
    respx.get(f"https://r.jina.ai/{url}").mock(return_value=httpx.Response(500))
    good = "Tavily-1 extracted content. " * 40
    respx.post("https://api.tavily.com/extract").mock(
        return_value=httpx.Response(
            200, json={"results": [{"url": url, "raw_content": good}], "failed_results": []}
        )
    )
    pipe = Pipeline.build(settings)
    try:
        await pipe.read(url)
    finally:
        await pipe.aclose()
    line = next((m for m in capture_logs if m.startswith("read url=")), None)
    assert line is not None
    assert "ok=true" in line
    assert "provider=tavily-1" in line
    assert "tried=" in line  # the chain that was walked
    assert "paid_calls=" in line
    assert "paid_calls=0" not in line  # tavily-1 is paid → at least one paid call


@respx.mock
async def test_read_all_fail_logs_failed_line(monkeypatch, settings, capture_logs):
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
    line = next((m for m in capture_logs if "FAILED" in m), None)
    assert line is not None
    assert "ok=false" in line
    assert url in line
    assert "tried=" in line


@respx.mock
async def test_read_pdf_logs_provider_pdf(monkeypatch, settings, capture_logs):
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
        await pipe.read(url)
    finally:
        await pipe.aclose()
    line = next((m for m in capture_logs if m.startswith("read url=")), None)
    assert line is not None
    assert "provider=pdf" in line
    assert "ok=true" in line


# -- per-instance proxy ----------------------------------------------------


@respx.mock
async def test_proxied_provider_still_serves(monkeypatch, settings):
    # A provider configured with a proxy uses a (separate) proxied client; respx
    # intercepts above the SOCKS transport, so the request still succeeds.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EXA_API_KEY", "k")
    monkeypatch.setenv("EXA_PROXY", "socks5://internal.lc:1080")
    respx.post("https://api.exa.ai/search").mock(
        return_value=httpx.Response(
            200, json={"results": [{"url": "https://e.test", "title": "Exa via proxy"}]}
        )
    )
    pipe = Pipeline.build(settings)
    try:
        results = await pipe.search("q", num_results=5, page=1, language=None)
        assert any(r.url == "https://e.test" for r in results)
        # The proxied exa client is distinct from the direct client.
        proxied = pipe._clients.client_for("socks5://internal.lc:1080")
        direct = pipe._clients.client_for(None)
        assert proxied is not direct
    finally:
        await pipe.aclose()


# -- trafilatura extraction contract (fix C) ------------------------------


def test_extract_markdown_uses_precision(monkeypatch):
    # Deterministic regression guard for fix C: assert the exact kwargs handed to
    # trafilatura.extract. A content-based assertion would be tautological (it
    # passes under both favor_recall and favor_precision); spying on the call
    # locks the contract so a revert to favor_recall fails the test.
    captured_kwargs: dict[str, object] = {}

    def spy(html, **kwargs):
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        return "# Fixed markdown\n\nNon-empty content."

    monkeypatch.setattr(
        "src.providers.trafilatura._trafilatura.extract", spy
    )

    out = extract_markdown("<html><body><article>hello</article></body></html>")

    assert out == "# Fixed markdown\n\nNon-empty content."  # spy result, non-empty
    assert captured_kwargs.get("favor_precision") is True
    assert captured_kwargs.get("favor_recall") is not True
    assert captured_kwargs.get("output_format") == "markdown"
    assert captured_kwargs.get("include_comments") is False
