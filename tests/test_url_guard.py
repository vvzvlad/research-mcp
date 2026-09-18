"""SSRF guard: read_page must not reach the internal network.

The guard is exercised through the real ``Pipeline.read`` path (respx mocks all
HTTP), because that is what matters: a blocked url must cost zero requests, and
a redirect into loopback must be stopped mid-flight by the client event hook.
"""

import socket

import httpx
import pytest
import respx

from src.pipeline import Pipeline
from src.providers import _url_guard
from src.providers._url_guard import UrlNotAllowed
from src.settings import Settings
from tests.conftest import _clear_provider_env

ARTICLE_HTML = (
    "<html><head><title>Test Article</title></head><body>"
    "<article><h1>Main Heading</h1>"
    + "<p>This is a substantial paragraph of the main article body. </p>" * 12
    + "</article></body></html>"
)

PRIVATE_URL = "http://192.168.1.10/admin"


def _permissive_settings() -> Settings:
    """Test settings with the SSRF escape hatch turned on."""
    return Settings(
        _env_file=None,
        request_timeout=5.0,
        fallback_min_chars=400,
        read_pages_concurrency=5,
        retries=1,
        allow_private_network=True,
    )


@respx.mock
async def test_private_ip_url_is_blocked_without_any_request(monkeypatch, settings):
    # The entry check in read() fires before the probe and before any provider,
    # so a blocked url costs zero HTTP requests.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    route = respx.get(PRIVATE_URL).mock(return_value=httpx.Response(200, text=ARTICLE_HTML))

    pipe = Pipeline.build(settings)
    try:
        with pytest.raises(UrlNotAllowed):
            await pipe.read(PRIVATE_URL)
    finally:
        await pipe.aclose()

    assert route.call_count == 0
    assert len(respx.calls) == 0


@respx.mock
async def test_redirect_to_loopback_is_blocked(monkeypatch, settings):
    # The public url passes the entry check; the 302 hop into loopback is caught
    # by the request event hook on the guarded client.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    url = "https://public.test/go"
    respx.get(url).mock(return_value=httpx.Response(302, headers={"Location": "http://127.0.0.1/"}))
    loopback = respx.get("http://127.0.0.1/").mock(
        return_value=httpx.Response(200, text=ARTICLE_HTML)
    )

    pipe = Pipeline.build(settings)
    try:
        with pytest.raises(UrlNotAllowed):
            await pipe.read(url)
    finally:
        await pipe.aclose()

    assert loopback.call_count == 0  # the hop never left the process


@respx.mock
async def test_hostname_resolving_to_private_is_blocked(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")

    async def _fake_resolve(host: str) -> list[str]:
        return ["10.1.2.3"]

    monkeypatch.setattr(_url_guard, "_resolve_host", _fake_resolve)
    url = "https://internal.example.com/secret"
    route = respx.get(url).mock(return_value=httpx.Response(200, text=ARTICLE_HTML))

    pipe = Pipeline.build(settings)
    try:
        with pytest.raises(UrlNotAllowed):
            await pipe.read(url)
    finally:
        await pipe.aclose()

    assert route.call_count == 0


@respx.mock
async def test_allow_private_network_reads_the_private_url(monkeypatch):
    # The escape hatch: the very same url reads normally when it is on.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    route = respx.get(PRIVATE_URL).mock(return_value=httpx.Response(200, text=ARTICLE_HTML))

    pipe = Pipeline.build(_permissive_settings())
    try:
        out = await pipe.read(PRIVATE_URL)
    finally:
        await pipe.aclose()

    assert "main article body" in out
    assert route.call_count == 1


@respx.mock
async def test_file_scheme_blocked_even_when_private_is_allowed(monkeypatch):
    # The scheme check is unconditional: allow_private_network only turns off
    # the address check.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")

    pipe = Pipeline.build(_permissive_settings())
    try:
        with pytest.raises(UrlNotAllowed):
            await pipe.read("file:///etc/passwd")
    finally:
        await pipe.aclose()

    assert len(respx.calls) == 0


@respx.mock
async def test_malformed_hostname_does_not_raise_out_of_read(monkeypatch, settings):
    # The REAL resolver (not the patched one): getaddrinfo pushes a str host
    # through the `idna` codec, which raises UnicodeError — not gaierror — on an
    # empty label. That must be handled like any failed lookup, otherwise
    # read_page answers with a traceback instead of a string.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    url = "http://a..b.test/article"
    respx.get(url).mock(return_value=httpx.Response(200, text=ARTICLE_HTML))

    pipe = Pipeline.build(settings)
    try:
        out = await pipe.read(url)
    finally:
        await pipe.aclose()

    assert "main article body" in out


@respx.mock
async def test_unresolvable_host_is_allowed_through(monkeypatch, settings):
    # Deliberate decision (see ensure_url_allowed): a host that does not resolve
    # has no address to connect to, so it is let through — httpx fails on its
    # own, and hosts resolved on the proxy side keep working.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")

    async def _fake_resolve(host: str) -> list[str]:
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(_url_guard, "_resolve_host", _fake_resolve)
    url = "https://nxdomain.test/article"
    respx.get(url).mock(return_value=httpx.Response(200, text=ARTICLE_HTML))

    pipe = Pipeline.build(settings)
    try:
        out = await pipe.read(url)
    finally:
        await pipe.aclose()

    assert "main article body" in out
