"""ClientManager: one cached httpx client per proxy URL (None = direct)."""

import httpx
import pytest

from src.pipeline import ClientManager


@pytest.fixture
async def manager():
    cm = ClientManager(request_timeout=5.0)
    try:
        yield cm
    finally:
        await cm.aclose()


async def test_same_proxy_returns_same_client(manager):
    a = manager.client_for("socks5://host:1080")
    b = manager.client_for("socks5://host:1080")
    assert a is b  # cached by proxy URL


async def test_different_proxies_return_different_clients(manager):
    a = manager.client_for("socks5://host-a:1080")
    b = manager.client_for("socks5://host-b:1080")
    assert a is not b


async def test_direct_client_is_separate(manager):
    direct = manager.client_for(None)
    proxied = manager.client_for("socks5://host:1080")
    assert direct is not proxied
    # The direct client is reused on repeat calls.
    assert manager.client_for(None) is direct


async def test_injected_direct_client_is_reused():
    injected = httpx.AsyncClient(timeout=5.0)
    cm = ClientManager(request_timeout=5.0, direct_client=injected)
    try:
        assert cm.client_for(None) is injected  # respx can intercept this one
    finally:
        await cm.aclose()


async def test_recreates_closed_client(manager):
    first = manager.client_for("socks5://host:1080")
    await first.aclose()
    second = manager.client_for("socks5://host:1080")
    assert second is not first  # self-healing: a closed client is replaced
    assert not second.is_closed


async def test_aclose_closes_all():
    cm = ClientManager(request_timeout=5.0)
    direct = cm.client_for(None)
    proxied = cm.client_for("socks5://host:1080")
    await cm.aclose()
    assert direct.is_closed
    assert proxied.is_closed


async def test_accepts_socks_and_http_schemes(manager):
    # All three supported schemes construct a usable client.
    for proxy in ("socks5://h:1080", "socks5h://h:1080", "http://h:8080"):
        client = manager.client_for(proxy)
        assert isinstance(client, httpx.AsyncClient)
        assert not client.is_closed
