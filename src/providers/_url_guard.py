"""SSRF guard: refuse to fetch urls that point into the internal network.

``read_page`` takes a url straight from the model and this process runs inside
the docker network next to searxng/crawl4ai, so an unchecked fetch reaches
private services, loopback and the cloud metadata endpoint (169.254.169.254).
Every url the MODEL gives us is checked here: once at the entry of
``Pipeline.read`` (a clean early error, zero HTTP requests) and once per outgoing
request — including every redirect hop — via the httpx event hook on the guarded
clients. Our own service endpoints (searxng, crawl4ai, provider APIs) are fetched
with the plain, unguarded clients: they are configured by us, not by the model.

``settings.allow_private_network`` is the deliberate escape hatch: it turns the
address check off (the scheme check stays on).
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

from loguru import logger

from src.providers.base import ProviderError

_IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class UrlNotAllowed(ProviderError):
    """The url points at a blocked address (or uses a blocked scheme).

    A ``ProviderError`` subclass on purpose: ``src/server.py`` already catches
    ``ProviderError`` and returns its message to the model unchanged.
    """


# Only these two schemes are ever fetched (no file://, gopher://, ftp://, ...).
_ALLOWED_SCHEMES = frozenset({"http", "https"})

# EXPLICIT blocklist. `ipaddress`'s own flags are not enough on their own: what
# counts as non-global differs between Python versions (100.64/10 CGNAT is the
# classic case), so the ranges that matter are named here and the flags below
# only act as a catch-all.
_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "0.0.0.0/8",  # "this network"
        "10.0.0.0/8",  # RFC1918
        "100.64.0.0/10",  # CGNAT
        "127.0.0.0/8",  # loopback
        "169.254.0.0/16",  # link-local, incl. the cloud metadata endpoint
        "172.16.0.0/12",  # RFC1918 (the docker default pool)
        "192.168.0.0/16",  # RFC1918
        "::1/128",  # IPv6 loopback
        "fc00::/7",  # IPv6 unique-local
        "fe80::/10",  # IPv6 link-local
    )
)


def _is_blocked(ip: _IpAddress) -> bool:
    """True if ``ip`` is not a public address we are allowed to fetch."""
    for network in _BLOCKED_NETWORKS:
        if ip.version == network.version and ip in network:
            return True
    # Catch-all for everything the table above does not name (reserved ranges,
    # broadcast, IPv4-mapped IPv6, ...).
    return not ip.is_global or ip.is_multicast


def _deny(url: str, reason: str) -> UrlNotAllowed:
    """Log the block and build the error. The single log site for a block."""
    logger.warning("url blocked url={} reason={}", url, reason)
    return UrlNotAllowed(f"URL заблокирован: {reason}.")


async def _resolve_host(host: str) -> list[str]:
    """Resolve ``host`` to its addresses (module level so tests can patch it)."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


async def ensure_url_allowed(url: str, *, allow_private: bool) -> None:
    """Raise ``UrlNotAllowed`` unless ``url`` is a public http(s) target.

    The scheme check applies ALWAYS, ``allow_private=True`` only skips the
    address check (and with it every DNS lookup). An IP literal is checked
    as-is; a hostname is resolved and EVERY returned address must be allowed.

    Deliberate decision: a host that fails to resolve is let THROUGH. There is
    no address to connect to, so there is nothing to protect — httpx will fail
    on its own — and letting it through keeps hosts that are resolved on the
    proxy side (``socks5://``) working. It is logged at debug only.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise _deny(url, f"схема '{parts.scheme}' не поддерживается, разрешены только http и https")
    # `hostname` lowercases and already strips the [] around an IPv6 literal.
    host = parts.hostname or ""
    if not host:
        raise _deny(url, "в URL не указан хост")

    if allow_private:
        return

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked(literal):
            raise _deny(url, f"адрес {literal} принадлежит внутренней сети")
        return

    try:
        addresses = await _resolve_host(host)
    except (OSError, UnicodeError) as exc:
        # UnicodeError (not an OSError): getaddrinfo runs a str host through the
        # `idna` codec, which rejects an empty or >63-character label — e.g.
        # "http://a..b.com/". Same outcome as a failed lookup: no address.
        logger.debug("url guard: host {} did not resolve ({}); allowing", host, exc)
        return
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if _is_blocked(ip):
            raise _deny(url, f"хост {host} разрешается в адрес внутренней сети {ip}")
