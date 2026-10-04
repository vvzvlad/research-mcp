"""Provider interfaces and shared types.

Two provider kinds:

- ``SearchProvider`` — turns a query into a list of ``SearchResult``.
- ``ReadProvider``  — turns a url into clean Markdown, or raises ``ProviderError``.

Concrete providers live one-per-module in this package and register themselves
via ``@register("type")`` (see ``registry.py``). They are constructed with a
``ProviderConfig`` carrying the already-resolved secrets/URLs and shared knobs —
provider code never touches ``os.environ`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx


class ProviderError(Exception):
    """A provider could not produce a usable result.

    Raised on hard failures (HTTP errors, empty/too-thin content, exhausted
    credits). The pipeline catches it and moves on to the next instance.
    ``status`` is set only by ``request_with_retry``'s plain-4xx branch (not
    402/429, not a credit-exhaustion 4xx, not 5xx) and is ``None`` otherwise —
    enough for a caller to tell "the site said 404" from other errors.
    ``reason`` is the failure category (a ``src.failure_reason`` constant), set
    by the raise site, which is the only place that knows it. ``None`` means
    "let ``classify()`` look at the exception chain, else ``other``".
    """

    def __init__(
        self, message: str, status: int | None = None, reason: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason


@dataclass(slots=True)
class SearchResult:
    """One search hit. ``source`` names the instance that produced it."""

    title: str
    url: str
    snippet: str
    source: str


@dataclass(slots=True)
class ProviderConfig:
    """Resolved configuration handed to a provider instance at build time.

    ``url`` / ``api_key`` / ``token`` / ``proxy`` are the *values* already read
    from the environment by the instance loader — never env var names. ``proxy``
    is a SOCKS5/HTTP proxy URL this instance must route through (None = direct);
    the pipeline picks the httpx client bound to it. ``options`` holds extra
    per-instance settings if ever needed.
    """

    name: str
    request_timeout: float = 25.0
    fallback_min_chars: int = 400
    retries: int = 1
    url: str | None = None
    api_key: str | None = None
    token: str | None = None
    proxy: str | None = None
    options: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class SearchProvider(Protocol):
    """A configured search instance."""

    name: str
    # Proxy URL this instance routes through (None = direct egress). The pipeline
    # reads it to pick the httpx client bound to this instance's proxy.
    proxy: str | None

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int,
        language: str | None,
    ) -> list[SearchResult]:
        """Return search results, or raise ``ProviderError`` on failure."""
        ...


@runtime_checkable
class ReadProvider(Protocol):
    """A configured read/extract instance."""

    name: str
    # Proxy URL this instance routes through (None = direct egress).
    proxy: str | None

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        """Return page content as Markdown, or raise ``ProviderError``.

        Returning content shorter than ``fallback_min_chars`` is treated by the
        pipeline as "too thin" and triggers the next provider, so providers may
        either return what they got or raise ``ProviderError`` for empties.
        """
        ...


@runtime_checkable
class UrlSpecificReader(ReadProvider, Protocol):
    """A reader that serves only some urls (a YouTube video, an Instagram post).

    The pipeline offers it only the urls it ``accepts``, BEFORE the probe, and
    its answer is final whatever its length.
    """

    def accepts(self, url: str) -> bool:
        """True if this reader serves ``url``."""
        ...


# Browser-like User-Agent so plain sites do not block the direct-HTTP read path.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
