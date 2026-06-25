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
    """


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

    ``url`` / ``api_key`` / ``token`` are the *values* already read from the
    environment by the instance loader — never env var names. ``options`` holds
    extra per-instance settings if ever needed.
    """

    name: str
    request_timeout: float = 25.0
    fallback_min_chars: int = 400
    retries: int = 1
    url: str | None = None
    api_key: str | None = None
    token: str | None = None
    options: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class SearchProvider(Protocol):
    """A configured search instance."""

    name: str

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int,
        page: int,
        language: str | None,
    ) -> list[SearchResult]:
        """Return search results, or raise ``ProviderError`` on failure."""
        ...


@runtime_checkable
class ReadProvider(Protocol):
    """A configured read/extract instance."""

    name: str

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        """Return page content as Markdown, or raise ``ProviderError``.

        Returning content shorter than ``fallback_min_chars`` is treated by the
        pipeline as "too thin" and triggers the next provider, so providers may
        either return what they got or raise ``ProviderError`` for empties.
        """
        ...


# Browser-like User-Agent so plain sites do not block the direct-HTTP read path.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
