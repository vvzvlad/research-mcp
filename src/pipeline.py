"""The provider pipeline: load enabled instances, run search and read.

At startup ``Pipeline.build`` resolves the ENV-variable NAMES from
``pipeline_config.INSTANCES`` into values, constructs the enabled instances
(skipping any whose required variables are unset, with a log line), and asserts
that at least one search and one read instance are enabled.

Search runs all enabled ``SEARCH_PIPELINE`` instances concurrently, merges and
deduplicates by normalized url (pipeline order wins), and trims to ``num_results``.

Read first detects PDFs (Content-Type / ``.pdf`` suffix / ``%PDF`` magic) and
extracts them with pypdf. Otherwise it tries the enabled ``READ_PIPELINE``
instances in order; the first to return content ``>= fallback_min_chars`` wins;
thin/empty/error → next instance. If all fail, it raises ``ProviderError`` with
an aggregated message.
"""

from __future__ import annotations

import asyncio
import os
import time
from urllib.parse import urlsplit, urlunsplit

import httpx
from loguru import logger

from src.config_errors import ConfigError
from src.pipeline_config import (
    INSTANCES,
    READ_PIPELINE,
    SEARCH_PIPELINE,
    Instance,
)
import src.providers  # noqa: F401  (import the package for its @register side effects)
from src.providers.base import (
    BROWSER_USER_AGENT,
    ProviderConfig,
    ProviderError,
    ReadProvider,
    SearchProvider,
    SearchResult,
)
from src.providers.pdf import extract_pdf_text, looks_like_pdf
from src.providers.registry import REGISTRY
from src.providers.trafilatura import TrafilaturaRead, extract_markdown
from src.settings import Settings


def _normalize_url(url: str) -> str:
    """Normalize a url for dedup: lowercase host, strip fragment & trailing /."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    scheme = parts.scheme.lower() or "http"
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _resolve_instance(inst: Instance) -> ProviderConfig | None:
    """Read an instance's ENV-named secrets; return a config or None if disabled.

    An instance is disabled (returns None) when a required variable is unset. A
    variable is required unless it is the api key of an ``optional_api_key``
    instance (e.g. jina).
    """
    url = os.getenv(inst.url_env) if inst.url_env else None
    token = os.getenv(inst.token_env) if inst.token_env else None
    api_key = os.getenv(inst.api_key_env) if inst.api_key_env else None

    if inst.url_env and not url:
        return None
    if inst.token_env and not token:
        return None
    if inst.api_key_env and not api_key and not inst.optional_api_key:
        return None

    return ProviderConfig(name=inst.name, url=url, token=token, api_key=api_key)


class Pipeline:
    """Holds the enabled provider instances and runs the search/read logic."""

    def __init__(
        self,
        settings: Settings,
        search_instances: list[SearchProvider],
        read_instances: list[ReadProvider],
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._search = search_instances
        self._read = read_instances
        # Do NOT create the client eagerly: build() runs before the event loop
        # starts, and an httpx.AsyncClient created here would be bound to (and
        # later closed by) the wrong loop/lifespan. It is created lazily inside
        # the running loop by _ensure_client (an injected client is kept as-is).
        self._client = client

    def _ensure_client(self) -> httpx.AsyncClient:
        # Create the client lazily inside the running event loop and recreate it
        # if a previous one was closed (e.g. a premature lifespan shutdown), so the
        # facade never serves "client has been closed" across sessions/requests.
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._settings.request_timeout,
                follow_redirects=True,
            )
        return self._client

    # -- construction -------------------------------------------------------

    @classmethod
    def build(cls, settings: Settings, client: httpx.AsyncClient | None = None) -> "Pipeline":
        """Resolve ENV, construct enabled instances, validate, and return.

        Raises ``ConfigError`` if no search or no read instance is enabled.
        """
        built: dict[str, object] = {}

        for inst in INSTANCES:
            config = _resolve_instance(inst)
            if config is None:
                logger.info("Provider instance '{}' disabled (missing ENV)", inst.name)
                continue
            cls_impl = REGISTRY.get(inst.type)
            if cls_impl is None:
                logger.warning("Unknown provider type '{}' for instance '{}'", inst.type, inst.name)
                continue
            # Carry shared knobs into the provider config.
            config = ProviderConfig(
                name=config.name,
                request_timeout=settings.request_timeout,
                fallback_min_chars=settings.fallback_min_chars,
                retries=settings.retries,
                url=config.url,
                api_key=config.api_key,
                token=config.token,
            )
            try:
                built[inst.name] = cls_impl(config)
            except Exception as exc:  # noqa: BLE001 — provider __init__ guard
                logger.warning("Provider instance '{}' failed to build: {}", inst.name, exc)
                continue
            logger.info("Provider instance '{}' ({}) enabled", inst.name, inst.type)

        # Every key in `built` came from INSTANCES, so membership in `built` is
        # the only condition needed to pick the enabled instances in order.
        search_instances = [built[name] for name in SEARCH_PIPELINE if name in built]
        read_instances = [built[name] for name in READ_PIPELINE if name in built]

        if not search_instances:
            raise ConfigError(
                "No search provider enabled. Set at least one of "
                "SEARXNG_URL / SERPER_API_KEY / EXA_API_KEY."
            )
        if not read_instances:
            raise ConfigError(
                "No read provider enabled. trafilatura needs no config, so this "
                "should not happen — check src/pipeline_config.py."
            )

        return cls(settings, search_instances, read_instances, client=client)  # type: ignore[arg-type]

    async def aclose(self) -> None:
        """Close the shared httpx client if one is open."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    @property
    def search_names(self) -> list[str]:
        return [p.name for p in self._search]

    @property
    def read_names(self) -> list[str]:
        return [p.name for p in self._read]

    # -- search -------------------------------------------------------------

    async def search(
        self,
        query: str,
        num_results: int,
        page: int,
        language: str | None,
    ) -> list[SearchResult]:
        """Run all search instances concurrently, merge + dedup, trim."""
        # Defend the public method too: a non-positive count would otherwise
        # silently return nothing. (The server already does max(1, ...).)
        num_results = max(1, num_results)
        started = time.monotonic()
        client = self._ensure_client()

        async def _one(provider: SearchProvider) -> tuple[str, list[SearchResult] | None]:
            # Returns (name, results) where results is None if the provider
            # failed/crashed (so it is NOT counted as "really worked").
            try:
                hits = await provider.search(client, query, num_results, page, language)
                return provider.name, hits
            except ProviderError as exc:
                logger.info("search '{}' failed: {}", provider.name, exc)
                return provider.name, None
            except Exception as exc:  # noqa: BLE001 — never break the merge
                logger.warning("search '{}' crashed: {}", provider.name, exc)
                return provider.name, None

        # Gather in pipeline order; results keep that order so dedup prefers the
        # earlier (higher-priority) provider.
        batches = await asyncio.gather(*(_one(p) for p in self._search))

        used: list[str] = []
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for name, hits in batches:
            if hits is None:
                continue
            used.append(name)
            for result in hits:
                key = _normalize_url(result.url)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(result)
        merged = merged[:num_results]

        elapsed_ms = int((time.monotonic() - started) * 1000)
        # One per-request line for the persistent log (no bodies/secrets).
        logger.info(
            "search query={!r} providers={} results={} elapsed_ms={}",
            query,
            used,
            len(merged),
            elapsed_ms,
        )
        return merged

    # -- read ---------------------------------------------------------------

    async def read(self, url: str) -> str:
        """Return clean Markdown for ``url`` (PDF-aware, with provider fallback).

        Raises ``ProviderError`` if every method fails.
        """
        started = time.monotonic()
        client = self._ensure_client()

        def _ms() -> int:
            return int((time.monotonic() - started) * 1000)

        # 1) One probe GET decides the path. If it is a PDF, we are done; if it
        #    is HTML, reuse that body for the trafilatura step (no second GET).
        pdf_text, probe_html = await self._probe(client, url)
        if pdf_text is not None:
            logger.info("read url={} -> provider=pdf ok=true elapsed_ms={}", url, _ms())
            return pdf_text

        # 2) HTML path: walk the read pipeline until one yields enough content.
        errors: list[str] = []
        best_thin: str | None = None
        best_thin_name: str | None = None
        for provider in self._read:
            try:
                content = await self._read_one(client, provider, url, probe_html)
            except ProviderError as exc:
                errors.append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 — treat as provider failure
                errors.append(f"{provider.name}: {exc}")
                continue
            if len(content) >= self._settings.fallback_min_chars:
                logger.info(
                    "read url={} -> provider={} ok=true elapsed_ms={}",
                    url,
                    provider.name,
                    _ms(),
                )
                return content
            # Too thin — remember the longest thin result as a last resort.
            if best_thin is None or len(content) > len(best_thin):
                best_thin = content
                best_thin_name = provider.name
            errors.append(f"{provider.name}: content too thin ({len(content)} chars)")

        if best_thin:
            logger.info(
                "read url={} -> provider={} ok=true elapsed_ms={} (thin fallback)",
                url,
                best_thin_name,
                _ms(),
            )
            return best_thin
        logger.warning("read url={} -> all providers failed elapsed_ms={}", url, _ms())
        raise ProviderError("Не удалось прочитать страницу. " + "; ".join(errors))

    async def _read_one(
        self,
        client: httpx.AsyncClient,
        provider: ReadProvider,
        url: str,
        probe_html: str | None,
    ) -> str:
        """Run one read provider, reusing the probe body for trafilatura.

        trafilatura is a pure HTML→Markdown extractor, so when the probe already
        downloaded the page we extract from that body instead of GETting it
        again (read_page is a hot path). All other providers fetch as usual.
        """
        if probe_html is not None and isinstance(provider, TrafilaturaRead):
            content = extract_markdown(probe_html)
            if not content:
                raise ProviderError(f"{provider.name}: no main content extracted")
            return content
        return await provider.read(client, url)

    async def _probe(self, client: httpx.AsyncClient, url: str) -> tuple[str | None, str | None]:
        """Fetch ``url`` once and classify it.

        Returns ``(pdf_text, html)``:
        - PDF detected → ``(extracted_text, None)``.
        - HTML fetched → ``(None, body_text)`` so the caller can reuse the body.
        - probe failed on a non-PDF url → ``(None, None)``; HTML providers retry.

        Raises ``ProviderError`` only when a clearly-PDF url cannot be downloaded.
        """
        suffix_pdf = url.split("?", 1)[0].rstrip().lower().endswith(".pdf")
        try:
            response = await client.get(
                url, headers={"User-Agent": BROWSER_USER_AGENT}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            if suffix_pdf:
                raise ProviderError(f"Не удалось загрузить PDF {url}: {exc}") from exc
            # Not obviously a PDF and the probe failed — let HTML providers try.
            return None, None

        content_type = response.headers.get("Content-Type")
        if looks_like_pdf(url, content_type, response.content[:8]):
            return extract_pdf_text(response.content), None
        return None, response.text
