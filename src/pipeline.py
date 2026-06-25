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
    PAID_TYPES,
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
    # Proxy is always optional: unset → direct egress (no instance is disabled
    # for a missing proxy var).
    proxy = os.getenv(inst.proxy_env) if inst.proxy_env else None

    if inst.url_env and not url:
        return None
    if inst.token_env and not token:
        return None
    if inst.api_key_env and not api_key and not inst.optional_api_key:
        return None

    return ProviderConfig(name=inst.name, url=url, token=token, api_key=api_key, proxy=proxy)


class ClientManager:
    """Lazily creates and caches one ``httpx.AsyncClient`` per proxy URL.

    Key = proxy URL string, with ``None`` for direct (no-proxy) egress. Clients
    are created lazily inside the running event loop and recreated if a previous
    one was closed (e.g. a premature lifespan shutdown), so the facade never
    serves "client has been closed" across sessions/requests. Reusing one client
    per proxy keeps connection pools warm instead of spawning a client per call.
    """

    def __init__(
        self,
        request_timeout: float,
        direct_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._request_timeout = request_timeout
        # The direct (None-proxy) client may be injected (tests pass one so respx
        # can intercept it); proxied clients are always created on demand.
        self._clients: dict[str | None, httpx.AsyncClient] = {}
        if direct_client is not None:
            self._clients[None] = direct_client

    def client_for(self, proxy: str | None) -> httpx.AsyncClient:
        """Return the client bound to ``proxy`` (None = direct), creating it lazily.

        A ``socks5://`` / ``socks5h://`` / ``http://`` URL is passed straight to
        httpx; with socksio installed, ``socks5://`` already resolves the target
        hostname remotely (proxy-side DNS), like ``curl --socks5-hostname``.
        """
        client = self._clients.get(proxy)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                timeout=self._request_timeout,
                follow_redirects=True,
                proxy=proxy,
            )
            self._clients[proxy] = client
        return client

    async def aclose(self) -> None:
        """Close every open client."""
        for client in self._clients.values():
            if not client.is_closed:
                await client.aclose()
        self._clients.clear()


class Pipeline:
    """Holds the enabled provider instances and runs the search/read logic."""

    def __init__(
        self,
        settings: Settings,
        search_instances: list[SearchProvider],
        read_instances: list[ReadProvider],
        client: httpx.AsyncClient | None = None,
        paid_names: set[str] | frozenset[str] | None = None,
    ) -> None:
        self._settings = settings
        self._search = search_instances
        self._read = read_instances
        # Names of the enabled instances whose TYPE bills per successful request.
        self._paid: frozenset[str] = frozenset(paid_names or ())
        # Cumulative BILLED counters for this process: paid calls and total
        # calls. They are in-memory and reset on restart; the per-request
        # `paid_calls=` field in each log line lets the full history be
        # re-aggregated from the log file across restarts.
        self._cum_paid = 0
        self._cum_calls = 0
        # Do NOT create clients eagerly: build() runs before the event loop
        # starts, and an httpx.AsyncClient created here would be bound to (and
        # later closed by) the wrong loop/lifespan. The manager creates them
        # lazily inside the running loop. An injected client (tests) becomes the
        # direct, no-proxy client so respx can intercept it.
        self._clients = ClientManager(settings.request_timeout, direct_client=client)

    # -- construction -------------------------------------------------------

    @classmethod
    def build(cls, settings: Settings, client: httpx.AsyncClient | None = None) -> "Pipeline":
        """Resolve ENV, construct enabled instances, validate, and return.

        Raises ``ConfigError`` if no search or no read instance is enabled.
        """
        built: dict[str, object] = {}
        paid_names: set[str] = set()

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
                proxy=config.proxy,
            )
            if config.proxy:
                logger.info("Provider instance '{}' routes via proxy", inst.name)
            try:
                built[inst.name] = cls_impl(config)
            except Exception as exc:  # noqa: BLE001 — provider __init__ guard
                logger.warning("Provider instance '{}' failed to build: {}", inst.name, exc)
                continue
            # Track which enabled instances bill per successful request (used
            # only for the paid-vs-free accounting in the per-request logs).
            if inst.type in PAID_TYPES:
                paid_names.add(inst.name)
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

        return cls(  # type: ignore[arg-type]
            settings,
            search_instances,
            read_instances,
            client=client,
            paid_names=paid_names,
        )

    async def aclose(self) -> None:
        """Close every per-proxy httpx client."""
        await self._clients.aclose()

    @property
    def search_names(self) -> list[str]:
        return [p.name for p in self._search]

    @property
    def read_names(self) -> list[str]:
        return [p.name for p in self._read]

    # -- usage accounting ---------------------------------------------------

    def _account(self, billed: list[str]) -> tuple[int, float]:
        """Fold one request's billed upstream calls into the cumulative counters.

        `billed` = names of provider instances whose upstream call returned data
        (a billed 200; thin results count, raised/errored calls do not). Returns
        (paid_calls_this_request, cumulative_paid_percent). Mutates the counters
        synchronously (no await), so it is safe under asyncio.gather concurrency.
        """
        paid = sum(1 for name in billed if name in self._paid)
        self._cum_paid += paid
        self._cum_calls += len(billed)
        pct = (100.0 * self._cum_paid / self._cum_calls) if self._cum_calls else 0.0
        return paid, pct

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

        async def _one(provider: SearchProvider) -> tuple[str, list[SearchResult] | None]:
            # Returns (name, results) where results is None if the provider
            # failed/crashed (so it is NOT counted as "really worked"). Each
            # provider uses the client bound to ITS proxy (None = direct).
            client = self._clients.client_for(provider.proxy)
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
        # The billed calls are exactly `used`: each successful search provider
        # returned data (a billed 200). Fold them into the cumulative counters.
        paid_calls, pct = self._account(used)
        # One per-request line for the persistent log (no bodies/secrets).
        logger.info(
            "search query={!r} providers={} results={} paid_calls={} "
            "cum_paid={} cum_calls={} paid_pct={:.1f}% elapsed_ms={}",
            query,
            used,
            len(merged),
            paid_calls,
            self._cum_paid,
            self._cum_calls,
            pct,
            elapsed_ms,
        )
        return merged

    # -- read ---------------------------------------------------------------

    async def read(self, url: str) -> str:
        """Return clean Markdown for ``url`` (PDF-aware, with provider fallback).

        Raises ``ProviderError`` if every method fails.
        """
        started = time.monotonic()

        def _ms() -> int:
            return int((time.monotonic() - started) * 1000)

        # Accounting state for this request: every provider entered in the
        # fallback chain (in order), and the subset whose upstream call returned
        # content without raising (a billed 200; thin results count too).
        tried: list[str] = []
        billed: list[str] = []

        def _log_ok(provider_name: str, suffix: str = "") -> None:
            # Fold the billed calls into the cumulative counters and emit the
            # single success line. Used by the pdf / full / thin branches so the
            # accounting fields stay identical everywhere.
            paid_calls, pct = self._account(billed)
            logger.info(
                "read url={} -> provider={} ok=true paid_calls={} cum_paid={} "
                "cum_calls={} paid_pct={:.1f}% tried={} elapsed_ms={}" + suffix,
                url,
                provider_name,
                paid_calls,
                self._cum_paid,
                self._cum_calls,
                pct,
                tried,
                _ms(),
            )

        # 1) One probe GET decides the path. If it is a PDF, we are done; if it
        #    is HTML, reuse that body for the trafilatura step (no second GET).
        #    The probe is a generic fetch + PDF/HTML detect, so it uses the
        #    direct client (the proxied providers fetch with their own client).
        #    The probe is NOT a provider call, so it is never billed.
        try:
            pdf_text, probe_html = await self._probe(self._clients.client_for(None), url)
        except ProviderError:
            paid_calls, pct = self._account(billed)  # billed is empty here
            logger.warning(
                "read url={} -> FAILED ok=false tried={} paid_calls={} "
                "cum_paid={} cum_calls={} paid_pct={:.1f}% elapsed_ms={} "
                "errors=pdf probe failed",
                url,
                tried,
                paid_calls,
                self._cum_paid,
                self._cum_calls,
                pct,
                _ms(),
            )
            raise
        if pdf_text is not None:
            _log_ok("pdf")
            return pdf_text

        # 2) HTML path: walk the read pipeline until one yields enough content.
        errors: list[str] = []
        best_thin: str | None = None
        best_thin_name: str | None = None
        for provider in self._read:
            tried.append(provider.name)
            try:
                content = await self._read_one(provider, url, probe_html)
            except ProviderError as exc:
                errors.append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 — treat as provider failure
                errors.append(f"{provider.name}: {exc}")
                continue
            # Returned without raising → a billed 200 (even if too thin).
            billed.append(provider.name)
            if len(content) >= self._settings.fallback_min_chars:
                _log_ok(provider.name)
                return content
            # Too thin — remember the longest thin result as a last resort.
            if best_thin is None or len(content) > len(best_thin):
                best_thin = content
                best_thin_name = provider.name
            errors.append(f"{provider.name}: content too thin ({len(content)} chars)")

        if best_thin:
            _log_ok(best_thin_name or "", suffix=" (thin fallback)")
            return best_thin
        paid_calls, pct = self._account(billed)
        logger.warning(
            "read url={} -> FAILED ok=false tried={} paid_calls={} cum_paid={} "
            "cum_calls={} paid_pct={:.1f}% elapsed_ms={} errors={}",
            url,
            tried,
            paid_calls,
            self._cum_paid,
            self._cum_calls,
            pct,
            _ms(),
            "; ".join(errors),
        )
        raise ProviderError("Не удалось прочитать страницу. " + "; ".join(errors))

    async def _read_one(self, provider: ReadProvider, url: str, probe_html: str | None) -> str:
        """Run one read provider, reusing the probe body for trafilatura.

        trafilatura is a pure HTML→Markdown extractor, so when the probe already
        downloaded the page we extract from that body instead of GETting it
        again (read_page is a hot path). All other providers fetch with the
        client bound to THEIR proxy (None = direct).
        """
        if probe_html is not None and isinstance(provider, TrafilaturaRead):
            content = extract_markdown(probe_html)
            if not content:
                raise ProviderError(f"{provider.name}: no main content extracted")
            return content
        return await provider.read(self._clients.client_for(provider.proxy), url)

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
