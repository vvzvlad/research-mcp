"""The provider pipeline: load enabled instances, run search and read.

At startup ``Pipeline.build`` resolves the ENV-variable NAMES from
``pipeline_config.INSTANCES`` into values, constructs the enabled instances
(skipping any whose required variables are unset, with a log line), and asserts
that at least one search and one read instance are enabled.

Search runs all enabled ``SEARCH_PIPELINE`` instances concurrently, merges and
deduplicates by normalized url (pipeline order wins), optionally reranks the
full merged list with the Jina reranker (``src/rerank.py`` — enabled by
``JINA_API_KEY`` + ``settings.search_rerank_enabled``, falls back to the merge
order on any failure), and trims to ``num_results``. It returns a
``SearchOutcome``: the results plus which instances answered, came back empty
or failed — so the caller can tell "nothing matched" from "the search itself
broke".

Read first detects PDFs (Content-Type / ``.pdf`` suffix / ``%PDF`` magic) and
extracts them with pypdf. Otherwise it tries the enabled ``READ_PIPELINE``
instances in order; the first to return content ``>= fallback_min_chars`` wins;
thin/empty/error → next instance. It returns a ``ReadOutcome``: the markdown
plus the winning provider, the chain that was walked and every failure along the
way tagged with a ``src.failure_reason`` category. If all fail, it raises
``ReadFailed`` (a ``ProviderError``) carrying the same telemetry.

``search_and_read`` composes the two: it runs an over-fetched search and reads
the top hits in waves (each wave only as wide as the previous one's failures)
until ``num_results`` pages have opened, returning a ``SearchReadOutcome``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import os
import ssl
import time
from urllib.parse import urlsplit, urlunsplit

import httpx
from loguru import logger

from src import failure_reason
from src.config_errors import ConfigError
from src.pipeline_config import (
    INSTANCES,
    PAID_TYPES,
    READ_PIPELINE,
    SEARCH_PIPELINE,
    Instance,
)
import src.providers  # noqa: F401  (import the package for its @register side effects)
from src.providers._url_guard import ensure_url_allowed
from src.providers.base import (
    BROWSER_USER_AGENT,
    ProviderConfig,
    ProviderError,
    ReadProvider,
    SearchProvider,
    SearchResult,
)
from src.providers.pdf import NO_TEXT_LAYER_NOTICE, extract_pdf_text, looks_like_pdf
from src.providers.registry import REGISTRY
from src.providers.trafilatura import TrafilaturaRead, extract_markdown
from src.rerank import JinaReranker
from src.settings import Settings


@dataclass(slots=True)
class SearchOutcome:
    """One search request: its results plus what the instances actually did.

    The three buckets are disjoint and together cover ``attempted``: an instance
    either answered with hits, answered with nothing, or failed. They exist so a
    caller can tell an honestly empty result set ("nobody has this") from a dead
    search ("every provider fell over"), which look identical when only
    ``results`` is returned.
    """

    results: list[SearchResult]  # merged, deduped, reranked, trimmed
    attempted: list[str]  # instances launched, in pipeline order
    answered: list[str]  # returned at least one hit
    empty: list[str]  # returned without error but zero hits
    failed: list[str]  # raised ProviderError or crashed
    failed_reasons: list[str]  # why, one src.failure_reason per entry of `failed`
    hits_before_dedup: int  # total hits across the answering providers
    reranked: bool
    elapsed_ms: int


@dataclass(slots=True)
class ReadOutcome:
    """One read request: the markdown plus how the pipeline got hold of it.

    ``provider`` is the instance that won (``"pdf"`` for the PDF path, which no
    instance serves), ``tried`` is the chain walked up to and including it, and
    ``failures`` pairs every instance that did not deliver with its
    ``src.failure_reason`` category. ``thin`` marks the last-resort branch: every
    instance came back under ``fallback_min_chars`` and the longest of those
    scraps is what is being returned.
    """

    markdown: str
    provider: str
    tried: list[str]
    failures: list[tuple[str, str]]  # (instance, reason constant)
    thin: bool
    elapsed_ms: int


@dataclass(slots=True)
class ReadItem:
    """One entry of a ``SearchReadOutcome``: a search hit plus what reading it gave.

    The search fields always carry the hit as the search returned it; the read
    fields are mutually exclusive — ``ok`` means ``markdown``, otherwise
    ``error`` (the message) and ``reason`` (a ``src.failure_reason`` constant,
    for the caller to label).
    """

    title: str
    url: str
    snippet: str
    ok: bool
    markdown: str | None = None
    error: str | None = None
    reason: str | None = None


@dataclass(slots=True)
class SearchReadOutcome:
    """One search_and_read request: the entries plus what they cost.

    ``search`` is the underlying search (over-fetched, so its result count is
    larger than ``items``), ``candidates`` is how many hits it brought back and
    ``read_attempts`` how many of them a read was actually spent on — the two
    numbers that say how much of the over-fetch the failures ate.
    """

    items: list[ReadItem]  # ok first in search order, then failed; <= num_results
    search: SearchOutcome
    candidates: int
    read_attempts: int


class ReadFailed(ProviderError):
    """Every read method failed. Carries the telemetry of the failed attempt.

    A plain ``ProviderError`` by inheritance — callers that only want the message
    keep working — with ``tried`` / ``failures`` attached for the ones that want
    to tell the model *why* nothing opened.
    """

    def __init__(self, message: str, tried: list[str], failures: list[tuple[str, str]]) -> None:
        super().__init__(message)
        self.tried = tried
        self.failures = failures


def classify_read_failure(exc: BaseException) -> str:
    """The one failure category to report for a url that did not open.

    A ``ReadFailed`` already carries a classified reason per provider, so the
    dominant one speaks for the whole chain; anything else (the SSRF guard, an
    unexpected crash) is classified on the spot.
    """
    if isinstance(exc, ReadFailed):
        return failure_reason.dominant_reason(exc.failures)
    return failure_reason.classify(exc)


def _is_tls_verify_error(exc: Exception) -> bool:
    """True if ``exc`` (or a cause in its chain) is a TLS certificate
    verification failure. httpx wraps these in ``httpx.ConnectError`` whose
    underlying cause is ``ssl.SSLCertVerificationError``."""
    seen = 0
    cur: BaseException | None = exc
    while cur is not None and seen < 6:
        if isinstance(cur, ssl.SSLCertVerificationError):
            return True
        if "CERTIFICATE_VERIFY_FAILED" in str(cur):
            return True
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return False


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

    There are TWO caches: plain clients (``client_for``) and SSRF-guarded ones
    (``guarded_client_for``), whose request event hook checks every outgoing url
    — the initial one and every redirect hop. Plain clients stay unguarded on
    purpose: searxng/crawl4ai are internal services on private addresses.
    """

    def __init__(
        self,
        request_timeout: float,
        direct_client: httpx.AsyncClient | None = None,
        allow_private: bool = False,
    ) -> None:
        self._request_timeout = request_timeout
        self._allow_private = allow_private
        # The direct (None-proxy) client may be injected (tests pass one so respx
        # can intercept it); proxied clients are always created on demand.
        self._clients: dict[str | None, httpx.AsyncClient] = {}
        if direct_client is not None:
            self._clients[None] = direct_client
        # Guarded clients are NEVER injected: an injected client carries no event
        # hook, so it could not enforce the guard.
        self._guarded_clients: dict[str | None, httpx.AsyncClient] = {}

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

    async def _guard_request(self, request: httpx.Request) -> None:
        """httpx request hook: block a url that points into the internal network.

        httpx runs it for the initial request AND for every redirect hop, which
        is what closes the "public url redirects to 127.0.0.1" hole.
        """
        await ensure_url_allowed(str(request.url), allow_private=self._allow_private)

    def guard_event_hooks(self) -> dict[str, list[Callable[[httpx.Request], Awaitable[None]]]]:
        """The ``event_hooks`` dict that makes an httpx client SSRF-guarded.

        Exposed so a client created outside this manager (the throwaway insecure
        retry client in the probe) carries the same hook.
        """
        return {"request": [self._guard_request]}

    def guarded_client_for(self, proxy: str | None) -> httpx.AsyncClient:
        """Like ``client_for``, but every request (and redirect) is SSRF-checked.

        Used for the urls WE fetch on the model's behalf. Always created here —
        an injected client is never reused as a guarded one.
        """
        client = self._guarded_clients.get(proxy)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                timeout=self._request_timeout,
                follow_redirects=True,
                proxy=proxy,
                event_hooks=self.guard_event_hooks(),
            )
            self._guarded_clients[proxy] = client
        return client

    async def aclose(self) -> None:
        """Close every open client (both caches)."""
        for cache in (self._clients, self._guarded_clients):
            for client in cache.values():
                if not client.is_closed:
                    await client.aclose()
            cache.clear()


class Pipeline:
    """Holds the enabled provider instances and runs the search/read logic."""

    def __init__(
        self,
        settings: Settings,
        search_instances: list[SearchProvider],
        read_instances: list[ReadProvider],
        client: httpx.AsyncClient | None = None,
        paid_names: set[str] | frozenset[str] | None = None,
        reranker: JinaReranker | None = None,
    ) -> None:
        self._settings = settings
        self._search = search_instances
        self._read = read_instances
        # Post-merge reranker (None = step disabled). Not a pipeline instance:
        # it transforms the merged list instead of producing results, so it is
        # wired separately from the provider lists.
        self._reranker = reranker
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
        self._clients = ClientManager(
            settings.request_timeout,
            direct_client=client,
            allow_private=settings.allow_private_network,
        )

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
            # Provider-specific knobs travel in `options` — the designated
            # slot per ProviderConfig's docstring, so no dedicated config
            # field is added per provider.
            options: dict[str, str] = {}
            if inst.type == "jina":
                # Per-request token cap for the Reader; jina.py sends it as
                # X-Token-Budget only in keyed (billed) mode.
                options["token_budget"] = str(settings.jina_token_budget)
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
                options=options,
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

        # Unreachable on any normal configuration since duckduckgo needs no ENV
        # and is therefore always enabled — kept as the guard against a
        # SEARCH_PIPELINE that names no buildable instance (a rename, a deleted
        # Instance, a provider whose __init__ raised).
        if not search_instances:
            raise ConfigError(
                "No search provider enabled. duckduckgo needs no config, so this "
                "should not happen — check src/pipeline_config.py."
            )
        if not read_instances:
            raise ConfigError(
                "No read provider enabled. trafilatura needs no config, so this "
                "should not happen — check src/pipeline_config.py."
            )

        # The post-merge reranker rides on the same key (and proxy) as the
        # other jina instances. Env access belongs here in build() like every
        # other secret lookup — rerank code never touches os.environ. No key →
        # the step silently disables itself; settings.search_rerank_enabled is
        # the explicit ops kill-switch on top of that.
        reranker: JinaReranker | None = None
        jina_key = os.getenv("JINA_API_KEY")
        if settings.search_rerank_enabled and jina_key:
            reranker = JinaReranker(jina_key, proxy=os.getenv("JINA_PROXY"))
            # The rerank call is metered (input tokens), so its name joins the
            # paid set for the per-request accounting.
            paid_names.add(reranker.name)
            logger.info("Search rerank enabled (jina-reranker-v3.5)")

        return cls(  # type: ignore[arg-type]
            settings,
            search_instances,
            read_instances,
            client=client,
            paid_names=paid_names,
            reranker=reranker,
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
        (a billed 200; thin read results count, raised/errored calls do not). A
        search instance that answered with ZERO hits is NOT billed either
        (SearXNG alive but its engines blocked, a keyed provider returning an
        empty page): it stays out of BOTH counters, so the ratio keeps a single
        meaning — of the calls that actually bought data, how many were paid.
        The list may also carry the pseudo-instance name "jina-rerank" for a
        successful rerank call — deliberately not an Instance in
        pipeline_config, so do not look for it in INSTANCES. Returns
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
    ) -> SearchOutcome:
        """Run all search instances concurrently, merge + dedup, trim.

        Returns a ``SearchOutcome`` — the results plus the per-instance
        answered/empty/failed telemetry.
        """
        # Defend the public method too: a non-positive count would otherwise
        # silently return nothing. (The server already does max(1, ...).)
        num_results = max(1, num_results)
        started = time.monotonic()

        async def _one(provider: SearchProvider) -> tuple[str, list[SearchResult] | None, str]:
            # Returns (name, results, reason) where results is None if the
            # provider failed/crashed (so it is NOT counted as "really worked")
            # and reason is its failure category (empty string when it worked).
            # Each provider uses the client bound to ITS proxy (None = direct).
            client = self._clients.client_for(provider.proxy)
            try:
                hits = await provider.search(client, query, num_results, page, language)
                return provider.name, hits, ""
            except ProviderError as exc:
                logger.info("search '{}' failed: {}", provider.name, exc)
                return provider.name, None, failure_reason.classify(exc)
            except Exception as exc:  # noqa: BLE001 — never break the merge
                logger.warning("search '{}' crashed: {}", provider.name, exc)
                return provider.name, None, failure_reason.classify(exc)

        attempted = [p.name for p in self._search]
        # Gather in pipeline order; results keep that order so dedup prefers the
        # earlier (higher-priority) provider.
        batches = await asyncio.gather(*(_one(p) for p in self._search))

        # Three disjoint buckets, filled from what _one already distinguishes:
        # None = the instance failed, [] = it answered with nothing.
        answered: list[str] = []
        empty: list[str] = []
        failed: list[str] = []
        failed_reasons: list[str] = []
        hits_before_dedup = 0
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for name, hits, reason in batches:
            if hits is None:
                failed.append(name)
                failed_reasons.append(reason or failure_reason.OTHER)
                continue
            if not hits:
                empty.append(name)
                continue
            answered.append(name)
            hits_before_dedup += len(hits)
            for result in hits:
                key = _normalize_url(result.url)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(result)

        # Rerank BEFORE the trim, on the FULL merged list — that is the point:
        # the reranker picks the best num_results from everything the providers
        # brought, instead of trimming blindly by pipeline order and shuffling
        # the surviving prefix. This is a serial await after the concurrent
        # provider gather, so its latency lands on every web_search; accepted
        # for the ordering quality, and settings.search_rerank_enabled is the
        # off-switch. A single result (or none) has nothing to reorder.
        reranked = False
        if self._reranker is not None and len(merged) > 1:
            try:
                merged = await self._reranker.rerank(
                    self._clients.client_for(self._reranker.proxy),
                    query,
                    merged,
                    top_n=num_results,
                )
                reranked = True
            except ProviderError as exc:
                # Graceful degradation: keep the original merge order below.
                logger.info("search rerank failed: {}", exc)
            except Exception as exc:  # noqa: BLE001 — never break the search
                logger.warning("search rerank failed: {}", exc)
        merged = merged[:num_results]

        elapsed_ms = int((time.monotonic() - started) * 1000)
        # The billed calls are exactly `answered`: those providers returned data
        # (a billed 200) — plus the rerank call when it went through (metered
        # too). An instance that answered with zero hits bought nothing, so it
        # stays out (see _account). A rerank that answered 200 with an anomalous
        # body (empty/malformed ranking → ProviderError) is deliberately NOT
        # billed, consistent with search providers whose response failed to
        # parse. The rerank joins only the accounting list, never the
        # providers=[...] one: it produced no results of its own.
        billed = [*answered, self._reranker.name] if reranked and self._reranker else answered
        paid_calls, pct = self._account(billed)
        # One per-request line for the persistent log (no bodies/secrets).
        logger.info(
            "search query={!r} providers={} empty={} failed={} reasons={} results={} "
            "reranked={} paid_calls={} cum_paid={} cum_calls={} paid_pct={:.1f}% elapsed_ms={}",
            query,
            answered,
            empty,
            failed,
            failed_reasons,
            len(merged),
            "true" if reranked else "false",
            paid_calls,
            self._cum_paid,
            self._cum_calls,
            pct,
            elapsed_ms,
        )
        return SearchOutcome(
            results=merged,
            attempted=attempted,
            answered=answered,
            empty=empty,
            failed=failed,
            failed_reasons=failed_reasons,
            hits_before_dedup=hits_before_dedup,
            reranked=reranked,
            elapsed_ms=elapsed_ms,
        )

    # -- read ---------------------------------------------------------------

    async def read(self, url: str) -> ReadOutcome:
        """Read ``url`` (PDF-aware, with provider fallback) into a ``ReadOutcome``.

        Raises ``ReadFailed`` if every method fails, or ``UrlNotAllowed`` (both
        are ``ProviderError``) if the url points into the internal network.
        """
        # SSRF entry check: a blocked url costs zero HTTP requests and gets a
        # clean error. The guarded clients re-check the same url when the probe
        # goes out — deliberate: this check is for the early error, the hook is
        # for every hop (redirects included).
        await ensure_url_allowed(url, allow_private=self._settings.allow_private_network)
        started = time.monotonic()

        def _ms() -> int:
            return int((time.monotonic() - started) * 1000)

        # Accounting state for this request: every provider entered in the
        # fallback chain (in order), and the subset whose upstream call returned
        # content without raising (a billed 200; thin results count too).
        tried: list[str] = []
        billed: list[str] = []
        # Model-facing telemetry: one (instance, reason) pair per attempt that
        # did not deliver — the structured twin of the `errors` texts below, so
        # both lists stay in step.
        failures: list[tuple[str, str]] = []

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
        # The probe never hard-fails: a fetch error or an unparseable "PDF"
        # defers to the read chain below (jina/tavily/firecrawl fetch server-side).
        pdf_text, probe_html = await self._probe(self._clients.guarded_client_for(None), url)
        # A PDF with a text layer is done here. A PDF WITHOUT one is a scan:
        # pypdf has nothing to give and used to return the notice as a success,
        # which meant a scan never reached the read chain at all. Fall through
        # instead, keeping the notice as the last resort if the chain also comes
        # back empty (the behaviour callers had before).
        #
        # What the chain can actually do with a scan, precisely: jina, tavily,
        # firecrawl and brightdata fetch the file server-side with their own
        # parsers and may find text pypdf could not. crawl4ai also fetches on
        # its own side, but it is a headless browser with no OCR, so on a scan
        # it is as useless as trafilatura. And trafilatura, which runs FIRST, is
        # an HTML-only extractor that always fails here — worse, probe_html is
        # None on this branch, so it cannot reuse the probe body and re-downloads
        # the whole file to our host before failing. That download is part of
        # the price of this fall-through.
        #
        # jina's OCR tier is the strongest chance, but it has TWO gates: the url
        # PATH must end in .pdf (_is_pdf_url in providers/jina.py) and jina must
        # be running keyed (the whole ladder sits behind `if api_key`). So a
        # keyless deployment never reaches OCR at all, and neither does a scan
        # served from something like /download?id=123 — this branch also fires
        # for PDFs recognised by Content-Type or %PDF magic alone. How often
        # that shape of url occurs we have not counted. Widening the gate would
        # mean passing the probe's verdict down into the provider, which the
        # ReadProvider protocol has no room for today.
        pdf_notice: str | None = None
        if pdf_text is not None:
            if pdf_text != NO_TEXT_LAYER_NOTICE:
                _log_ok("pdf")
                # tried stays empty: the probe is not a provider call.
                return ReadOutcome(
                    markdown=pdf_text,
                    provider="pdf",
                    tried=list(tried),
                    failures=list(failures),
                    thin=False,
                    elapsed_ms=_ms(),
                )
            pdf_notice = pdf_text

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
                failures.append((provider.name, failure_reason.classify(exc)))
                continue
            except Exception as exc:  # noqa: BLE001 — treat as provider failure
                errors.append(f"{provider.name}: {exc}")
                failures.append((provider.name, failure_reason.classify(exc)))
                continue
            # Returned without raising → a billed 200 (even if too thin). One
            # jina success may hide up to three extra billed upstream calls
            # (readerlm-v2 at 3x, +x-proxy at 5x, jina-ocr-v1 at 40x) not
            # reflected here — the provider emits one "... escalation for url="
            # line per step, only after that step returned, so grepping that
            # suffix counts the extra billed calls. See _ESCALATIONS in
            # src/providers/jina.py for the current list of labels.
            billed.append(provider.name)
            if len(content) >= self._settings.fallback_min_chars:
                _log_ok(provider.name)
                return ReadOutcome(
                    markdown=content,
                    provider=provider.name,
                    tried=list(tried),
                    failures=list(failures),
                    thin=False,
                    elapsed_ms=_ms(),
                )
            # Too thin — remember the longest thin result as a last resort.
            if best_thin is None or len(content) > len(best_thin):
                best_thin = content
                best_thin_name = provider.name
            errors.append(f"{provider.name}: content too thin ({len(content)} chars)")
            failures.append((provider.name, failure_reason.EMPTY))

        if best_thin:
            _log_ok(best_thin_name or "", suffix=" (thin fallback)")
            return ReadOutcome(
                markdown=best_thin,
                provider=best_thin_name or "",
                tried=list(tried),
                failures=list(failures),
                thin=True,
                elapsed_ms=_ms(),
            )
        if pdf_notice:
            # Scanned PDF and nothing in the chain could read it either: hand
            # back the same notice this branch returned before OCR existed. The
            # winner is still "pdf" — the notice came from the probe, not from a
            # provider — while `tried`/`failures` carry the chain that was spent
            # trying to OCR it.
            _log_ok("pdf", suffix=" (no text layer)")
            return ReadOutcome(
                markdown=pdf_notice,
                provider="pdf",
                tried=list(tried),
                failures=list(failures),
                thin=False,
                elapsed_ms=_ms(),
            )
        paid_calls, pct = self._account(billed)
        logger.warning(
            "read url={} -> FAILED ok=false tried={} reasons={} paid_calls={} cum_paid={} "
            "cum_calls={} paid_pct={:.1f}% elapsed_ms={} errors={}",
            url,
            tried,
            # Flat categories, like the search line's reasons= — the instance
            # names are already in tried=, and a grep over reasons= must see the
            # same shape in both lines.
            [reason for _, reason in failures],
            paid_calls,
            self._cum_paid,
            self._cum_calls,
            pct,
            _ms(),
            "; ".join(errors),
        )
        raise ReadFailed(
            "Не удалось прочитать страницу. " + "; ".join(errors),
            tried=list(tried),
            failures=list(failures),
        )

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
        if isinstance(provider, TrafilaturaRead):
            # trafilatura is the only read provider that fetches the target url
            # from OUR process, so it is the only one that needs the SSRF-guarded
            # client. Every other read provider POSTs the url to its own API and
            # never fetches it from here.
            return await provider.read(self._clients.guarded_client_for(provider.proxy), url)
        return await provider.read(self._clients.client_for(provider.proxy), url)

    async def _probe(self, client: httpx.AsyncClient, url: str) -> tuple[str | None, str | None]:
        """Fetch ``url`` once and classify it.

        Returns ``(pdf_text, html)``:
        - PDF detected → ``(extracted_text, None)``.
        - HTML fetched → ``(None, body_text)`` so the caller can reuse the body.
        - probe could not fetch the page → ``(None, None)``; the read pipeline then
          tries its providers (jina/tavily/firecrawl fetch server-side and can often
          retrieve pages/PDFs the direct client cannot — SSL/403/anti-bot).

        A TLS certificate-verification failure triggers ONE retry without
        verification (some legitimate hosts ship a broken chain). The probe never
        hard-fails anymore — a failed fetch always defers to the provider chain,
        and a body that looks like a PDF but fails to parse ALSO defers to the
        provider chain instead of raising.
        """
        response = await self._probe_fetch(client, url)
        if response is None:
            return None, None
        content_type = response.headers.get("Content-Type")
        if looks_like_pdf(url, content_type, response.content[:8]):
            try:
                return extract_pdf_text(response.content), None
            except ProviderError:
                # Looked like a PDF (usually just the .pdf suffix) but the bytes
                # do not parse as one — typically an antibot/HTML interstitial
                # served at a .pdf URL, or a truncated download. Do NOT hard-fail:
                # defer to the read chain, where jina/tavily/firecrawl fetch
                # server-side and can retrieve the real PDF (mirrors the 403
                # defer-to-provider behaviour above).
                logger.warning(
                    "read url={} -> looks like PDF but bytes did not parse; "
                    "deferring to provider chain",
                    url,
                )
                return None, None
        return None, response.text

    async def _probe_fetch(self, client: httpx.AsyncClient, url: str) -> httpx.Response | None:
        """GET ``url`` for the probe; return the response or ``None`` if unfetchable.

        On a TLS certificate-verification error, retry once with verification
        disabled (public read-only fetch; we accept the MITM risk and warn).
        """
        try:
            response = await client.get(url, headers={"User-Agent": BROWSER_USER_AGENT})
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            if not _is_tls_verify_error(exc):
                return None
        # TLS verification failed → one insecure retry on a throwaway client.
        logger.warning("read url={} -> TLS verification failed; retrying without verification", url)
        try:
            async with httpx.AsyncClient(
                verify=False,
                timeout=self._settings.request_timeout,
                follow_redirects=True,
                # Same SSRF hook as the guarded clients: dropping verification
                # must not also drop the address check.
                event_hooks=self._clients.guard_event_hooks(),
            ) as insecure:
                response = await insecure.get(url, headers={"User-Agent": BROWSER_USER_AGENT})
                response.raise_for_status()
                _ = response.content  # force-read the body before the client closes
                return response
        except httpx.HTTPError:
            return None

    # -- search + read (combined) -------------------------------------------

    async def search_and_read(
        self,
        query: str,
        num_results: int,
        page: int,
        language: str | None,
        candidates: int,
    ) -> SearchReadOutcome:
        """Search, then read the top hits, and return both in one outcome.

        Pure composition of ``search`` and ``read``: every provider decision,
        failover and per-request log line stays where it already lives.

        ``candidates`` is the OVER-FETCHED search count, computed by the caller
        (the tool owns the cap on how many results a search may ask for): some
        urls never open, so the search is asked for more hits than the
        ``num_results`` pages we owe. Those extra candidates are NOT all read up
        front — a read costs money — but in WAVES: the first ``num_results``
        candidates are read concurrently under ``read_pages_concurrency``, and
        each following wave reads exactly as many untouched candidates as the
        previous wave failed to open, until enough pages are in hand or the
        candidates run out.
        """
        num_results = max(1, num_results)
        # A caller that asked for fewer candidates than pages would cap the
        # answer below what it requested; the over-fetch is never negative.
        candidates = max(num_results, candidates)
        outcome = await self.search(query, candidates, page, language)
        hits = outcome.results

        semaphore = asyncio.Semaphore(self._settings.read_pages_concurrency)

        async def _one(hit: SearchResult) -> ReadItem:
            # Never raises: a url that did not open becomes a failed entry, so a
            # single dead link cannot take the whole wave down.
            async with semaphore:
                try:
                    read = await self.read(hit.url)
                except ProviderError as exc:
                    error, reason = str(exc), classify_read_failure(exc)
                except Exception as exc:  # noqa: BLE001 — never break the wave
                    error = f"Непредвиденная ошибка: {exc}"
                    reason = classify_read_failure(exc)
                else:
                    return ReadItem(
                        title=hit.title,
                        url=hit.url,
                        snippet=hit.snippet,
                        ok=True,
                        markdown=read.markdown,
                    )
            return ReadItem(
                title=hit.title,
                url=hit.url,
                snippet=hit.snippet,
                ok=False,
                error=error,
                reason=reason,
            )

        opened: list[ReadItem] = []
        failed: list[ReadItem] = []
        next_hit = 0
        attempts = 0
        while len(opened) < num_results and next_hit < len(hits):
            # Take exactly what is still missing: num_results on the first pass,
            # then one candidate per url the previous wave failed to open.
            wave = hits[next_hit : next_hit + (num_results - len(opened))]
            next_hit += len(wave)
            attempts += len(wave)
            for item in await asyncio.gather(*(_one(hit) for hit in wave)):
                (opened if item.ok else failed).append(item)

        # Pages that opened come first, in search order; the failed ones fill
        # whatever is left of the quota, so the list never exceeds num_results.
        items = opened[:num_results]
        items.extend(failed[: num_results - len(items)])

        # Per-url lines are emitted by read() and the query line by search();
        # this one ties them together with what the over-fetch actually cost.
        logger.info(
            "search_and_read query={!r} candidates={} attempts={} read={} results={}",
            query,
            len(hits),
            attempts,
            len(opened),
            len(items),
        )
        return SearchReadOutcome(
            items=items,
            search=outcome,
            candidates=len(hits),
            read_attempts=attempts,
        )
