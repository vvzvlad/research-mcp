"""SearXNG search provider (self-hosted metasearch).

API: GET ``{url}/search?q=&format=json&pageno=&language=`` →
``{"results": [{"url", "title", "content", ...}], ...}``.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

# Minimum spacing between two SearXNG queries from this process.
#
# Measured on prod 2026-08-09: the only engine still working from our IP is
# DuckDuckGo (google/brave/startpage/mojeek/qwant/yandex/bing are disabled as
# broken). DDG cuts us off after ~4 back-to-back queries and keeps the block for
# 7-8 minutes (8 rapid queries → only the first 4 returned hits). With a 45s gap
# 8 of 8 queries came back with a full result set, so 45s is the safe pace.
_MIN_INTERVAL_SECONDS = 45.0


@register("searxng")
class SearxngSearch:
    """Search via a self-hosted SearXNG instance (requires ``url``).

    Rate-limited locally to one query per ``_MIN_INTERVAL_SECONDS`` — by SKIPPING,
    never by waiting (see ``search``).
    """

    def __init__(self, config: ProviderConfig) -> None:
        if not config.url:
            raise ValueError("searxng requires a url")
        self.name = config.name
        self.proxy = config.proxy
        self._base = config.url.rstrip("/")
        self._config = config
        # monotonic() timestamp of the last query this instance let through.
        # -inf (not 0.0): monotonic()'s zero point is arbitrary and can be in the
        # future relative to process start, so 0.0 could throttle the very first
        # query. -inf guarantees it always passes.
        self._last_call: float = float("-inf")

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int,
        page: int,
        language: str | None,
    ) -> list[SearchResult]:
        # Local throttle, SKIP semantics: if the slot is taken this instance drops
        # out of the current search immediately. It must NEVER sleep here —
        # Pipeline.search() runs every search provider concurrently and awaits all
        # of them, so waiting 45s would stall the whole web_search behind us
        # (median gap between our searches is 2s; ~88% of them would wait).
        #
        # Raise ProviderError instead of returning []: Pipeline._one() catches it,
        # logs "search '<name>' failed: ..." and leaves the instance out of the
        # providers=[...] list. An empty list would instead be recorded as a
        # successful (and billed-looking) run and the log would claim searxng took
        # part when it did not.
        #
        # The check and the assignment are adjacent with no await between them, so
        # under asyncio they are atomic — no lock is needed here, do not add one.
        #
        # _last_call is stamped BEFORE the request, so an attempt that fails
        # spends the slot too — even failures DDG never saw (searxng down,
        # connection refused, our own timeout). That is a deliberate fail-closed
        # choice: from here we cannot tell how far a failed request got, and
        # pacing one wasted slot is far cheaper than a 7-8 minute block. Do NOT
        # "fix" this by moving the assignment below the request.
        now = time.monotonic()
        if now - self._last_call < _MIN_INTERVAL_SECONDS:
            raise ProviderError(
                f"{self.name}: throttled (min interval {_MIN_INTERVAL_SECONDS:.0f}s)"
            )
        self._last_call = now

        params: dict[str, Any] = {"q": query, "format": "json", "pageno": page}
        if language:
            params["language"] = language
        response = await request_with_retry(
            client,
            "GET",
            f"{self._base}/search",
            params=params,
            # retries=0 on purpose, NOT self._config.retries: our bottleneck is a
            # per-IP quota, not a transient blip. A retry fires 0.3s after the
            # first attempt and spends the same 45s slot, so it re-creates
            # exactly the back-to-back pattern that earns the 7-8 minute DDG
            # block — negative expected value. A failed query simply waits for
            # the next slot instead.
            retries=0,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON (is format=json enabled?)") from exc
        results = data.get("results")
        if not isinstance(results, list):
            return []
        out: list[SearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            if not url:
                continue
            out.append(
                SearchResult(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("content") or "").strip(),
                    source=self.name,
                )
            )
        return out
