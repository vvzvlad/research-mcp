"""Brave Search API provider (independent web index).

API (verified live 2026-08-09): GET
``https://api.search.brave.com/res/v1/web/search`` with headers ``Accept:
application/json``, ``Accept-Encoding: gzip`` and ``X-Subscription-Token``;
query params ``q``, ``count``, ``country``, ``search_lang``, ``offset`` →
top-level keys ``["type", "query", "mixed", "web"]``, where ``web`` is
``{"type", "results", "family_friendly"}`` and every ``web.results[]`` item
carries ``title`` / ``url`` / ``description`` (plus meta_url, profile, thumbnail,
language, ...). The snippet lives in ``description`` here — NOT ``content`` as in
searxng, nor ``snippet`` as in serper. A response may omit ``web`` entirely (e.g.
nothing found), which is a normal empty answer, not a failure.

Free-plan quota, read off the response headers (verified 2026-08-09):
``x-ratelimit-limit: 1, 2000`` / ``x-ratelimit-policy: 1;w=1, 2000;w=2678400`` —
one request per second and 2000 per month. Hence the local throttle below.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Brave caps `count` (results per page) at 20.
BRAVE_COUNT_MAX = 20

# Brave caps `offset` (the 0-based page index) at 9, i.e. page 10 is the deepest
# page it will serve; anything above that is a 422. The tool's `page` argument
# has no upper bound of its own, so this provider refuses deeper pages itself —
# see `search` for why it refuses rather than clamping.
BRAVE_OFFSET_MAX = 9

# Every value Brave's `search_lang` accepts, verbatim from the `Language` enum
# of the official Web Search OpenAPI schema (52 codes, read 2026-08-09 from
# api-dashboard.search.brave.com/api-reference/web/search/get → `search_lang` →
# `#/components/schemas/Language`). Anything outside this set is a 422 — note
# how little of it is plain ISO 639-1: no bare `zh` or `pt`, and `en-gb` /
# `pt-br` / `pt-pt` are regional.
_BRAVE_LANGS = frozenset(
    {
        "ar", "eu", "bn", "bg", "ca", "zh-hans", "zh-hant", "hr", "cs", "da",
        "nl", "en", "en-gb", "et", "fi", "fr", "gl", "de", "el", "gu",
        "he", "hi", "hu", "is", "it", "ja", "jp", "kn", "ko", "lv",
        "lt", "ms", "ml", "mr", "nb", "pl", "pt-br", "pt-pt", "pa", "ro",
        "ru", "sr", "sk", "sl", "es", "sv", "ta", "te", "th", "tr",
        "uk", "vi",
    }
)  # fmt: skip

# Tags a caller is likely to send that are NOT in the list above but do map onto
# one that is. Chinese exists only as script variants there and Portuguese only
# as regional ones, so `zh` / `zh-CN` / `pt` would otherwise be dropped for no
# reason. `ja` → `jp`: the enum happens to accept both, so this only
# canonicalises on Brave's own spelling for Japanese.
_LANG_ALIASES = {
    "ja": "jp",
    "zh": "zh-hans",
    "zh-cn": "zh-hans",
    "zh-sg": "zh-hans",
    "zh-tw": "zh-hant",
    "zh-hk": "zh-hant",
    "pt": "pt-br",
}


def _brave_lang(value: str) -> str | None:
    """Map a caller language tag onto a code Brave actually accepts.

    ``language`` is whatever the LLM passed — ``ru-RU``, ``en_US``, ``EN``,
    ``" ru "`` — while Brave validates ``search_lang`` against the closed list
    above. So normalise separator and case, try the full tag first (that keeps
    regional targeting wherever Brave has it: ``en-gb``, ``pt-br``), then fall
    back to the bare language subtag.

    Returns None when nothing matches: omitting search_lang lets Brave pick its
    default, which beats a guaranteed 422 that would drop us from the merge.
    """
    code = value.strip().replace("_", "-").lower()
    code = _LANG_ALIASES.get(code, code)
    if code in _BRAVE_LANGS:
        return code
    short = code.split("-")[0]
    short = _LANG_ALIASES.get(short, short)
    return short if short in _BRAVE_LANGS else None

# Minimum spacing between two Brave queries from this process.
#
# The free plan allows exactly 1 request per second (``x-ratelimit-policy:
# 1;w=1``, verified 2026-08-09). Our median gap between searches is 2s, but
# bursts of several searches inside one second do happen, so we pace slightly
# above one second to stay clear of the window boundary.
#
# The reaction to a taken slot is a SKIP, not a wait — deliberately, even though
# waiting here would cost 1.1s versus searxng's 45s. Pipeline.search() awaits
# every search provider, so even a one-second wait stretches EVERY web_search by
# that much; and parallel searches from the LLM would just queue up against
# Brave's own per-second limit anyway, so the wait buys nothing.
_MIN_INTERVAL_SECONDS = 1.1


@register("brave")
class BraveSearch:
    """Web search via the Brave Search API (requires ``api_key``).

    Rate-limited locally to one query per ``_MIN_INTERVAL_SECONDS`` — by SKIPPING,
    never by waiting (see ``search``).
    """

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("brave requires an api_key")
        self.name = config.name
        self.proxy = config.proxy
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
        # Deeper than Brave can serve → refuse, do not clamp to the last page.
        # Clamping would spend a throttle slot and one of the 2000 monthly
        # queries just to hand back page 10 again, and pages 11, 12, ... would
        # each re-inject those same hits into the merge (dedup runs within a
        # single search() call, never across calls). This check sits ABOVE the
        # throttle on purpose: a query we never send must not cost a slot.
        if page > BRAVE_OFFSET_MAX + 1:
            raise ProviderError(
                f"{self.name}: page {page} is beyond brave's depth "
                f"(max {BRAVE_OFFSET_MAX + 1})"
            )

        # Local throttle, SKIP semantics (same design as searxng, different
        # reason: here it is Brave's own 1 req/s plan limit). If the slot is taken
        # this instance drops out of the current search immediately. It must NEVER
        # sleep here — Pipeline.search() runs every search provider concurrently
        # and awaits all of them, so a sleeping provider stalls the whole
        # web_search behind it.
        #
        # Raise ProviderError instead of returning []: Pipeline._one() catches it,
        # logs "search '<name>' failed: ..." and leaves the instance out of the
        # providers=[...] list. An empty list would instead be recorded as a
        # successful (and billed-looking) run, and the log would claim brave took
        # part when it did not.
        #
        # The check and the assignment are adjacent with no await between them, so
        # under asyncio they are atomic — no lock is needed here, do not add one.
        #
        # _last_call is stamped BEFORE the request, so an attempt that fails
        # spends the slot too (including failures Brave never saw). Deliberate
        # fail-closed choice: from here we cannot tell how far a failed request
        # got, and one wasted slot is cheaper than tripping the limit. Do NOT
        # "fix" this by moving the assignment below the request.
        now = time.monotonic()
        if now - self._last_call < _MIN_INTERVAL_SECONDS:
            raise ProviderError(
                f"{self.name}: throttled (min interval {_MIN_INTERVAL_SECONDS:.1f}s)"
            )
        self._last_call = now

        params: dict[str, Any] = {
            "q": query,
            "count": max(1, min(num_results, BRAVE_COUNT_MAX)),
            # Brave's `offset` is a 0-based PAGE index, not a result offset.
            # max(0, ...) covers page <= 0: nothing upstream clamps `page`
            # (src/server.py only clamps num_results), and offset=-1 is a 422.
            "offset": max(0, page - 1),
        }
        if language:
            # Unlike serper's Google-compatible `hl` or searxng's `language`,
            # both of which take a regional tag as-is, `search_lang` is checked
            # against a closed list — see _brave_lang. An unmappable tag means
            # no search_lang at all (Brave's default), never a guessed value.
            lang = _brave_lang(language)
            if lang:
                params["search_lang"] = lang
        # `country` is deliberately NOT sent: the tool contract has no country
        # input, and deriving one from `language` would be wrong for languages
        # spoken in many countries (en, es, ru, ...). Brave then uses its own
        # default, which is the better guess.
        response = await request_with_retry(
            client,
            "GET",
            BRAVE_ENDPOINT,
            params=params,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._config.api_key,
            },
            # retries=0 on purpose, NOT self._config.retries: the bottleneck here
            # is a quota (1 req/s), not a transient blip. A retry 0.3s later
            # lands inside the same throttle slot and inside the same rate-limit
            # window, so it can only make things worse. _http.py already treats
            # 429 as a hard failure without retrying; this extends the same logic
            # to 5xx/transport errors for a rate-limited provider.
            retries=0,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        # No `web` block at all (nothing found, or only other verticals matched)
        # is a normal empty answer.
        web = data.get("web")
        results = web.get("results") if isinstance(web, dict) else None
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
                    # Brave calls the snippet "description".
                    snippet=(item.get("description") or "").strip(),
                    url=url,
                    source=self.name,
                )
            )
        return out
