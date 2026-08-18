"""Jina Search (s.jina.ai) provider.

API (verified 2026-08-18 from https://docs.jina.ai/ and
https://api.jina.ai/docs): POST ``https://s.jina.ai/`` with ``Authorization:
Bearer <key>`` — the key is REQUIRED here (unlike the r.jina.ai reader,
keyless access to the search endpoint is blocked) — plus ``Content-Type:
application/json``, ``Accept: application/json`` and ``X-Respond-With:
no-content``. The ``no-content`` header asks for the SERP only
(title/url/description); without it s.jina.ai VISITS every hit and returns
the full page contents, which is slow, expensive, and redundant next to our
own read pipeline.

JSON body: ``q`` (required); optional ``num`` (results per page), ``page``
(pagination — semantics ambiguous, see the in-code comment), ``hl``
(two-letter language code), ``gl`` / ``location`` (never sent, see below).

Response is the reader-style envelope ``{"code": 200, "status": ...,
"data": [{"title", "url", "description"}, ...]}``. A missing or empty ``data``
list is a normal empty answer (nothing found), not a failure.

Billing: a fixed 10000 tokens per request (~$0.0005 at the $50/1B pack). The
rate limit is 100 RPM on a paid key — generous enough that NO local throttle
is needed (unlike brave's 1 req/s free plan).
"""

from __future__ import annotations

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

JINA_SEARCH_ENDPOINT = "https://s.jina.ai/"

# No documented ceiling for `num` (checked 2026-08-18); clamped to the tool's
# own num_results cap for symmetry with brave/exa, so a future server-side
# change cannot push an unvalidated value upstream.
JINA_SEARCH_NUM_MAX = 50


@register("jina_search")
class JinaSearch:
    """Web search via s.jina.ai (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("jina_search requires an api_key")
        self.name = config.name
        self.proxy = config.proxy
        self._config = config

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int,
        page: int,
        language: str | None,
    ) -> list[SearchResult]:
        body: dict[str, object] = {"q": query}
        # The default SERP depth already covers up to 10 hits, and the docs
        # advise against `num`: "Using num may cause latency and exclude
        # specialized result types. Omit unless you specifically need more
        # results per page." So it is sent only when the caller really wants
        # more than the default page can hold.
        if num_results > 10:
            body["num"] = min(num_results, JINA_SEARCH_NUM_MAX)
        if language:
            # Jina validates `hl` as a two-letter language code, so reduce
            # whatever the LLM passed ("ru-RU", "en_US", " EN ") to its primary
            # subtag and send it only when that really is two ascii letters.
            # Anything else is omitted — jina's own default beats a guaranteed
            # validation error that would drop us from the merge.
            lang = language.strip().replace("_", "-").split("-")[0].lower()
            if len(lang) == 2 and lang.isascii() and lang.isalpha():
                body["hl"] = lang
        # `gl` (country) is deliberately NOT sent: the tool contract has no
        # country input, and deriving one from `language` would be wrong for
        # languages spoken in many countries (en, es, ru, ...) — the same
        # reasoning as brave.py's `country` comment.
        if page > 1:
            # The docs are ambiguous about `page`: the NAME suggests a 1-based
            # page index, but the description reads "The result offset. It
            # skips the given number of results. It's used for pagination." —
            # a skip-count. We cannot live-test which reading wins, so the raw
            # page number goes through as-is: deep pagination through jina is
            # a rare path, and the within-call dedup in Pipeline.search()
            # absorbs a wrong guess (re-served hits collapse against the other
            # providers' pages).
            body["page"] = page
        response = await request_with_retry(
            client,
            "POST",
            JINA_SEARCH_ENDPOINT,
            json=body,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # SERP only (title/url/description). Without this header
                # s.jina.ai fetches every hit's page content — see the module
                # docstring.
                "X-Respond-With": "no-content",
            },
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        # The reader-style envelope can carry an application error with HTTP
        # 200 (e.g. `{"code": 422, "message": ...}`) — HTTP 200 does not mean
        # success for this API. Without this check the instance would be
        # logged and billed as having worked while returning nothing. The code
        # is compared as a trimmed string so an envelope that says "200" as a
        # string still counts as success rather than misfiring.
        if isinstance(data, dict):
            code = data.get("code")
            if code is not None and str(code).strip() != "200":
                raise ProviderError(f"{self.name}: API error (code {code!r})")
        # A missing/empty `data` list (nothing found) is a normal empty answer.
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        out: list[SearchResult] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            if not url:
                continue
            out.append(
                SearchResult(
                    title=(item.get("title") or "").strip(),
                    # The snippet normally lives in `description`; reader-style
                    # payloads sometimes carry `content` instead, so fall back
                    # to it rather than losing the snippet.
                    snippet=(item.get("description") or item.get("content") or "").strip(),
                    url=url,
                    source=self.name,
                )
            )
        return out
