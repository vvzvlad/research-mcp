"""Octen Web Search API provider (live web index, minute-fresh).

API (contract read 2026-09-18 from the official reference,
https://docs.octen.ai/api-reference/search.md — NOT verified against a live
call, we hold no key yet): POST ``https://api.octen.ai/search``. The key goes
either in ``x-api-key`` (apiKey scheme) or as ``Authorization: Bearer <key>``
(http bearer); both are accepted, this module sends ``x-api-key``.

Request body: ``query`` (REQUIRED, maxLength 500, supports the ``site:`` /
``-site:`` operators), ``count`` (1..100, default 5), ``language`` (an ARRAY of
ISO 639-1 codes from a closed 18-value enum, default ``[]`` = no filter),
``topic`` (``general``/``news``), ``time_basis`` / ``time_range`` /
``start_time`` / ``end_time``, ``include_domains`` / ``exclude_domains``,
``include_text`` / ``exclude_text``, ``highlight`` (``{enable: true,
max_tokens: 512}`` by default), ``full_content`` (off by default), ``format``
(``markdown``/``text``, default ``text``), ``safesearch`` (default ``strict``),
``include_images``. There is no page/offset/cursor field: ``count`` is the only
volume knob.

Response: ``{"code", "msg", "request_id", "data", "meta"}`` where ``code: 0``
means success, ``data`` is ``{"query", "results": [...]}`` and every result has
``title`` / ``url`` / ``highlight`` / ``authors`` / ``time_published`` /
``time_last_crawled`` / ``favicon`` (plus ``full_content`` and images when those
options are on). The snippet is ``highlight`` — "Query-relevant highlight
snippets. Returned only if highlight.enable is true" — not ``description``
(brave), ``content`` (linkup/searxng) or ``snippet`` (serper). A missing or
empty ``results`` list is a normal empty answer.

Documented errors: 400 "Missing parameter query", 401 "Invalid API Key", 403
"Insufficient balance in account" (what a depleted account returns — no special
handling belongs in this module, but note that ``_http.py`` matches that wording
against ``_CREDIT_MARKERS`` and reports it as "out of credits (HTTP 403)", so an
empty balance here reads as a billing state and not as a broken request),
429 "Exceeding the rate limit", 500 "Internal error".
"""

from __future__ import annotations

from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

OCTEN_SEARCH_ENDPOINT = "https://api.octen.ai/search"

# `count` is documented as 1..100; anything outside is a 400.
OCTEN_COUNT_MIN = 1
OCTEN_COUNT_MAX = 100

# Every value Octen's `language` enum accepts, verbatim from the request schema
# (18 ISO 639-1 codes, read 2026-09-18). Plain two-letter codes only — no
# regional tags, so `en-GB` or `ru-RU` must be reduced before being sent.
_OCTEN_LANGS = frozenset(
    {
        "ar", "de", "en", "es", "fr", "hi", "id", "it", "ja",
        "ko", "nl", "pl", "pt", "ru", "th", "tr", "vi", "zh",
    }
)  # fmt: skip


def _octen_lang(value: str) -> str | None:
    """Map a caller language tag onto a code Octen's enum accepts.

    ``language`` is whatever the LLM passed — ``ru-RU``, ``en_US``, ``RU``,
    ``" ru "`` — while Octen validates against the closed list above, which holds
    bare subtags only. So normalise separator and case, drop the region, and
    return None when the result is not in the enum: no language filter at all
    (Octen's own default) beats a guaranteed 400 that would drop this instance
    from the merge.
    """
    code = value.strip().replace("_", "-").lower().split("-")[0]
    return code if code in _OCTEN_LANGS else None


def _highlight_text(value: object) -> str:
    """Flatten Octen's ``highlight`` field into one snippet string.

    The docs call these "highlight snippets" (plural) and the request side has a
    ``highlight.max_tokens`` knob, so a LIST of fragments is as plausible as a
    single string — and the neighbouring vendors settle it both ways: Parallel
    returns a list in ``excerpts``, You.com a list in ``snippets`` beside a
    string ``description``. We have no Octen key to settle it by observation, so
    accept both shapes: calling ``.strip()`` on a list would raise
    AttributeError, which ``Pipeline._one`` would log as a crashed provider on
    EVERY query — the instance would simply never return anything.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(part.strip() for part in value if isinstance(part, str) and part.strip())
    return ""


@register("octen_search")
class OctenSearch:
    """Web search via the Octen API (requires ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("octen_search requires an api_key")
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
        # No pagination in the API: page 2+ would re-run the same search and bill
        # for the hits the caller already has, so refuse rather than silently
        # hand back page 1 (same rule as brave beyond its depth limit).
        if page > 1:
            raise ProviderError(
                f"{self.name}: page {page} is unavailable (octen has no pagination)"
            )

        body: dict[str, Any] = {
            "query": query,
            "count": max(OCTEN_COUNT_MIN, min(num_results, OCTEN_COUNT_MAX)),
        }
        if language:
            # `language` is a LIST here, not a scalar like serper's `hl`, and it
            # is checked against a closed enum — see _octen_lang. An unmappable
            # tag means no language key at all, never a guessed value.
            lang = _octen_lang(language)
            if lang:
                body["language"] = [lang]
        # `highlight` is left at its default (enabled, 512 tokens): it is the
        # snippet this provider returns. `full_content` stays off — fetching page
        # bodies is the read pipeline's job and is separately billed here.
        response = await request_with_retry(
            client,
            "POST",
            OCTEN_SEARCH_ENDPOINT,
            json=body,
            headers={
                "x-api-key": self._config.api_key,
                "Content-Type": "application/json",
            },
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        # Application-level failure inside a 200, the way s.jina.ai does it —
        # see the identical guard in jina_search.py. Without this an error
        # envelope would be recorded as a successful (and billed) empty answer.
        # Compared as a string: the documented success value is the number 0,
        # but "0" from a stricter-typing day must not read as an error.
        code = data.get("code")
        if code is not None and str(code).strip() != "0":
            raise ProviderError(f"{self.name}: API error (code {code!r})")
        # Results live one level down, under `data` (the envelope also carries
        # `code`/`msg`/`request_id`/`meta`).
        payload = data.get("data")
        results = payload.get("results") if isinstance(payload, dict) else None
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
                    # Octen calls the snippet "highlight".
                    snippet=_highlight_text(item.get("highlight")),
                    source=self.name,
                )
            )
        return out
