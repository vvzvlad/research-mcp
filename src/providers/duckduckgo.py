"""DuckDuckGo search provider — keyless, and the only one that needs no config.

API: POST ``https://html.duckduckgo.com/html/`` with a form-encoded body
``{"q": <query>, "kl": <region>}`` (plus ``s`` for deeper pages) and a browser
User-Agent. This is the no-JS SERP: unlike duckduckgo.com itself it needs no
``vqd`` token handshake, so ONE request returns the whole result page as HTML.

Response shape: each hit is a ``.result`` row whose ``a.result__a`` carries the
title text and the href, and whose ``.result__snippet`` carries the snippet. The
hrefs are WRAPPED in a redirector — ``/l/?uddg=<url-encoded real url>``, often
protocol-relative (``//duckduckgo.com/l/?uddg=...``) — so every url has to be
unwrapped (see ``_unwrap_href``).

There is no key and no quota here, so the failure modes are a block or a markup
change rather than a billing error; ``_parse`` is careful to tell "DuckDuckGo
found nothing" apart from "we did not get a SERP at all". See also the HTTP 202
note in ``search``.
"""

from __future__ import annotations

import time
from urllib.parse import parse_qs, urlparse

import httpx
import lxml.etree
import lxml.html

from src.providers._http import request_with_retry
from src.providers.base import (
    BROWSER_USER_AGENT,
    ProviderConfig,
    ProviderError,
    SearchResult,
)
from src.providers.registry import register

DDG_ENDPOINT = "https://html.duckduckgo.com/html/"

# Results per SERP page. The endpoint's own "next page" form posts `s` as a
# RESULT offset (not a page index), and one no-JS page carries 30 results.
DDG_PAGE_SIZE = 30

# `kl` value meaning "no region at all" — the default, and the fallback for a
# language tag we cannot map.
DDG_REGION_ANY = "wt-wt"

# `kl` values for the languages we can map. DuckDuckGo has no language parameter:
# `kl` is a REGION (locale) code, so a bare language subtag has to be resolved to
# one region — the largest/most neutral one for that language. Anything outside
# this table falls back to DDG_REGION_ANY rather than to a guessed region: a
# wrong region silently skews the whole result set, while `wt-wt` merely declines
# to bias it.
_DDG_REGIONS = {
    "en": "us-en",
    "ru": "ru-ru",
    "de": "de-de",
    "fr": "fr-fr",
    "es": "es-es",
    "it": "it-it",
}

# Minimum spacing between two DuckDuckGo queries from this process.
#
# The endpoint publishes no quota, so this number is NOT ours to guess: it is
# the one measured against this very upstream from this very egress IP, and it
# lives in searxng.py (prod, 2026-08-09 — DuckDuckGo cuts us off after ~4
# back-to-back queries and holds the block for 7-8 minutes; at a 45s gap 8 of 8
# came back whole). A faster pace here would not only cost us this instance: our
# SearXNG reaches DuckDuckGo over the same address, and DuckDuckGo is the only
# engine it still has, so a block taken here takes the main search down with it.
_MIN_INTERVAL_SECONDS = 45.0


def _class_xpath(token: str) -> str:
    """XPath predicate matching ONE whitespace-separated class token.

    ``@class`` on a DDG result row is a token list (``"result results_links
    web-result"``), so a plain ``contains(@class, 'result')`` would also match
    ``result__body`` / ``results``. Padding both sides with spaces makes the
    match exact per token.
    """
    return f"contains(concat(' ', normalize-space(@class), ' '), ' {token} ')"


def _ddg_region(value: str) -> str:
    """Map a caller language tag onto a ``kl`` region code.

    ``language`` is whatever the LLM passed — ``ru-RU``, ``en_US``, ``EN``,
    ``" ru "`` — so normalise separator and case exactly like ``_brave_lang``
    does, then key on the bare language subtag: ``kl`` is a region code, and the
    regional part of the caller's tag is not one (``ru-RU`` is not a ``kl``
    value). Unmappable → ``wt-wt`` (no regional bias).
    """
    code = value.strip().replace("_", "-").lower()
    short = code.split("-")[0]
    return _DDG_REGIONS.get(short, DDG_REGION_ANY)


def _unwrap_href(href: str) -> str:
    """Return the real target url behind a ``/l/?uddg=...`` redirector href.

    DuckDuckGo wraps most result hrefs in its own redirector, sometimes
    protocol-relative (``//duckduckgo.com/l/?uddg=...``) — ``urlparse`` handles
    both forms.

    ``parse_qs`` percent-decodes the value exactly once, which is the whole of
    the wrapping. Decoding a SECOND time would eat the target's own escapes and
    silently produce a different resource: ``...?id=x%23frag`` would turn into a
    fragment, ``%26`` into an extra parameter, ``group%2Fsub`` into a path. A
    Wikipedia link therefore comes back as ``..._%28programming_language%29``
    rather than with literal parentheses — the same page, spelled the way DDG
    spells it.

    No ``uddg`` parameter → the href is already a plain url (DDG serves some
    unwrapped), so it is returned as-is.
    """
    values = parse_qs(urlparse(href).query).get("uddg")
    if not values:
        return href.strip()
    return values[0].strip()


@register("duckduckgo")
class DuckDuckGoSearch:
    """Keyless web search via the no-JS DuckDuckGo SERP (no config at all).

    Always enabled — it is the search half of the zero-config floor, next to the
    ``trafilatura`` reader. Rate-limited locally to one query per
    ``_MIN_INTERVAL_SECONDS`` — by SKIPPING, never by waiting (see ``search``).
    """

    def __init__(self, config: ProviderConfig) -> None:
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
        # Local throttle, SKIP semantics (same design as searxng and brave): if
        # the slot is taken this instance drops out of the current search
        # immediately. It must NEVER sleep here — Pipeline.search() runs every
        # search provider concurrently and awaits all of them, so a sleeping
        # provider stalls the whole web_search behind it.
        #
        # Raise ProviderError instead of returning []: Pipeline._one() catches it
        # and logs "search '<name>' failed: ...", so the instance is recorded as
        # having failed. Returning [] would instead claim duckduckgo ran and
        # found nothing — a skipped slot must never look like an honest empty
        # result, all the more so here, where duckduckgo may be the only search
        # provider a deployment has.
        #
        # The check and the assignment are adjacent with no await between them, so
        # under asyncio they are atomic — no lock is needed here, do not add one.
        #
        # _last_call is stamped BEFORE the request, so an attempt that fails
        # spends the slot too — even failures DuckDuckGo never saw (our own
        # timeout, a dead link). That is a deliberate fail-closed choice: from
        # here we cannot tell how far a failed request got, and pacing one wasted
        # slot is far cheaper than earning a block. Do NOT "fix" this by moving
        # the assignment below the request.
        now = time.monotonic()
        if now - self._last_call < _MIN_INTERVAL_SECONDS:
            raise ProviderError(
                f"{self.name}: throttled (min interval {_MIN_INTERVAL_SECONDS:.0f}s)"
            )
        self._last_call = now

        # `num_results` is deliberately unused: the no-JS SERP has no count
        # parameter, it always serves a full page and the pipeline trims.
        data: dict[str, str] = {
            "q": query,
            "kl": _ddg_region(language) if language else DDG_REGION_ANY,
        }
        # Page 1 omits `s` entirely, exactly like the endpoint's own first
        # request; `page > 1` also covers page <= 0 (nothing upstream clamps it).
        if page > 1:
            data["s"] = str((page - 1) * DDG_PAGE_SIZE)
        response = await request_with_retry(
            client,
            "POST",
            DDG_ENDPOINT,
            data=data,
            # Without a browser UA this endpoint answers with the block page.
            headers={"User-Agent": BROWSER_USER_AGENT},
            # retries=0 on purpose, NOT self._config.retries: the bottleneck here
            # is a burst detector, not a transient blip. A retry fires 0.3s after
            # the first attempt and spends the same throttle slot, so it re-creates
            # exactly the back-to-back pattern that earns a block. Same reasoning
            # as searxng and brave.
            retries=0,
            provider=self.name,
        )
        # HTTP 202 is a SOFT RATE-LIMIT BLOCK here, not a success: DuckDuckGo
        # answers 202 with a "Ratelimit" body instead of a SERP. request_with_retry
        # returns any 2xx as a success, so this provider has to recognise it
        # itself — otherwise the block would reach _parse and be reported as
        # broken markup.
        if response.status_code == 202:
            raise ProviderError(f"{self.name}: rate limited (HTTP 202)")
        return self._parse(response.text)

    def _parse(self, html: str) -> list[SearchResult]:
        """Turn a SERP into results, or raise if this is not a SERP.

        Outcomes, deliberately distinguished (a keyless scraper has no status
        code to tell them apart):

        - usable result rows found → return them;
        - rows were there but not one href could be read → raise: that is the
          redirector format changing under us;
        - DuckDuckGo's own "no results" marker, or rows that were all ads → an
          honest empty answer, return ``[]``: the search ran, it simply left us
          nothing to offer;
        - no rows and no marker → raise: a block page, a captcha or a redesign.

        Listed in the order the checks run, which is the order that matters.

        The two raises exist so a broken parser never reports itself as a
        successful empty search.
        """
        try:
            doc = lxml.html.fromstring(html)
        except lxml.etree.ParserError as exc:
            raise ProviderError(f"{self.name}: empty or unparseable response") from exc

        out: list[SearchResult] = []
        # Whether the page carried any result row at all — including rows we
        # drop below. A page full of ads only is still a page that answered, so
        # it must not be reported as a block.
        rows_present = False
        # A row we dropped because its href made no sense to us. Harmless-looking
        # and NOT the same as an ad: it is what a change to the redirector format
        # looks like from here.
        unusable_href = False
        for row in doc.xpath(f"//*[{_class_xpath('result')}]"):
            links = row.xpath(f".//a[{_class_xpath('result__a')}]")
            # Rows without a title link are the SERP's own furniture (the
            # "more results" row carrying the next-page form), not hits.
            if not links:
                continue
            rows_present = True
            # Ads are shaped exactly like organic hits (`result result--ad`)
            # and would otherwise be handed to the model as search results.
            if row.xpath(f"self::*[{_class_xpath('result--ad')}]"):
                continue
            url = _unwrap_href(links[0].get("href") or "")
            # DDG's own click wrappers (`//duckduckgo.com/y.js?...`) carry no
            # `uddg` and come back scheme-less, which read_page then refuses
            # with an unhelpful error. Anything that is not plain http(s) is
            # not a result we can offer.
            if not url.startswith(("http://", "https://")):
                unusable_href = True
                continue
            snippets = row.xpath(f".//*[{_class_xpath('result__snippet')}]")
            out.append(
                SearchResult(
                    title=links[0].text_content().strip(),
                    url=url,
                    snippet=snippets[0].text_content().strip() if snippets else "",
                    source=self.name,
                )
            )
        if out:
            return out
        if unusable_href:
            # Organic rows were there and we could not read a single href out of
            # them. Returning [] here would report a broken parser as an honest
            # empty search — exactly the silence this provider must not produce.
            raise ProviderError(
                f"{self.name}: every result row had an unrecognised href "
                "(the redirector format changed)"
            )
        if rows_present or doc.xpath(f"//*[{_class_xpath('no-results')}]"):
            # Either DDG said so itself, or every row it sent was an ad: the
            # search ran, it just gave us nothing to offer.
            return []
        raise ProviderError(
            f"{self.name}: no results and no 'no results' marker "
            "(blocked, captcha, or the markup changed)"
        )
