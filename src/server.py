"""FastMCP server wiring: build the facade and register the 4 research tools.

Tool descriptions are in English, like the code and comments; the per-answer
status lines rendered by ``src/formatting.py`` are still in Russian.
Each tool wraps the pipeline call in ``try/except`` and returns a clean value (a
string, or a ``{summary, pages}`` / ``{summary, results}`` dict for the batch
tools) so the LLM always gets a usable result instead of a traceback. Every
answer carries a short status line of pipeline telemetry rendered by
``src/formatting.py`` — two of them for ``search_and_read``, since a search and
a batch of reads both ran — on results, empty results and failed reads alike;
the only exception is a url the SSRF guard rejected before any provider ran,
where there is no pipeline run to report.

The descriptions cross-reference each other (when to use this tool and when to
use one of the others), so the model gets a small routing graph instead of four
independent texts.

Transport: streamable-http on ``mcp_host:mcp_port`` (endpoint ``/mcp``). The
server itself does NO auth — Traefik + basicAuth in front of it handles that.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from loguru import logger
from mcp.server.fastmcp import FastMCP

from src.formatting import (
    format_batch_status,
    format_read_failure_status,
    format_read_status,
    format_search_read_status,
    format_search_results,
    format_search_status,
    reason_label,
    truncate_markdown,
)
from src.pipeline import Pipeline, ReadFailed, classify_read_failure
from src.providers.base import ProviderError
from src.settings import Settings

# Hard caps baked into the tool descriptions (the docstrings promise these exact
# numbers to the LLM), so they are constants — NOT settings — to keep the
# contract honest regardless of environment overrides.
SEARCH_RESULTS_MAX = 50
READ_PAGES_MAX = 20
# Default number of pages search_and_read reads per call. Small on purpose: each
# result costs a real read, and the answer carries page text, not snippets.
SEARCH_AND_READ_DEFAULT = 5


# NOTE on `num_results="8"`: weak local models routinely send numbers as
# strings, and the tools accept that already — FastMCP validates arguments in
# pydantic's lax mode, which coerces a numeric string to int and still rejects
# "восемь". A BeforeValidator here would be a no-op, so there is none; the
# behaviour is pinned by tests/test_server.py instead, which is what protects it
# if a future SDK version tightens the mode.


def build_server(settings: Settings, pipeline: Pipeline | None = None) -> FastMCP:
    """Build a FastMCP facade exposing the 4 research tools.

    A single ``Pipeline`` (shared httpx client + enabled provider instances) is
    closed over by all tools and closed when the server shuts down.
    """
    pipeline = pipeline or Pipeline.build(settings)

    @asynccontextmanager
    async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
        """Close the shared httpx client on server shutdown."""
        try:
            yield
        finally:
            await pipeline.aclose()

    mcp = FastMCP(
        "research-mcp",
        host=settings.mcp_host,
        port=settings.mcp_port,
        lifespan=_lifespan,
    )

    @mcp.tool(
        name="web_search",
        description=(
            "Web search. Aggregates several sources (DuckDuckGo out of the box, plus "
            "a self-hosted SearXNG and every paid vendor whose key is configured), "
            "merges and deduplicates the results. Returns a ranked list: title, URL, "
            "snippet.\n\n"
            "Parameters:\n"
            "- query: the search query. One query = one topic; for different topics "
            "call it separately.\n"
            "- num_results: how many results to return (default 8, maximum 50).\n"
            "- page: result page number (default 1) — for deeper results.\n"
            '- language: language code to prioritise (e.g. "ru", "en"); unrestricted '
            "by default.\n\n"
            "This is ONLY search, it does NOT read pages. To get the content, take a "
            "url from a result and pass it to read_page (or several urls to "
            "read_pages).\n\n"
            "When to take this one: you need links and snippets only — to survey a "
            "topic, collect sources, check whether something exists at all. If the "
            "page content will be needed anyway, do not spend two calls on search and "
            "read — take search_and_read (search + content in one call)."
        ),
    )
    async def web_search(
        query: str,
        num_results: int = 8,
        page: int = 1,
        language: str | None = None,
    ) -> str:
        count = max(1, min(num_results, SEARCH_RESULTS_MAX))
        try:
            outcome = await pipeline.search(query, count, page, language)
        except ProviderError as exc:
            return str(exc)
        body = format_search_results(outcome, query=query, page=page)
        # The status line rides under every answer, including the "nothing
        # found" / "search is broken" texts — that is exactly when the model
        # needs to know how many instances were behind the verdict.
        return f"{body}\n\n{format_search_status(outcome)}"

    @mcp.tool(
        name="read_page",
        description=(
            "Download ONE web page or PDF by url and return its main text as clean "
            "Markdown. It detects the type and picks the extraction method itself.\n\n"
            "Parameter:\n"
            "- url: the full http(s) address of a page or a PDF.\n\n"
            "How it works: HTML is stripped of navigation/footer/sidebar/ads; for "
            "JS-rendered pages and sites behind bot protection heavier extraction "
            "methods are engaged automatically; for a PDF the text layer is "
            "extracted, a scan without one may go through recognition, and if no text "
            "could be obtained the answer says so.\n\n"
            "It returns an error only if the page is unreachable by every method. Do "
            "NOT retry such a url — this is not a transient failure.\n"
            "For several urls in one call — read_pages.\n\n"
            "When to take this one: there is exactly one url and it is already known; "
            "this is the only tool that returns a page whole, with no size cap. "
            "Several known urls — read_pages; urls not known yet and the topic has to "
            "be researched — search_and_read."
        ),
    )
    async def read_page(url: str) -> str:
        try:
            outcome = await pipeline.read(url)
        except ReadFailed as exc:
            # A failed read carries telemetry too: the category says whether
            # another attempt could ever help. Same shape as a failed url in a
            # read_pages batch.
            status = format_read_failure_status(classify_read_failure(exc), len(exc.tried))
            return f"{exc}\n\n---\n{status}"
        except ProviderError as exc:
            # Raised before any provider ran (the SSRF guard) — nothing to report.
            return str(exc)
        # Behind a horizontal rule: the status line is about the page, not part
        # of it, and must not read as page text.
        return f"{outcome.markdown}\n\n---\n{format_read_status(outcome)}"

    @mcp.tool(
        name="read_pages",
        description=(
            "Download SEVERAL pages or PDFs in one call (up to 20) — each into clean "
            "Markdown, the way read_page does it (same type detection, same chain of "
            "extraction methods). Use this instead of a loop of read_page calls when "
            "a batch of urls has to be read.\n\n"
            "Parameter:\n"
            "- urls: a list of http(s) addresses (up to 20).\n\n"
            "Returns an object {summary, pages}: summary is one status line for the "
            "batch, pages is a list of {url, ok, markdown|error, reason} — ok=false "
            "with the error text and a failure category for the urls that did not "
            "open by any method, the rest with markdown.\n\n"
            "Page text that is too long is truncated, with an explicit marker naming "
            "how many characters were dropped.\n"
            "When to take this one: the urls are already known — from an earlier "
            "result list or from the user. If they are not known yet, do not spend "
            "two steps on web_search + read_pages (an extra round-trip and extra "
            "context) — call search_and_read. One url whole, uncapped — read_page."
        ),
    )
    async def read_pages(urls: list[str]) -> dict[str, Any]:
        capped = urls[:READ_PAGES_MAX]
        semaphore = asyncio.Semaphore(settings.read_pages_concurrency)

        async def _one(url: str) -> tuple[dict[str, Any], str | None]:
            # Returns (entry, reason) — the reason constant is None for a url
            # that opened, and feeds the batch summary for one that did not.
            async with semaphore:
                try:
                    outcome = await pipeline.read(url)
                    # Batch budget: 20 unbounded pages would wreck the context
                    # window, so each page is capped (read_page is not).
                    markdown = truncate_markdown(outcome.markdown, settings.read_batch_max_chars)
                    return {"url": url, "ok": True, "markdown": markdown}, None
                except ProviderError as exc:
                    reason = classify_read_failure(exc)
                    entry = {"url": url, "ok": False, "error": str(exc)}
                except Exception as exc:  # noqa: BLE001 — never break the batch
                    reason = classify_read_failure(exc)
                    entry = {"url": url, "ok": False, "error": f"Непредвиденная ошибка: {exc}"}
                entry["reason"] = reason_label(reason)
                return entry, reason

        outcomes = await asyncio.gather(*(_one(url) for url in capped))
        pages = [entry for entry, _ in outcomes]
        reasons = [reason for _, reason in outcomes if reason is not None]
        # Per-url lines are emitted by pipeline.read; add one batch summary line.
        # (read already logs each url's winning provider/latency individually.)
        ok_count = sum(1 for page in pages if page["ok"])
        logger.info("read_pages count={} ok={}", len(capped), ok_count)
        return {
            "summary": format_batch_status(ok_count, len(pages) - ok_count, reasons),
            "pages": pages,
        }

    @mcp.tool(
        name="search_and_read",
        description=(
            "Web search + the content of the top results in ONE call. The default "
            "tool for researching a topic: no need to call web_search first and "
            "read_pages after — it saves a round-trip and the context the "
            "intermediate result list would cost.\n\n"
            "Parameters:\n"
            "- query: the search query. One query = one topic; for different topics "
            "call it separately.\n"
            "- num_results: how many READ pages to return (default 5, maximum 20).\n"
            "- page: result page number (default 1).\n"
            '- language: language code to prioritise (e.g. "ru", "en"); unrestricted '
            "by default.\n\n"
            "How it works: the search is over-fetched (some urls will not open), the "
            "top results are read in waves until num_results pages have opened; the "
            "successfully read ones come first (in result order), the failed ones "
            "after them, within what is left of the limit. Each page is clean "
            "Markdown, as in read_page; text that is too long is truncated, with an "
            "explicit marker naming how many characters were dropped.\n\n"
            "Returns an object {summary, results}: summary is the status lines for "
            "the search and for the reads, results is a list of "
            "{title, url, snippet, ok, markdown|error, reason}.\n\n"
            "When to take this one: a topic is being researched from scratch and the "
            "page text is needed, not just links. ONLY links and snippets — "
            "web_search. Urls already known — read_pages (several) or read_page (one "
            "url, whole and uncapped)."
        ),
    )
    async def search_and_read(
        query: str,
        num_results: int = SEARCH_AND_READ_DEFAULT,
        page: int = 1,
        language: str | None = None,
    ) -> dict[str, Any]:
        # Every result costs a real read, so the per-call cap is the read one.
        count = max(1, min(num_results, READ_PAGES_MAX))
        # Over-fetch: ask the search for more candidates than the pages we owe,
        # because some of those urls will not open. The cap on a single search
        # request lives here, so the pipeline is handed the final number.
        candidates = min(count * 2 + 2, SEARCH_RESULTS_MAX)
        try:
            outcome = await pipeline.search_and_read(query, count, page, language, candidates)
        except ProviderError as exc:
            # Raised before any provider ran — there is no run to report.
            return {"summary": str(exc), "results": []}

        results: list[dict[str, Any]] = []
        reasons: list[str] = []
        for item in outcome.items:
            entry: dict[str, Any] = {
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
                "ok": item.ok,
            }
            if item.ok:
                # Same per-page budget as read_pages: this answer carries page
                # text for several urls at once.
                entry["markdown"] = truncate_markdown(
                    item.markdown or "", settings.read_batch_max_chars
                )
            else:
                entry["error"] = item.error
                entry["reason"] = reason_label(item.reason or "")
                reasons.append(item.reason or "")
            results.append(entry)

        read_ok = sum(1 for entry in results if entry["ok"])
        # Two lines, because two pipelines ran: the search that produced the
        # candidates and the reads that were spent on them.
        read_status = format_search_read_status(
            read_ok, outcome.read_attempts, outcome.candidates, reasons
        )
        summary = f"{format_search_status(outcome.search)}\n{read_status}"
        if not outcome.search.results:
            # No candidates at all: an empty `results` list on its own reads as
            # "the topic does not exist" whichever way the search died. Carry the
            # same prose web_search gives, which says in words whether this was a
            # search failure worth repeating or an honestly empty result.
            # Keyed on the SEARCH results, not on `results`: when the search did
            # find urls and only the reads failed, the entries are there and speak
            # for themselves.
            body = format_search_results(outcome.search, query=query, page=page)
            summary = f"{body}\n\n{summary}"
        return {"summary": summary, "results": results}

    return mcp
