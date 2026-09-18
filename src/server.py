"""FastMCP server wiring: build the facade and register the 4 research tools.

Tool descriptions are in Russian (LLM-facing); code and comments are in English.
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
            "Поиск в вебе. Агрегирует несколько источников (SearXNG-метапоиск + при "
            "наличии Brave/Serper/Exa), мёржит и дедуплицирует результаты. Возвращает "
            "ранжированный список: заголовок, URL, сниппет.\n\n"
            "Параметры:\n"
            "- query: поисковый запрос. Один запрос = одна тема; для разных тем "
            "вызывай отдельно.\n"
            "- num_results: сколько результатов вернуть (по умолчанию 8, максимум 50).\n"
            "- page: номер страницы выдачи (по умолчанию 1) — для более глубоких "
            "результатов.\n"
            '- language: код языка для приоритета (например "ru", "en"); по умолчанию '
            "без ограничения.\n\n"
            "Это ТОЛЬКО поиск, он НЕ читает страницы. Чтобы получить содержимое — "
            "возьми url из результата и передай в read_page (или несколько url в "
            "read_pages).\n\n"
            "Когда брать именно его: нужны только ссылки и сниппеты — осмотреться по "
            "теме, набрать источники, проверить, существует ли что-то вообще. Если "
            "содержимое страниц всё равно понадобится — не делай поиск и чтение двумя "
            "вызовами, бери search_and_read (поиск + содержимое за один вызов)."
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
            "Скачать ОДНУ веб-страницу или PDF по url и вернуть основной текст в "
            "чистом Markdown. Сам определяет тип и сам выбирает способ извлечения.\n\n"
            "Параметр:\n"
            "- url: полный http(s)-адрес страницы или PDF.\n\n"
            "Как работает: HTML — чистится от навигации/футера/сайдбара/рекламы; для "
            "JS-страниц и сайтов за бот-защитой автоматически задействуются более "
            "тяжёлые методы извлечения; PDF — извлекается текстовый слой (OCR нет, "
            "отсканированные PDF без текста вернут пусто).\n\n"
            "Ошибку вернёт только если страница недоступна всеми способами. НЕ ретрай "
            "такой url повторно — это не транзиентный сбой.\n"
            "Для нескольких url за один вызов — read_pages.\n\n"
            "Когда брать именно его: url ровно один и он уже известен; только здесь "
            "страница возвращается целиком, без обрезки по объёму. Несколько известных "
            "url — read_pages; url ещё неизвестны и тему надо исследовать — "
            "search_and_read."
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
            "Скачать НЕСКОЛЬКО страниц или PDF за один вызов (до 20) — каждую в чистый "
            "Markdown, как read_page (с тем же авто-определением типа и перебором "
            "способов извлечения). Используй это вместо цикла из read_page, когда "
            "нужно прочитать пачку url.\n\n"
            "Параметр:\n"
            "- urls: список http(s)-адресов (до 20).\n\n"
            "Возвращает объект {summary, pages}: summary — строка состояния по батчу, "
            "pages — список объектов {url, ok, markdown|error, reason}: ok=false с "
            "текстом ошибки и категорией причины для тех url, что не открылись всеми "
            "способами, остальные — с markdown.\n\n"
            "Слишком длинный текст страницы обрезается с явной пометкой "
            "[содержимое обрезано на N символах].\n"
            "Когда брать именно его: url уже известны — из прошлой выдачи или от "
            "пользователя. Если url ещё неизвестны, не делай web_search + read_pages "
            "двумя шагами (лишний round-trip и лишний контекст) — вызови "
            "search_and_read. Один url целиком, без обрезки — read_page."
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
            "Поиск в вебе + содержимое верхних результатов за ОДИН вызов. Дефолтный "
            "инструмент для исследования темы: не нужно сначала звать web_search, а "
            "потом read_pages — экономит round-trip и контекст на промежуточной "
            "выдаче.\n\n"
            "Параметры:\n"
            "- query: поисковый запрос. Один запрос = одна тема; для разных тем "
            "вызывай отдельно.\n"
            "- num_results: сколько ПРОЧИТАННЫХ страниц вернуть (по умолчанию 5, "
            "максимум 20).\n"
            "- page: номер страницы выдачи (по умолчанию 1).\n"
            '- language: код языка для приоритета (например "ru", "en"); по умолчанию '
            "без ограничения.\n\n"
            "Как работает: поиск запрашивается с запасом (часть url не открывается), "
            "верхние результаты читаются волнами, пока не наберётся num_results "
            "прочитанных страниц; сначала идут успешно прочитанные (в порядке выдачи), "
            "затем неудачные — в остаток лимита. Каждая страница — чистый Markdown, "
            "как в read_page; слишком длинный текст обрезается с явной пометкой "
            "[содержимое обрезано на N символах].\n\n"
            "Возвращает объект {summary, results}: summary — строки состояния по "
            "поиску и по чтению, results — список объектов "
            "{title, url, snippet, ok, markdown|error, reason}.\n\n"
            "Когда брать именно его: тема исследуется с нуля и нужен текст страниц, а "
            "не только ссылки. Нужны ТОЛЬКО ссылки и сниппеты — web_search. Url уже "
            "известны — read_pages (несколько) или read_page (один url, целиком и без "
            "обрезки)."
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
