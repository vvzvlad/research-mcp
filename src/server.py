"""FastMCP server wiring: build the facade and register the 3 research tools.

Tool descriptions are in Russian (LLM-facing); code and comments are in English.
Each tool wraps the pipeline call in ``try/except`` and returns a clean value (a
string, or a ``{summary, pages}`` dict for read_pages) so the LLM always gets a
usable result instead of a traceback. Every answer carries one short status line
of pipeline telemetry rendered by ``src/formatting.py`` — results, empty results
and failed reads alike; the only exception is a url the SSRF guard rejected
before any provider ran, where there is no pipeline run to report.

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

from src.failure_reason import classify, dominant_reason
from src.formatting import (
    format_batch_status,
    format_read_failure_status,
    format_read_status,
    format_search_results,
    format_search_status,
    reason_label,
)
from src.pipeline import Pipeline, ReadFailed
from src.providers.base import ProviderError
from src.settings import Settings

# Hard caps baked into the tool descriptions (the docstrings promise these exact
# numbers to the LLM), so they are constants — NOT settings — to keep the
# contract honest regardless of environment overrides.
SEARCH_RESULTS_MAX = 50
READ_PAGES_MAX = 20


def _failure_reason(exc: BaseException) -> str:
    """The one failure category to report for a url that did not open.

    A ``ReadFailed`` already carries a classified reason per provider, so the
    dominant one speaks for the whole chain; anything else (the SSRF guard, an
    unexpected crash) is classified on the spot.
    """
    if isinstance(exc, ReadFailed):
        return dominant_reason(exc.failures)
    return classify(exc)


def build_server(settings: Settings, pipeline: Pipeline | None = None) -> FastMCP:
    """Build a FastMCP facade exposing the 3 research tools.

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
            "read_pages)."
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
            "Для нескольких url за один вызов — read_pages."
        ),
    )
    async def read_page(url: str) -> str:
        try:
            outcome = await pipeline.read(url)
        except ReadFailed as exc:
            # A failed read carries telemetry too: the category says whether
            # another attempt could ever help. Same shape as a failed url in a
            # read_pages batch.
            status = format_read_failure_status(_failure_reason(exc), len(exc.tried))
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
            "способами, остальные — с markdown."
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
                    return {"url": url, "ok": True, "markdown": outcome.markdown}, None
                except ProviderError as exc:
                    reason = _failure_reason(exc)
                    entry = {"url": url, "ok": False, "error": str(exc)}
                except Exception as exc:  # noqa: BLE001 — never break the batch
                    reason = _failure_reason(exc)
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

    return mcp
