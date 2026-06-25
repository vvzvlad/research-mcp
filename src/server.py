"""FastMCP server wiring: build the facade and register the 3 research tools.

Tool descriptions are in Russian (LLM-facing); code and comments are in English.
Each tool wraps the pipeline call in ``try/except`` and returns a clean value (a
string, or a list of dicts for read_pages) so the LLM always gets a usable
result instead of a traceback.

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

from src.formatting import format_search_results
from src.pipeline import Pipeline
from src.providers.base import ProviderError
from src.settings import Settings

# Hard caps baked into the tool descriptions (the docstrings promise these exact
# numbers to the LLM), so they are constants — NOT settings — to keep the
# contract honest regardless of environment overrides.
SEARCH_RESULTS_MAX = 50
READ_PAGES_MAX = 20


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
            "наличии Serper/Exa), мёржит и дедуплицирует результаты. Возвращает "
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
            results = await pipeline.search(query, count, page, language)
        except ProviderError as exc:
            return str(exc)
        return format_search_results(results, query=query, page=page)

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
            return await pipeline.read(url)
        except ProviderError as exc:
            return str(exc)

    @mcp.tool(
        name="read_pages",
        description=(
            "Скачать НЕСКОЛЬКО страниц или PDF за один вызов (до 20) — каждую в чистый "
            "Markdown, как read_page (с тем же авто-определением типа и перебором "
            "способов извлечения). Используй это вместо цикла из read_page, когда "
            "нужно прочитать пачку url.\n\n"
            "Параметр:\n"
            "- urls: список http(s)-адресов (до 20).\n\n"
            "Возвращает список объектов {url, ok, markdown|error}: ok=false с текстом "
            "ошибки для тех url, что не открылись всеми способами, остальные — "
            "с markdown."
        ),
    )
    async def read_pages(urls: list[str]) -> list[dict[str, Any]]:
        capped = urls[:READ_PAGES_MAX]
        semaphore = asyncio.Semaphore(settings.read_pages_concurrency)

        async def _one(url: str) -> dict[str, Any]:
            async with semaphore:
                try:
                    markdown = await pipeline.read(url)
                    return {"url": url, "ok": True, "markdown": markdown}
                except ProviderError as exc:
                    return {"url": url, "ok": False, "error": str(exc)}
                except Exception as exc:  # noqa: BLE001 — never break the batch
                    return {"url": url, "ok": False, "error": f"Непредвиденная ошибка: {exc}"}

        results = await asyncio.gather(*(_one(url) for url in capped))
        # Per-url lines are emitted by pipeline.read; add one batch summary line.
        # (read already logs each url's winning provider/latency individually.)
        ok_count = sum(1 for r in results if r["ok"])
        logger.info("read_pages count={} ok={}", len(capped), ok_count)
        return results

    return mcp
