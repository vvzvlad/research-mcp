"""Pure formatting helpers for the research facade.

No I/O here — these functions take parsed pipeline results and render compact,
LLM-friendly strings, so they are trivially unit-testable.
"""

from __future__ import annotations

from src.providers.base import SearchResult


def format_search_results(results: list[SearchResult], query: str, page: int) -> str:
    """Render merged search results as a numbered Markdown list.

    Each item is ``N. **title**\\n   url\\n   snippet``. Returns a friendly
    notice when there are no hits.
    """
    if not results:
        return f'По запросу "{query}" (стр. {page}) ничего не найдено.'

    lines: list[str] = [f'Результаты поиска: "{query}" (стр. {page})', ""]
    for index, item in enumerate(results, start=1):
        title = (item.title or "(без заголовка)").strip()
        url = (item.url or "").strip()
        snippet = (item.snippet or "").strip().replace("\n", " ")
        lines.append(f"{index}. **{title}**")
        if url:
            lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)
