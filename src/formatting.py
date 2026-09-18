"""Pure formatting helpers for the research facade.

No I/O here — these functions take parsed pipeline results and render compact,
LLM-friendly strings, so they are trivially unit-testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import for typing only — keeps this module I/O-free
    from src.pipeline import SearchOutcome


def format_search_results(outcome: SearchOutcome, query: str, page: int) -> str:
    """Render a search outcome as a numbered Markdown list.

    Each item is ``N. **title**\\n   url\\n   snippet``. With no results there
    are two distinct answers: a SEARCH FAILURE when every launched instance
    raised (retrying makes sense), and the plain "nothing matched" notice when
    at least one instance answered. ``Pipeline.build`` refuses to start without
    a search instance, so "nothing was launched at all" cannot happen here.
    """
    if not outcome.results:
        if not outcome.answered and not outcome.empty:
            return (
                f'Поиск по запросу "{query}" (стр. {page}) не выполнен: все поисковые '
                "провайдеры вернули ошибку. Это сбой поиска, а не пустая выдача — "
                "имеет смысл повторить запрос позже."
            )
        return f'По запросу "{query}" (стр. {page}) ничего не найдено.'

    lines: list[str] = [f'Результаты поиска: "{query}" (стр. {page})', ""]
    for index, item in enumerate(outcome.results, start=1):
        title = (item.title or "(без заголовка)").strip()
        url = (item.url or "").strip()
        snippet = (item.snippet or "").strip().replace("\n", " ")
        lines.append(f"{index}. **{title}**")
        if url:
            lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)
