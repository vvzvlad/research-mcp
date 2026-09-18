"""Pure formatting helpers for the research facade.

No I/O here — these functions take parsed pipeline results and render compact,
LLM-friendly strings, so they are trivially unit-testable.

Besides the results themselves, this module renders the per-call STATUS LINE: a
single short line of pipeline telemetry (who answered, what was dropped, what
broke and why, how long it took). One line per call on purpose — it is a signal,
not a dump, so it never enumerates the individual attempts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from src import failure_reason

if TYPE_CHECKING:  # import for typing only — keeps this module I/O-free
    from src.pipeline import ReadOutcome, SearchOutcome

# The model-facing names of the failure categories (the constants stay English).
_REASON_LABELS = {
    failure_reason.TIMEOUT: "таймаут",
    failure_reason.RATE_LIMIT: "лимит запросов",
    failure_reason.NO_CREDITS: "нет кредитов",
    failure_reason.ACCESS_DENIED: "отказ в доступе",
    failure_reason.BOT_PROTECTION: "бот-защита",
    failure_reason.TLS: "TLS",
    failure_reason.DNS: "DNS",
    failure_reason.NETWORK: "сеть",
    failure_reason.EMPTY: "пусто",
    failure_reason.OTHER: "прочее",
}


def reason_label(reason: str) -> str:
    """Russian label for a ``src.failure_reason`` constant (unknown → прочее)."""
    return _REASON_LABELS.get(reason, _REASON_LABELS[failure_reason.OTHER])


def _reason_labels(reasons: Iterable[str]) -> list[str]:
    """Labels for ``reasons``, deduplicated, first occurrence first."""
    labels: list[str] = []
    for reason in reasons:
        label = reason_label(reason)
        if label not in labels:
            labels.append(label)
    return labels


def _seconds(elapsed_ms: int) -> str:
    return f"{elapsed_ms / 1000:.1f} с"


def _failed_part(count: int, reasons: Iterable[str]) -> str:
    """``ошибок: N`` plus the categories behind them, when there are any."""
    labels = _reason_labels(reasons) if count else []
    if labels:
        return f"ошибок: {count} ({', '.join(labels)})"
    return f"ошибок: {count}"


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


def format_search_status(outcome: SearchOutcome) -> str:
    """One line of search telemetry, appended under every web_search answer.

    Tells the model what the number of results is worth: how many of the
    launched instances actually answered (and which), how much the dedup/trim
    dropped, how many came back empty, and what broke — by category, not by
    exception text.
    """
    answered = f"ответили {len(outcome.answered)} из {len(outcome.attempted)}"
    if outcome.answered:
        answered += f" ({', '.join(outcome.answered)})"
    return "; ".join(
        (
            f"Статус поиска: {answered}",
            f"хитов {outcome.hits_before_dedup} → результатов {len(outcome.results)}",
            f"пусто: {len(outcome.empty)}",
            _failed_part(len(outcome.failed), outcome.failed_reasons),
            _seconds(outcome.elapsed_ms),
        )
    )


def format_read_status(outcome: ReadOutcome) -> str:
    """One line of read telemetry: who delivered, after how many providers.

    The caller puts it behind a ``---`` separator so it cannot be mistaken for
    part of the page.

    The PDF branch gets its own wording: no read provider is involved there (the
    probe body is extracted locally), and "провайдеров испробовано: 0" next to a
    named winner reads as a contradiction.
    """
    if not outcome.tried:
        return f"Статус чтения: {outcome.provider} (извлечено локально); {_seconds(outcome.elapsed_ms)}"
    return (
        f"Статус чтения: {outcome.provider} "
        f"(провайдеров испробовано: {len(outcome.tried)}); {_seconds(outcome.elapsed_ms)}"
    )


def format_read_failure_status(reason: str, tried: int) -> str:
    """One line for a read that delivered nothing: the category and the chain.

    The counterpart of ``format_read_status`` for the error path, so a failed
    read_page carries the same categorized signal as a failed url inside a batch
    instead of the raw aggregated exception text.
    """
    return f"Статус чтения: не прочитано ({reason_label(reason)}); провайдеров испробовано: {tried}"


def truncate_markdown(markdown: str, max_chars: int) -> str:
    """Cut ``markdown`` to ``max_chars`` and mark how much was dropped.

    The marker sits on its own line so it cannot be read as page text, and it
    carries the NUMBER of dropped characters — "there is more, this much more"
    is what lets the model decide whether to go after the rest. ``max_chars <= 0``
    disables the cut; content that fits is returned untouched (no marker).
    """
    if max_chars <= 0 or len(markdown) <= max_chars:
        return markdown
    dropped = len(markdown) - max_chars
    return f"{markdown[:max_chars]}\n\n[содержимое обрезано на {dropped} символах]"


def format_batch_status(read_ok: int, failed: int, reasons: Sequence[str]) -> str:
    """One line of read_pages telemetry: how much of the batch actually opened.

    ``reasons`` holds one ``src.failure_reason`` constant per failed url; the
    line shows the distinct categories, never the per-url detail.
    """
    total = read_ok + failed
    return f"Статус чтения: прочитано {read_ok} из {total}; {_failed_part(failed, reasons)}"


def format_search_read_status(
    read_ok: int, attempted: int, candidates: int, reasons: Sequence[str]
) -> str:
    """One line of search_and_read telemetry: what the reads cost and yielded.

    ``format_batch_status`` cannot serve here: that tool is handed a fixed list
    of urls, while this one PICKS its urls — it over-fetches ``candidates`` hits
    and spends reads on them in waves until enough pages opened. So the line
    counts the reads actually attempted, not the size of the answer.

    ``reasons`` holds the categories of the failed reads that are IN the returned
    list; a failure that a later wave topped up is counted but has no entry left
    to explain, so the categories can cover only part of the failures.
    """
    failed = max(0, attempted - read_ok)
    return (
        f"Статус чтения: прочитано {read_ok} из {attempted} "
        f"(кандидатов: {candidates}); {_failed_part(failed, reasons)}"
    )
