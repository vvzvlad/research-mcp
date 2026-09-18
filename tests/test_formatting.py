"""Pure formatting tests: the search-result renderer and the 3 status lines."""

from src.formatting import (
    format_batch_status,
    format_read_failure_status,
    format_read_status,
    format_search_read_status,
    format_search_results,
    format_search_status,
    truncate_markdown,
)
from src.pipeline import ReadOutcome, SearchOutcome
from src.providers.base import SearchResult


def _outcome(
    results,
    *,
    answered=("searxng",),
    empty=(),
    failed=(),
    failed_reasons=(),
    hits=None,
    elapsed_ms=7,
) -> SearchOutcome:
    """Wrap `results` in a minimal outcome; the renderer only reads the results
    and the answered/empty buckets."""
    return SearchOutcome(
        results=list(results),
        attempted=[*answered, *empty, *failed],
        answered=list(answered),
        empty=list(empty),
        failed=list(failed),
        failed_reasons=list(failed_reasons),
        hits_before_dedup=len(results) if hits is None else hits,
        reranked=False,
        elapsed_ms=elapsed_ms,
    )


def _results():
    return [
        SearchResult(
            title="First result",
            url="https://example.com/a",
            snippet="Snippet about the first thing.",
            source="searxng",
        ),
        SearchResult(
            title="Second result",
            url="https://example.com/b",
            snippet="Snippet about the second thing.",
            source="serper",
        ),
    ]


def test_renders_title_url_snippet():
    out = format_search_results(_outcome(_results()), query="foo", page=1)
    assert "First result" in out
    assert "https://example.com/a" in out
    assert "Snippet about the first thing." in out
    assert "1. **First result**" in out
    assert "2. **Second result**" in out


def test_empty_results_message():
    # At least one instance answered → an honestly empty result set.
    out = format_search_results(_outcome([]), query="nothing here", page=2)
    assert "ничего не найдено" in out.lower()
    assert "nothing here" in out


def test_snippet_newlines_collapsed():
    results = [SearchResult(title="t", url="u", snippet="line one\nline two", source="x")]
    out = format_search_results(_outcome(results), query="q", page=1)
    assert "line one line two" in out


# -- status lines ----------------------------------------------------------
#
# These pin the exact rendered text: the status line is a contract with the
# model, so a silent reword must fail the suite.


def test_search_status_line_format():
    outcome = _outcome(
        _results(),
        answered=("searxng", "serper"),
        empty=("exa",),
        failed=("brave",),
        failed_reasons=("rate-limit",),
        hits=14,
        elapsed_ms=1234,
    )
    assert format_search_status(outcome) == (
        "Статус поиска: ответили 2 из 4 (searxng, serper); "
        "хитов 14 → результатов 2; пусто: 1; ошибок: 1 (лимит запросов); 1.2 с"
    )


def test_search_status_line_when_everything_failed():
    # No answering instance → no name list, and the reason categories carry the
    # whole signal.
    outcome = _outcome(
        [],
        answered=(),
        failed=("searxng", "exa"),
        failed_reasons=("timeout", "no-credits"),
        hits=0,
        elapsed_ms=2000,
    )
    assert format_search_status(outcome) == (
        "Статус поиска: ответили 0 из 2; хитов 0 → результатов 0; пусто: 0; "
        "ошибок: 2 (таймаут, нет кредитов); 2.0 с"
    )


def test_read_status_line_format():
    outcome = ReadOutcome(
        markdown="# page",
        provider="crawl4ai",
        tried=["trafilatura", "jina", "crawl4ai"],
        failures=[("trafilatura", "empty"), ("jina", "rate-limit")],
        thin=False,
        elapsed_ms=1900,
    )
    assert format_read_status(outcome) == (
        "Статус чтения: crawl4ai (провайдеров испробовано: 3); 1.9 с"
    )


def test_read_status_line_names_the_local_pdf_path():
    # The PDF branch runs no read provider at all, so "провайдеров испробовано: 0"
    # next to a named winner would read as a contradiction.
    outcome = ReadOutcome(
        markdown="text",
        provider="pdf",
        tried=[],
        failures=[],
        thin=False,
        elapsed_ms=1500,
    )
    assert format_read_status(outcome) == "Статус чтения: pdf (извлечено локально); 1.5 с"


def test_read_failure_status_line_format():
    line = format_read_failure_status("bot-protection", 4)
    assert line == "Статус чтения: не прочитано (бот-защита); провайдеров испробовано: 4"


def test_batch_status_line_format():
    line = format_batch_status(3, 2, ["bot-protection", "timeout"])
    assert line == "Статус чтения: прочитано 3 из 5; ошибок: 2 (бот-защита, таймаут)"


def test_batch_status_line_deduplicates_categories():
    line = format_batch_status(0, 3, ["timeout", "timeout", "dns"])
    assert line == "Статус чтения: прочитано 0 из 3; ошибок: 3 (таймаут, DNS)"


def test_batch_status_line_without_failures_lists_no_categories():
    assert format_batch_status(2, 0, []) == "Статус чтения: прочитано 2 из 2; ошибок: 0"


def test_search_read_status_line_format():
    # 3 pages out of 7 reads spent on 12 over-fetched candidates.
    line = format_search_read_status(3, 7, 12, ["timeout", "dns"])
    assert line == (
        "Статус чтения: прочитано 3 из 7 (кандидатов: 12); ошибок: 4 (таймаут, DNS)"
    )


def test_search_read_status_line_counts_failures_it_cannot_name():
    # Every failed read was topped up by a later wave, so no entry is left to
    # explain them — the count must still be honest.
    line = format_search_read_status(2, 5, 6, [])
    assert line == "Статус чтения: прочитано 2 из 5 (кандидатов: 6); ошибок: 3"


# -- the per-page content budget -------------------------------------------


def test_truncate_marks_the_number_of_dropped_characters():
    out = truncate_markdown("x" * 100, 40)
    assert out == "x" * 40 + "\n\n[содержимое обрезано на 60 символах]"


def test_truncate_leaves_content_that_fits_untouched():
    assert truncate_markdown("short page", 40) == "short page"
    # Exactly at the budget is still "fits" — nothing was dropped.
    assert truncate_markdown("x" * 40, 40) == "x" * 40


def test_truncate_disabled_by_a_non_positive_budget():
    assert truncate_markdown("x" * 100, 0) == "x" * 100
