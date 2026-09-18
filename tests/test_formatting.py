"""Pure formatting tests for the search-result renderer."""

from src.formatting import format_search_results
from src.pipeline import SearchOutcome
from src.providers.base import SearchResult


def _outcome(results, *, answered=("searxng",), empty=(), failed=()) -> SearchOutcome:
    """Wrap `results` in a minimal outcome; the renderer only reads the results
    and the answered/empty buckets."""
    return SearchOutcome(
        results=list(results),
        attempted=[*answered, *empty, *failed],
        answered=list(answered),
        empty=list(empty),
        failed=list(failed),
        hits_before_dedup=len(results),
        reranked=False,
        elapsed_ms=7,
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
