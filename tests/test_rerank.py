"""Jina reranker: index-based reordering, top_n slice, malformed responses.

The response shape mirrors the RerankerV3Request contract read 2026-08-18 from
https://api.jina.ai/openapi.json: ``results`` is a list of ``{"index",
"relevance_score"}`` sorted by relevance descending, where ``index`` points
into the ORIGINAL documents list. Network I/O is mocked with respx.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ProviderError, SearchResult
from src.rerank import (
    RERANK_ENDPOINT,
    RERANK_MODEL,
    RERANK_TIMEOUT_SECONDS,
    JinaReranker,
)


def _results(n: int) -> list[SearchResult]:
    return [
        SearchResult(title=f"t{i}", url=f"https://x.test/{i}", snippet=f"s{i}", source="searxng")
        for i in range(n)
    ]


def _response(indices: list[int]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": RERANK_MODEL,
            "object": "list",
            "usage": {"total_tokens": 100},
            "results": [
                {"index": i, "relevance_score": 1.0 - 0.1 * pos}
                for pos, i in enumerate(indices)
            ],
        },
    )


@respx.mock
async def test_reorders_by_the_returned_indices():
    route = respx.post(RERANK_ENDPOINT).mock(return_value=_response([2, 0, 1]))
    reranker = JinaReranker("k", proxy=None)
    results = _results(3)
    async with httpx.AsyncClient() as client:
        out = await reranker.rerank(client, "q", results, top_n=3)
    assert [r.url for r in out] == [
        "https://x.test/2",
        "https://x.test/0",
        "https://x.test/1",
    ]
    # Request contract: model, query, title+snippet documents, top_n, and
    # return_documents=false (indices are enough — the objects stay local).
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == RERANK_MODEL
    assert body["query"] == "q"
    assert body["documents"] == ["t0\ns0", "t1\ns1", "t2\ns2"]
    assert body["top_n"] == 3
    assert body["return_documents"] is False
    assert route.calls.last.request.headers["Authorization"] == "Bearer k"


@respx.mock
async def test_slices_to_top_n_even_if_the_api_returns_more():
    # The API applies top_n itself, but a response that ignores it must not
    # leak extra results past the caller's limit.
    respx.post(RERANK_ENDPOINT).mock(return_value=_response([2, 0, 1]))
    reranker = JinaReranker("k", proxy=None)
    async with httpx.AsyncClient() as client:
        out = await reranker.rerank(client, "q", _results(3), top_n=2)
    assert [r.url for r in out] == ["https://x.test/2", "https://x.test/0"]


@respx.mock
async def test_duplicate_indices_are_deduplicated():
    respx.post(RERANK_ENDPOINT).mock(return_value=_response([1, 1, 0]))
    reranker = JinaReranker("k", proxy=None)
    async with httpx.AsyncClient() as client:
        out = await reranker.rerank(client, "q", _results(2), top_n=5)
    assert [r.url for r in out] == ["https://x.test/1", "https://x.test/0"]


@respx.mock
async def test_partial_ranking_pads_with_the_unmentioned_results():
    # Rerank must reorder, never filter: a partial ranking (fewer indices than
    # documents) gets the unmentioned results appended in their original merge
    # order, after the ranked ones.
    respx.post(RERANK_ENDPOINT).mock(return_value=_response([1]))
    reranker = JinaReranker("k", proxy=None)
    async with httpx.AsyncClient() as client:
        out = await reranker.rerank(client, "q", _results(4), top_n=4)
    assert [r.url for r in out] == [
        "https://x.test/1",  # the one ranked result comes first
        "https://x.test/0",  # then the unmentioned ones, in original order
        "https://x.test/2",
        "https://x.test/3",
    ]


@respx.mock
async def test_partial_ranking_padding_still_respects_top_n():
    # The pad happens BEFORE the top_n slice, so a partial answer cannot leak
    # more results past the caller's limit.
    respx.post(RERANK_ENDPOINT).mock(return_value=_response([2]))
    reranker = JinaReranker("k", proxy=None)
    async with httpx.AsyncClient() as client:
        out = await reranker.rerank(client, "q", _results(4), top_n=2)
    assert [r.url for r in out] == ["https://x.test/2", "https://x.test/0"]


@respx.mock
async def test_empty_ranking_for_a_nonempty_input_raises():
    # A valid-looking `{"results": []}` is an anomaly, not an answer: returned
    # as-is it would wipe every found result while the pipeline logs
    # reranked=true. It must raise so the caller keeps the merge order.
    respx.post(RERANK_ENDPOINT).mock(return_value=_response([]))
    reranker = JinaReranker("k", proxy=None)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await reranker.rerank(client, "q", _results(3), top_n=3)
    assert "empty ranking" in str(excinfo.value)


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"results": "not-a-list"},  # results of a wrong shape
        {"nothing": "here"},  # results missing entirely
        {"results": [{"index": 99, "relevance_score": 1.0}]},  # index out of range
        {"results": [{"index": -1, "relevance_score": 1.0}]},  # negative index
        {"results": [{"index": "0", "relevance_score": 1.0}]},  # non-int index
        {"results": [{"index": True, "relevance_score": 1.0}]},  # bool is not an index
        {"results": ["not-a-dict"]},  # malformed item
    ],
)
async def test_malformed_response_raises_provider_error(payload):
    # `index` points into OUR list; a bad value would silently reorder the SERP
    # into garbage, so anything suspicious is a hard ProviderError (the
    # pipeline then falls back to the original merge order).
    respx.post(RERANK_ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    reranker = JinaReranker("k", proxy=None)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await reranker.rerank(client, "q", _results(3), top_n=3)


@respx.mock
async def test_hit_with_no_title_and_no_snippet_sends_the_url_as_its_document():
    # A hit with neither title nor snippet must not become an empty document
    # string — the API may 422 the whole batch over one blank entry. The URL
    # stands in as the document then.
    route = respx.post(RERANK_ENDPOINT).mock(return_value=_response([0, 1]))
    reranker = JinaReranker("k", proxy=None)
    results = _results(2)
    results[1] = SearchResult(title="", url="https://x.test/1", snippet="", source="searxng")
    async with httpx.AsyncClient() as client:
        await reranker.rerank(client, "q", results, top_n=2)
    body = json.loads(route.calls.last.request.content)
    assert body["documents"] == ["t0\ns0", "https://x.test/1"]


@respx.mock
async def test_documents_are_truncated_to_1000_chars():
    # jina_search's snippet can fall back to a whole-page "content" field; an
    # unbounded document would silently multiply the rerank input-token cost.
    route = respx.post(RERANK_ENDPOINT).mock(return_value=_response([0, 1]))
    reranker = JinaReranker("k", proxy=None)
    results = _results(2)
    results[0] = SearchResult(
        title="t0", url="https://x.test/0", snippet="s" * 5000, source="searxng"
    )
    async with httpx.AsyncClient() as client:
        await reranker.rerank(client, "q", results, top_n=2)
    body = json.loads(route.calls.last.request.content)
    assert body["documents"][0] == ("t0\n" + "s" * 5000)[:1000]
    assert len(body["documents"][0]) == 1000
    assert body["documents"][1] == "t1\ns1"  # short documents pass untouched


@respx.mock
async def test_top_n_larger_than_the_input_is_clamped_in_the_request():
    # jina's docs default top_n to len(documents); a server-side validation of
    # top_n <= len(documents) would 422 exactly on thin result sets, so the
    # client clamps before sending.
    route = respx.post(RERANK_ENDPOINT).mock(return_value=_response([1, 0]))
    reranker = JinaReranker("k", proxy=None)
    async with httpx.AsyncClient() as client:
        out = await reranker.rerank(client, "q", _results(2), top_n=10)
    body = json.loads(route.calls.last.request.content)
    assert body["top_n"] == 2
    assert [r.url for r in out] == ["https://x.test/1", "https://x.test/0"]


@respx.mock
async def test_request_timeout_reaches_httpx():
    # The dedicated short timeout must actually override the client's shared
    # timeout for this one call — otherwise a hung api.jina.ai would stall
    # every web_search for the full shared timeout before falling back.
    route = respx.post(RERANK_ENDPOINT).mock(return_value=_response([0, 1]))
    reranker = JinaReranker("k", proxy=None)
    # The client timeout is deliberately NON-default: httpx's default (5.0s)
    # happens to equal RERANK_TIMEOUT_SECONDS, so a default client would make
    # this assertion pass even with the per-request override removed.
    async with httpx.AsyncClient(timeout=30.0) as client:
        await reranker.rerank(client, "q", _results(2), top_n=2)
    assert route.calls.last.request.extensions["timeout"] == {
        "connect": RERANK_TIMEOUT_SECONDS,
        "read": RERANK_TIMEOUT_SECONDS,
        "write": RERANK_TIMEOUT_SECONDS,
        "pool": RERANK_TIMEOUT_SECONDS,
    }


@respx.mock
async def test_invalid_json_raises_provider_error():
    respx.post(RERANK_ENDPOINT).mock(return_value=httpx.Response(200, text="nope"))
    reranker = JinaReranker("k", proxy=None)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await reranker.rerank(client, "q", _results(2), top_n=2)


@respx.mock
async def test_retries_zero_means_a_single_attempt():
    # The hardcoded retries=0 is deliberate: the rerank is a serial step, so a
    # retry would delay the whole web_search while the fallback (original
    # order) is fine.
    route = respx.post(RERANK_ENDPOINT).mock(return_value=httpx.Response(500))
    reranker = JinaReranker("k", proxy=None)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await reranker.rerank(client, "q", _results(2), top_n=2)
    assert route.call_count == 1
