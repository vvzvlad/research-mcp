"""Jina reranker: reorder the merged search results by relevance to the query.

Not a pipeline provider — it produces no results of its own, it transforms the
list the search providers already brought — so it lives outside
``src/providers/`` and is wired into ``Pipeline`` directly (see
``Pipeline.build`` / ``Pipeline.search``).

API (verified 2026-08-18 from https://api.jina.ai/openapi.json): POST
``https://api.jina.ai/v1/rerank`` with ``Authorization: Bearer <key>`` and JSON
body ``{"model", "query", "documents": [<str>, ...], "top_n",
"return_documents": false}`` — the ``RerankerV3Request`` schema, whose model
enum is ``jina-reranker-v3`` / ``jina-reranker-v3.5``. Response: ``{"model",
"object": "list", "usage": {...}, "results": [{"index": <int, position in the
ORIGINAL documents list>, "relevance_score": <float>}, ...]}``, sorted by
relevance descending.

Billing counts INPUT tokens. Our documents are title+snippet strings (or the
URL when both are empty), each capped at 1000 chars; a TYPICAL call costs
~1-3k input tokens, and even the worst case — a few hundred capped documents —
stays at fractions of a cent, which is why the rerank can afford to run on
every web_search by default.
"""

from __future__ import annotations

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderError, SearchResult

RERANK_ENDPOINT = "https://api.jina.ai/v1/rerank"
RERANK_MODEL = "jina-reranker-v3.5"

# Own short timeout instead of the shared 25s request timeout: the rerank is an
# optional step with a graceful fallback (the original merge order), but it is
# a SERIAL await inside every web_search — a hung api.jina.ai would stall each
# search for the full shared timeout before falling back. 5s is plenty for a
# ~1-3k-token rerank call and caps the worst-case stall.
RERANK_TIMEOUT_SECONDS = 5.0


class JinaReranker:
    """Rerank a merged result list via the Jina Reranker API."""

    def __init__(self, api_key: str, proxy: str | None) -> None:
        self.name = "jina-rerank"
        self.proxy = proxy
        self._api_key = api_key

    async def rerank(
        self,
        client: httpx.AsyncClient,
        query: str,
        results: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        """Return ``results`` reordered by relevance, at most ``top_n`` of them.

        Reorders, never filters: a partial ranking is padded with the
        unmentioned results in their original order before the ``top_n`` slice.
        Raises ``ProviderError`` on HTTP failure, a malformed response, or an
        empty ranking for a non-empty input; the caller (``Pipeline.search``)
        falls back to the original merge order then. The caller guarantees a
        non-trivial input — the pipeline reranks only for ``len(merged) > 1``.
        """
        # A hit with neither title nor snippet must not become an empty
        # document string — the API may reject the whole batch with a 422 over
        # one blank entry. The URL stands in then: it is a useful relevance
        # signal in its own right (host and path words often match the query).
        # Each document is capped at 1000 chars: snippets are normally short,
        # but jina_search falls back to an item's "content", which in
        # reader-style responses can be a whole page — an unbounded document
        # would silently multiply the rerank's input-token cost.
        documents = [((f"{r.title}\n{r.snippet}").strip() or r.url)[:1000] for r in results]
        response = await request_with_retry(
            client,
            "POST",
            RERANK_ENDPOINT,
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": documents,
                # jina's docs default top_n to len(documents) and probably
                # clamp server-side, but a version that validates
                # top_n <= len(documents) would 422 exactly on thin result
                # sets; the client-side min is free.
                "top_n": min(top_n, len(results)),
                # Indices are enough: the SearchResult objects stay local, so
                # echoing the documents back would only waste bandwidth.
                "return_documents": False,
            },
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            # retries=0 ON PURPOSE (not settings.retries): the rerank is a
            # serial await after the concurrent provider gather, so every retry
            # (backoff + another round trip) delays the whole web_search — and
            # the fallback on failure (the original merge order) is perfectly
            # graceful. Giving up fast beats stalling the search for a nicer
            # ordering.
            retries=0,
            provider=self.name,
            # Forwarded to httpx: overrides the client's shared timeout for
            # this one call — see RERANK_TIMEOUT_SECONDS.
            timeout=RERANK_TIMEOUT_SECONDS,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        ranked = data.get("results") if isinstance(data, dict) else None
        if not isinstance(ranked, list):
            raise ProviderError(f"{self.name}: malformed response (no results list)")
        # The API already applies top_n and sorts by relevance, but validate
        # and slice anyway: `index` points into OUR list, so a bad value here
        # would silently reorder the SERP into garbage. An out-of-range or
        # non-int index is a malformed answer (raise, never guess); a duplicate
        # index is dropped.
        out: list[SearchResult] = []
        seen: set[int] = set()
        for item in ranked:
            if not isinstance(item, dict):
                raise ProviderError(f"{self.name}: malformed response item")
            index = item.get("index")
            # bool is excluded explicitly: it subclasses int, and a JSON `true`
            # here would otherwise pass as index 1.
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(results)
            ):
                raise ProviderError(f"{self.name}: bad result index {index!r}")
            if index in seen:
                continue
            seen.add(index)
            out.append(results[index])
        # A syntactically valid `{"results": []}` for a non-empty input is an
        # anomaly, not an answer: returning [] here would let the caller wipe
        # every found result while logging reranked=true. Raise instead — the
        # caller then keeps the merge order and logs reranked=false.
        if not out and results:
            raise ProviderError(f"{self.name}: empty ranking for {len(results)} documents")
        # Rerank must reorder, never filter: a partial ranking (fewer indices
        # than documents) would otherwise silently drop the unmentioned
        # results. Pad with them, in their original merge order, after the
        # ranked ones — the top_n slice below still applies.
        if len(out) < len(results):
            out.extend(r for i, r in enumerate(results) if i not in seen)
        return out[:top_n]
