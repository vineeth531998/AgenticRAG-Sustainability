"""End-to-end retrieval for one subquery: embed → parallel-union hybrid search → rerank.

Design notes:

1. The planner's `must_phrases` used to be a HARD Qdrant `MatchText` filter —
   chunks that didn't contain those terms never entered the pool. We changed
   this. Two things broke that model:

   (a) Qdrant's WORD tokenizer + MatchText fails on multi-word phrases: a
       filter like `MatchText("Independent Directors")` returned 0 hits
       across an entire 77-page governance report where those tokens
       clearly co-occurred. Probe-verified. Suspected cause is
       phrase-adjacency semantics of the tokenizer combined with our
       index settings, but the fix is more valuable than the diagnosis.

   (b) Even when strict matched something, the sequential cascade
       (strict → relaxed if empty) short-circuited on strict-succeeded-but-
       wrong-chunks — the answer chunk never reached the reranker because
       we thought strict "worked".

2. `query_type` reshapes the leg sizes:
     - narrative: balanced dense+sparse pool
     - factual_lookup: bias to sparse (keyword) leg
     - comparison: same as narrative but the orchestrator runs sufficiency
       differently (expects both sides of the comparison to be hit)

   Note: `is_table_only` was removed from RetrievalFilter entirely. Under
   the three-wing architecture the composite wing reads non-table chunks in
   parallel with the table wing, so restricting retrieval to tables only
   starves the composite wing without any upside.

PARALLEL-UNION CASCADE (replaces the old sequential fallback)
═════════════════════════════════════════════════════════════
For every subquery we now fire TWO Qdrant queries concurrently:

   strict   — planner's full filters + must_phrases (what the planner intended)
   relaxed  — drop everything the planner guessed; keep only report_ids

Both queries run in parallel via `asyncio.gather`, then we UNION their
results (dedupe by chunk_id, keep the higher score per chunk). The union
goes through the client-side `must_phrases` post-filter and then to the
reranker.

Why this is strictly better than sequential:
  • Strict's precise hits still bubble up when the filter DOES work
  • Relaxed guarantees the answer chunk reaches the reranker even when the
    strict filter is over-restrictive or under-recalls (e.g. broken
    MatchText on multi-word phrases)
  • The `if strict returned anything, skip relaxed` heuristic is gone —
    no more silent drops when strict happens to return non-answer chunks
  • Wall clock is the same as one round-trip because both queries run
    concurrently on Qdrant

`report_ids` is never dropped in either query — it's the user-selected
document scope, not a planner guess.
"""
from __future__ import annotations

import asyncio
import re

from agentic_rag.embeddings.dense import embed_dense_one
from agentic_rag.retrieval.bm25 import to_sparse
from agentic_rag.retrieval.reranker import rerank
from agentic_rag.schemas import RetrievalFilter, RetrievedChunk, Subquery
from agentic_rag.vectordb.qdrant import QdrantStore
from config.settings import settings

# Minimum candidates we're willing to send to the reranker after post-filter.
# Below this we accept the un-filtered pool — better to let the reranker sort
# noise than to feed it 1–2 chunks that happened to be exact-string matches.
POST_FILTER_MIN_KEEP = 3

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _text_contains_all_phrases(text: str, must_phrases: list[str]) -> bool:
    """Same semantics as Qdrant's MatchText: all tokens of every phrase
    must appear in the text, case-insensitive, no stemming."""
    if not must_phrases:
        return True
    haystack = set(_tokens(text))
    for phrase in must_phrases:
        needles = _tokens(phrase)
        if not needles:
            continue
        if not all(tok in haystack for tok in needles):
            return False
    return True


def _post_filter_by_must_phrases(
    chunks: list[RetrievedChunk], must_phrases: list[str]
) -> list[RetrievedChunk]:
    if not must_phrases:
        return chunks
    return [
        c for c in chunks
        if _text_contains_all_phrases(c.chunk.text, must_phrases)
    ]


def _leg_sizes(sub: Subquery) -> tuple[int, int]:
    if sub.query_type == "factual_lookup":
        # Keyword-first: bigger sparse pool, smaller dense pool.
        return (settings.top_k_dense // 2, settings.top_k_sparse * 2)
    return (settings.top_k_dense, settings.top_k_sparse)


def _fallback_stages(sub: Subquery) -> list[tuple[str, RetrievalFilter, list[str]]]:
    """Return the (label, filters, must_phrases) tiers.

    Only referenced by `scripts/probe_retrieval.py` now — production uses
    the parallel-union path in `retrieve_for_subquery`. The probe still
    wants to inspect each stage independently for diagnostic output.

    Two stages: strict (planner intent) and relaxed (report_ids only).
    """
    strict_filters = sub.filters
    relaxed_filters = RetrievalFilter()  # every field defaults to None / False
    return [
        ("strict", strict_filters, list(sub.must_phrases)),
        ("relaxed", relaxed_filters, []),
    ]


def _union_by_chunk_id(
    *pools: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Merge multiple retrieval pools, keeping the highest-score chunk per id.

    Order in the returned list is by descending score so downstream
    heuristics (post-filter keep-order, terminal log ordering) reflect
    relevance rather than pool origin.
    """
    by_id: dict[str, RetrievedChunk] = {}
    for pool in pools:
        for c in pool:
            existing = by_id.get(c.chunk.chunk_id)
            if existing is None or c.score > existing.score:
                by_id[c.chunk.chunk_id] = c
    return sorted(by_id.values(), key=lambda c: c.score, reverse=True)


async def retrieve_for_subquery(
    sub: Subquery,
    *,
    report_ids: list[str],
    store: QdrantStore,
) -> list[RetrievedChunk]:
    """Run one subquery through the full retrieval pipeline (parallel-union).

    Pipeline shape:
        1. Embed dense (HyDE doc if provided, else raw query)
        2. Embed sparse (query + soft keywords)
        3. Run STRICT and RELAXED Qdrant queries in PARALLEL
        4. Union results by chunk_id, keep highest score per chunk
        5. Optional client-side post-filter by must_phrases (when it
           doesn't over-cull the pool)
        6. Rerank the union, return top-K
    """
    # Dense: Fireworks qwen3-embedding-8b (HyDE doc if present, else raw query).
    # Sparse: BM25 over the query plus the soft `keywords` — exact terms the
    # planner wants weighted without making them must-have.
    dense_text = sub.hyde_doc or sub.query
    sparse_text = (
        sub.query + " " + " ".join(sub.keywords) if sub.keywords else sub.query
    )

    dense_vec = await embed_dense_one(dense_text)
    sparse_vec = to_sparse(sparse_text)

    limit_dense, limit_sparse = _leg_sizes(sub)
    limit_final = max(limit_dense, limit_sparse)

    # ── Parallel-union: fire STRICT + RELAXED concurrently. ────────────────
    # Strict uses the planner's full filters + must_phrases. Relaxed drops
    # every guess and keeps only the user's report scope. Both are bounded
    # by `limit_final` so the union has at most ~2×limit_final unique chunks
    # (usually much less after dedup because both queries hit similar top
    # results). Wall clock is one Qdrant round-trip because they run
    # concurrently.
    strict_filters = sub.filters
    relaxed_filters = RetrievalFilter()

    strict_task = store.hybrid_search(
        dense_query=dense_vec,
        sparse_query=sparse_vec,
        report_ids=report_ids,
        filters=strict_filters,
        must_phrases=list(sub.must_phrases),
        limit_dense=limit_dense,
        limit_sparse=limit_sparse,
        limit_final=limit_final,
    )
    relaxed_task = store.hybrid_search(
        dense_query=dense_vec,
        sparse_query=sparse_vec,
        report_ids=report_ids,
        filters=relaxed_filters,
        must_phrases=[],
        limit_dense=limit_dense,
        limit_sparse=limit_sparse,
        limit_final=limit_final,
    )
    strict_pool, relaxed_pool = await asyncio.gather(strict_task, relaxed_task)
    raw = _union_by_chunk_id(strict_pool, relaxed_pool)

    # Retrieval debug log — visible in the Streamlit terminal. The split
    # into strict/relaxed/union tells you exactly whether the planner's
    # hard filter was doing real work (strict > 0) or just contributing
    # noise/nothing (strict = 0 → union == relaxed).
    print(
        f"[retrieval] strict={len(strict_pool)} relaxed={len(relaxed_pool)} "
        f"union={len(raw)} | query={sub.query[:80]!r} | "
        f"must={sub.must_phrases} | "
        f"filters={sub.filters.model_dump(exclude_none=True)}",
        flush=True,
    )

    if not raw:
        return []

    # Post-filter: re-apply must_phrases on the union before the reranker
    # sees it. This tightens precision — the union always contains the
    # relaxed pool (no must_phrase gate at Qdrant), so if the planner had
    # meaningful must_phrases we want to prefer chunks that actually
    # contain them. If the filter would be too aggressive (would leave
    # < POST_FILTER_MIN_KEEP chunks), keep the broader pool so the
    # reranker has enough to sort.
    if sub.must_phrases:
        filtered = _post_filter_by_must_phrases(raw, sub.must_phrases)
        if len(filtered) >= POST_FILTER_MIN_KEEP:
            print(
                f"[retrieval] post-filter kept {len(filtered)}/{len(raw)} "
                f"chunks matching must={sub.must_phrases}",
                flush=True,
            )
            raw = filtered
        elif filtered:
            print(
                f"[retrieval] post-filter would keep only "
                f"{len(filtered)}/{len(raw)}; too aggressive — "
                f"passing full pool to reranker",
                flush=True,
            )

    return await rerank(sub.query, raw, top_k=settings.top_k_rerank)
