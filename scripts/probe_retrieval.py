"""Trace a query through the retrieval pipeline and find where the answer chunk drops out.

Point it at a query that FAILED (final answer said "not available" but you
know the answer is in the corpus) and a target substring that only appears
in the correct chunk (e.g. "D. Sundaram" for the Infosys directors question).
The probe reports, at EACH pipeline stage, whether the target chunk is still
in the pool — and if not, which stage killed it.

Every stage the retrieval pipeline runs is instrumented:

  1. Existence check      — is the target chunk even in Qdrant for this report?
  2. Planner              — what subquery / must_phrases / filters / keywords?
  3. Dense-only prefetch  — did the answer chunk make dense top-K?
  4. Sparse-only prefetch — did it make sparse top-K?
  5. Fused RRF            — did the fusion keep it?
  6. Cascade fallback     — did strict stage fire, or did we fall back to relaxed?
  7. Post-filter          — did must_phrases (client-side) drop it?
  8. Rerank               — did the reranker keep it in top-K?

Usage:

    uv run python scripts/probe_retrieval.py \\
        --query "Name of all independent directors in the company" \\
        --report-id InfosysTest \\
        --target-substring "D. Sundaram"

Optional:
    --skip-planner           Skip planner; use the raw query with no must_phrases
                             (isolates whether planner's decisions are the cause)
    --must-phrases "a,b,c"   Override planner and inject specific must_phrases
    --no-hyde                Force HyDE off (settings default is off anyway)
    --dump-chunks N          Dump full text of top-N chunks at each stage
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Path shim so this runs with `uv run python scripts/probe_retrieval.py`
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from qdrant_client import models  # noqa: E402

from agentic_rag.agents.planner import plan  # noqa: E402
from agentic_rag.embeddings.dense import embed_dense_one  # noqa: E402
from agentic_rag.retrieval.bm25 import to_sparse  # noqa: E402
from agentic_rag.retrieval.hybrid import (  # noqa: E402
    POST_FILTER_MIN_KEEP,
    _fallback_stages,
    _leg_sizes,
    _text_contains_all_phrases,
)
from agentic_rag.retrieval.reranker import rerank  # noqa: E402
from agentic_rag.schemas import (  # noqa: E402
    PlannerOutput,
    RetrievalFilter,
    RetrievedChunk,
    Subquery,
)
from agentic_rag.vectordb.qdrant import (  # noqa: E402
    QdrantStore,
    _build_filter,
    _point_to_retrieved,
)
from config.settings import settings  # noqa: E402


# ── Small helpers ───────────────────────────────────────────────────────────

def _hit_marker(rank: int | None, score: float | None) -> str:
    if rank is None:
        return "MISSED"
    return f"FOUND @ rank #{rank + 1}  score={score:.4f}"


def _find_target(
    chunks: list[RetrievedChunk], target_substring: str
) -> tuple[int | None, float | None]:
    """Return (rank_zero_based, score) of the FIRST chunk whose text contains
    target_substring. Case-insensitive. Returns (None, None) if not present.
    """
    tgt = target_substring.lower()
    for i, c in enumerate(chunks):
        if tgt in c.chunk.text.lower():
            return i, c.score
    return None, None


def _short(text: str, n: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ── Direct Qdrant queries for diagnostic isolation ──────────────────────────

async def _dense_only(
    store: QdrantStore,
    *,
    dense_vec: list[float],
    q_filter: models.Filter | None,
    limit: int,
) -> list[RetrievedChunk]:
    """Dense leg only, no RRF, no sparse. Same filter as production."""
    resp = await store.client.query_points(
        collection_name=store.collection,
        query=dense_vec,
        using=settings.dense_vector_name,
        limit=limit,
        query_filter=q_filter,
        with_payload=True,
    )
    return [_point_to_retrieved(p) for p in resp.points]


async def _sparse_only(
    store: QdrantStore,
    *,
    sparse_vec: dict[int, float],
    q_filter: models.Filter | None,
    limit: int,
) -> list[RetrievedChunk]:
    """Sparse leg only, no RRF, no dense. Same filter as production."""
    resp = await store.client.query_points(
        collection_name=store.collection,
        query=models.SparseVector(
            indices=list(sparse_vec.keys()),
            values=list(sparse_vec.values()),
        ),
        using=settings.sparse_vector_name,
        limit=limit,
        query_filter=q_filter,
        with_payload=True,
    )
    return [_point_to_retrieved(p) for p in resp.points]


async def _fused(
    store: QdrantStore,
    *,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    q_filter: models.Filter | None,
    limit_dense: int,
    limit_sparse: int,
    limit_final: int,
) -> list[RetrievedChunk]:
    """The production hybrid — dense + sparse prefetch, RRF fusion."""
    prefetch = [
        models.Prefetch(
            query=dense_vec,
            using=settings.dense_vector_name,
            limit=limit_dense,
            filter=q_filter,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=list(sparse_vec.keys()),
                values=list(sparse_vec.values()),
            ),
            using=settings.sparse_vector_name,
            limit=limit_sparse,
            filter=q_filter,
        ),
    ]
    resp = await store.client.query_points(
        collection_name=store.collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit_final,
        with_payload=True,
        query_filter=q_filter,
    )
    return [_point_to_retrieved(p) for p in resp.points]


# ── Existence check: does the target chunk exist in the index at all? ──────

async def _find_target_in_index(
    store: QdrantStore, report_id: str, target_substring: str, max_scan: int = 5000
) -> list[dict[str, Any]]:
    """Scroll through all chunks for the report and return every one whose
    text contains the target substring. Case-insensitive.

    This is expensive (touches every chunk of the report) but it's the ground
    truth: if this returns nothing, the target isn't in the index and no
    downstream retrieval fix can help.
    """
    matches: list[dict[str, Any]] = []
    tgt = target_substring.lower()
    q_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.report_id",
                match=models.MatchValue(value=report_id),
            )
        ]
    )
    offset: Any = None
    scanned = 0
    while scanned < max_scan:
        resp, next_offset = await store.client.scroll(
            collection_name=store.collection,
            scroll_filter=q_filter,
            limit=200,
            offset=offset,
            with_payload=True,
        )
        for point in resp:
            payload = point.payload or {}
            text = payload.get("text", "")
            if tgt in text.lower():
                matches.append({
                    "chunk_id": payload.get("chunk_id"),
                    "page": payload.get("metadata", {}).get("page"),
                    "section": payload.get("metadata", {}).get("section"),
                    "is_table": payload.get("metadata", {}).get("is_table"),
                    "label": payload.get("metadata", {}).get("label"),
                    "text_preview": _short(text, 240),
                })
        scanned += len(resp)
        if next_offset is None:
            break
        offset = next_offset
    return matches


# ── Trace one subquery through all stages ──────────────────────────────────

@dataclass
class StageResult:
    stage: str
    filter_desc: str
    pool: list[RetrievedChunk]
    dense_only: list[RetrievedChunk]
    sparse_only: list[RetrievedChunk]


async def _trace_subquery(
    store: QdrantStore,
    sub: Subquery,
    report_ids: list[str],
    target_substring: str,
    dump_chunks: int,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], str]:
    """Trace one subquery. Returns (final_reranked, post_filter_input, verdict)."""

    # 1) Build the vectors — same as production.
    dense_text = sub.hyde_doc or sub.query
    sparse_text = (
        sub.query + " " + " ".join(sub.keywords) if sub.keywords else sub.query
    )
    print(f"\n[probe]   dense query text  : {_short(dense_text, 140)!r}")
    print(f"[probe]   sparse query text : {_short(sparse_text, 140)!r}")

    dense_vec = await embed_dense_one(dense_text)
    sparse_vec = to_sparse(sparse_text)
    limit_dense, limit_sparse = _leg_sizes(sub)
    limit_final = max(limit_dense, limit_sparse)
    print(
        f"[probe]   leg sizes         : dense={limit_dense} "
        f"sparse={limit_sparse} fused_final={limit_final}"
    )

    # 2) Run each fallback stage until one returns results (matching production).
    stages_run: list[StageResult] = []
    chosen: StageResult | None = None
    for label, filters, must_phrases in _fallback_stages(sub):
        q_filter = _build_filter(report_ids, filters, must_phrases)
        filter_desc = (
            f"report_ids={report_ids}  "
            f"must_phrases={must_phrases}  "
            f"filters={filters.model_dump(exclude_none=True)}"
        )
        print(f"\n[probe] ── STAGE: {label.upper()} ──")
        print(f"[probe]   filter: {filter_desc}")

        # Dense-only leg (diagnostic)
        dense_only = await _dense_only(
            store, dense_vec=dense_vec, q_filter=q_filter, limit=limit_dense
        )
        d_rank, d_score = _find_target(dense_only, target_substring)
        print(
            f"[probe]   dense-only top-{limit_dense}  hits={len(dense_only)}  "
            f"target: {_hit_marker(d_rank, d_score)}"
        )

        # Sparse-only leg (diagnostic)
        sparse_only = await _sparse_only(
            store, sparse_vec=sparse_vec, q_filter=q_filter, limit=limit_sparse
        )
        s_rank, s_score = _find_target(sparse_only, target_substring)
        print(
            f"[probe]   sparse-only top-{limit_sparse} hits={len(sparse_only)}  "
            f"target: {_hit_marker(s_rank, s_score)}"
        )

        # Fused RRF (production)
        pool = await _fused(
            store,
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            q_filter=q_filter,
            limit_dense=limit_dense,
            limit_sparse=limit_sparse,
            limit_final=limit_final,
        )
        f_rank, f_score = _find_target(pool, target_substring)
        print(
            f"[probe]   fused (RRF)      hits={len(pool)}          "
            f"target: {_hit_marker(f_rank, f_score)}"
        )

        stages_run.append(StageResult(
            stage=label,
            filter_desc=filter_desc,
            pool=pool,
            dense_only=dense_only,
            sparse_only=sparse_only,
        ))
        if pool:
            chosen = stages_run[-1]
            print(f"[probe]   ↳ this stage returned results — cascade stops here")
            break
        else:
            print(f"[probe]   ↳ empty pool — cascade falls through to next stage")

    if chosen is None:
        return [], [], "ALL STAGES RETURNED EMPTY (target never entered any pool)"

    # 3) Post-filter (client-side) — production only applies if must_phrases set.
    print(f"\n[probe] ── POST-FILTER (client-side must_phrases) ──")
    pool = chosen.pool
    if not sub.must_phrases:
        print(f"[probe]   skipped: subquery has no must_phrases")
        post_filter_input = pool
    else:
        filtered = [
            c for c in pool
            if _text_contains_all_phrases(c.chunk.text, sub.must_phrases)
        ]
        pf_rank, pf_score = _find_target(filtered, target_substring)
        print(
            f"[probe]   would-keep       hits={len(filtered)}/{len(pool)}  "
            f"must={sub.must_phrases}"
        )
        print(f"[probe]   target after post-filter: {_hit_marker(pf_rank, pf_score)}")
        if len(filtered) >= POST_FILTER_MIN_KEEP:
            print(f"[probe]   decision: KEEP filtered ({len(filtered)} ≥ min_keep={POST_FILTER_MIN_KEEP})")
            pool = filtered
        else:
            print(
                f"[probe]   decision: KEEP full pool (filtered would leave only "
                f"{len(filtered)} < min_keep={POST_FILTER_MIN_KEEP})"
            )
        post_filter_input = pool

    # 4) Rerank
    print(f"\n[probe] ── RERANK ──")
    print(f"[probe]   input pool: {len(pool)} chunks  (reranker top_n={settings.top_k_rerank})")
    rerank_input_rank, _ = _find_target(pool, target_substring)
    if rerank_input_rank is None:
        return [], post_filter_input, (
            "TARGET DROPPED BEFORE RERANK — never entered the reranker's input pool"
        )
    print(f"[probe]   target in input pool at rank #{rerank_input_rank + 1}")

    reranked = await rerank(sub.query, pool, top_k=settings.top_k_rerank)
    r_rank, r_score = _find_target(reranked, target_substring)
    print(
        f"[probe]   reranker output top-{len(reranked)}  "
        f"target: {_hit_marker(r_rank, r_score)}"
    )

    if dump_chunks > 0:
        print(f"\n[probe] ── TOP-{dump_chunks} RERANKED CHUNKS ──")
        for i, c in enumerate(reranked[:dump_chunks]):
            m = c.chunk.metadata
            is_target = target_substring.lower() in c.chunk.text.lower()
            marker = "★ TARGET" if is_target else "       "
            print(
                f"[probe]   {marker}  #{i + 1}  score={c.score:.4f}  "
                f"chunk_id={c.chunk.chunk_id}  page={m.page}  "
                f"section={(m.section or '-')[:60]!r}  label={m.label}"
            )
            print(f"[probe]              text: {_short(c.chunk.text, 220)!r}")

    verdict = (
        "TARGET IN FINAL TOP-K RERANK" if r_rank is not None
        else "TARGET FELL OFF DURING RERANK (was in input pool, not in output)"
    )
    return reranked, post_filter_input, verdict


# ── Main ────────────────────────────────────────────────────────────────────

async def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", required=True, help="The user query to trace")
    ap.add_argument("--report-id", required=True, help="Which report_id to search")
    ap.add_argument(
        "--target-substring", required=True,
        help="A substring that appears ONLY in the correct answer chunk "
             "(e.g. a director name, a specific numeric value with unit)",
    )
    ap.add_argument(
        "--skip-planner", action="store_true",
        help="Skip the planner; use raw query + no must_phrases + no keywords "
             "(isolates whether planner decisions are the cause)",
    )
    ap.add_argument(
        "--must-phrases", type=str, default=None,
        help="Override planner's must_phrases with a comma-separated list "
             "(e.g. 'Independent Directors,D. Sundaram')",
    )
    ap.add_argument(
        "--no-hyde", action="store_true",
        help="Force HyDE off (settings default is off anyway)",
    )
    ap.add_argument(
        "--dump-chunks", type=int, default=5,
        help="Dump full text of top-N reranked chunks. Default 5.",
    )
    args = ap.parse_args()

    print("=" * 88)
    print(f"[probe] query           : {args.query!r}")
    print(f"[probe] report_id       : {args.report_id!r}")
    print(f"[probe] target_substring: {args.target_substring!r}")
    print("=" * 88)

    store = QdrantStore()

    # ── Stage 0: does the target even exist in the index? ─────────────────
    print(f"\n[probe] ── STAGE 0: existence check ──")
    print(
        f"[probe] scanning all chunks for report_id={args.report_id!r} "
        f"looking for substring {args.target_substring!r}..."
    )
    matches = await _find_target_in_index(
        store, args.report_id, args.target_substring
    )
    if not matches:
        print(f"[probe] ❌ TARGET NOT IN INDEX")
        print(
            f"[probe]    No chunk under report_id={args.report_id!r} contains "
            f"{args.target_substring!r}. Either:\n"
            f"[probe]    (a) the report wasn't ingested\n"
            f"[probe]    (b) Chandra emitted this content under a different "
            f"substring — check the raw JSON via probe_chandra.py\n"
            f"[probe]    (c) the substring you passed has different casing "
            f"or punctuation than the index copy — try a shorter, less "
            f"punctuated substring.\n"
        )
        return
    print(f"[probe] ✅ FOUND {len(matches)} matching chunk(s) in the index:")
    for i, m in enumerate(matches):
        print(
            f"[probe]    match #{i + 1}: chunk_id={m['chunk_id']}  "
            f"page={m['page']}  section={(m['section'] or '-')[:60]!r}  "
            f"is_table={m['is_table']}  label={m['label']}"
        )
        print(f"[probe]              preview: {m['text_preview']!r}")

    # ── Stage 1: planner (or synthetic subquery) ──────────────────────────
    print(f"\n[probe] ── STAGE 1: planner ──")
    if args.skip_planner:
        print(f"[probe] --skip-planner set; synthesizing a raw subquery")
        must_phrases = (
            [p.strip() for p in args.must_phrases.split(",") if p.strip()]
            if args.must_phrases else []
        )
        subs = [Subquery(
            query=args.query,
            must_phrases=must_phrases,
            keywords=[],
            hyde_doc=None,
            filters=RetrievalFilter(),
            query_type="narrative",
            target_cells=[],
            rationale="synthesized by probe_retrieval --skip-planner",
        )]
        plan_result = PlannerOutput(
            subqueries=subs,
            reasoning="[skip-planner mode] no planner run",
        )
    else:
        plan_result = await plan(
            args.query,
            enable_hyde=False if args.no_hyde else None,
        )
        subs = list(plan_result.subqueries)
        if args.must_phrases is not None:
            override = [
                p.strip() for p in args.must_phrases.split(",") if p.strip()
            ]
            print(
                f"[probe] overriding planner must_phrases with {override!r}"
            )
            subs = [s.model_copy(update={"must_phrases": override}) for s in subs]

    print(f"[probe] planner returned {len(subs)} subquery(ies):")
    for i, s in enumerate(subs):
        print(
            f"[probe]   SQ{i + 1}  type={s.query_type}  "
            f"must_phrases={s.must_phrases}  "
            f"filters={s.filters.model_dump(exclude_none=True)}"
        )
        print(f"[probe]         query: {_short(s.query, 140)!r}")
        if s.keywords:
            print(f"[probe]         keywords: {s.keywords}")
        if s.hyde_doc:
            print(f"[probe]         hyde: {_short(s.hyde_doc, 140)!r}")

    # ── Stage 2..N: trace each subquery ───────────────────────────────────
    for i, s in enumerate(subs):
        print("\n" + "=" * 88)
        print(f"[probe] TRACING SUBQUERY {i + 1} / {len(subs)}")
        print("=" * 88)
        reranked, _, verdict = await _trace_subquery(
            store, s, [args.report_id], args.target_substring, args.dump_chunks
        )
        print(f"\n[probe] SUBQUERY {i + 1} VERDICT: {verdict}")

    print("\n" + "=" * 88)
    print("[probe] DONE")
    print("=" * 88)


if __name__ == "__main__":
    asyncio.run(_main())
