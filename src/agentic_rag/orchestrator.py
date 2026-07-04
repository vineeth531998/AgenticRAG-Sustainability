"""The agentic loop:

    user_query
        │
        ▼
    Planner ──> subqueries[]  (each tagged narrative | factual_lookup | comparison)
        │
        ▼
    for each subquery: keyword-gated hybrid retrieve → rerank
        │
        ▼
    for each factual_lookup subquery: TableExtractor on its table chunks ─► TableValue[]
        │
        ▼
    Critic ──> sufficient?  ─── no ──> follow-up subqueries (loop, capped)
        │
        yes
        ▼
    Synthesizer (chunks + accumulated TableValues) ──> cited answer

The audit trail (planner output, every subquery including follow-ups, every
critic decision, every extracted TableValue) is exposed in `QueryResult` —
mandatory for sustainability/compliance reuse of the answer.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from agentic_rag.agents.composite_extractor import extract_composite_values
from agentic_rag.agents.composite_vlm_verifier import (
    verify_and_merge_composite_values,
    vlm_extract_unfound_composite_targets,
)
from agentic_rag.agents.critic import critique
from agentic_rag.agents.planner import plan
from agentic_rag.agents.synthesizer import synthesize
from agentic_rag.agents.table_extractor import extract_table_values
from agentic_rag.agents.table_vlm_verifier import (
    verify_and_merge_table_values,
    vlm_extract_unfound_targets,
)
from agentic_rag.ingestion.metadata import load_metadata
from agentic_rag.retrieval.hybrid import retrieve_for_subquery
from agentic_rag.schemas import (
    CriticOutput,
    PlannerOutput,
    ReportMetadata,
    RetrievedChunk,
    Subquery,
    SynthesizerOutput,
    TableValue,
    UnfoundTargetContext,
    UnfoundVLMEvidence,
)
from agentic_rag.vectordb.qdrant import QdrantStore
from config.settings import settings


@dataclass
class VLMVerificationEvent:
    """One row for the audit log — full VLM response for a single verification.

    Captures every field the VLM returned so we can see WHY it made its call —
    especially critical for `found=False` events where the note explains what
    the VLM actually saw (e.g. "table shows split by gender, no combined
    total"). Without the note in the audit trail, `found=false` looks
    identical whether the VLM correctly detected a segment-vs-combined
    mismatch or whether it just gave up.
    """
    target: str
    stage: str  # "skip" | "start" | "done" | "error" | *_not_found variants
    reason: str | None = None            # populated when stage == "skip"
    chunk_id: str | None = None          # source chunk the VLM was reading
    original_value: str | None = None
    original_confidence: str | None = None
    # VLM's own reading — captured whether found=True OR found=False
    vlm_found: bool | None = None
    vlm_value: str | None = None
    vlm_confidence: str | None = None
    vlm_row_label: str | None = None
    vlm_column_label: str | None = None
    vlm_unit: str | None = None
    vlm_note: str | None = None          # ← the "why" for found=False
    # Merged output after applying the R1..R7 merge rules
    merged_value: str | None = None
    merged_confidence: str | None = None
    merged_note: str | None = None
    error: str | None = None


@dataclass
class QueryTrace:
    planner: PlannerOutput
    all_subqueries: list[Subquery] = field(default_factory=list)
    critic_decisions: list[CriticOutput] = field(default_factory=list)
    iterations: int = 0
    unfound_targets: list[str] = field(default_factory=list)
    vlm_verifications: list[VLMVerificationEvent] = field(default_factory=list)
    # Subqueries the sanitizer removed before execution — from ANY stage
    # (planner + every critic iteration). Each entry has a "stage" field
    # ("planner" or e.g. "critic_iter_2"). Kept for the audit log so we
    # can see what got leaked and where.
    sanitized_subqueries: list[dict[str, str]] = field(default_factory=list)


@dataclass
class QueryResult:
    answer: SynthesizerOutput
    chunks_used: list[RetrievedChunk]
    table_values: list[TableValue]
    trace: QueryTrace


EventCallback = Callable[[str, Any], None]


async def answer_query(
    user_query: str,
    report_ids: list[str],
    *,
    store: QdrantStore | None = None,
    on_event: EventCallback | None = None,
) -> QueryResult:
    """Run the agentic loop.

    `on_event(stage, data)` is invoked at each pipeline boundary so a UI can
    show live progress. A failing callback never breaks the pipeline. Stages
    in order:
        planner_start           data: {"query": str, "report_ids": [str]}
        planner_done            data: PlannerOutput
        iteration_start         data: int (1-indexed iteration number)
        retrieval_start         data: list[Subquery]
        retrieval_done          data: {"subqueries": [...], "chunks_per_sub": [...]}
        target_cells_autofill      data: {"subquery": Subquery, "synthesized": str}
        table_extraction_start     data: {"subquery": Subquery, "table_count": int}
        table_extraction_done      data: TableExtractorOutput
        table_extraction_skip      data: {"subquery": Subquery, "reason": str}
        composite_extraction_start data: {"subquery": Subquery, "chunk_count": int}
        composite_extraction_done  data: TableExtractorOutput
        composite_extraction_skip  data: {"subquery": Subquery, "reason": str}
        composite_vlm_verify_batch_start / done  data: per-batch dict
        composite_vlm_verify_skip / start / done / error  data: per-value dict
        composite_vlm_extract_batch_start / done  data: per-batch dict
        composite_vlm_extract_start / done / not_found / error  data: per-target dict
        critic_start               data: None
        critic_done             data: CriticOutput
        synthesizer_start       data: None
        synthesizer_done        data: SynthesizerOutput
    """
    if not report_ids:
        raise ValueError("report_ids must be non-empty")

    # We build the trace first so `emit` can also record VLM verification events
    # into `trace.vlm_verifications` — that's what makes them visible in the
    # audit log after the run, not just in the live UI callback.
    trace_ref: dict[str, Any] = {}

    def emit(stage: str, data: Any = None) -> None:
        # 1. Record VLM events into the trace so the audit log picks them up
        #    regardless of whether a UI callback is attached. Four families
        #    of stages flow through the same VLMVerificationEvent list —
        #    the `stage` field on each event distinguishes:
        #      • table_vlm_verify_*      — table verify (skip/start/done/error)
        #      • table_vlm_extract_*     — table extract-from-unfound rescue
        #      • composite_vlm_verify_*  — composite verify (skip/start/done/error)
        #      • composite_vlm_extract_* — composite extract-from-unfound rescue
        if "trace" in trace_ref and (
            stage.startswith("table_vlm_verify_")
            or stage.startswith("table_vlm_extract_")
            or stage.startswith("composite_vlm_verify_")
            or stage.startswith("composite_vlm_extract_")
        ):
            _record_vlm_event(trace_ref["trace"], stage, data)
        # 2. Forward to the caller's callback if any.
        if on_event is None:
            return
        try:
            on_event(stage, data)
        except Exception:  # noqa: BLE001 — never let a UI bug break a query
            pass

    store = store or QdrantStore()

    # 1. Planner (with optional content-aware planning context)
    # Load per-report content distribution metadata (written at ingestion
    # end) for every report the query targets. If a report was ingested
    # before this feature existed, load_metadata returns None and we skip
    # that report — planner degrades to default behavior.
    report_metadatas = [
        m for m in (load_metadata(rid) for rid in report_ids) if m is not None
    ]
    report_context = _format_report_context(report_ids, report_metadatas)

    emit("planner_start", {"query": user_query, "report_ids": list(report_ids)})
    planner_out = await plan(user_query, report_context=report_context)

    # DEFENSIVE SANITIZATION: even with a tightened prompt, some models
    # leak planning-scratchpad subqueries (FINAL_REVIEW, HOWTO_*, self-
    # flagged duplicates) into the executable array. Strip them out BEFORE
    # they burn retrieval + LLM/VLM budget downstream. The same sanitizer
    # runs after every critic iteration too, sharing the `seen_signatures`
    # set so cross-iteration cosmetic reissues are caught (critic tries to
    # rephrase what the planner already ran).
    seen_signatures: set[tuple[str, tuple[str, ...]]] = set()
    kept, filtered = _sanitize_subqueries(
        planner_out.subqueries,
        stage="planner",
        seen_signatures=seen_signatures,
    )
    if filtered:
        print(
            f"[planner-sanitize] dropped {len(filtered)}/"
            f"{len(planner_out.subqueries)} subqueries "
            f"(reasons: {', '.join(sorted({f['reason'] for f in filtered}))})",
            flush=True,
        )
        emit("planner_sanitized",
             {"original_count": len(planner_out.subqueries),
              "kept_count": len(kept),
              "filtered": filtered})
        planner_out.subqueries = kept

    emit("planner_done", planner_out)
    trace = QueryTrace(
        planner=planner_out,
        all_subqueries=list(planner_out.subqueries),
        sanitized_subqueries=list(filtered),
    )
    trace_ref["trace"] = trace

    chunks_by_id: dict[str, RetrievedChunk] = {}
    table_values: list[TableValue] = []
    new_subs = list(planner_out.subqueries)

    while True:
        emit("iteration_start", trace.iterations + 1)

        # 2. Retrieve for the new subqueries (parallel) + merge into the pool
        emit("retrieval_start", list(new_subs))
        per_sub_results = await _retrieve_per_subquery(new_subs, report_ids, store)
        for r in (c for batch in per_sub_results for c in batch):
            existing = chunks_by_id.get(r.chunk.chunk_id)
            if existing is None or r.score > existing.score:
                chunks_by_id[r.chunk.chunk_id] = r
        emit(
            "retrieval_done",
            {"subqueries": list(new_subs), "chunks_per_sub": per_sub_results},
        )

        # 3. Extraction: scoped to ONLY the chunks that came back from THAT
        #    subquery, so noise from other subqueries doesn't pollute the
        #    extraction. Two extraction wings exist:
        #      • TABLE wing     — text extractor (markdown/JSON tables) +
        #                         mandatory VLM verify + VLM extract-from-unfound
        #      • COMPOSITE wing — LLM extractor over prose / list /
        #                         infographic-transcribed chunks (Chandra
        #                         label=Composite, which is where MOST report
        #                         content actually lives)
        #
        #    A third "narrative wing" is implicit: retrieved chunks flow
        #    directly to the synthesizer regardless of extraction. That path
        #    always runs — no code here.
        #
        #    Both extraction wings fire IN PARALLEL for every factual_lookup
        #    subquery. A target is only truly unfound if BOTH wings missed it.
        #    For narrative-typed subqueries, both extraction wings are
        #    skipped and chunks reach the synthesizer via the narrative wing.
        for sub, sub_chunks in zip(new_subs, per_sub_results, strict=True):
            if sub.query_type != "factual_lookup":
                continue

            # SAFETY NET: planner is REQUIRED to fill target_cells for every
            # factual_lookup subquery, but LLMs sometimes ship it empty. Silent
            # skip here → synthesizer re-parses raw chunks and can flip
            # labels. So autofill from the subquery string.
            target_cells = list(sub.target_cells)
            if not target_cells:
                synthesized = sub.query.strip()
                print(
                    f"[orchestrator] WARN planner shipped empty target_cells "
                    f"for factual_lookup subquery {sub.query!r} — auto-synthesizing "
                    f"target from the subquery string so extraction still runs.",
                    flush=True,
                )
                emit(
                    "target_cells_autofill",
                    {"subquery": sub, "synthesized": synthesized},
                )
                target_cells = [synthesized]

            # Split the subquery's chunks by type. is_table → table wing.
            # Everything else (Composite label + Image-with-description +
            # any other non-tabular chunk) → composite wing, capped at
            # composite_extract_top_n to keep the LLM prompt bounded.
            table_chunks = [
                c for c in sub_chunks
                if c.chunk.metadata.is_table and c.chunk.metadata.table_data
            ]
            composite_chunks = [
                c for c in sub_chunks
                if not c.chunk.metadata.is_table
            ][: settings.composite_extract_top_n]

            if not table_chunks and not composite_chunks:
                emit(
                    "table_extraction_skip",
                    {"subquery": sub, "reason": "no_extraction_candidates"},
                )
                trace.unfound_targets.extend(target_cells)
                continue

            # Fire both wings in parallel. Each returns (values, unfound).
            table_path_task = asyncio.create_task(
                _run_table_path(sub, target_cells, table_chunks, emit)
            )
            composite_path_task = asyncio.create_task(
                _run_composite_path(sub, target_cells, composite_chunks, emit)
            )
            (table_path_values, table_path_unfound), (
                composite_path_values,
                composite_path_unfound,
            ) = await asyncio.gather(table_path_task, composite_path_task)

            # Merge both wings' values into the shared list — the synthesizer
            # treats them identically (both are TableValue); `note` on each
            # carries the provenance.
            table_values.extend(table_path_values)
            table_values.extend(composite_path_values)

            # A target is truly unfound only when BOTH wings missed it.
            table_unfound_set = set(table_path_unfound)
            composite_unfound_set = set(composite_path_unfound)
            truly_unfound = [
                t for t in target_cells
                if t in table_unfound_set and t in composite_unfound_set
            ]
            trace.unfound_targets.extend(truly_unfound)

        # 4. Critic
        emit("critic_start", None)
        critic_out = await critique(
            user_query,
            trace.all_subqueries,
            list(chunks_by_id.values()),
            table_values=table_values,
            unfound_targets=trace.unfound_targets,
        )
        emit("critic_done", critic_out)
        trace.critic_decisions.append(critic_out)
        trace.iterations += 1

        if critic_out.sufficient or trace.iterations >= settings.max_critic_iterations:
            break
        if not critic_out.follow_up_subqueries:
            break

        # Sanitize critic follow-ups the same way we sanitize planner
        # output — same scratchpad + self-flag rules, plus CROSS-ITERATION
        # dedup via the shared `seen_signatures` set. If the critic tries
        # to rephrase a subquery the planner already ran (or a prior
        # critic iteration already ran), it gets dropped here.
        c_stage = f"critic_iter_{trace.iterations}"
        c_kept, c_filtered = _sanitize_subqueries(
            critic_out.follow_up_subqueries,
            stage=c_stage,
            seen_signatures=seen_signatures,
        )
        if c_filtered:
            print(
                f"[critic-sanitize] {c_stage}: dropped "
                f"{len(c_filtered)}/{len(critic_out.follow_up_subqueries)} "
                f"follow-ups (reasons: "
                f"{', '.join(sorted({f['reason'] for f in c_filtered}))})",
                flush=True,
            )
            emit("critic_sanitized",
                 {"stage": c_stage,
                  "original_count": len(critic_out.follow_up_subqueries),
                  "kept_count": len(c_kept),
                  "filtered": c_filtered})
            critic_out.follow_up_subqueries = c_kept
            trace.sanitized_subqueries.extend(c_filtered)

        if not c_kept:
            # Every follow-up was garbage. Stop the loop rather than
            # burning another empty iteration.
            print(
                f"[critic-sanitize] {c_stage}: 0 follow-ups survived — "
                f"exiting critic loop",
                flush=True,
            )
            break

        new_subs = c_kept
        trace.all_subqueries.extend(new_subs)

    # 5. Synthesizer
    emit("synthesizer_start", None)
    final_chunks = sorted(chunks_by_id.values(), key=lambda c: c.score, reverse=True)

    # Aggregate VLM's semantic notes for every unfound target so the
    # synthesizer can use them as authoritative evidence for "what IS
    # disclosed" (rather than re-deriving from raw chunk markdown).
    unfound_context = _build_unfound_context(
        trace.unfound_targets, trace.vlm_verifications,
    )

    answer = await synthesize(
        user_query,
        final_chunks,
        table_values=table_values,
        unfound_context=unfound_context,
    )
    emit("synthesizer_done", answer)

    return QueryResult(
        answer=answer,
        chunks_used=final_chunks,
        table_values=table_values,
        trace=trace,
    )


def _record_vlm_event(trace: QueryTrace, stage: str, data: Any) -> None:
    """Fold a VLM verification / extract event into the audit trace.

    Verify path stages:
      table_vlm_verify_skip / start / done / error
    Extract-from-unfound path stages:
      table_vlm_extract_start / done / not_found / error
    """
    if not isinstance(data, dict):
        return
    target = data.get("target") or "?"

    # ── Verify path ────────────────────────────────────────────────────────
    if stage == "table_vlm_verify_skip":
        trace.vlm_verifications.append(VLMVerificationEvent(
            target=target, stage="skip", reason=data.get("reason"),
        ))
        return

    if stage == "table_vlm_verify_start":
        original = data.get("original")
        trace.vlm_verifications.append(VLMVerificationEvent(
            target=target,
            stage="start",
            chunk_id=getattr(original, "chunk_id", None) if original else None,
            original_value=getattr(original, "value", None) if original else None,
            original_confidence=getattr(original, "confidence", None) if original else None,
        ))
        return

    if stage == "table_vlm_verify_done":
        verified = data.get("verified")
        merged = data.get("merged")
        for ev in reversed(trace.vlm_verifications):
            if ev.target == target and ev.stage == "start":
                ev.stage = "done"
                _capture_vlm_response(ev, verified)
                _capture_merged(ev, merged)
                return
        return

    if stage == "table_vlm_verify_error":
        for ev in reversed(trace.vlm_verifications):
            if ev.target == target and ev.stage == "start":
                ev.stage = "error"
                ev.error = data.get("error")
                return
        trace.vlm_verifications.append(VLMVerificationEvent(
            target=target, stage="error", error=data.get("error"),
        ))
        return

    # ── Extract-from-unfound path ──────────────────────────────────────────
    if stage == "table_vlm_extract_start":
        trace.vlm_verifications.append(VLMVerificationEvent(
            target=target,
            stage="extract_start",
            reason=f"chunk={data.get('chunk_id')}",
            chunk_id=data.get("chunk_id"),
        ))
        return

    if stage == "table_vlm_extract_done":
        verified = data.get("verified")
        result = data.get("result")
        for ev in reversed(trace.vlm_verifications):
            if ev.target == target and ev.stage == "extract_start":
                ev.stage = "extract_done"
                _capture_vlm_response(ev, verified, force_found_true=True)
                _capture_merged(ev, result)
                return
        return

    if stage == "table_vlm_extract_not_found":
        for ev in reversed(trace.vlm_verifications):
            if ev.target == target and ev.stage == "extract_start":
                ev.stage = "extract_not_found"
                ev.vlm_found = False
                # VLM's "why" — passed as `vlm_note` in the emit data by
                # vlm_extract_unfound_targets. Critical for diagnosing whether
                # the segment-vs-combined rule fired vs the VLM just failed.
                ev.vlm_note = data.get("vlm_note")
                return
        return

    if stage == "table_vlm_extract_error":
        for ev in reversed(trace.vlm_verifications):
            if ev.target == target and ev.stage == "extract_start":
                ev.stage = "extract_error"
                ev.error = data.get("error")
                return

    # ── Composite VLM verifier path ────────────────────────────────────────
    # Symmetric with the table VLM verifier events but under the
    # `composite_vlm_verify_` prefix so the audit log distinguishes which
    # wing produced the verification. Same event shapes as the table verifier
    # (skip / start / done / error) — the merge logic is shared.
    if stage == "composite_vlm_verify_skip":
        trace.vlm_verifications.append(VLMVerificationEvent(
            target=target, stage="composite_verify_skip",
            reason=data.get("reason"),
        ))
        return

    if stage == "composite_vlm_verify_start":
        original = data.get("original")
        trace.vlm_verifications.append(VLMVerificationEvent(
            target=target,
            stage="composite_verify_start",
            chunk_id=getattr(original, "chunk_id", None) if original else None,
            original_value=getattr(original, "value", None) if original else None,
            original_confidence=getattr(original, "confidence", None) if original else None,
        ))
        return

    if stage == "composite_vlm_verify_done":
        verified = data.get("verified")
        merged = data.get("merged")
        for ev in reversed(trace.vlm_verifications):
            if ev.target == target and ev.stage == "composite_verify_start":
                ev.stage = "composite_verify_done"
                _capture_vlm_response(ev, verified)
                _capture_merged(ev, merged)
                return
        return

    if stage == "composite_vlm_verify_error":
        for ev in reversed(trace.vlm_verifications):
            if ev.target == target and ev.stage == "composite_verify_start":
                ev.stage = "composite_verify_error"
                ev.error = data.get("error")
                return
        trace.vlm_verifications.append(VLMVerificationEvent(
            target=target, stage="composite_verify_error",
            error=data.get("error"),
        ))
        return

    # ── Composite VLM extract-from-unfound path ────────────────────────────
    # Symmetric with the table wing's extract-from-unfound events but under
    # the `composite_vlm_extract_` prefix. Fires when the composite text
    # extractor gave up on a target and VLM tries fresh on the top chunks.
    if stage == "composite_vlm_extract_start":
        trace.vlm_verifications.append(VLMVerificationEvent(
            target=target,
            stage="composite_extract_start",
            reason=f"chunk={data.get('chunk_id')}",
            chunk_id=data.get("chunk_id"),
        ))
        return

    if stage == "composite_vlm_extract_done":
        verified = data.get("verified")
        result = data.get("result")
        for ev in reversed(trace.vlm_verifications):
            if ev.target == target and ev.stage == "composite_extract_start":
                ev.stage = "composite_extract_done"
                _capture_vlm_response(ev, verified, force_found_true=True)
                _capture_merged(ev, result)
                return
        return

    if stage == "composite_vlm_extract_not_found":
        for ev in reversed(trace.vlm_verifications):
            if ev.target == target and ev.stage == "composite_extract_start":
                ev.stage = "composite_extract_not_found"
                ev.vlm_found = False
                # VLM's "why" — passed as `vlm_note` in the emit data by
                # vlm_extract_unfound_composite_targets. Critical for
                # diagnosing whether segment-vs-combined fired vs VLM failed.
                ev.vlm_note = data.get("vlm_note")
                return
        return

    if stage == "composite_vlm_extract_error":
        for ev in reversed(trace.vlm_verifications):
            if ev.target == target and ev.stage == "composite_extract_start":
                ev.stage = "composite_extract_error"
                ev.error = data.get("error")
                return
        trace.vlm_verifications.append(VLMVerificationEvent(
            target=target, stage="composite_extract_error",
            error=data.get("error"),
        ))
        return


# ── Planner output sanitizer ────────────────────────────────────────────────
# Belt-and-suspenders defence against models that occasionally leak their
# own reasoning scratchpad into the executable subqueries array. The
# PLANNER_SYSTEM prompt now has an explicit OUTPUT DISCIPLINE section
# forbidding this, but we still filter defensively — a stray scratchpad
# subquery burns retrieval + LLM + VLM budget for zero value.

# Query strings that indicate the "subquery" is actually a scratchpad note.
# Compared as a case-insensitive whole-string OR prefix match on the query
# after trimming whitespace.
_SCRATCHPAD_QUERY_PREFIXES = (
    "howto",
    "howto_",
    "final_review",
    "final review",
    "validity_check",
    "validity check",
    "sanity_check",
    "sanity check",
    "ask_cell_count",
    "ask_",
    "check:",
    "review:",
    "todo:",
    "todo ",
    "remediate ",
    "cost_budget",
    "cost budget",
    "plan_review",
    "plan review",
)

# Rationale phrases that mean the model itself does NOT want this
# subquery to execute. Case-insensitive substring match on the rationale.
_SELF_FLAGGED_DROP_PHRASES = (
    "should be dropped",
    "should be ignored",
    "should be discarded",
    "void this subquery",
    "voids this subquery",
    "no-op",
    "drops on the duplicate",
    "duplicate of subquery",
    "near-duplicate of subquery",
    "alternate phrasing",
    "planning artifact only",
    "planning artifact",
    "planning scratchpad",
    "scratchpad only",
    "dropping this",
    "drop this subquery",
    "removed by",
    "cut from the plan",
    "does not execute",
    "downstream should ignore",
    "downstream pipeline should ignore",
)


def _canonicalize_subquery_key(sub: Subquery) -> tuple[str, tuple[str, ...]]:
    """Lower-cased, whitespace-collapsed (query, target_cells) signature.

    Used for deduplication — two subqueries with cosmetic differences
    (extra whitespace, different casing) collide on this key.
    """
    q = " ".join((sub.query or "").lower().split())
    tcs = tuple(sorted(
        " ".join(t.lower().split()) for t in (sub.target_cells or [])
    ))
    return (q, tcs)


def _sanitize_subqueries(
    subqueries: list[Subquery],
    *,
    stage: str = "planner",
    seen_signatures: set[tuple[str, tuple[str, ...]]] | None = None,
    max_new: int = 3,
) -> tuple[list[Subquery], list[dict[str, str]]]:
    """Drop scratchpad-shaped and self-flagged subqueries; dedupe; cap.

    Used for BOTH the planner's initial output AND the critic's follow-ups
    on each iteration. When `seen_signatures` is provided (a mutable set
    of canonical `(query, target_cells)` signatures from prior stages /
    iterations), incoming subqueries are also deduplicated against it —
    so if the critic's iteration-2 follow-up cosmetically restates a
    subquery the planner already ran in iteration 1, we catch it.

    Kept subqueries' signatures are ADDED to `seen_signatures` in place,
    so the same set can be threaded through subsequent stages.

    Args:
        subqueries: subqueries emitted by the LLM (planner or critic)
        stage:     label stamped on every filtered entry — "planner" or
                   e.g. "critic_iter_2". Shows up in the audit log so
                   readers can see which stage a drop happened at.
        seen_signatures: canonical signatures already accepted from prior
                   stages/iterations. Mutated in-place — kept subqueries'
                   signatures are added. Pass None on the first call
                   (which creates a fresh set); pass the same set on
                   subsequent calls to enable cross-stage dedup.
        max_new:   cap on NEW subqueries accepted in this call. Follow-ups
                   from the critic get max_new=3 too — the "3 max" HARD
                   RULE applies per stage, not cumulatively (each iteration
                   can legitimately try 1–3 new angles).

    Returns:
        (kept, filtered). Every entry in `filtered` has a "stage" field
        stamped with the passed-in `stage` label.
    """
    kept: list[Subquery] = []
    filtered: list[dict[str, str]] = []
    if seen_signatures is None:
        seen_signatures = set()

    for sub in subqueries:
        q_stripped = (sub.query or "").strip().lower()
        rationale_lower = (sub.rationale or "").lower()

        # Rule 1: scratchpad-shaped query string
        matched_prefix = next(
            (p for p in _SCRATCHPAD_QUERY_PREFIXES
             if q_stripped == p.rstrip() or q_stripped.startswith(p)),
            None,
        )
        if matched_prefix is not None:
            filtered.append({
                "stage": stage,
                "query": sub.query,
                "reason": f"scratchpad_query_prefix:{matched_prefix!r}",
            })
            continue

        # Rule 2: rationale self-flags for dropping
        matched_phrase = next(
            (ph for ph in _SELF_FLAGGED_DROP_PHRASES if ph in rationale_lower),
            None,
        )
        if matched_phrase is not None:
            filtered.append({
                "stage": stage,
                "query": sub.query,
                "reason": f"self_flagged_drop:{matched_phrase!r}",
            })
            continue

        # Rule 3: dedupe by canonical (query, target_cells) signature.
        # `seen_signatures` may already contain signatures from prior
        # stages/iterations — that's how we catch cross-iteration
        # cosmetic reissues (critic tries to rephrase what the planner
        # already ran).
        key = _canonicalize_subquery_key(sub)
        if key in seen_signatures:
            filtered.append({
                "stage": stage,
                "query": sub.query,
                "reason": "duplicate_of_earlier_subquery",
            })
            continue
        seen_signatures.add(key)

        # Rule 4: cap at max_new (drop the excess with a specific reason)
        if len(kept) >= max_new:
            filtered.append({
                "stage": stage,
                "query": sub.query,
                "reason": f"exceeds_max_subqueries={max_new}",
            })
            continue

        kept.append(sub)

    return kept, filtered


def _format_report_context(
    report_ids: list[str], metadatas: list[ReportMetadata],
) -> str | None:
    """Render a Report context block for the planner.

    Returns None when NO reports have metadata on disk — the planner then
    receives no context block and applies default rules. When SOME reports
    have metadata and others don't, we render only what we have (and label
    the missing ones) so the planner has partial signal rather than none.
    """
    if not metadatas:
        return None

    by_id = {m.report_id: m for m in metadatas}
    lines: list[str] = [
        f"Reports being queried: {len(report_ids)}",
        "",
    ]
    for rid in report_ids:
        meta = by_id.get(rid)
        if meta is None:
            lines.append(f"Report: {rid}")
            lines.append("  (metadata unavailable — likely ingested before content-aware planning)")
            lines.append("")
            continue
        total = max(1, meta.total_chunks)  # avoid /0
        tbl_pct = 100 * meta.table_chunks / total
        cmp_pct = 100 * meta.composite_chunks / total
        img_pct = 100 * meta.infographic_chunks / total
        lines.append(f"Report: {meta.report_id}")
        lines.append(
            f"  Total chunks: {meta.total_chunks} "
            f"({tbl_pct:.0f}% tables, {cmp_pct:.0f}% composite prose, "
            f"{img_pct:.0f}% infographic-derived)"
        )
        lines.append(f"  Dominant content type: {meta.dominant_content_type}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_unfound_context(
    unfound_targets: list[str],
    vlm_events: list[VLMVerificationEvent],
) -> list[UnfoundTargetContext]:
    """Aggregate every VLM's semantic note tied to each unfound target.

    For each target the pipeline concluded is not disclosed, collect the
    notes from every VLM extract-from-unfound event that tried to answer it
    (across both wings — table and composite). The synthesizer then uses
    these notes as authoritative evidence, per the "UNFOUND IS AUTHORITATIVE"
    rule in SYNTHESIZER_SYSTEM.

    Only includes events where `vlm_note` is populated. Events where the VLM
    failed silently (empty note) don't contribute — there's nothing useful
    for the synthesizer to cite.
    """
    if not unfound_targets:
        return []

    # Consider all "not_found" stages from both wings' extract-from-unfound
    # paths. Verify events aren't included because if verify fired at all,
    # the text extractor found something (so the target wasn't unfound).
    UNFOUND_STAGES = {"extract_not_found", "composite_extract_not_found"}

    by_target: dict[str, list[UnfoundVLMEvidence]] = {t: [] for t in unfound_targets}
    for ev in vlm_events:
        if ev.stage not in UNFOUND_STAGES:
            continue
        if ev.target not in by_target:
            continue
        if not ev.vlm_note or not ev.vlm_note.strip():
            continue
        # Deduplicate: if the exact same (chunk_id, note) appears twice,
        # keep only one (can happen when the same chunk is examined by both
        # wings — usually doesn't, but be defensive).
        existing_pairs = {(e.chunk_id, e.note) for e in by_target[ev.target]}
        chunk_id = ev.chunk_id or "unknown"
        if (chunk_id, ev.vlm_note) in existing_pairs:
            continue
        by_target[ev.target].append(UnfoundVLMEvidence(
            chunk_id=chunk_id, note=ev.vlm_note,
        ))

    return [
        UnfoundTargetContext(target=t, vlm_evidence=by_target[t])
        for t in unfound_targets
    ]


def _capture_vlm_response(
    ev: VLMVerificationEvent, verified: Any, *, force_found_true: bool = False,
) -> None:
    """Populate every VLM-side field on the event from the raw response object.

    `verified` may be a `TableVLMVerification` (VLM's raw output) OR a
    `TableValue` (VLM's output already wrapped for merge). Both have the
    same field names (found / value / row_label / column_label / unit /
    confidence / note), so `getattr` handles both.

    `force_found_true` is set by the extract-from-unfound `done` handler
    where reaching that handler already implies the VLM found the target
    (its emit event fires only after the found=True branch executed).
    """
    if verified is None:
        return
    ev.vlm_found = True if force_found_true else getattr(verified, "found", None)
    ev.vlm_value = getattr(verified, "value", None)
    ev.vlm_confidence = getattr(verified, "confidence", None)
    ev.vlm_row_label = getattr(verified, "row_label", None)
    ev.vlm_column_label = getattr(verified, "column_label", None)
    ev.vlm_unit = getattr(verified, "unit", None)
    ev.vlm_note = getattr(verified, "note", None)


def _capture_merged(ev: VLMVerificationEvent, merged: Any) -> None:
    """Populate every merged-side field on the event from the merged TableValue."""
    if merged is None:
        return
    ev.merged_value = getattr(merged, "value", None)
    ev.merged_confidence = getattr(merged, "confidence", None)
    ev.merged_note = getattr(merged, "note", None)


BatchEventCallback = Callable[[int, str, str, Any], None]


@dataclass
class BatchOutcome:
    """One slot in the batch result list — a query plus either its result or its error."""
    query: str
    result: QueryResult | None
    error: str | None  # populated iff result is None


async def answer_queries(
    queries: list[str],
    report_ids: list[str],
    *,
    store: QdrantStore | None = None,
    on_query_event: BatchEventCallback | None = None,
) -> list[BatchOutcome]:
    """Run a batch of queries sequentially, one QueryResult per query.

    Sequential (not parallel) because:
      - Local-file Qdrant only allows one client, and we're sharing `store`
      - Fireworks rate limits are gentler at 1 concurrent query
      - Progress is easier to reason about when the UI shows one at a time

    `on_query_event(index, query, stage, data)` fires for every pipeline
    stage of every query so the caller can drive a live progress UI. The
    stage names come from `answer_query`'s existing events; we prefix each
    call with (index, query) so the UI can attach the update to the right
    row/card.

    Failures in one query are captured as `BatchOutcome.error` and DON'T
    abort the batch — critical for a 10-query compliance sweep.
    """
    if not queries:
        return []
    if not report_ids:
        raise ValueError("report_ids must be non-empty")

    if store is None:
        store = QdrantStore()

    outcomes: list[BatchOutcome] = []
    for idx, q in enumerate(queries):
        def _forward(stage: str, data: Any = None, _idx: int = idx, _q: str = q) -> None:
            if on_query_event is not None:
                try:
                    on_query_event(_idx, _q, stage, data)
                except Exception:  # noqa: BLE001
                    pass

        _forward("batch_query_start")
        try:
            result = await answer_query(
                q, report_ids=report_ids, store=store, on_event=_forward
            )
            _forward("batch_query_done", result)
            outcomes.append(BatchOutcome(query=q, result=result, error=None))
        except Exception as e:  # noqa: BLE001
            _forward("batch_query_error", str(e))
            outcomes.append(BatchOutcome(query=q, result=None, error=str(e)))
    return outcomes


async def _run_table_path(
    sub: Subquery,
    target_cells: list[str],
    table_chunks: list[RetrievedChunk],
    emit: Callable[[str, Any], None],
) -> tuple[list[TableValue], list[str]]:
    """Table extraction wing: text extractor + mandatory VLM verify + VLM
    extract-from-unfound. Returns (values_to_store, still_unfound_targets).

    Skips entirely when there are no table chunks — every target starts as
    unfound from this wing's perspective, and the composite wing (running
    in parallel) may still rescue them.
    """
    if not table_chunks:
        emit(
            "table_extraction_skip",
            {"subquery": sub, "reason": "no_tables_in_results"},
        )
        return [], list(target_cells)

    emit(
        "table_extraction_start",
        {"subquery": sub, "table_count": len(table_chunks)},
    )
    extracted = await extract_table_values(target_cells, table_chunks)
    emit("table_extraction_done", extracted)

    # MANDATORY visual verification of every extracted cell — Chandra can't
    # detect its own header-flattening errors (spanning headers like
    # [Male | Female | Others] over [Number | Median] shift columns without
    # lowering the extractor's stated confidence). See table_vlm_verifier.
    values_to_store = extracted.values
    if settings.enable_table_vlm_verify and extracted.values:
        table_chunks_by_id = {c.chunk.chunk_id: c for c in table_chunks}
        emit(
            "table_vlm_verify_batch_start",
            {"subquery": sub, "n_to_verify": len(extracted.values)},
        )
        values_to_store = await verify_and_merge_table_values(
            extracted.values, table_chunks_by_id, on_event=emit
        )
        emit("table_vlm_verify_batch_done", values_to_store)

    # VLM RESCUE for targets the text extractor gave up on (markdown too
    # broken for text parsing, but the visual table is still legible).
    unfound_after = list(extracted.unfound)
    if (
        settings.enable_table_vlm_extract_unfound
        and unfound_after
        and table_chunks
    ):
        emit(
            "table_vlm_extract_batch_start",
            {"subquery": sub, "n_unfound": len(unfound_after)},
        )
        rescued, still_unfound = await vlm_extract_unfound_targets(
            unfound_after, table_chunks, on_event=emit,
        )
        emit(
            "table_vlm_extract_batch_done",
            {"rescued": rescued, "still_unfound": still_unfound},
        )
        values_to_store = list(values_to_store) + list(rescued)
        unfound_after = still_unfound

    return list(values_to_store), unfound_after


async def _run_composite_path(
    sub: Subquery,
    target_cells: list[str],
    composite_chunks: list[RetrievedChunk],
    emit: Callable[[str, Any], None],
) -> tuple[list[TableValue], list[str]]:
    """Composite extraction wing: LLM extractor over prose / list /
    infographic-transcribed chunks + mandatory VLM verification of every
    extracted value. Returns (values, unfound_targets) with the same shape
    as the table path.

    Skipped when disabled or when there are no composite chunks in the
    subquery's results (in which case every target is unfound from this
    wing's perspective — the table wing may still catch them). Chandra
    puts most report content into Composite chunks, so this wing carries
    the majority of factual answers on non-BRSR reports.

    VLM verification: every value the extractor produced gets cross-checked
    against its source chunk's cropped bbox. Chandra loses the visual origin
    when it OCRs infographic text into Composite chunks (the is_infographic
    flag only covers ~10% of visually-derived Composite content), so we
    verify all extractor outputs regardless. Concurrency-bounded by
    vlm_concurrency; typical query produces 1-3 verifiable values → 1 round.
    """
    if not settings.enable_composite_extraction or not composite_chunks:
        emit(
            "composite_extraction_skip",
            {"subquery": sub, "reason": "no_composite_chunks_or_disabled"},
        )
        return [], list(target_cells)

    emit(
        "composite_extraction_start",
        {"subquery": sub, "chunk_count": len(composite_chunks)},
    )
    result = await extract_composite_values(target_cells, composite_chunks)
    emit("composite_extraction_done", result)

    # MANDATORY visual verification. Chandra doesn't preserve infographic
    # origin for Composite chunks it OCR'd, and the extractor working on
    # flattened prose can mis-attribute values. VLM re-reads the source
    # region and the shared merge logic (see table_vlm_verifier) applies:
    # agreement → high; VLM-high disagreement → REPLACE; VLM-medium/low
    # disagreement → DOWNGRADE to low with both readings in note.
    values_to_store = result.values
    if settings.enable_composite_vlm_verify and result.values:
        composite_chunks_by_id = {c.chunk.chunk_id: c for c in composite_chunks}
        emit(
            "composite_vlm_verify_batch_start",
            {"subquery": sub, "n_to_verify": len(result.values)},
        )
        values_to_store = await verify_and_merge_composite_values(
            result.values, composite_chunks_by_id, on_event=emit,
        )
        emit("composite_vlm_verify_batch_done", values_to_store)

    # VLM RESCUE for targets the composite extractor gave up on. Symmetric
    # with the table wing's extract-from-unfound path — independent VLM
    # extraction over the top composite chunks. Catches ambiguous prose
    # cases or infographic-transcribed text where the text extractor
    # couldn't lock onto a value but the visual layout is still legible.
    unfound_after = list(result.unfound)
    if (
        settings.enable_composite_vlm_extract_unfound
        and unfound_after
        and composite_chunks
    ):
        emit(
            "composite_vlm_extract_batch_start",
            {"subquery": sub, "n_unfound": len(unfound_after)},
        )
        rescued, still_unfound = await vlm_extract_unfound_composite_targets(
            unfound_after, composite_chunks, on_event=emit,
        )
        emit(
            "composite_vlm_extract_batch_done",
            {"rescued": rescued, "still_unfound": still_unfound},
        )
        values_to_store = list(values_to_store) + list(rescued)
        unfound_after = still_unfound

    return list(values_to_store), unfound_after


async def _retrieve_per_subquery(
    subs: list[Subquery],
    report_ids: list[str],
    store: QdrantStore,
) -> list[list[RetrievedChunk]]:
    """Run retrieval for each subquery in parallel and keep per-subquery groupings.

    We keep results grouped (rather than flattening to a dict) because the table
    extractor needs to know which tables came back for which subquery — that's
    the only way to scope target_cells to the right factual_lookup subquery.
    """
    return list(
        await asyncio.gather(
            *(retrieve_for_subquery(s, report_ids=report_ids, store=store) for s in subs)
        )
    )
