"""Per-query audit log.

After every `answer_query` call, we serialize EVERYTHING the orchestrator saw
and did into a flat text file on disk + return the string for inline display.
This is the audit/debugging surface — when an answer looks wrong, you read the
log and see:

  • what the planner decomposed the question into
  • what subqueries actually went to Qdrant (with must_phrases, filters, etc.)
  • every chunk that came back, with full text, score breakdown, and metadata
  • which chunks were cited by the synthesizer
  • what cells the Table Extractor pulled (and which it couldn't find)
  • every critic decision in the loop
  • the final answer

Logs land in `data/query_logs/<timestamp>_<slug>.txt` so you can grep/diff
across queries later. The Streamlit UI also renders the same content inline.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from agentic_rag.orchestrator import QueryResult

LOG_DIR = Path("data/query_logs")
RULE = "=" * 88
SUB_RULE = "-" * 88


def save_query_log(
    question: str,
    report_ids: list[str],
    result: QueryResult,
    *,
    log_dir: Path | None = None,
) -> tuple[Path, str]:
    """Format + persist the log. Returns (file_path, log_text)."""
    log_dir = log_dir or LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = _slugify(question)[:60] or "query"
    path = log_dir / f"{ts}_{slug}.txt"

    text = format_query_log(question, report_ids, result)
    path.write_text(text, encoding="utf-8")
    return path, text


def format_query_log(question: str, report_ids: list[str], result: QueryResult) -> str:
    cited_chunk_ids = {c.chunk_id for c in result.answer.citations}
    parts: list[str] = []

    # ── Header ─────────────────────────────────────────────────────────────
    parts.append(RULE)
    parts.append(f"QUERY        : {question}")
    parts.append(f"REPORTS      : {', '.join(report_ids)}")
    parts.append(f"TIMESTAMP    : {datetime.now().isoformat(timespec='seconds')}")
    parts.append(f"CONFIDENCE   : {result.answer.confidence}")
    parts.append(f"ITERATIONS   : {result.trace.iterations}")
    parts.append(f"CHUNKS USED  : {len(result.chunks_used)}  (cited: {len(cited_chunk_ids)})")
    parts.append(f"TABLE VALUES : {len(result.table_values)}")
    parts.append(f"VLM VERIFY   : {len(result.trace.vlm_verifications)} event(s)")
    parts.append(RULE)

    # ── Planner ────────────────────────────────────────────────────────────
    parts.append("\n## PLANNER REASONING\n")
    parts.append(result.trace.planner.reasoning.strip())

    parts.append(f"\n## SUBQUERIES ({len(result.trace.all_subqueries)} total — incl. critic follow-ups)\n")
    for i, sq in enumerate(result.trace.all_subqueries, start=1):
        parts.append(f"[{i}] type={sq.query_type}")
        parts.append(f"    query        : {sq.query!r}")
        parts.append(f"    must_phrases : {sq.must_phrases}")
        parts.append(f"    keywords     : {sq.keywords}")
        parts.append(f"    target_cells : {sq.target_cells}")
        parts.append(f"    filters      : {sq.filters.model_dump(exclude_none=True)}")
        if sq.hyde_doc:
            parts.append(f"    hyde_doc     : {sq.hyde_doc[:200]}{'…' if len(sq.hyde_doc) > 200 else ''}")
        parts.append(f"    rationale    : {sq.rationale!r}")
        parts.append("")

    # ── Planner sanitizer output ──────────────────────────────────────────
    # Any subqueries dropped before execution — scratchpad noise,
    # self-flagged duplicates, over-cap subqueries. Empty on a clean plan.
    if result.trace.planner_sanitized_out:
        parts.append(
            f"\n## PLANNER SANITIZER — dropped "
            f"{len(result.trace.planner_sanitized_out)} subquery(ies) "
            f"before execution\n"
        )
        for i, f in enumerate(result.trace.planner_sanitized_out, start=1):
            parts.append(f"[{i}] reason: {f['reason']}")
            parts.append(f"    query : {f['query']!r}")
            parts.append("")

    # ── Critic decisions ───────────────────────────────────────────────────
    parts.append(f"\n## CRITIC DECISIONS ({result.trace.iterations} iterations)\n")
    for i, d in enumerate(result.trace.critic_decisions, start=1):
        verdict = "SUFFICIENT" if d.sufficient else "NEEDS_MORE"
        parts.append(f"[Iteration {i}] {verdict}")
        if d.missing_info:
            parts.append(f"    missing: {d.missing_info}")
        if d.follow_up_subqueries:
            parts.append(f"    → issued {len(d.follow_up_subqueries)} follow-up subquery(s):")
            for j, fsq in enumerate(d.follow_up_subqueries, start=1):
                parts.append(f"       ({j}) {fsq.query!r}  must={fsq.must_phrases}")
        parts.append("")

    if result.trace.unfound_targets:
        parts.append("\n## UNFOUND TABLE TARGETS\n")
        for t in result.trace.unfound_targets:
            parts.append(f"  - {t}")

    # ── VLM events (all 4 stage families) ──────────────────────────────────
    # Records the FULL VLM response — including row_label / column_label /
    # unit / note — for every VLM verify + extract-from-unfound call across
    # both the table wing and the composite wing. The `note` field is
    # especially important for `found=False` events: it carries the VLM's
    # semantic reasoning ("Table shows split by Male/Female, no combined
    # total") which is the difference between "the segment-vs-combined rule
    # worked" and "VLM just failed."
    if result.trace.vlm_verifications:
        parts.append(f"\n## VLM EVENTS ({len(result.trace.vlm_verifications)})\n")
        parts.append("Stage families:")
        parts.append("  table verify        : {skip / start / done / error}")
        parts.append("  table extract       : {extract_start / done / not_found / error}")
        parts.append("  composite verify    : {composite_verify_skip / start / done / error}")
        parts.append("  composite extract   : {composite_extract_start / done / not_found / error}")
        parts.append("")
        for ev in result.trace.vlm_verifications:
            parts.append(SUB_RULE)
            parts.append(f"target      : {ev.target}")
            parts.append(f"stage       : {ev.stage}")
            if ev.reason:
                parts.append(f"skip reason : {ev.reason}")
            if ev.original_value is not None or ev.original_confidence is not None:
                parts.append(
                    f"original    : value={ev.original_value!r}  "
                    f"confidence={ev.original_confidence!r}"
                )
            if ev.vlm_found is not None or ev.vlm_value is not None or ev.vlm_note is not None:
                parts.append(
                    f"vlm         : found={ev.vlm_found}  "
                    f"value={ev.vlm_value!r}  confidence={ev.vlm_confidence!r}"
                )
                # Show VLM's labels + unit when any of them is populated —
                # these are what let us tell segment-vs-combined mismatches
                # apart from simple label agreement.
                if (ev.vlm_row_label is not None
                        or ev.vlm_column_label is not None
                        or ev.vlm_unit is not None):
                    parts.append(
                        f"vlm labels  : row={ev.vlm_row_label!r}  "
                        f"col={ev.vlm_column_label!r}  unit={ev.vlm_unit!r}"
                    )
                if ev.vlm_note:
                    parts.append(f"vlm note    : {ev.vlm_note}")
            if ev.merged_value is not None or ev.merged_note is not None:
                parts.append(
                    f"merged      : value={ev.merged_value!r}  "
                    f"confidence={ev.merged_confidence!r}"
                )
                if ev.merged_note:
                    parts.append(f"merged note : {ev.merged_note}")
            if ev.error:
                parts.append(f"error       : {ev.error}")
            parts.append("")

    # ── Table values ───────────────────────────────────────────────────────
    parts.append(f"\n## TABLE VALUES EXTRACTED ({len(result.table_values)})\n")
    if not result.table_values:
        parts.append("(none)")
    for tv in result.table_values:
        parts.append(f"  target     : {tv.target_description}")
        parts.append(f"  row        : {tv.row_label}")
        parts.append(f"  column     : {tv.column_label}")
        parts.append(f"  value      : {tv.value}  {tv.unit or ''}")
        parts.append(f"  confidence : {tv.confidence}")
        parts.append(f"  chunk_id   : {tv.chunk_id}")
        if tv.note:
            parts.append(f"  note       : {tv.note}")
        parts.append("")

    # ── Final answer ───────────────────────────────────────────────────────
    parts.append("\n## FINAL ANSWER\n")
    parts.append(result.answer.answer.strip())
    if result.answer.caveats:
        parts.append(f"\nCAVEATS: {result.answer.caveats}")

    parts.append(f"\n## CITATIONS ({len(result.answer.citations)})\n")
    for c in result.answer.citations:
        parts.append(f"  chunk_id : {c.chunk_id}")
        parts.append(f"  report   : {c.report_id}")
        parts.append(f"  page     : {c.page}")
        parts.append(f"  section  : {c.section}")
        if c.quote:
            parts.append(f"  quote    : {c.quote}")
        parts.append("")

    # ── Every chunk used (cited ones marked) ──────────────────────────────
    parts.append(f"\n## CHUNKS USED (sorted by score)\n")
    for i, rc in enumerate(result.chunks_used, start=1):
        m = rc.chunk.metadata
        marker = "★ CITED" if rc.chunk.chunk_id in cited_chunk_ids else "       "
        parts.append(SUB_RULE)
        parts.append(f"{marker}  #{i}  chunk_id={rc.chunk.chunk_id}")
        parts.append(
            f"  score={rc.score:.4f}  label={m.label or '-'}  page={m.page}  "
            f"is_table={m.is_table}"
        )
        meta_line = []
        if m.section:
            meta_line.append(f"section={m.section!r}")
        if m.company:
            meta_line.append(f"company={m.company!r}")
        if m.report_year:
            meta_line.append(f"year={m.report_year}")
        if m.framework:
            meta_line.append(f"framework={m.framework.value}")
        if meta_line:
            parts.append("  " + "  ".join(meta_line))
        score_bits = []
        if rc.dense_score is not None:
            score_bits.append(f"dense={rc.dense_score:.4f}")
        if rc.sparse_score is not None:
            score_bits.append(f"sparse={rc.sparse_score:.4f}")
        if rc.rerank_score is not None:
            score_bits.append(f"rerank={rc.rerank_score:.4f}")
        if score_bits:
            parts.append("  scores: " + "  ".join(score_bits))

        parts.append("\nTEXT:")
        parts.append(rc.chunk.text)

        if m.is_table and m.table_data:
            parts.append("\nSTRUCTURED TABLE:")
            parts.append(json.dumps(m.table_data.model_dump(), indent=2, ensure_ascii=False))
        parts.append("")

    parts.append(RULE)
    return "\n".join(parts)


def _slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[-\s]+", "-", s)
    return s.strip("-")
