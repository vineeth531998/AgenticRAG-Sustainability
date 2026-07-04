"""Synthesizer: produce the final cited answer from retrieved chunks + extracted table values."""
from __future__ import annotations

import json

from agentic_rag.agents.prompts import SYNTHESIZER_SYSTEM
from agentic_rag.llm import structured_call
from agentic_rag.schemas import (
    RetrievedChunk,
    SynthesizerOutput,
    TableValue,
    UnfoundTargetContext,
)
from config.settings import settings


def _render_chunks_for_synth(chunks: list[RetrievedChunk]) -> str:
    """Render chunks so the synthesizer can cite them precisely.

    For tables we include both the markdown view (good for prose context) and
    the structured JSON view (good for cross-checking against TableValues).
    """
    blocks: list[str] = []
    for r in chunks:
        m = r.chunk.metadata
        header = (
            f"[chunk_id={r.chunk.chunk_id} | report={m.report_id} | page={m.page} | "
            f"section={m.section or '-'} | company={m.company or '-'} | "
            f"year={m.report_year or '-'} | framework={m.framework.value if m.framework else '-'} | "
            f"is_table={m.is_table}]"
        )
        body = r.chunk.text
        if m.is_table and m.table_data:
            body += "\n\nStructured table:\n" + json.dumps(m.table_data.model_dump(), indent=2)
        blocks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(blocks)


def _render_table_values(values: list[TableValue]) -> str:
    if not values:
        return "(none)"
    return "\n".join(
        f"- chunk={v.chunk_id} | target={v.target_description!r} | "
        f"row={v.row_label!r} | col={v.column_label!r} | "
        f"value={v.value!r} | unit={v.unit!r} | "
        f"confidence={v.confidence}"
        + (f" | note={v.note!r}" if v.note else "")
        for v in values
    )


def _render_unfound_context(items: list[UnfoundTargetContext]) -> str:
    """Render unfound targets + VLM evidence for the synthesizer.

    Each unfound target is followed by every VLM note that examined it,
    tied to the source chunk_id. Empty vlm_evidence lists are still shown
    so the synthesizer knows the target was investigated (just without a
    semantic explanation).
    """
    if not items:
        return "(none — the pipeline found every requested target)"
    blocks: list[str] = []
    for it in items:
        parts = [f"TARGET: {it.target}"]
        if not it.vlm_evidence:
            parts.append("  (no VLM notes — pipeline could not answer this target)")
        for ev in it.vlm_evidence:
            parts.append(f"  ↳ chunk={ev.chunk_id}: {ev.note}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


async def synthesize(
    user_query: str,
    retrieved: list[RetrievedChunk],
    *,
    table_values: list[TableValue] | None = None,
    unfound_context: list[UnfoundTargetContext] | None = None,
) -> SynthesizerOutput:
    """Synthesize the final answer.

    Three inputs beyond the user query and retrieved chunks:
      • `table_values` — values the extraction pipeline (text + VLM merged)
        successfully pulled out of chunks. Prefer these for numeric KPIs.
      • `unfound_context` — for every target the pipeline concluded was
        NOT disclosed, the VLM's semantic notes explaining WHY (and what
        IS disclosed instead). These are AUTHORITATIVE per the
        UNFOUND CONTEXT section of SYNTHESIZER_SYSTEM — do not re-derive
        their values from raw chunk markdown.
    """
    table_values = table_values or []
    unfound_context = unfound_context or []
    user_content = (
        f"User question:\n{user_query}\n\n"
        f"Retrieved evidence ({len(retrieved)} chunks):\n"
        f"{_render_chunks_for_synth(retrieved)}\n\n"
        f"Pre-extracted table values ({len(table_values)} — PREFER these when "
        f"reporting numeric KPIs):\n{_render_table_values(table_values)}\n\n"
        f"Unfound targets with VLM evidence "
        f"({len(unfound_context)} — these are AUTHORITATIVE; do NOT re-derive "
        f"their values from chunk markdown — see UNFOUND CONTEXT rules):\n"
        f"{_render_unfound_context(unfound_context)}\n\n"
        "Write the final answer now, with inline [^chunk_id] citations on every claim."
    )

    return await structured_call(
        model=settings.synthesizer_model,
        system_prompt=SYNTHESIZER_SYSTEM,
        user_content=user_content,
        output_model=SynthesizerOutput,
        # Synthesizer writes long answers with inline citations; with Qwen3.7's
        # thinking overhead, 32k keeps both fitting comfortably.
        max_tokens=32000,
    )
