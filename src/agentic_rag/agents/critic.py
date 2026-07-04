"""Sufficiency critic: are the retrieved chunks + extracted table values enough?"""
from __future__ import annotations

from agentic_rag.agents.prompts import CRITIC_SYSTEM
from agentic_rag.llm import structured_call
from agentic_rag.schemas import (
    CriticOutput,
    RetrievedChunk,
    Subquery,
    TableValue,
)
from config.settings import settings


def _render_chunks(chunks: list[RetrievedChunk]) -> str:
    out: list[str] = []
    for r in chunks:
        m = r.chunk.metadata
        header = (
            f"[chunk_id={r.chunk.chunk_id} | report={m.report_id} | page={m.page} | "
            f"section={m.section or '-'} | is_table={m.is_table} | score={r.score:.3f}]"
        )
        out.append(f"{header}\n{r.chunk.text}\n")
    return "\n---\n".join(out)


def _render_subqueries(subs: list[Subquery]) -> str:
    return "\n".join(
        f"- type={s.query_type} | q={s.query!r} | must={s.must_phrases} | "
        f"keywords={s.keywords} | filters={s.filters.model_dump(exclude_none=True)} | "
        f"targets={s.target_cells}"
        for s in subs
    )


def _render_table_values(values: list[TableValue]) -> str:
    if not values:
        return "(none)"
    return "\n".join(
        f"- chunk={v.chunk_id} | target={v.target_description!r} | "
        f"{v.row_label!r} × {v.column_label!r} = {v.value!r} "
        f"({v.unit or 'no unit'}) [{v.confidence}]"
        + (f" — {v.note}" if v.note else "")
        for v in values
    )


async def critique(
    user_query: str,
    subqueries_run: list[Subquery],
    retrieved: list[RetrievedChunk],
    *,
    table_values: list[TableValue] | None = None,
    unfound_targets: list[str] | None = None,
) -> CriticOutput:
    table_values = table_values or []
    unfound_targets = unfound_targets or []

    user_content = (
        f"Original user question:\n{user_query}\n\n"
        f"Subqueries already run:\n{_render_subqueries(subqueries_run)}\n\n"
        f"Chunks retrieved ({len(retrieved)} total):\n{_render_chunks(retrieved)}\n\n"
        f"Table values extracted ({len(table_values)}):\n{_render_table_values(table_values)}\n\n"
        f"Unfound table targets ({len(unfound_targets)}):\n"
        + ("\n".join(f"- {t}" for t in unfound_targets) if unfound_targets else "(none)")
        + "\n\nAssess sufficiency now."
    )

    return await structured_call(
        model=settings.critic_model,
        system_prompt=CRITIC_SYSTEM,
        user_content=user_content,
        output_model=CriticOutput,
    )
