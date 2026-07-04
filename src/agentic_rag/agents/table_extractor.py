"""Table Extractor agent — pulls specific cells out of retrieved table chunks.

This is what makes pointed queries like "what was wastewater discharge in
FY2023?" reliable. After retrieval narrows the universe to candidate tables,
this agent reads the STRUCTURED JSON (`headers` + `rows`) — not the markdown —
and returns typed TableValue objects with row/column/value/unit provenance.

Why a separate agent (rather than letting the synthesizer do it):
- The task is mechanical (find row, find column, read cell) and benefits from
  a focused prompt that allows no narrative wiggle room.
- TableValue is a strong type the synthesizer can cite verbatim — no markdown
  parsing in the synth path.
- It surfaces 'unfound' explicitly, which the critic uses to decide whether
  to issue follow-up subqueries.
"""
from __future__ import annotations

import json

from agentic_rag.agents.prompts import TABLE_EXTRACTOR_SYSTEM
from agentic_rag.llm import structured_call
from agentic_rag.schemas import RetrievedChunk, TableExtractorOutput
from config.settings import settings


def _render_tables(chunks: list[RetrievedChunk]) -> str:
    """Render each table chunk as a chunk_id-tagged JSON block.

    Only chunks where metadata.is_table and table_data is populated are emitted;
    everything else is filtered out by the orchestrator before this is called.
    """
    blocks: list[str] = []
    for r in chunks:
        m = r.chunk.metadata
        if not (m.is_table and m.table_data):
            continue
        td = m.table_data.model_dump()
        head = (
            f"chunk_id: {r.chunk.chunk_id}\n"
            f"report_id: {m.report_id} | page: {m.page} | "
            f"section: {m.section or '-'} | company: {m.company or '-'} | "
            f"year: {m.report_year or '-'}\n"
            f"caption: {td.get('caption') or '-'}"
        )
        body = json.dumps(
            {"headers": td["headers"], "rows": td["rows"]},
            indent=2,
            ensure_ascii=False,
        )
        blocks.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(blocks)


async def extract_table_values(
    target_cells: list[str],
    table_chunks: list[RetrievedChunk],
) -> TableExtractorOutput:
    """Given target cell descriptions + candidate table chunks, return TableValues."""
    if not target_cells:
        return TableExtractorOutput(values=[], unfound=[])

    table_blob = _render_tables(table_chunks)
    if not table_blob:
        # No usable tables — every target is unfound by definition.
        return TableExtractorOutput(values=[], unfound=list(target_cells))

    user_content = (
        "target_cells:\n"
        + "\n".join(f"- {t}" for t in target_cells)
        + "\n\nCandidate tables:\n\n"
        + table_blob
        + "\n\nExtract values for each target_cell now."
    )

    return await structured_call(
        model=settings.synthesizer_model,
        system_prompt=TABLE_EXTRACTOR_SYSTEM,
        user_content=user_content,
        output_model=TableExtractorOutput,
    )
