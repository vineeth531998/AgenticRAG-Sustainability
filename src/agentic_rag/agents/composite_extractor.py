"""Composite Extractor — pulls specific target cells out of prose / list chunks.

Companion to `table_extractor.py`. Where the Table Extractor reads structured
markdown tables, this reads Chandra's `Composite` chunks — prose paragraphs,
list groups, and infographic-transcribed text. Chandra puts most report
content into Composite chunks (KPI callouts, dashboard cards, sankey labels,
board rosters, policy statements). So for a huge fraction of "list of X" or
"what is X" questions, the answer lives in a Composite chunk, not a table.

Same output contract as `table_extractor` (`TableExtractorOutput`) so the
orchestrator can concatenate table + composite results into a single
`table_values` list without downstream code caring which wing produced which
value. The `note` field on each `TableValue` distinguishes provenance:
    • "composite-extracted from prose" — this extractor
    • "VLM-verified"                    — table extractor + VLM verify
    • (no marker)                       — pure table extractor

Row / column labels for a non-tabular chunk are SEMANTIC IDENTIFIERS:
    • row_label    = the metric or entity being measured
    • column_label = the period, dimension, or breakdown
    • value        = the actual value as it appears in the prose (verbatim)

For a director-roster question the answer TableValue might look like:
    row_label    = "Independent Directors"
    column_label = "Names as of March 31, 2025"
    value        = "D. Sundaram | Michael Gibbs | Bobby Parikh | Chitra Nayak
                    | Govind Iyer | Helene Auriol Potier | Nitin Paranjpe"

Single LLM call per subquery — the model receives ALL target_cells and ALL
candidate chunks at once (bounded by settings.composite_extract_top_n) and
returns a batched output. No VLM, no per-chunk fanout.
"""
from __future__ import annotations

from agentic_rag.agents.prompts import COMPOSITE_EXTRACTOR_SYSTEM
from agentic_rag.llm import structured_call
from agentic_rag.schemas import RetrievedChunk, TableExtractorOutput
from config.settings import settings


def _render_composite_chunks(chunks: list[RetrievedChunk]) -> str:
    """Render each candidate chunk with enough metadata for the LLM to cite it.

    Order matches the input list (which is reranker-sorted highest first).
    Each block carries chunk_id + section + page + label + a short header so
    the extractor knows which chunk to attribute a found value to.
    """
    blocks: list[str] = []
    for r in chunks:
        m = r.chunk.metadata
        head = (
            f"chunk_id: {r.chunk.chunk_id}\n"
            f"report_id: {m.report_id} | page: {m.page} | "
            f"section: {m.section or '-'} | label: {m.label or '-'} | "
            f"company: {m.company or '-'} | year: {m.report_year or '-'}"
        )
        body = r.chunk.text
        blocks.append(f"{head}\n\n{body}")
    return "\n\n---\n\n".join(blocks)


async def extract_composite_values(
    target_cells: list[str],
    composite_chunks: list[RetrievedChunk],
) -> TableExtractorOutput:
    """Given target cell descriptions + top-N composite chunks, return TableValues.

    Symmetric with `extract_table_values` — same input pattern, same output
    schema. The orchestrator can therefore run both extractors under a single
    `asyncio.gather` and merge their outputs by simple list concatenation.
    """
    if not target_cells:
        return TableExtractorOutput(values=[], unfound=[])
    if not composite_chunks:
        return TableExtractorOutput(values=[], unfound=list(target_cells))

    chunks_blob = _render_composite_chunks(composite_chunks)

    user_content = (
        "target_cells:\n"
        + "\n".join(f"- {t}" for t in target_cells)
        + "\n\nCandidate composite chunks (prose / lists / infographic text):\n\n"
        + chunks_blob
        + "\n\nExtract values for each target_cell now."
    )

    return await structured_call(
        model=settings.synthesizer_model,
        system_prompt=COMPOSITE_EXTRACTOR_SYSTEM,
        user_content=user_content,
        output_model=TableExtractorOutput,
    )
