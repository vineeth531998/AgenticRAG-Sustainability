"""Pydantic schemas for chunks, citations, agent IO, and orchestrator state.

These are the contracts that flow between OCR → vectordb → retrieval → agents.
Keep them stable; everything else can change behind them.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---- Sustainability domain enums ----

class Framework(str, Enum):
    BRSR = "BRSR"
    GRI = "GRI"
    SASB = "SASB"
    TCFD = "TCFD"
    IR = "IR"            # Integrated Reporting
    CDP = "CDP"
    OTHER = "OTHER"


# ---- Chunk + metadata (what we store in Qdrant payload) ----

class TableData(BaseModel):
    """Normalized table representation stored alongside markdown."""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: Optional[str] = None


class ChunkMetadata(BaseModel):
    """Payload stored in Qdrant alongside the dense+sparse vectors.

    Every chunk MUST have report_id + page so we can cite it. Other fields are
    extracted at ingestion time when available.
    """
    report_id: str
    page: int                                        # primary page (first the chunk appears on)
    pages: list[int] = Field(default_factory=list)   # full page span for citations
    chunk_index: int                                 # ordinal within the report
    section: Optional[str] = None                    # section/heading path
    is_table: bool = False
    table_data: Optional[TableData] = None           # populated iff is_table
    is_infographic: bool = False                     # True for Chandra Image blocks
    #   the ingestion VLM confirmed as non-decorative (charts, dashboard cards,
    #   sankey diagrams, KPI callouts). Text field carries a 3-part description
    #   (DESCRIPTION / METRIC NAMES / DATA HINTS) used for retrieval; at query
    #   time these chunks flow through the composite extraction wing along with
    #   other non-table chunks.
    company: Optional[str] = None
    report_year: Optional[int] = None
    framework: Optional[Framework] = None
    bbox: Optional[list[float]] = None               # union bbox on the primary page
    label: Optional[str] = None                      # post-assembly label
    #   "Composite" — assembled from one or more Text/List-Group blocks
    #   "Table"     — single-table chunk with structured TableData
    #   "Image"     — visual chunk (is_infographic=True after VLM description)
    #   (Page-Header / Page-Footer are dropped at ingestion; Section-Header is
    #   never its own chunk — it becomes the `section` field on the next chunk.)


class Chunk(BaseModel):
    """A retrievable unit. `text` is markdown (incl. table flattened to MD).

    For tables, `text` holds the markdown view (good for embeddings + LLM input)
    and `metadata.table_data` holds the structured JSON view (good for KPI lookup).
    """
    chunk_id: str
    text: str
    metadata: ChunkMetadata


# ---- Retrieval result ----

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float                       # final score (post-rerank if reranked)
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None


# ---- Planner agent IO ----

class RetrievalFilter(BaseModel):
    """Hard filters the planner can apply at retrieval time.

    Anything left None is unconstrained. The orchestrator AND-combines these
    with the report_id scope.

    NOTE: `is_table_only` was removed. Under the three-wing architecture the
    composite wing reads non-table (prose/list/infographic-text) chunks in
    parallel with the table wing, so filtering retrieval to is_table=True
    only starves the composite wing without any upside.
    """
    company: Optional[str] = None
    report_year: Optional[int] = None
    framework: Optional[Framework] = None
    section_contains: Optional[str] = None


QueryType = Literal["narrative", "factual_lookup", "comparison"]


class Subquery(BaseModel):
    """A single self-contained retrieval task derived from the user query.

    Keyword fields are first-class for sustainability QA — semantic similarity
    alone routinely misses chunks that contain the right number but use slightly
    different prose. `must_phrases` is a HARD filter (chunk must contain every
    listed phrase), `keywords` is a SOFT signal (boosts the sparse leg).
    """
    query: str = Field(..., description="Self-contained natural language query for embedding (dense leg)")

    must_phrases: list[str] = Field(
        default_factory=list,
        description=(
            "Phrases the chunk MUST contain. Applied as Qdrant MatchText filters before vector search. "
            "Use sparingly and only when the planner is confident the report uses these exact terms. "
            "Each entry can be multi-word; tokens are AND-ed within an entry."
        ),
    )

    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Soft keyword signal appended to the sparse-leg query text. Less strict than must_phrases. "
            "KPI names, framework codes, units, year tokens (e.g. 'Scope 1', 'tCO2e', 'FY2023')."
        ),
    )

    hyde_doc: Optional[str] = Field(
        None,
        description="Hypothetical answer paragraph for HyDE-style dense retrieval. Optional.",
    )

    filters: RetrievalFilter = Field(default_factory=RetrievalFilter)

    query_type: QueryType = Field(
        "narrative",
        description=(
            "How retrieval and downstream extraction should treat this subquery. "
            "'factual_lookup' triggers is_table-only retrieval + the Table Extractor agent. "
            "'comparison' signals the orchestrator that multiple sides must be present before sufficiency."
        ),
    )

    target_cells: list[str] = Field(
        default_factory=list,
        description=(
            "Only meaningful when query_type=='factual_lookup'. Human descriptions of the cells to extract, "
            "e.g. 'wastewater discharge for FY2023', 'Scope 1 emissions for FY2023 in tCO2e'."
        ),
    )

    rationale: str = Field(..., description="Why this subquery was generated")


class PlannerOutput(BaseModel):
    subqueries: list[Subquery] = Field(..., min_length=1)
    reasoning: str = Field(..., description="High-level decomposition rationale")


# ---- Critic agent IO ----

class CriticOutput(BaseModel):
    sufficient: bool
    missing_info: Optional[str] = Field(
        None,
        description="What's still missing to fully answer the user query. None iff sufficient=true.",
    )
    follow_up_subqueries: list[Subquery] = Field(
        default_factory=list,
        description="New subqueries to fetch the missing information. Empty iff sufficient=true.",
    )


# ---- Synthesizer agent IO ----

class Citation(BaseModel):
    chunk_id: str
    report_id: str
    page: int
    section: Optional[str] = None
    quote: Optional[str] = None       # short excerpt the claim rests on


class SynthesizerOutput(BaseModel):
    answer: str = Field(..., description="Final answer with inline [^chunk_id] citation markers")
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    caveats: Optional[str] = None
    answer_available: bool = Field(
        True,
        description=(
            "True when the evidence directly supports at least one substantive claim in the answer. "
            "False when the retrieved chunks do not contain the information needed — in that case "
            "the answer field must be a short 'Not available in the provided document' statement, "
            "citations must be empty, and confidence must be 'low'."
        ),
    )


# ---- Table extractor IO ----

class TableValue(BaseModel):
    """A single value extracted from a chunk, with full provenance.

    UNIFIED SCHEMA — produced by BOTH text extractors (table + composite) AND
    VLM extractors (table verify + composite verify). Downstream merge logic
    compares two `TableValue`s (one from text, one from VLM) on the same
    fields, symmetrically. The `source` field tracks which extractor
    produced this reading, and `found` distinguishes "extracted a value"
    from "looked but couldn't find it in this chunk."

    Fields other than provenance + `found` are Optional because a
    not-found reading carries only the `note` explaining why. When
    `found=True` an extractor is expected to fill row_label + column_label
    + value at minimum.
    """
    # ── Provenance (filled by the caller, not by the extractor LLM/VLM) ────
    chunk_id: str = Field(..., description="The chunk this reading was extracted from")
    target_description: str = Field(..., description="The target_cell entry this reading answers")
    source: Literal["text", "vlm", "merged"] = Field(
        "text",
        description=(
            "Which extractor produced this reading. 'text' = LLM reading "
            "Chandra's OCR text; 'vlm' = VLM reading pixels independently; "
            "'merged' = result of merging one text + one vlm reading."
        ),
    )

    # ── Did the extractor find the target in this chunk? ───────────────────
    found: bool = Field(
        True,
        description=(
            "True when the extractor produced a reading. False means the "
            "extractor looked but could not answer the target from this "
            "chunk — see `note` for why. Text extractors typically use "
            "batch-level `unfound: list[str]` and always set found=True on "
            "the values they DO return; VLM extractors set found=False "
            "in-line when they can't answer."
        ),
    )

    # ── The reading — Optional so a not-found reading is representable ─────
    row_label: Optional[str] = Field(
        None, description="Verbatim row label / semantic identifier"
    )
    column_label: Optional[str] = Field(
        None, description="Verbatim column label / period / breakdown identifier"
    )
    value: Optional[str] = Field(None, description="The value as it appears in the source")
    unit: Optional[str] = Field(
        None, description="Unit if present in header / row label / cell / adjacent text"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        "low",
        description=(
            "Confidence in the reading. Defaults to 'low' for not-found "
            "cases; extractors set explicitly when found=True."
        ),
    )
    note: Optional[str] = Field(
        None,
        description=(
            "Optional explanation. Required when confidence < high, when "
            "found=False, or when the extractor detected a semantic "
            "ambiguity (e.g. segment-vs-combined mismatch)."
        ),
    )


class TableExtractorOutput(BaseModel):
    values: list[TableValue] = Field(default_factory=list)
    unfound: list[str] = Field(
        default_factory=list,
        description="target_cells entries that could not be answered from the provided tables",
    )


class TableVLMVerification(BaseModel):
    """VLM's opinion on a specific cell after seeing the cropped table image.

    Used as a fallback when the text/markdown-based TableExtractor returned a
    non-high-confidence value. `found=False` means the VLM couldn't locate the
    cell in the image; in that case the other fields may be null.
    """
    found: bool = Field(
        ..., description="True if a matching cell was located in the cropped table image"
    )
    row_label: Optional[str] = None
    column_label: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    confidence: Optional[Literal["high", "medium", "low"]] = None
    note: Optional[str] = Field(
        None,
        description="Short explanation, especially when confidence < high",
    )


# ---- Synthesizer input: context around unfound targets ---------------------

class UnfoundVLMEvidence(BaseModel):
    """One VLM's semantic explanation of why a target is unfound in a chunk.

    Produced by the extract-from-unfound VLM path (both table and composite
    wings) when the VLM returned `found=False` with a non-empty note. The
    note typically explains what IS disclosed in the chunk even though the
    target isn't — e.g. "Table shows Total Learning Hours split by
    Male/Female; no combined figure disclosed."
    """
    chunk_id: str = Field(..., description="Chunk the VLM was reading")
    note: str = Field(..., description="VLM's semantic explanation")


class UnfoundTargetContext(BaseModel):
    """Full context for one unfound target, passed to the synthesizer.

    Aggregates every VLM's note tied to this target across both wings
    (table + composite extract-from-unfound paths). Feeds the synthesizer's
    "UNFOUND IS AUTHORITATIVE" rule — the synthesizer treats these notes
    as its source of truth for what the report actually discloses about
    the target, and must NOT re-derive the target's value from raw chunk
    markdown.
    """
    target: str = Field(..., description="The target_cell that was unfound")
    vlm_evidence: list[UnfoundVLMEvidence] = Field(
        default_factory=list,
        description="Per-chunk VLM explanations. Empty means neither extractor gave a semantic reason.",
    )


# ---- Report metadata (content-aware planning) -----------------------------

class ReportMetadata(BaseModel):
    """Per-report content distribution, computed at ingestion end.

    Written to `data/reports/<report_id>.metadata.json` after all chunks
    are indexed. Loaded at query time and passed to the planner so
    decomposition can adapt to report shape (BRSR = tabular, Integrated
    Report = narrative-heavy, etc.).
    """
    report_id: str
    total_chunks: int
    table_chunks: int         # metadata.is_table == True
    composite_chunks: int     # metadata.label == "Composite"
    infographic_chunks: int   # metadata.is_infographic == True
    dominant_content_type: Literal["tabular", "narrative", "mixed"]
    created_at: str           # ISO-format timestamp
