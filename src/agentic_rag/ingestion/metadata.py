"""Per-report content-distribution metadata.

Computed at the END of ingestion once all chunks are indexed, written to
`data/reports/<report_id>.metadata.json`, loaded at query time and passed
to the planner so it can adapt decomposition to report shape.

Why this exists: our three-wing pipeline (table / composite / narrative) is
best-suited to different query types depending on how content is
distributed in the source. A BRSR filing is mostly structured tables — a
KPI question maps cleanly to `factual_lookup` + tight target_cells. An
Integrated Report is mostly prose — the composite wing carries most
answers, and supplementary narrative subqueries add value for "how / why"
questions. Without metadata the planner is guessing.

Storage: one JSON file per report at `data/reports/<report_id>.metadata.json`
(matches the PDF storage path convention in `pipeline.py`). Per-report
files avoid write contention during concurrent ingestion and make it
trivial to inspect / delete individual reports' metadata.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from agentic_rag.schemas import Chunk, ReportMetadata


REPORTS_DIR = Path("data/reports")


def metadata_path(report_id: str) -> Path:
    """Return the on-disk metadata file path for a report_id."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR / f"{report_id}.metadata.json"


def compute_metadata(report_id: str, chunks: Iterable[Chunk]) -> ReportMetadata:
    """Tally chunk types and derive the dominant content type.

    Rules for dominant_content_type:
      • tabular    if table_chunks / total > 0.5
      • narrative  if composite_chunks / total > 0.6
      • mixed      otherwise (including the tabular < 0.5 AND composite < 0.6 case)

    The thresholds are intentionally different — 0.5 for tables, 0.6 for
    composites — because ALL infographic-derived text lands in composite
    chunks too. A 50/50 table/composite split still puts most FACTUAL
    KPI values in tables, so we don't want to call it "narrative"
    prematurely.
    """
    total = 0
    tables = 0
    composites = 0
    infographics = 0
    for c in chunks:
        total += 1
        m = c.metadata
        if m.is_table:
            tables += 1
        if m.label == "Composite":
            composites += 1
        if m.is_infographic:
            infographics += 1

    if total == 0:
        dominant: str = "mixed"
    else:
        if tables / total > 0.5:
            dominant = "tabular"
        elif composites / total > 0.6:
            dominant = "narrative"
        else:
            dominant = "mixed"

    return ReportMetadata(
        report_id=report_id,
        total_chunks=total,
        table_chunks=tables,
        composite_chunks=composites,
        infographic_chunks=infographics,
        dominant_content_type=dominant,  # type: ignore[arg-type]
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def save_metadata(meta: ReportMetadata) -> Path:
    """Serialize to `data/reports/<report_id>.metadata.json`. Overwrites any prior."""
    path = metadata_path(meta.report_id)
    path.write_text(json.dumps(meta.model_dump(), indent=2))
    return path


def load_metadata(report_id: str) -> ReportMetadata | None:
    """Load a report's metadata. Returns None if the file doesn't exist.

    Callers must handle None (report ingested before this feature, deleted
    metadata file, etc.). Downstream consumers should degrade to
    default behavior when metadata is missing.
    """
    path = metadata_path(report_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        return ReportMetadata.model_validate(payload)
    except Exception as e:  # noqa: BLE001
        # Corrupt / schema-mismatched file — return None so callers degrade
        # gracefully. Don't crash the whole query on a stale metadata file.
        print(
            f"[metadata] WARN could not read {path}: {type(e).__name__}: {e}",
            flush=True,
        )
        return None
