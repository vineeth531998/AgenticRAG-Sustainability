"""Ingestion pipeline: PDF → Chandra OCR → Qwen3 dense + BM25 sparse → Qdrant upsert.

One report at a time (matches the UX: user uploads, then queries). If the same
report_id is ingested twice, the old chunks are deleted first so re-ingestion
is idempotent.

We also persist the source PDF to `data/reports/<report_id>.pdf` so the
Streamlit UI can render cited pages later for traceability.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from agentic_rag.embeddings.dense import ChunkVectors, embed_dense
from agentic_rag.ingestion.chandra_client import ChandraOCRClient
from agentic_rag.ingestion.metadata import compute_metadata, save_metadata
from agentic_rag.retrieval.bm25 import to_sparse
from agentic_rag.schemas import Framework
from agentic_rag.vectordb.qdrant import QdrantStore
from config.settings import settings

PDF_STORAGE_DIR = Path("data/reports")


def stored_pdf_path(report_id: str) -> Path:
    return PDF_STORAGE_DIR / f"{report_id}.pdf"


async def ingest_report(
    pdf_path: str | Path,
    report_id: str,
    *,
    company: str | None = None,
    report_year: int | None = None,
    framework: Framework | None = None,
    replace_existing: bool = True,
    store: QdrantStore | None = None,
) -> int:
    """Ingest a single report. Returns the number of chunks indexed.

    Pass `store=` to share a QdrantStore across calls (required in local-file
    Qdrant mode where only one client can hold the directory lock).
    """
    ocr = ChandraOCRClient(
        endpoint_url=settings.chandra_ocr_url,
        timeout_s=settings.chandra_ocr_timeout_s,
        page_concurrency=settings.chandra_page_concurrency,
    )
    if store is None:
        store = QdrantStore()
    await store.ensure_collection()

    if replace_existing:
        await store.delete_report(report_id)

    # Persist the source PDF for later page previews. Skips the copy when
    # the source is already at the target path (CLI may pass the canonical
    # location directly).
    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    target_pdf = stored_pdf_path(report_id)
    pdf_path_obj = Path(pdf_path)
    if pdf_path_obj.resolve() != target_pdf.resolve():
        shutil.copy2(pdf_path_obj, target_pdf)

    chunks = await ocr.ocr(
        pdf_path,
        report_id=report_id,
        company=company,
        report_year=report_year,
        framework=framework,
    )
    if not chunks:
        return 0

    # Dense via Fireworks embeddings (batched); sparse via local BM25 tokenizer.
    vectors: list[ChunkVectors] = []
    for i in range(0, len(chunks), 32):
        batch = chunks[i : i + 32]
        batch_texts = [c.text for c in batch]
        dense_vecs = await embed_dense(batch_texts)
        for c, dv in zip(batch, dense_vecs, strict=True):
            vectors.append(ChunkVectors(dense=dv, sparse=to_sparse(c.text)))

    await store.upsert_chunks(chunks, vectors)

    # Compute and persist content-distribution metadata for content-aware
    # planning at query time (see agents/planner.py + prompts.py). Failures
    # here MUST NOT break ingestion — metadata is a soft hint, not a hard
    # dependency.
    try:
        meta = compute_metadata(report_id, chunks)
        meta_path = save_metadata(meta)
        print(
            f"[ingest] SUMMARY report={report_id!r} chunks={meta.total_chunks} "
            f"tables={meta.table_chunks} composites={meta.composite_chunks} "
            f"infographics={meta.infographic_chunks} "
            f"dominant={meta.dominant_content_type} → wrote {meta_path}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[ingest] WARN metadata computation failed for {report_id!r}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

    return len(chunks)
