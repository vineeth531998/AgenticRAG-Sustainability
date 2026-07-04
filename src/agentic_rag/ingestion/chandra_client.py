"""Client for the Chandra OCR endpoint on Modal — per-page concurrent fan-out.

Why per-page: Chandra runs on vLLM, which gets its throughput from continuous
batching. To hit the published ~1.44 pages/s on H100, we need ~96 concurrent
in-flight page requests so vLLM has a full batch to schedule. Sending one
whole PDF as one request collapses the batch to size 1 and throughput tanks.

Pipeline:
    PDF ─► pypdfium2 renders pages to PIL images locally
        ─► asyncio.gather over all pages with Semaphore(CHANDRA_PAGE_CONCURRENCY)
        ─► each task: POST /ocr_page with page image + page number
        ─► reassemble per-page chunk lists into one list, ordered by page
        ─► post-hoc: propagate `section` forward (last heading sticks until the
                     next heading, possibly across pages)
"""
from __future__ import annotations

import asyncio
import io
import uuid
from pathlib import Path

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentic_rag.ingestion.vlm_extractor import (
    DECORATIVE_TOKEN,
    crop_bbox_from_png,
    extract_image_content,
)
from agentic_rag.schemas import Chunk, ChunkMetadata, Framework, TableData
from config.settings import settings


class ChandraOCRError(RuntimeError):
    pass


class ChandraOCRClient:
    def __init__(
        self,
        endpoint_url: str,
        *,
        timeout_s: int,
        page_concurrency: int,
        render_scale: float = 2.0,
    ):
        # endpoint_url is expected to point at the base URL of the Modal app
        # (e.g. https://...modal.run/ocr_page). We POST the same URL per page.
        self.endpoint_url = endpoint_url
        self.timeout_s = timeout_s
        self.page_concurrency = page_concurrency
        self.render_scale = render_scale

    async def ocr(
        self,
        pdf_path: str | Path,
        report_id: str,
        *,
        company: str | None = None,
        report_year: int | None = None,
        framework: Framework | None = None,
    ) -> list[Chunk]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        # 1. Render every page to a PNG (in memory). pypdfium2 is fast; ~0.1s/page.
        page_images = await asyncio.to_thread(_render_pages, pdf_path, self.render_scale)
        if not page_images:
            raise ChandraOCRError(f"No pages rendered from {pdf_path}")

        # 2. Fan out per-page OCR requests under a semaphore.
        sem = asyncio.Semaphore(self.page_concurrency)

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:

            async def _do_page(idx_one_based: int, png_bytes: bytes) -> list[dict]:
                async with sem:
                    return await self._post_page(client, idx_one_based, png_bytes, report_id)

            tasks = [_do_page(i, b) for i, b in enumerate(page_images, start=1)]
            per_page_chunks = await asyncio.gather(*tasks)

        # 3. VLM extraction pass over Image chunks (charts, infographics,
        #    dashboard cards). Chandra tags these but doesn't OCR them; we
        #    crop the page image at their bbox and ask a vision model to
        #    transcribe the data. Skipped entirely when disabled.
        if settings.enable_vlm_extraction:
            await _enrich_image_chunks(per_page_chunks, page_images)

        # 4. Flatten in page order, then propagate `section` forward.
        flat = [c for page_chunks in per_page_chunks for c in page_chunks]
        flat = _propagate_sections(flat)

        # 4. Map to typed Chunk objects with full metadata.
        out: list[Chunk] = []
        for idx, c in enumerate(flat):
            table_data = None
            if c.get("is_table") and c.get("table"):
                t = c["table"]
                table_data = TableData(
                    headers=t.get("headers", []),
                    rows=t.get("rows", []),
                    caption=t.get("caption"),
                )

            pages = c.get("pages") or [int(c["page"])]
            meta = ChunkMetadata(
                report_id=report_id,
                page=int(c["page"]),
                pages=[int(p) for p in pages],
                chunk_index=idx,
                section=c.get("section"),
                is_table=bool(c.get("is_table", False)),
                table_data=table_data,
                is_infographic=bool(c.get("is_infographic", False)),
                company=company,
                report_year=report_year,
                framework=framework,
                bbox=c.get("bbox"),
                label=c.get("label"),
            )
            out.append(
                Chunk(
                    chunk_id=f"{report_id}::{idx}::{uuid.uuid4().hex[:8]}",
                    text=c["text"],
                    metadata=meta,
                )
            )
        return out

    async def _post_page(
        self,
        client: httpx.AsyncClient,
        page_number: int,
        png_bytes: bytes,
        report_id: str,
    ) -> list[dict]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=2, max=20),
            retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                files = {"image": (f"page_{page_number}.png", png_bytes, "image/png")}
                data = {"page_number": str(page_number), "report_id": report_id}
                resp = await client.post(self.endpoint_url, files=files, data=data)
                if not resp.is_success:
                    # Surface the server-side detail before raising. Without this
                    # the user only sees "500 Internal Server Error" with no clue
                    # what FastAPI raised on the Modal side.
                    try:
                        body = resp.json()
                        detail = body.get("detail") or str(body)[:800]
                    except Exception:  # noqa: BLE001
                        detail = (resp.text or "")[:800]
                    raise httpx.HTTPStatusError(
                        f"Chandra endpoint {resp.status_code} on page {page_number}: {detail}",
                        request=resp.request,
                        response=resp,
                    )
                payload = resp.json()

        return payload.get("chunks", [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_pages(pdf_path: Path, scale: float) -> list[bytes]:
    """Render every page of a PDF to PNG bytes. Sync — call via asyncio.to_thread."""
    import pypdfium2 as pdfium

    out: list[bytes] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for page in pdf:
            pil = page.render(scale=scale).to_pil()
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            out.append(buf.getvalue())
    finally:
        pdf.close()
    return out


def _propagate_sections(chunks: list[dict]) -> list[dict]:
    """Fill `section` on chunks that started a page before any Section-Header.

    The server's per-page assembler sets `section` for each chunk based on the
    Section-Header seen earlier on THAT page. But when a section spans pages,
    the next page often starts mid-section (no Section-Header at the top), so
    chunks at the start of that page come back with section=None. Walk in
    document order and inherit the section from the most recent set value.
    """
    current: str | None = None
    for c in chunks:
        if c.get("section"):
            current = c["section"]
        elif current is not None:
            c["section"] = current
    return chunks


async def _enrich_image_chunks(
    per_page_chunks: list[list[dict]],
    page_images: list[bytes],
) -> None:
    """Run bounded-concurrent VLM extraction on every Image chunk in place.

    For each Image chunk (label == "Image"):
      1. Crop the corresponding page PNG at the chunk's bbox
      2. Send the crop to Fireworks VLM with the extract prompt
      3. If the VLM returns the DECORATIVE token, mark the chunk for drop
      4. Otherwise, replace the placeholder text with the VLM's extraction
         AND flag chunk["is_infographic"] = True so query-time routing finds it

    Mutates `per_page_chunks` in place (removes dropped chunks). Never raises
    on VLM failure — falls back to the alt-text placeholder.

    Prints a per-image outcome line and a final summary. Without this, VLM
    failures during ingestion (finish_reason='length', rate limits, network
    errors) are invisible until you find yourself with an index full of
    zombie image chunks that no query-time wing can rescue.
    """
    sem = asyncio.Semaphore(settings.vlm_concurrency)
    # Outcome counters keyed by category — printed as a summary at the end.
    counters = {
        "bad_bbox": 0,
        "bad_page_idx": 0,
        "decorative": 0,
        "vlm_empty_or_error": 0,
        "success": 0,
    }

    async def _process(page_idx_zero_based: int, chunk: dict) -> None:
        page_1based = page_idx_zero_based + 1
        bbox = chunk.get("bbox")
        if not bbox or len(bbox) != 4:
            chunk["_drop"] = True
            counters["bad_bbox"] += 1
            print(f"[vlm-ingest page={page_1based}] SKIP bad_bbox={bbox!r}", flush=True)
            return
        if not (0 <= page_idx_zero_based < len(page_images)):
            chunk["_drop"] = True
            counters["bad_page_idx"] += 1
            print(
                f"[vlm-ingest page={page_1based}] SKIP bad_page_idx "
                f"(only {len(page_images)} page images rendered)",
                flush=True,
            )
            return

        # Diagnostic tag threaded into extract_image_content so its own
        # error prints include the page + rough area.
        x0, y0, x1, y1 = bbox
        area = max(0.0, (x1 - x0) * (y1 - y0))
        tag = f"page={page_1based} area={int(area)}"

        crop_png = crop_bbox_from_png(page_images[page_idx_zero_based], bbox)
        async with sem:
            vlm_text = await extract_image_content(crop_png, diagnostic_tag=tag)

        if not vlm_text:
            # VLM failed / empty — chunk stays but WITHOUT is_infographic=True.
            # It becomes a zombie chunk (placeholder text, useless for retrieval,
            # invisible to the query-time image wing). This is bad state. The
            # error was already printed inside extract_image_content.
            counters["vlm_empty_or_error"] += 1
            print(
                f"[vlm-ingest {tag}] KEPT_AS_ZOMBIE (no VLM description → "
                f"is_infographic will remain False → image wing cannot rescue "
                f"at query time)",
                flush=True,
            )
            return
        if vlm_text.strip().upper() == DECORATIVE_TOKEN:
            chunk["_drop"] = True
            counters["decorative"] += 1
            print(f"[vlm-ingest {tag}] DROPPED decorative", flush=True)
            return
        chunk["text"] = vlm_text
        # The VLM confirmed non-decorative and produced the 3-part
        # DESCRIPTION / METRIC NAMES / DATA HINTS description. Mark the
        # chunk so downstream routing can find it via metadata.is_infographic
        # without depending on the label string. Currently is_infographic is
        # informational only — the query-time composite wing reads is_table=False
        # (which includes infographics) so no separate routing is needed.
        chunk["is_infographic"] = True
        counters["success"] += 1
        print(
            f"[vlm-ingest {tag}] OK chars={len(vlm_text)} "
            f"→ marked is_infographic=True",
            flush=True,
        )

    tasks: list = []
    for page_idx, page_chunks in enumerate(per_page_chunks):
        for chunk in page_chunks:
            if chunk.get("label") == "Image":
                tasks.append(_process(page_idx, chunk))

    if not tasks:
        print("[vlm-ingest] SUMMARY no Image blocks emitted by Chandra", flush=True)
        return

    print(
        f"[vlm-ingest] START processing {len(tasks)} Image blocks "
        f"(vlm_model={settings.vlm_model}, vlm_max_tokens={settings.vlm_max_tokens}, "
        f"concurrency={settings.vlm_concurrency})",
        flush=True,
    )
    await asyncio.gather(*tasks)

    # End-of-ingest summary — the ONE line the user should always see.
    kept = counters["success"]
    dropped = counters["decorative"]
    zombies = counters["vlm_empty_or_error"]
    skipped = counters["bad_bbox"] + counters["bad_page_idx"]
    total = kept + dropped + zombies + skipped
    print(
        f"[vlm-ingest] SUMMARY total={total}  "
        f"success={kept} (is_infographic=True → image wing can use these)  "
        f"decorative_dropped={dropped}  "
        f"zombies={zombies} (VLM failed → is_infographic=False → INVISIBLE to image wing)  "
        f"skipped_bad_metadata={skipped}",
        flush=True,
    )
    if zombies > 0:
        print(
            f"[vlm-ingest] WARN {zombies} chunk(s) failed VLM extraction and are "
            f"now zombies in the index. Common causes: VLM_MAX_TOKENS too low "
            f"for thinking model (see EMPTY_CONTENT lines above), Fireworks rate "
            f"limit (see EXCEPTION lines above with 429), or model doesn't accept "
            f"multimodal input. Re-ingest after fixing the root cause.",
            flush=True,
        )

    # Filter out chunks marked as decorative after the VLM pass.
    for i, page_chunks in enumerate(per_page_chunks):
        per_page_chunks[i] = [c for c in page_chunks if not c.get("_drop")]


