"""Probe Chandra OCR on specific pages of a PDF and inspect what it detects.

Use this when the ingest summary shows a suspicious pattern — e.g. 89/99
Image blocks called "decorative" — and you want to see, per page:

  • what block labels Chandra emitted (was it even tagged as Image?)
  • the exact bbox of each Image block on the page
  • the cropped Image region as a PNG you can eyeball
  • (optional) what the VLM says when you ask it "informative or decorative
    AND WHY" instead of the production one-word prompt

Everything lands on disk under --out so you can compare page-by-page.

Usage:

    uv run python scripts/probe_chandra.py \\
        --pdf data/reports/Infosys-Data.pdf \\
        --pages 21,22,23 \\
        --out data/probe/infosys_directors \\
        --vlm            # optional: run diagnostic VLM on each Image crop

    # or a page range:
    uv run python scripts/probe_chandra.py \\
        --pdf data/reports/PersistentSystem.pdf \\
        --pages 8-12 \\
        --out data/probe/persistent_esg

The probe never touches Qdrant and never runs the assembler — it just calls
the same /ocr_page endpoint your ingest hits, dumps the raw response, and
optionally runs a diagnostic-mode VLM pass on Image crops. Safe to run
against any deployed Chandra endpoint without side-effects on your index.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import sys
from pathlib import Path

import httpx
import pypdfium2 as pdfium
from PIL import Image
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Path shim so `uv run python scripts/probe_chandra.py` picks up src/ + config/
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from agentic_rag.ingestion.vlm_extractor import crop_bbox_from_png  # noqa: E402
from agentic_rag.llm import get_client  # noqa: E402
from config.settings import settings  # noqa: E402


# ── Diagnostic VLM prompt (deliberately DIFFERENT from the production one) ──
# The production prompt asks for a one-word DECORATIVE token OR the 3-part
# description. Here we force the VLM to also EXPLAIN itself so we can see
# WHY it's dropping things. Never used in production — probe-only.
DIAGNOSTIC_VLM_PROMPT = (
    "You are diagnosing an infographic-vs-decorative decision inside a "
    "sustainability report. Look at this cropped region and answer these "
    "four questions in the EXACT format shown, one per line, no extra text:\n\n"
    "TYPE: <what kind of visual — bar chart / pie / sankey / dashboard card / "
    "KPI callout / diagram / photo-of-board / org chart / logo / banner icon / "
    "section divider / page furniture / other>\n"
    "INFO: <YES or NO> — <one clause: what informational content is visible; "
    "if NO, what's actually in the region>\n"
    "CLASSIFICATION: <INFORMATIVE or DECORATIVE>\n"
    "REASON: <one sentence justifying the classification>\n"
)


RENDER_SCALE = 2.0  # match ChandraOCRClient default


# ── Page-range parsing (accepts "21,22,25-30,50") ───────────────────────────
def _parse_pages(spec: str) -> list[int]:
    """Return a sorted deduplicated list of 1-indexed page numbers."""
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", token)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            out.update(range(a, b + 1))
            continue
        if token.isdigit():
            out.add(int(token))
            continue
        raise ValueError(f"Bad page token: {token!r}")
    return sorted(out)


# ── Render specific pages of a PDF to PNG bytes ─────────────────────────────
def render_pages(pdf_path: Path, pages_1based: list[int]) -> dict[int, bytes]:
    """Render only the requested pages. Returns {page_num_1based: png_bytes}."""
    out: dict[int, bytes] = {}
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        n = len(pdf)
        for p in pages_1based:
            if not (1 <= p <= n):
                print(f"[probe] SKIP page={p} (PDF only has {n} pages)", flush=True)
                continue
            pil = pdf[p - 1].render(scale=RENDER_SCALE).to_pil()
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            out[p] = buf.getvalue()
    finally:
        pdf.close()
    return out


# ── One /ocr_page call (with retry — Modal cold starts can 3xx/5xx once) ───
async def ocr_one_page(
    client: httpx.AsyncClient,
    endpoint_url: str,
    report_id: str,
    page_num: int,
    png_bytes: bytes,
    *,
    max_attempts: int = 4,
) -> dict:
    """POST one page. Retries on ANY non-2xx (Modal cold starts return 303
    while the container boots vLLM — the same retry pattern the production
    ingest client uses)."""
    files = {"image": (f"page_{page_num}.png", png_bytes, "image/png")}
    data = {"page_number": str(page_num), "report_id": report_id}

    last_err: str = ""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=3, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    ):
        with attempt:
            resp = await client.post(endpoint_url, files=files, data=data)
            if not resp.is_success:
                # Dump everything the response tells us so we can diagnose
                # whether this is a cold start, an auth wall, a route
                # mismatch, or a real server error.
                try:
                    body_text = resp.text or ""
                except Exception:  # noqa: BLE001
                    body_text = "(could not read body)"
                headers_line = "; ".join(
                    f"{k}={v}" for k, v in resp.headers.items()
                    if k.lower() in {
                        "location", "server", "content-type",
                        "content-length", "modal-call-id", "www-authenticate",
                    }
                )
                last_err = (
                    f"status={resp.status_code} "
                    f"headers[{headers_line}] "
                    f"body[{body_text[:400]!r}]"
                )
                print(
                    f"[probe] page={page_num} attempt={attempt.retry_state.attempt_number} "
                    f"NON_2XX {last_err}",
                    flush=True,
                )
                # Raise an HTTPError subclass so tenacity retries this attempt.
                raise httpx.HTTPStatusError(
                    last_err,
                    request=resp.request,
                    response=resp,
                )
            return resp.json()

    # If we exit the retry loop without returning, tenacity re-raised — this
    # is unreachable, but keeps type checkers happy.
    raise RuntimeError(f"Chandra kept returning non-2xx on page {page_num}: {last_err}")


# ── Diagnostic VLM (the WHY-are-you-calling-this-decorative prompt) ─────────
async def diagnostic_vlm(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    try:
        resp = await get_client().chat.completions.create(
            model=settings.vlm_model,
            max_tokens=settings.vlm_max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DIAGNOSTIC_VLM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001
        return f"[VLM-EXCEPTION {type(e).__name__}: {e}]"
    if not resp.choices:
        return "[VLM-NO-CHOICES]"
    choice = resp.choices[0]
    text = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", None)
    if not text:
        return f"[VLM-EMPTY finish_reason={finish_reason!r}]"
    return text.strip()


# ── Per-page dump ───────────────────────────────────────────────────────────
async def dump_page(
    out_dir: Path,
    pdf_stem: str,
    page_num: int,
    page_png: bytes,
    ocr_response: dict,
    run_vlm: bool,
) -> dict:
    """Save the page image, per-Image-block crops, raw JSON, and a summary.

    Returns a per-page summary dict for the top-level report.
    """
    page_dir = out_dir / f"{pdf_stem}_page{page_num:04d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    # 1) Rendered page image.
    (page_dir / "page.png").write_bytes(page_png)

    # 2) Raw Chandra JSON.
    (page_dir / "chandra_raw.json").write_text(
        json.dumps(ocr_response, indent=2, ensure_ascii=False)
    )

    # 3) Enumerate every block, extract label/bbox, save Image crops.
    chunks = ocr_response.get("chunks", []) or []
    per_label: dict[str, int] = {}
    image_reports: list[dict] = []

    for i, blk in enumerate(chunks):
        label = blk.get("label") or "?"
        per_label[label] = per_label.get(label, 0) + 1
        bbox = blk.get("bbox")

        if label != "Image":
            continue
        if not bbox or len(bbox) != 4:
            image_reports.append({
                "index": i,
                "bbox": bbox,
                "reason_dropped": "bad_bbox",
            })
            continue

        x0, y0, x1, y1 = bbox
        area = int(max(0.0, (x1 - x0) * (y1 - y0)))

        try:
            crop_png = crop_bbox_from_png(page_png, bbox)
        except Exception as e:  # noqa: BLE001
            image_reports.append({
                "index": i, "bbox": bbox, "area_px2": area,
                "reason_dropped": f"crop_error: {type(e).__name__}: {e}",
            })
            continue

        crop_path = page_dir / f"image_block_{i:03d}_area{area}.png"
        crop_path.write_bytes(crop_png)

        entry = {
            "index": i,
            "bbox": bbox,
            "area_px2": area,
            "crop_file": crop_path.name,
            "alt_text_snippet": (blk.get("content") or "")[:200],
        }

        if run_vlm:
            entry["diagnostic_vlm"] = await diagnostic_vlm(crop_png)

        image_reports.append(entry)

    # 4) Human-readable summary.
    lines = [
        f"# Page {page_num} — {pdf_stem}",
        "",
        f"Total blocks: {sum(per_label.values())}",
        f"By label:     {json.dumps(per_label, indent=2)}",
        f"Image blocks: {len(image_reports)}",
        "",
    ]
    for r in image_reports:
        lines.append(f"[block #{r['index']}]")
        lines.append(f"  bbox        : {r.get('bbox')}")
        if "area_px2" in r:
            lines.append(f"  area_px2    : {r['area_px2']}")
        if "reason_dropped" in r:
            lines.append(f"  DROPPED     : {r['reason_dropped']}")
        if "crop_file" in r:
            lines.append(f"  crop_file   : {r['crop_file']}")
        if r.get("alt_text_snippet"):
            lines.append(f"  alt_snippet : {r['alt_text_snippet']!r}")
        if "diagnostic_vlm" in r:
            lines.append("  diagnostic_vlm:")
            for vl in r["diagnostic_vlm"].splitlines():
                lines.append(f"    {vl}")
        lines.append("")
    (page_dir / "summary.txt").write_text("\n".join(lines))

    return {
        "page": page_num,
        "output_dir": str(page_dir),
        "total_blocks": sum(per_label.values()),
        "by_label": per_label,
        "image_block_count": len(image_reports),
        "image_reports": image_reports,
    }


# ── Main ────────────────────────────────────────────────────────────────────
async def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, type=Path, help="Path to the PDF")
    ap.add_argument(
        "--pages", required=True, type=str,
        help="Comma-sep 1-indexed pages, e.g. '21,22,25-30'",
    )
    ap.add_argument(
        "--out", required=True, type=Path,
        help="Output dir. Created if missing.",
    )
    ap.add_argument(
        "--report-id", type=str, default=None,
        help="report_id sent to Chandra. Defaults to the PDF stem.",
    )
    ap.add_argument(
        "--endpoint", type=str, default=None,
        help="Chandra /ocr_page URL. Defaults to CHANDRA_OCR_URL in .env.",
    )
    ap.add_argument(
        "--vlm", action="store_true",
        help="Also run the diagnostic VLM prompt on every Image crop. Costs "
             "one VLM call per Image block. Prints the 4-line classification.",
    )
    ap.add_argument(
        "--concurrency", type=int, default=8,
        help="Parallel /ocr_page requests. Default 8.",
    )
    args = ap.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)

    pages = _parse_pages(args.pages)
    if not pages:
        raise ValueError("No pages requested.")

    endpoint = args.endpoint or settings.chandra_ocr_url
    if not endpoint:
        raise RuntimeError(
            "No Chandra endpoint. Pass --endpoint or set CHANDRA_OCR_URL in .env"
        )

    report_id = args.report_id or args.pdf.stem
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"[probe] pdf={args.pdf.name} pages={pages} out={args.out} "
        f"endpoint={endpoint} report_id={report_id} vlm={args.vlm}",
        flush=True,
    )

    print(f"[probe] rendering {len(pages)} page(s)...", flush=True)
    page_pngs = render_pages(args.pdf, pages)
    print(f"[probe] rendered {len(page_pngs)} page(s)", flush=True)

    per_page_summaries: list[dict] = []
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        timeout=settings.chandra_ocr_timeout_s,
        # Some Modal cold-starts respond 303 See Other. httpx doesn't follow
        # 3xx on POST by default. If a redirect DOES resolve to the same
        # endpoint (rare), this saves a retry round.
        follow_redirects=True,
    ) as client:

        async def _do(pn: int, png: bytes) -> dict:
            async with sem:
                print(f"[probe] page={pn} → POST /ocr_page ...", flush=True)
                try:
                    ocr = await ocr_one_page(client, endpoint, report_id, pn, png)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[probe] page={pn} OCR-ERROR {type(e).__name__}: {e}",
                        flush=True,
                    )
                    return {"page": pn, "error": str(e)}
            summary = await dump_page(
                args.out, args.pdf.stem, pn, png, ocr, run_vlm=args.vlm
            )
            print(
                f"[probe] page={pn} DONE  images={summary['image_block_count']}  "
                f"labels={summary['by_label']}",
                flush=True,
            )
            return summary

        results = await asyncio.gather(
            *(_do(pn, png) for pn, png in page_pngs.items())
        )
        per_page_summaries.extend(results)

    # ── Top-level report ────────────────────────────────────────────────────
    overall = {
        "pdf": str(args.pdf),
        "report_id": report_id,
        "endpoint": endpoint,
        "pages": pages,
        "used_vlm": args.vlm,
        "per_page": per_page_summaries,
    }
    (args.out / "probe_report.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False)
    )
    print(
        f"[probe] wrote overall report → {args.out / 'probe_report.json'}",
        flush=True,
    )

    # ── One-line human summary ─────────────────────────────────────────────
    total_images = sum(
        p.get("image_block_count", 0) for p in per_page_summaries
    )
    label_totals: dict[str, int] = {}
    for p in per_page_summaries:
        for k, v in (p.get("by_label") or {}).items():
            label_totals[k] = label_totals.get(k, 0) + v
    print(
        f"\n[probe] SUMMARY pages={len(pages)}  "
        f"total_image_blocks={total_images}  label_totals={label_totals}",
        flush=True,
    )
    if args.vlm:
        print(
            "[probe] Open the per-page summary.txt files under --out to see "
            "the diagnostic VLM classifications (INFORMATIVE vs DECORATIVE "
            "with reasons).",
            flush=True,
        )
    else:
        print(
            "[probe] Rerun with --vlm to see the VLM's classification + "
            "reason per Image crop.",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(_main())
