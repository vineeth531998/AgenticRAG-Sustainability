"""Modal deployment for Chandra OCR-2 — per-page endpoint, vLLM-backed.

Architecture inside the container:

    vllm serve  (background subprocess started in @modal.enter; loads
                 datalab-to/chandra-ocr-2 onto the H100, serves OpenAI-compatible
                 endpoint on localhost:8000)
         ▲
         │ HTTP
         │
    InferenceManager(method="vllm")   ← Chandra's Python client, kept on self.manager
         │
         ▼
    BatchOutputItem with .chunks (list[{label, content, bbox}]) → mapped into
    our pipeline schema and returned as JSON.

Concurrency:
    Modal accepts up to 64 async page requests per container; each runs
    InferenceManager.generate via asyncio.to_thread, all of which fan into
    the same vLLM server which continuous-batches them on the GPU. Matches
    Chandra's published config (max-num-seqs=64).

Deploy:
    modal deploy deploy/modal_chandra.py
After deploy, Modal prints the URL. Set in .env:
    CHANDRA_OCR_URL=https://<workspace>--agentic-rag-chandra-chandraendpoint-web.modal.run/ocr_page

NOTE: We deliberately do NOT use `from __future__ import annotations` here.
With future annotations enabled, parameter types like `image: UploadFile`
become string forward-refs, and FastAPI's `get_type_hints()` lookup runs
against the module's globals — which lack UploadFile (the fastapi imports
live inside `web()` because they're container-only deps). The symptom is
a Pydantic "TypeAdapter not fully defined" error at request time. Keeping
annotations eager dodges this; the vllm-openai:v0.17.0 image ships
Python 3.12, so PEP-604 (`X | None`) and PEP-585 (`list[T]`) syntax works
natively without the future import.
"""

import modal

app = modal.App("agentic-rag-chandra")

# ─── Image ──────────────────────────────────────────────────────────────────
# Base = the same Docker image chandra_vllm would have launched. Comes with
# CUDA toolkit + nvcc + PyTorch + vLLM 0.17.0 pre-installed.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.17.0")
    # The vllm-openai image has python3 but no `python` symlink; Modal's
    # builder calls `python -m pip ...` which 127s without it.
    .run_commands(
        "ln -sf $(command -v python3) /usr/local/bin/python && python --version"
    )
    .apt_install("libgl1", "libglib2.0-0", "poppler-utils")
    .pip_install(
        "chandra-ocr",            # InferenceManager + BatchInputItem
        "beautifulsoup4",         # HTML table → headers/rows for our TableData
        "fastapi[standard]",
        "python-multipart",
        "pillow",
        "requests",
        "huggingface_hub",
    )
    .entrypoint([])               # override vllm-openai's `vllm serve` entrypoint
)

# ─── Chandra's published vLLM serve config ──────────────────────────────────
# Copied verbatim from the chandra_vllm wrapper's debug log so we match the
# 1.44 pages/s benchmark setup. --served-model-name="chandra" is what
# InferenceManager looks for.
CHANDRA_MODEL_ID = "datalab-to/chandra-ocr-2"
CHANDRA_SERVED_NAME = "chandra"
MAX_CONCURRENT_SEQS = 64
VLLM_SERVE_ARGS = [
    "vllm", "serve", CHANDRA_MODEL_ID,
    "--no-enforce-eager",
    "--max-num-seqs", str(MAX_CONCURRENT_SEQS),
    "--dtype", "bfloat16",
    "--max-model-len", "18000",
    "--max-num-batched-tokens", "8192",
    "--gpu-memory-utilization", "0.85",
    "--enable-prefix-caching",
    "--mm-processor-kwargs", '{"min_pixels": 3136, "max_pixels": 6291456}',
    "--served-model-name", CHANDRA_SERVED_NAME,
    "--host", "0.0.0.0",
    "--port", "8000",
]

# Persistent volume so model weights download once across cold starts.
weights_vol = modal.Volume.from_name("chandra-weights", create_if_missing=True)

# Block labels Chandra emits that have zero retrieval value.
# Note: Image is NOT dropped here — see the Image branch in `_assemble` for
# how substantial-size images become chunks that get a VLM extraction pass
# on the client side. Tiny decorative icons (logos, banner glyphs) are still
# filtered out via ICON_MAX_BBOX_AREA below.
DROP_LABELS = {"Page-Header", "Page-Footer"}

# Bbox pixel-area cutoff for decorative icons vs substantial figures. Chandra
# bboxes are in the pixel space of the image we sent at scale=2.0, so 40k sq
# px ≈ 200×200 px at 2x, ≈ 100×100 in the source PDF — big enough to hold
# real infographic content, small enough to keep dashboard icons out.
ICON_MAX_BBOX_AREA = 40000

# Chunk-assembly tuning. Chandra's chunks are FINE-GRAINED — a single
# "a. Number of locations" list item becomes its own chunk if we don't pack
# them. We pack Text + List-Group blocks under a Section-Header into chunks of
# ~2400 chars (≈600 tokens) before they leave the server.
CHUNK_MAX_CHARS = 2400
# When a Table follows a small dangling text block, attach the small block to
# the table as caption-like context instead of emitting it as its own chunk.
CHUNK_MIN_DANGLING_CHARS = 120


@app.cls(
    image=image,
    gpu="H100",
    timeout=900,
    scaledown_window=300,
    volumes={"/cache": weights_vol},
    max_containers=4,
    # secrets=[modal.Secret.from_name("hf-token")],  # uncomment if weights gated
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_SEQS)
class ChandraEndpoint:

    @modal.enter()
    def setup(self) -> None:
        """Start vLLM, wait for ready, prepare InferenceManager. Runs once per container."""
        import os
        import subprocess
        import sys
        import threading
        import time

        os.environ["HF_HOME"] = "/cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/cache/transformers"

        print(f"[setup] Launching: {' '.join(VLLM_SERVE_ARGS)}", flush=True)
        self.vllm_proc = subprocess.Popen(
            VLLM_SERVE_ARGS,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _pipe_logs() -> None:
            assert self.vllm_proc.stdout is not None
            for line in iter(self.vllm_proc.stdout.readline, ""):
                sys.stdout.write(f"[vllm] {line}")
                sys.stdout.flush()

        threading.Thread(target=_pipe_logs, daemon=True).start()

        # Wait for /v1/models to come up. First-ever container does model
        # download (slow); subsequent cold starts read weights from the volume.
        import requests
        deadline = time.time() + 600
        ready = False
        while time.time() < deadline:
            if self.vllm_proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with code {self.vllm_proc.returncode} "
                    f"before becoming ready. See [vllm] log lines above."
                )
            try:
                r = requests.get("http://localhost:8000/v1/models", timeout=2)
                if r.status_code == 200:
                    print(f"[setup] ✓ vLLM ready: {r.json()}", flush=True)
                    ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(3)
        if not ready:
            raise RuntimeError("vLLM did not become ready within 10 minutes")

        from chandra.model import InferenceManager  # type: ignore[import-not-found]
        self.manager = InferenceManager(method="vllm")
        print("[setup] ✓ InferenceManager initialized", flush=True)

    @modal.asgi_app()
    def web(self):
        import asyncio
        import io
        from typing import Any

        from chandra.model.schema import BatchInputItem  # type: ignore[import-not-found]
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from PIL import Image

        api = FastAPI(title="Chandra OCR-2")

        @api.get("/healthz")
        def healthz() -> dict[str, str]:
            return {"status": "ok"}

        @api.post("/ocr_page")
        async def ocr_page(
            image: UploadFile = File(...),
            page_number: int = Form(...),
            report_id: str = Form(...),
            prompt_type: str = Form("ocr_layout"),
        ) -> dict[str, Any]:
            import sys
            import traceback

            # Outer net so NO exception escapes as the FastAPI default
            # "Internal Server Error" with no detail. We always surface the
            # type name + repr, and dump the traceback to Modal logs.
            try:
                try:
                    img_bytes = await image.read()
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    print(
                        f"[ocr_page] page={page_number} image-decode-error: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    raise HTTPException(400, f"Could not decode image: {e}") from e

                try:
                    batch = [BatchInputItem(image=pil_img, prompt_type=prompt_type)]
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[ocr_page] page={page_number} batch-build-error: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    traceback.print_exc(file=sys.stdout)
                    raise HTTPException(
                        500,
                        f"BatchInputItem construction failed ({type(e).__name__}): {e}",
                    ) from e

                # InferenceManager.generate is sync (HTTP to local vLLM).
                # asyncio.to_thread keeps the asyncio loop unblocked so
                # @concurrent(max_inputs=64) actually runs in parallel.
                try:
                    results = await asyncio.to_thread(self.manager.generate, batch)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[ocr_page] page={page_number} generate-error: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    traceback.print_exc(file=sys.stdout)
                    raise HTTPException(
                        500,
                        f"Chandra generate failed ({type(e).__name__}): {e}",
                    ) from e

                if not results:
                    print(f"[ocr_page] page={page_number} empty-results", flush=True)
                    raise HTTPException(500, "Chandra returned no results")
                result = results[0]
                if getattr(result, "error", False):
                    err_attr = getattr(result, "error", None)
                    print(
                        f"[ocr_page] page={page_number} chandra-error-flag={err_attr!r}",
                        flush=True,
                    )
                    raise HTTPException(
                        500,
                        f"Chandra reported error on page {page_number}: {err_attr!r}",
                    )

                try:
                    chunks = _result_to_chunks(result, page_number)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[ocr_page] page={page_number} assembly-error: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    traceback.print_exc(file=sys.stdout)
                    raise HTTPException(
                        500,
                        f"Chunk assembly failed ({type(e).__name__}): {e}",
                    ) from e

                print(
                    f"[ocr_page] page={page_number} ok: {len(chunks)} chunks, "
                    f"token_count={getattr(result, 'token_count', None)}",
                    flush=True,
                )
                return {
                    "report_id": report_id,
                    "page_number": page_number,
                    "chunks": chunks,
                    "token_count": getattr(result, "token_count", None),
                }
            except HTTPException:
                raise
            except Exception as e:  # noqa: BLE001
                print(
                    f"[ocr_page] page={page_number} UNHANDLED: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                traceback.print_exc(file=sys.stdout)
                raise HTTPException(
                    500,
                    f"Unhandled ({type(e).__name__}): {e}",
                ) from e

        return api


# ---------------------------------------------------------------------------
# Assemble Chandra's BLOCK-level output into retrieval CHUNKS.
#
# Chandra emits very fine-grained blocks — a single list-item label like
# "a. Number of locations" becomes its own block. Embedding/retrieving those
# 1:1 is useless: no context, no semantic content, garbage chunks.
#
# We assemble them into coherent chunks before returning:
#   - Section-Header  → sets current section, NOT emitted as its own chunk
#   - Table           → standalone chunk (must stay intact for the
#                       TableExtractor agent). Small dangling text immediately
#                       before a table is prepended as caption-like context.
#   - Text/List-Group → accumulated into a buffer; flushed at section breaks,
#                       at tables, or when buffer ≥ CHUNK_MAX_CHARS.
#   - Page-Header / Page-Footer → dropped (noise).
#
# Every emitted chunk gets the current section prepended as `# section` in the
# text so embeddings see section context, and stored as metadata.
#
# Output contract (consumed by chandra_client.py):
#   {text, page, pages, section, is_table, table, bbox, label}
# ---------------------------------------------------------------------------
def _result_to_chunks(result, page_number: int) -> list[dict]:
    raw_blocks = getattr(result, "chunks", None) or []
    return _assemble(raw_blocks, page_number)


def _assemble(raw_blocks, page_number: int) -> list[dict]:
    chunks: list[dict] = []
    current_section: str | None = None
    buffer: list[dict] = []  # accumulator of {"text", "bbox"}

    def buffer_chars() -> int:
        return sum(len(b["text"]) for b in buffer)

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        body = "\n\n".join(b["text"] for b in buffer)

        # Footnote / dangling-text rule: if the buffer is tiny AND the most
        # recent emitted chunk is still in the same section, fold the buffer
        # into that chunk instead of emitting a near-useless mini-chunk. This
        # catches the common BRSR pattern: Table followed by a small
        # "*Material suppliers of the business verticals…" footnote.
        if (
            len(body) < CHUNK_MIN_DANGLING_CHARS
            and chunks
            and chunks[-1].get("section") == current_section
        ):
            chunks[-1]["text"] = chunks[-1]["text"] + "\n\n" + body
            buffer = []
            return

        text = f"# {current_section}\n\n{body}" if current_section else body
        bbox = _union_bboxes([b["bbox"] for b in buffer if b["bbox"]])
        chunks.append({
            "text": text,
            "page": page_number,
            "pages": [page_number],
            "section": current_section,
            "is_table": False,
            "table": None,
            "bbox": bbox,
            "label": "Composite",
        })
        buffer = []

    for blk in raw_blocks:
        label = _attr(blk, "label")
        if not label or label in DROP_LABELS:
            continue
        bbox = _attr(blk, "bbox")
        bbox_list = list(bbox) if bbox else None
        content = _attr(blk, "content") or ""

        if label == "Section-Header":
            flush()
            current_section = _html_to_text(content) or current_section
            continue

        if label == "Image":
            # Emit substantial images as their own chunk carrying just the bbox
            # (and the alt-text as placeholder body). The client will crop the
            # page image at this bbox and run a VLM extraction pass to fill in
            # the real content. Small icons (logos, section banners) are
            # dropped here so we don't waste VLM calls on them.
            if not bbox_list or len(bbox_list) != 4:
                continue
            x0, y0, x1, y1 = bbox_list
            area = max(0.0, (x1 - x0) * (y1 - y0))
            if area < ICON_MAX_BBOX_AREA:
                continue
            flush()  # close any pending prose buffer
            alt_text = _html_to_text(content).strip() or "(figure — awaiting VLM extraction)"
            chunks.append({
                "text": alt_text,
                "page": page_number,
                "pages": [page_number],
                "section": current_section,
                "is_table": False,
                "table": None,
                "bbox": bbox_list,
                "label": "Image",
            })
            continue

        if label == "Table":
            headers, rows = _parse_html_table(content)
            if not (headers or rows):
                continue  # drop empty tables (OCR noise)
            md = _table_to_markdown(headers, rows)

            # If the buffer has only a small dangling text block (likely a
            # table caption / sub-label), prepend it to the table chunk
            # instead of emitting a tiny standalone chunk.
            prefix_text = ""
            if buffer and buffer_chars() < CHUNK_MIN_DANGLING_CHARS:
                prefix_text = "\n\n".join(b["text"] for b in buffer)
                buffer = []
            elif buffer:
                flush()

            parts: list[str] = []
            if current_section:
                parts.append(f"# {current_section}")
            if prefix_text:
                parts.append(prefix_text)
            parts.append(md)
            chunks.append({
                "text": "\n\n".join(parts),
                "page": page_number,
                "pages": [page_number],
                "section": current_section,
                "is_table": True,
                "table": {"headers": headers, "rows": rows, "caption": None},
                "bbox": bbox_list,
                "label": "Table",
            })
            continue

        # Text, List-Group, and any other prose-like label → accumulate.
        text = _html_to_text(content)
        if not text:
            continue
        buffer.append({"text": text, "bbox": bbox_list})
        if buffer_chars() >= CHUNK_MAX_CHARS:
            flush()

    flush()
    return chunks


def _union_bboxes(bboxes: list) -> list[float] | None:
    """Smallest enclosing rectangle of a set of bboxes, or None if empty."""
    if not bboxes:
        return None
    xs0 = [b[0] for b in bboxes]
    ys0 = [b[1] for b in bboxes]
    xs1 = [b[2] for b in bboxes]
    ys1 = [b[3] for b in bboxes]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def _attr(obj, name):
    """Read either an object attribute or a dict key; chandra-ocr may evolve."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _html_to_text(html: str) -> str:
    """Strip HTML to clean plain text for embedding/retrieval."""
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "html.parser").get_text(separator=" ").strip()


def _parse_html_table(html: str) -> tuple[list[str], list[list[str]]]:
    """Parse Chandra's HTML table into (headers, rows). Cell text is stripped."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return [], []

    def cell(c) -> str:
        return c.get_text(separator=" ").strip()

    headers: list[str] = []
    thead = table.find("thead")
    if thead:
        first_tr = thead.find("tr")
        if first_tr:
            headers = [cell(c) for c in first_tr.find_all(["th", "td"])]

    body_rows: list[list[str]] = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = [cell(c) for c in tr.find_all(["th", "td"])]
        if cells:
            body_rows.append(cells)

    # No <thead> → assume first row was a header
    if not headers and body_rows:
        headers = body_rows[0]
        body_rows = body_rows[1:]

    # If thead existed but rows include the header again (some Chandra outputs),
    # don't worry — duplicate row is harmless downstream.

    return headers, body_rows


def _table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table (pipe syntax) for embeddings + LLM context."""
    if not headers and not rows:
        return ""
    parts: list[str] = []
    if headers:
        parts.append("| " + " | ".join(headers) + " |")
        parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        # Pad/trim row to header width so the markdown stays valid even if a
        # row's cell count drifts from the header.
        if headers and len(row) != len(headers):
            row = (row + [""] * len(headers))[: len(headers)]
        parts.append("| " + " | ".join(row) + " |")
    return "\n".join(parts)
