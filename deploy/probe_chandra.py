"""Standalone Chandra probe — ship one PDF page to Modal, dump the response.

This is intentionally NOT wired into the main pipeline. Its only job is to
let us see what Chandra-OCR-2 actually returns for one page so we can write
the right parser in `modal_chandra.py` / `chandra_client.py`.

Architecture inside the Modal container:

    chandra_vllm  (background subprocess; loads model, runs vLLM)
          ▲
          │ HTTP localhost:8000
          │
    InferenceManager(method="vllm")   ← Chandra's Python client
          │
          ▼
    result objects with .markdown (and possibly more — we dump everything)

Usage:
    pip install modal pypdfium2 pillow
    modal setup    # one-time

    modal run deploy/probe_chandra.py --pdf path/to/report.pdf --page 5

    # Different prompt_type (chandra exposes a small set; "ocr_layout" is the
    # default per their HF example)
    modal run deploy/probe_chandra.py --pdf path/to/report.pdf --page 5 \\
        --prompt-type ocr_layout

Output:
    - Prints result.markdown to stdout
    - Writes the FULL recursive object dump (every attribute, every nested
      field) to chandra_probe_output.json so we can see what beyond markdown
      Chandra exposes (tables? bboxes? confidence? layout blocks?)
"""
from __future__ import annotations

import modal

app = modal.App("chandra-probe")

# ─── Image ──────────────────────────────────────────────────────────────────
# pip name for chandra is a guess — if `pip install chandra-ocr` fails, try
# `chandra`, `marker-chandra`, or whatever the model card lists. Easy to change
# here; everything else stays.
# Base = the same Docker image chandra_vllm would have launched. Comes with
# CUDA toolkit + nvcc + PyTorch + vLLM 0.17.0 pre-installed and pre-tested by
# the Chandra team. debian_slim was missing nvcc which broke vLLM's startup.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.17.0")
    # vllm-openai ships python3 but no `python` symlink. Modal's builder calls
    # `python -m pip ...` which 127s without it. Symlink into the SAME Python
    # that already has vLLM, so pip layers chandra-ocr next to it (don't use
    # add_python — that installs a separate Python and you end up with vLLM
    # in one and chandra-ocr in another, unable to see each other).
    .run_commands(
        "ln -sf $(command -v python3) /usr/local/bin/python && python --version"
    )
    .apt_install("libgl1", "libglib2.0-0", "poppler-utils")
    .pip_install(
        "chandra-ocr",            # provides InferenceManager + BatchInputItem
        "pillow",
        "requests",
        "huggingface_hub",
    )
    # The vllm-openai image has ENTRYPOINT=["python3","-m","vllm.entrypoints.openai.api_server"]
    # — Modal overrides this with its own entrypoint, so we just run `vllm serve`
    # as a subprocess in the function body like before.
    .entrypoint([])
)

# Chandra's recommended vLLM flags, copied verbatim from the chandra_vllm
# debug log so we match the published throughput config exactly.
CHANDRA_MODEL_ID = "datalab-to/chandra-ocr-2"
CHANDRA_SERVED_NAME = "chandra"   # InferenceManager expects this exact name
VLLM_SERVE_ARGS = [
    "vllm", "serve", CHANDRA_MODEL_ID,
    "--no-enforce-eager",
    "--max-num-seqs", "64",
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

# Reuse the production weights volume — chandra_vllm will populate it on first
# run, every later run reads from disk.
weights_vol = modal.Volume.from_name("chandra-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=900,
    volumes={"/cache": weights_vol},
    scaledown_window=300,  # keep container warm 5 min for fast iteration
    # secrets=[modal.Secret.from_name("hf-token")],  # if Chandra weights are gated
)
def probe(image_bytes: bytes, prompt_type: str) -> dict:
    """Start chandra_vllm, run one page through Chandra, return everything."""
    import io
    import os
    import subprocess
    import sys
    import threading
    import time

    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/transformers"

    # ── 1. Launch vLLM directly (chandra_vllm uses Docker; not usable in Modal) ──
    print(f"[1/4] Launching: {' '.join(VLLM_SERVE_ARGS)}", flush=True)
    proc = subprocess.Popen(
        VLLM_SERVE_ARGS,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream its logs to our stdout so Modal captures them and we see startup
    # errors live.
    def _pipe_logs():
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            sys.stdout.write(f"[vllm] {line}")
            sys.stdout.flush()

    threading.Thread(target=_pipe_logs, daemon=True).start()

    # ── 2. Wait for the server to become ready ────────────────────────────
    # Model load on H100 is the long step (~60–120s). Allow generously.
    print("[2/4] Waiting for vLLM server to become ready (up to 10 min)...", flush=True)
    import requests
    ready = False
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vllm serve exited with code {proc.returncode} before becoming ready. "
                f"See [vllm] log lines above for the reason."
            )
        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=2)
            if r.status_code == 200:
                print(f"   ✓ Server ready. /v1/models: {r.json()}", flush=True)
                ready = True
                break
        except requests.RequestException:
            pass
        time.sleep(3)

    if not ready:
        raise RuntimeError("vLLM did not become ready within 10 minutes")

    # ── 3. Run inference via Chandra's InferenceManager ───────────────────
    print(f"[3/4] Running InferenceManager.generate (prompt_type={prompt_type!r})...", flush=True)
    from PIL import Image

    from chandra.model import InferenceManager  # type: ignore[import-not-found]
    from chandra.model.schema import BatchInputItem  # type: ignore[import-not-found]

    manager = InferenceManager(method="vllm")
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    print(f"   image size: {pil_img.size}", flush=True)

    batch = [BatchInputItem(image=pil_img, prompt_type=prompt_type)]
    results = manager.generate(batch)
    result = results[0]
    print("   ✓ generate() returned", flush=True)

    # ── 4. Inspect the result object exhaustively ─────────────────────────
    print("[4/4] Building dump of result attributes...", flush=True)
    return {
        "prompt_type": prompt_type,
        # The one attribute we know exists from the model card example.
        "markdown": getattr(result, "markdown", None),
        # ALL public attribute names — useful for spotting tables, bboxes,
        # layout, confidence, blocks, etc. that aren't in __dict__.
        "dir_attrs": sorted(a for a in dir(result) if not a.startswith("_")),
        # The result's __dict__ keys (dataclass / pydantic fields).
        "vars_keys": sorted(vars(result).keys()) if hasattr(result, "__dict__") else [],
        # Full recursive walk — every nested object's fields.
        "full_dump": _recursive_dump(result),
        # repr() as a fallback if our walker missed something.
        "repr": repr(result)[:5000],
        # If there are >1 results, dump them all (shouldn't happen for batch=1
        # but cheap insurance).
        "all_results_dump": [_recursive_dump(r) for r in results],
    }


def _recursive_dump(obj, max_depth: int = 6, depth: int = 0):
    """Walk any object into JSON-able primitives. Goal is visibility, not pretty."""
    if depth >= max_depth:
        return f"<truncated: {type(obj).__name__}>"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    # PIL images and tensors — note but don't try to serialize
    type_name = type(obj).__name__
    if type_name in {"Image", "Tensor", "ndarray"}:
        return f"<{type_name} repr={repr(obj)[:120]}>"
    if isinstance(obj, (list, tuple)):
        return [_recursive_dump(x, max_depth, depth + 1) for x in obj[:50]]
    if isinstance(obj, dict):
        return {str(k): _recursive_dump(v, max_depth, depth + 1) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {
            "_type": type_name,
            **{
                k: _recursive_dump(v, max_depth, depth + 1)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            },
        }
    return repr(obj)


@app.local_entrypoint()
def main(
    pdf: str,
    page: int = 1,
    prompt_type: str = "ocr_layout",
    out: str = "chandra_probe_output.json",
    render_scale: float = 2.0,
):
    """Render one page locally, ship to Modal, dump the Chandra response."""
    import io
    import json
    from pathlib import Path

    import pypdfium2 as pdfium

    pdf_path = Path(pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    print(f"Reading {pdf_path}...")
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        n = len(doc)
        if page < 1 or page > n:
            raise ValueError(f"page {page} out of range (PDF has {n} pages)")
        print(f"Rendering page {page}/{n} at scale={render_scale}...")
        pil = doc[page - 1].render(scale=render_scale).to_pil()
    finally:
        doc.close()

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    print(f"Page rendered: {pil.size}, PNG size: {len(img_bytes):,} bytes")
    print("Dispatching to Modal (first run is slow: model + chandra_vllm startup)...")

    result = probe.remote(img_bytes, prompt_type)

    # Pretty-print the important bits
    print("\n" + "=" * 78)
    print(f"prompt_type:  {result['prompt_type']}")
    print(f"dir_attrs:    {result['dir_attrs']}")
    print(f"vars_keys:    {result['vars_keys']}")
    print("=" * 78)
    print("\n--- result.markdown ---")
    print(result["markdown"] or "<None>")
    print("=" * 78)

    out_path = Path(out)
    out_path.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    print(f"\nFull recursive dump (look for non-markdown fields here): {out_path}")
