"""VLM extraction pass for Image chunks that Chandra tagged but didn't OCR.

Chandra-OCR-2 classifies figures/infographics/charts as `Image` blocks and
returns only their bbox + alt-text. For anything more informational than a
banner icon (dashboard cards, charts, sankey diagrams, callout boxes with
embedded KPIs), we crop the page image at that bbox and feed the crop to a
vision-capable Fireworks model to write a THREE-PART DESCRIPTION:

    DESCRIPTION: 1-2 sentence semantic summary — chart type, topic, period
    METRIC NAMES: canonical KPI names / acronyms that appear (no values)
    DATA HINTS: verbatim numbers with their labels, best-effort

The three-part output serves retrieval:
  • DESCRIPTION supports dense / semantic matching
  • METRIC NAMES + DATA HINTS support BM25 (a query for "Scope 1 1.2M tCO2e"
    needs to find those literal tokens in the chunk).

Query-time extraction on these chunks flows through the COMPOSITE wing —
the 3-part text is treated like any other prose chunk. There is no
query-time VLM re-crop; if the ingestion description missed a value, we
lose it. This is a deliberate simplification (the query-time image path
we used to have added ~4 VLM calls per query and rarely produced value
beyond what the composite wing already caught).

If the VLM confirms the region is decorative (no data), it returns the exact
token DECORATIVE and the caller drops the chunk.

Reuses the same `AsyncOpenAI` client we use for the agents — Fireworks's
`qwen3p7-plus` (default in .env) accepts multimodal `image_url` content, so
no second client or SDK.
"""
from __future__ import annotations

import base64
import io

from PIL import Image

from agentic_rag.llm import get_vlm_config
from config.settings import settings  # noqa: F401 — kept for existing callers

VLM_EXTRACT_PROMPT = (
    "You are describing an infographic / chart / dashboard card from a "
    "sustainability disclosure report. Your output goes into a retrieval "
    "index — it must be indexable AND informative.\n\n"
    "Return EXACTLY the three-section markdown format below. Do NOT add "
    "preamble, headings, or commentary before or after these three sections.\n\n"
    "═══════════════════════════════════════════════════════════════════\n"
    "DESCRIPTION\n"
    "One or two sentences that name (a) the visual type (bar chart / pie / "
    "sankey / dashboard card / KPI callout / diagram / wheel), (b) the "
    "topic in domain terms (Scope 1 emissions, DEI split, revenue by "
    "segment, water withdrawal), and (c) the temporal scope if shown "
    "(FY2023, FY2022-24, five-year trend). No numbers here — just what "
    "this figure is ABOUT.\n\n"
    "METRIC NAMES\n"
    "A comma-separated list of the canonical KPI names, framework codes, "
    "acronyms, and unit tokens that appear anywhere in the figure. These "
    "power BM25 retrieval, so include VERBATIM forms: 'Scope 1', 'Scope 2', "
    "'GHG', 'tCO2e', 'GRI 305-1', 'BRSR Principle 3', 'FY2023', 'DEI', "
    "'LTIFR', etc. No values, no descriptions — just the tokens.\n\n"
    "DATA HINTS\n"
    "Every numeric value you can read from the figure, verbatim, paired "
    "with its label. Keep commas and units exactly as shown. One value per "
    "line, format: `<label>: <value> <unit>`. Examples:\n"
    "  Scope 1 FY2023: 1,247,850 tCO2e\n"
    "  Female Board Members: 3\n"
    "  Employee attrition rate FY24: 14.2%\n"
    "If a value is illegible or the figure has no numeric content, write "
    "the literal string `(none)` on its own line under DATA HINTS.\n"
    "═══════════════════════════════════════════════════════════════════\n\n"
    "SPECIAL CASE — DECORATIVE\n"
    "If the region is purely decorative (a logo, a section banner icon, a "
    "background illustration, an arrow, page furniture, an author photo) "
    "and carries NO informational content, respond with EXACTLY the single "
    "token on its own line: DECORATIVE\n"
    "Do NOT emit the three-section format in that case — just the one word."
)

DECORATIVE_TOKEN = "DECORATIVE"


def crop_bbox_from_png(page_png: bytes, bbox: list[float]) -> bytes:
    """Return a cropped PNG for the given bbox on the page image."""
    img = Image.open(io.BytesIO(page_png))
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img.width, x1)
    y1 = min(img.height, y1)
    if x1 <= x0 or y1 <= y0:
        # Degenerate bbox — return the whole page rather than fail.
        crop = img
    else:
        crop = img.crop((x0, y0, x1, y1))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


async def extract_image_content(
    png_bytes: bytes, *, diagnostic_tag: str = ""
) -> str:
    """Ask the Fireworks vision model what's in this image crop.

    Returns the model's markdown text. Returns "DECORATIVE" (the sentinel
    token defined above) if the model decided the region has no data content.
    Empty string on failure — caller keeps the fallback alt text.

    `diagnostic_tag` is prepended to every log line from this call (e.g.
    "page=5 area=42000") so batch ingestion runs stay grep-able. Every
    failure mode is now printed instead of silently swallowed — otherwise
    a run with 200 quietly-failed images looks identical to a healthy run
    and you can't tell the difference until query time.
    """
    tag = f"[vlm-ingest {diagnostic_tag}]" if diagnostic_tag else "[vlm-ingest]"
    b64 = base64.b64encode(png_bytes).decode("ascii")

    client, model, max_tokens = get_vlm_config()

    try:
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_EXTRACT_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            },
                        },
                    ],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001 — VLM failure shouldn't break ingestion
        # Print the actual exception so the user can see rate limits, network
        # errors, model-not-found, etc. This is critical: previously every
        # failure returned "" silently and the ingest looked healthy.
        print(
            f"{tag} EXCEPTION {type(e).__name__}: {_truncate(str(e))}",
            flush=True,
        )
        return ""

    if not resp.choices:
        print(f"{tag} EMPTY_RESPONSE (no choices in response)", flush=True)
        return ""

    choice = resp.choices[0]
    text = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", None)

    if not text:
        # Common cause: thinking model burned every token before emitting
        # anything (finish_reason='length'). Surface it — otherwise the user
        # only sees "no infographic chunks in retrieval" downstream with no
        # explanation.
        print(
            f"{tag} EMPTY_CONTENT finish_reason={finish_reason!r} "
            f"(likely VLM_MAX_TOKENS too low for this thinking model)",
            flush=True,
        )
        return ""

    return text.strip()


def _truncate(msg: str, max_len: int = 300) -> str:
    msg = msg.replace("\n", " ").strip()
    if len(msg) <= max_len:
        return msg
    return msg[: max_len - 3] + "..."
