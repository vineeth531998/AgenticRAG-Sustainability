"""PDF page rendering for citation traceability.

Each chunk has a `bbox` in metadata — the rectangle on the page where the
content sits. We render the page back to a PNG at the same scale Chandra used
at ingestion (default 2.0) so the bbox values map directly to pixel
coordinates, then overlay a translucent highlight on the cited region.

Used by the Streamlit UI to make every citation visually verifiable:
click → see the page → see the highlighted region the answer came from.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

from agentic_rag.ingestion.pipeline import stored_pdf_path

# Match the scale used in chandra_client.ChandraOCRClient (default 2.0).
# Re-rendering at the same scale means bboxes line up without conversion.
DEFAULT_RENDER_SCALE = 2.0


def get_pdf_path(report_id: str) -> Path | None:
    p = stored_pdf_path(report_id)
    return p if p.exists() else None


def render_pdf_page(
    pdf_path: str | Path,
    page: int,
    *,
    scale: float = DEFAULT_RENDER_SCALE,
) -> bytes:
    """Render one PDF page to PNG bytes (no highlight). 1-indexed page number."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        if page < 1 or page > len(doc):
            raise ValueError(
                f"page {page} out of range (PDF has {len(doc)} pages)"
            )
        pil = doc[page - 1].render(scale=scale).to_pil()
    finally:
        doc.close()

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def render_page_crop(
    pdf_path: str | Path,
    page: int,
    bbox: Sequence[float],
    *,
    padding_px: int = 40,
    scale: float = DEFAULT_RENDER_SCALE,
) -> bytes:
    """Render a PDF page, crop tight around bbox (with padding), return PNG bytes.

    Used by the Table VLM Verifier — we want just the table region to reach
    the vision model, not the whole page, so the model can focus on cell
    boundaries rather than surrounding prose.
    """
    from PIL import Image

    png = render_pdf_page(pdf_path, page, scale=scale)
    img = Image.open(io.BytesIO(png)).convert("RGB")

    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0 = max(0, x0 - padding_px)
    y0 = max(0, y0 - padding_px)
    x1 = min(img.width, x1 + padding_px)
    y1 = min(img.height, y1 + padding_px)

    if x1 <= x0 or y1 <= y0:
        cropped = img  # degenerate bbox — return whole page
    else:
        cropped = img.crop((x0, y0, x1, y1))

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def overlay_bbox(
    png_bytes: bytes,
    bbox: Sequence[float],
    *,
    fill: tuple[int, int, int, int] = (255, 230, 0, 90),     # translucent yellow
    outline: tuple[int, int, int, int] = (255, 50, 0, 255),  # solid red-orange border
    width: int = 4,
) -> bytes:
    """Overlay a translucent highlight rectangle on a rendered page.

    `bbox` is expected in the same pixel-space as `png_bytes` was rendered —
    i.e. Chandra-emitted coordinates at scale=2.0 with no further transform.
    """
    from PIL import Image, ImageDraw

    if not bbox or len(bbox) != 4:
        return png_bytes

    base = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x0, y0, x1, y1 = (float(v) for v in bbox)
    # Clamp into the image just in case OCR's bbox extends a hair beyond.
    x0 = max(0, min(x0, base.width))
    x1 = max(0, min(x1, base.width))
    y0 = max(0, min(y0, base.height))
    y1 = max(0, min(y1, base.height))
    if x1 <= x0 or y1 <= y0:
        return png_bytes  # degenerate; nothing to draw

    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=width)

    combined = Image.alpha_composite(base, overlay).convert("RGB")
    out = io.BytesIO()
    combined.save(out, format="PNG")
    return out.getvalue()
