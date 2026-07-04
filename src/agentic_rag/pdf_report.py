"""PDF batch-report generator — audit-ready output with a clean visual design.

Produces a single PDF from a list of `BatchOutcome`s:
  • cover page with title bar, stat card grid, and a queries-in-batch table
  • one bookmarked section per query with a colored header bar
  • pill-shaped confidence badges (filled rectangles, white text)
  • answer with citations renumbered to [1] [2] …
  • numbered source cards with page + section metadata
  • compressed JPEG evidence images cropped tight around the cited bbox
  • compact trace footer (planner reasoning + subqueries + critic verdict)

Uses `fpdf2` (pure Python, no system deps).
"""
from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from PIL import Image

from agentic_rag.citations import renumber
from agentic_rag.orchestrator import BatchOutcome
from agentic_rag.pdf_preview import get_pdf_path, render_pdf_page

# ═══ Layout constants ══════════════════════════════════════════════════════

A4_W = 210.0
A4_H = 297.0
MARGIN = 18.0
CONTENT_W = A4_W - 2 * MARGIN

# Palette — muted, professional, consistent across the report.
INK = (33, 37, 41)          # near-black body text
MUTED = (108, 117, 125)     # secondary text / captions
BORDER = (222, 226, 230)    # subtle divider lines
CARD_BG = (248, 249, 250)   # very light grey card background
ACCENT = (44, 82, 130)      # deep blue for section headers / title bar

CONF_COLOR = {
    "high": (39, 174, 96),        # green
    "medium": (241, 196, 15),     # yellow
    "low": (231, 76, 60),         # red
    "unavailable": (127, 127, 127),  # grey
}

# Evidence image sizing — max 60mm tall, cropped tight to bbox.
EVIDENCE_MAX_H_MM = 60.0
EVIDENCE_MAX_W_MM = CONTENT_W - 4  # small breathing room
EVIDENCE_JPEG_QUALITY = 78
EVIDENCE_MAX_PIXELS_W = 1200  # cap width to keep file size sane


# ═══ PDF class ═════════════════════════════════════════════════════════════

class BatchReportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(MARGIN, 20, MARGIN)
        self.alias_nb_pages()

    def header(self) -> None:
        if self.page_no() == 1:
            return  # cover
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*MUTED)
        self.cell(0, 8, "Sustainability RAG - Batch Query Report",
                  0, new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_draw_color(*BORDER)
        y = self.get_y()
        self.line(MARGIN, y, A4_W - MARGIN, y)
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 10, f"Page {self.page_no()} / {{nb}}", 0, 0, "C")


# ═══ Public entry point ════════════════════════════════════════════════════

def generate_batch_pdf(
    outcomes: list[BatchOutcome],
    report_ids: list[str],
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = BatchReportPDF()

    with tempfile.TemporaryDirectory(prefix="ragbatch_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        _write_cover(pdf, outcomes, report_ids)
        for i, outcome in enumerate(outcomes, start=1):
            pdf.add_page()
            if outcome.error is not None or outcome.result is None:
                _write_error_section(pdf, i, outcome)
            else:
                _write_query_section(pdf, i, outcome, tmpdir)
        pdf.output(str(output_path))
    return output_path


# ═══ Cover ═════════════════════════════════════════════════════════════════

def _write_cover(
    pdf: BatchReportPDF,
    outcomes: list[BatchOutcome],
    report_ids: list[str],
) -> None:
    pdf.add_page()

    # Colored title bar
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 25, A4_W, 32, style="F")

    pdf.set_y(30)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "Sustainability RAG",
             0, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 13)
    pdf.cell(0, 7, "Batch Query Report",
             0, new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_y(66)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5, f"Generated {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}",
             0, new_x="LMARGIN", new_y="NEXT", align="C")

    # Stat card grid ------------------------------------------------------
    total = len(outcomes)
    answered = sum(1 for o in outcomes if o.result and o.result.answer.answer_available)
    unavailable = sum(1 for o in outcomes if o.result and not o.result.answer.answer_available)
    errored = sum(1 for o in outcomes if o.error is not None)

    cards = [
        ("TOTAL QUERIES", str(total), INK),
        ("ANSWERED", str(answered), CONF_COLOR["high"]),
        ("NOT AVAILABLE", str(unavailable), CONF_COLOR["unavailable"]),
    ]
    if errored:
        cards.append(("ERRORED", str(errored), CONF_COLOR["low"]))

    pdf.set_y(85)
    card_gap = 4
    card_w = (CONTENT_W - card_gap * (len(cards) - 1)) / len(cards)
    card_h = 22
    y = pdf.get_y()
    for i, (label, value, value_color) in enumerate(cards):
        x = MARGIN + i * (card_w + card_gap)
        _draw_stat_card(pdf, x, y, card_w, card_h, label, value, value_color)

    # Reports queried ------------------------------------------------------
    pdf.set_y(120)
    _section_heading(pdf, "Reports queried")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*INK)
    for rid in report_ids:
        pdf.cell(4, 5.5, "", 0, 0, "L")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 5.5, _s(f"- {rid}"), 0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(3)

    # Queries in this batch ------------------------------------------------
    _section_heading(pdf, "Queries in this batch")
    for i, o in enumerate(outcomes, start=1):
        _cover_query_row(pdf, i, o)


def _draw_stat_card(pdf, x, y, w, h, label, value, value_color):
    # Card background
    pdf.set_fill_color(*CARD_BG)
    pdf.set_draw_color(*BORDER)
    pdf.rect(x, y, w, h, style="DF")
    # Label
    pdf.set_xy(x, y + 3)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*MUTED)
    pdf.cell(w, 4, label, 0, 0, "C")
    # Value
    pdf.set_xy(x, y + 9)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*value_color)
    pdf.cell(w, 10, value, 0, 0, "C")
    pdf.set_text_color(*INK)


def _section_heading(pdf, text):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 6, text, 0, new_x="LMARGIN", new_y="NEXT", align="L")
    # Underline
    y = pdf.get_y()
    pdf.set_draw_color(*ACCENT)
    pdf.line(MARGIN, y, MARGIN + 40, y)
    pdf.ln(3)
    pdf.set_text_color(*INK)


def _cover_query_row(pdf, i, outcome):
    """One row: [n] query text ...........badge"""
    status = _status_key(outcome)
    color = CONF_COLOR[status]
    label = _status_label_text(outcome)

    # Number
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(7, 5, f"[{i}]", 0, 0, "L")

    # Query text (truncated)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*INK)
    q_text = _s(_truncate(outcome.query, 90))
    # Reserve space for the badge on the right
    text_w = CONTENT_W - 7 - 24
    pdf.cell(text_w, 5, q_text, 0, 0, "L")

    # Right-aligned badge
    x_badge = MARGIN + 7 + text_w + 1
    y_badge = pdf.get_y()
    _draw_pill(pdf, x_badge, y_badge, label, color, w=22, h=5)
    pdf.ln(6)


# ═══ Per-query sections ════════════════════════════════════════════════════

def _write_query_section(
    pdf: BatchReportPDF,
    idx: int,
    outcome: BatchOutcome,
    tmpdir: Path,
) -> None:
    result = outcome.result
    assert result is not None
    ans = result.answer
    is_available = ans.answer_available

    # Bookmark
    pdf.start_section(_s(f"[{idx}] {_truncate(outcome.query, 60)}"), level=0)

    # ── Colored header bar with number chip + confidence badge ────────────
    _header_bar(pdf, idx, outcome)

    # ── Query text ────────────────────────────────────────────────────────
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*INK)
    pdf.multi_cell(CONTENT_W, 6, _s(outcome.query), 0, "L")
    pdf.ln(2)

    # ── Answer ────────────────────────────────────────────────────────────
    _labelled_section(pdf, "Answer")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK)
    clean_answer, ordered_cits = renumber(ans.answer, ans.citations)
    pdf.multi_cell(CONTENT_W, 5.2, _s(clean_answer), 0, "L")
    pdf.ln(2)

    if ans.caveats:
        pdf.set_font("Helvetica", "I", 9.5)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(CONTENT_W, 5, _s(f"Caveats: {ans.caveats}"), 0, "L")
        pdf.set_text_color(*INK)
        pdf.ln(1)

    # ── Sources ───────────────────────────────────────────────────────────
    if is_available and ordered_cits:
        _labelled_section(pdf, "Sources")
        chunk_by_id = {rc.chunk.chunk_id: rc for rc in result.chunks_used}
        for j, cit in enumerate(ordered_cits, start=1):
            _source_card(pdf, j, cit, chunk_by_id, tmpdir)
    elif not is_available:
        _labelled_section(pdf, "Sources")
        pdf.set_font("Helvetica", "I", 9.5)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(CONTENT_W, 5,
                       "No supporting evidence found in the provided document.",
                       0, "L")
        pdf.set_text_color(*INK)
        pdf.ln(1)

    # ── Trace ─────────────────────────────────────────────────────────────
    _write_trace(pdf, result,
                 show_full=(not is_available or ans.confidence == "low"))


def _header_bar(pdf, idx, outcome):
    """Full-width colored bar with query number chip + confidence pill."""
    status_key = _status_key(outcome)
    conf_color = CONF_COLOR[status_key]

    y = pdf.get_y()
    bar_h = 10
    # Background bar
    pdf.set_fill_color(*ACCENT)
    pdf.rect(MARGIN, y, CONTENT_W, bar_h, style="F")
    # Query number chip on the left
    pdf.set_xy(MARGIN + 4, y + 1.5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(20, bar_h - 3, f"Query {idx}", 0, 0, "L")
    # Confidence pill on the right
    label = _status_label_text(outcome)
    pill_w = 30
    _draw_pill(pdf, A4_W - MARGIN - pill_w - 2, y + 2.5,
               label, conf_color, w=pill_w, h=bar_h - 5)
    pdf.set_y(y + bar_h)
    pdf.set_text_color(*INK)


def _labelled_section(pdf, label):
    """Small section label with underline, used for 'Answer', 'Sources', 'Trace'."""
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 5, label, 0, new_x="LMARGIN", new_y="NEXT", align="L")
    y = pdf.get_y()
    pdf.set_draw_color(*ACCENT)
    pdf.line(MARGIN, y, MARGIN + 25, y)
    pdf.ln(2)
    pdf.set_text_color(*INK)


def _source_card(pdf, number, cit, chunk_by_id, tmpdir):
    """One source card: numbered circle badge + metadata + tight evidence image."""
    # Reserve space; force new page if we won't fit the whole card
    est_h = 12 + (EVIDENCE_MAX_H_MM + 3 if get_pdf_path(cit.report_id) else 0)
    if pdf.get_y() + est_h > A4_H - 25:
        pdf.add_page()

    y_start = pdf.get_y()

    # Numbered circle badge (drawn as small filled circle)
    circle_r = 3.2
    cx = MARGIN + circle_r
    cy = y_start + circle_r
    pdf.set_fill_color(*ACCENT)
    pdf.ellipse(cx - circle_r, cy - circle_r, circle_r * 2, circle_r * 2, style="F")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(cx - circle_r, cy - circle_r + 0.5)
    pdf.cell(circle_r * 2, circle_r * 2, str(number), 0, 0, "C")

    # Metadata to the right of the circle
    pdf.set_xy(MARGIN + 10, y_start)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*INK)
    header_bits = [f"Page {cit.page}"]
    if cit.section:
        header_bits.append(_truncate(cit.section, 60))
    header_bits.append(cit.report_id)
    pdf.cell(0, 5, _s(" · ".join(header_bits)),
             0, new_x="LMARGIN", new_y="NEXT", align="L")

    # Chunk ID on the next line, muted
    pdf.set_x(MARGIN + 10)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 4, _s(cit.chunk_id),
             0, new_x="LMARGIN", new_y="NEXT", align="L")

    if cit.quote:
        pdf.set_x(MARGIN + 10)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(CONTENT_W - 10, 4.2, _s(f'"{cit.quote[:220]}"'), 0, "L")

    pdf.set_text_color(*INK)
    pdf.ln(1)

    # Evidence image
    pdf_path_ = get_pdf_path(cit.report_id)
    if pdf_path_ is not None:
        try:
            rc = chunk_by_id.get(cit.chunk_id)
            bbox = rc.chunk.metadata.bbox if rc else None
            img_path = _prepare_evidence_jpeg(
                str(pdf_path_), cit.page, bbox,
                tmpdir / f"{cit.report_id}_p{cit.page}_{cit.chunk_id.replace('::', '_')}.jpg",
            )
            _embed_image(pdf, img_path,
                         max_w_mm=EVIDENCE_MAX_W_MM, max_h_mm=EVIDENCE_MAX_H_MM)
        except Exception:  # noqa: BLE001
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(*CONF_COLOR["low"])
            pdf.multi_cell(CONTENT_W, 5,
                           "(could not render page preview)", 0, "L")
            pdf.set_text_color(*INK)

    pdf.ln(4)


def _prepare_evidence_jpeg(pdf_path, page, bbox, out_path):
    """Render + crop + compress the evidence page as a JPEG.

    Cropping tight around the bbox (with a small context margin) drops the
    image size to ~30–80 KB from ~500 KB for a full page, which is where
    the batch PDF's file-size bloat was coming from.
    """
    png_bytes = render_pdf_page(pdf_path, page)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

    if bbox and len(bbox) == 4:
        # Crop with ~40 px context margin above/below/left/right; if the
        # bbox is very small (an icon-ish chunk), give it more air.
        x0, y0, x1, y1 = (int(v) for v in bbox)
        pad_x = max(40, int((x1 - x0) * 0.15))
        pad_y = max(50, int((y1 - y0) * 0.35))
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(img.width, x1 + pad_x)
        y1 = min(img.height, y1 + pad_y)
        img = img.crop((x0, y0, x1, y1))

        # Draw the actual chunk bbox as a red outline inside the crop
        # (relative to the crop origin).
        from PIL import ImageDraw
        crop_offset_x = -(bbox[0] - x0) + (bbox[0] - x0)  # noop; just clarity
        # Re-open on img for drawing
        draw = ImageDraw.Draw(img)
        rel = (
            int(bbox[0]) - x0,
            int(bbox[1]) - y0,
            int(bbox[2]) - x0,
            int(bbox[3]) - y0,
        )
        draw.rectangle(rel, outline=(255, 50, 0), width=4)

    # Downscale if wider than the max pixel width
    if img.width > EVIDENCE_MAX_PIXELS_W:
        scale = EVIDENCE_MAX_PIXELS_W / img.width
        img = img.resize(
            (EVIDENCE_MAX_PIXELS_W, int(img.height * scale)),
            Image.LANCZOS,
        )

    img.save(out_path, "JPEG",
             quality=EVIDENCE_JPEG_QUALITY, optimize=True, progressive=True)
    return out_path


def _embed_image(pdf, img_path, *, max_w_mm, max_h_mm):
    with Image.open(img_path) as im:
        px_w, px_h = im.size
    aspect = px_h / px_w
    w = min(max_w_mm, max_h_mm / aspect)
    h = w * aspect
    if h > max_h_mm:
        h = max_h_mm
        w = h / aspect
    x = MARGIN + (CONTENT_W - w) / 2
    if pdf.get_y() + h > A4_H - 20:
        pdf.add_page()
    pdf.image(str(img_path), x=x, w=w, h=h)
    pdf.ln(h + 1)


def _write_trace(pdf, result, *, show_full):
    trace = result.trace
    pdf.ln(2)
    pdf.set_draw_color(*BORDER)
    y = pdf.get_y()
    pdf.line(MARGIN, y, A4_W - MARGIN, y)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 4, "TRACE",
             0, new_x="LMARGIN", new_y="NEXT", align="L")

    pdf.set_font("Helvetica", "", 8)
    n_sq = len(trace.all_subqueries)
    pdf.cell(0, 4,
             _s(f"{n_sq} subquery(ies) - {trace.iterations} iteration(s)"),
             0, new_x="LMARGIN", new_y="NEXT", align="L")
    if trace.planner.reasoning:
        pdf.multi_cell(CONTENT_W, 4,
                       _s(f"Planner: {_truncate(trace.planner.reasoning, 400)}"),
                       0, "L")

    if show_full:
        for i, sq in enumerate(trace.all_subqueries, start=1):
            pdf.multi_cell(CONTENT_W, 4,
                           _s(f"  sq{i} ({sq.query_type}): {_truncate(sq.query, 130)}"),
                           0, "L")
            if sq.must_phrases:
                pdf.multi_cell(CONTENT_W, 4,
                               _s(f"       must_phrases={sq.must_phrases}"),
                               0, "L")
        for i, d in enumerate(trace.critic_decisions, start=1):
            verdict = "SUFFICIENT" if d.sufficient else "NEEDS_MORE"
            line = f"  critic[{i}] {verdict}"
            if d.missing_info:
                line += f" - {_truncate(d.missing_info, 180)}"
            pdf.multi_cell(CONTENT_W, 4, _s(line), 0, "L")
        if trace.unfound_targets:
            pdf.multi_cell(
                CONTENT_W, 4,
                _s(f"  unfound: {'; '.join(_truncate(t, 80) for t in trace.unfound_targets)}"),
                0, "L",
            )
    pdf.set_text_color(*INK)


def _write_error_section(pdf, idx, outcome):
    pdf.start_section(_s(f"[{idx}] {_truncate(outcome.query, 60)}  (error)"), level=0)
    _header_bar(pdf, idx, outcome)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*INK)
    pdf.multi_cell(CONTENT_W, 6, _s(outcome.query), 0, "L")
    pdf.ln(2)
    _labelled_section(pdf, "Pipeline error")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*CONF_COLOR["low"])
    pdf.multi_cell(CONTENT_W, 5.2,
                   _s(outcome.error or "(no error message)"), 0, "L")
    pdf.set_text_color(*INK)


# ═══ Pill badge helper ════════════════════════════════════════════════════

def _draw_pill(pdf, x, y, text, bg_color, *, w=25, h=5,
               text_color=(255, 255, 255)):
    """Draw a filled pill-style badge with centered text."""
    pdf.set_fill_color(*bg_color)
    pdf.set_draw_color(*bg_color)
    pdf.rect(x, y, w, h, style="F")
    pdf.set_xy(x, y - 0.3)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*text_color)
    pdf.cell(w, h + 0.5, text, 0, 0, "C")
    pdf.set_text_color(*INK)


# ═══ Small helpers ═════════════════════════════════════════════════════════

def _truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def _status_key(o: BatchOutcome) -> str:
    if o.error is not None:
        return "low"
    if o.result is None:
        return "unavailable"
    if not o.result.answer.answer_available:
        return "unavailable"
    return o.result.answer.confidence


def _status_label_text(o: BatchOutcome) -> str:
    if o.error is not None:
        return "ERROR"
    if o.result is None or not o.result.answer.answer_available:
        return "NOT AVAILABLE"
    return o.result.answer.confidence.upper()


def _s(text: str) -> str:
    """Replace non-latin-1 characters fpdf2's core Helvetica can't render.

    Latin-1 is the safe subset for the built-in Helvetica core font. Rather
    than bundling a TTF (heavy), we substitute common Unicode punctuation.
    """
    if not text:
        return ""
    replacements = {
        "—": "-", "–": "-", "…": "...",
        """: '"', """: '"', "‘": "'", "’": "'",
        "•": "-", "→": "->", "★": "*", "▸": ">", "▾": "v",
        "🟢": "[OK]", "🟡": "[~]", "🔴": "[!]", "⭐": "*",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")
