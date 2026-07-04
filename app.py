"""Streamlit UI for the agentic RAG pipeline.

Run with:
    streamlit run app.py

Layout:
    Sidebar      → upload a PDF (with company / year / framework) + list of
                   indexed reports with delete buttons.
    Main pane    → query against selected reports; displays the final answer,
                   any extracted TableValues, the cited evidence chunks (with
                   the actually-cited ones highlighted and expanded by default),
                   and the planner/critic trace.

Notes:
    - All async functions in the pipeline are driven via asyncio.run(...) per
      Streamlit interaction.
    - We share a single QdrantStore across reruns via @st.cache_resource —
      necessary for the default local-file Qdrant backend which doesn't allow
      multiple concurrent clients on the same path.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from agentic_rag.citations import renumber as _renumber_citations
from agentic_rag.ingestion.pipeline import ingest_report
from agentic_rag.orchestrator import answer_queries, answer_query, BatchOutcome
from agentic_rag.pdf_preview import get_pdf_path, overlay_bbox, render_pdf_page
from agentic_rag.pdf_report import generate_batch_pdf
from agentic_rag.query_log import save_query_log
from agentic_rag.schemas import Framework, RetrievedChunk
from agentic_rag.vectordb.qdrant import QdrantStore


st.set_page_config(
    page_title="Sustainability Agentic RAG",
    layout="wide",
    page_icon="🌱",
)


# ───── Shared Qdrant store ──────────────────────────────────────────────────

@st.cache_resource
def _get_store() -> QdrantStore:
    """Single QdrantStore for the whole Streamlit process."""
    return QdrantStore()


@st.cache_data(show_spinner=False)
def _cached_render_page(pdf_path_str: str, page: int) -> bytes:
    """Cache raw PNG renders per (pdf, page) so re-expanding is instant."""
    return render_pdf_page(pdf_path_str, page)


async def _ensure_collection() -> None:
    await _get_store().ensure_collection()


async def _list_reports() -> dict[str, int]:
    """Walk Qdrant payloads, count chunks per report_id."""
    store = _get_store()
    await store.ensure_collection()
    seen: dict[str, int] = {}
    next_offset = None
    while True:
        points, next_offset = await store.client.scroll(
            collection_name=store.collection,
            with_payload=["metadata"],
            limit=256,
            offset=next_offset,
        )
        for p in points:
            rid = (p.payload or {}).get("metadata", {}).get("report_id")
            if rid:
                seen[rid] = seen.get(rid, 0) + 1
        if next_offset is None:
            break
    return seen


async def _delete_report(report_id: str) -> None:
    await _get_store().delete_report(report_id)


async def _do_ingest(
    pdf_bytes: bytes,
    pdf_name: str,
    report_id: str,
    company: str | None,
    year: int | None,
    framework: Framework | None,
) -> int:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        return await ingest_report(
            tmp_path,
            report_id=report_id,
            company=company,
            report_year=year,
            framework=framework,
            store=_get_store(),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


async def _do_query(question: str, report_ids: list[str], on_event=None):
    return await answer_query(
        question, report_ids=report_ids, store=_get_store(), on_event=on_event
    )


async def _do_batch(queries: list[str], report_ids: list[str], on_query_event=None):
    return await answer_queries(
        queries, report_ids=report_ids, store=_get_store(), on_query_event=on_query_event
    )


# ───── Batch card renderer (used after a Batch mode run) ────────────────────

# ── Shared rendering helpers used by both single-query and batch views ──────

def _render_confidence_badge(ans, is_available: bool) -> None:
    """Colored pill: green / yellow / red / grey for high / medium / low / n/a."""
    if not is_available:
        color, label = "#7f7f7f", "NOT AVAILABLE"
    else:
        color = {"high": "#27ae60", "medium": "#f1c40f", "low": "#e74c3c"}.get(
            ans.confidence, "#7f7f7f"
        )
        label = ans.confidence.upper()
    st.markdown(
        f"<span style='background:{color};color:white;"
        f"padding:2px 8px;border-radius:4px;font-size:12px;'>{label}</span>",
        unsafe_allow_html=True,
    )


def _render_sources_block(
    ordered_cits,
    chunks_used,
    *,
    header: str = "**Sources**",
    quote_max: int = 200,
) -> None:
    """Numbered sources list with expandable PDF page previews + bbox highlight."""
    if not ordered_cits:
        return
    st.markdown(header)
    chunk_by_id = {rc.chunk.chunk_id: rc for rc in chunks_used}
    for j, c in enumerate(ordered_cits, start=1):
        bits = [f"**[{j}]**", f"page **{c.page}**"]
        if c.section:
            bits.append(f"_{c.section}_")
        bits.append(f"`{c.chunk_id}`")
        st.markdown(" · ".join(bits))
        if c.quote:
            st.caption(f"> {c.quote[:quote_max]}")
        pdf_path = get_pdf_path(c.report_id)
        if pdf_path is None:
            st.caption(
                "⚠️ Source PDF not on disk — re-upload this report to enable page previews."
            )
            continue
        with st.expander(f"📄 Open page {c.page} in source PDF", expanded=False):
            try:
                png = _cached_render_page(str(pdf_path), c.page)
                rc = chunk_by_id.get(c.chunk_id)
                bbox = rc.chunk.metadata.bbox if rc else None
                if bbox:
                    png = overlay_bbox(png, bbox)
                st.image(png, use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not render page {c.page}: {e}")


def _render_trace_expander(result, *, key_suffix: str = "") -> None:
    with st.expander("Trace"):
        st.write(
            f"Subqueries: {len(result.trace.all_subqueries)}, "
            f"iterations: {result.trace.iterations}"
        )
        st.write(f"Planner: {result.trace.planner.reasoning[:400]}")
        for j, d in enumerate(result.trace.critic_decisions, start=1):
            verdict = "✅ sufficient" if d.sufficient else "❌ insufficient"
            st.write(f"Critic[{j}]: {verdict} — {d.missing_info or ''}")


def _render_batch_card(idx: int, outcome: BatchOutcome) -> None:
    """One result card per query in a Batch run."""
    with st.container(border=True):
        st.markdown(f"**[{idx}] {outcome.query}**")

        if outcome.error is not None or outcome.result is None:
            st.error(f"❌ Pipeline error: {outcome.error or 'unknown'}")
            return

        result = outcome.result
        ans = result.answer
        _render_confidence_badge(ans, ans.answer_available)

        if not ans.answer_available:
            st.write(ans.answer)
            if ans.caveats:
                st.caption(f"Caveats: {ans.caveats}")
            _render_trace_expander(result)
            return

        clean_answer, ordered_cits = _renumber_citations(ans.answer, ans.citations)
        st.markdown(clean_answer)
        if ans.caveats:
            st.caption(f"Caveats: {ans.caveats}")
        _render_sources_block(ordered_cits, result.chunks_used)
        _render_trace_expander(result)


# Citation renumbering lives in agentic_rag.citations so it stays consistent
# across the Streamlit UI and the PDF batch report.


# ───── Chunk renderer (defined before use; Streamlit is top-down) ───────────

def _render_chunk(retrieved: RetrievedChunk, *, is_cited: bool) -> None:
    """Render one retrieved chunk in an expander, with metadata + body."""
    c = retrieved.chunk
    m = c.metadata
    if m.pages and m.pages != [m.page]:
        page_str = f"p.{min(m.pages)}–{max(m.pages)}"
    else:
        page_str = f"p.{m.page}"
    marker = "⭐ " if is_cited else ""
    label_emoji = "📊" if m.is_table else "📝"
    short_id = c.chunk_id.split("::")[-1] if "::" in c.chunk_id else c.chunk_id[:8]

    summary = (
        f"{marker}{label_emoji} **{m.label or 'Chunk'}** · {page_str} · "
        f"score={retrieved.score:.3f} · `{short_id}`"
    )
    if m.section:
        summary += f" · _{m.section[:80]}_"

    with st.expander(summary, expanded=is_cited):
        meta_bits = []
        if m.company:
            meta_bits.append(f"company: **{m.company}**")
        if m.report_year:
            meta_bits.append(f"year: **{m.report_year}**")
        if m.framework:
            meta_bits.append(f"framework: **{m.framework.value}**")
        meta_bits.append(f"chunk_id: `{c.chunk_id}`")
        st.caption("  ·  ".join(meta_bits))

        if m.is_table and m.table_data and (m.table_data.headers or m.table_data.rows):
            # Chandra sometimes emits rows whose cell count differs from the
            # header count (merged cells, missing data). Pad/truncate so the
            # DataFrame constructor doesn't raise.
            headers = m.table_data.headers or None
            rows = m.table_data.rows or []
            if headers:
                n = len(headers)
                rows = [(list(r) + [""] * n)[:n] for r in rows]
            df = pd.DataFrame(rows, columns=headers)
            st.dataframe(df, use_container_width=True, hide_index=True)
            with st.expander("Markdown view (what the embedder / synthesizer sees)"):
                st.markdown(c.text)
        else:
            st.markdown(c.text)


# ───── Sidebar: upload + manage ─────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Reports")

    with st.expander("Upload a new report", expanded=True):
        pdf_file = st.file_uploader("PDF", type=["pdf"])
        report_id = st.text_input(
            "Report ID",
            placeholder="voltas-brsr-fy25",
            help="Stable, unique identifier used in citations.",
        )
        company = st.text_input("Company", placeholder="Voltas")
        year = st.number_input(
            "Report year",
            min_value=2000,
            max_value=2100,
            value=2025,
            step=1,
        )
        framework_str = st.selectbox(
            "Framework",
            ["", "BRSR", "GRI", "SASB", "TCFD", "IR", "CDP", "OTHER"],
            index=1,
        )

        can_ingest = pdf_file is not None and bool(report_id.strip())
        if st.button("Ingest", type="primary", disabled=not can_ingest):
            framework = Framework(framework_str) if framework_str else None
            with st.status("Ingesting report…", expanded=True) as status:
                status.write(f"Reading **{pdf_file.name}** ({pdf_file.size:,} bytes)")
                status.write("Rendering pages locally with pypdfium2…")
                status.write("Fanning out to Chandra on Modal (64-way concurrent)…")
                status.write("Embedding via Ollama + BM25 → upserting to Qdrant…")
                try:
                    n = asyncio.run(
                        _do_ingest(
                            pdf_file.getvalue(),
                            pdf_file.name,
                            report_id.strip(),
                            company.strip() or None,
                            int(year) if year else None,
                            framework,
                        )
                    )
                    status.update(label=f"✅ Indexed {n} chunks", state="complete")
                    st.session_state["last_ingested"] = report_id.strip()
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    status.update(label=f"❌ Ingestion failed: {e}", state="error")
                    raise

    st.divider()
    st.subheader("Indexed reports")
    try:
        reports = asyncio.run(_list_reports())
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read Qdrant: {e}")
        reports = {}

    if not reports:
        st.caption("Nothing indexed yet. Upload a report above to begin.")
    else:
        for rid, n in sorted(reports.items()):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"`{rid}` · **{n}** chunks")
            if c2.button("🗑", key=f"del-{rid}", help=f"Delete {rid}"):
                asyncio.run(_delete_report(rid))
                st.rerun()


# ───── Main pane: query + answer ────────────────────────────────────────────

st.title("🌱 Agentic RAG — Sustainability Reports")
st.caption(
    "Planner → keyword-gated hybrid retrieval → Qwen3 reranker → "
    "Critic-loop → Synthesizer with inline citations."
)

if not reports:
    st.info("👈 Upload a report from the sidebar to get started.")
    st.stop()

mode = st.radio(
    "Mode",
    options=["Single query", "Batch queries"],
    horizontal=True,
    label_visibility="collapsed",
)

with st.form("query_form"):
    col1, col2 = st.columns([3, 2])

    if mode == "Single query":
        with col1:
            question = st.text_area(
                "Ask a question",
                placeholder="What was Voltas's Scope 1 emissions in FY 2025?",
                height=120,
            )
            batch_text = ""
    else:
        with col1:
            batch_text = st.text_area(
                "Enter one query per line",
                placeholder=(
                    "What is the total employee count?\n"
                    "What are the Scope 1 emissions for FY2024?\n"
                    "Has the company performed any LCAs on its products?\n"
                    "..."
                ),
                height=200,
                help="Blank lines and lines starting with # are ignored.",
            )
            question = ""

    with col2:
        selected = st.multiselect(
            "Query against report(s)",
            options=sorted(reports.keys()),
            default=sorted(reports.keys())[:1],
        )
        if mode == "Single query":
            st.caption(
                "Tip: try a pointed numeric question to exercise the "
                "TableExtractor path."
            )
        else:
            st.caption(
                "Tip: paste a compliance checklist. The batch runs "
                "sequentially and produces a PDF audit report."
            )

    btn_label = "Ask" if mode == "Single query" else "Run batch"
    submitted = st.form_submit_button(btn_label, type="primary", use_container_width=True)


if submitted and mode == "Batch queries":
    # Parse the batch textarea. Skip blank lines + #-prefixed comments.
    queries = [
        line.strip()
        for line in batch_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not queries:
        st.warning("Paste at least one query in the textarea.")
        st.stop()
    if not selected:
        st.warning("Select at least one report.")
        st.stop()

    # Live results panel — cards get appended as each query completes.
    results_container = st.container()
    st.divider()

    with st.status(
        f"Running batch of {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}…",
        expanded=True,
    ) as status:
        current_idx_holder = {"i": 0}

        def on_query_event(qi: int, q: str, stage: str, data) -> None:
            if stage == "batch_query_start":
                status.write(f"---\n**[{qi + 1}/{len(queries)}]** _{q[:100]}_")
                current_idx_holder["i"] = qi
            elif stage == "planner_done":
                status.write(
                    f"&nbsp;&nbsp;📋 Plan: {len(data.subqueries)} subquer"
                    f"{'y' if len(data.subqueries) == 1 else 'ies'}"
                )
            elif stage == "retrieval_done":
                total = sum(len(c) for c in data["chunks_per_sub"])
                status.write(f"&nbsp;&nbsp;🔍 {total} chunks retrieved")
            elif stage == "critic_done":
                if data.sufficient:
                    status.write("&nbsp;&nbsp;✅ Critic: sufficient")
                else:
                    status.write("&nbsp;&nbsp;🔁 Critic: insufficient")
            elif stage == "synthesizer_done":
                if data.answer_available:
                    status.write(
                        f"&nbsp;&nbsp;✍️ Synthesized (confidence: {data.confidence})"
                    )
                else:
                    status.write("&nbsp;&nbsp;⚫ Not available")

        try:
            outcomes = asyncio.run(_do_batch(queries, selected, on_query_event))
            status.update(label=f"✅ Batch complete ({len(outcomes)} queries)", state="complete")
        except Exception as e:  # noqa: BLE001
            status.update(label=f"❌ Batch failed: {e}", state="error")
            raise

    # Persist a per-query audit log for each outcome
    for outcome in outcomes:
        if outcome.result is not None:
            save_query_log(outcome.query, selected, outcome.result)

    # Generate the PDF now that we have all outcomes
    from datetime import datetime as _dt
    from pathlib import Path as _P
    pdf_dir = _P("data/batch_reports")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"batch_{_dt.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    try:
        generate_batch_pdf(outcomes, selected, pdf_path)
    except Exception as e:  # noqa: BLE001
        st.error(f"PDF generation failed: {e}")
        pdf_path = None

    # Render result cards
    with results_container:
        st.header(f"Batch results — {len(outcomes)} quer{'y' if len(outcomes) == 1 else 'ies'}")

        if pdf_path and pdf_path.exists():
            st.download_button(
                "📄 Download PDF audit report",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
            st.caption(f"Saved to `{pdf_path}`")

        st.divider()

        for i, outcome in enumerate(outcomes, start=1):
            _render_batch_card(i, outcome)

    st.session_state["batch_outcomes"] = outcomes
    st.stop()


if submitted:
    if not question.strip():
        st.warning("Type a question first.")
        st.stop()
    if not selected:
        st.warning("Select at least one report.")
        st.stop()

    with st.status("Running agentic RAG loop…", expanded=True) as status:
        # Each event from the orchestrator gets translated into a status.write
        # line. They render progressively as the pipeline runs.
        def on_event(stage: str, data) -> None:
            if stage == "planner_start":
                status.write("🧠 **Planner** — decomposing query…")
            elif stage == "planner_done":
                n = len(data.subqueries)
                status.write(
                    f"📋 **Plan ready** — {n} subquer{'y' if n == 1 else 'ies'}"
                )
                for i, sq in enumerate(data.subqueries, start=1):
                    emoji = "📊" if sq.query_type == "factual_lookup" else "📝"
                    status.write(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;{emoji} `[{i}]` _{sq.query_type}_ — {sq.query}"
                    )
                    if sq.must_phrases:
                        status.write(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;must_phrases: {sq.must_phrases}"
                        )
                    if sq.target_cells:
                        status.write(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_cells: {sq.target_cells}"
                        )
            elif stage == "iteration_start":
                if data > 1:
                    status.write(f"---\n🔁 **Iteration {data}** (critic loop)")
            elif stage == "retrieval_start":
                n = len(data)
                status.write(
                    f"🔍 **Retrieving** across {n} subquer{'y' if n == 1 else 'ies'}…"
                )
            elif stage == "retrieval_done":
                total = sum(len(c) for c in data["chunks_per_sub"])
                per_sub = ", ".join(
                    f"sq{i + 1}={len(c)}" for i, c in enumerate(data["chunks_per_sub"])
                )
                status.write(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;→ {total} chunks retrieved ({per_sub})"
                )
            elif stage == "table_extraction_start":
                sub = data["subquery"]
                n = data["table_count"]
                status.write(
                    f"📊 **Table extraction** — {n} candidate table(s) for "
                    f"_{sub.query[:80]}_"
                )
            elif stage == "table_extraction_done":
                n_found = len(data.values)
                n_unfound = len(data.unfound)
                msg = f"&nbsp;&nbsp;&nbsp;&nbsp;→ {n_found} cell(s) extracted"
                if n_unfound:
                    msg += f", {n_unfound} unfound"
                status.write(msg)
            elif stage == "table_extraction_skip":
                sub = data["subquery"]
                status.write(
                    f"⏭️ Table extraction skipped for _{sub.query[:60]}_ — "
                    f"no table chunks in retrieval results"
                )
            elif stage == "critic_start":
                status.write("🤔 **Critic** — checking sufficiency…")
            elif stage == "critic_done":
                if data.sufficient:
                    status.write("&nbsp;&nbsp;&nbsp;&nbsp;✅ Sufficient")
                else:
                    miss = data.missing_info or "(no detail)"
                    status.write(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;❌ Insufficient — _{miss[:120]}_"
                    )
                    if data.follow_up_subqueries:
                        status.write(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;→ issuing "
                            f"{len(data.follow_up_subqueries)} follow-up subquer"
                            f"{'y' if len(data.follow_up_subqueries) == 1 else 'ies'}"
                        )
            elif stage == "synthesizer_start":
                status.write("✍️ **Synthesizer** — writing the cited answer…")
            elif stage == "synthesizer_done":
                status.write(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;✅ Answer ready — confidence: "
                    f"**{data.confidence}**, {len(data.citations)} citation(s)"
                )

        try:
            result = asyncio.run(_do_query(question.strip(), selected, on_event))
            status.update(label="✅ Done", state="complete")
        except Exception as e:  # noqa: BLE001
            status.update(label=f"❌ Failed: {e}", state="error")
            raise

    # Persist a full per-query audit log to disk + keep its text for inline view.
    log_path, log_text = save_query_log(question.strip(), selected, result)
    st.session_state["result"] = result
    st.session_state["log_path"] = log_path
    st.session_state["log_text"] = log_text


# ───── Render the last result ───────────────────────────────────────────────

if "result" in st.session_state:
    result = st.session_state["result"]
    cited_chunk_ids = {c.chunk_id for c in result.answer.citations}

    # ── Answer ──────────────────────────────────────────────────────────────
    st.header("Answer")
    clean_answer, ordered_cits = _renumber_citations(
        result.answer.answer, result.answer.citations
    )
    st.markdown(clean_answer)

    if ordered_cits:
        with st.container(border=True):
            _render_sources_block(
                ordered_cits, result.chunks_used,
                header="**📖 Sources**", quote_max=240,
            )

    if result.answer.caveats:
        st.warning(f"**Caveats:** {result.answer.caveats}")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Confidence", result.answer.confidence)
    metric_cols[1].metric("Iterations", result.trace.iterations)
    metric_cols[2].metric("Chunks retrieved", len(result.chunks_used))
    metric_cols[3].metric("Chunks cited", len(cited_chunk_ids))

    # ── Extracted table values ──────────────────────────────────────────────
    if result.table_values:
        st.subheader("📊 Extracted table values")
        tv_df = pd.DataFrame(
            [
                {
                    "target": v.target_description,
                    "row": v.row_label,
                    "column": v.column_label,
                    "value": v.value,
                    "unit": v.unit or "—",
                    "conf": v.confidence,
                    "chunk_id": v.chunk_id,
                    "note": v.note or "",
                }
                for v in result.table_values
            ]
        )
        st.dataframe(tv_df, use_container_width=True, hide_index=True)

    # ── Cited chunks (highlighted) + retrieved chunks ───────────────────────
    st.subheader("📑 Evidence")
    st.caption(
        "Chunks the synthesizer pulled in. ⭐ = chunk was actually cited "
        "in the final answer (chunk_id appeared in citations[])."
    )

    # Sort: cited first, then by score
    sorted_chunks = sorted(
        result.chunks_used,
        key=lambda c: (c.chunk.chunk_id not in cited_chunk_ids, -c.score),
    )

    for chunk in sorted_chunks:
        _render_chunk(chunk, is_cited=chunk.chunk.chunk_id in cited_chunk_ids)

    # ── Full audit log (saved to disk + viewable inline) ────────────────────
    if "log_path" in st.session_state:
        log_path = st.session_state["log_path"]
        log_text = st.session_state["log_text"]
        st.subheader("📝 Audit log")
        c1, c2 = st.columns([3, 1])
        c1.caption(f"Saved to: `{log_path}`")
        c2.download_button(
            "Download .txt",
            log_text,
            file_name=log_path.name,
            mime="text/plain",
            use_container_width=True,
        )
        with st.expander("View full log inline", expanded=False):
            st.code(log_text, language="text")

    # ── Trace (planner / critic / subqueries) ───────────────────────────────
    with st.expander("🔍 Trace — planner / subqueries / critic", expanded=False):
        st.markdown("**Planner reasoning:**")
        st.write(result.trace.planner.reasoning)

        st.markdown(f"**All subqueries ({len(result.trace.all_subqueries)}):**")
        sq_df = pd.DataFrame(
            [
                {
                    "type": s.query_type,
                    "query": s.query,
                    "must_phrases": ", ".join(s.must_phrases) or "—",
                    "keywords": ", ".join(s.keywords) or "—",
                    "targets": "; ".join(s.target_cells) or "—",
                    "filters": str(s.filters.model_dump(exclude_none=True)) or "—",
                }
                for s in result.trace.all_subqueries
            ]
        )
        st.dataframe(sq_df, use_container_width=True, hide_index=True)

        st.markdown(f"**Critic decisions ({result.trace.iterations} iterations):**")
        for i, decision in enumerate(result.trace.critic_decisions, start=1):
            badge = "✅ sufficient" if decision.sufficient else "🔁 needs more"
            st.markdown(f"- Iteration **{i}** — {badge}")
            if decision.missing_info:
                st.caption(f"  missing: {decision.missing_info}")
            if decision.follow_up_subqueries:
                st.caption(
                    f"  → issued {len(decision.follow_up_subqueries)} follow-up subqueries"
                )

        if result.trace.unfound_targets:
            st.warning(
                "**Unfound table targets** (Table Extractor couldn't locate these cells):\n"
                + "\n".join(f"- {t}" for t in result.trace.unfound_targets)
            )


