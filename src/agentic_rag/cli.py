"""CLI: `arag upload <pdf>` and `arag ask "<question>"`.

The workflow matches what we agreed: user uploads a report, then asks questions.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_rag.ingestion.pipeline import ingest_report
from agentic_rag.orchestrator import answer_queries, answer_query
from agentic_rag.pdf_report import generate_batch_pdf
from agentic_rag.schemas import Framework
from agentic_rag.vectordb.qdrant import QdrantStore

app = typer.Typer(no_args_is_help=True, help="Agentic RAG for sustainability reports.")
console = Console()


@app.command()
def upload(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    report_id: Optional[str] = typer.Option(None, help="Stable ID for this report. Defaults to a uuid."),
    company: Optional[str] = typer.Option(None),
    year: Optional[int] = typer.Option(None),
    framework: Optional[Framework] = typer.Option(None, case_sensitive=False),
) -> None:
    """Upload a sustainability report: OCR via Chandra, embed, upsert to Qdrant."""
    rid = report_id or f"report-{uuid.uuid4().hex[:8]}"
    console.print(f"[bold]Ingesting[/bold] {pdf.name} as [cyan]{rid}[/cyan]...")
    n = asyncio.run(
        ingest_report(
            pdf,
            report_id=rid,
            company=company,
            report_year=year,
            framework=framework,
        )
    )
    console.print(f"[green]✓[/green] Indexed [bold]{n}[/bold] chunks. report_id: [cyan]{rid}[/cyan]")


@app.command()
def ask(
    question: str = typer.Argument(...),
    report_id: list[str] = typer.Option(..., "--report-id", "-r", help="One or more report_ids to query."),
    show_trace: bool = typer.Option(False, "--trace", help="Print planner/critic trace."),
) -> None:
    """Ask a question against one or more uploaded reports."""
    result = asyncio.run(answer_query(question, report_ids=report_id))

    console.print(Panel(result.answer.answer, title="Answer", border_style="green"))

    if result.answer.caveats:
        console.print(Panel(result.answer.caveats, title="Caveats", border_style="yellow"))

    console.print(f"[bold]Confidence:[/bold] {result.answer.confidence}")

    if result.table_values:
        tv_table = Table(title="Extracted table values", show_lines=True)
        tv_table.add_column("target", style="cyan")
        tv_table.add_column("row")
        tv_table.add_column("column")
        tv_table.add_column("value", style="bold")
        tv_table.add_column("unit")
        tv_table.add_column("conf")
        tv_table.add_column("chunk_id")
        for v in result.table_values:
            tv_table.add_row(
                v.target_description,
                v.row_label,
                v.column_label,
                v.value,
                v.unit or "-",
                v.confidence,
                v.chunk_id,
            )
        console.print(tv_table)

    cit_table = Table(title="Citations", show_lines=True)
    cit_table.add_column("chunk_id", style="cyan", no_wrap=True)
    cit_table.add_column("report")
    cit_table.add_column("page")
    cit_table.add_column("section")
    cit_table.add_column("quote")
    for c in result.answer.citations:
        cit_table.add_row(c.chunk_id, c.report_id, str(c.page), c.section or "-", (c.quote or "")[:80])
    console.print(cit_table)

    if show_trace:
        console.print(Panel(result.trace.planner.reasoning, title="Planner reasoning"))
        sub_table = Table(title="All subqueries", show_lines=True)
        sub_table.add_column("type", style="magenta")
        sub_table.add_column("query")
        sub_table.add_column("must_phrases", style="bold yellow")
        sub_table.add_column("keywords")
        sub_table.add_column("targets")
        sub_table.add_column("filters")
        for s in result.trace.all_subqueries:
            sub_table.add_row(
                s.query_type,
                s.query,
                ", ".join(s.must_phrases) or "-",
                ", ".join(s.keywords) or "-",
                "; ".join(s.target_cells) or "-",
                str(s.filters.model_dump(exclude_none=True)) or "-",
            )
        console.print(sub_table)
        if result.trace.unfound_targets:
            console.print(
                "[yellow]Unfound table targets:[/yellow] "
                + "; ".join(result.trace.unfound_targets)
            )
        console.print(f"[bold]Critic iterations:[/bold] {result.trace.iterations}")
        for i, c in enumerate(result.trace.critic_decisions):
            console.print(f"  [{i+1}] sufficient={c.sufficient}  missing={c.missing_info or '-'}")


@app.command(name="list-reports")
def list_reports() -> None:
    """List all report_ids currently indexed in Qdrant."""
    async def _run() -> list[str]:
        store = QdrantStore()
        await store.ensure_collection()
        seen: set[str] = set()
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
                    seen.add(rid)
            if next_offset is None:
                break
        return sorted(seen)

    ids = asyncio.run(_run())
    if not ids:
        console.print("[yellow]No reports indexed.[/yellow]")
        return
    for rid in ids:
        console.print(f"  - {rid}")


@app.command()
def delete(report_id: str = typer.Argument(...)) -> None:
    """Delete all chunks for a given report_id."""
    async def _run() -> None:
        store = QdrantStore()
        await store.delete_report(report_id)

    asyncio.run(_run())
    console.print(f"[green]✓[/green] Deleted [cyan]{report_id}[/cyan]")


@app.command()
def batch(
    queries_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True,
                                        help="Text file with one query per line. Blank lines and #-comments are ignored."),
    report_id: list[str] = typer.Option(..., "--report-id", "-r",
                                        help="One or more report_ids to query against."),
    output: Path = typer.Option(Path("batch_report.pdf"), "--output", "-o",
                                help="Output PDF path. Defaults to ./batch_report.pdf."),
) -> None:
    """Run a batch of queries and write a PDF audit report."""
    # Parse queries
    lines = queries_file.read_text(encoding="utf-8").splitlines()
    queries = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not queries:
        console.print("[red]No queries found in the file[/red] (all lines empty or #-commented)")
        raise typer.Exit(code=1)

    console.print(f"[bold]Running batch of {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}[/bold] "
                  f"against report(s): {', '.join(report_id)}")

    # Progress callback
    def on_query_event(qi: int, q: str, stage: str, data) -> None:
        if stage == "batch_query_start":
            console.print(f"\n[cyan][{qi + 1}/{len(queries)}][/cyan] {q}")
        elif stage == "planner_done":
            console.print(f"    plan: {len(data.subqueries)} subquer"
                          f"{'y' if len(data.subqueries) == 1 else 'ies'}")
        elif stage == "retrieval_done":
            total = sum(len(c) for c in data["chunks_per_sub"])
            console.print(f"    retrieved: {total} chunks")
        elif stage == "critic_done":
            verdict = "sufficient" if data.sufficient else "insufficient → follow-up"
            console.print(f"    critic: {verdict}")
        elif stage == "synthesizer_done":
            if data.answer_available:
                console.print(f"    ✓ answered (confidence: {data.confidence})")
            else:
                console.print("    ⚫ not available")

    outcomes = asyncio.run(answer_queries(
        queries, report_ids=report_id, on_query_event=on_query_event,
    ))

    # Persist per-query audit logs on disk (same as Streamlit does)
    from agentic_rag.query_log import save_query_log
    for outcome in outcomes:
        if outcome.result is not None:
            save_query_log(outcome.query, report_id, outcome.result)

    # PDF
    console.print("\n[bold]Generating PDF report…[/bold]")
    pdf_path = generate_batch_pdf(outcomes, report_id, output)
    console.print(f"[green]✓[/green] Wrote [cyan]{pdf_path}[/cyan]")

    # Summary
    answered = sum(1 for o in outcomes if o.result and o.result.answer.answer_available)
    unavailable = sum(1 for o in outcomes if o.result and not o.result.answer.answer_available)
    errored = sum(1 for o in outcomes if o.error is not None)
    summary = Table(title="Batch summary", show_header=False)
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Total", str(len(outcomes)))
    summary.add_row("Answered", f"[green]{answered}[/green]")
    summary.add_row("Not available", f"[grey50]{unavailable}[/grey50]")
    if errored:
        summary.add_row("Errored", f"[red]{errored}[/red]")
    console.print(summary)


if __name__ == "__main__":
    app()
