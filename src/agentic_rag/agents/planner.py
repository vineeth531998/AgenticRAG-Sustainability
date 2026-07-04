"""Planner agent: user query → list of self-contained subqueries."""
from __future__ import annotations

from agentic_rag.agents.prompts import PLANNER_SYSTEM
from agentic_rag.llm import structured_call
from agentic_rag.schemas import PlannerOutput
from config.settings import settings


async def plan(
    user_query: str,
    *,
    enable_hyde: bool | None = None,
    report_context: str | None = None,
) -> PlannerOutput:
    """Decompose the user query into subqueries.

    `report_context` (optional) is a pre-formatted block describing the
    content distribution of the target report(s) — chunk counts by type
    and dominant content style. When provided, the planner adapts its
    decomposition strategy per the CONTENT-AWARE PLANNING section of
    PLANNER_SYSTEM. When None, planner uses default behavior.
    """
    use_hyde = settings.enable_hyde if enable_hyde is None else enable_hyde

    context_block = (
        f"Report context:\n{report_context}\n\n" if report_context else ""
    )

    user_content = (
        f"User question:\n{user_query}\n\n"
        f"{context_block}"
        f"HyDE: {'enabled' if use_hyde else 'disabled — do not include hyde_doc fields'}\n\n"
        "Produce the subquery plan now."
    )

    return await structured_call(
        model=settings.planner_model,
        system_prompt=PLANNER_SYSTEM,
        user_content=user_content,
        output_model=PlannerOutput,
    )
