"""Citation renumbering — turns [^chunk_id] markers into [1] [2] ordered by
first appearance, and returns the ordered unique Citation list.

Shared by the Streamlit UI and the PDF batch report so both renderings stay
in lock-step.
"""
from __future__ import annotations

import re

from agentic_rag.schemas import Citation

_MARKER_RE = re.compile(r"\[\^([^\]\s]+)\]")


def renumber(
    answer: str, citations: list[Citation]
) -> tuple[str, list[Citation]]:
    """Rewrite inline [^chunk_id] markers as [1], [2], … in display order.

    Returns (new_answer, ordered_unique_citations). Chunk_ids that don't
    appear in the citations list still get numbered — we don't drop them,
    because the answer already relies on them; they just won't have a source
    entry.
    """
    seen: dict[str, int] = {}
    ordered: list[Citation] = []
    cit_by_id = {c.chunk_id: c for c in citations}

    def replace(match: "re.Match[str]") -> str:
        cid = match.group(1)
        if cid not in seen:
            seen[cid] = len(seen) + 1
            if cid in cit_by_id:
                ordered.append(cit_by_id[cid])
        return f" [{seen[cid]}]"

    new_answer = _MARKER_RE.sub(replace, answer)
    # Collapse any double spaces introduced by the leading-space in replacement.
    new_answer = re.sub(r" +", " ", new_answer)
    return new_answer.strip(), ordered
