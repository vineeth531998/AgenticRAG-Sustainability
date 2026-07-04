"""Reranker via Fireworks /v1/rerank (Qwen3-Reranker-8B by default).

Fireworks's rerank endpoint takes a single query + a list of documents and
returns a list of `{index, relevance_score}` entries sorted by score. One HTTP
call per subquery, no logprob parsing, no per-pair fan-out. Much simpler and
simpler than the per-pair logprob-yes/no approach a local reranker would need.

Wire format:
    POST {fireworks_base_url}/rerank
    Authorization: Bearer <key>
    Content-Type: application/json
    {
        "model": "accounts/fireworks/models/qwen3-reranker-8b",
        "query": "<the user's subquery>",
        "documents": ["<doc1>", "<doc2>", ...],
        "top_n": 10
    }
    →
    {
        "object": "list",
        "model": "...",
        "data": [
            {"index": 3, "relevance_score": 0.92, "document": "..."},
            {"index": 0, "relevance_score": 0.87, "document": "..."},
            ...
        ],
        "usage": {...}
    }

Note: Fireworks uses OpenAI's list shape (top-level `data`), not Cohere's
(`results`). We accept both for robustness.
"""
from __future__ import annotations

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentic_rag.schemas import RetrievedChunk
from config.settings import settings


class RerankerError(RuntimeError):
    pass


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


async def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Score every candidate against the query, return the top_k by relevance."""
    if not candidates:
        return []

    documents = [
        _truncate(c.chunk.text, settings.reranker_doc_max_chars) for c in candidates
    ]
    payload = {
        "model": settings.reranker_model,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(candidates)),
    }
    headers = {
        "Authorization": f"Bearer {settings.fireworks_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.fireworks_base_url.rstrip('/')}/rerank"

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=1, max=5),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    ):
        with attempt:
            async with httpx.AsyncClient(timeout=settings.reranker_timeout_s) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if not resp.is_success:
                    # Surface the error detail — without this we'd just see
                    # the bare HTTP status with no hint as to what's wrong.
                    try:
                        body = resp.json()
                        detail = body.get("error") or body
                    except Exception:  # noqa: BLE001
                        detail = (resp.text or "")[:600]
                    raise httpx.HTTPStatusError(
                        f"Fireworks rerank {resp.status_code}: {detail}",
                        request=resp.request,
                        response=resp,
                    )
                data = resp.json()

    # Fireworks's rerank endpoint returns `data: [...]` (OpenAI list shape).
    # Some Cohere-compatible providers use `results: [...]` — accept either.
    results = data.get("data") or data.get("results") or []
    if not results:
        raise RerankerError(f"Fireworks rerank returned no results: {data!r}")

    out: list[RetrievedChunk] = []
    for r in results:
        idx = r.get("index")
        score = r.get("relevance_score")
        if idx is None or score is None:
            continue
        if not (0 <= idx < len(candidates)):
            continue
        original = candidates[idx]
        out.append(
            original.model_copy(
                update={"rerank_score": float(score), "score": float(score)}
            )
        )
    return out[:top_k]
