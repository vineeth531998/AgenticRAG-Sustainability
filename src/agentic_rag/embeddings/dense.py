"""Dense embeddings via Fireworks.ai.

We reuse the shared `AsyncOpenAI` client from `llm.py` — Fireworks's embeddings
endpoint (`/v1/embeddings`) lives at the same base URL as chat completions, so
no second client needed.

This module is dense-only. The sparse / BM25 leg lives in `retrieval.bm25`.
"""
from __future__ import annotations

from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentic_rag.llm import get_client
from config.settings import settings


@dataclass(frozen=True)
class ChunkVectors:
    """One chunk's full vector representation for upsert / query."""
    dense: list[float]
    sparse: dict[int, float]


class EmbeddingError(RuntimeError):
    pass


async def embed_dense(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one dense vector per input, in order."""
    if not texts:
        return []

    client = get_client().with_options(timeout=settings.embedding_timeout_s)

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            resp = await client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
            )

    vecs = [item.embedding for item in resp.data]
    if len(vecs) != len(texts):
        raise EmbeddingError(
            f"Fireworks returned {len(vecs)} embeddings for {len(texts)} inputs"
        )
    return vecs


async def embed_dense_one(text: str) -> list[float]:
    return (await embed_dense([text]))[0]
