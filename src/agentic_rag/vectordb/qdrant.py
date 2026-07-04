"""Qdrant client wrapper for hybrid (dense + sparse) retrieval.

The collection uses NAMED VECTORS: one dense ("dense") + one sparse ("sparse").
A single Qdrant Query API call runs both with prefetch and fuses via RRF.

We index payload fields that the planner can filter on (report_id, company,
report_year, framework, is_table) so retrieval-time filters are cheap.
"""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from agentic_rag.embeddings.dense import ChunkVectors
from agentic_rag.schemas import Chunk, RetrievalFilter, RetrievedChunk
from config.settings import settings


class QdrantStore:
    def __init__(self) -> None:
        # See settings.py for the three-mode logic. Local file > in-memory > server.
        if settings.qdrant_local_path:
            from pathlib import Path
            Path(settings.qdrant_local_path).mkdir(parents=True, exist_ok=True)
            self.client = AsyncQdrantClient(path=settings.qdrant_local_path)
        elif settings.qdrant_in_memory:
            self.client = AsyncQdrantClient(location=":memory:")
        elif settings.qdrant_url:
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            raise RuntimeError(
                "No Qdrant backend configured. Set QDRANT_LOCAL_PATH (recommended), "
                "QDRANT_IN_MEMORY=true, or QDRANT_URL."
            )
        self.collection = settings.collection_name

    # ---- collection mgmt ----

    async def ensure_collection(self) -> None:
        existing = await self.client.get_collections()
        if any(c.name == self.collection for c in existing.collections):
            return

        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                settings.dense_vector_name: models.VectorParams(
                    size=settings.dense_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                # IDF modifier turns our client-side TF counts into BM25-style
                # scoring (TF * IDF) inside Qdrant. Don't drop this — without
                # it, the sparse leg becomes raw dot-product on TFs and gets
                # dominated by long chunks.
                settings.sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                    modifier=models.Modifier.IDF,
                ),
            },
        )

        # Keyword / integer / bool payload indexes for filtered retrieval.
        for field, schema in [
            ("metadata.report_id", models.PayloadSchemaType.KEYWORD),
            ("metadata.company", models.PayloadSchemaType.KEYWORD),
            ("metadata.framework", models.PayloadSchemaType.KEYWORD),
            ("metadata.report_year", models.PayloadSchemaType.INTEGER),
            ("metadata.is_table", models.PayloadSchemaType.BOOL),
        ]:
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=schema,
            )

        # Explicit case-insensitive text index on the section field so the
        # planner's `section_contains` filter matches "human capital" against
        # both "Human Capital" and "HUMAN CAPITAL". Default PayloadSchemaType
        # .TEXT does NOT guarantee lowercase across Qdrant versions; force it.
        await self.client.create_payload_index(
            collection_name=self.collection,
            field_name="metadata.section",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
                min_token_len=2,
            ),
        )

        # Full-text index on the chunk body so the planner's must_phrases can
        # become hard MatchText filters at retrieval time. This is the
        # BM25-style hard gate that complements the dense+sparse vector pool.
        await self.client.create_payload_index(
            collection_name=self.collection,
            field_name="text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
                min_token_len=2,
            ),
        )

    async def delete_report(self, report_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(
                        key="metadata.report_id",
                        match=models.MatchValue(value=report_id),
                    )
                ])
            ),
        )

    # ---- upsert ----

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[ChunkVectors],
    ) -> None:
        assert len(chunks) == len(vectors), "chunks/vectors length mismatch"

        points = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            points.append(
                models.PointStruct(
                    # Qdrant point IDs must be int or UUID — our chunk_id is a string,
                    # so we derive a deterministic UUID5 from it and keep the original
                    # in the payload.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    vector={
                        settings.dense_vector_name: vec.dense,
                        settings.sparse_vector_name: models.SparseVector(
                            indices=list(vec.sparse.keys()),
                            values=list(vec.sparse.values()),
                        ),
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata.model_dump(mode="json"),
                    },
                )
            )

        # Batch upsert in chunks of 64 to keep request size sane
        for i in range(0, len(points), 64):
            await self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + 64],
                wait=True,
            )

    # ---- query ----

    async def hybrid_search(
        self,
        *,
        dense_query: list[float],
        sparse_query: dict[int, float],
        report_ids: list[str],
        filters: RetrievalFilter,
        must_phrases: list[str] | None = None,
        limit_dense: int,
        limit_sparse: int,
        limit_final: int,
    ) -> list[RetrievedChunk]:
        """Hybrid query: dense prefetch + sparse prefetch fused via RRF.

        `must_phrases` becomes a set of MatchText conditions in the `must` filter,
        i.e. chunks that don't contain those tokens never enter the candidate pool.
        That's the BM25-style hard gate the planner controls.
        """
        q_filter = _build_filter(report_ids, filters, must_phrases or [])

        prefetch = [
            models.Prefetch(
                query=dense_query,
                using=settings.dense_vector_name,
                limit=limit_dense,
                filter=q_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(sparse_query.keys()),
                    values=list(sparse_query.values()),
                ),
                using=settings.sparse_vector_name,
                limit=limit_sparse,
                filter=q_filter,
            ),
        ]

        resp = await self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit_final,
            with_payload=True,
            query_filter=q_filter,
        )

        return [_point_to_retrieved(p) for p in resp.points]


# ---- helpers ----

def _build_filter(
    report_ids: list[str],
    f: RetrievalFilter,
    must_phrases: list[str],
) -> models.Filter | None:
    must: list[Any] = []
    if report_ids:
        must.append(
            models.FieldCondition(
                key="metadata.report_id",
                match=models.MatchAny(any=report_ids),
            )
        )
    # BM25-style hard gate: each phrase becomes a MatchText condition. Multiple
    # phrases are AND-ed. Within one phrase, tokens are also AND-ed by Qdrant's
    # text index. So `["wastewater discharge", "FY2023"]` requires all 3 tokens.
    for phrase in must_phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        must.append(
            models.FieldCondition(
                key="text",
                match=models.MatchText(text=phrase),
            )
        )
    if f.company:
        must.append(
            models.FieldCondition(
                key="metadata.company",
                match=models.MatchValue(value=f.company),
            )
        )
    if f.report_year is not None:
        must.append(
            models.FieldCondition(
                key="metadata.report_year",
                match=models.MatchValue(value=f.report_year),
            )
        )
    if f.framework:
        must.append(
            models.FieldCondition(
                key="metadata.framework",
                match=models.MatchValue(value=f.framework.value),
            )
        )
    if f.section_contains:
        must.append(
            models.FieldCondition(
                key="metadata.section",
                match=models.MatchText(text=f.section_contains),
            )
        )

    return models.Filter(must=must) if must else None


def _point_to_retrieved(p: Any) -> RetrievedChunk:
    """Build a RetrievedChunk from a Qdrant point payload.

    Uses dict-based `model_validate` (not direct constructor calls) so we're
    immune to Pydantic class-identity mismatches. Those happen when Streamlit
    auto-reloads `schemas.py` while a cached QdrantStore is still holding
    Chunk instances built against the previous class object — pydantic then
    rejects those instances as "not a valid Chunk". Validating through a dict
    boundary sidesteps the identity check entirely.
    """
    payload = p.payload or {}
    return RetrievedChunk.model_validate(
        {
            "chunk": {
                "chunk_id": payload["chunk_id"],
                "text": payload["text"],
                "metadata": payload["metadata"],
            },
            "score": float(p.score),
        }
    )
