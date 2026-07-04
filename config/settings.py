from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider (Fireworks via their OpenAI-compatible endpoint)
    fireworks_api_key: str = Field(..., alias="FIREWORKS_API_KEY")
    fireworks_base_url: str = Field(
        "https://api.fireworks.ai/inference/v1",
        alias="FIREWORKS_BASE_URL",
    )

    # VLM provider (Groq via their OpenAI-compatible endpoint) — used ONLY
    # for vision calls (ingestion image description + query-time verifier
    # and extract-from-unfound in table & composite wings). Text LLMs
    # (planner, critic, synthesizer, extractors) still route to Fireworks.
    # Toggle `use_groq_vlm=false` to fall back to Fireworks's multimodal
    # model everywhere; useful if Groq is down or the current vision model
    # underperforms on a specific report type.
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    groq_base_url: str = Field(
        "https://api.groq.com/openai/v1",
        alias="GROQ_BASE_URL",
    )
    groq_vlm_model: str = Field(
        "meta-llama/llama-4-scout-17b-16e-instruct",
        alias="GROQ_VLM_MODEL",
    )
    # Groq's Llama vision models are non-thinking — they don't burn tokens
    # on internal reasoning before emitting JSON, so we don't need the 16k
    # headroom Fireworks's Qwen3.7-Plus needed. 4000 is plenty for our
    # structured outputs (TableVLMVerification schema).
    groq_vlm_max_tokens: int = Field(4000, alias="GROQ_VLM_MAX_TOKENS")
    use_groq_vlm: bool = Field(True, alias="USE_GROQ_VLM")

    chandra_ocr_url: str = Field(..., alias="CHANDRA_OCR_URL")
    chandra_ocr_timeout_s: int = Field(600, alias="CHANDRA_OCR_TIMEOUT_S")
    # Per-page fan-out to Modal. Should match Modal's @concurrent(max_inputs=...)
    # and vLLM's max_num_seqs. Default mirrors Chandra's published benchmark config.
    chandra_page_concurrency: int = Field(96, alias="CHANDRA_PAGE_CONCURRENCY")

    # Qdrant: three mutually-exclusive modes, picked in this priority order:
    #   1. qdrant_local_path  → persistent local file store (no Docker, survives restart)
    #   2. qdrant_in_memory   → pure RAM (lost on process exit)
    #   3. qdrant_url         → talk to a real server
    qdrant_local_path: str | None = Field("./data/qdrant_local", alias="QDRANT_LOCAL_PATH")
    qdrant_in_memory: bool = Field(False, alias="QDRANT_IN_MEMORY")
    qdrant_url: str | None = Field(None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(None, alias="QDRANT_API_KEY")

    planner_model: str = Field("accounts/fireworks/models/qwen3p7-plus", alias="PLANNER_MODEL")
    critic_model: str = Field("accounts/fireworks/models/qwen3p7-plus", alias="CRITIC_MODEL")
    synthesizer_model: str = Field("accounts/fireworks/models/qwen3p7-plus", alias="SYNTHESIZER_MODEL")

    # Dense embeddings (Fireworks, served via /v1/embeddings on the same
    # OpenAI-compatible endpoint as the chat models).
    embedding_model: str = Field(
        "accounts/fireworks/models/qwen3-embedding-8b",
        alias="EMBEDDING_MODEL",
    )
    embedding_timeout_s: int = Field(60, alias="EMBEDDING_TIMEOUT_S")
    embedding_batch_size: int = Field(32, alias="EMBEDDING_BATCH_SIZE")

    # Reranker (Fireworks Qwen3-Reranker, /v1/rerank). One batched call per
    # subquery returns relevance scores for all candidates, sorted.
    reranker_model: str = Field(
        "accounts/fireworks/models/qwen3-reranker-8b",
        alias="RERANKER_MODEL",
    )
    reranker_doc_max_chars: int = Field(2000, alias="RERANKER_DOC_MAX_CHARS")
    reranker_timeout_s: int = Field(60, alias="RERANKER_TIMEOUT_S")

    # VLM extraction for infographics / charts / dashboard cards that Chandra
    # tagged as `Image` but didn't OCR. Runs at ingestion after Chandra
    # returns per-page chunks — crops the page at the Image bbox and sends to
    # a vision-capable model. Default model is the same qwen3p7-plus we use
    # for the agents (Fireworks confirmed it accepts multimodal image_url
    # content). Set enable_vlm_extraction=False to skip entirely.
    enable_vlm_extraction: bool = Field(True, alias="ENABLE_VLM_EXTRACTION")
    vlm_model: str = Field(
        "accounts/fireworks/models/qwen3p7-plus",
        alias="VLM_MODEL",
    )
    vlm_concurrency: int = Field(4, alias="VLM_CONCURRENCY")
    # Qwen3-VL and similar thinking-VL models spend a lot of tokens on
    # internal reasoning before emitting the JSON body. 16000 matches the
    # planner default and gives it plenty of room to think AND finish the
    # structured output; bump higher if you STILL see `finish_reason='length'`
    # errors in the VLM verify trace (or switch to a non-thinking VL model).
    vlm_max_tokens: int = Field(16000, alias="VLM_MAX_TOKENS")

    # Table VLM verifier — MANDATORY visual verification of every text-extracted
    # cell. When ON, every value produced by the text-based Table Extractor is
    # cross-checked against the cropped table image (fanned out concurrently,
    # bounded by vlm_concurrency). This exists because Chandra's markdown
    # flattener silently collapses multi-row / spanning / merged headers, and
    # the text extractor cannot detect its own upstream error — it will happily
    # report `confidence=high` on a value that belongs to the wrong column.
    # Only the visual layout catches that class of bug.
    # Requires: chunk has a bbox AND the source PDF is persisted (any cell
    # without both passes through unverified).
    enable_table_vlm_verify: bool = Field(True, alias="ENABLE_TABLE_VLM_VERIFY")

    # Table VLM extract-from-unfound — the more aggressive fallback. When the
    # text-based Table Extractor completely fails to locate a target cell
    # (returns it as "unfound"), send the top table images to the VLM to try
    # extracting the cell from scratch. Catches the case where markdown table
    # structure is too broken for text parsing but a vision model can read
    # the actual visual layout.
    enable_table_vlm_extract_unfound: bool = Field(
        True, alias="ENABLE_TABLE_VLM_EXTRACT_UNFOUND"
    )

    # Composite extraction — the parallel wing for prose / list / infographic-
    # transcribed chunks (Chandra label=Composite). Fires alongside the table
    # wing for every factual_lookup subquery. Chandra puts most report content
    # (including infographic-extracted values, named rosters, callout numbers)
    # into Composite chunks, so this wing carries the majority of factual
    # answers for non-BRSR reports. Single LLM call per subquery (no VLM),
    # runs in parallel with the table wing.
    enable_composite_extraction: bool = Field(
        True, alias="ENABLE_COMPOSITE_EXTRACTION"
    )
    # After rerank, only the top-N non-table chunks per subquery are handed
    # to the composite extractor. Bounded to keep per-query LLM cost
    # predictable — 5 gives enough breadth without ballooning prompt size.
    composite_extract_top_n: int = Field(5, alias="COMPOSITE_EXTRACT_TOP_N")

    # Composite VLM verifier — symmetric with the table verifier, but for
    # composite-extracted values. Chandra loses the visual origin when it
    # OCRs infographic-derived text into Composite chunks (the flag
    # `is_infographic` only covers ~10% of visually-derived Composite
    # content). To close that gap we visually verify EVERY composite-
    # extracted value against its source chunk's cropped bbox. Same
    # semaphore-bounded concurrency and shared merge logic as the table
    # verifier. Runs in parallel with the table wing's VLM chain.
    # Set to false to skip and save cost.
    enable_composite_vlm_verify: bool = Field(
        True, alias="ENABLE_COMPOSITE_VLM_VERIFY"
    )

    # Composite VLM extract-from-unfound — symmetric with the table wing's
    # rescue path. When the Composite Extractor gives up on a target (returns
    # it in `unfound`), send the top composite chunk images to the VLM to
    # try extracting the value from scratch. Independent extraction (no prior
    # guess). Catches the case where the composite prose is ambiguous or the
    # infographic-transcribed text lost enough structure that the text
    # extractor couldn't lock onto a value, but the visual layout is still
    # legible.
    enable_composite_vlm_extract_unfound: bool = Field(
        True, alias="ENABLE_COMPOSITE_VLM_EXTRACT_UNFOUND"
    )

    top_k_dense: int = Field(30, alias="TOP_K_DENSE")
    top_k_sparse: int = Field(30, alias="TOP_K_SPARSE")
    top_k_rerank: int = Field(10, alias="TOP_K_RERANK")
    max_critic_iterations: int = Field(2, alias="MAX_CRITIC_ITERATIONS")
    enable_hyde: bool = Field(False, alias="ENABLE_HYDE")

    collection_name: str = "sustainability_chunks"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    dense_dim: int = Field(4096, alias="DENSE_DIM")  # qwen3-embedding-8b → 4096


settings = Settings()  # type: ignore[call-arg]
