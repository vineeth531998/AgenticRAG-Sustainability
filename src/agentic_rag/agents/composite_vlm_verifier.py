"""Composite VLM Verifier — independent visual extraction + symmetric merge.

Companion to `table_vlm_verifier` but for composite chunks (prose / list /
infographic-transcribed text). Chandra loses the visual origin when it OCRs
infographic text into `Composite` chunks (only ~10% of visually-derived
Composite content carries `is_infographic=True`). So the composite extractor
working on that flattened text can mis-attribute values, and Chandra's OCR
itself can misread numbers in visually-laid-out regions (dashboard cards,
KPI callouts, board rosters).

Same architecture as the table verifier:

  1. VLM does INDEPENDENT extraction — it never sees the composite
     extractor's guess. That kills confirmation bias.
  2. Both extractors produce `TableValue` (unified schema). No shape
     translation.
  3. Merge logic is shared from `table_vlm_verifier._merge_extractions` —
     one source of truth for how text + VLM readings combine.

Gated by `settings.enable_composite_vlm_verify`.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Callable

from agentic_rag.agents.prompts import COMPOSITE_VLM_VERIFIER_SYSTEM
# Reuse the shared merge logic + small helpers from the table verifier so
# both wings apply identical merge rules. If the merge logic changes there
# it applies here automatically.
from agentic_rag.agents.table_vlm_verifier import (
    _append_note,
    _merge_extractions,
    _truncate_err,
    vlm_reading_to_table_value,
)
from agentic_rag.ingestion.pipeline import stored_pdf_path
from agentic_rag.llm import (
    _force_additional_properties_false,
    _force_all_properties_required,
    get_vlm_config,
)
from agentic_rag.pdf_preview import render_page_crop
from agentic_rag.schemas import (
    Chunk,
    RetrievedChunk,
    TableValue,
    TableVLMVerification,
)
from config.settings import settings


EventCallback = Callable[[str, Any], None] | None


async def verify_and_merge_composite_values(
    values: list[TableValue],
    chunks_by_id: dict[str, RetrievedChunk],
    *,
    on_event: EventCallback = None,
) -> list[TableValue]:
    """Independently extract via VLM, then merge with the text-extracted value.

    For each composite-extracted value: crop the source chunk's bbox and ask
    the VLM to independently find the target in the visual layout (prose,
    list, dashboard card, or infographic). Wrap VLM's reading into a
    TableValue with source="vlm", then merge symmetrically with the text
    TableValue via the shared merge function.

    Values without a bbox or without a persisted PDF pass through untouched.
    """
    if not values:
        return values

    sem = asyncio.Semaphore(max(1, settings.vlm_concurrency))

    async def _one(text_val: TableValue) -> TableValue:
        retrieved = chunks_by_id.get(text_val.chunk_id)
        if retrieved is None:
            print(
                f"[composite-vlm-verify] SKIP chunk_not_found "
                f"target={text_val.target_description!r} chunk={text_val.chunk_id}",
                flush=True,
            )
            _emit(on_event, "composite_vlm_verify_skip",
                  {"target": text_val.target_description, "reason": "chunk_not_found"})
            return text_val

        chunk = retrieved.chunk
        if not chunk.metadata.bbox:
            print(
                f"[composite-vlm-verify] SKIP no_bbox "
                f"target={text_val.target_description!r} chunk={text_val.chunk_id}",
                flush=True,
            )
            _emit(on_event, "composite_vlm_verify_skip",
                  {"target": text_val.target_description, "reason": "no_bbox"})
            return text_val

        pdf_path = stored_pdf_path(chunk.metadata.report_id)
        if not pdf_path.exists():
            print(
                f"[composite-vlm-verify] SKIP pdf_not_stored "
                f"target={text_val.target_description!r} expected={pdf_path}",
                flush=True,
            )
            _emit(on_event, "composite_vlm_verify_skip",
                  {"target": text_val.target_description, "reason": "pdf_not_stored",
                   "expected_pdf_path": str(pdf_path)})
            return text_val

        async with sem:
            print(
                f"[composite-vlm-verify] RUNNING target={text_val.target_description!r} "
                f"chunk={text_val.chunk_id} page={chunk.metadata.page} "
                f"text_val_confidence={text_val.confidence} "
                f"text_val_value={text_val.value!r}",
                flush=True,
            )
            _emit(on_event, "composite_vlm_verify_start",
                  {"target": text_val.target_description, "original": text_val})

            try:
                # Independent extraction — VLM never sees text_val's guess.
                vlm_reading = await _run_vlm_composite(
                    text_val.target_description, chunk,
                )
            except Exception as e:  # noqa: BLE001
                annotated = text_val.model_copy(update={
                    "note": _append_note(
                        text_val.note,
                        f"Composite VLM extraction failed ({type(e).__name__}): "
                        f"{_truncate_err(str(e))}",
                    ),
                })
                print(
                    f"[composite-vlm-verify] ERROR "
                    f"target={text_val.target_description!r} "
                    f"{type(e).__name__}: {e}", flush=True,
                )
                _emit(on_event, "composite_vlm_verify_error",
                      {"target": text_val.target_description, "error": str(e)})
                return annotated

            vlm_val = vlm_reading_to_table_value(
                text_val.chunk_id, text_val.target_description, vlm_reading,
            )
            merged = _merge_extractions(text_val, vlm_val)
            print(
                f"[composite-vlm-verify] DONE target={text_val.target_description!r} "
                f"vlm_found={vlm_val.found} vlm_confidence={vlm_val.confidence} "
                f"vlm_value={vlm_val.value!r} → merged_confidence={merged.confidence} "
                f"merged_value={merged.value!r}",
                flush=True,
            )
            _emit(on_event, "composite_vlm_verify_done",
                  {"target": text_val.target_description,
                   "original": text_val, "verified": vlm_val, "merged": merged})
            return merged

    return await asyncio.gather(*(_one(tv) for tv in values))


# ── Extract-from-unfound path ──────────────────────────────────────────────

async def vlm_extract_unfound_composite_targets(
    unfound: list[str],
    composite_chunks: list[RetrievedChunk],
    *,
    max_chunks_per_target: int = 2,
    on_event: EventCallback = None,
) -> tuple[list[TableValue], list[str]]:
    """For each target the Composite Extractor gave up on, try VLM extraction
    on the top-N composite chunks (independent extraction, no prior guess).

    Symmetric with `vlm_extract_unfound_targets` in table_vlm_verifier — same
    concurrency model (parallel across targets, serial with break-on-hit
    within each target), same TableValue output schema, same VLM prompt
    (COMPOSITE_VLM_VERIFIER_SYSTEM with segment-vs-combined rule).

    Returns (rescued_values, still_unfound). Rescued values are TableValues
    with `source="vlm"` and a note marking them as composite-extract-from-
    unfound rescues.

    `composite_chunks` should already be reranker-sorted (highest relevance
    first).
    """
    if not unfound or not composite_chunks:
        return [], list(unfound)

    sem = asyncio.Semaphore(max(1, settings.vlm_concurrency))

    async def _process_target(target: str) -> TableValue | None:
        for retrieved in composite_chunks[:max_chunks_per_target]:
            chunk = retrieved.chunk

            if not chunk.metadata.bbox:
                continue
            pdf_path = stored_pdf_path(chunk.metadata.report_id)
            if not pdf_path.exists():
                continue

            async with sem:
                print(
                    f"[composite-vlm-extract] RUNNING target={target!r} "
                    f"chunk={chunk.chunk_id} page={chunk.metadata.page}",
                    flush=True,
                )
                _emit(on_event, "composite_vlm_extract_start",
                      {"target": target, "chunk_id": chunk.chunk_id})

                try:
                    vlm_reading = await _run_vlm_composite(target, chunk)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[composite-vlm-extract] ERROR target={target!r} "
                        f"{type(e).__name__}: {e}", flush=True,
                    )
                    _emit(on_event, "composite_vlm_extract_error",
                          {"target": target, "error": str(e)})
                    continue  # try next chunk for this target

            if not vlm_reading.found:
                print(
                    f"[composite-vlm-extract] NOT_FOUND target={target!r} "
                    f"chunk={chunk.chunk_id}"
                    + (f" — VLM note: {vlm_reading.note}" if vlm_reading.note else ""),
                    flush=True,
                )
                _emit(on_event, "composite_vlm_extract_not_found",
                      {"target": target, "chunk_id": chunk.chunk_id,
                       "vlm_note": vlm_reading.note})
                continue  # try next chunk

            # VLM found it — wrap into a TableValue with source="vlm".
            rescued_value = TableValue(
                chunk_id=chunk.chunk_id,
                target_description=target,
                source="vlm",
                found=True,
                row_label=vlm_reading.row_label or "?",
                column_label=vlm_reading.column_label or "?",
                value=vlm_reading.value or "?",
                unit=vlm_reading.unit,
                confidence=vlm_reading.confidence or "medium",
                note=_append_note(
                    vlm_reading.note,
                    "VLM-extracted from unfound (composite extractor could not "
                    "locate this value in the chunk text)",
                ),
            )
            print(
                f"[composite-vlm-extract] DONE target={target!r} "
                f"value={rescued_value.value!r} "
                f"confidence={rescued_value.confidence}",
                flush=True,
            )
            _emit(on_event, "composite_vlm_extract_done",
                  {"target": target, "verified": vlm_reading, "result": rescued_value})
            return rescued_value

        return None  # exhausted all chunks without a hit

    # Fan out across targets; gather preserves order so we can zip back.
    results = await asyncio.gather(*(_process_target(t) for t in unfound))

    rescued: list[TableValue] = []
    still_unfound: list[str] = []
    for target, result in zip(unfound, results, strict=True):
        if result is None:
            still_unfound.append(target)
        else:
            rescued.append(result)

    return rescued, still_unfound


# ── VLM call (INDEPENDENT — no prior guess ever shown) ──────────────────────

async def _run_vlm_composite(
    target_desc: str, chunk: Chunk,
) -> TableVLMVerification:
    """Crop the composite chunk's region and INDEPENDENTLY extract the target.

    VLM sees only the target and the cropped region. It commits to a reading
    from the visual alone. External code compares with the text extractor's
    reading afterwards.
    """
    assert chunk.metadata.bbox is not None
    pdf_path = stored_pdf_path(chunk.metadata.report_id)

    # 60px padding matches the table verifier — composite regions often
    # carry section headings, adjacent labels, or list bullets just outside
    # the tight bbox that Chandra emitted.
    png_bytes = render_page_crop(
        pdf_path, chunk.metadata.page, chunk.metadata.bbox, padding_px=60
    )
    b64 = base64.b64encode(png_bytes).decode("ascii")

    prompt_body = (
        f"Target cell to extract:\n  {target_desc}\n\n"
        "Read the CROPPED REGION in the image and INDEPENDENTLY identify the "
        "value that answers the target. The region may show prose, a list, "
        "a dashboard card, an infographic, or a mix. Return your reading "
        "VERBATIM with your own confidence assessment. Return `found=false` "
        "ONLY if the value is genuinely not present in the region, or if "
        "the target implies a value the source does not disclose as a "
        "single reading (see the segment-vs-combined rule in your "
        "instructions)."
    )

    user_content = [
        {"type": "text", "text": prompt_body},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]

    schema = _schema_for_verification()

    client, model, max_tokens = get_vlm_config()
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": COMPOSITE_VLM_VERIFIER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "TableVLMVerification",
                "schema": schema,
                "strict": True,
            },
        },
    )

    if not resp.choices:
        raise RuntimeError(f"Composite VLM returned no choices: {resp!r}")
    text = resp.choices[0].message.content
    if not text:
        raise RuntimeError(
            f"Composite VLM returned empty content "
            f"(finish_reason={resp.choices[0].finish_reason!r})"
        )
    return TableVLMVerification.model_validate(json.loads(text))


def _schema_for_verification() -> dict[str, Any]:
    schema = TableVLMVerification.model_json_schema()
    _force_additional_properties_false(schema)
    _force_all_properties_required(schema)  # strict-mode compliance for Groq
    return schema


def _emit(cb: EventCallback, stage: str, data: Any) -> None:
    if cb is None:
        return
    try:
        cb(stage, data)
    except Exception:  # noqa: BLE001
        pass
