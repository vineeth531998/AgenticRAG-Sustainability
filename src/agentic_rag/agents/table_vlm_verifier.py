"""Table VLM Verifier — independent visual extraction + symmetric merge.

The text-based Table Extractor reads Chandra's markdown/HTML representation of
each table. This works when the markdown is faithful — but there is a
systematic failure mode it cannot detect: multi-row / merged / spanning
headers. When a header row like `[Male | Female | Others]` spans two
sub-columns each (`[Number | Median]`), Chandra's markdown flattener collapses
the hierarchy. The extractor still reports `confidence=high` because the
markdown LOOKS valid — but its column labels are shifted, and the value it
returns belongs to the wrong dimension.

═══════════════════════════════════════════════════════════════════════════════
INDEPENDENT EXTRACTION (not confirmation-biased "verification")
═══════════════════════════════════════════════════════════════════════════════

The VLM does NOT see the text extractor's guess. It receives only the target
and the cropped image, and produces its own independent reading. Then we
compare two symmetric extractions — one from the text extractor, one from
the VLM — and merge with a rule set that treats both sources equally.

This kills the confirmation bias where the VLM was previously "verifying"
against a number it could see in the prompt: given `value='1,337,472.73'` in
the guess, VLM would scan the image, find that exact number (which IS in the
image, just for the wrong dimension), and confirm. Under independent
extraction, the VLM commits to a value FROM the image, and only then do we
compare with the text extractor.

═══════════════════════════════════════════════════════════════════════════════
UNIFIED SCHEMA — both extractors produce `TableValue`
═══════════════════════════════════════════════════════════════════════════════

`TableValue` now has `source: "text" | "vlm" | "merged"` and `found: bool`.
The text extractor produces `TableValue(source="text", found=True)`. The VLM
extractor produces `TableValue(source="vlm", found=True_or_False)`. The
merge function compares two `TableValue`s field-by-field on the same schema.

═══════════════════════════════════════════════════════════════════════════════
MERGE RULES (see `_merge_extractions`)
═══════════════════════════════════════════════════════════════════════════════

  R1. VLM not found          → keep text; downgrade to `low` if VLM had a note
                               (VLM saw a real reason it couldn't answer)
  R2. Values agree + labels agree + no VLM concern
                             → mutual vote, upgrade to `high`
  R3. Values agree + VLM note is NON-EMPTY (semantic concern)
                             → keep text; confidence = `medium` (agreement but
                               VLM flagged something worth attention)
  R4. Values agree + VLM's row/col label reveals a SEGMENT the text didn't name
                             → DISPUTED-LABELS: keep text; downgrade to `low`,
                               surface the label mismatch in the note. This is
                               the shifted-header case where both saw the same
                               NUMBER (e.g. 1,337,472.73) but VLM correctly
                               labelled it "Male" while text called it "Total".
  R5. Values differ + one is HIGH and other isn't
                             → the HIGH-confidence reading wins (REPLACE if VLM,
                               keep-with-hedge if text). VLM winning is the
                               shifted-header rescue.
  R6. Values differ + BOTH high
                             → DISPUTED — DO NOT blindly override. Keep text,
                               downgrade to `low`, both readings in note.
  R7. Values differ + neither high
                             → DISPUTED — keep text, downgrade to `low`, both
                               readings in note.

The public entry point `verify_and_merge_table_values` fans out all
verifications concurrently (respecting `settings.vlm_concurrency`) and returns
the merged list.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
from typing import Any, Callable

from agentic_rag.agents.prompts import TABLE_VLM_VERIFIER_SYSTEM
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


# ── Public API ──────────────────────────────────────────────────────────────

async def verify_and_merge_table_values(
    values: list[TableValue],
    chunks_by_id: dict[str, RetrievedChunk],
    *,
    on_event: EventCallback = None,
) -> list[TableValue]:
    """Independently extract every value via VLM and merge symmetrically.

    For each text-extracted value: crop its source chunk's bbox, ask the VLM
    to independently extract the same target (WITHOUT showing the text
    extractor's guess), wrap the VLM's reading into a `TableValue` with
    `source="vlm"`, then merge symmetrically with the text `TableValue`.

    Values without a bbox or without a persisted PDF pass through untouched.
    Concurrency-bounded by `settings.vlm_concurrency`.
    """
    if not values:
        return values

    sem = asyncio.Semaphore(max(1, settings.vlm_concurrency))

    async def _one(text_val: TableValue) -> TableValue:
        # Fast-path skips (no image available). These pass through untouched.
        retrieved = chunks_by_id.get(text_val.chunk_id)
        if retrieved is None:
            print(
                f"[vlm-verify] SKIP chunk_not_found "
                f"target={text_val.target_description!r} chunk={text_val.chunk_id}",
                flush=True,
            )
            _emit(on_event, "table_vlm_verify_skip",
                  {"target": text_val.target_description, "reason": "chunk_not_found"})
            return text_val

        chunk = retrieved.chunk
        if not chunk.metadata.bbox:
            print(
                f"[vlm-verify] SKIP no_bbox "
                f"target={text_val.target_description!r} chunk={text_val.chunk_id}",
                flush=True,
            )
            _emit(on_event, "table_vlm_verify_skip",
                  {"target": text_val.target_description, "reason": "no_bbox"})
            return text_val

        pdf_path = stored_pdf_path(chunk.metadata.report_id)
        if not pdf_path.exists():
            print(
                f"[vlm-verify] SKIP pdf_not_stored "
                f"target={text_val.target_description!r} expected={pdf_path}",
                flush=True,
            )
            _emit(on_event, "table_vlm_verify_skip",
                  {"target": text_val.target_description, "reason": "pdf_not_stored",
                   "expected_pdf_path": str(pdf_path)})
            return text_val

        async with sem:
            print(
                f"[vlm-verify] RUNNING target={text_val.target_description!r} "
                f"chunk={text_val.chunk_id} page={chunk.metadata.page} "
                f"text_val_confidence={text_val.confidence} "
                f"text_val_value={text_val.value!r}",
                flush=True,
            )
            _emit(on_event, "table_vlm_verify_start",
                  {"target": text_val.target_description, "original": text_val})

            try:
                # Independent extraction — VLM does NOT see text_val's guess.
                vlm_reading = await _run_vlm(text_val.target_description, chunk)
            except Exception as e:  # noqa: BLE001
                annotated = text_val.model_copy(update={
                    "note": _append_note(
                        text_val.note,
                        f"VLM extraction failed ({type(e).__name__}): "
                        f"{_truncate_err(str(e))}",
                    ),
                })
                print(
                    f"[vlm-verify] ERROR target={text_val.target_description!r} "
                    f"{type(e).__name__}: {e}", flush=True,
                )
                _emit(on_event, "table_vlm_verify_error",
                      {"target": text_val.target_description, "error": str(e)})
                return annotated

            # Wrap VLM's reading into TableValue with source="vlm" so merge
            # compares two TableValues on the same schema.
            vlm_val = vlm_reading_to_table_value(
                text_val.chunk_id, text_val.target_description, vlm_reading,
            )
            merged = _merge_extractions(text_val, vlm_val)
            print(
                f"[vlm-verify] DONE target={text_val.target_description!r} "
                f"vlm_found={vlm_val.found} vlm_confidence={vlm_val.confidence} "
                f"vlm_value={vlm_val.value!r} → merged_confidence={merged.confidence} "
                f"merged_value={merged.value!r}",
                flush=True,
            )
            _emit(on_event, "table_vlm_verify_done",
                  {"target": text_val.target_description,
                   "original": text_val, "verified": vlm_val, "merged": merged})
            return merged

    return await asyncio.gather(*(_one(tv) for tv in values))


# ── Extract-from-unfound path (unchanged semantics) ─────────────────────────

async def vlm_extract_unfound_targets(
    unfound: list[str],
    table_chunks: list[RetrievedChunk],
    *,
    max_chunks_per_target: int = 2,
    on_event: EventCallback = None,
) -> tuple[list[TableValue], list[str]]:
    """For each unfound target, try VLM extraction on the top N table chunks.

    Returns (rescued_values, still_unfound). Rescued values are TableValues
    with `source="vlm"` and a note marking them as extract-from-unfound
    rescues.

    Concurrency model: targets run IN PARALLEL (bounded by `vlm_concurrency`),
    but the chunk fallback WITHIN each target is sequential with break-on-hit.
    """
    if not unfound or not table_chunks:
        return [], list(unfound)

    sem = asyncio.Semaphore(max(1, settings.vlm_concurrency))

    async def _process_target(target: str) -> TableValue | None:
        for retrieved in table_chunks[:max_chunks_per_target]:
            chunk = retrieved.chunk

            if not chunk.metadata.bbox:
                continue
            pdf_path = stored_pdf_path(chunk.metadata.report_id)
            if not pdf_path.exists():
                continue

            async with sem:
                print(
                    f"[vlm-extract] RUNNING target={target!r} "
                    f"chunk={chunk.chunk_id} page={chunk.metadata.page}",
                    flush=True,
                )
                _emit(on_event, "table_vlm_extract_start",
                      {"target": target, "chunk_id": chunk.chunk_id})

                try:
                    vlm_reading = await _run_vlm(target, chunk)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[vlm-extract] ERROR target={target!r} "
                        f"{type(e).__name__}: {e}", flush=True,
                    )
                    _emit(on_event, "table_vlm_extract_error",
                          {"target": target, "error": str(e)})
                    continue

            if not vlm_reading.found:
                print(
                    f"[vlm-extract] NOT_FOUND target={target!r} "
                    f"chunk={chunk.chunk_id}"
                    + (f" — VLM note: {vlm_reading.note}" if vlm_reading.note else ""),
                    flush=True,
                )
                _emit(on_event, "table_vlm_extract_not_found",
                      {"target": target, "chunk_id": chunk.chunk_id,
                       "vlm_note": vlm_reading.note})
                continue

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
                    "VLM-extracted from unfound (text extractor could not "
                    "locate this cell in the markdown)",
                ),
            )
            print(
                f"[vlm-extract] DONE target={target!r} value={rescued_value.value!r} "
                f"confidence={rescued_value.confidence}", flush=True,
            )
            _emit(on_event, "table_vlm_extract_done",
                  {"target": target, "verified": vlm_reading, "result": rescued_value})
            return rescued_value

        return None

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

async def _run_vlm(target_desc: str, chunk: Chunk) -> TableVLMVerification:
    """Crop the chunk region and ask the VLM to INDEPENDENTLY extract the target.

    VLM is NEVER shown the text extractor's guess — that would introduce
    confirmation bias. The VLM commits to a reading from the image alone,
    and only then does the caller compare it with the text extractor's
    value externally.
    """
    assert chunk.metadata.bbox is not None
    pdf_path = stored_pdf_path(chunk.metadata.report_id)

    # 60px padding — multi-row / merged headers (Male/Female over Number/Median)
    # live in the top strip and get clipped when Chandra's bbox is tight.
    png_bytes = render_page_crop(
        pdf_path, chunk.metadata.page, chunk.metadata.bbox, padding_px=60
    )
    b64 = base64.b64encode(png_bytes).decode("ascii")

    prompt_body = (
        f"Target cell to extract:\n  {target_desc}\n\n"
        "Read the VISUAL table in the image and independently identify the "
        "cell that answers the target. Return your reading VERBATIM (row_label, "
        "column_label, value, unit) with your own confidence assessment. "
        "Return `found=false` ONLY if the cell is genuinely not present in the "
        "image, or if the target implies a value the table does not disclose "
        "as a single cell (see the segment-vs-combined rule in your instructions)."
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
            {"role": "system", "content": TABLE_VLM_VERIFIER_SYSTEM},
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
        raise RuntimeError(f"VLM returned no choices: {resp!r}")
    text = resp.choices[0].message.content
    if not text:
        raise RuntimeError(
            f"VLM returned empty content "
            f"(finish_reason={resp.choices[0].finish_reason!r})"
        )
    return TableVLMVerification.model_validate(json.loads(text))


def _schema_for_verification() -> dict[str, Any]:
    schema = TableVLMVerification.model_json_schema()
    _force_additional_properties_false(schema)
    _force_all_properties_required(schema)  # strict-mode compliance for Groq
    return schema


def vlm_reading_to_table_value(
    chunk_id: str, target_description: str, reading: TableVLMVerification,
) -> TableValue:
    """Wrap a VLM's TableVLMVerification into a TableValue with source="vlm".

    Public because composite_vlm_verifier reuses this pattern.
    """
    return TableValue(
        chunk_id=chunk_id,
        target_description=target_description,
        source="vlm",
        found=reading.found,
        row_label=reading.row_label,
        column_label=reading.column_label,
        value=reading.value,
        unit=reading.unit,
        confidence=reading.confidence or "low",
        note=reading.note,
    )


# ── Symmetric merge over two TableValues (text + vlm) ───────────────────────

def _merge_extractions(text_val: TableValue, vlm_val: TableValue) -> TableValue:
    """Merge one text-extracted TableValue with one VLM-extracted TableValue.

    See module docstring for the R1..R7 rule set. Both inputs speak the same
    schema, so every branch is a field-by-field comparison — no shape
    translation.
    """
    # R1: VLM couldn't find. If VLM had a note, that's a semantic signal to
    # downgrade the text reading. If not, keep text as-is but still note that
    # VLM tried and couldn't locate.
    if not vlm_val.found:
        if vlm_val.note:
            return text_val.model_copy(update={
                "source": "merged",
                "confidence": "low",
                "note": _append_note(
                    text_val.note,
                    f"VLM could not locate this cell in the image "
                    f"(VLM note: {vlm_val.note}) — original reading treated "
                    f"as unreliable",
                ),
            })
        return text_val.model_copy(update={
            "source": "merged",
            "note": _append_note(
                text_val.note, "VLM could not locate this cell in the image"
            ),
        })

    values_agree = _values_match(text_val.value, vlm_val.value)
    labels_agree = _labels_align(text_val, vlm_val)
    vlm_has_concern = bool(vlm_val.note and vlm_val.note.strip())

    # R2: values match + labels agree + no VLM concern → mutual vote, HIGH.
    if values_agree and labels_agree and not vlm_has_concern:
        return text_val.model_copy(update={
            "source": "merged",
            "confidence": "high",
            "note": _append_note(
                text_val.note,
                f"VLM ({vlm_val.confidence}) independently agreed on value + labels",
            ),
        })

    # R3: values match + VLM raised a concern in its note → agreement is
    # weaker than it looks. Keep text but cap confidence at "medium" and
    # surface the concern.
    if values_agree and labels_agree and vlm_has_concern:
        return text_val.model_copy(update={
            "source": "merged",
            "confidence": _cap_at_medium(text_val.confidence),
            "note": _append_note(
                text_val.note,
                f"VLM ({vlm_val.confidence}) agreed on value but flagged: {vlm_val.note}",
            ),
        })

    # R4: values match but LABELS reveal a segment mismatch (e.g. VLM said
    # "Male" and text said no segment at all). This is the learning-hours
    # case — both saw 1,337,472.73 but VLM correctly labelled it "Male".
    if values_agree and not labels_agree:
        return text_val.model_copy(update={
            "source": "merged",
            "confidence": "low",
            "note": _append_note(
                text_val.note,
                f"DISPUTED-LABELS: value {text_val.value!r} is the same in both "
                f"readings but the LABELS reveal a semantic mismatch. "
                f"text=[{text_val.row_label!r} × {text_val.column_label!r}], "
                f"vlm=[{vlm_val.row_label!r} × {vlm_val.column_label!r}]"
                + (f" — VLM note: {vlm_val.note}" if vlm_val.note else "")
                + ". This typically means the reading is a SEGMENT of the "
                "target (e.g. Male-only when target wanted combined). "
                "Treat as unreliable.",
            ),
        })

    # Values DIFFER from here on.

    # R5: One side is HIGH and the other isn't → the HIGH reading wins.
    if vlm_val.confidence == "high" and text_val.confidence != "high":
        # VLM confidently disagrees — REPLACE (shifted-header rescue).
        return vlm_val.model_copy(update={
            "source": "merged",
            "note": _append_note(
                vlm_val.note,
                f"VLM-OVERRIDE (text extractor was {text_val.confidence}, "
                f"VLM high-confidence read differed): "
                f"original text value was {text_val.value!r} at "
                f"row={text_val.row_label!r} × col={text_val.column_label!r}",
            ),
        })
    if text_val.confidence == "high" and vlm_val.confidence != "high":
        # Text confidently disagrees — keep text but note VLM's dissent.
        return text_val.model_copy(update={
            "source": "merged",
            "confidence": "medium",  # hedge slightly, VLM saw something different
            "note": _append_note(
                text_val.note,
                f"VLM ({vlm_val.confidence}) DISAGREED: read {vlm_val.value!r} "
                f"at row={vlm_val.row_label!r} × col={vlm_val.column_label!r}"
                + (f" — {vlm_val.note}" if vlm_val.note else "")
                + ". Kept text (high) reading with a hedge to medium.",
            ),
        })

    # R6 + R7: values differ + no clear confidence winner → DISPUTED.
    return text_val.model_copy(update={
        "source": "merged",
        "confidence": "low",
        "note": _append_note(
            text_val.note,
            f"DISPUTED: text-extractor read {text_val.value!r} "
            f"({text_val.confidence}); VLM ({vlm_val.confidence}) read "
            f"{vlm_val.value!r} at row={vlm_val.row_label!r} × "
            f"col={vlm_val.column_label!r}"
            + (f" — {vlm_val.note}" if vlm_val.note else "")
            + ". Both readings preserved; treat as unreliable — the "
            "underlying data may be structured such that no single value "
            "matches the target (e.g. gender-split column with no combined "
            "total).",
        ),
    })


# ── Small helpers ───────────────────────────────────────────────────────────

def _values_match(a: str | None, b: str | None) -> bool:
    """Loose numeric-friendly equality — ignore commas, whitespace, casing."""
    if a is None or b is None:
        return False
    def _norm(s: str) -> str:
        return s.replace(",", "").replace(" ", "").strip().lower()
    return _norm(a) == _norm(b)


# Words that when present in a label indicate a SEGMENT (Male/Female/etc.)
# rather than a combined total. Used by `_labels_align`.
_SEGMENT_MARKERS = frozenset({
    "male", "female", "other", "others",
    "man", "woman", "men", "women",
    "regular", "contractual", "permanent", "contract",
    "north", "south", "east", "west",
    "domestic", "international",
    "urban", "rural",
})


def _labels_align(text_val: TableValue, vlm_val: TableValue) -> bool:
    """Do the two readings' labels describe the SAME slice of the source?

    Loose equality — normalize casing / whitespace / separator characters
    (`/`, `|`, `-`), then check whether the concatenated (row,col) label
    strings match — OR both lack any segment marker (both "combined" reads).

    The "segment marker" check catches the learning-hours case: text said
    row='' × col='Total Learning Hours', VLM said row='' × col='Total
    Learning Hours / Male'. Their label strings differ AND the VLM's has a
    segment marker the text's doesn't. That's a mismatch.
    """
    text_joined = _join_labels(text_val)
    vlm_joined = _join_labels(vlm_val)
    if text_joined == vlm_joined:
        return True

    text_segs = _extract_segments(text_joined)
    vlm_segs = _extract_segments(vlm_joined)
    # If one side has segment markers the other doesn't, that's a mismatch.
    if text_segs != vlm_segs:
        return False
    # Same segment markers (or none on either side) but labels differ in
    # non-segment ways — treat as ALIGNED. Cosmetic label differences
    # (like a caption vs a header) shouldn't count as a mismatch.
    return True


def _join_labels(tv: TableValue) -> str:
    parts = [tv.row_label or "", tv.column_label or ""]
    return _norm_label(" | ".join(parts))


def _norm_label(s: str) -> str:
    s = s.lower()
    # Collapse separators to spaces so "Total Learning Hours / Male" tokens
    # cleanly.
    s = re.sub(r"[|/\-–—]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_segments(joined: str) -> frozenset[str]:
    tokens = joined.split()
    return frozenset(t for t in tokens if t in _SEGMENT_MARKERS)


def _cap_at_medium(
    confidence: str,
) -> str:
    if confidence == "high":
        return "medium"
    return confidence  # "medium" or "low" pass through


def _append_note(existing: str | None, addition: str) -> str:
    if not existing:
        return addition
    return f"{existing}; {addition}"


def _truncate_err(msg: str, max_len: int = 200) -> str:
    msg = msg.replace("\n", " ").strip()
    if len(msg) <= max_len:
        return msg
    return msg[: max_len - 3] + "..."


def _emit(cb: EventCallback, stage: str, data: Any) -> None:
    if cb is None:
        return
    try:
        cb(stage, data)
    except Exception:  # noqa: BLE001
        pass
