"""LLM client + helpers used by every agent.

We talk to Fireworks.ai via their OpenAI-compatible API. The `openai` Python
SDK works against any OpenAI-compatible endpoint — we just point its `base_url`
at Fireworks. Same agents (Planner / Critic / Synthesizer / Table Extractor),
zero changes to prompts.

Structured outputs are constrained via the `response_format=json_schema` form,
which Fireworks supports for most modern models. The schema is generated from
the caller's Pydantic class, sent with the request, and the response comes
back as JSON we can `model_validate` directly.

Prompt-caching note: Anthropic's explicit `cache_control` is gone. Fireworks
does its own prefix caching automatically for repeated stable prefixes, so we
get cache hits "for free" as long as agent system prompts stay byte-stable —
which they do, since they live as module-level constants in `agents/prompts.py`.
"""
from __future__ import annotations

import json
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from config.settings import settings

T = TypeVar("T", bound=BaseModel)

_client: AsyncOpenAI | None = None
_groq_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Single shared AsyncOpenAI instance for FIREWORKS, lazy-initialized.

    Used by all text agents (planner, critic, synthesizer, extractors) and
    by the VLM path when `use_groq_vlm=False`.
    """
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.fireworks_api_key,
            base_url=settings.fireworks_base_url,
        )
    return _client


def get_groq_client() -> AsyncOpenAI:
    """Single shared AsyncOpenAI instance for GROQ, lazy-initialized.

    Used only for VLM calls when `use_groq_vlm=True`. Groq exposes an
    OpenAI-compatible /v1 endpoint so the same SDK works — only the
    base_url + api_key change.
    """
    global _groq_client
    if _groq_client is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "USE_GROQ_VLM=true but GROQ_API_KEY is empty. Set GROQ_API_KEY "
                "in .env or flip USE_GROQ_VLM=false to fall back to Fireworks."
            )
        _groq_client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
    return _groq_client


def get_vlm_config() -> tuple[AsyncOpenAI, str, int]:
    """Return (client, model, max_tokens) for a VLM call, per USE_GROQ_VLM.

    Routes ALL vision calls (ingestion image description, query-time
    table/composite VLM verifier, extract-from-unfound) through a single
    knob. Keeping this in one place means the three call sites don't each
    need to know the provider — they just ask for "the VLM config" and
    fire the call.
    """
    if settings.use_groq_vlm:
        return get_groq_client(), settings.groq_vlm_model, settings.groq_vlm_max_tokens
    return get_client(), settings.vlm_model, settings.vlm_max_tokens


def _schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema hardened for OpenAI/Groq strict mode.

    Applies two transforms:
      1. additionalProperties=false on every nested object (Pydantic omits)
      2. `required` array widened to include every property key on every
         nested object (Pydantic emits only fields without defaults). Groq
         enforces this per the OpenAI structured-output spec; Fireworks
         doesn't but accepts the stricter shape.
    """
    schema = model.model_json_schema()
    _force_additional_properties_false(schema)
    _force_all_properties_required(schema)
    return schema


def _force_additional_properties_false(schema: dict[str, Any]) -> None:
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
    for v in schema.values():
        if isinstance(v, dict):
            _force_additional_properties_false(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _force_additional_properties_false(item)


def _force_all_properties_required(schema: dict[str, Any]) -> None:
    """OpenAI/Groq strict-mode compliance: every property key must be in `required`.

    Pydantic emits `required` containing only fields that lack a default value.
    OpenAI's strict-mode JSON schema spec (which Groq enforces aggressively;
    Fireworks is more lenient) requires ALL properties to be listed in
    `required`, with optionality expressed via nullable type unions like
    `anyOf: [{"type": "string"}, {"type": "null"}]`. Pydantic already emits
    Optional[X] as such a nullable union — this helper just widens the
    `required` array so schema is fully compliant with strict mode.

    Safe for both providers: a stricter schema still describes the same
    shape, and Fireworks accepts it without complaint.
    """
    if schema.get("type") == "object" and "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    for v in schema.values():
        if isinstance(v, dict):
            _force_all_properties_required(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _force_all_properties_required(item)


async def structured_call(
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    output_model: type[T],
    max_tokens: int = 32000,
) -> T:
    """Call the LLM, return a typed instance of `output_model`.

    Uses Fireworks's OpenAI-compatible `response_format=json_schema` mode, so
    the model is constrained to return JSON matching the Pydantic schema. We
    then `model_validate` the JSON for full type-checking on our side too.

    Default max_tokens is 16k because Qwen3.7-Plus is a thinking model and
    spends tokens on internal reasoning before emitting the JSON — small
    budgets cause mid-string truncation that surfaces as JSONDecodeError.
    Bump higher for agents that produce long outputs (e.g. synthesizer → 32k).
    """
    client = get_client()
    schema = _schema_for(output_model)

    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": output_model.__name__,
                "schema": schema,
                "strict": True,
            },
        },
    )

    if not resp.choices:
        raise RuntimeError(f"{model} returned no choices: {resp!r}")
    choice = resp.choices[0]
    text = choice.message.content
    finish_reason = choice.finish_reason

    if not text:
        raise RuntimeError(
            f"{model} returned empty content (finish_reason={finish_reason!r})"
        )

    # If the model ran out of tokens mid-JSON, json.loads will throw a useless
    # "Unterminated string" error. Catch it here and surface the real cause +
    # a snippet of what we got so the fix (bump max_tokens) is obvious.
    if finish_reason == "length":
        raise RuntimeError(
            f"{model} hit max_tokens={max_tokens} before completing the JSON "
            f"for {output_model.__name__}. Increase max_tokens for this agent. "
            f"Got {len(text)} chars; tail: …{text[-200:]!r}"
        )

    try:
        return output_model.model_validate(json.loads(text))
    except json.JSONDecodeError as e:
        # Bad JSON despite finish_reason != "length" — show the head and tail
        # so we can see whether the model emitted thinking tokens, prose, etc.
        raise RuntimeError(
            f"{model} returned malformed JSON for {output_model.__name__} "
            f"(finish_reason={finish_reason!r}, {len(text)} chars). "
            f"Head: {text[:200]!r}  Tail: {text[-200:]!r}  "
            f"Decoder error: {e}"
        ) from e
