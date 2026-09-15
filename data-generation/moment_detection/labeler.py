"""Phase 2: label Moment Detection outputs from saved inputs (spec v1.3)."""

from __future__ import annotations

import time
from typing import Any

from story_chunking.providers import Provider
from story_chunking.transformers_provider import TransformersProvider

from .prompts import LABEL_SYSTEM_PROMPT, build_label_user_prompt
from .validation import validate_output_row


def label_input_row(
    provider: Provider,
    input_row: dict,
) -> tuple[dict | None, dict[str, Any]]:
    inp = input_row["input"]
    original_text = inp["original_text"]
    source = inp["source"]
    input_id = input_row["id"]

    stats: dict[str, Any] = {"labeling_s": 0.0}
    prompt = build_label_user_prompt(original_text, source)

    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {}
    if isinstance(provider, TransformersProvider):
        kwargs = {"temperature": 0.1, "max_new_tokens": 1536, "max_retries": 3}
    parsed = provider.generate_json(LABEL_SYSTEM_PROMPT, prompt, **kwargs)
    stats["labeling_s"] = round(time.perf_counter() - t0, 1)

    if not parsed:
        stats["validation_reason"] = "labeler returned no JSON"
        return None, stats

    row = {"input_id": input_id, "output": parsed}
    is_valid, reason = validate_output_row(row, input_row=input_row)
    if not is_valid:
        stats["validation_reason"] = reason
        return None, stats

    return row, stats
