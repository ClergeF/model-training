"""Phase 1: generate Moment Detection input rows."""

from __future__ import annotations

import time
from typing import Any

from story_chunking.providers import Provider
from story_chunking.transformers_provider import TransformersProvider

from .prompts import INPUT_SYSTEM_PROMPT, build_input_user_prompt
from .validation import validate_input_row, word_count


def generate_input_row(
    provider: Provider,
    *,
    row_id: str,
    source: str = "synthetic",
) -> tuple[dict | None, dict[str, Any]]:
    user_prompt, target_words = build_input_user_prompt()
    stats: dict[str, Any] = {"target_words": target_words, "words": 0, "generation_s": 0.0}

    t0 = time.perf_counter()
    if isinstance(provider, TransformersProvider):
        text = provider.generate_text(
            INPUT_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.9,
            max_new_tokens=512,
        )
    else:
        result = provider.generate_json(INPUT_SYSTEM_PROMPT, user_prompt)
        text = str(result.get("original_text", "")).strip() if result else ""
    stats["generation_s"] = round(time.perf_counter() - t0, 1)

    if not text:
        return None, stats

    row = {
        "id": row_id,
        "input": {
            "original_text": text.strip(),
            "source": source,
            "categories": [],
        },
    }
    stats["words"] = word_count(text)

    is_valid, reason = validate_input_row(row)
    if not is_valid:
        stats["validation_reason"] = reason
        return None, stats

    return row, stats
