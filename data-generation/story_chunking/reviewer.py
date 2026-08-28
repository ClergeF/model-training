"""
Optional reviewer pass for V2 story segmentation quality.
"""

from __future__ import annotations

from typing import Any

from .prompts_v2 import REVIEW_SYSTEM_PROMPT, build_review_prompt
from .providers import Provider
from .validation_v2 import validate_example_v2


MAX_LABEL_RETRIES = 2


def run_review(
    provider: Provider,
    transcript_text: str,
    duration: str,
    sections: list[dict],
    meeting_summary: str,
) -> dict[str, Any] | None:
    """Run reviewer LLM call. Returns parsed review dict or None on failure."""
    prompt = build_review_prompt(transcript_text, duration, sections)
    result = provider.generate_json(
        REVIEW_SYSTEM_PROMPT,
        prompt,
        temperature=0.1,
        max_new_tokens=2048,
        max_retries=2,
    )
    if not result:
        return None

    approved = result.get("approved")
    issues = result.get("issues", [])
    suggested = result.get("suggested_story_sections", [])

    return {
        "approved": bool(approved),
        "issues": issues if isinstance(issues, list) else [],
        "suggested_story_sections": suggested if isinstance(suggested, list) else [],
        "meeting_summary": meeting_summary,
    }


def apply_review_result(
    review: dict[str, Any],
    transcript_text: str,
    duration: str,
    transcript_id: str,
    meeting_summary: str,
    sections: list[dict],
) -> tuple[list[dict], str, dict[str, Any]] | None:
    """
    If reviewer approved or supplied valid suggested sections, return updated
    (sections, meeting_summary, review_metadata). Else return None.
    """
    if review.get("approved"):
        return sections, meeting_summary, review

    suggested = review.get("suggested_story_sections") or []
    if suggested:
        candidate_row = {
            "instruction": "",
            "input": {
                "transcript_id": transcript_id,
                "duration": duration,
                "transcript_text": transcript_text,
            },
            "output": {
                "transcript_id": transcript_id,
                "meeting_summary": meeting_summary,
                "story_sections": suggested,
            },
            "metadata": {},
        }
        is_valid, reason = validate_example_v2(candidate_row)
        if is_valid:
            review["approved"] = True
            review["issues"] = review.get("issues", []) + ["repaired from reviewer suggestions"]
            return suggested, meeting_summary, review

    return None
