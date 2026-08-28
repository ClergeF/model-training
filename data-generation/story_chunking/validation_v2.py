"""
Deterministic validation for V2 synthetic Story Chunking examples.
"""

from __future__ import annotations

from .prompts_v2 import BANNED_SECTION_TITLES
from .timestamps import (
    has_artificial_even_blocks,
    meeting_duration_from_lines,
    parse_transcript,
    validate_section_boundaries,
)

MIN_TRANSCRIPT_CHARS = 300


def _title_is_banned(title: str) -> bool:
    normalized = title.strip().lower()
    for banned in BANNED_SECTION_TITLES:
        if banned.lower() in normalized:
            return True
    return False


def validate_example_v2(row: dict) -> tuple[bool, str | None]:
    """Return (True, None) if valid, else (False, reason)."""
    for key in ("instruction", "input", "output", "metadata"):
        if key not in row:
            return False, f"missing top-level key: {key}"

    input_obj = row["input"]
    output_obj = row["output"]

    for key in ("transcript_id", "duration", "transcript_text"):
        if not input_obj.get(key):
            return False, f"input missing or empty: {key}"

    for key in ("transcript_id", "meeting_summary", "story_sections"):
        if key not in output_obj:
            return False, f"output missing: {key}"

    if input_obj["transcript_id"] != output_obj["transcript_id"]:
        return False, "transcript_id mismatch between input and output"

    transcript_text = input_obj["transcript_text"]
    if len(transcript_text) < MIN_TRANSCRIPT_CHARS:
        return False, f"transcript_text too short ({len(transcript_text)} chars)"

    lines = parse_transcript(transcript_text)
    if len(lines) < 3:
        return False, f"too few parseable transcript lines ({len(lines)})"

    meeting_summary = str(output_obj.get("meeting_summary", "")).strip()
    if not meeting_summary:
        return False, "meeting_summary is empty"

    sections = output_obj["story_sections"]
    if not isinstance(sections, list) or not sections:
        return False, "story_sections must be a non-empty list"

    for index, section in enumerate(sections):
        for key in ("start_time", "end_time", "section_title", "summary"):
            if key not in section or not str(section[key]).strip():
                return False, f"section {index} missing or empty: {key}"

        title = str(section["section_title"]).strip()
        if _title_is_banned(title):
            return False, f"section {index} uses banned generic title: {title!r}"

        # Model target must not include these fields
        for forbidden in ("section_text", "confidence", "speakers"):
            if forbidden in section:
                return False, f"section {index} must not include {forbidden} in model target"

    ok, reason = validate_section_boundaries(sections, lines)
    if not ok:
        return False, reason

    if has_artificial_even_blocks(sections):
        return False, "too many minute-rounded section boundaries (artificial pattern)"

    expected_duration = meeting_duration_from_lines(lines)
    if expected_duration and input_obj["duration"] != expected_duration:
        return False, (
            f"duration mismatch: input has {input_obj['duration']!r}, "
            f"expected {expected_duration!r} from transcript"
        )

    return True, None
