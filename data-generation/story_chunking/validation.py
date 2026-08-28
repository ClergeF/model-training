"""
Python-only validation for synthetic Story Chunking examples.

Checks schema completeness and input/output alignment. No LLM calls here.
"""

from __future__ import annotations

import re

TIME_PATTERN = re.compile(r"^\d{1,3}:\d{2}$")

MIN_SECTIONS_PER_HOUR = 4


def parse_mmss(value: str) -> int | None:
    """Convert 'MM:SS' (minutes can exceed 59) to total seconds."""
    if not isinstance(value, str) or not TIME_PATTERN.match(value.strip()):
        return None
    minutes, seconds = value.strip().split(":")
    if int(seconds) >= 60:
        return None
    return int(minutes) * 60 + int(seconds)


def min_sections_for_duration(duration_minutes: int) -> int:
    scaled = round(MIN_SECTIONS_PER_HOUR * duration_minutes / 60)
    return max(3, scaled)


def validate_example(row: dict, duration_minutes: int) -> tuple[bool, str | None]:
    """Return (True, None) if the row is valid, else (False, reason)."""
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
    if len(transcript_text) < 300:
        return False, f"transcript_text too short ({len(transcript_text)} chars)"

    sections = output_obj["story_sections"]
    if not isinstance(sections, list):
        return False, "story_sections is not a list"

    needed = min_sections_for_duration(duration_minutes)
    if len(sections) < needed:
        return False, f"too few sections ({len(sections)}, need >= {needed})"

    duration_seconds = duration_minutes * 60
    previous_end = 0

    for index, section in enumerate(sections):
        for key in (
            "start_time",
            "end_time",
            "section_title",
            "summary",
            "section_text",
            "speakers",
            "confidence",
        ):
            if key not in section:
                return False, f"section {index} missing: {key}"

        start = parse_mmss(section["start_time"])
        end = parse_mmss(section["end_time"])
        if start is None:
            return False, f"section {index} bad start_time: {section['start_time']}"
        if end is None:
            return False, f"section {index} bad end_time: {section['end_time']}"
        if end <= start:
            return False, f"section {index} end_time not after start_time"

        if index == 0 and start != 0:
            return False, "first section does not start at 00:00"
        if start < previous_end:
            return False, f"section {index} overlaps previous section"
        previous_end = end

        if not str(section["section_text"]).strip():
            return False, f"section {index} has empty section_text"

        speakers = section["speakers"]
        if not isinstance(speakers, list) or len(speakers) < 2:
            return False, f"section {index} needs at least 2 speakers"

        confidence = section["confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False, f"section {index} confidence out of range"

    # Coverage: last section should reach close to the full duration.
    if previous_end < duration_seconds - 120:
        return False, "sections do not cover the full meeting duration"

    return True, None
