"""
Section-first orchestration for synthetic Story Chunking examples.

Flow per example:
  1. Pick topics + speakers.
  2. Ask the provider for a meeting plan (titles, summaries, speakers).
  3. Assign even, contiguous time slots so boundaries are always valid.
  4. Ask the provider for messy transcript lines per section.
  5. Assemble transcript_text and per-section section_text.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

from .speakers import choose_speakers
from .prompts import (
    INSTRUCTION,
    TOPIC_POOL,
    build_plan_prompt,
    build_section_prompt,
    PLAN_SYSTEM_PROMPT,
    SECTION_SYSTEM_PROMPT,
)
from .ollama_schemas import PLAN_JSON_SCHEMA, SECTION_JSON_SCHEMA
from .providers import Provider
from .validation import parse_mmss

MIN_TOPICS = 5
MAX_TOPICS = 8


def seconds_to_mmss(seconds: int) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def choose_topics() -> list[str]:
    count = random.randint(MIN_TOPICS, min(MAX_TOPICS, len(TOPIC_POOL)))
    return random.sample(TOPIC_POOL, count)


def allocate_slots(duration_seconds: int, num_sections: int) -> list[tuple[int, int]]:
    """Even, contiguous [start, end) slots covering the full duration."""
    slot = duration_seconds // num_sections
    slots: list[tuple[int, int]] = []
    for i in range(num_sections):
        start = i * slot
        end = duration_seconds if i == num_sections - 1 else (i + 1) * slot
        slots.append((start, end))
    return slots


def render_lines(lines: list[dict], slot_start: int, slot_end: int, speakers: list[str]) -> str:
    """Render model lines as '[MM:SS - MM:SS] Speaker: text', clamped to the slot."""
    rendered: list[str] = []
    fallback_speaker = speakers[0] if speakers else "Speaker 1"
    span = max(1, slot_end - slot_start)
    step = max(1, span // max(1, len(lines)))

    for index, line in enumerate(lines):
        text = str(line.get("text", "")).strip()
        if not text:
            continue

        speaker = str(line.get("speaker", "")).strip() or fallback_speaker

        start = parse_mmss(str(line.get("start", "")))
        end = parse_mmss(str(line.get("end", "")))
        if start is None or start < slot_start or start >= slot_end:
            start = slot_start + index * step
        if end is None or end <= start or end > slot_end:
            end = min(slot_end, start + step)

        rendered.append(
            f"[{seconds_to_mmss(start)} - {seconds_to_mmss(end)}] {speaker}: {text}"
        )

    return "\n".join(rendered)


def generate_section_text(
    provider: Provider,
    section: dict,
    slot_start: int,
    slot_end: int,
    speakers: list[str],
) -> str:
    section_with_times = dict(section)
    section_with_times["start_time"] = seconds_to_mmss(slot_start)
    section_with_times["end_time"] = seconds_to_mmss(slot_end)

    prompt = build_section_prompt(section_with_times, speakers)
    result = provider.generate_json(
        SECTION_SYSTEM_PROMPT, prompt, json_schema=SECTION_JSON_SCHEMA
    )
    if not result:
        return ""

    lines = result.get("lines")
    if not isinstance(lines, list) or not lines:
        return ""

    return render_lines(lines, slot_start, slot_end, speakers)


def build_example(
    provider: Provider,
    *,
    transcript_id: str,
    duration_minutes: int,
) -> dict | None:
    """Generate one full synthetic example, or None if generation failed."""
    topics = choose_topics()
    speakers = choose_speakers()
    duration_seconds = duration_minutes * 60

    plan = provider.generate_json(
        PLAN_SYSTEM_PROMPT,
        build_plan_prompt(duration_minutes, topics, speakers),
        json_schema=PLAN_JSON_SCHEMA,
    )
    if not plan:
        return None

    planned_sections = plan.get("story_sections")
    meeting_summary = str(plan.get("meeting_summary", "")).strip()
    if not isinstance(planned_sections, list) or not planned_sections or not meeting_summary:
        return None

    slots = allocate_slots(duration_seconds, len(planned_sections))

    sections_out: list[dict] = []
    transcript_chunks: list[str] = []

    for section, (slot_start, slot_end) in zip(planned_sections, slots):
        section_speakers = section.get("speakers")
        if not isinstance(section_speakers, list) or len(section_speakers) < 2:
            section_speakers = random.sample(speakers, k=min(3, len(speakers)))

        section_text = generate_section_text(
            provider, section, slot_start, slot_end, section_speakers
        )
        if not section_text:
            return None

        confidence = section.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            confidence = round(random.uniform(0.7, 0.95), 2)

        sections_out.append(
            {
                "start_time": seconds_to_mmss(slot_start),
                "end_time": seconds_to_mmss(slot_end),
                "section_title": str(section.get("section_title", "")).strip()
                or "Untitled section",
                "summary": str(section.get("summary", "")).strip(),
                "section_text": section_text,
                "speakers": section_speakers,
                "confidence": round(float(confidence), 2),
            }
        )
        transcript_chunks.append(section_text)

    transcript_text = "\n\n".join(transcript_chunks)
    duration_label = f"{duration_minutes:02d}:00"

    return {
        "instruction": INSTRUCTION,
        "input": {
            "transcript_id": transcript_id,
            "duration": duration_label,
            "transcript_text": transcript_text,
        },
        "output": {
            "transcript_id": transcript_id,
            "meeting_summary": meeting_summary,
            "story_sections": sections_out,
        },
        "metadata": {
            "synthetic": True,
            "provider": provider.name,
            "topics": topics,
            "speakers": speakers,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
