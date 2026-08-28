"""
Transcript-first orchestration for V2 synthetic Story Chunking examples.

Flow per example:
  1. Pick scenario + speakers (transcript generator does NOT see boundaries).
  2. Generate full messy transcript.
  3. Label pass: meeting_summary + story_sections from transcript.
  4. Optional reviewer pass.
  5. Deterministic validation.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any

from .prompts_v2 import (
    INSTRUCTION_V2,
    LABEL_SYSTEM_PROMPT,
    TRANSCRIPT_SYSTEM_PROMPT,
    build_label_prompt,
    build_transcript_prompt,
)
from .providers import Provider
from .reviewer import apply_review_result, run_review
from .scenarios_v2 import ScenarioTemplate, pick_scenario, scenario_duration_minutes
from .speakers import choose_speakers
from .timestamps import (
    meeting_duration_from_lines,
    parse_transcript,
    section_text_for_range,
)
from .transformers_provider import TransformersProvider
from .validation_v2 import validate_example_v2

MAX_ATTEMPTS_V2 = 8


def _generate_transcript(
    provider: Provider,
    scenario: ScenarioTemplate,
    duration_minutes: int,
    speakers: list[str],
    target_story_count: int,
) -> str | None:
    prompt = build_transcript_prompt(scenario, duration_minutes, speakers, target_story_count)

    if isinstance(provider, TransformersProvider):
        text = provider.generate_text(
            TRANSCRIPT_SYSTEM_PROMPT,
            prompt,
            temperature=0.9,
            max_new_tokens=4096,
        )
    else:
        result = provider.generate_json(TRANSCRIPT_SYSTEM_PROMPT, prompt)
        if result and "transcript_text" in result:
            return str(result["transcript_text"]).strip()
        if result and "lines" in result:
            lines = result["lines"]
            if isinstance(lines, list):
                parts = []
                for line in lines:
                    start = line.get("start", "00:00")
                    end = line.get("end", "00:05")
                    speaker = line.get("speaker", "Speaker")
                    text_part = line.get("text", "")
                    parts.append(f"[{start} - {end}] {speaker}: {text_part}")
                return "\n".join(parts)
        return None

    return text.strip() if text else None


def _run_label_pass(
    provider: Provider,
    transcript_text: str,
    duration: str,
) -> dict | None:
    prompt = build_label_prompt(transcript_text, duration)
    kwargs: dict[str, Any] = {}
    if isinstance(provider, TransformersProvider):
        kwargs = {"temperature": 0.1, "max_new_tokens": 2048, "max_retries": 3}
    return provider.generate_json(LABEL_SYSTEM_PROMPT, prompt, **kwargs)


def _clean_sections(raw_sections: list) -> list[dict]:
    clean: list[dict] = []
    for section in raw_sections:
        if not isinstance(section, dict):
            continue
        clean.append(
            {
                "start_time": str(section.get("start_time", "")).strip(),
                "end_time": str(section.get("end_time", "")).strip(),
                "section_title": str(section.get("section_title", "")).strip(),
                "summary": str(section.get("summary", "")).strip(),
            }
        )
    return clean


def choose_speakers_for_scenario(
    speaker_count_range: tuple[int, int],
    rng: random.Random | None = None,
) -> list[str]:
    rng = rng or random
    low, high = speaker_count_range
    count = rng.randint(low, high)
    speakers = choose_speakers()
    while len(speakers) < count:
        speakers = choose_speakers()
    return speakers[:count]


def build_example_v2(
    provider: Provider,
    *,
    transcript_id: str,
    target_story_count: int,
    scenario: ScenarioTemplate | None = None,
    enable_review: bool = True,
    rng: random.Random | None = None,
) -> tuple[dict | None, dict[str, Any]]:
    """
    Generate one V2 example. Returns (row, stats) where stats has timing info.
    row is None if generation failed.
    """
    rng = rng or random.Random()
    scenario = scenario or pick_scenario(target_story_count, rng)
    duration_minutes = scenario_duration_minutes(scenario, rng)
    speakers = choose_speakers_for_scenario(scenario.speaker_count_range, rng)

    stats: dict[str, Any] = {
        "transcript_s": 0.0,
        "label_s": 0.0,
        "review_s": 0.0,
        "sections": 0,
        "validation": "FAIL",
        "scenario": scenario.key,
        "target_story_count": target_story_count,
    }

    t0 = time.perf_counter()
    transcript_text = _generate_transcript(
        provider, scenario, duration_minutes, speakers, target_story_count
    )
    stats["transcript_s"] = round(time.perf_counter() - t0, 1)

    if not transcript_text:
        return None, stats

    lines = parse_transcript(transcript_text)
    if len(lines) < 3:
        return None, stats

    duration = meeting_duration_from_lines(lines)
    if not duration:
        return None, stats

    t1 = time.perf_counter()
    label_result = _run_label_pass(provider, transcript_text, duration)
    stats["label_s"] = round(time.perf_counter() - t1, 1)

    if not label_result:
        return None, stats

    meeting_summary = str(label_result.get("meeting_summary", "")).strip()
    raw_sections = label_result.get("story_sections")
    if not meeting_summary or not isinstance(raw_sections, list) or not raw_sections:
        return None, stats

    sections = _clean_sections(raw_sections)
    review_meta: dict[str, Any] = {"approved": True, "issues": []}

    if enable_review:
        t2 = time.perf_counter()
        review = run_review(
            provider, transcript_text, duration, sections, meeting_summary
        )
        stats["review_s"] = round(time.perf_counter() - t2, 1)

        if review and not review.get("approved"):
            repaired = apply_review_result(
                review,
                transcript_text,
                duration,
                transcript_id,
                meeting_summary,
                sections,
            )
            if repaired:
                sections, meeting_summary, review_meta = repaired
            else:
                review_meta = review
                return None, stats
        elif review:
            review_meta = review

    # Audit metadata: section text by range (not in model target)
    section_text_by_range = {
        f"{s['start_time']}-{s['end_time']}": section_text_for_range(
            lines, s["start_time"], s["end_time"]
        )
        for s in sections
    }

    row = {
        "instruction": INSTRUCTION_V2,
        "input": {
            "transcript_id": transcript_id,
            "duration": duration,
            "transcript_text": transcript_text,
        },
        "output": {
            "transcript_id": transcript_id,
            "meeting_summary": meeting_summary,
            "story_sections": sections,
        },
        "metadata": {
            "synthetic": True,
            "version": "v2",
            "provider": provider.name,
            "target_story_count": target_story_count,
            "scenario": scenario.key,
            "scenario_title": scenario.title,
            "speakers": speakers,
            "generation_timings": {
                "transcript_s": stats["transcript_s"],
                "label_s": stats["label_s"],
                "review_s": stats["review_s"],
            },
            "review": review_meta,
            "section_text_by_range": section_text_by_range,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    is_valid, reason = validate_example_v2(row)
    if not is_valid:
        stats["validation_reason"] = reason
        return None, stats

    stats["validation"] = "PASS"
    stats["sections"] = len(sections)
    return row, stats
