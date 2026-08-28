"""
Transcript timestamp parsing and story-section boundary validation.

Boundary convention (V2):
- Transcript lines use: [MM:SS - MM:SS] Speaker: text
- Allowed timestamps are every line start and end time in the transcript.
- section[0].start_time = first line start
- For i < n-1: section[i].end_time == section[i+1].start_time (shared boundary)
- Final section end_time = last line end time
- Every section start_time and end_time must exactly match a value in the allowed set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .validation import parse_mmss

LINE_PATTERN = re.compile(
    r"^\[(?P<start>\d{1,3}:\d{2})\s*-\s*(?P<end>\d{1,3}:\d{2})\]\s*"
    r"(?P<speaker>[^:]+):\s*(?P<text>.+)$"
)


@dataclass(frozen=True)
class TranscriptLine:
    start_seconds: int
    end_seconds: int
    start_time: str
    end_time: str
    speaker: str
    text: str
    raw: str


def seconds_to_mmss(seconds: int) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def parse_transcript(transcript_text: str) -> list[TranscriptLine]:
    lines: list[TranscriptLine] = []
    for raw_line in transcript_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_PATTERN.match(line)
        if not match:
            continue
        start = parse_mmss(match.group("start"))
        end = parse_mmss(match.group("end"))
        if start is None or end is None or end <= start:
            continue
        lines.append(
            TranscriptLine(
                start_seconds=start,
                end_seconds=end,
                start_time=match.group("start"),
                end_time=match.group("end"),
                speaker=match.group("speaker").strip(),
                text=match.group("text").strip(),
                raw=line,
            )
        )
    return lines


def allowed_timestamp_set(lines: list[TranscriptLine]) -> set[str]:
    allowed: set[str] = set()
    for line in lines:
        allowed.add(line.start_time)
        allowed.add(line.end_time)
    return allowed


def allowed_start_timestamps(lines: list[TranscriptLine]) -> set[str]:
    return {line.start_time for line in lines}


def meeting_duration_from_lines(lines: list[TranscriptLine]) -> str | None:
    if not lines:
        return None
    return seconds_to_mmss(lines[-1].end_seconds)


def lines_in_section(
    lines: list[TranscriptLine],
    start_time: str,
    end_time: str,
) -> list[TranscriptLine]:
    start_seconds = parse_mmss(start_time)
    end_seconds = parse_mmss(end_time)
    if start_seconds is None or end_seconds is None:
        return []
    return [
        line
        for line in lines
        if line.start_seconds >= start_seconds and line.start_seconds < end_seconds
    ]


def section_text_for_range(
    lines: list[TranscriptLine],
    start_time: str,
    end_time: str,
) -> str:
    return "\n".join(line.raw for line in lines_in_section(lines, start_time, end_time))


def validate_section_boundaries(
    sections: list[dict],
    lines: list[TranscriptLine],
) -> tuple[bool, str | None]:
    if not lines:
        return False, "transcript has no parseable timestamped lines"
    if not sections:
        return False, "story_sections is empty"

    allowed = allowed_timestamp_set(lines)
    allowed_starts = allowed_start_timestamps(lines)
    first_start = lines[0].start_time
    last_end = lines[-1].end_time

    for index, section in enumerate(sections):
        start_time = str(section.get("start_time", "")).strip()
        end_time = str(section.get("end_time", "")).strip()

        if start_time not in allowed:
            return False, f"section {index} start_time {start_time!r} not in transcript"
        if end_time not in allowed:
            return False, f"section {index} end_time {end_time!r} not in transcript"

        start_seconds = parse_mmss(start_time)
        end_seconds = parse_mmss(end_time)
        if start_seconds is None or end_seconds is None:
            return False, f"section {index} has invalid timestamp format"
        if end_seconds <= start_seconds:
            return False, f"section {index} end_time must be after start_time"

        if index == 0:
            if start_time != first_start:
                return False, f"first section must start at {first_start}, got {start_time}"
        else:
            prev_end = str(sections[index - 1].get("end_time", "")).strip()
            if start_time != prev_end:
                return False, (
                    f"section {index} start_time {start_time} must equal "
                    f"previous end_time {prev_end}"
                )
            if start_time not in allowed_starts:
                return False, f"section {index} boundary {start_time} is not an utterance start"

        if index < len(sections) - 1:
            next_start = str(sections[index + 1].get("start_time", "")).strip()
            if end_time != next_start:
                return False, (
                    f"section {index} end_time {end_time} must equal "
                    f"next start_time {next_start}"
                )
        else:
            if end_time != last_end:
                return False, f"final section must end at {last_end}, got {end_time}"

    return True, None


def count_minute_rounded_starts(sections: list[dict]) -> int:
    count = 0
    for section in sections:
        start_time = str(section.get("start_time", "")).strip()
        parsed = parse_mmss(start_time)
        if parsed is not None and parsed % 60 == 0 and parsed > 0:
            count += 1
    return count


def has_artificial_even_blocks(sections: list[dict]) -> bool:
    """Heuristic: >=3 section starts at exact minute marks (01:00, 02:00, ...)."""
    return count_minute_rounded_starts(sections) >= 3
