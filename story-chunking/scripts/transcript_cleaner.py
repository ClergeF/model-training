#!/usr/bin/env python3
"""
Rule-based transcript cleaning for the Story Chunking pipeline.

Multi-pass cleanup: parse Recall raw → normalize → same-speaker gap merge
→ tiny-fragment merge → readable export. No LLM or model training.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STORY_CHUNKING_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = STORY_CHUNKING_ROOT / "data" / "transcripts" / "raw"
CLEANED_DIR = STORY_CHUNKING_ROOT / "data" / "transcripts" / "cleaned"
READABLE_DIR = STORY_CHUNKING_ROOT / "data" / "transcripts" / "readable"

CLEANUP_VERSION = "1"
DEFAULT_MAX_GAP_SECONDS = 1.75
DEFAULT_MAX_FRAGMENT_GAP_SECONDS = 2.0
MIN_FRAGMENT_MERGE_SCORE = 3.0

SENTENCE_ENDINGS = (".", "?", "!")
DANGLING_TOKENS = frozenset(
    {
        "the", "a", "an", "under", "for", "but", "and", "to", "of", "like",
        "been", "on", "at", "i", "you", "we", "my", "your", "our", "their",
        "this", "that", "with", "from", "in", "is", "are", "was", "were",
        "have", "has", "had", "just", "some", "no", "not", "what", "when",
        "where", "who", "how", "all", "right", "well", "so", "if", "or",
    }
)
CONNECTOR_WORDS = frozenset(
    {
        "but", "and", "because", "yeah", "so", "it", "weather", "or", "then",
        "also", "though", "though", "well", "right", "okay", "ok", "yes",
        "no", "like", "just", "really", "actually", "maybe", "though",
    }
)

UNCERTAIN_CONFIDENCE = "uncertain"
CROSS_SPEAKER_NOTE = "merged tiny fragment across speaker labels"


@dataclass
class CleanConfig:
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS
    max_fragment_gap_seconds: float = DEFAULT_MAX_FRAGMENT_GAP_SECONDS


@dataclass
class Segment:
    start_time: str = ""
    end_time: str = ""
    speaker: str = ""
    text: str = ""
    speaker_confidence: str | None = None
    cleanup_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker": self.speaker,
            "text": self.text,
        }
        if self.speaker_confidence:
            result["speaker_confidence"] = self.speaker_confidence
        if self.cleanup_note:
            result["cleanup_note"] = self.cleanup_note
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        return cls(
            start_time=str(data.get("start_time", "")),
            end_time=str(data.get("end_time", "")),
            speaker=str(data.get("speaker", "")),
            text=str(data.get("text", "")),
            speaker_confidence=data.get("speaker_confidence"),
            cleanup_note=data.get("cleanup_note"),
        )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_time(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def word_count(text: str) -> int:
    text = normalize_whitespace(text)
    if not text:
        return 0
    return len(text.split())


def is_fragment(segment: Segment) -> bool:
    return word_count(segment.text) <= 2


def ends_sentence(text: str) -> bool:
    text = normalize_whitespace(text)
    return bool(text) and text[-1] in SENTENCE_ENDINGS


def last_token(text: str) -> str:
    tokens = normalize_whitespace(text).lower().split()
    return tokens[-1] if tokens else ""


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return str(seconds)


def format_timestamp_obj(timestamp_obj: dict[str, Any] | None) -> str:
    if not timestamp_obj:
        return ""
    relative = timestamp_obj.get("relative")
    if relative is not None:
        return str(relative)
    absolute = timestamp_obj.get("absolute")
    if absolute:
        return str(absolute)
    return ""


def resolve_speaker(participant: dict[str, Any] | None, speaker_map: dict[str, str]) -> str:
    participant = participant or {}
    name = participant.get("name")
    if isinstance(name, str) and name.strip():
        return normalize_whitespace(name)

    participant_id = participant.get("id")
    key = f"id:{participant_id}" if participant_id is not None else "unknown"
    if key not in speaker_map:
        speaker_map[key] = f"Speaker {len(speaker_map) + 1}"
    return speaker_map[key]


def normalize_transcript_blocks(raw_data: Any) -> list[dict[str, Any]]:
    if isinstance(raw_data, list):
        return [block for block in raw_data if isinstance(block, dict)]

    if isinstance(raw_data, dict):
        for key in ("results", "segments", "utterances", "transcript"):
            value = raw_data.get(key)
            if isinstance(value, list):
                return [block for block in value if isinstance(block, dict)]

    raise ValueError("Unrecognized Recall transcript download format")


def block_to_segment(block: dict[str, Any], speaker_map: dict[str, str]) -> Segment | None:
    words = block.get("words") or []
    text_parts = [
        normalize_whitespace(str(word.get("text", "")))
        for word in words
        if normalize_whitespace(str(word.get("text", "")))
    ]
    text = normalize_whitespace(" ".join(text_parts))
    if not text:
        return None

    start_time = format_timestamp_obj(words[0].get("start_timestamp")) if words else ""
    end_time = format_timestamp_obj(words[-1].get("end_timestamp")) if words else ""

    return Segment(
        start_time=start_time,
        end_time=end_time,
        speaker=resolve_speaker(block.get("participant"), speaker_map),
        text=text,
    )


def parse_recall_raw(raw_data: Any) -> list[Segment]:
    blocks = normalize_transcript_blocks(raw_data)
    speaker_map: dict[str, str] = {}
    segments: list[Segment] = []
    for block in blocks:
        segment = block_to_segment(block, speaker_map)
        if segment:
            segments.append(segment)
    return segments


def normalize_segments(segments: list[Segment]) -> list[Segment]:
    cleaned: list[Segment] = []
    for segment in segments:
        text = normalize_whitespace(segment.text)
        if not text:
            continue
        cleaned.append(
            Segment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                speaker=segment.speaker,
                text=text,
                speaker_confidence=segment.speaker_confidence,
                cleanup_note=segment.cleanup_note,
            )
        )

    with_time = [s for s in cleaned if parse_time(s.start_time) is not None]
    without_time = [s for s in cleaned if parse_time(s.start_time) is None]
    with_time.sort(key=lambda s: parse_time(s.start_time) or 0.0)
    return with_time + without_time


def gap_seconds(previous: Segment, current: Segment) -> float | None:
    prev_end = parse_time(previous.end_time)
    curr_start = parse_time(current.start_time)
    if prev_end is None or curr_start is None:
        return None
    return curr_start - prev_end


def merge_into_previous(target: Segment, fragment: Segment, *, cross_speaker: bool) -> None:
    target.text = normalize_whitespace(f"{target.text} {fragment.text}")
    if fragment.end_time:
        target.end_time = fragment.end_time
    if cross_speaker:
        target.speaker_confidence = UNCERTAIN_CONFIDENCE
        target.cleanup_note = CROSS_SPEAKER_NOTE


def merge_into_next(target: Segment, fragment: Segment, *, cross_speaker: bool) -> None:
    target.text = normalize_whitespace(f"{fragment.text} {target.text}")
    if fragment.start_time:
        target.start_time = fragment.start_time
    if cross_speaker:
        target.speaker_confidence = UNCERTAIN_CONFIDENCE
        target.cleanup_note = CROSS_SPEAKER_NOTE


def prefix_score(fragment: Segment, target: Segment, *, gap: float | None) -> float:
    """Score whether a tiny fragment should prepend to the next segment."""
    score = 0.0
    if gap is None:
        return score

    if gap <= 0.5:
        score += 3.0
    elif gap <= 1.0:
        score += 2.0
    elif gap <= DEFAULT_MAX_FRAGMENT_GAP_SECONDS:
        score += 1.0
    else:
        return score

    if fragment.speaker == target.speaker:
        score += 2.0

    target_text = normalize_whitespace(target.text)
    if target_text and target_text[0].islower():
        score += 1.5

    fragment_text = normalize_whitespace(fragment.text)
    first_word = fragment_text.lower().split()[0] if fragment_text else ""
    if first_word in CONNECTOR_WORDS:
        score += 1.5

    if not ends_sentence(fragment_text):
        score += 1.0

    return score


def merge_same_speaker_with_gap(
    segments: list[Segment],
    max_gap_seconds: float,
) -> list[Segment]:
    if not segments:
        return []

    merged: list[Segment] = [deepcopy(segments[0])]

    for segment in segments[1:]:
        previous = merged[-1]
        gap = gap_seconds(previous, segment)
        same_speaker = segment.speaker == previous.speaker
        within_gap = gap is not None and gap <= max_gap_seconds

        if same_speaker and within_gap:
            merge_into_previous(previous, segment, cross_speaker=False)
            continue

        merged.append(deepcopy(segment))

    return merged


def continuation_score(target: Segment, fragment: Segment, *, gap: float | None) -> float:
    score = 0.0
    if gap is None:
        return score

    if gap <= 0.5:
        score += 3.0
    elif gap <= 1.0:
        score += 2.0
    elif gap <= DEFAULT_MAX_FRAGMENT_GAP_SECONDS:
        score += 1.0
    else:
        return score

    if target.speaker == fragment.speaker:
        score += 2.0

    if not ends_sentence(target.text):
        score += 1.5

    if last_token(target.text) in DANGLING_TOKENS:
        score += 2.0

    fragment_text = normalize_whitespace(fragment.text)
    if fragment_text and fragment_text[0].islower():
        score += 1.5

    first_word = fragment_text.lower().split()[0] if fragment_text else ""
    if first_word in CONNECTOR_WORDS:
        score += 1.5

    return score


def merge_tiny_fragments_once(
    segments: list[Segment],
    max_fragment_gap_seconds: float,
) -> tuple[list[Segment], bool]:
    if len(segments) < 2:
        return segments, False

    result = deepcopy(segments)
    changed = False
    skip_next = False

    for index in range(len(result)):
        if skip_next:
            skip_next = False
            continue

        segment = result[index]
        if not is_fragment(segment):
            continue

        prev_index = index - 1 if index > 0 else None
        next_index = index + 1 if index + 1 < len(result) else None

        prev_score = -1.0
        next_score = -1.0
        prev_gap = None
        next_gap = None

        if prev_index is not None:
            prev_gap = gap_seconds(result[prev_index], segment)
            if prev_gap is not None and prev_gap <= max_fragment_gap_seconds:
                prev_score = continuation_score(result[prev_index], segment, gap=prev_gap)

        if next_index is not None:
            next_gap = gap_seconds(segment, result[next_index])
            if next_gap is not None and next_gap <= max_fragment_gap_seconds:
                next_score = prefix_score(segment, result[next_index], gap=next_gap)

        if prev_score < MIN_FRAGMENT_MERGE_SCORE and next_score < MIN_FRAGMENT_MERGE_SCORE:
            continue

        merge_into_next_neighbor = next_score > prev_score

        if merge_into_next_neighbor:
            target_index = next_index
            cross_speaker = result[target_index].speaker != segment.speaker
            merge_into_next(result[target_index], segment, cross_speaker=cross_speaker)
        else:
            target_index = prev_index
            cross_speaker = result[target_index].speaker != segment.speaker
            merge_into_previous(result[target_index], segment, cross_speaker=cross_speaker)

        if target_index is None:
            continue

        result.pop(index)
        changed = True
        skip_next = True
        break

    return result, changed


def merge_tiny_fragments(
    segments: list[Segment],
    max_fragment_gap_seconds: float,
) -> list[Segment]:
    result = segments
    max_iterations = max(len(segments) * 2, 1)
    for _ in range(max_iterations):
        result, changed = merge_tiny_fragments_once(result, max_fragment_gap_seconds)
        if not changed:
            break
    return result


def clean_segments(segments: list[Segment], config: CleanConfig | None = None) -> list[Segment]:
    config = config or CleanConfig()
    segments = normalize_segments(segments)
    segments = merge_same_speaker_with_gap(segments, config.max_gap_seconds)
    segments = merge_tiny_fragments(segments, config.max_fragment_gap_seconds)
    segments = merge_same_speaker_with_gap(segments, config.max_gap_seconds)
    return segments


def parse_and_clean(
    raw_data: Any,
    *,
    transcript_id: str,
    recording_id: str,
    created_at: str,
    config: CleanConfig | None = None,
) -> dict[str, Any]:
    initial = parse_recall_raw(raw_data)
    cleaned = clean_segments(initial, config)
    return {
        "transcript_id": transcript_id,
        "recording_id": recording_id,
        "created_at": created_at,
        "source": "recall_ai",
        "cleanup_version": CLEANUP_VERSION,
        "segments": [segment.to_dict() for segment in cleaned],
    }


def seconds_to_mmss(seconds: float | None) -> str:
    if seconds is None:
        return "00:00"
    total = max(0, int(seconds))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


def render_readable_transcript(cleaned_doc: dict[str, Any]) -> str:
    lines: list[str] = []
    for segment in cleaned_doc.get("segments") or []:
        start = seconds_to_mmss(parse_time(str(segment.get("start_time", ""))))
        end = seconds_to_mmss(parse_time(str(segment.get("end_time", ""))))
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "")
        lines.append(f"[{start} - {end}] {speaker}: {text}")
    return "\n\n".join(lines) + ("\n" if lines else "")


def write_cleaned_outputs(
    cleaned_doc: dict[str, Any],
    *,
    cleaned_path: Path,
    readable_path: Path,
) -> None:
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    readable_path.parent.mkdir(parents=True, exist_ok=True)

    with cleaned_path.open("w", encoding="utf-8") as handle:
        json.dump(cleaned_doc, handle, ensure_ascii=False, indent=2)

    with readable_path.open("w", encoding="utf-8") as handle:
        handle.write(render_readable_transcript(cleaned_doc))


def parse_ids_from_filename(filename: str) -> tuple[str, str, str]:
    """Return (created_date, recording_id, transcript_id) from stable filename."""
    stem = Path(filename).stem
    parts = stem.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected raw filename format: {filename}")
    return parts[0], parts[1], parts[2]


def clean_raw_file(
    raw_path: Path,
    *,
    config: CleanConfig | None = None,
    force: bool = False,
) -> tuple[int, int] | None:
    """
    Clean one raw transcript file. Returns (before_count, after_count) or None if skipped.
    """
    config = config or CleanConfig()
    filename = raw_path.name
    cleaned_path = CLEANED_DIR / filename
    readable_path = READABLE_DIR / filename.replace(".json", ".txt")

    if not force and cleaned_path.exists() and readable_path.exists():
        return None

    with raw_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    created_date, recording_id, transcript_id = parse_ids_from_filename(filename)
    created_at = f"{created_date}T00:00:00Z"

    initial = parse_recall_raw(raw_data)
    before_count = len(initial)

    cleaned_doc = parse_and_clean(
        raw_data,
        transcript_id=transcript_id,
        recording_id=recording_id,
        created_at=created_at,
        config=config,
    )
    after_count = len(cleaned_doc["segments"])

    write_cleaned_outputs(
        cleaned_doc,
        cleaned_path=cleaned_path,
        readable_path=readable_path,
    )

    return before_count, after_count


def relative_path(path: Path) -> str:
    return str(path.relative_to(STORY_CHUNKING_ROOT)).replace("\\", "/")


def update_manifest_cleaned(
    manifest: dict[str, dict[str, Any]],
    *,
    transcript_id: str,
    recording_id: str,
    created_at: str,
    raw_path: Path,
    cleaned_path: Path,
    readable_path: Path,
    status: str = "done",
) -> None:
    entry = manifest.get(transcript_id, {})
    entry.update(
        {
            "transcript_id": transcript_id,
            "recording_id": recording_id,
            "created_at": created_at,
            "status": status,
            "raw_file_path": relative_path(raw_path),
            "cleaned_file_path": relative_path(cleaned_path),
            "readable_file_path": relative_path(readable_path),
            "cleaned_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if "processed_at" not in entry:
        entry["processed_at"] = entry["cleaned_at"]
    manifest[transcript_id] = entry
