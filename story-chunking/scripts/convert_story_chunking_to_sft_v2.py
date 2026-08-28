#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path

SYSTEM_PROMPT = """You are Transcript Architect, a meeting transcript segmentation model.

Your job is to divide a timestamped meeting transcript into coherent story sections based on meaningful semantic or topic changes.

Return ONLY valid JSON in this exact structure:

{
  "transcript_id": "string",
  "meeting_summary": "string",
  "story_sections": [
    {
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "section_title": "string",
      "summary": "string",
      "speakers": ["string"]
    }
  ]
}

SEGMENTATION RULES:
- Segment by meaningful topic/story changes, not fixed time intervals.
- Keep sections in chronological order.
- Sections must not overlap.
- Adjacent sections should share a boundary when one topic ends and another begins.
- Use timestamps grounded in the transcript timeline.
- The final section must end at the provided meeting duration.
- Prefer a few coherent sections over many tiny sections.
- Do not create a new section for a short interruption unless it becomes a meaningful discussion.
- If a discussion temporarily drifts and returns to the same topic, keep it in the same section when appropriate.

OUTPUT RULES:
- section_title should be short, specific, and grounded in the transcript.
- summary should describe the main discussion, activity, decision, problem, or outcome.
- speakers must contain only speakers who actually speak in that section.
- Do not invent speakers, topics, events, timestamps, decisions, or outcomes.
- Do not reproduce the transcript in the output.
- Do not output section_text.
- Do not output confidence scores.
- Do not include markdown or commentary outside the JSON object."""

TIMESTAMP_PATTERN = re.compile(
    r"\[(?:\d{1,2}:)?\d{1,2}:\d{2}\s*-\s*(?:\d{1,2}:)?\d{1,2}:\d{2}\]"
)

def validate_source_row(row, line_number):
    if not isinstance(row, dict):
        raise ValueError(f"Line {line_number}: row must be a JSON object.")

    for key in ("input", "output"):
        if key not in row:
            raise ValueError(f"Line {line_number}: missing '{key}'.")

    inp = row["input"]
    out = row["output"]

    for key in ("transcript_id", "duration", "transcript_text"):
        if key not in inp:
            raise ValueError(f"Line {line_number}: input missing '{key}'.")

    for key in ("transcript_id", "meeting_summary", "story_sections"):
        if key not in out:
            raise ValueError(f"Line {line_number}: output missing '{key}'.")

    if not isinstance(inp["transcript_text"], str) or not inp["transcript_text"].strip():
        raise ValueError(f"Line {line_number}: transcript_text must be non-empty.")

    if not TIMESTAMP_PATTERN.search(inp["transcript_text"]):
        raise ValueError(
            f"Line {line_number}: transcript_text does not appear to contain timestamp ranges."
        )

    if not isinstance(out["story_sections"], list) or not out["story_sections"]:
        raise ValueError(f"Line {line_number}: story_sections must be a non-empty list.")

def clean_target(row, line_number):
    out = row["output"]
    clean_sections = []

    required = ("start_time", "end_time", "section_title", "summary", "speakers")

    for index, section in enumerate(out["story_sections"]):
        missing = [field for field in required if field not in section]
        if missing:
            raise ValueError(
                f"Line {line_number}, section {index}: missing {', '.join(missing)}."
            )

        clean_sections.append({
            "start_time": section["start_time"],
            "end_time": section["end_time"],
            "section_title": section["section_title"],
            "summary": section["summary"],
            "speakers": section["speakers"],
        })

    return {
        "transcript_id": out["transcript_id"],
        "meeting_summary": out["meeting_summary"],
        "story_sections": clean_sections,
    }

def convert_row(row, line_number):
    validate_source_row(row, line_number)

    inp = row["input"]
    target = clean_target(row, line_number)

    user_prompt = (
        f"Transcript ID: {inp['transcript_id']}\n"
        f"Meeting duration: {inp['duration']}\n\n"
        f"Transcript:\n{inp['transcript_text']}"
    )

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps(
                    target,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        ],
    }

def load_and_convert(path):
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                source_row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_number}: invalid JSON: {exc}"
                ) from exc

            rows.append(convert_row(source_row, line_number))

    return rows

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Convert Transcript Architect JSONL into TRL SFT format."
    )
    parser.add_argument("input_file")
    parser.add_argument("--output-dir", default="story_chunking_sft")
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    if not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")

    rows = load_and_convert(source_path)

    if len(rows) < 2:
        raise ValueError("At least 2 valid examples are required.")

    random.Random(args.seed).shuffle(rows)

    validation_count = max(1, round(len(rows) * args.validation_size))
    validation_rows = rows[:validation_count]
    train_rows = rows[validation_count:]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    print(f"Total: {len(rows)}")
    print(f"Train: {len(train_rows)} -> {train_path}")
    print(f"Validation: {len(validation_rows)} -> {validation_path}")
    print("section_text and confidence were removed from SFT targets.")
    print("Reconstruct section_text later from timestamps + original transcript.")

if __name__ == "__main__":
    main()
