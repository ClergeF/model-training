#!/usr/bin/env python3
"""
Re-clean existing raw Recall transcripts using rule-based merge logic.

Reads from data/transcripts/raw/ only — raw files are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transcript_cleaner import (
    CLEANED_DIR,
    RAW_DIR,
    READABLE_DIR,
    CleanConfig,
    clean_raw_file,
    parse_ids_from_filename,
    relative_path,
    update_manifest_cleaned,
)

STORY_CHUNKING_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = STORY_CHUNKING_ROOT / "data" / "transcripts" / "manifest.jsonl"


def load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}

    entries: dict[str, dict] = {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            transcript_id = row.get("transcript_id")
            if transcript_id:
                entries[transcript_id] = row
    return entries


def save_manifest(manifest: dict[str, dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        for row in manifest.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-clean raw Recall transcripts with rule-based merge logic."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cleaned and readable outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N raw transcripts (useful for testing).",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=CleanConfig.max_gap_seconds,
        help="Max seconds between same-speaker segments to merge (default: 1.75).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CleanConfig(max_gap_seconds=args.max_gap)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    READABLE_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.json"))
    if args.limit is not None:
        raw_files = raw_files[: max(args.limit, 0)]

    if not raw_files:
        print("No raw transcript files found.")
        return 0

    manifest = load_manifest()
    processed_count = 0
    skipped_count = 0

    print(f"Found {len(raw_files)} raw transcript(s)")

    for index, raw_path in enumerate(raw_files, start=1):
        filename = raw_path.name
        cleaned_path = CLEANED_DIR / filename
        readable_path = READABLE_DIR / filename.replace(".json", ".txt")

        if not args.force and cleaned_path.exists() and readable_path.exists():
            print(f"Skipping {index}/{len(raw_files)} ({filename}): already cleaned")
            skipped_count += 1
            continue

        print(f"Cleaning {index}/{len(raw_files)} ({filename})")
        result = clean_raw_file(raw_path, config=config, force=True)

        if result is None:
            skipped_count += 1
            continue

        before_count, after_count = result
        print(f"  {before_count} -> {after_count} segments")

        try:
            created_date, recording_id, transcript_id = parse_ids_from_filename(filename)
        except ValueError as exc:
            print(f"  Warning: {exc}")
            processed_count += 1
            continue

        created_at = manifest.get(transcript_id, {}).get(
            "created_at", f"{created_date}T00:00:00Z"
        )
        status = manifest.get(transcript_id, {}).get("status", "done")

        update_manifest_cleaned(
            manifest,
            transcript_id=transcript_id,
            recording_id=recording_id,
            created_at=created_at,
            raw_path=raw_path,
            cleaned_path=cleaned_path,
            readable_path=readable_path,
            status=status,
        )
        processed_count += 1

    save_manifest(manifest)
    print(f"Done. Cleaned {processed_count} transcript(s), skipped {skipped_count}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
