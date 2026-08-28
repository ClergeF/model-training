#!/usr/bin/env python3
"""
Sync completed Recall AI transcript artifacts for the MyVillage pipeline.

List → download → clean → save. Does not run story chunking or model training.

API reference (List Transcript):
  curl --request GET \\
       --url 'https://us-east-1.recall.ai/api/v1/transcript/?status_code=done' \\
       --header 'Authorization: RECALL_API_KEY' \\
       --header 'accept: application/json'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from dotenv import load_dotenv

from transcript_cleaner import (
    CLEANED_DIR,
    RAW_DIR,
    READABLE_DIR,
    CleanConfig,
    parse_and_clean,
    update_manifest_cleaned,
    write_cleaned_outputs,
)

STORY_CHUNKING_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = STORY_CHUNKING_ROOT / "data" / "transcripts" / "manifest.jsonl"

DEFAULT_RECALL_BASE_URL = "https://us-east-1.recall.ai"
LIST_TRANSCRIPT_PATH = "/api/v1/transcript/"

LIST_REQUEST_INTERVAL_SECONDS = 1.1
DOWNLOAD_REQUEST_INTERVAL_SECONDS = 0.25


def load_env() -> None:
    load_dotenv(STORY_CHUNKING_ROOT / ".env")


def load_api_key() -> str:
    load_env()
    api_key = os.getenv("RECALL_API_KEY", "").strip().strip('"').strip("'")
    if not api_key or api_key == "your_recall_api_key_here":
        raise SystemExit(
            "RECALL_API_KEY is not set. Copy .env.example to .env and add your Recall API key."
        )
    return api_key


def recall_base_url() -> str:
    load_env()
    raw = os.getenv("RECALL_API_BASE_URL", DEFAULT_RECALL_BASE_URL).strip().rstrip("/")
    if not raw:
        return DEFAULT_RECALL_BASE_URL

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"

    return DEFAULT_RECALL_BASE_URL


def recall_headers(api_key: str) -> dict[str, str]:
    key = api_key.strip().strip('"').strip("'")
    auth_value = key if key.lower().startswith("token ") else f"Token {key}"
    return {
        "Authorization": auth_value,
        "accept": "application/json",
    }


def extract_cursor(next_url: str | None) -> str | None:
    if not next_url:
        return None
    parsed = urlparse(next_url)
    values = parse_qs(parsed.query).get("cursor")
    return values[0] if values else None


def list_completed_transcripts(api_key: str) -> list[dict[str, Any]]:
    session = requests.Session()
    headers = recall_headers(api_key)
    base = recall_base_url()
    url = f"{base}{LIST_TRANSCRIPT_PATH}"
    params: dict[str, str] | None = {"status_code": "done"}

    artifacts: list[dict[str, Any]] = []
    page = 1

    while True:
        response = session.get(url, headers=headers, params=params, timeout=60)
        if response.status_code == 401:
            raise SystemExit(
                "Recall API returned 401 Unauthorized. Check that RECALL_API_KEY is valid "
                f"and RECALL_API_BASE_URL matches your workspace region "
                f"(currently {base}). Keys from us-west-2 will not work against us-east-1."
            )
        response.raise_for_status()
        payload = response.json()

        results = payload.get("results") or []
        artifacts.extend(results)
        print(f"Fetched page {page}: {len(results)} transcript artifact(s)")

        next_url = payload.get("next")
        cursor = extract_cursor(next_url)
        if not cursor:
            break

        page += 1
        params = {"status_code": "done", "cursor": cursor}
        time.sleep(LIST_REQUEST_INTERVAL_SECONDS)

    return artifacts


def created_date_prefix(created_at: str) -> str:
    if not created_at:
        return "unknown-date"
    return created_at[:10]


def stable_filename(created_at: str, recording_id: str, transcript_id: str) -> str:
    date_part = created_date_prefix(created_at)
    return f"{date_part}_{recording_id}_{transcript_id}.json"


def load_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}

    entries: dict[str, dict[str, Any]] = {}
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


def save_manifest(manifest: dict[str, dict[str, Any]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        for row in manifest.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def already_processed(
    transcript_id: str,
    raw_path: Path,
    cleaned_path: Path,
    readable_path: Path,
    manifest: dict[str, dict[str, Any]],
) -> bool:
    if transcript_id not in manifest:
        return False
    return raw_path.exists() and cleaned_path.exists() and readable_path.exists()


def download_transcript(download_url: str) -> Any:
    response = requests.get(download_url, timeout=120)
    response.raise_for_status()
    return response.json()


def artifact_fields(artifact: dict[str, Any]) -> tuple[str, str, str, str, str]:
    transcript_id = str(artifact.get("id", "")).strip()
    recording = artifact.get("recording") or {}
    recording_id = str(recording.get("id", "")).strip()
    created_at = str(artifact.get("created_at", "")).strip()
    status = str((artifact.get("status") or {}).get("code", "")).strip()
    download_url = str((artifact.get("data") or {}).get("download_url") or "").strip()
    return transcript_id, recording_id, created_at, status, download_url


def run_cleaning_step(
    raw_data: Any,
    *,
    transcript_id: str,
    recording_id: str,
    created_at: str,
    raw_path: Path,
    cleaned_path: Path,
    readable_path: Path,
    manifest: dict[str, dict[str, Any]],
    status: str,
    config: CleanConfig,
) -> None:
    cleaned_doc = parse_and_clean(
        raw_data,
        transcript_id=transcript_id,
        recording_id=recording_id,
        created_at=created_at,
        config=config,
    )
    write_cleaned_outputs(
        cleaned_doc,
        cleaned_path=cleaned_path,
        readable_path=readable_path,
    )
    print("Saved cleaned transcript")
    print("Saved readable transcript")

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


def process_artifact(
    artifact: dict[str, Any],
    *,
    index: int,
    total: int,
    force: bool,
    manifest: dict[str, dict[str, Any]],
    config: CleanConfig,
    re_clean_only: bool,
) -> str:
    transcript_id, recording_id, created_at, status, download_url = artifact_fields(artifact)

    if not transcript_id:
        print(f"Skipping artifact {index}/{total}: missing transcript id")
        return "skipped"

    filename = stable_filename(created_at, recording_id, transcript_id)
    raw_path = RAW_DIR / filename
    cleaned_path = CLEANED_DIR / filename
    readable_path = READABLE_DIR / filename.replace(".json", ".txt")

    if not re_clean_only and not download_url:
        print(f"Skipping transcript {index}/{total} ({transcript_id}): no download URL")
        return "skipped"

    if not force and already_processed(
        transcript_id, raw_path, cleaned_path, readable_path, manifest
    ):
        print(f"Skipping transcript {index}/{total} ({transcript_id}): already processed")
        return "skipped"

    if re_clean_only:
        if not raw_path.exists():
            print(f"Skipping transcript {index}/{total} ({transcript_id}): raw file missing")
            return "skipped"
        print(f"Re-cleaning transcript {index}/{total} ({transcript_id})")
        with raw_path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
    else:
        print(f"Downloading transcript {index}/{total} ({transcript_id})")
        raw_data = download_transcript(download_url)
        time.sleep(DOWNLOAD_REQUEST_INTERVAL_SECONDS)

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as handle:
            json.dump(raw_data, handle, ensure_ascii=False, indent=2)
        print("Saved raw transcript")

    run_cleaning_step(
        raw_data,
        transcript_id=transcript_id,
        recording_id=recording_id,
        created_at=created_at,
        raw_path=raw_path,
        cleaned_path=cleaned_path,
        readable_path=readable_path,
        manifest=manifest,
        status=status or "done",
        config=config,
    )
    return "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync completed Recall AI transcripts for story chunking data prep."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-clean transcripts even if already processed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N transcripts (useful for testing).",
    )
    parser.add_argument(
        "--re-clean-only",
        action="store_true",
        help="Skip API download; re-run rule-based cleaning on existing raw files.",
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
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    if args.re_clean_only:
        raw_files = sorted(RAW_DIR.glob("*.json"))
        if args.limit is not None:
            raw_files = raw_files[: max(args.limit, 0)]
        print(f"Re-cleaning {len(raw_files)} raw transcript(s)")
        processed_count = 0
        skipped_count = 0
        total = len(raw_files)

        for index, raw_path in enumerate(raw_files, start=1):
            stem = raw_path.stem
            parts = stem.split("_", 2)
            if len(parts) != 3:
                print(f"Skipping {raw_path.name}: unexpected filename format")
                skipped_count += 1
                continue

            created_date, recording_id, transcript_id = parts
            created_at = manifest.get(transcript_id, {}).get("created_at", f"{created_date}T00:00:00Z")
            status = manifest.get(transcript_id, {}).get("status", "done")

            artifact = {
                "id": transcript_id,
                "recording": {"id": recording_id},
                "created_at": created_at,
                "status": {"code": status},
                "data": {"download_url": ""},
            }
            result = process_artifact(
                artifact,
                index=index,
                total=total,
                force=args.force,
                manifest=manifest,
                config=config,
                re_clean_only=True,
            )
            if result == "processed":
                processed_count += 1
            else:
                skipped_count += 1

        save_manifest(manifest)
        print(f"Done. Processed {processed_count} transcript(s), skipped {skipped_count}.")
        return 0

    api_key = load_api_key()
    print("Listing completed Recall AI transcript artifacts...")
    artifacts = list_completed_transcripts(api_key)
    print(f"Found {len(artifacts)} transcript artifacts")

    if args.limit is not None:
        artifacts = artifacts[: max(args.limit, 0)]
        print(f"Processing first {len(artifacts)} transcript(s) due to --limit")

    processed_count = 0
    skipped_count = 0
    total = len(artifacts)

    for index, artifact in enumerate(artifacts, start=1):
        result = process_artifact(
            artifact,
            index=index,
            total=total,
            force=args.force,
            manifest=manifest,
            config=config,
            re_clean_only=False,
        )
        if result == "processed":
            processed_count += 1
        else:
            skipped_count += 1

    save_manifest(manifest)
    print(f"Done. Processed {processed_count} transcript(s), skipped {skipped_count}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
