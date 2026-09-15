"""JSONL helpers for Moment Detection dataset files."""

from __future__ import annotations

import json
import re
from pathlib import Path

INPUT_ID_PATTERN = re.compile(r"^moment_input_(\d+)$")


def load_input_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("id")
            if isinstance(row_id, str):
                ids.add(row_id)
    return ids


def load_output_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            input_id = row.get("input_id")
            if isinstance(input_id, str):
                ids.add(input_id)
    return ids


def load_inputs_by_id(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("id")
            if isinstance(row_id, str):
                rows[row_id] = row
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def next_input_index(existing_ids: set[str], start_index: int) -> int:
    max_index = start_index - 1
    for row_id in existing_ids:
        match = INPUT_ID_PATTERN.match(row_id)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max(max_index + 1, start_index)


def input_id_for(index: int) -> str:
    return f"moment_input_{index:05d}"


def ids_needed_for_target(
    existing_ids: set[str],
    target_count: int,
    start_index: int,
) -> list[str]:
    """Return sorted list of input IDs that should exist for target_count total."""
    needed: list[str] = []
    for index in range(start_index, start_index + target_count):
        row_id = input_id_for(index)
        if row_id not in existing_ids:
            needed.append(row_id)
    return needed


def missing_output_input_ids(
    input_rows: dict[str, dict],
    output_ids: set[str],
) -> list[str]:
    missing = [row_id for row_id in sorted(input_rows) if row_id not in output_ids]
    return missing
