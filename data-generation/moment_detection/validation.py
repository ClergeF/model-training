"""Deterministic validation for Moment Detection inputs and outputs."""

from __future__ import annotations

import re

from .schemas import (
    CATEGORIES,
    DETECTION_STATUSES,
    FORBIDDEN_INPUT_KEYS,
    INPUT_SOURCES,
    MOMENT_OBJECT_KEYS,
    MOMENT_TYPES,
    OUTPUT_TOP_KEYS,
)

META_PREFIXES = (
    "here is your requested conversation",
    "here's your requested conversation",
    "sure, here is",
    "below is",
)

MIN_WORDS = 25
MAX_WORDS = 400
PREFERRED_MIN_WORDS = 50
PREFERRED_MAX_WORDS = 250


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def _contains_forbidden_keys(obj: dict, prefix: str = "") -> str | None:
    for key, value in obj.items():
        key_lower = key.lower()
        if key_lower in FORBIDDEN_INPUT_KEYS or key in FORBIDDEN_INPUT_KEYS:
            return f"forbidden key {prefix}{key}"
        if isinstance(value, dict):
            nested = _contains_forbidden_keys(value, prefix=f"{prefix}{key}.")
            if nested:
                return nested
    return None


def validate_input_row(row: dict) -> tuple[bool, str | None]:
    if not isinstance(row, dict):
        return False, "row must be a JSON object"

    forbidden = _contains_forbidden_keys(row)
    if forbidden:
        return False, forbidden

    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id.startswith("moment_input_"):
        return False, "id must be moment_input_NNNNN"

    if "output" in row or "input_id" in row:
        return False, "input row must not contain output fields"

    inp = row.get("input")
    if not isinstance(inp, dict):
        return False, "missing input object"

    original_text = inp.get("original_text")
    if not isinstance(original_text, str) or not original_text.strip():
        return False, "original_text must be non-empty string"

    lower = original_text.strip().lower()
    for prefix in META_PREFIXES:
        if lower.startswith(prefix):
            return False, "original_text contains generator meta preamble"

    words = word_count(original_text)
    if words < MIN_WORDS:
        return False, f"original_text too short ({words} words)"
    if words > MAX_WORDS:
        return False, f"original_text too long ({words} words)"

    source = inp.get("source")
    if source not in INPUT_SOURCES:
        return False, f"invalid source: {source!r}"

    categories = inp.get("categories")
    if categories is None:
        return False, "categories field required (use empty list)"
    if not isinstance(categories, list):
        return False, "categories must be a list"

    allowed_input_keys = {"original_text", "source", "categories"}
    extra = set(inp.keys()) - allowed_input_keys
    if extra:
        return False, f"unexpected input keys: {sorted(extra)}"

    return True, None


def _names_appear_in_text(name: str, text: str) -> bool:
    if not name or not isinstance(name, str):
        return False
    return name.lower() in text.lower()


def validate_output_row(
    row: dict,
    *,
    input_row: dict | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(row, dict):
        return False, "row must be a JSON object"

    input_id = row.get("input_id")
    if not isinstance(input_id, str) or not input_id.startswith("moment_input_"):
        return False, "input_id must be moment_input_NNNNN"

    output = row.get("output")
    if not isinstance(output, dict):
        return False, "missing output object"

    extra_top = set(row.keys()) - {"input_id", "output"}
    if extra_top:
        return False, f"unexpected top-level keys: {sorted(extra_top)}"

    missing_keys = OUTPUT_TOP_KEYS - set(output.keys())
    if missing_keys:
        return False, f"output missing keys: {sorted(missing_keys)}"

    if input_row is not None:
        if input_row.get("id") != input_id:
            return False, "input_id does not match provided input row"
        inp = input_row.get("input") or {}
        expected_text = inp.get("original_text", "")
        expected_source = inp.get("source", "")
        if output.get("originalText") != expected_text:
            return False, "originalText must match input original_text exactly"
        if output.get("source") != expected_source:
            return False, "source must match input source exactly"

    status = output.get("detectionStatus")
    if status not in DETECTION_STATUSES:
        return False, f"invalid detectionStatus: {status!r}"

    moments = output.get("moments")
    if not isinstance(moments, list):
        return False, "moments must be a list"

    moment_count = output.get("momentCount")
    if not isinstance(moment_count, int) or moment_count < 0:
        return False, "momentCount must be a non-negative integer"

    if moment_count != len(moments):
        return False, "momentCount must equal len(moments)"

    missing_info = output.get("missingInformation")
    if not isinstance(missing_info, list):
        return False, "missingInformation must be a list"

    original_text = str(output.get("originalText", ""))

    if status == "CONFIRMED" and moment_count == 0:
        return False, "CONFIRMED requires at least one moment"

    if status in ("POTENTIAL", "NOT_A_MOMENT") and moment_count != 0:
        return False, f"{status} requires momentCount 0"

    for index, moment in enumerate(moments):
        if not isinstance(moment, dict):
            return False, f"moment {index} must be an object"
        for key in ("contributor", "beneficiary", "summary", "momentType", "category"):
            if key not in moment or not str(moment[key]).strip():
                return False, f"moment {index} missing {key}"
        if moment.get("momentType") not in MOMENT_TYPES:
            return False, f"moment {index} invalid momentType"
        if moment.get("category") not in CATEGORIES:
            return False, f"moment {index} invalid category"
        unknown = set(moment.keys()) - MOMENT_OBJECT_KEYS
        if unknown:
            return False, f"moment {index} unknown keys: {sorted(unknown)}"
        if not _names_appear_in_text(str(moment.get("contributor")), original_text):
            return False, f"moment {index} contributor not found in originalText"
        if not _names_appear_in_text(str(moment.get("beneficiary")), original_text):
            return False, f"moment {index} beneficiary not found in originalText"

    return True, None
