import json
import os
import sys

import requests

from skill_map import (
    FIELD_TO_SKILLS,
    SKILL_LIST_TEXT,
    SKILL_MAP,
    normalize_field_name,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Source file — the Model 3 knowledge score dataset.
# The script adds a "skill" field to each row after labeling, but never
# removes or changes "input_text" or "knowledge_score".
INPUT_FILE = "cic_knowledge_score_dataset.jsonl"

# Output file — the new Model 2 skill/field dataset.
OUTPUT_FILE = "cic_skill_field_dataset.jsonl"

# How long to wait for each Ollama labeling call.
LABEL_TIMEOUT_SECONDS = 180

# How many times to retry a row when Ollama returns an invalid skill.
MAX_RETRIES = 3

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

LABEL_SYSTEM_PROMPT = f"""You are a skill classifier for a student coding and AI internship program.

Your job is to read a student statement and pick the ONE skill from the approved list that best matches what the student is talking about.

Approved skills (you must use one of these exactly):
{SKILL_LIST_TEXT}

Rules:
- Pick the skill that best matches the topic of the student's statement.
- You MUST use the exact skill name from the list above — no variations, no lowercase, no similar names.
- The field is determined by the skill. Do not invent a field.
- If the student's statement is vague or generic, pick the skill that is the closest reasonable match.
- Return ONLY valid JSON in this exact format:
{{"skill": "API Creation", "field": "Production"}}
- Return ONLY the JSON object, nothing else."""

RETRY_SYSTEM_PROMPT = f"""You are a skill classifier. You returned an invalid skill name on your last attempt.

You MUST pick from this exact list only:
{SKILL_LIST_TEXT}

Copy the skill name exactly as shown — including capitalization and spacing.
Return ONLY valid JSON:
{{"skill": "API Creation", "field": "Production"}}"""


def build_field_pick_prompt(field, skills):
    """Prompt that forces Ollama to pick one skill from a single field."""
    skill_lines = "\n".join(f'  - "{s}"' for s in skills)
    return f"""You are a skill classifier. The student's statement belongs to field "{field}".

Pick the ONE skill below that best matches the statement. Use the exact skill name.

Skills in this field only:
{skill_lines}

Return ONLY valid JSON:
{{"skill": "{skills[0]}", "field": "{field}"}}"""


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(system_prompt, prompt):
    """
    Sends a request to the local Ollama API and returns the parsed JSON response.
    Returns None if the call fails for any reason.
    """
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {
            "temperature": 0.1,   # Low temperature for consistent classification
            "num_predict": 60,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=LABEL_TIMEOUT_SECONDS)
        response.raise_for_status()
        api_data = response.json()
        raw_text = api_data.get("response", "")
        result = json.loads(raw_text)
        return result

    except requests.exceptions.ConnectionError:
        print("\nError: Ollama is not running. Open Ollama or start it, then try again.")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        error_str = str(e).lower()
        if "404" in error_str or "not found" in error_str:
            print(f"\nError: Model issue. Make sure {MODEL_NAME} exists locally.")
            sys.exit(1)
        print(f"  HTTP error from Ollama: {e}")
        return None

    except requests.exceptions.ReadTimeout:
        print(f"  Timed out after {LABEL_TIMEOUT_SECONDS}s — skipping")
        return None

    except json.JSONDecodeError:
        return None

    except Exception as e:
        print(f"  Unexpected error: {e}")
        return None


def validate_label(result):
    """
    Checks that Ollama returned a valid skill from the approved skill map.

    Returns (skill, field) if valid, or (None, reason_string) if invalid.
    The field is always taken from the Python skill map — not from Ollama's response.
    """
    if not isinstance(result, dict):
        return None, "Response is not a dict"

    skill = result.get("skill", "")

    if not isinstance(skill, str) or not skill.strip():
        return None, "Missing or empty skill"

    skill = skill.strip()

    if skill not in SKILL_MAP:
        return None, f'"{skill}" is not in the approved skill map'

    field = SKILL_MAP[skill]
    return skill, field


def resolve_skill_from_field(input_text, result):
    """
    When Ollama returns a field name (e.g. "Game Development") instead of a skill,
    pick the best skill from that field's approved list.

    Returns (skill, field) or (None, None).
    """
    if not isinstance(result, dict):
        return None, None

    skill_raw = (result.get("skill") or "").strip()
    field_raw = (result.get("field") or "").strip()

    # Model may put the field in either key.
    field = normalize_field_name(skill_raw) or normalize_field_name(field_raw)
    if not field:
        return None, None

    skills = FIELD_TO_SKILLS[field]
    if len(skills) == 1:
        return skills[0], field

    print(f"  Field match: \"{field}\" — picking best skill from {len(skills)} options...")
    system = build_field_pick_prompt(field, skills)
    user = (
        f"Student statement:\n\"{input_text}\"\n\n"
        f"Pick the best skill from the list for field \"{field}\". Return ONLY JSON."
    )

    pick = call_ollama(system, user)
    if pick is None:
        return None, None

    skill, _ = validate_label(pick)
    if skill is not None:
        return skill, field

    # Last resort: skill name in pick might still be wrong — match case-insensitively
    pick_skill = (pick.get("skill") or "").strip()
    for s in skills:
        if s.lower() == pick_skill.lower():
            return s, field

    return None, None


def label_input_text(input_text, row_num):
    """
    Asks Ollama to classify one input_text into a skill and field.
    Retries up to MAX_RETRIES times if Ollama returns an invalid skill.

    Returns (skill, field) if successful, or (None, None) if all retries fail.
    """
    prompt = (
        f"Student statement:\n\"{input_text}\"\n\n"
        f"Which skill from the approved list best matches this statement?\n"
        f"Return ONLY the JSON object."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        # Use the stricter retry prompt after the first failure
        system = LABEL_SYSTEM_PROMPT if attempt == 1 else RETRY_SYSTEM_PROMPT

        result = call_ollama(system, prompt)

        if result is None:
            print(f"  Attempt {attempt}/{MAX_RETRIES}: call failed")
            continue

        skill, field_or_reason = validate_label(result)

        if skill is not None:
            return skill, field_or_reason

        # Ollama may have returned a field name instead of a skill — resolve it.
        skill, field = resolve_skill_from_field(input_text, result)
        if skill is not None:
            print(f"  Resolved field → skill: {skill} / {field}")
            return skill, field

        print(f"  Attempt {attempt}/{MAX_RETRIES}: invalid — {field_or_reason}")

    return None, None


def load_input_rows():
    """
    Reads all input_text values from the Model 3 source file.
    Returns a list of input_text strings in order.
    Skips any lines that are missing input_text.
    """
    rows = []

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Source file not found: {INPUT_FILE}")
        print("Make sure you are running this script from the project folder.")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                text = row.get("input_text", "")
                if text and isinstance(text, str):
                    rows.append(text)
                else:
                    print(f"  Skipped line {i}: missing input_text")
            except json.JSONDecodeError:
                print(f"  Skipped line {i}: invalid JSON")

    return rows


def load_already_labeled():
    """
    Reads the output file if it already exists and returns a set of
    input_text strings that have already been labeled.
    This enables the resume feature.
    """
    already_labeled = set()

    if not os.path.exists(OUTPUT_FILE):
        return already_labeled

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                text = row.get("input_text", "")
                if text:
                    already_labeled.add(text)
            except json.JSONDecodeError:
                pass

    return already_labeled


def save_labeled_row(input_text, skill, field):
    """
    Appends one labeled row to the output JSONL file.
    Only saves input_text, skill, and field.
    """
    row = {
        "input_text": input_text,
        "skill": skill,
        "field": field,
    }
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def merge_skills_into_knowledge_file():
    """
    Reads cic_skill_field_dataset.jsonl to get the current skill labels,
    then rewrites cic_knowledge_score_dataset.jsonl so that every row that
    has been labeled gets a "skill" field added between "input_text" and
    "knowledge_score".

    Rows that have not been labeled yet are left unchanged.
    Rows that already have a "skill" field are updated if the label changed.

    Returns the number of rows that were updated (added or changed skill).
    """
    # Build a lookup of input_text -> skill from the skill/field output file.
    skill_lookup = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    text = row.get("input_text", "")
                    skill = row.get("skill", "")
                    if text and skill:
                        skill_lookup[text] = skill
                except json.JSONDecodeError:
                    pass

    if not skill_lookup:
        return 0

    # Read all rows from the knowledge score file.
    if not os.path.exists(INPUT_FILE):
        return 0

    original_rows = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                original_rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Merge skill into each row, preserving field order:
    # input_text → skill → knowledge_score
    updated_count = 0
    merged_rows = []

    for row in original_rows:
        text = row.get("input_text", "")
        if text in skill_lookup:
            new_skill = skill_lookup[text]
            old_skill = row.get("skill")

            # Build a new ordered dict with skill inserted in the right position.
            merged = {"input_text": text, "skill": new_skill}
            # Copy any other fields except input_text and skill (preserves knowledge_score etc.)
            for key, value in row.items():
                if key not in ("input_text", "skill"):
                    merged[key] = value

            if old_skill != new_skill:
                updated_count += 1

            merged_rows.append(merged)
        else:
            merged_rows.append(row)

    # Write the updated rows back to the knowledge score file.
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row) + "\n")

    return updated_count


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  CIC Skill + Field Labeling Script (Model 2)")
    print("=" * 55)
    print(f"  Source file:  {INPUT_FILE}")
    print(f"  Output file:  {OUTPUT_FILE}")
    print(f"  Model:        {MODEL_NAME}")
    print("=" * 55)
    print()

    # Load all input rows from the Model 3 dataset.
    # This is read fresh every run so new rows added to the file are picked up automatically.
    all_texts = load_input_rows()
    print(f"Rows found in {INPUT_FILE}: {len(all_texts)}  (read live — no hardcoded count)")

    # Load already-labeled rows for resume support.
    already_labeled = load_already_labeled()
    if already_labeled:
        print(f"Resuming — {len(already_labeled)} rows already labeled, skipping those.")
    else:
        print("No existing output file found. Starting fresh.")

    # Filter down to only the rows that still need labeling.
    pending = [t for t in all_texts if t not in already_labeled]
    print(f"Rows still to label: {len(pending)}")
    print("Press Ctrl+C at any time to stop. Progress is saved after each row.\n")

    if not pending:
        print("All rows are already labeled.")
        print("\nMerging skills into knowledge score file...")
        updated = merge_skills_into_knowledge_file()
        print(f"  {updated} rows updated in {INPUT_FILE}")
        return

    labeled_count = len(already_labeled)
    skipped_count = 0

    try:
        for i, input_text in enumerate(pending, start=1):
            # Show a short preview of the text.
            preview = input_text[:70] + "..." if len(input_text) > 70 else input_text
            print(f"[{labeled_count + 1} labeled | Row {i}/{len(pending)}]  \"{preview}\"")

            skill, field = label_input_text(input_text, i)

            if skill is None:
                print(f"  Result: Skipped — could not get a valid label after {MAX_RETRIES} attempts\n")
                skipped_count += 1
                continue

            save_labeled_row(input_text, skill, field)
            labeled_count += 1
            print(f"  Result: {skill} / {field}\n")

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C).")

    # ── Merge skills back into the knowledge score file ──
    print("\nMerging skills into knowledge score file...")
    updated = merge_skills_into_knowledge_file()
    print(f"  {updated} rows updated in {INPUT_FILE}")

    print("=" * 55)
    print(f"  Rows in source file:      {len(all_texts)}")
    print(f"  Labeled rows saved:       {labeled_count}")
    print(f"  Rows updated in {INPUT_FILE.split('.')[0]}: {updated}")
    print(f"  Skipped (no valid label): {skipped_count}")
    print(f"  Skill/field output file:  {OUTPUT_FILE}")
    print("=" * 55)

    if skipped_count > 0:
        print(f"\nNote: {skipped_count} rows were skipped. Run the script again to retry them.")


if __name__ == "__main__":
    main()
