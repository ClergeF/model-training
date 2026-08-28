"""
Checker for Model 2 skill + field dataset (cic_skill_field_dataset.jsonl).
Writes flagged rows to changes_needed_skills.json for adjust_datasets.py.
"""

import json
import os
import random
import sys

import requests
from dotenv import load_dotenv

from skill_map import SKILL_LIST_TEXT, SKILL_MAP, validate_skill_field_pair

load_dotenv()

OUTPUT_FILE = "cic_skill_field_dataset.jsonl"
KNOWLEDGE_FILE = "cic_knowledge_score_dataset.jsonl"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/owl-alpha")

SAMPLE_SIZE = 99999
REVIEW_TIMEOUT_SECONDS = 30
CHANGES_NEEDED_FILE = "changes_needed_skills.json"

SKILL_CHECK_SYSTEM_PROMPT = f"""You are a fair reviewer for a student skill labeling dataset.

Students are in a coding/AI internship. Each row has a statement plus a skill and field label.

Approved skills (field is fixed per skill):
{SKILL_LIST_TEXT}

Rules:
- The skill must be the ONE best match from the list above (exact name).
- The field must match that skill's field from the list — never invent fields.
- If the label is reasonable, label_ok is true.
- If the skill is clearly wrong but the field is roughly right, suggest a different skill from the SAME field when possible.

Return ONLY valid JSON:
{{"label_ok": true, "suggested_skill": "API Creation", "suggested_field": "Production", "note": "one sentence"}}

- suggested_field must match suggested_skill from the map.
- Return ONLY the JSON object."""


def load_dataset():
    valid_rows = []
    bad_lines = []
    seen = set()
    duplicates = 0

    if not os.path.exists(OUTPUT_FILE):
        print(f"File not found: {OUTPUT_FILE}")
        sys.exit(1)

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                bad_lines.append((line_num, "empty line"))
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines.append((line_num, "invalid JSON"))
                continue

            if "input_text" not in row or "skill" not in row or "field" not in row:
                bad_lines.append((line_num, "missing input_text, skill, or field"))
                continue

            text = row["input_text"]
            skill = row["skill"]
            field = row["field"]

            if not isinstance(text, str) or not text.strip():
                bad_lines.append((line_num, "invalid input_text"))
                continue

            if not validate_skill_field_pair(skill, field):
                bad_lines.append((line_num, f"invalid pair: {skill!r} / {field!r}"))
                continue

            if text in seen:
                duplicates += 1
                bad_lines.append((line_num, "duplicate input_text"))
                continue

            seen.add(text)
            valid_rows.append(row)

    return valid_rows, bad_lines, duplicates


def get_counts(rows, key):
    counts = {}
    for row in rows:
        val = row[key]
        counts[val] = counts.get(val, 0) + 1
    return counts


def print_section(title):
    print(f"\n{title}")
    print("  " + "-" * (len(title) - 2))


def print_distribution(rows, key, title):
    print_section(title)
    counts = get_counts(rows, key)
    total = len(rows)
    for name in sorted(counts, key=lambda x: (-counts[x], x)):
        n = counts[name]
        pct = (n / total * 100) if total else 0
        print(f"  {name}:  {n:4d}  ({pct:5.1f}%)")


def check_api_key():
    return bool(OPENROUTER_API_KEY)


def ask_openrouter_label_check(row):
    user_message = (
        f"Student statement: {json.dumps(row['input_text'])}\n"
        f"Saved skill: {row['skill']}\n"
        f"Saved field: {row['field']}\n"
        f"Is this label correct? Return ONLY JSON."
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SKILL_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=REVIEW_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        return None


def normalize_suggestion(result):
    """Enforce suggested skill/field against the map."""
    skill = (result.get("suggested_skill") or "").strip()
    if skill not in SKILL_MAP:
        return None, None
    return skill, SKILL_MAP[skill]


def run_openrouter_check(valid_rows):
    print_section("OPENROUTER SKILL LABEL CHECK")

    sample_count = min(SAMPLE_SIZE, len(valid_rows))
    sample = random.sample(valid_rows, sample_count)

    print(f"  Model:    {OPENROUTER_MODEL}")
    print(f"  Checking {sample_count} rows...\n")

    ok_count = 0
    flagged = []
    skipped = 0

    for i, row in enumerate(sample, start=1):
        print(f"  Checking row {i}/{sample_count}...", end=" ", flush=True)
        result = ask_openrouter_label_check(row)

        if result is None:
            print("failed — skipped")
            skipped += 1
            continue

        label_ok = result.get("label_ok", True)
        sug_skill, sug_field = normalize_suggestion(result)
        if sug_skill is None:
            sug_skill = row["skill"]
            sug_field = row["field"]

        wrong_skill = sug_skill != row["skill"]
        wrong_field = sug_field != row["field"]

        if not label_ok or wrong_skill or wrong_field:
            print(f"FLAGGED  ({row['skill']} → {sug_skill})")
            flagged.append({
                "input_text": row["input_text"],
                "saved_skill": row["skill"],
                "saved_field": row["field"],
                "suggested_skill": sug_skill,
                "suggested_field": sug_field,
                "note": result.get("note", ""),
            })
        else:
            print("OK")
            ok_count += 1

    checked = sample_count - skipped
    print(f"\n  Results: OK {ok_count}/{checked}, flagged {len(flagged)}/{checked}")
    return flagged


def save_changes_needed(flagged, path):
    with open(path, "w", encoding="utf-8") as out:
        json.dump(flagged, out, indent=2)
    print(f"\n  Wrote {len(flagged)} change(s) to {path}")
    print("  Apply with: python adjust_datasets.py skills")


def main():
    print("=" * 55)
    print("  CIC Skill + Field Dataset Checker (Model 2)")
    print("=" * 55)
    print(f"  File: {OUTPUT_FILE}")
    print("=" * 55)

    valid_rows, bad_lines, duplicates = load_dataset()
    total = len(valid_rows) + len(bad_lines)

    print_section("ROW COUNT AND FORMAT AUDIT")
    print(f"  Total lines read:  {total}")
    print(f"  Valid rows:        {len(valid_rows)}")
    print(f"  Bad/skipped lines: {len(bad_lines)}")
    print(f"  Duplicate texts:   {duplicates}")

    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            knowledge_count = sum(1 for line in f if line.strip())
        print(f"  Rows in {KNOWLEDGE_FILE}: {knowledge_count}")
        print(f"  Labeled in skill file:     {len(valid_rows)}")
        print(f"  Not yet in skill file:     {max(0, knowledge_count - len(valid_rows))} (approx.)")

    if bad_lines:
        print("\n  Bad line details (first 10):")
        for line_num, reason in bad_lines[:10]:
            print(f"    Line {line_num}: {reason}")

    if not valid_rows:
        print("\nNo valid rows to analyze.")
        return

    print_distribution(valid_rows, "field", "DISTRIBUTION BY FIELD")
    print_distribution(valid_rows, "skill", "DISTRIBUTION BY SKILL")

    print_section("OPENROUTER SKILL LABEL CHECK")
    if not check_api_key():
        print("  No API key — add OPENROUTER_API_KEY to .env and run again.")
        return

    flagged = run_openrouter_check(valid_rows)
    if flagged:
        save_changes_needed(flagged, CHANGES_NEEDED_FILE)
    else:
        if os.path.exists(CHANGES_NEEDED_FILE):
            os.remove(CHANGES_NEEDED_FILE)
        print("\n  No changes file written (no flagged rows).")

    print("\n" + "=" * 55)
    print("  Check complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
