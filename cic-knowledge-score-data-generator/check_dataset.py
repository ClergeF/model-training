import json
import os
import random
import sys

import requests
from dotenv import load_dotenv

# Load API key from .env file in the same folder as this script.
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "cic_knowledge_score_dataset.jsonl"
TARGET_ROWS = 1000

# OpenRouter API settings.
# The API key is loaded from the OPENROUTER_API_KEY variable in your .env file.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model to use for score sanity checking.
# gpt-4o-mini is cheap, fast, and follows JSON instructions reliably.
# You can swap this for any model listed at https://openrouter.ai/models
OPENROUTER_MODEL = "openrouter/owl-alpha"

# How many rows to check during the score sanity check.
# Set to 9999 so the entire dataset is always checked regardless of size.
# The script automatically caps this at however many valid rows exist.
SAMPLE_SIZE = 99999

# How far a suggested score can differ from the saved score before
# we consider the row "flagged" as potentially wrong.
FLAG_THRESHOLD = 0.15

# Timeout for each OpenRouter API call.
REVIEW_TIMEOUT_SECONDS = 30

# Written when rows are flagged during the OpenRouter check (for adjust_datasets.py).
CHANGES_NEEDED_FILE = "changes_needed_knowledge.json"

# ─────────────────────────────────────────────────────────────────────────────
# BAND DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Five main bands — the same ones used by the generator.
MAIN_BANDS = [
    {"label": "0.00–0.20", "min": 0.00, "max": 0.20},
    {"label": "0.21–0.40", "min": 0.21, "max": 0.40},
    {"label": "0.41–0.60", "min": 0.41, "max": 0.60},
    {"label": "0.61–0.80", "min": 0.61, "max": 0.80},
    {"label": "0.81–1.00", "min": 0.81, "max": 1.00},
]

# Ten granular sub-bands — 0.10-wide steps across the full 0.0–1.0 range.
GRANULAR_BANDS = [
    {"label": f"{lo:.2f}–{lo + 0.10:.2f}", "min": round(lo, 2), "max": round(lo + 0.10, 2)}
    for lo in [round(i * 0.10, 2) for i in range(10)]
]

# ─────────────────────────────────────────────────────────────────────────────
# REVIEW PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SANITY_CHECK_SYSTEM_PROMPT = """You are a fair reviewer checking whether knowledge scores are accurate for an intern student program.

These are high school or early college students in a coding, AI, and robotics internship. Score them at their level, not as senior engineers.

You will receive a student statement and a saved score.

Check if the score correctly matches this rubric:

0.00-0.20 — No evidence
The student gives filler, vague agreement, or says nothing technical.
Examples: "Yeah I agree." / "I worked on it today." / "It was confusing."

0.21-0.40 — Basic awareness
The student names a tool or concept but does not explain anything about it.
Examples: "I think we need an API." / "We should use a database." / "I used Python."

0.41-0.60 — Partial understanding
The student explains part of the idea but is still missing details or has an open question.
Examples: "I used Flask for the routes but I'm not sure how to handle authentication yet."

0.61-0.80 — Solid understanding
The student explains what they did AND why it makes sense or how it helps the project.
A student does NOT need to discuss tradeoffs or architecture to reach this band.
Examples:
  "I deployed the API to Heroku so it's accessible via the production URL now." → 0.72
  "I created an API key to authenticate requests to our web app so it's secure when users interact with the database." → 0.74
  "The bot needs to be invited into the private Slack channel before it can read messages because private channels block access." → 0.76

0.81-1.00 — Advanced understanding
The student explains tradeoffs, limitations, architecture decisions, debugging steps, or cause and effect.
Examples: "I separated the auth service from the main API because if auth goes down it shouldn't take the whole app offline."

IMPORTANT CALIBRATION:
- If a student explains what they did AND gives a reason or result, that is solid understanding (0.61-0.80).
- Only push a score below 0.40 if the student truly shows no understanding of what they are working on.
- Do not penalize students for not mentioning advanced concepts like tradeoffs or scalability unless the band specifically requires it.

Return ONLY valid JSON:
{"score_ok": true, "suggested_score": 0.72, "note": "one sentence"}

Rules:
- score_ok is true if the saved score is reasonable (within ~0.15 of correct)
- score_ok is false if the score is clearly wrong
- suggested_score is what you think the correct score should be
- note is one short sentence
- Return ONLY the JSON, nothing else"""


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    """
    Reads every line of the JSONL file and separates valid rows from bad ones.

    Returns:
        valid_rows   — list of dicts that passed all format checks
        bad_lines    — list of (line_number, reason) for lines that failed
        duplicate_count — number of duplicate input_text values found
    """
    valid_rows = []
    bad_lines = []
    seen_texts = set()
    duplicate_count = 0

    if not os.path.exists(OUTPUT_FILE):
        print(f"File not found: {OUTPUT_FILE}")
        print("Run generate_dataset.py first to create the dataset.")
        sys.exit(1)

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                bad_lines.append((line_num, "empty line"))
                continue

            # Try parsing JSON
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines.append((line_num, "invalid JSON"))
                continue

            # Must have both required fields
            if "input_text" not in row or "knowledge_score" not in row:
                bad_lines.append((line_num, "missing input_text or knowledge_score"))
                continue

            input_text = row["input_text"]
            score = row["knowledge_score"]

            # input_text must be a string
            if not isinstance(input_text, str):
                bad_lines.append((line_num, "input_text is not a string"))
                continue

            # knowledge_score must be a number
            if not isinstance(score, (int, float)):
                bad_lines.append((line_num, "knowledge_score is not a number"))
                continue

            # Score must be in valid range
            if not (0.0 <= score <= 1.0):
                bad_lines.append((line_num, f"score {score} is out of range"))
                continue

            # Text length check
            if len(input_text) < 8:
                bad_lines.append((line_num, f"input_text too short ({len(input_text)} chars)"))
                continue

            if len(input_text) > 500:
                bad_lines.append((line_num, f"input_text too long ({len(input_text)} chars)"))
                continue

            # Duplicate check
            if input_text in seen_texts:
                duplicate_count += 1
                bad_lines.append((line_num, "duplicate input_text"))
                continue

            seen_texts.add(input_text)
            valid_rows.append(row)

    return valid_rows, bad_lines, duplicate_count


def get_band_counts(rows, bands):
    """
    Counts how many rows fall into each band.
    Returns a dict mapping band label to count.
    """
    counts = {b["label"]: 0 for b in bands}
    for row in rows:
        score = row["knowledge_score"]
        for band in bands:
            if band["min"] <= score <= band["max"]:
                counts[band["label"]] += 1
                break
    return counts


def print_section(title):
    """Prints a section header."""
    print(f"\n{title}")
    print("  " + "-" * (len(title) - 2))


def print_row_audit(total_lines, valid_count, bad_lines, duplicate_count):
    """Prints the row count and format audit section."""
    print_section("ROW COUNT AND FORMAT AUDIT")
    print(f"  Total lines read:  {total_lines}")
    print(f"  Valid rows:        {valid_count}")
    print(f"  Bad/skipped lines: {len(bad_lines)}")
    print(f"  Duplicate texts:   {duplicate_count}")

    if bad_lines:
        print(f"\n  Bad line details:")
        for line_num, reason in bad_lines[:10]:  # Show at most 10 bad lines
            print(f"    Line {line_num}: {reason}")
        if len(bad_lines) > 10:
            print(f"    ... and {len(bad_lines) - 10} more")


def print_main_band_distribution(rows, valid_count):
    """Prints the five-band distribution with percentage and target."""
    print_section("MAIN BAND DISTRIBUTION")
    counts = get_band_counts(rows, MAIN_BANDS)
    target_per_band = TARGET_ROWS // len(MAIN_BANDS)

    for band in MAIN_BANDS:
        label = band["label"]
        count = counts[label]
        pct = (count / valid_count * 100) if valid_count > 0 else 0
        # Build a simple bar using block characters
        bar_width = int(pct / 2)
        bar = "█" * bar_width
        print(f"  {label}:  {count:4d} rows  ({pct:5.1f}%)  [target: {target_per_band}]  {bar}")


def print_granular_band_distribution(rows, valid_count):
    """Prints the ten granular sub-bands (0.10-wide steps)."""
    print_section("GRANULAR SUB-BAND BREAKDOWN  (0.10-wide steps)")
    counts = get_band_counts(rows, GRANULAR_BANDS)

    max_count = max(counts.values()) if counts else 1

    for band in GRANULAR_BANDS:
        label = band["label"]
        count = counts[label]
        pct = (count / valid_count * 100) if valid_count > 0 else 0
        # Scale bar to max count so it fits the terminal
        bar_width = int((count / max_count) * 30) if max_count > 0 else 0
        bar = "█" * bar_width
        print(f"  {label}:  {count:4d} rows  ({pct:5.1f}%)  {bar}")


def check_api_key():
    """
    Returns True if an OpenRouter API key is present in the environment.
    Returns False if the key is missing or empty.
    """
    return bool(OPENROUTER_API_KEY)


def ask_openrouter_score_check(row):
    """
    Sends one row to the OpenRouter API and asks if the score makes sense.

    Uses the OpenAI-compatible chat completions endpoint.
    Returns a dict with score_ok, suggested_score, and note.
    Returns None if the call fails or times out.
    """
    user_message = (
        f"Student statement: {json.dumps(row['input_text'])}\n"
        f"Saved score: {row['knowledge_score']}\n"
        f"Does this score match the rubric? Return ONLY the JSON object."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SANITY_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 100,
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
        api_data = response.json()

        # OpenAI-compatible response: the model's text is inside choices[0].message.content
        content = api_data["choices"][0]["message"]["content"]
        result = json.loads(content)
        return result

    except requests.exceptions.ReadTimeout:
        print("timed out — skipping")
        return None
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            print("invalid API key — check OPENROUTER_API_KEY in your .env file")
        else:
            print(f"HTTP error {response.status_code} — skipping")
        return None
    except (json.JSONDecodeError, KeyError, Exception):
        return None


def save_changes_needed(flagged, path):
    """Saves flagged rows to JSON for adjust_datasets.py."""
    changes = []
    for f in flagged:
        changes.append({
            "input_text": f["input_text"],
            "saved_knowledge_score": f["saved_score"],
            "suggested_knowledge_score": f["suggested_score"],
            "note": f.get("note", ""),
        })
    with open(path, "w", encoding="utf-8") as out:
        json.dump(changes, out, indent=2)
    print(f"\n  Wrote {len(changes)} change(s) to {path}")
    print("  Apply with: python adjust_datasets.py knowledge")


def run_openrouter_sanity_check(valid_rows):
    """
    Checks rows with OpenRouter and returns the list of flagged items.
    """
    print_section("OPENROUTER SCORE SANITY CHECK")

    sample_count = min(SAMPLE_SIZE, len(valid_rows))
    sample = random.sample(valid_rows, sample_count)

    print(f"  Model:    {OPENROUTER_MODEL}")
    print(f"  Sampling {sample_count} rows at random...")
    print(f"  A row is flagged if suggested score differs by more than {FLAG_THRESHOLD:.2f}\n")

    ok_count = 0
    flagged = []
    skipped = 0

    for i, row in enumerate(sample, start=1):
        print(f"  Checking row {i}/{sample_count}...", end=" ", flush=True)
        result = ask_openrouter_score_check(row)

        if result is None:
            print("failed — skipped")
            skipped += 1
            continue

        score_ok = result.get("score_ok", True)
        suggested = result.get("suggested_score", row["knowledge_score"])
        note = result.get("note", "")

        # Calculate difference between saved and suggested score
        diff = abs(suggested - row["knowledge_score"])

        # Flag the row if score_ok is false OR if the suggested score differs a lot
        if not score_ok or diff > FLAG_THRESHOLD:
            print(f"FLAGGED  (saved: {row['knowledge_score']}, suggested: {suggested})")
            flagged.append({
                "input_text": row["input_text"],
                "saved_score": row["knowledge_score"],
                "suggested_score": suggested,
                "note": note,
            })
        else:
            print(f"OK  (saved: {row['knowledge_score']}, suggested: {suggested})")
            ok_count += 1

    # Summary line
    checked = sample_count - skipped
    print(f"\n  Results from {checked} checked rows ({skipped} skipped due to timeout/error):")
    print(f"    Score OK:      {ok_count} / {checked}")
    print(f"    Flagged rows:  {len(flagged)} / {checked}")

    # Show flagged examples
    if flagged:
        print(f"\n  Flagged examples:")
        for f in flagged:
            # Truncate long text for display
            display_text = f["input_text"]
            if len(display_text) > 80:
                display_text = display_text[:77] + "..."
            print(f"\n    Text:            \"{display_text}\"")
            print(f"    Saved score:     {f['saved_score']}")
            print(f"    Suggested score: {f['suggested_score']}")
            if f["note"]:
                print(f"    Note:            {f['note']}")
    else:
        print("\n  All sampled scores look reasonable.")

    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  CIC Dataset Checker")
    print("=" * 55)
    print(f"  File:        {OUTPUT_FILE}")
    print(f"  Target rows: {TARGET_ROWS}")
    print("=" * 55)

    # ── Step 1: Load and audit every row ──
    valid_rows, bad_lines, duplicate_count = load_dataset()
    total_lines = len(valid_rows) + len(bad_lines)

    print_row_audit(total_lines, len(valid_rows), bad_lines, duplicate_count)

    if not valid_rows:
        print("\nNo valid rows found. Nothing else to check.")
        return

    # ── Step 2: Main band distribution ──
    print_main_band_distribution(valid_rows, len(valid_rows))

    # ── Step 3: Granular sub-band breakdown ──
    print_granular_band_distribution(valid_rows, len(valid_rows))

    # ── Step 4: Optional OpenRouter score sanity check ──
    print_section("OPENROUTER SCORE SANITY CHECK")

    if not check_api_key():
        print("  No API key found — skipping score check.")
        print("  Add your key to .env as OPENROUTER_API_KEY= and run again.")
    else:
        flagged = run_openrouter_sanity_check(valid_rows)
        if flagged:
            save_changes_needed(flagged, CHANGES_NEEDED_FILE)
        else:
            if os.path.exists(CHANGES_NEEDED_FILE):
                os.remove(CHANGES_NEEDED_FILE)
            print(f"\n  No changes file written (no flagged rows).")

    print("\n" + "=" * 55)
    print("  Check complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
