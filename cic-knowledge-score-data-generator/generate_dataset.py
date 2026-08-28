import json
import os
import random
import sys

import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# Change these values if you want to adjust behavior.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TARGET_ROWS = 1000
OUTPUT_FILE = "cic_knowledge_score_dataset.jsonl"
MAX_REJECTS_IN_A_ROW = 50  # Print a warning if this many rejections happen in a row
GENERATION_TIMEOUT_SECONDS = 180
REVIEW_TIMEOUT_SECONDS = 240

# ─────────────────────────────────────────────────────────────────────────────
# SKILL MAP
# These skills are used as hidden topic inspiration during generation.
# They are NOT saved in the final JSONL file.
# ─────────────────────────────────────────────────────────────────────────────
SKILL_MAP = [
    {"skill": "Web Development", "field": "Applications"},
    {"skill": "Robotics", "field": "Robotics"},
    {"skill": "Data Curation", "field": "Data"},
    {"skill": "Data Collection", "field": "Data"},
    {"skill": "Game Design", "field": "Game Development"},
    {"skill": "MCP Server", "field": "Artificial Intelligence"},
    {"skill": "Prompt Engineering", "field": "Artificial Intelligence"},
    {"skill": "Game Coding", "field": "Game Development"},
    {"skill": "Workflow Building", "field": "Artificial Intelligence"},
    {"skill": "Mathematics", "field": "Soft Skills"},
    {"skill": "Knowledge Articulation", "field": "Soft Skills"},
    {"skill": "Model Training", "field": "Artificial Intelligence"},
    {"skill": "System Architecture", "field": "Artificial Intelligence"},
    {"skill": "App Development", "field": "Applications"},
    {"skill": "Animation", "field": "Game Development"},
    {"skill": "Game Concept", "field": "Game Development"},
    {"skill": "Sound Design", "field": "Production"},
    {"skill": "Game Production", "field": "Production"},
    {"skill": "Problem Solving", "field": "Soft Skills"},
    {"skill": "Teamwork", "field": "Soft Skills"},
    {"skill": "Group Leadership", "field": "Soft Skills"},
    {"skill": "3D Character Modeling", "field": "Digital Art"},
    {"skill": "API Creation", "field": "Production"},
    {"skill": "App Deployment", "field": "Production"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SCORE BANDS
# The dataset must be balanced across these five bands.
# ─────────────────────────────────────────────────────────────────────────────
BANDS = [
    {
        "name": "no_evidence",
        "label": "0.00–0.20",
        "min": 0.00,
        "max": 0.20,
        "description": (
            "The student gives filler, agreement, or vague language. "
            "No proof of understanding. Examples: 'Yeah I agree.' or 'I worked on it today.'"
        ),
    },
    {
        "name": "basic_awareness",
        "label": "0.21–0.40",
        "min": 0.21,
        "max": 0.40,
        "description": (
            "The student knows the topic exists but does not explain anything. "
            "Examples: 'I think we need an API.' or 'We should use a database.'"
        ),
    },
    {
        "name": "partial_understanding",
        "label": "0.41–0.60",
        "min": 0.41,
        "max": 0.60,
        "description": (
            "The student explains part of the idea but is still missing details or depth. "
            "Examples: 'The Slack API can get messages, but I still need to figure out permissions.'"
        ),
    },
    {
        "name": "solid_understanding",
        "label": "0.61–0.80",
        "min": 0.61,
        "max": 0.80,
        "description": (
            "The student explains what they did, why it works, or how it applies. "
            "Examples: 'The bot needs to be invited into the private channel because private channels block access.'"
        ),
    },
    {
        "name": "advanced_understanding",
        "label": "0.81–1.00",
        "min": 0.81,
        "max": 1.00,
        "description": (
            "The student explains tradeoffs, architecture, debugging, or cause and effect. "
            "Examples: 'I separated Slack message tracking from Huddle capture because the API handles them differently.'"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# These are the instructions Ollama receives for each type of call.
# ─────────────────────────────────────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = """You are a synthetic data generator for a student knowledge scoring dataset.

Your job is to write ONE realistic student statement from a coding and AI internship program called Coding in Color.

Students in this program work on projects involving web apps, AI models, robotics, game development, APIs, data collection, and more.

The student might be speaking during a Slack Huddle, a project meeting, a standup, a demo, or a code review.

The statement must sound natural and realistic — like something a real student would actually say, not a textbook.

You will be given:
- A target score band (how much knowledge the student should show)
- A skill or topic for inspiration
- A description of what that score band means

You must return ONLY valid JSON in this exact format:
{"input_text": "student statement here", "knowledge_score": 0.72}

Rules:
- input_text must be the student's actual words, spoken naturally
- input_text must be between 10 and 400 characters
- knowledge_score must be a number that falls within the target band range
- Do NOT include student names (real or fake)
- Do NOT use markdown (no **, ##, ```, or bullet points)
- Do NOT write a description — write the student's actual words
- Return ONLY the JSON object, nothing else, no explanation"""

REVIEW_SYSTEM_PROMPT = """You are a strict reviewer for a student knowledge scoring dataset.

Approve only if the student statement is realistic, useful training data, not too generic, and the score fits the rubric.

Rubric:
0.00-0.20 no evidence
0.21-0.40 basic awareness
0.41-0.60 partial understanding
0.61-0.80 solid understanding
0.81-1.00 advanced understanding

Return ONLY valid JSON in this exact format:
{"approved": true, "corrected_score": 0.72, "reason": "short explanation here"}

Rules:
- approved must be true or false
- corrected_score must be a number between 0.0 and 1.0
- corrected_score should be your best score estimate
- reason must be short
- Return ONLY the JSON object, nothing else"""


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(system_prompt, prompt, timeout_seconds=GENERATION_TIMEOUT_SECONDS, max_tokens=200):
    """
    Sends a request to the local Ollama API and returns the parsed JSON response.

    Ollama wraps the model's text output inside data["response"].
    We parse that text as JSON since we ask Ollama to use format="json".

    Returns a dict if successful, or None if anything goes wrong.
    """
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {
            "temperature": 0.4,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout_seconds)
        response.raise_for_status()

        # Ollama returns the full API response as JSON
        api_data = response.json()

        # The actual model output is inside the "response" key
        raw_text = api_data.get("response", "")

        # Parse the model's output as JSON (since we set format="json")
        result = json.loads(raw_text)
        return result

    except requests.exceptions.ConnectionError:
        print("\nError: Ollama is not running. Open Ollama or start it, then try again.")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        error_str = str(e).lower()
        if "404" in error_str or "model" in error_str or "not found" in error_str:
            print(f"\nError: Model issue. Make sure {MODEL_NAME} exists locally.")
            sys.exit(1)
        # Other HTTP errors — log and skip this attempt
        print(f"  HTTP error from Ollama: {e}")
        return None

    except requests.exceptions.ReadTimeout:
        print(f"  Ollama timed out after {timeout_seconds} seconds")
        return None

    except json.JSONDecodeError:
        # The model returned text that is not valid JSON — skip this attempt
        return None

    except Exception as e:
        print(f"  Unexpected error calling Ollama: {e}")
        return None


def get_score_band(score):
    """
    Returns the band dict that a given score falls into.
    Returns None if the score is out of range or not a number.
    """
    if not isinstance(score, (int, float)):
        return None
    for band in BANDS:
        if band["min"] <= score <= band["max"]:
            return band
    return None


def choose_underfilled_band(band_counts):
    """
    Returns the band that currently has the fewest accepted rows.
    If there is a tie, picks randomly among the tied bands.
    This keeps the dataset balanced across all five score bands.
    """
    min_count = min(band_counts.values())
    tied_bands = [name for name, count in band_counts.items() if count == min_count]
    chosen_name = random.choice(tied_bands)

    # Find and return the full band dict
    for band in BANDS:
        if band["name"] == chosen_name:
            return band

    # Fallback (should not happen)
    return BANDS[0]


def generate_candidate(target_band, skill_entry):
    """
    Asks Ollama to generate one candidate training row.

    Uses the target band and a skill/topic for context.
    Returns a dict with input_text and knowledge_score, or None if it fails.
    """
    prompt = (
        f"Generate a student statement for this target:\n\n"
        f"Target score band: {target_band['label']} ({target_band['name']})\n"
        f"What this band means: {target_band['description']}\n"
        f"Skill or topic for inspiration: {skill_entry['skill']} (field: {skill_entry['field']})\n\n"
        f"The student is in a coding and AI internship program.\n"
        f"Write one realistic statement that fits this score band.\n"
        f"Return ONLY the JSON object."
    )

    return call_ollama(
        GENERATION_SYSTEM_PROMPT,
        prompt,
        timeout_seconds=GENERATION_TIMEOUT_SECONDS,
        max_tokens=180,
    )


def review_candidate(candidate, target_band, band_counts):
    """
    Asks Ollama to review a candidate row for quality and score accuracy.

    Returns a dict with approved, corrected_score, and reason.
    Returns None if the review call fails.
    """
    prompt = (
        f"Candidate JSON: {json.dumps(candidate)}\n"
        f"Target band: {target_band['label']} ({target_band['name']})\n"
        f"Question: approve this row and correct the score if needed."
    )

    return call_ollama(
        REVIEW_SYSTEM_PROMPT,
        prompt,
        timeout_seconds=REVIEW_TIMEOUT_SECONDS,
        max_tokens=100,
    )


def validate_candidate(candidate, existing_texts, target_band):
    """
    Validates a candidate row using Python logic only (not Ollama).

    This is an independent check on top of the Ollama reviewer.
    Returns (True, None) if valid, or (False, "reason string") if invalid.
    """
    # Must be a dict
    if not isinstance(candidate, dict):
        return False, "Response is not a dict"

    # Must have both required fields
    if "input_text" not in candidate:
        return False, "Missing field: input_text"
    if "knowledge_score" not in candidate:
        return False, "Missing field: knowledge_score"

    input_text = candidate.get("input_text", "")
    score = candidate.get("knowledge_score")

    # input_text must be a string
    if not isinstance(input_text, str):
        return False, "input_text is not a string"

    # knowledge_score must be a number
    if not isinstance(score, (int, float)):
        return False, "knowledge_score is not a number"

    # Score must be in valid range
    if not (0.0 <= score <= 1.0):
        return False, f"knowledge_score {score} is out of range (must be 0.0–1.0)"

    # Text must be long enough to be meaningful
    if len(input_text) < 8:
        return False, f"input_text too short ({len(input_text)} chars, minimum 8)"

    # Text must not be too long
    if len(input_text) > 500:
        return False, f"input_text too long ({len(input_text)} chars, maximum 500)"

    # Text must not contain markdown formatting
    markdown_markers = ["**", "##", "```", "- [", "* "]
    for marker in markdown_markers:
        if marker in input_text:
            return False, f"input_text contains markdown: '{marker}'"

    # Text must not contain obvious fake student name patterns
    lower_text = input_text.lower()
    name_indicators = ["student:", "name:", "my name is", "i am called", "speaker:"]
    for indicator in name_indicators:
        if indicator in lower_text:
            return False, f"input_text contains name indicator: '{indicator}'"

    # Text must not be a duplicate of something already saved
    if input_text in existing_texts:
        return False, "Duplicate input_text"

    # Score must fall in a valid band
    band = get_score_band(score)
    if band is None:
        return False, f"knowledge_score {score} does not fall in any valid band"

    return True, None


def load_existing_dataset():
    """
    Loads all valid rows from the JSONL output file if it already exists.

    This enables the resume feature — if you stop the script and run it again,
    it will pick up from where it left off instead of starting over.

    Returns:
        rows         — list of all valid row dicts loaded from the file
        existing_texts — set of all input_text strings (for duplicate checking)
        band_counts  — dict mapping band name to count of rows in that band
    """
    rows = []
    existing_texts = set()
    band_counts = {b["name"]: 0 for b in BANDS}

    if not os.path.exists(OUTPUT_FILE):
        print("No existing dataset file found. Starting fresh.")
        return rows, existing_texts, band_counts

    print(f"Existing dataset found: {OUTPUT_FILE}")
    print("Loading existing rows...")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Only count rows that have both required fields
                if "input_text" in row and "knowledge_score" in row:
                    rows.append(row)
                    existing_texts.add(row["input_text"])
                    band = get_score_band(row["knowledge_score"])
                    if band:
                        band_counts[band["name"]] += 1
                else:
                    print(f"  Skipped line {i}: missing required fields")
            except json.JSONDecodeError:
                print(f"  Skipped line {i}: not valid JSON")

    print(f"Loaded {len(rows)} valid rows from existing file.")
    return rows, existing_texts, band_counts


def save_row(row):
    """
    Appends one accepted row to the JSONL output file.

    Only saves input_text and knowledge_score.
    Everything else (skill, field, band, review reason) is discarded.
    Each row is written immediately so no data is lost if the script is interrupted.
    """
    clean_row = {
        "input_text": row["input_text"],
        "knowledge_score": row["knowledge_score"],
    }
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean_row) + "\n")


def print_progress(accepted_count, rejected_count, band_counts):
    """
    Prints the current generation progress and band distribution.
    """
    print(f"\nAccepted: {accepted_count} / {TARGET_ROWS}")
    print(f"Rejected: {rejected_count}")
    print("Current band counts:")
    for band in BANDS:
        print(f"  {band['label']}: {band_counts[band['name']]}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Controls the active generation loop.

    The loop:
    1. Picks the score band that needs more examples.
    2. Picks a random skill for topic inspiration.
    3. Asks Ollama to generate one candidate row.
    4. Validates the row with Python.
    5. Asks Ollama to review the row.
    6. Accepts or rejects the row.
    7. Saves accepted rows immediately.
    8. Repeats until TARGET_ROWS is reached or the user stops with Ctrl+C.
    """
    print("=" * 55)
    print("  CIC Knowledge Score Dataset Generator")
    print("=" * 55)
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Target rows: {TARGET_ROWS}")
    print(f"  Output file: {OUTPUT_FILE}")
    print("=" * 55)
    print()

    # Load existing dataset (enables resume feature)
    rows, existing_texts, band_counts = load_existing_dataset()
    accepted_count = len(rows)
    rejected_count = 0
    consecutive_rejects = 0

    # Check if we already hit the target
    if accepted_count >= TARGET_ROWS:
        print(f"\nDataset already contains {accepted_count} rows. Target of {TARGET_ROWS} already reached.")
        print_progress(accepted_count, rejected_count, band_counts)
        return

    print(f"Starting from {accepted_count} accepted rows. Need {TARGET_ROWS - accepted_count} more.")
    print("Press Ctrl+C at any time to stop. Saved rows will not be lost.\n")

    try:
        while accepted_count < TARGET_ROWS:

            # ── Step 1: Pick the band that needs more examples ──
            target_band = choose_underfilled_band(band_counts)

            # ── Step 2: Pick a random skill for topic inspiration ──
            skill_entry = random.choice(SKILL_MAP)

            print(f"[Generating] Band: {target_band['label']}  |  Skill: {skill_entry['skill']}")

            # ── Step 3: Ask Ollama to generate a candidate row ──
            candidate = generate_candidate(target_band, skill_entry)

            if candidate is None:
                print("  Result: Generation failed — skipping\n")
                rejected_count += 1
                consecutive_rejects += 1
                _check_consecutive_rejects(consecutive_rejects)
                continue

            # Show a preview of what was generated
            preview_text = str(candidate.get("input_text", ""))[:70]
            print(f"  Generated score: {candidate.get('knowledge_score')}  |  Text: {preview_text}...")

            # ── Step 4: Validate with Python before sending to reviewer ──
            is_valid, rejection_reason = validate_candidate(candidate, existing_texts, target_band)

            if not is_valid:
                print(f"  Result: Rejected by Python validation — {rejection_reason}\n")
                rejected_count += 1
                consecutive_rejects += 1
                _check_consecutive_rejects(consecutive_rejects)
                continue

            # ── Step 5: Ask Ollama to review the candidate ──
            review = review_candidate(candidate, target_band, band_counts)

            if review is None:
                print("  Result: Review call failed — skipping\n")
                rejected_count += 1
                consecutive_rejects += 1
                _check_consecutive_rejects(consecutive_rejects)
                continue

            approved = review.get("approved", False)
            corrected_score = review.get("corrected_score", candidate["knowledge_score"])
            reason = review.get("reason", "")

            # Show the reviewer's decision
            print(f"  Reviewer: approved={approved}  |  corrected_score={corrected_score}")
            if reason:
                print(f"  Reason: {str(reason)[:100]}")

            # ── Step 6: Reject if the reviewer said no ──
            if not approved:
                print("  Result: Rejected by Ollama reviewer\n")
                rejected_count += 1
                consecutive_rejects += 1
                _check_consecutive_rejects(consecutive_rejects)
                continue

            # ── Step 7: Use the corrected score if it is valid ──
            if isinstance(corrected_score, (int, float)) and 0.0 <= corrected_score <= 1.0:
                final_score = round(float(corrected_score), 4)
            else:
                # Fall back to the originally generated score
                final_score = round(float(candidate["knowledge_score"]), 4)

            # ── Step 8: Build and save the final row ──
            final_row = {
                "input_text": candidate["input_text"],
                "knowledge_score": final_score,
            }

            save_row(final_row)
            existing_texts.add(final_row["input_text"])

            # Update band count for the final score (may differ from target band)
            saved_band = get_score_band(final_score)
            if saved_band:
                band_counts[saved_band["name"]] += 1

            accepted_count += 1
            consecutive_rejects = 0  # Reset the streak counter

            print(f"  Result: Accepted  |  Final score: {final_score}\n")

            # ── Print progress every 10 accepted rows ──
            if accepted_count % 10 == 0 or accepted_count == TARGET_ROWS:
                print_progress(accepted_count, rejected_count, band_counts)

    except KeyboardInterrupt:
        # User pressed Ctrl+C — stop gracefully without losing any saved data
        print("\n\nStopped by user (Ctrl+C).")

    # Final summary
    print("=" * 55)
    print(f"  Accepted rows saved: {accepted_count}")
    print(f"  Total rejected:      {rejected_count}")
    print(f"  Output file:         {OUTPUT_FILE}")
    print("=" * 55)

    if accepted_count < TARGET_ROWS:
        print(f"\nRun the script again to continue from {accepted_count} rows.")
    else:
        print("\nTarget reached. Dataset is complete.")


def _check_consecutive_rejects(consecutive_rejects):
    """
    Prints a warning if too many rejections happen in a row.
    This helps the user notice if something is wrong (e.g., bad prompts or Ollama issues).
    """
    if consecutive_rejects > 0 and consecutive_rejects % MAX_REJECTS_IN_A_ROW == 0:
        print(
            f"\n  Warning: {consecutive_rejects} consecutive rejections. "
            "This may indicate a prompt issue or Ollama instability. "
            "Continuing anyway...\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
