"""
Generate more training examples for specific skills (Ollama).

Usage (from skill_classifier folder, Ollama running):

  # Add 20 examples for one skill
  python scripts/add_skill_examples.py --skill "Robotics" --count 20

  # Bring every skill up to at least 40 examples
  python scripts/add_skill_examples.py --fill-under --min-count 40

  # Preview without saving
  python scripts/add_skill_examples.py --skill "Sound Design" --count 5 --dry-run
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import requests

_DATA_GEN_ROOT = Path(__file__).resolve().parents[2]
if str(_DATA_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_GEN_ROOT))

from shared_utils.skill_to_field import SKILL_TO_FIELD

CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = CLASSIFIER_ROOT / "datasets" / "skill_dataset.jsonl"

MODEL_NAME = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT_SECONDS = 180
MAX_RETRIES = 3


def call_ollama(system_prompt, prompt):
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {"temperature": 0.7, "num_predict": 120},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        raw = response.json().get("response", "")
        return json.loads(raw)
    except requests.exceptions.ConnectionError:
        print("\nError: Ollama is not running. Start Ollama and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"  Generation failed: {e}")
        return None


def build_system_prompt(skill, field):
    return f"""You write synthetic student statements for a coding/AI internship dataset.

Each statement must clearly relate to this skill:
  Skill: {skill}
  Field: {field}

The student might be in a standup, Slack Huddle, demo, or code review.

Return ONLY valid JSON:
{{"input_text": "natural student sentence here"}}

Rules:
- input_text is what the student actually says (10–400 characters)
- Must be obviously about {skill}, not a generic filler line
- No student names, no markdown
- Return ONLY the JSON object"""


def generate_one(skill, field, existing_texts):
    system = build_system_prompt(skill, field)
    prompt = (
        f"Write ONE new unique student statement about {skill}.\n"
        f"Make it different from typical standup filler.\n"
        f"Return ONLY JSON with input_text."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        result = call_ollama(system, prompt)
        if not result or not isinstance(result, dict):
            continue

        text = result.get("input_text", "")
        if not isinstance(text, str):
            continue
        text = text.strip()

        if len(text) < 8:
            continue
        if len(text) > 500:
            continue
        if text in existing_texts:
            print("  Duplicate text — retrying")
            continue

        return text

    return None


def load_existing(dataset_path):
    texts = set()
    counts = Counter()
    if not dataset_path.exists():
        return texts, counts

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                t = row.get("input_text", "")
                s = (row.get("skill") or "").strip()
                if t:
                    texts.add(t)
                if s:
                    counts[s] += 1
            except json.JSONDecodeError:
                pass
    return texts, counts


def append_row(dataset_path, input_text, skill, dry_run):
    row = {"input_text": input_text, "skill": skill}
    if dry_run:
        print(f"  [dry-run] would save: {skill} | {input_text[:70]}...")
        return
    with open(dataset_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def add_for_skill(skill, count, dataset_path, existing_texts, counts, dry_run):
    if skill not in SKILL_TO_FIELD:
        print(f"Unknown skill: {skill}")
        print("Use an exact name from the approved skill map.")
        return 0

    field = SKILL_TO_FIELD[skill]
    added = 0
    print(f"\nAdding up to {count} examples for: {skill} ({field})")
    print(f"  Current count: {counts.get(skill, 0)}")

    for i in range(1, count + 1):
        print(f"  [{i}/{count}] generating...", end=" ", flush=True)
        text = generate_one(skill, field, existing_texts)
        if text is None:
            print("failed")
            continue

        append_row(dataset_path, text, skill, dry_run)
        existing_texts.add(text)
        counts[skill] += 1
        added += 1
        preview = text[:65] + "..." if len(text) > 65 else text
        print(f"OK — \"{preview}\"")

    return added


def skills_needing_fill(counts, min_count):
    needs = []
    for skill in SKILL_TO_FIELD:
        current = counts.get(skill, 0)
        if current < min_count:
            needs.append((skill, min_count - current))
    # Lowest count first
    needs.sort(key=lambda x: (counts.get(x[0], 0), x[0]))
    return needs


def main():
    parser = argparse.ArgumentParser(
        description="Add synthetic examples for underrepresented skills."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="skill_dataset.jsonl to append to",
    )
    parser.add_argument(
        "--skill",
        type=str,
        help='Target skill, e.g. "Robotics"',
    )
    parser.add_argument(
        "--count",
        type=int,
        default=15,
        help="How many examples to generate (with --skill)",
    )
    parser.add_argument(
        "--fill-under",
        action="store_true",
        help="For every skill below --min-count, generate enough to reach it",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=40,
        help="Target minimum per skill when using --fill-under",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate but do not write to the dataset file",
    )
    args = parser.parse_args()

    if not args.skill and not args.fill_under:
        parser.error("Use --skill NAME --count N  OR  --fill-under --min-count N")

    dataset_path = args.dataset.resolve()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    existing_texts, counts = load_existing(dataset_path)

    print("=" * 60)
    print("  Add skill examples (Model 2)")
    print("=" * 60)
    print(f"  Dataset: {dataset_path}")
    print(f"  Model:   {MODEL_NAME}")
    if args.dry_run:
        print("  Mode:    DRY RUN")
    print("=" * 60)

    total_added = 0

    try:
        if args.fill_under:
            plan = skills_needing_fill(counts, args.min_count)
            if not plan:
                print("\nAll skills already meet min-count. Nothing to add.")
                return
            print(f"\nPlan: fill {len(plan)} skill(s) up to {args.min_count} examples each")
            for skill, need in plan:
                total_added += add_for_skill(
                    skill, need, dataset_path, existing_texts, counts, args.dry_run
                )
        else:
            total_added += add_for_skill(
                args.skill.strip(),
                args.count,
                dataset_path,
                existing_texts,
                counts,
                args.dry_run,
            )
    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C).")

    print("\n" + "=" * 60)
    print(f"  Examples added this run: {total_added}")
    if not args.dry_run and total_added:
        print(f"  Saved to: {dataset_path}")
        print("  Run: python scripts/check_skill_balance.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
