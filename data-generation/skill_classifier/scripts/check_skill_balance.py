"""
Check how many training examples each skill has.

Usage (from skill_classifier folder):
  python scripts/check_skill_balance.py

Options:
  python scripts/check_skill_balance.py --min-count 30
  python scripts/check_skill_balance.py --dataset datasets/skill_dataset.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# shared_utils lives under data-generation/
_DATA_GEN_ROOT = Path(__file__).resolve().parents[2]
if str(_DATA_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_GEN_ROOT))

from shared_utils.skill_to_field import SKILL_TO_FIELD

CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = CLASSIFIER_ROOT / "datasets" / "skill_dataset.jsonl"


def load_skill_counts(dataset_path):
    counts = Counter()
    unknown_skills = Counter()
    total = 0
    duplicates = 0
    seen_texts = set()

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Warning: invalid JSON on line {line_num}")
                continue

            text = row.get("input_text", "")
            skill = (row.get("skill") or "").strip()
            if not text or not skill:
                continue

            total += 1
            if text in seen_texts:
                duplicates += 1
            seen_texts.add(text)

            if skill in SKILL_TO_FIELD:
                counts[skill] += 1
            else:
                unknown_skills[skill] += 1

    return counts, unknown_skills, total, duplicates


def bar(count, max_count, width=30):
    if max_count <= 0:
        return ""
    filled = int((count / max_count) * width)
    return "#" * filled


def main():
    parser = argparse.ArgumentParser(
        description="Show class balance (examples per skill) for Model 2 training."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to skill_dataset.jsonl",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=30,
        help="Skills below this count are flagged as LOW (default: 30)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    counts, unknown, total, duplicates = load_skill_counts(dataset_path)
    min_count = args.min_count

    print("=" * 60)
    print("  Skill class balance (Model 2)")
    print("=" * 60)
    print(f"  Dataset:     {dataset_path}")
    print(f"  Total rows:  {total}")
    print(f"  Duplicates:  {duplicates}  (same input_text seen more than once)")
    print(f"  Low threshold: fewer than {min_count} examples")
    print("=" * 60)

    # Every approved skill, sorted by count (lowest first).
    all_skills = sorted(SKILL_TO_FIELD.keys(), key=lambda s: (counts.get(s, 0), s))
    max_count = max(counts.values()) if counts else 1

    low_skills = []
    ok_skills = []
    missing_skills = []

    print(f"\n{'Skill':<28} {'Count':>6}   Bar")
    print("-" * 60)

    for skill in all_skills:
        n = counts.get(skill, 0)
        b = bar(n, max_count)
        flag = ""
        if n == 0:
            missing_skills.append(skill)
            flag = "  [ZERO]"
        elif n < min_count:
            low_skills.append((skill, n))
            flag = "  [LOW]"
        else:
            ok_skills.append((skill, n))

        print(f"  {skill:<26} {n:>6}   {b}{flag}")

    if unknown:
        print("\n  Unknown skills (not in approved map):")
        for skill, n in unknown.most_common():
            print(f"    {skill}: {n}")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Skills OK (>= {min_count}):     {len(ok_skills)}")
    print(f"  Skills LOW (< {min_count}):     {len(low_skills)}")
    print(f"  Skills with ZERO examples:  {len(missing_skills)}")

    if low_skills or missing_skills:
        print("\n  Recommended: add examples for low/zero skills:")
        print("    python scripts/add_skill_examples.py --skill \"Robotics\" --count 20")
        print("    python scripts/add_skill_examples.py --fill-under --min-count 40")
        print("\n  Or merge rare labels / exclude them from training if too few.")

        print("\n  Lowest counts:")
        combined = [(s, counts.get(s, 0)) for s in all_skills]
        combined.sort(key=lambda x: x[1])
        for skill, n in combined[:10]:
            if n < min_count:
                need = min_count - n
                print(f"    {skill}: {n}  (need ~{need} more for min {min_count})")
    else:
        print("\n  All skills meet the minimum count. Good balance.")

    print("=" * 60)


if __name__ == "__main__":
    main()
