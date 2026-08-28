"""
Apply fixes from checker output files.

Usage:
  python adjust_datasets.py knowledge   # apply changes_needed_knowledge.json
  python adjust_datasets.py skills      # apply changes_needed_skills.json
  python adjust_datasets.py all         # apply both
  python adjust_datasets.py merge       # sync skills into knowledge file only

  python adjust_datasets.py knowledge --dry-run   # preview without writing
"""

import argparse
import json
import os
import sys

from skill_map import SKILL_MAP, validate_skill_field_pair

KNOWLEDGE_FILE = "cic_knowledge_score_dataset.jsonl"
SKILL_FILE = "cic_skill_field_dataset.jsonl"
CHANGES_KNOWLEDGE = "changes_needed_knowledge.json"
CHANGES_SKILLS = "changes_needed_skills.json"


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_changes(path):
    if not os.path.exists(path):
        print(f"  No file: {path}")
        print("  Run the matching check script first.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_knowledge_changes(dry_run=False):
    changes = load_changes(CHANGES_KNOWLEDGE)
    if not changes:
        return 0

    lookup = {c["input_text"]: c for c in changes}
    rows = load_jsonl(KNOWLEDGE_FILE)
    updated = 0

    for row in rows:
        text = row.get("input_text", "")
        if text not in lookup:
            continue
        change = lookup[text]
        new_score = change.get("suggested_knowledge_score")
        if new_score is None:
            continue
        new_score = round(float(new_score), 4)
        old_score = row.get("knowledge_score")
        if old_score == new_score:
            continue
        if not dry_run:
            row["knowledge_score"] = new_score
        updated += 1
        print(f"  score {old_score} → {new_score}  |  {text[:60]}...")

    if not dry_run and updated:
        write_jsonl(KNOWLEDGE_FILE, rows)
    print(f"\n  Knowledge scores updated: {updated}" + (" (dry run)" if dry_run else ""))
    return updated


def apply_skill_changes(dry_run=False):
    changes = load_changes(CHANGES_SKILLS)
    if not changes:
        return 0

    lookup = {c["input_text"]: c for c in changes}
    rows = load_jsonl(SKILL_FILE)
    updated = 0

    for row in rows:
        text = row.get("input_text", "")
        if text not in lookup:
            continue
        change = lookup[text]
        new_skill = change.get("suggested_skill", "").strip()
        new_field = change.get("suggested_field", "").strip()

        if not validate_skill_field_pair(new_skill, new_field):
            # Force field from map
            if new_skill in SKILL_MAP:
                new_field = SKILL_MAP[new_skill]
            else:
                print(f"  Skip invalid suggestion: {new_skill!r}")
                continue

        if row.get("skill") == new_skill and row.get("field") == new_field:
            continue

        if not dry_run:
            row["skill"] = new_skill
            row["field"] = new_field
        updated += 1
        print(f"  {row.get('skill')} → {new_skill}  |  {text[:55]}...")

    if not dry_run and updated:
        write_jsonl(SKILL_FILE, rows)
        from label_skills import merge_skills_into_knowledge_file
        merged = merge_skills_into_knowledge_file()
        print(f"  Synced {merged} row(s) in {KNOWLEDGE_FILE}")

    print(f"\n  Skill labels updated: {updated}" + (" (dry run)" if dry_run else ""))
    return updated


def run_merge_only():
    from label_skills import merge_skills_into_knowledge_file
    n = merge_skills_into_knowledge_file()
    print(f"  Merged skills into {KNOWLEDGE_FILE}: {n} row(s) touched.")


def main():
    parser = argparse.ArgumentParser(
        description="Apply dataset fixes from check_dataset / check_skill_dataset output."
    )
    parser.add_argument(
        "command",
        choices=["knowledge", "skills", "all", "merge"],
        help="Which dataset changes to apply",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  CIC Dataset Adjust")
    print("=" * 55)
    if args.dry_run:
        print("  Mode: DRY RUN (no files written)\n")

    if args.command == "knowledge":
        apply_knowledge_changes(dry_run=args.dry_run)
    elif args.command == "skills":
        apply_skill_changes(dry_run=args.dry_run)
    elif args.command == "all":
        apply_knowledge_changes(dry_run=args.dry_run)
        print()
        apply_skill_changes(dry_run=args.dry_run)
    elif args.command == "merge":
        if args.dry_run:
            print("  merge does not support --dry-run")
            sys.exit(1)
        run_merge_only()

    print("\n" + "=" * 55)
    print("  Done.")
    print("=" * 55)


if __name__ == "__main__":
    main()
