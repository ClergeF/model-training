#!/usr/bin/env python3
"""
Generate synthetic full-meeting transcripts paired with story-section labels
for the Story Chunking Model.

V2 (default): transcript-first generation with GPU Transformers provider support.
V1 (--version v1): legacy section-first pipeline.

Usage:
  python data-generation/generate_story_chunking_dataset.py --smoke-test --provider transformers
  python data-generation/generate_story_chunking_dataset.py --count 20 --provider transformers
  python data-generation/generate_story_chunking_dataset.py --version v1 --count 3 --provider ollama
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from story_chunking.generator import build_example as build_example_v1
from story_chunking.generator_v2 import MAX_ATTEMPTS_V2, build_example_v2
from story_chunking.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    check_server,
    ensure_model_available,
    resolve_base_url,
    warmup_model,
)
from story_chunking.providers import get_provider
from story_chunking.scenarios_v2 import build_story_count_schedule
from story_chunking.transformers_provider import DEFAULT_TRANSFORMERS_MODEL, TransformersProvider
from story_chunking.validation import validate_example as validate_example_v1

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_GEN_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_V1 = "story-chunking/data/training/synthetic_story_chunking.jsonl"
DEFAULT_OUTPUT_V2 = "story-chunking/data/training/v2/synthetic_story_chunking_v2.jsonl"
MAX_ATTEMPTS_V1 = 3


def transcript_id_for(version: str, index: int) -> str:
    if version == "v2":
        return f"synthetic_v2_{index:05d}"
    return f"synthetic_{index:05d}"


def load_existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            transcript_id = (row.get("input") or {}).get("transcript_id")
            if transcript_id:
                ids.add(transcript_id)
    return ids


def load_section_histogram(output_path: Path) -> dict[str, int]:
    histogram = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    if not output_path.exists():
        return histogram

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sections = (row.get("output") or {}).get("story_sections") or []
            count = len(sections)
            if count >= 5:
                histogram["5+"] += 1
            elif count >= 1:
                histogram[str(count)] += 1
    return histogram


def drop_ids(output_path: Path, ids_to_drop: set[str]) -> None:
    if not output_path.exists() or not ids_to_drop:
        return

    kept: list[str] = []
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            transcript_id = (row.get("input") or {}).get("transcript_id")
            if transcript_id in ids_to_drop:
                continue
            kept.append(stripped)

    with output_path.open("w", encoding="utf-8") as handle:
        for line in kept:
            handle.write(line + "\n")


def append_row(output_path: Path, row: dict) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_output_path(raw_output: str) -> Path:
    path = Path(raw_output)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Story Chunking training data."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Generator version (default: v2).",
    )
    parser.add_argument("--count", type=int, default=20, help="Examples to generate.")
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="ollama, anthropic, openrouter, or transformers.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for transformers provider.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (relative to Model Training folder).",
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Meeting length in minutes (V1 only)."
    )
    parser.add_argument(
        "--start-index", type=int, default=1, help="First synthetic id index."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate and overwrite rows for the indices being generated.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate exactly one example and print provider/GPU summary.",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip the reviewer pass (V2 only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for V2 story-count schedule.",
    )
    return parser.parse_args()


def setup_ollama() -> int | None:
    base_url = resolve_base_url()
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    print(f"  Ollama base: {base_url}")
    print(f"  Ollama model: {model}")
    try:
        check_server(base_url)
        ensure_model_available(model, base_url)
        print("  Ollama: server reachable, model installed")
        print("  Ollama: warming up model for bulk generation...")
        warmup_model(model, base_url)
        print("  Ollama: model ready")
    except OllamaConnectionError as exc:
        print(f"\nError: {exc}")
        return 1
    except OllamaModelNotFoundError as exc:
        print(f"\nError: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\nError: Ollama warmup failed: {exc}")
        return 1
    return None


def print_transformers_summary(provider: TransformersProvider) -> None:
    import torch

    print(f"  Provider: transformers")
    print(f"  Model: {provider.model_name}")
    print(f"  CUDA: {'available' if torch.cuda.is_available() else 'unavailable'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        print(f"  dtype: {dtype}")
        print(f"  VRAM: {provider.vram_summary()}")


def run_v1(args: argparse.Namespace, provider_name: str, output_path: Path) -> int:
    target_ids = [
        transcript_id_for("v1", args.start_index + offset) for offset in range(args.count)
    ]

    provider = get_provider(provider_name)

    if provider_name == "ollama":
        err = setup_ollama()
        if err:
            return err

    if args.force:
        drop_ids(output_path, set(target_ids))

    existing_ids = load_existing_ids(output_path)
    generated = skipped = failed = 0

    for position, transcript_id in enumerate(target_ids, start=1):
        if transcript_id in existing_ids:
            print(f"[{position}/{args.count}] {transcript_id}: already exists, skipping")
            skipped += 1
            continue

        print(f"[{position}/{args.count}] Generating {transcript_id}...")

        row = None
        for attempt in range(1, MAX_ATTEMPTS_V1 + 1):
            candidate = build_example_v1(
                provider,
                transcript_id=transcript_id,
                duration_minutes=args.duration,
            )
            if candidate is None:
                print(f"  Attempt {attempt}: generation failed (empty response)")
                continue

            is_valid, reason = validate_example_v1(candidate, args.duration)
            if is_valid:
                row = candidate
                break
            print(f"  Attempt {attempt}: rejected — {reason}")

        if row is None:
            print(f"  Giving up on {transcript_id} after {MAX_ATTEMPTS_V1} attempts")
            failed += 1
            continue

        append_row(output_path, row)
        existing_ids.add(transcript_id)
        generated += 1
        section_count = len(row["output"]["story_sections"])
        chars = len(row["input"]["transcript_text"])
        print(f"  Accepted: {section_count} sections, {chars} transcript chars")

    print("=" * 60)
    print(f"  Generated: {generated}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Output:    {output_path}")
    print("=" * 60)
    return 0


def run_v2(args: argparse.Namespace, provider_name: str, output_path: Path) -> int:
    count = 1 if args.smoke_test else args.count
    target_ids = [
        transcript_id_for("v2", args.start_index + offset) for offset in range(count)
    ]
    story_schedule = build_story_count_schedule(count, seed=args.seed)

    model_name = args.model or os.getenv("TRANSFORMERS_MODEL", DEFAULT_TRANSFORMERS_MODEL)
    provider = get_provider(provider_name, model=model_name if provider_name == "transformers" else None)

    if provider_name == "ollama":
        err = setup_ollama()
        if err:
            return err

    if isinstance(provider, TransformersProvider):
        print_transformers_summary(provider)

    if args.force:
        drop_ids(output_path, set(target_ids))

    existing_ids = load_existing_ids(output_path)
    generated = skipped = failed = retried = 0
    generation_times: list[float] = []
    run_start = time.perf_counter()

    for position, transcript_id in enumerate(target_ids, start=1):
        if transcript_id in existing_ids and not args.smoke_test:
            print(f"Generating example {position}/{count}...")
            print(f"  {transcript_id}: already exists, skipping")
            skipped += 1
            continue

        target_stories = story_schedule[position - 1] if position <= len(story_schedule) else 1
        print(f"Generating example {position}/{count}...")

        row = None
        stats: dict = {}
        example_start = time.perf_counter()

        for attempt in range(1, MAX_ATTEMPTS_V2 + 1):
            if attempt > 1:
                retried += 1
            candidate, attempt_stats = build_example_v2(
                provider,
                transcript_id=transcript_id,
                target_story_count=target_stories,
                enable_review=not args.no_review,
            )
            stats = attempt_stats
            if candidate is None:
                reason = attempt_stats.get("validation_reason", "generation failed")
                print(f"  Attempt {attempt}: rejected — {reason}")
                continue

            row = candidate
            break

        elapsed = time.perf_counter() - example_start
        generation_times.append(elapsed)

        if row is None:
            print(f"  Giving up on {transcript_id} after {MAX_ATTEMPTS_V2} attempts")
            failed += 1
            continue

        if args.smoke_test and transcript_id in existing_ids:
            drop_ids(output_path, {transcript_id})

        append_row(output_path, row)
        existing_ids.add(transcript_id)
        generated += 1

        print(f"  transcript generation: {stats.get('transcript_s', 0)}s")
        print(f"  labeling: {stats.get('label_s', 0)}s")
        if not args.no_review:
            print(f"  review: {stats.get('review_s', 0)}s")
        print(f"  sections: {stats.get('sections', 0)}")
        print(f"  validation: {stats.get('validation', 'PASS')}")
        print(f"  saved: {transcript_id}")

    total_time = time.perf_counter() - run_start
    histogram = load_section_histogram(output_path)

    print("=" * 60)
    print(f"  Generated: {generated}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Rejected/retried: {retried}")
    print(f"  1-story: {histogram['1']}")
    print(f"  2-story: {histogram['2']}")
    print(f"  3-story: {histogram['3']}")
    print(f"  4-story: {histogram['4']}")
    print(f"  5+-story: {histogram['5+']}")
    if generation_times:
        avg_time = sum(generation_times) / len(generation_times)
        print(f"  Average generation time: {avg_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Output: {output_path}")

    if isinstance(provider, TransformersProvider):
        try:
            import torch

            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"  Peak GPU VRAM: {peak_gb:.2f} GB")
        except Exception:
            pass

    print("=" * 60)
    return 0 if failed == 0 or generated > 0 else 1


def main() -> int:
    args = parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    load_dotenv(DATA_GEN_ROOT / ".env")

    version = args.version.strip().lower()
    default_output = DEFAULT_OUTPUT_V2 if version == "v2" else DEFAULT_OUTPUT_V1
    output_path = resolve_output_path(args.output or default_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    provider_name = (
        args.provider
        or os.getenv("GENERATOR_PROVIDER", "transformers" if version == "v2" else "ollama")
    ).strip().lower()

    print("=" * 60)
    print("  Story Chunking - Synthetic Dataset Generator")
    print("=" * 60)
    print(f"  Version:     {version}")
    print(f"  Provider:    {provider_name}")
    print(f"  Count:       {1 if args.smoke_test else args.count}")
    print(f"  Output:      {output_path}")
    print(f"  Smoke test:  {args.smoke_test}")
    if version == "v1":
        print(f"  Duration:    {args.duration} min")
    if version == "v2":
        print(f"  Review:      {not args.no_review}")
        if provider_name == "transformers":
            model = args.model or os.getenv("TRANSFORMERS_MODEL", DEFAULT_TRANSFORMERS_MODEL)
            print(f"  Model:       {model}")
    print(f"  Force:       {args.force}")
    print("=" * 60)

    if version == "v1":
        return run_v1(args, provider_name, output_path)
    return run_v2(args, provider_name, output_path)


if __name__ == "__main__":
    sys.exit(main())
