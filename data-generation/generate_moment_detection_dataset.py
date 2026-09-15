#!/usr/bin/env python3
"""
Generate synthetic Moment Detection training data (two-phase, resume-safe).

Phase 1: short story chunks (inputs only)
Phase 2: Moment labels per spec v1.3 (outputs)

Usage:
  python data-generation/generate_moment_detection_dataset.py --smoke-test --provider transformers
  python data-generation/generate_moment_detection_dataset.py --count 500 --phase all --provider transformers
  python data-generation/generate_moment_detection_dataset.py --count 500 --phase inputs --provider transformers
  python data-generation/generate_moment_detection_dataset.py --phase outputs --provider transformers
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

from moment_detection.generator import generate_input_row
from moment_detection.io_utils import (
    append_jsonl,
    count_lines,
    ids_needed_for_target,
    input_id_for,
    load_input_ids,
    load_inputs_by_id,
    load_output_ids,
    missing_output_input_ids,
)
from moment_detection.labeler import label_input_row
from story_chunking.providers import get_provider
from story_chunking.transformers_provider import DEFAULT_TRANSFORMERS_MODEL, TransformersProvider

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_GEN_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = "moment-detection/data/training/synthetic_moment_inputs.jsonl"
DEFAULT_OUTPUTS = "moment-detection/data/training/synthetic_moment_outputs.jsonl"
SMOKE_INPUTS = "moment-detection/data/training/smoke_moment_inputs.jsonl"
SMOKE_OUTPUTS = "moment-detection/data/training/smoke_moment_outputs.jsonl"
MAX_ATTEMPTS = 8


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


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


def run_phase_inputs(
    provider,
    *,
    target_count: int,
    start_index: int,
    inputs_path: Path,
) -> tuple[int, int]:
    existing_ids = load_input_ids(inputs_path)
    target_id_set = {
        input_id_for(i) for i in range(start_index, start_index + target_count)
    }
    needed_ids = ids_needed_for_target(existing_ids, target_count, start_index)
    existing_count = len(existing_ids & target_id_set)

    print("PHASE 1 — INPUT GENERATION")
    print(f"Existing: {existing_count}")
    print(f"Needed: {len(needed_ids)}")
    print()

    generated = 0
    retries = 0

    for position, row_id in enumerate(needed_ids, start=1):
        total_index = int(row_id.split("_")[-1])
        print(f"[{total_index}/{start_index + target_count - 1}]")
        print(f"ID: {row_id}")

        row = None
        stats = {}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt > 1:
                retries += 1
            candidate, attempt_stats = generate_input_row(provider, row_id=row_id)
            stats = attempt_stats
            if candidate is not None:
                row = candidate
                break
            reason = attempt_stats.get("validation_reason", "generation failed")
            print(f"  Attempt {attempt}: rejected — {reason}")

        if row is None:
            print(f"  Giving up on {row_id}")
            continue

        append_jsonl(inputs_path, row)
        existing_ids.add(row_id)
        generated += 1
        print(f"Words: {stats.get('words', 0)}")
        print(f"Generation: {stats.get('generation_s', 0)}s")
        print("Validation: PASS")
        print()

    final = count_lines(inputs_path)
    print("INPUT TARGET REACHED" if final >= target_count else "PHASE 1 INCOMPLETE")
    print(f"{final} / {target_count}")
    print()
    return generated, retries


def run_phase_outputs(
    provider,
    *,
    inputs_path: Path,
    outputs_path: Path,
    target_count: int,
    start_index: int,
) -> tuple[int, int]:
    input_rows = load_inputs_by_id(inputs_path)
    target_ids = {
        input_id_for(i) for i in range(start_index, start_index + target_count)
    }
    input_rows = {k: v for k, v in input_rows.items() if k in target_ids}

    output_ids = load_output_ids(outputs_path)
    missing = missing_output_input_ids(input_rows, output_ids)

    print("PHASE 2 — MOMENT LABELING")
    print(f"Existing Outputs: {len(output_ids & target_ids)}")
    print(f"Remaining: {len(missing)}")
    print()

    if not input_rows:
        print("No inputs available for labeling.")
        return 0, 0

    generated = 0
    retries = 0

    for position, input_id in enumerate(missing, start=1):
        input_row = input_rows[input_id]
        print(f"[{position}/{len(missing)}]")
        print(f"Input: {input_id}")

        row = None
        stats = {}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt > 1:
                retries += 1
            candidate, attempt_stats = label_input_row(provider, input_row)
            stats = attempt_stats
            if candidate is not None:
                row = candidate
                break
            reason = attempt_stats.get("validation_reason", "labeling failed")
            print(f"  Attempt {attempt}: rejected — {reason}")

        if row is None:
            print(f"  Giving up on {input_id}")
            continue

        append_jsonl(outputs_path, row)
        output_ids.add(input_id)
        generated += 1
        print(f"Labeling: {stats.get('labeling_s', 0)}s")
        print(f"status: {row['output'].get('detectionStatus')}")
        print(f"momentCount: {row['output'].get('momentCount')}")
        print("Validation: PASS")
        print()

    labeled = len(load_output_ids(outputs_path) & target_ids)
    print("DATASET COMPLETE" if labeled >= len(input_rows) else "PHASE 2 INCOMPLETE")
    print(f"Outputs: {labeled} / {len(input_rows)}")
    print()
    return generated, retries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Moment Detection synthetic inputs and labels."
    )
    parser.add_argument("--count", type=int, default=500, help="Target total inputs.")
    parser.add_argument(
        "--phase",
        choices=["all", "inputs", "outputs"],
        default="all",
        help="Run input generation, labeling, or both (default: all).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="transformers (recommended), ollama, anthropic, or openrouter",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model for transformers provider.",
    )
    parser.add_argument(
        "--inputs",
        default=None,
        help="Input JSONL path (relative to repo root).",
    )
    parser.add_argument(
        "--outputs",
        default=None,
        help="Output JSONL path (relative to repo root).",
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate 3 inputs (+ labels) to smoke test paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    load_dotenv(DATA_GEN_ROOT / ".env")

    if args.smoke_test:
        args.count = 3
        inputs_path = resolve_path(args.inputs or SMOKE_INPUTS)
        outputs_path = resolve_path(args.outputs or SMOKE_OUTPUTS)
    else:
        inputs_path = resolve_path(args.inputs or DEFAULT_INPUTS)
        outputs_path = resolve_path(args.outputs or DEFAULT_OUTPUTS)

    provider_name = (
        args.provider or os.getenv("GENERATOR_PROVIDER", "transformers")
    ).strip().lower()
    model_name = args.model or os.getenv("TRANSFORMERS_MODEL", DEFAULT_TRANSFORMERS_MODEL)

    print("=" * 60)
    print("  MOMENT DETECTION DATASET GENERATOR")
    print("=" * 60)
    print(f"  Target Inputs: {args.count}")
    print(f"  Phase:         {args.phase}")
    print(f"  Provider:      {provider_name}")
    print(f"  Inputs:        {inputs_path}")
    print(f"  Outputs:       {outputs_path}")
    print(f"  Smoke test:    {args.smoke_test}")
    print("=" * 60)

    provider = get_provider(
        provider_name,
        model=model_name if provider_name == "transformers" else None,
    )
    if isinstance(provider, TransformersProvider):
        print_transformers_summary(provider)

    run_start = time.perf_counter()
    input_generated = input_retries = output_generated = output_retries = 0

    if args.phase in ("all", "inputs"):
        input_generated, input_retries = run_phase_inputs(
            provider,
            target_count=args.count,
            start_index=args.start_index,
            inputs_path=inputs_path,
        )

    if args.phase in ("all", "outputs"):
        output_generated, output_retries = run_phase_outputs(
            provider,
            inputs_path=inputs_path,
            outputs_path=outputs_path,
            target_count=args.count,
            start_index=args.start_index,
        )

    total_inputs = count_lines(inputs_path)
    total_outputs = count_lines(outputs_path)
    elapsed = time.perf_counter() - run_start

    print("=" * 60)
    print(f"  Inputs:  {total_inputs} / {args.count}")
    print(f"  Outputs: {total_outputs} / {min(total_inputs, args.count)}")
    print(f"  Input retries:  {input_retries}")
    print(f"  Output retries: {output_retries}")
    print(f"  Total runtime: {elapsed:.1f}s")
    if isinstance(provider, TransformersProvider):
        try:
            import torch

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"  Peak GPU VRAM: {peak:.2f} GB")
        except Exception:
            pass
    print("=" * 60)

    if args.smoke_test and total_inputs > 0:
        print("\nSmoke test sample (first input):")
        with inputs_path.open("r", encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        print(json.dumps(first, ensure_ascii=False, indent=2)[:1200])
        if total_outputs > 0:
            print("\nSmoke test sample (first output):")
            with outputs_path.open("r", encoding="utf-8") as handle:
                first_out = json.loads(handle.readline())
            print(json.dumps(first_out, ensure_ascii=False, indent=2)[:1200])

    return 0


if __name__ == "__main__":
    sys.exit(main())
