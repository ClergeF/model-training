# Moment Detection Model — Data & Generation

Moment Detection runs **after Story Chunking**. It receives **one short story chunk at a time** (not a full meeting) and identifies zero or more **Moments** — bounded person-to-person check-ins.

```text
Raw Transcript
  → Story Chunking (Model 1)
  → Story chunk (paragraph-sized segment)
  → Moment Detection
  → Moment label(s)
  → downstream intelligence models (future)
```

Canonical labeling rules: [`docs/MOMENT_DETECTION_SPEC_v1.3.txt`](docs/MOMENT_DETECTION_SPEC_v1.3.txt) (from *Moment_Detection_Model_Logic.docx* v1.3).

---

## Dataset layout

| File | Contents |
|------|----------|
| `data/training/synthetic_moment_inputs.jsonl` | Inputs only (no labels) |
| `data/training/synthetic_moment_outputs.jsonl` | Labels keyed by `input_id` |
| `data/training/smoke_moment_*.jsonl` | Smoke-test artifacts |

**Input row (provisional):**

```json
{
  "id": "moment_input_00001",
  "input": {
    "original_text": "...",
    "source": "synthetic",
    "categories": []
  }
}
```

**Output row:**

```json
{
  "input_id": "moment_input_00001",
  "output": {
    "originalText": "...",
    "source": "synthetic",
    "detectionStatus": "CONFIRMED",
    "momentCount": 1,
    "moments": [ { "contributor": "...", "beneficiary": "...", "summary": "...", "momentType": "MENTORSHIP", "category": "Education & Innovation" } ],
    "missingInformation": []
  }
}
```

Inputs and outputs are **separate files** so you can relabel without regenerating conversations.

---

## Two-phase generator

Entry point: [`../data-generation/generate_moment_detection_dataset.py`](../data-generation/generate_moment_detection_dataset.py)

| Phase | Meaning |
|-------|---------|
| **Phase 1** | Reach `--count` valid **inputs** (resume-safe; only fills missing IDs) |
| **Phase 2** | Write one **output** per input (resume-safe; only missing labels) |

`--count 500` means **500 valid inputs total**, not “500 new lines every run”.

### CLI

| Flag | Default | Description |
|------|---------|-------------|
| `--count` | 500 | Target number of inputs |
| `--phase` | `all` | `all`, `inputs`, or `outputs` |
| `--provider` | `transformers` | Uses shared Story Chunking GPU stack |
| `--model` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Transformers model |
| `--inputs` / `--outputs` | paths above | Override JSONL locations |
| `--start-index` | 1 | First `moment_input_{index}` id |
| `--smoke-test` | off | Generate **3** examples to smoke paths |

### Commands (from repo root)

**Smoke test (run first on Colab A100):**

```bash
pip install -r data-generation/requirements.txt
huggingface-cli login   # Llama 3.1 license + token

python data-generation/generate_moment_detection_dataset.py \
  --smoke-test \
  --provider transformers \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Inputs only:**

```bash
python data-generation/generate_moment_detection_dataset.py \
  --count 500 \
  --phase inputs \
  --provider transformers
```

**Labels only (after inputs exist):**

```bash
python data-generation/generate_moment_detection_dataset.py \
  --count 500 \
  --phase outputs \
  --provider transformers
```

**Full pipeline:**

```bash
python data-generation/generate_moment_detection_dataset.py \
  --count 500 \
  --phase all \
  --provider transformers
```

---

## GPU (Colab A100)

Uses the **same Transformers provider** as Story Chunking V2:

- Requires **CUDA** (no silent CPU fallback)
- Loads **Llama 3.1 8B once** per run
- BF16 on A100 when supported
- Startup logs show GPU name and VRAM

Confirm with `nvidia-smi` while the script runs.

---

## Resume behavior

- **Phase 1:** If 327/500 inputs exist, generates only `moment_input_00328` … `moment_input_00500`.
- **Phase 2:** If 183/500 outputs exist, labels only the remaining 317 inputs.
- Re-running `--phase all` skips completed work in each phase.

---

## What is not in this folder

- **Model training** — not implemented yet (data generation only)
- **SFT conversion** — combine input/output JSONL later when training is ready

Story Chunking, Skill Classification, and Knowledge Scoring pipelines are unchanged.
