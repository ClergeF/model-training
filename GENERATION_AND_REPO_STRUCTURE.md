# Model Training — Generation Scripts & Repo Structure

**Purpose of this document:** Snapshot of the current repo layout, data-generation pipelines, and how pieces connect. Use this to discuss rearranging the project for better model results.

**Workspace root:** `Model Training/`

**Last snapshot:** 2026-08-26

---

## High-Level System Goal

**Coding in Color Student Intelligence System** — analyze student technical conversations to understand:

1. What technical skills students demonstrate
2. Whether students show evidence of understanding
3. How knowledge develops over time

### Intended multi-model pipeline

```text
Raw Transcript
  → Model 1: Story Chunking (segment full meeting into topic sections)
  → Model 2: Skill Classification (chunk → technical skill)
  → Model 3: Knowledge Scoring (chunk → 0.0–1.0 understanding score)
  → Student Intelligence Aggregation (future)
```

Models 2 and 3 operate on **story chunks / technical statements**, not full transcripts. Model 1 is the gatekeeper that turns messy meetings into structured sections.

---

## Top-Level Directory Tree

```text
Model Training/
├── .venv/                              # shared Python venv (local)
├── .vscode/
│
├── data-generation/                    # PRIMARY synthetic generation hub (newer layout)
│   ├── generate_story_chunking_dataset.py   # Model 1 synthetic data entry point
│   ├── story_chunking/                 # Model 1 generation library
│   ├── skill_classifier/               # Model 2 training + synthetic fill scripts
│   ├── shared_utils/                   # skill → field map
│   ├── chunking_model/                 # README pointer only
│   ├── knowledge_scoring_model/          # README placeholder only
│   ├── requirements.txt
│   ├── .env / .env.example
│   └── README.md
│
├── story-chunking/                     # Model 1 REAL transcript ingestion + training data
│   ├── data/
│   │   ├── transcripts/                # Recall AI pipeline outputs
│   │   │   ├── raw/
│   │   │   ├── cleaned/
│   │   │   ├── readable/
│   │   │   └── manifest.jsonl
│   │   └── training/
│   │       ├── synthetic_story_chunking.jsonl   # synthetic Model 1 labels
│   │       └── sft/
│   │           ├── train.jsonl                  # TRL SFT format
│   │           └── validation.jsonl
│   ├── scripts/
│   │   ├── sync_recall_transcripts.py
│   │   ├── clean_transcripts.py
│   │   ├── transcript_cleaner.py
│   │   └── convert_story_chunking_to_sft_v2.py
│   ├── requirements.txt
│   ├── .env / .env.example
│   └── README.md
│
├── cic-knowledge-score-data-generator/ # LEGACY / standalone Model 2+3 synthetic gen
│   ├── generate_dataset.py             # Model 3 knowledge score generation
│   ├── label_skills.py                 # Model 2 skill labeling from knowledge rows
│   ├── check_dataset.py
│   ├── check_skill_dataset.py
│   ├── adjust_datasets.py
│   ├── skill_map.py
│   ├── cic_knowledge_score_dataset.jsonl
│   ├── cic_skill_field_dataset.jsonl
│   ├── requirements.txt
│   ├── .env / .env.example
│   └── README.md
│
└── training/                           # High-level architecture docs only (no code yet)
    └── README.md
```

---

## Current Dataset Sizes (approx.)

| File | Lines | Notes |
|------|------:|-------|
| `story-chunking/data/training/synthetic_story_chunking.jsonl` | 200 | Model 1 synthetic full-meeting examples |
| `story-chunking/data/training/sft/train.jsonl` | 180 | Converted SFT training split |
| `story-chunking/data/training/sft/validation.jsonl` | 20 | Converted SFT validation split |
| `cic-knowledge-score-data-generator/cic_knowledge_score_dataset.jsonl` | 738 | Model 3: `input_text` + `knowledge_score` |
| `cic-knowledge-score-data-generator/cic_skill_field_dataset.jsonl` | 510 | Model 2 labels: `input_text` + `skill` + `field` |
| `data-generation/skill_classifier/datasets/skill_dataset.jsonl` | 510 | Model 2 training copy (`input_text` + `skill` only) |
| `story-chunking/data/transcripts/manifest.jsonl` | 3 | Real Recall transcripts synced so far |

---

## Pipeline 1 — Story Chunking (Model 1)

### Two parallel data sources

| Source | Location | Status |
|--------|----------|--------|
| **Synthetic** (LLM-generated meetings + labels) | `data-generation/` → writes to `story-chunking/data/training/` | Active, 200 examples |
| **Real** (Recall AI transcripts) | `story-chunking/scripts/` → `story-chunking/data/transcripts/` | Active, 3 transcripts |

Real and synthetic data are **not yet merged** into a single training set. Synthetic data has labels; real transcripts are cleaned but **not yet chunked/labeled**.

---

### 1A. Synthetic generation (`data-generation/`)

**Entry point:** `data-generation/generate_story_chunking_dataset.py`

**What it does:** Generates synthetic full-meeting transcripts paired with correct story-section labels. Does **not** train any model.

**Generation strategy (section-first):**

```text
1. Pick random topics (from TOPIC_POOL) + speaker names
2. LLM "plan" call  → meeting outline (sections, titles, summaries, speakers)
3. Python allocates even contiguous time slots (boundaries always valid)
4. LLM "section" call per section → messy multi-speaker transcript lines
5. Assemble full transcript_text + per-section section_text
6. Python validation (schema, alignment, min sections, min length)
7. Append to JSONL (resume-safe; skips existing IDs unless --force)
```

**Module breakdown (`data-generation/story_chunking/`):**

| File | Role |
|------|------|
| `generator.py` | Orchestration: `build_example()` — plan → slots → per-section text → row |
| `prompts.py` | All LLM prompts, topic pool, style hints, instruction string |
| `speakers.py` | Random speaker name selection per meeting |
| `providers.py` | Unified `Provider` interface: Ollama, Anthropic, OpenRouter |
| `ollama_client.py` | Ollama HTTP helpers, warmup, model checks |
| `ollama_schemas.py` | JSON schemas passed to providers for structured output |
| `validation.py` | Python-only validation (no LLM); rejects bad rows |

**Providers (env in `data-generation/.env`):**

| Provider | Env vars |
|----------|----------|
| `ollama` (default) | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` (default `llama3.1:8b`) |
| `anthropic` | `ANTHROPIC_API_KEY`, model in code |
| `openrouter` | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` |

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--count` | 500 | Examples to generate |
| `--provider` | `GENERATOR_PROVIDER` from `.env` | `ollama`, `anthropic`, `openrouter` |
| `--output` | `story-chunking/data/training/synthetic_story_chunking.jsonl` | Output path (relative to repo root) |
| `--duration` | 60 | Meeting length in minutes |
| `--start-index` | 1 | First `synthetic_{index}` id |
| `--force` | off | Overwrite rows for indices being generated |

**Example commands (from repo root):**

```bash
pip install -r data-generation/requirements.txt
cp data-generation/.env.example data-generation/.env

python data-generation/generate_story_chunking_dataset.py --count 3 --provider openrouter
python data-generation/generate_story_chunking_dataset.py --count 500 --provider ollama
```

**Output schema (one JSON object per line):**

```json
{
  "instruction": "Read the full meeting transcript and organize it into story sections based on topic changes.",
  "input": {
    "transcript_id": "synthetic_00001",
    "duration": "60:00",
    "transcript_text": "[00:00 - 00:08] Name: ..."
  },
  "output": {
    "transcript_id": "synthetic_00001",
    "meeting_summary": "...",
    "story_sections": [
      {
        "start_time": "00:00",
        "end_time": "08:00",
        "section_title": "...",
        "summary": "...",
        "section_text": "...",
        "speakers": ["Name", "Name"],
        "confidence": 0.9
      }
    ]
  },
  "metadata": {
    "synthetic": true,
    "provider": "ollama",
    "topics": ["game programming", "AI"],
    "speakers": ["..."],
    "generated_at": "2026-06-23T..."
  }
}
```

**Topic pool** (from `prompts.py`): 2D art, 3D art, game design, game programming, AI, community engagement, student updates, project demos, technical help, program logistics, mentor feedback, casual check-ins.

---

### 1B. Real transcript ingestion (`story-chunking/`)

**Pipeline:**

```text
Recall AI API (completed transcripts)
  → sync_recall_transcripts.py   (download raw JSON)
  → transcript_cleaner.py        (rule-based cleanup)
  → data/transcripts/cleaned/ + readable/
  → (future) Story Chunking Model labels these
```

**Scripts:**

| Script | Purpose |
|--------|---------|
| `sync_recall_transcripts.py` | List Recall transcripts → download → clean → update manifest |
| `clean_transcripts.py` | Re-clean existing raw files without re-downloading |
| `transcript_cleaner.py` | Core cleaning logic (normalize, gap merge, fragment merge, readable export) |

**Cleaning passes:**

1. Normalize — sort by `start_time`, drop empty lines
2. Same-speaker gap merge (default gap ≤ 1.75s)
3. Tiny-fragment merge (1–2 word fragments into neighbors)
4. Readable `.txt` export

**Cleaned JSON shape:**

```json
{
  "transcript_id": "",
  "recording_id": "",
  "created_at": "",
  "source": "recall_ai",
  "cleanup_version": "1",
  "segments": [
    {
      "start_time": "11.2",
      "end_time": "17.2",
      "speaker": "Christopher Jett",
      "text": "yeah you'll be all right..."
    }
  ]
}
```

**Env (`story-chunking/.env`):** `RECALL_API_KEY`, optional `RECALL_API_BASE_URL`

---

### 1C. SFT conversion (`story-chunking/scripts/`)

**Script:** `convert_story_chunking_to_sft_v2.py`

Converts `synthetic_story_chunking.jsonl` → TRL-compatible SFT format.

- Strips `section_text` and `confidence` from training targets
- Uses a fixed system prompt ("Transcript Architect")
- Shuffles + splits into train/validation (default 90/10)

**SFT row shape:**

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Transcript ID: ...\nMeeting duration: ...\n\nTranscript:\n..."}
  ],
  "completion": [
    {"role": "assistant", "content": "{\"transcript_id\":\"...\",\"meeting_summary\":\"...\",\"story_sections\":[...]}"}
  ]
}
```

**Example:**

```bash
python story-chunking/scripts/convert_story_chunking_to_sft_v2.py \
  story-chunking/data/training/synthetic_story_chunking.jsonl \
  --output-dir story-chunking/data/training/sft
```

---

## Pipeline 2 — Skill Classification (Model 2)

### Data source

Primarily `cic-knowledge-score-data-generator/` (older standalone project), with a copy in `data-generation/skill_classifier/`.

**Training dataset:** `data-generation/skill_classifier/datasets/skill_dataset.jsonl`

```json
{"input_text": "...", "skill": "System Architecture"}
```

**24 skills** mapped to fields via `data-generation/shared_utils/skill_to_field.py` (field is derived after prediction, not trained).

### Generation / labeling scripts

| Location | Script | Purpose |
|----------|--------|---------|
| `cic-knowledge-score-data-generator/` | `label_skills.py` | Label `input_text` rows with skill + field (Ollama) |
| `cic-knowledge-score-data-generator/` | `check_skill_dataset.py` | Audit labels (OpenRouter review) → `changes_needed_skills.json` |
| `cic-knowledge-score-data-generator/` | `adjust_datasets.py skills` | Apply skill fixes |
| `data-generation/skill_classifier/scripts/` | `add_skill_examples.py` | Generate more examples for weak skills (Ollama) |
| `data-generation/skill_classifier/scripts/` | `check_skill_balance.py` | Show per-skill counts |

### Training scripts

| Script | Purpose |
|--------|---------|
| `train_skill_model.py` | TF-IDF + LinearSVC and Logistic Regression; saves best by macro F1 |
| `predict_skill.py` | Run inference with saved model |

**Saved model path (expected):** `data-generation/skill_classifier/saved_models/technical_development_skill_categorization_model.joblib`

**Approach:** Classical ML baseline (TF-IDF → classifier), not LLM fine-tuning.

---

## Pipeline 3 — Knowledge Scoring (Model 3)

### Data source

`cic-knowledge-score-data-generator/cic_knowledge_score_dataset.jsonl`

```json
{"input_text": "...", "knowledge_score": 0.76}
```

### Generation script

**Entry point:** `cic-knowledge-score-data-generator/generate_dataset.py`

**Flow:**

```text
1. Choose underfilled score band (balanced across 5 bands)
2. Pick random skill from SKILL_MAP (hidden topic inspiration, not saved)
3. Ollama generates candidate student statement
4. Ollama reviews candidate for quality + score accuracy
5. Python validates format, length, duplicates
6. Append accepted row (resume-safe, target 1000 rows)
```

**Score bands:**

| Band | Range | Meaning |
|------|-------|---------|
| No evidence | 0.00–0.20 | Filler, vague agreement |
| Basic awareness | 0.21–0.40 | Knows topic exists |
| Partial understanding | 0.41–0.60 | Partial explanation |
| Solid understanding | 0.61–0.80 | Real reasoning |
| Advanced understanding | 0.81–1.00 | Tradeoffs, architecture, debugging |

**Supporting scripts:**

| Script | Purpose |
|--------|---------|
| `check_dataset.py` | Audit scores (OpenRouter) → `changes_needed_knowledge.json` |
| `adjust_datasets.py knowledge` | Apply score fixes |
| `adjust_datasets.py merge` | Copy skills from skill file into knowledge file |
| `skill_map.py` | Skill/field definitions + validation helpers |

**Config:** Ollama `llama3.1:8b` local; checkers use OpenRouter.

**Note:** `data-generation/knowledge_scoring_model/` is a README placeholder pointing back to the cic generator.

---

## How the Pipelines Relate Today

```text
                    ┌─────────────────────────────────────┐
                    │  cic-knowledge-score-data-generator │
                    │  generate_dataset.py (Model 3)      │
                    │  → cic_knowledge_score_dataset     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  label_skills.py (Model 2 labels)   │
                    │  → cic_skill_field_dataset.jsonl  │
                    └──────────────┬──────────────────────┘
                                   │ (manual copy / merge)
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  skill_classifier/datasets/         │
                    │  skill_dataset.jsonl                │
                    │  → train_skill_model.py             │
                    └─────────────────────────────────────┘


  ┌──────────────────────────┐         ┌──────────────────────────┐
  │ data-generation/         │         │ story-chunking/          │
  │ generate_story_chunking  │ writes  │ data/training/           │
  │ _dataset.py (Model 1)    │ ──────► │ synthetic_story_chunking │
  └──────────────────────────┘         │ .jsonl                   │
                                       └──────────┬───────────────┘
                                                  │
                                                  ▼
                                       ┌──────────────────────────┐
                                       │ convert_story_chunking_    │
                                       │ to_sft_v2.py               │
                                       │ → sft/train + validation   │
                                       └──────────────────────────┘

  ┌──────────────────────────┐
  │ story-chunking/          │
  │ sync_recall_transcripts  │  (real data, no labels yet)
  │ → transcripts/cleaned/   │
  └──────────────────────────┘
```

**Key disconnects:**

1. **Model 1 synthetic data** lives under `story-chunking/data/` but is **generated from** `data-generation/`.
2. **Models 2 & 3** live in a separate top-level folder (`cic-knowledge-score-data-generator/`) with partial mirrors under `data-generation/`.
3. **Model 1 output (story sections)** is not yet fed into Models 2 & 3 — those models train on standalone `input_text` snippets, not extracted from chunked meetings.
4. **`training/`** documents the intended 3-model architecture but contains no implementation code.
5. **`data-generation/README.md`** describes a cleaner 3-folder layout (`chunking_model/`, `skill_classifier/`, `knowledge_scoring_model/`) that is only partially realized.

---

## Environment & Dependencies

| Folder | requirements.txt | .env location |
|--------|------------------|---------------|
| `data-generation/` | `requests`, `python-dotenv`, `anthropic` | `data-generation/.env` |
| `story-chunking/` | (Recall sync deps) | `story-chunking/.env` |
| `cic-knowledge-score-data-generator/` | own requirements | `cic-knowledge-score-data-generator/.env` |
| `data-generation/skill_classifier/` | own requirements (sklearn, etc.) | uses Ollama locally |

Three separate `.env` files with different API keys/providers.

---

## Skills & Fields (shared taxonomy)

Defined in both `cic-knowledge-score-data-generator/skill_map.py` and `data-generation/shared_utils/skill_to_field.py`:

Web Development, App Development, Robotics, Data Curation, Data Collection, Game Design, Game Coding, Game Concept, Animation, MCP Server, Prompt Engineering, Workflow Building, Model Training, System Architecture, Mathematics, Knowledge Articulation, Problem Solving, Teamwork, Group Leadership, Sound Design, Game Production, API Creation, App Deployment, 3D Character Modeling

**Fields:** Applications, Robotics, Data, Game Development, Artificial Intelligence, Soft Skills, Production, Digital Art

---

## What Exists vs. What Is Planned

| Component | Status |
|-----------|--------|
| Model 1 synthetic generation | **Working** — 200 examples |
| Model 1 real transcript sync/clean | **Working** — 3 transcripts |
| Model 1 SFT conversion | **Working** — 180/20 split |
| Model 1 actual fine-tuning | **Not started** (data prep only) |
| Model 1 labeling real transcripts | **Not started** |
| Model 2 dataset + TF-IDF training | **Working** — 510 examples |
| Model 2 synthetic fill (`add_skill_examples.py`) | **Working** |
| Model 3 synthetic generation | **Working** — 738/1000 target |
| Model 3 actual model training | **Not started** |
| End-to-end pipeline (chunk → skill → score) | **Not wired** |
| Unified repo layout under `data-generation/` | **Partial** |

---

## Questions to Discuss (reorganization for better model results)

1. Should Model 1 generation move entirely under `story-chunking/` (since outputs live there)?
2. Should `cic-knowledge-score-data-generator/` be merged into `data-generation/knowledge_scoring_model/` and `data-generation/skill_classifier/`?
3. Should Models 2 & 3 train on chunks **extracted from Model 1 outputs** instead of standalone synthetic snippets?
4. Should real Recall transcripts get human or LLM labels to mix with synthetic Model 1 data?
5. Is the section-first synthetic strategy (even time slots, plan-then-fill) producing realistic enough boundaries for fine-tuning?
6. Should `section_text` stay in training targets or continue being stripped in SFT conversion?
7. One shared `.env` and `requirements.txt` vs. per-pipeline configs?
8. Where should fine-tuning scripts live — `story-chunking/`, `data-generation/chunking_model/`, or new `training/` subfolders?

---

## Quick Reference — All Entry Points

```bash
# Model 1 — synthetic data
python data-generation/generate_story_chunking_dataset.py --count 500 --provider openrouter

# Model 1 — real transcript sync
python story-chunking/scripts/sync_recall_transcripts.py

# Model 1 — SFT conversion
python story-chunking/scripts/convert_story_chunking_to_sft_v2.py \
  story-chunking/data/training/synthetic_story_chunking.jsonl \
  --output-dir story-chunking/data/training/sft

# Model 3 — knowledge score synthetic data
cd cic-knowledge-score-data-generator && python generate_dataset.py

# Model 2 — skill labeling from knowledge rows
cd cic-knowledge-score-data-generator && python label_skills.py

# Model 2 — balance check + fill weak skills
cd data-generation/skill_classifier
python scripts/check_skill_balance.py
python scripts/add_skill_examples.py --fill-under --min-count 40

# Model 2 — train classifier
python scripts/train_skill_model.py
```
