# Model Training

Layout for three pipelines plus shared helpers.

| Folder | Purpose |
|--------|---------|
| `chunking_model/` | Story chunking (TBD) |
| `skill_classifier/` | Model 2: text → technical skill |
| `knowledge_scoring_model/` | Evidence-based knowledge score (TBD) |
| `shared_utils/` | Shared maps and helpers (e.g. skill → field) |
| `global_datasets/` | Cross-model datasets (optional) |

Each model folder has `datasets/`, `scripts/`, `saved_models/`, `outputs/`, and a `README.md`.

---

# Story Chunking Dataset Generator

`generate_story_chunking_dataset.py` produces synthetic full-meeting transcripts
paired with the correct story-section labels. This is training data for the
future Story Chunking Model — it does not train or run any model.

Generation is section-first: the script asks the provider for a meeting outline,
assigns even time slots so section boundaries are always valid, then generates
messy multi-speaker transcript lines per section and assembles the full meeting.

## Setup

From the `Model Training` folder:

```bash
pip install -r data-generation/requirements.txt
cp data-generation/.env.example data-generation/.env
```

Set the provider in `data-generation/.env`:

```env
GENERATOR_PROVIDER=openrouter     # or ollama, anthropic
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=anthropic/claude-sonnet-4.6
```

Or for local / direct Anthropic:

```env
GENERATOR_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
# ANTHROPIC_API_KEY=...          # required only for the anthropic provider
```

The `--provider` flag overrides `GENERATOR_PROVIDER`.

## Run

Generate 3 test examples before a large run:

```bash
python data-generation/generate_story_chunking_dataset.py --count 3 --provider openrouter
```

Full run:

```bash
python data-generation/generate_story_chunking_dataset.py --count 500 --provider openrouter
```

Other providers:

```bash
python data-generation/generate_story_chunking_dataset.py --count 500 --provider ollama
python data-generation/generate_story_chunking_dataset.py --count 500 --provider anthropic
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--count` | 500 | Number of examples to generate |
| `--provider` | env value | `ollama`, `anthropic`, or `openrouter` |
| `--output` | `story-chunking/data/training/synthetic_story_chunking.jsonl` | Output path (relative to `Model Training`) |
| `--duration` | 60 | Meeting length in minutes |
| `--start-index` | 1 | First `synthetic_{index}` id |
| `--force` | off | Regenerate and overwrite rows for the indices being generated |

The script is safe to stop and rerun: existing `transcript_id`s are skipped
unless `--force` is passed, and each accepted row is appended immediately.

## Output schema

One JSON object per line:

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
    "generated_at": "2026-06-23T..."
  }
}
```

Topics are drawn from the real meeting subjects (2D/3D art, game design, game
programming, AI, community engagement, student updates, project demos, technical
help, program logistics, mentor feedback, casual check-ins). Generated meetings
imitate the style of the 29 real meetings without copying them.
