# CIC Knowledge Score Dataset Generator

This project generates synthetic training data for the **Coding in Color Evidence-Based Knowledge Scoring Model**.

This is **Part 1** of the project. It creates the starter dataset before real transcripts are available.

---

## What This Does

The script uses a local Ollama instance running `llama3.1` to generate and review realistic student statements paired with knowledge scores.

- **Ollama generates** one student statement at a time
- **Ollama reviews** each statement for quality and score accuracy
- **Python validates** the format, length, and structure
- **Accepted rows** are saved immediately to a JSONL file
- **The script resumes** automatically if you stop and run it again

Each accepted row looks like this:

```
{"input_text": "The bot needs to be invited into a private channel before it can read messages because private channels block access by default.", "knowledge_score": 0.76}
```

---

## Requirements

- Python 3.8+
- Ollama already installed and running locally
- `llama3.1` already pulled in Ollama

---

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Script

```bash
python generate_dataset.py
```

The script loops until **1,000 rows** are accepted. Press **Ctrl+C** at any time to stop. All rows saved so far will be kept in the output file.

Run the script again at any time to resume from where you left off.

---

## Output File

`cic_knowledge_score_dataset.jsonl`

Each line is a JSON object with exactly two fields:

| Field | Type | Description |
|---|---|---|
| `input_text` | string | A realistic student statement or transcript-like text |
| `knowledge_score` | float | A number from 0.0 to 1.0 representing evidence of understanding |

---

## Score Bands

The dataset is balanced across five ranges:

| Band | Range | Meaning |
|---|---|---|
| No evidence | 0.00–0.20 | Filler, vague agreement, no technical content |
| Basic awareness | 0.21–0.40 | Knows the topic exists, no real explanation |
| Partial understanding | 0.41–0.60 | Explains part of the idea, missing depth |
| Solid understanding | 0.61–0.80 | Explains what, why, or how with real reasoning |
| Advanced understanding | 0.81–1.00 | Tradeoffs, architecture, debugging, cause and effect |

---

## Scripts

| Command | Purpose |
|---------|---------|
| `python generate_dataset.py` | Generate Model 3 knowledge score data (Ollama) |
| `python label_skills.py` | Label Model 2 skill + field from existing `input_text` (Ollama) |
| `python check_dataset.py` | Audit knowledge dataset; writes `changes_needed_knowledge.json` |
| `python check_skill_dataset.py` | Audit skill dataset; writes `changes_needed_skills.json` |
| `python adjust_datasets.py knowledge` | Apply score fixes from knowledge changes file |
| `python adjust_datasets.py skills` | Apply skill/field fixes and sync into knowledge file |
| `python adjust_datasets.py all` | Apply both change files |
| `python adjust_datasets.py merge` | Copy skills from skill file into knowledge file only |

Use `--dry-run` on adjust to preview without writing.

## Notes

- Generation and labeling use local Ollama (`llama3.1:8b`).
- Checkers use OpenRouter (set `OPENROUTER_API_KEY` in `.env`).
- If labeling returns a **field** name (e.g. "Game Development"), `label_skills.py` picks the best skill from that field automatically.
