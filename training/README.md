# Student Intelligence Model Training

This folder contains the training pipelines, datasets, scripts, and saved models for the Coding in Color Student Intelligence System.

The overall goal of this project is to analyze technical conversations, transcript chunks, Slack discussions, Huddles, project updates, and other student communication in order to understand:

- what technical skills students demonstrate
- whether students show evidence of understanding
- how student knowledge develops over time

The system is designed as a multi-model pipeline.

---

# Current Pipeline

```text
Raw Transcript
→ Chunking Model
→ Skill Classification Model
→ Knowledge Scoring Model
→ Student Intelligence Aggregation
```

---

# Folder Layout

| Folder                     | Purpose                                                                     |
| -------------------------- | --------------------------------------------------------------------------- |
| `chunking_model/`          | Model 1: extracts meaningful story/evidence chunks from raw transcript text |
| `skill_classifier/`        | Model 2: predicts the technical development skill from a story chunk        |
| `knowledge_scoring_model/` | Model 3: predicts a continuous knowledge score from 0.0–1.0                 |
| `shared_utils/`            | Shared helpers, mappings, configs, and utility functions                    |
| `global_datasets/`         | Shared or cross-model datasets                                              |

Each model folder contains:

```text
datasets/
scripts/
saved_models/
outputs/
README.md
```

---

# Model 1 — Chunking Model

## Purpose

The chunking model extracts meaningful technical statements from messy transcript conversations.

Its job is to:

- remove filler
- remove repeated speech
- isolate technical explanations
- isolate debugging discoveries
- isolate implementation reasoning
- isolate useful evidence chunks

## Input

```text
Raw transcript text
```

## Output

```text
Clean evidence/story chunk
```

## Example

### Input

```text
"Yeah so the webhook kept failing and then I realized the payload was actually nested under body.event.text."
```

### Output

```text
"The webhook was failing because the payload was nested under body.event.text."
```

---

# Model 2 — Skill Classification Model

## Purpose

The skill classification model predicts which technical development skill is demonstrated by a story chunk.

This is a supervised NLP text classification model.

## Input

```text
Story chunk / technical statement
```

## Output

```text
Predicted skill
```

Example:

```json
{
  "input_text": "I separated the API route from the database function so debugging would be easier.",
  "skill": "System Architecture"
}
```

The model predicts the skill only.

The field is automatically mapped from the skill using a Python dictionary so that skills and fields cannot mismatch.

Example:

```python
"System Architecture" → "Artificial Intelligence"
```

## Current Approach

The current implementation uses:

```text
TF-IDF → Regression-based classifier
```

The model learns relationships between technical language patterns and skill categories.

Example phrases:

- webhook
- payload
- API route
- deployment
- entity system
- training loss
- prompt structure

are converted into numerical features and used for prediction.

---

# Model 3 — Knowledge Scoring Model

## Purpose

The knowledge scoring model predicts how much evidence a student's statement gives that they understand the topic they are discussing.

This is not an engagement model.

It does not measure:

- participation quantity
- message count
- speaking time

It measures:

- demonstrated understanding
- reasoning
- technical explanation quality
- implementation awareness
- debugging knowledge
- architectural thinking

## Input

```text
Story chunk / technical statement
```

## Output

```text
Knowledge score from 0.0 → 1.0
```

Example:

```json
{
  "input_text": "The Slack bot has to be invited into the private channel before conversations.history can access messages.",
  "knowledge_score": 0.78
}
```

## Score Meaning

| Score Range | Meaning                |
| ----------- | ---------------------- |
| 0.00–0.20   | No clear evidence      |
| 0.21–0.40   | Basic awareness        |
| 0.41–0.60   | Partial understanding  |
| 0.61–0.80   | Solid understanding    |
| 0.81–1.00   | Advanced understanding |

---

# Training Philosophy

The project currently focuses on:

- local experimentation
- supervised NLP pipelines
- synthetic dataset generation
- iterative improvement
- scalable architecture

The system is intentionally modular so each model can improve independently.

---

# Synthetic Data Generation

Some datasets are generated synthetically using local LLMs through Ollama.

This allows early experimentation before large amounts of real transcript data are available.

The synthetic generation pipeline:

- generates candidate rows
- validates outputs
- balances score ranges
- removes duplicates
- saves accepted examples into JSONL datasets

---

# Long-Term Goals

Future versions of the system may include:

- real Slack transcript ingestion
- Slack Huddle processing
- Zoom transcript analysis
- autonomous student intelligence agents
- skill progression tracking
- knowledge trend analysis
- student growth reports
- team capability analysis

---

# Notes

This repository focuses on experimentation and training infrastructure.

Production deployment, APIs, and orchestration systems are handled separately.
