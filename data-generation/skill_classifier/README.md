# Technical Development Skill Categorization Model (Model 2)

Supervised text classification: **story chunk → predicted skill**.

## Dataset

`datasets/skill_dataset.jsonl` — one JSON object per line:

```json
{"input_text": "...", "skill": "System Architecture"}
```

## Setup

```bash
pip install -r requirements.txt
```

## Check class balance

See how many examples each skill has (flags low/zero counts):

```bash
cd data-generation/skill_classifier
python scripts/check_skill_balance.py
```

Optional:

```bash
python scripts/check_skill_balance.py --min-count 40
python scripts/check_skill_balance.py --dataset datasets/skill_dataset.jsonl
```

## Add more examples for weak skills

Requires Ollama running with `llama3.1:8b`.

One skill:

```bash
python scripts/add_skill_examples.py --skill "Robotics" --count 20
```

Bring every skill up to at least 40 examples:

```bash
python scripts/add_skill_examples.py --fill-under --min-count 40
```

Preview without saving:

```bash
python scripts/add_skill_examples.py --skill "Sound Design" --count 5 --dry-run
```

Then re-run the balance check and train again.

## Train

From this folder:

```bash
python scripts/train_skill_model.py
```

Trains TF-IDF + LinearSVC and TF-IDF + Logistic Regression, picks the best by **macro F1**, and saves:

**Technical Development Skill Categorization model** → `saved_models/technical_development_skill_categorization_model.joblib`

## Predict

```bash
python scripts/predict_skill.py
```

## Notes

I trained Model 2 (Technical Development Skill Categorization model) as a supervised text classification model. The input is a story chunk, and the output is the predicted technical skill. I tested TF-IDF + LinearSVC and TF-IDF + Logistic Regression as baseline models.

## Source data

Regenerate labels in `cic-knowledge-score-data-generator/`, then copy into `datasets/skill_dataset.jsonl` keeping only `input_text` and `skill` (drop `field` — this model trains on skill only).
