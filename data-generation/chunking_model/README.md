# Chunking Model — Data Generation

Synthetic training data for the Story Chunking Model. The generator lives one
level up at [`../generate_story_chunking_dataset.py`](../generate_story_chunking_dataset.py).

This step only produces training data. It does not train or run the model.

## What it produces

Each JSONL row pairs a full messy meeting transcript (input) with the correct
story sections (output), so the model can later learn where topics change.

See the [data-generation README](../README.md) for full usage, flags, and the
JSONL schema.

## Quick start

```bash
# from the Model Training folder
pip install -r data-generation/requirements.txt
cp data-generation/.env.example data-generation/.env   # set provider

# generate 3 test examples first (OpenRouter recommended on slow local hardware)
python data-generation/generate_story_chunking_dataset.py --count 3 --provider openrouter

# then a full run
python data-generation/generate_story_chunking_dataset.py --count 500 --provider openrouter
```

Output is written to
`story-chunking/data/training/synthetic_story_chunking.jsonl`.
