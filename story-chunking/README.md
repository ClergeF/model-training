# Story Chunking Model — Data Prep

Training and evaluation data for the Story Chunking Model (Model 1). This folder holds transcript ingestion, cleaning, and later chunking — not model training yet.

## Pipeline

```text
Recall AI (completed transcripts)
→ sync_recall_transcripts.py (download + save raw)
→ transcript_cleaner.py (rule-based cleanup)
→ data/transcripts/cleaned/ + readable/
→ (next) Story Chunking Model
```

## Recall AI transcript sync

Pulls completed transcript artifacts from Recall AI, downloads each transcript, and runs rule-based cleaning.

Based on the [Recall List Transcript API](https://docs.recall.ai/reference/transcript_list):

```bash
curl --request GET \
     --url 'https://us-east-1.recall.ai/api/v1/transcript/?status_code=done' \
     --header 'Authorization: RECALL_API_KEY' \
     --header 'accept: application/json'
```

### Setup

1. Install dependencies (from this folder):

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set your Recall API key:

   ```bash
   cp .env.example .env
   ```

   In `.env`:

   ```env
   RECALL_API_KEY=your_recall_api_key_here
   ```

   Optional region override (host only, no `/api/v1` path):

   ```env
   RECALL_API_BASE_URL=https://us-west-2.recall.ai
   ```

   Do not commit `.env` — it is listed in `.gitignore`.

### Run sync (download from Recall)

From the `story-chunking` folder:

```bash
python scripts/sync_recall_transcripts.py
```

Test with a small batch first:

```bash
python scripts/sync_recall_transcripts.py --limit 3
```

Re-download and re-clean everything:

```bash
python scripts/sync_recall_transcripts.py --force
```

The script is safe to rerun: already-processed transcripts are skipped unless `--force` is passed.

### Run cleaning only (no API download)

Re-clean existing raw files after rule changes, without touching raw data:

```bash
python scripts/clean_transcripts.py
python scripts/clean_transcripts.py --force
python scripts/clean_transcripts.py --limit 3
python scripts/clean_transcripts.py --max-gap 1.75
```

Or via the sync script:

```bash
python scripts/sync_recall_transcripts.py --re-clean-only --force
```

## Rule-based cleanup

Cleaning logic lives in `scripts/transcript_cleaner.py`. It runs these passes on Recall raw data:

1. **Normalize** — sort by numeric `start_time`, drop empty lines, trim whitespace
2. **Same-speaker gap merge** — merge consecutive same-speaker segments when the gap is ≤ 1.75s (configurable via `--max-gap`)
3. **Tiny-fragment merge** — merge 1–2 word fragments into the nearest neighbor when continuation heuristics score high enough (dangling tokens, lowercase starts, connector words, small time gaps)
4. **Readable export** — write a human-readable `.txt` transcript

Cross-speaker fragment merges are flagged in cleaned JSON:

```json
{
  "speaker_confidence": "uncertain",
  "cleanup_note": "merged tiny fragment across speaker labels"
}
```

Readable line format:

```text
[00:11 - 00:17] Christopher Jett: yeah you'll be all right just take some vitamin c i've been under the weather
```

Text stays as transcribed (lowercase, no LLM punctuation fix).

### Output locations

| Path | Contents |
|------|----------|
| `data/transcripts/raw/` | Raw JSON from Recall (never modified by cleaning) |
| `data/transcripts/cleaned/` | Rule-cleaned JSON, training-ready for Story Chunking Model |
| `data/transcripts/readable/` | Human-readable `.txt` transcripts |
| `data/transcripts/manifest.jsonl` | One JSON line per transcript (ids, paths, timestamps) |

Raw filenames: `{created_date}_{recording_id}_{transcript_id}.json`

Cleaned JSON shape:

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
      "text": "yeah you'll be all right just take some vitamin c i've been under the weather"
    }
  ]
}
```

### Next step

After transcripts are synced and cleaned, the next step is the **Story Chunking Model** — extracting meaningful technical story/evidence chunks from the cleaned segment data. Model training is not part of this step.

## Folder layout

```text
story-chunking/
├── data/transcripts/
│   ├── raw/
│   ├── cleaned/
│   ├── readable/
│   └── manifest.jsonl
├── scripts/
│   ├── sync_recall_transcripts.py
│   ├── clean_transcripts.py
│   └── transcript_cleaner.py
├── .env.example
├── requirements.txt
└── README.md
```
