# Coding in Color Knowledge Evidence Scoring Dataset Generator

## Purpose

This project creates synthetic training data for a future **Evidence-Based Knowledge Scoring Model**.

The model is meant to help Coding in Color understand whether students genuinely understand what they are working on — based on what they say during transcripts, Slack Huddles, project demos, standups, and technical conversations.

The model is not measuring how much a student talks.
The model is not measuring participation or engagement.
The model is measuring whether the student's words provide **evidence of real understanding**.

---

## Future Model

**Input:**
A student story, statement, or transcript chunk — a few words up to a few sentences.

**Output:**
A single number from 0.0 to 1.0.

That number represents how strong the evidence is that the student understands the concept they are discussing.

**Example:**

```json
{
  "input_text": "The bot needs to be invited into a private Slack channel before it can read messages because private channels block access unless the bot is a member.",
  "knowledge_score": 0.76
}
```

---

## Why Synthetic Data

Right now, Coding in Color does not have enough real Slack Huddle transcripts, meeting notes, or student recordings to train a scoring model.

So this project uses Ollama — a local language model tool — to generate a realistic **starter dataset**.

This lets the team:
- Build and test the data pipeline before real data is available
- Define and refine the scoring rubric
- Check whether the scoring scale is learnable
- Prepare the model training environment early

Once real transcript data is collected, it can replace or supplement this synthetic data.

---

## Why Ollama

Ollama is a free, local tool that runs large language models on your own machine.

The Python script talks to Ollama through a simple HTTP API:
```
POST http://localhost:11434/api/generate
```

This approach:
- Costs nothing (no external API fees)
- Keeps data private
- Works offline
- Is fast enough for generating a 1,000-row dataset

The model used is `llama3.1`, which must be pulled before running the script.

---

## What Cursor / Copilot Is Doing

Cursor or Copilot built the Python project files — the generator script, this context file, the README, and the requirements file.

Cursor is **not** generating the dataset directly.

The layers are:

```
Cursor / Copilot
  → builds the Python project files

generate_dataset.py
  → controls the active generation loop

Ollama llama3.1 (Generator role)
  → generates one synthetic training row at a time

Ollama llama3.1 (Reviewer role)
  → reviews that row for quality and score accuracy

generate_dataset.py
  → validates, accepts or rejects, saves, and loops
```

---

## Active Loop Design

The script does not ask for 1,000 examples in one big request. That would produce poor quality and unbalanced data.

Instead, the script runs a tight loop:

1. Look at the current dataset to find which score band needs more examples.
2. Pick a random skill or topic from the skill map for inspiration.
3. Ask Ollama to generate **one** candidate row targeting that band and skill.
4. Run Python validation on the row (format, length, duplicates, etc.).
5. Ask Ollama to review that candidate row for quality and score accuracy.
6. Accept or reject the row based on the reviewer's response.
7. If accepted, save the row immediately to the JSONL file.
8. Repeat until the target number of rows is reached.
9. If the user presses Ctrl+C, stop gracefully and keep all saved rows.

This approach:
- Keeps each row independent
- Allows quality control at every step
- Balances the dataset across score bands
- Saves rows as they are accepted so no data is lost if the script stops

---

## Ollama as Generator and Reviewer

Ollama plays **two separate roles** in this pipeline.

### Role 1: Generator

Ollama receives:
- A target score band (which range of scores to aim for)
- A description of what that band means
- A skill or topic for thematic inspiration

Ollama returns a JSON object:
```json
{"input_text": "student statement here", "knowledge_score": 0.72}
```

### Role 2: Reviewer

Ollama receives:
- The candidate row that was just generated
- The target score band
- The current band counts (so the reviewer has dataset context)

Ollama returns a review decision:
```json
{"approved": true, "corrected_score": 0.74, "reason": "short explanation"}
```

If `approved` is `true`:
- Use `corrected_score` as the final score (it may differ from the generated score)
- Save the row to the dataset

If `approved` is `false`:
- Discard the row
- Do not save it
- Continue to the next generation attempt

The reviewer's `reason` field is used for internal debugging only. It is never saved to the dataset.

---

## Python Validation

In addition to the Ollama reviewer, the Python script runs its own validation checks.

A row is valid only if:
1. It is valid JSON
2. It has both `input_text` and `knowledge_score` fields
3. `input_text` is a string
4. `knowledge_score` is a number between 0.0 and 1.0
5. `input_text` is between 8 and 500 characters
6. `input_text` does not contain markdown formatting (`**`, `##`, ` ``` `, etc.)
7. `input_text` does not contain student name indicators (`student:`, `name:`, `my name is`, etc.)
8. `input_text` is not a duplicate of anything already saved
9. `knowledge_score` falls within a valid score band

If any check fails, the row is rejected and the reason is printed to the console.

---

## Scoring Scale

| Score Range | Band Name | What It Means |
|---|---|---|
| 0.00–0.20 | No evidence | Filler, vague agreement, no technical content |
| 0.21–0.40 | Basic awareness | Knows the topic exists, no real explanation |
| 0.41–0.60 | Partial understanding | Explains part of the idea, missing depth or details |
| 0.61–0.80 | Solid understanding | Explains what, why, or how with real context and reasoning |
| 0.81–1.00 | Advanced understanding | Tradeoffs, architecture, debugging, limitations, cause and effect |

---

## Annotated Examples

### Low score (0.00–0.20): No evidence

**Input text:**
> "Yeah I agree with that."

**Knowledge score:** `0.08`

**Why:** The student is participating but shows no evidence of technical understanding. There is no information about what they know.

---

### Basic awareness (0.21–0.40)

**Input text:**
> "I think we need an API to connect the app to Slack."

**Knowledge score:** `0.32`

**Why:** The student knows an API is involved, but does not explain what the API does, how it works, or why it is needed. This is a name-drop, not an explanation.

---

### Partial understanding (0.41–0.60)

**Input text:**
> "The Slack API can get messages from channels, but I still need to figure out the permissions side of it."

**Knowledge score:** `0.55`

**Why:** The student understands part of the idea — the API can retrieve messages — but acknowledges a gap (permissions). They know something is missing but have not resolved it yet.

---

### Solid understanding (0.61–0.80)

**Input text:**
> "The bot needs to be invited into a private Slack channel before it can read messages because private channels block access unless the bot is a member."

**Knowledge score:** `0.76`

**Why:** The student explains the rule clearly, explains why it works that way, and connects it to the project. No gap is visible in this statement.

---

### Advanced understanding (0.81–1.00)

**Input text:**
> "I separated Slack message tracking from Huddle transcript capture because the Slack API can read channel messages with the right permissions, but it does not expose raw Huddle audio — so those need different tools and processing pipelines."

**Knowledge score:** `0.93`

**Why:** The student explains an architectural decision, distinguishes two different systems, understands a limitation of the Slack API, and reasons about why the pipelines need to be separate. This is the highest tier of evidence.

---

## Final Dataset Format

The output file is `cic_knowledge_score_dataset.jsonl`.

Each line is one JSON object with exactly two fields:

```json
{"input_text": "student statement here", "knowledge_score": 0.74}
```

**What is saved:**
- `input_text` — the student statement
- `knowledge_score` — the final accepted score

**What is NOT saved:**
- Skill or field used for topic inspiration
- Target band name
- Reviewer reason or notes
- Any metadata

The script uses skill, field, and reasoning internally during generation and review, but none of that is stored in the final dataset. The dataset is intentionally simple so it can be used directly for model training.

---

## Data Balancing

The dataset must be evenly distributed across all five score bands.

If `TARGET_ROWS = 1000`, the target distribution is:

| Band | Target count |
|---|---|
| 0.00–0.20 | ~200 |
| 0.21–0.40 | ~200 |
| 0.41–0.60 | ~200 |
| 0.61–0.80 | ~200 |
| 0.81–1.00 | ~200 |

At each loop iteration, the script checks which band currently has the fewest accepted rows and targets that band next. If two or more bands are tied for fewest, it picks one at random.

**Why balance matters:**
- If the dataset has too many high scores, the future model may overpredict knowledge.
- If the dataset has too many low scores, the future model may be too harsh.
- A balanced dataset helps the model learn the full 0.0–1.0 range and generalize properly.

---

## Skills and Fields Used for Topic Inspiration

The script picks one skill at random for each generation attempt. The skill is passed to Ollama as thematic context, but is never saved in the output file.

| Skill | Field |
|---|---|
| Web Development | Applications |
| Robotics | Robotics |
| Data Curation | Data |
| Data Collection | Data |
| Game Design | Game Development |
| MCP Server | Artificial Intelligence |
| Prompt Engineering | Artificial Intelligence |
| Game Coding | Game Development |
| Workflow Building | Artificial Intelligence |
| Mathematics | Soft Skills |
| Knowledge Articulation | Soft Skills |
| Model Training | Artificial Intelligence |
| System Architecture | Artificial Intelligence |
| App Development | Applications |
| Animation | Game Development |
| Game Concept | Game Development |
| Sound Design | Production |
| Game Production | Production |
| Problem Solving | Soft Skills |
| Teamwork | Soft Skills |
| Group Leadership | Soft Skills |
| 3D Character Modeling | Digital Art |
| API Creation | Production |
| App Deployment | Production |

---

## Resume Feature

If the script is stopped (by Ctrl+C or any other reason), it can be resumed simply by running it again.

When the script starts, it checks whether `cic_knowledge_score_dataset.jsonl` already exists. If it does, the script loads all existing valid rows and continues from where it left off.

The existing rows are also loaded into a set so the script can avoid saving duplicates.

---

## Error Handling

| Error | Behavior |
|---|---|
| Ollama not running | Prints an error message and exits immediately |
| Model not found | Prints a model error message and exits immediately |
| JSON parse failure | Rejects the row and continues |
| Missing required fields | Rejects the row and continues |
| Duplicate input_text | Rejects the row and continues |
| Score out of range | Rejects the row and continues |
| Too many consecutive rejects | Prints a warning but keeps trying |

---

## Key Config Values

These are defined near the top of `generate_dataset.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `"llama3.1"` | The Ollama model to use |
| `OLLAMA_URL` | `"http://localhost:11434/api/generate"` | The local Ollama API endpoint |
| `TARGET_ROWS` | `1000` | How many accepted rows to collect |
| `OUTPUT_FILE` | `"cic_knowledge_score_dataset.jsonl"` | Output file name |
| `MAX_REJECTS_IN_A_ROW` | `50` | How many consecutive rejects before printing a warning |

---

## Key Functions in generate_dataset.py

| Function | What It Does |
|---|---|
| `call_ollama(system_prompt, prompt)` | Sends a request to Ollama and returns the parsed JSON response |
| `generate_candidate(target_band, skill)` | Asks Ollama to generate one candidate training row |
| `review_candidate(candidate, target_band, band_counts)` | Asks Ollama to review a candidate row for quality and score accuracy |
| `validate_candidate(candidate, existing_texts, target_band)` | Validates a row using Python logic only |
| `get_score_band(score)` | Returns the band dict for a given score |
| `choose_underfilled_band(band_counts)` | Returns the band with the fewest accepted rows |
| `load_existing_dataset()` | Loads existing rows from the JSONL file for resume support |
| `save_row(row)` | Appends one accepted row to the JSONL file |
| `print_progress(accepted, rejected, band_counts)` | Prints current progress to the console |
| `main()` | Controls the active generation loop |

---

## This Is Part 1 Only

This project is the first step in a larger pipeline.

Later parts may include:

- **Part 2:** Real transcript collection from Slack and Zoom
- **Part 3:** Transcript chunking and preprocessing
- **Part 4:** Skill categorization and tagging
- **Part 5:** Knowledge scoring using the trained model
- **Part 6:** Student knowledge reports and dashboards
- **Part 7:** Autonomous agent analysis and instructor alerts

Right now, only the local synthetic data generator is being built. The goal is to have a usable dataset ready before the rest of the pipeline is in place.
