"""
Prompts for V2 transcript-first Story Chunking data generation.

Flow per example:
  1. Transcript call — full messy meeting, NO story boundaries supplied
  2. Label call    — read transcript, output meeting_summary + story_sections
  3. Review call   — optional quality check on proposed segmentation
"""

from __future__ import annotations

from .scenarios_v2 import ScenarioTemplate

INSTRUCTION_V2 = (
    "Read the full meeting transcript and divide it into genuine story sections "
    "based on meaningful topic or objective changes — not subtopics within the same story."
)

BANNED_SECTION_TITLES = [
    "Casual Opening",
    "Student Updates",
    "Technical Help",
    "Next Steps",
    "Wrap-up",
    "Closing",
]

STORY_VS_SUBTOPIC_RULES = """
STORY vs SUBTOPIC — critical distinction:

DO NOT create a new story merely because the discussion moves to a subtopic.

These may remain ONE story:
- ML project: dataset → model → metrics → train/test split
- Community event: basketball → vendors → registration → sponsors
- Video project: editing → noise cleanup → captions → export
- Website launch: design → Stripe → hosting → deployment

These SHOULD be separate stories:
- Website client work → unrelated robotics club discussion
- Game programming → upcoming field trip logistics
- AI chatbot project → unrelated personal/student issue

Ask: "Did the conversational objective genuinely change enough that a person
reviewing the recording would want this as a separate story?"
"""

TRANSCRIPT_SYSTEM_PROMPT = """You are a transcript writer producing a realistic, messy group meeting transcript for a youth creative-technology program.

You will receive a meeting scenario, duration, and speaker list. Write the COMPLETE meeting transcript.

Output format — one line per utterance:
[MM:SS - MM:SS] Speaker Name: what they said

Rules:
- Cover the full meeting duration with natural, uneven timestamps (NOT every minute on the dot).
- Use ONLY the provided speaker names, exact spelling.
- Make it sound real: filler words, interruptions, restarts, short replies, long monologues.
- Vary speaking style across speakers.
- Include tangents, questions, clarifications, and occasional returns to earlier subjects.
- Do NOT use a formulaic opening (greetings, camera checks) unless the scenario calls for it.
- Do NOT use formulaic closing or "next steps" unless they naturally occur.
- Do NOT label or mark story sections in the output.
- Do NOT invent story boundaries or section titles.
- Output ONLY the transcript lines, no JSON, no markdown, no commentary."""

TRANSCRIPT_USER_PROMPT = """Scenario: {scenario_title}
Duration: approximately {duration_minutes} minutes (end around {end_time})
Target number of genuine stories in this meeting: {target_story_count} (for realism only — do NOT mark boundaries)

Speakers (use ONLY these names):
{speakers_text}

Narrative guidance:
{narrative_hints}

{Diversity_hints}

Write the complete messy transcript now. Output ONLY timestamped lines."""

LABEL_SYSTEM_PROMPT = """You are a meeting transcript segmentation labeler.

Read the complete transcript and identify genuine STORY sections — not subtopics.

Return ONLY valid JSON:
{
  "meeting_summary": "2-4 sentence summary of the whole meeting",
  "story_sections": [
    {
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "section_title": "short specific title grounded in transcript",
      "summary": "1-2 sentence summary of this story"
    }
  ]
}

""" + STORY_VS_SUBTOPIC_RULES + """

TIMESTAMP RULES:
- Use ONLY timestamps that appear in the transcript (line start or end times).
- First section start_time = start of first transcript line.
- Adjacent sections share a boundary: section[i].end_time == section[i+1].start_time.
- Final section end_time = end of last transcript line.
- Do NOT invent timestamps (e.g. 01:00 when transcript has 00:58, 01:04).
- Do NOT create evenly spaced minute boundaries.

OUTPUT RULES:
- section_title: short, specific, grounded in actual discussion (avoid generic titles).
- summary: describe main discussion, activity, decision, or outcome.
- Do NOT include section_text, speakers, confidence, or transcript_id.
- Return ONLY the JSON object, no markdown."""

LABEL_USER_PROMPT = """Meeting duration: {duration}

Transcript:
{transcript_text}

Segment this transcript into genuine story sections. Return ONLY the JSON object."""

REVIEW_SYSTEM_PROMPT = """You are a quality reviewer for meeting transcript segmentation labels.

You will receive a transcript and proposed story_sections. Judge whether the segmentation is correct.

Return ONLY valid JSON:
{
  "approved": true,
  "issues": [],
  "suggested_story_sections": []
}

If approved is false, populate issues with specific problems and optionally provide
suggested_story_sections (same schema as input: start_time, end_time, section_title, summary).

Check:
1. Is every section grounded in the transcript?
2. Did the labeler invent topics, events, or timestamps?
3. Are adjacent sections really different stories (not subtopics of one story)?
4. Could adjacent sections reasonably be merged?
5. Is the transcript over-segmented?
6. Are any genuine major topic changes missing?
7. Do all timestamps exist in the transcript?

If suggesting fixes, use ONLY timestamps from the transcript.
Return ONLY the JSON object."""

REVIEW_USER_PROMPT = """Meeting duration: {duration}

Transcript:
{transcript_text}

Proposed segmentation:
{sections_json}

Review this segmentation. Return ONLY the JSON object."""

DIVERSITY_HINTS = """Diversity notes:
- Vary meeting structure: not every meeting needs opening, demos, help, and closing.
- Mix short and long utterances, technical and casual language.
- Some meetings are mostly one person explaining; others are rapid back-and-forth.
- Avoid repeating the same section title patterns across meetings."""


def build_transcript_prompt(
    scenario: ScenarioTemplate,
    duration_minutes: int,
    speakers: list[str],
    target_story_count: int,
) -> str:
    end_time = f"{duration_minutes:02d}:00"
    speakers_text = "\n".join(f"- {name}" for name in speakers)
    return TRANSCRIPT_USER_PROMPT.format(
        scenario_title=scenario.title,
        duration_minutes=duration_minutes,
        end_time=end_time,
        target_story_count=target_story_count,
        speakers_text=speakers_text,
        narrative_hints=scenario.narrative_hints,
        Diversity_hints=DIVERSITY_HINTS,
    )


def build_label_prompt(transcript_text: str, duration: str) -> str:
    return LABEL_USER_PROMPT.format(
        duration=duration,
        transcript_text=transcript_text,
    )


def build_review_prompt(transcript_text: str, duration: str, sections: list[dict]) -> str:
    import json

    sections_json = json.dumps(
        {"meeting_summary": "", "story_sections": sections},
        ensure_ascii=False,
        indent=2,
    )
    return REVIEW_USER_PROMPT.format(
        duration=duration,
        transcript_text=transcript_text,
        sections_json=sections_json,
    )
