"""
Prompts for Moment Detection synthetic data generation.

Labeling rules follow moment-detection/docs/MOMENT_DETECTION_SPEC_v1.3.txt
(canonical reference: Moment_Detection_Model_Logic.docx v1.3).
"""

from __future__ import annotations

import random

SCENARIO_HINTS = [
    "students discussing a technical project or debugging session",
    "someone asking a peer or mentor a specific question",
    "someone explaining how something works step by step",
    "someone helping another person troubleshoot a problem",
    "someone sharing useful information or a resource",
    "follow-up about something discussed earlier",
    "brief project update among teammates",
    "planning next steps for a community or school event",
    "informal demo of work in progress",
    "casual conversation that may or may not contain a meaningful check-in",
    "multiple people talking about one subject with several back-and-forth turns",
    "someone referencing help they gave another person earlier",
    "normal workplace or program conversation without obvious structure",
]

INPUT_SYSTEM_PROMPT = """You write realistic short conversational excerpts for training data.

Each excerpt is a STORY CHUNK: a paragraph-sized slice of dialogue or narrative
from a group meeting or conversation (roughly 50-250 words, natural variation allowed).

Rules:
- Write natural speech or transcript-style text with named people when appropriate.
- Include realistic variety: questions, explanations, updates, troubleshooting, casual talk.
- Do NOT decide whether the text contains a "Moment".
- Do NOT count interactions, classify moments, or label categories.
- Do NOT add JSON, markdown, titles, or meta commentary.
- Do NOT start with phrases like "Here is your requested conversation:".
- Output ONLY the conversational text itself."""

INPUT_USER_PROMPT = """Write one realistic story chunk.

Style hint: {scenario_hint}
Target length: about {target_words} words (natural variation is fine).

Output ONLY the text."""


LABEL_SYSTEM_PROMPT = """You are the Moment Detection labeling model (spec v1.3).

A Moment is one meaningful, bounded check-in where one person intentionally
reaches out to, checks on, supports, helps, informs, asks, follows up with,
mentors, or otherwise meaningfully engages another identifiable person.

Counting rule: 1 bounded interaction × 1 identifiable beneficiary = 1 Moment record.
Multiple actions in one continuous check-in with one beneficiary = 1 Moment.
One group interaction with N named beneficiaries = N Moment objects.

Historical interactions CAN be Moments if evidence is sufficient.
Vague statements like "I helped Sarah yesterday" alone → POTENTIAL, not CONFIRMED.

Return ONLY valid JSON with this exact top-level shape:
{
  "originalText": "<copy input text exactly>",
  "source": "<copy input source exactly>",
  "detectionStatus": "CONFIRMED" | "POTENTIAL" | "NOT_A_MOMENT",
  "momentCount": 0,
  "moments": [
    {
      "contributor": "string",
      "beneficiary": "string",
      "summary": "string",
      "momentType": "WELLNESS_CHECK|INFO_SHARING|INFO_REQUEST|FOLLOW_UP|SUPPORT|MENTORSHIP|MONETARY|TIME|DONATION|SERVICE|OTHER",
      "category": "Outreach Notes|Family & History|Business & Finance|Community & Health|Faith & Spirituality|Education & Innovation",
      "themes": ["optional"],
      "tone": "optional string"
    }
  ],
  "missingInformation": ["string"]
}

Rules:
- Extract names only from the text; never invent people.
- momentCount must equal len(moments) for CONFIRMED.
- For POTENTIAL or NOT_A_MOMENT, moments is [] and momentCount is 0.
- category is inferred from text; separate from momentType.
- summary must be grounded in the text; do not add facts.
- Copy originalText and source unchanged from the user message."""


def build_input_user_prompt() -> tuple[str, int]:
    scenario = random.choice(SCENARIO_HINTS)
    target_words = random.randint(50, 250)
    return (
        INPUT_USER_PROMPT.format(scenario_hint=scenario, target_words=target_words),
        target_words,
    )


def build_label_user_prompt(original_text: str, source: str) -> str:
    return (
        f"Label this input.\n\n"
        f"originalText:\n{original_text}\n\n"
        f"source: {source}\n\n"
        f"Return ONLY the JSON object."
    )
