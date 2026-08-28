"""
Prompts for synthetic Story Chunking data generation.

Edit this file to change what the LLM sees. Each generation example makes two
kinds of API calls:

  1. Plan call   — PLAN_SYSTEM_PROMPT + PLAN_USER_PROMPT (filled per meeting)
  2. Section call — SECTION_SYSTEM_PROMPT + SECTION_USER_PROMPT (per section)

Placeholders in user prompts (filled at runtime):
  Plan:    {duration_minutes}, {end_time}, {topics_text}, {speakers_text}, {style_hints}
  Section: {section_title}, {section_summary}, {start_time}, {end_time},
           {speakers_text}, {style_hints}

INSTRUCTION is stored in the output JSONL only; it is not sent to the generator LLM.
Speaker names are chosen per meeting in speakers.py. Topics are sampled from TOPIC_POOL.
"""

INSTRUCTION = (
    "Read the full meeting transcript and organize it into story sections "
    "based on topic changes."
)

TOPIC_POOL = [
    "2D art",
    "3D art",
    "game design",
    "game programming",
    "AI",
    "using art, games, and AI for community engagement",
    "student updates",
    "project demos",
    "technical help",
    "program logistics",
    "mentor feedback",
    "casual meeting check-ins",
]

STYLE_HINTS = """Style notes for these meetings:
- A few mentors lead, many students join. People talk over each other.
- Casual openings: greetings, jokes, asking people to turn cameras on, health check-ins.
- Mentors give long motivational or logistics monologues.
- Students give short updates, ask technical questions, or demo work.
- Topics shift naturally: someone finishes a demo, then talk drifts to AI, then logistics.
- Speech is messy: filler words, restarts, interruptions, half sentences.
- Subjects rotate through art, game dev, AI, community engagement, and program logistics.
"""

# ---------------------------------------------------------------------------
# Plan call — meeting outline (system + user)
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """You are a planner that designs realistic synthetic meeting outlines for a youth creative-technology program (art, games, AI, community engagement).

You will be given:
- A meeting duration in minutes
- A list of topics the meeting should cover
- A list of likely speaker names

You must return ONLY valid JSON in this exact shape:
{
  "meeting_summary": "2-4 sentence summary of the whole meeting",
  "story_sections": [
    {
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "section_title": "short title",
      "summary": "1-2 sentence summary of this section",
      "speakers": ["Name", "Name"],
      "confidence": 0.9
    }
  ]
}

Rules:
- Cover the FULL meeting duration with no gaps and no overlaps.
- The first section starts at 00:00 and the last section ends at the meeting end time.
- Produce one section per distinct topic; sections must be in time order.
- Each section needs at least 2 speakers drawn ONLY from the provided names list.
- Use the exact spellings from the provided names list for speakers.
- confidence is a float between 0.6 and 1.0 (how clear the topic boundary is).
- Return ONLY the JSON object, no markdown, no explanation."""

PLAN_USER_PROMPT = """Meeting duration: {duration_minutes} minutes (ends at {end_time}).
Cover these topics, one section each, in a natural order:
{topics_text}

Likely speakers (use ONLY these names, exact spelling):
{speakers_text}

{style_hints}

Design the meeting outline now. Return ONLY the JSON object."""

# ---------------------------------------------------------------------------
# Section call — messy transcript lines (system + user)
# ---------------------------------------------------------------------------

SECTION_SYSTEM_PROMPT = """You are a transcript writer that produces ONE messy section of a real group meeting transcript for a youth creative-technology program.

You will be given:
- The section title, summary, time range, and speakers
- Style notes describing how the meeting sounds

You must return ONLY valid JSON in this exact shape:
{
  "lines": [
    {"start": "MM:SS", "end": "MM:SS", "speaker": "Name", "text": "what they say"}
  ]
}

Rules:
- Lines must stay within the given section time range and be in time order.
- Use ONLY the provided speaker names, with exact spelling.
- Distribute lines across multiple speakers; do not give one person all the lines.
- Make it sound real and messy: interruptions, filler words, restarts, short replies, the odd long monologue.
- Keep the conversation focused on the section's topic, but allow a natural lead-in or transition.
- Write 6 to 16 lines.
- Return ONLY the JSON object, no markdown, no explanation."""

SECTION_USER_PROMPT = """Section title: {section_title}
Section summary: {section_summary}
Time range: {start_time} to {end_time}
Speakers for this section (use ONLY these names, exact spelling): {speakers_text}

{style_hints}

Write the messy transcript lines for this section now. Return ONLY the JSON object."""


def build_plan_prompt(duration_minutes: int, topics: list[str], speakers: list[str]) -> str:
    end_time = f"{duration_minutes:02d}:00"
    topics_text = "\n".join(f"- {topic}" for topic in topics)
    speakers_text = ", ".join(speakers)
    return PLAN_USER_PROMPT.format(
        duration_minutes=duration_minutes,
        end_time=end_time,
        topics_text=topics_text,
        speakers_text=speakers_text,
        style_hints=STYLE_HINTS,
    )


def build_section_prompt(section: dict, speakers: list[str]) -> str:
    speakers_text = ", ".join(speakers)
    return SECTION_USER_PROMPT.format(
        section_title=section.get("section_title", ""),
        section_summary=section.get("summary", ""),
        start_time=section.get("start_time", ""),
        end_time=section.get("end_time", ""),
        speakers_text=speakers_text,
        style_hints=STYLE_HINTS,
    )
