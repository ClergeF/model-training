"""JSON schemas for Ollama structured output on plan and section calls."""

PLAN_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "meeting_summary": {"type": "string"},
        "story_sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "section_title": {"type": "string"},
                    "summary": {"type": "string"},
                    "speakers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                },
                "required": [
                    "start_time",
                    "end_time",
                    "section_title",
                    "summary",
                    "speakers",
                    "confidence",
                ],
            },
        },
    },
    "required": ["meeting_summary", "story_sections"],
}

SECTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "speaker", "text"],
            },
        }
    },
    "required": ["lines"],
}
