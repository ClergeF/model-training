"""
Shared approved skill map for Model 2 labeling and validation.
"""

from collections import defaultdict

SKILL_MAP = {
    "Web Development": "Applications",
    "Robotics": "Robotics",
    "Data Curation": "Data",
    "Data Collection": "Data",
    "Game Design": "Game Development",
    "MCP Server": "Artificial Intelligence",
    "Prompt Engineering": "Artificial Intelligence",
    "Game Coding": "Game Development",
    "Workflow Building": "Artificial Intelligence",
    "Mathematics": "Soft Skills",
    "Knowledge Articulation": "Soft Skills",
    "Model Training": "Artificial Intelligence",
    "System Architecture": "Artificial Intelligence",
    "App Development": "Applications",
    "Animation": "Game Development",
    "Game Concept": "Game Development",
    "Sound Design": "Production",
    "Game Production": "Production",
    "Problem Solving": "Soft Skills",
    "Teamwork": "Soft Skills",
    "Group Leadership": "Soft Skills",
    "3D Character Modeling": "Digital Art",
    "API Creation": "Production",
    "App Deployment": "Production",
}

# field name -> list of skills in that field
FIELD_TO_SKILLS = defaultdict(list)
for _skill, _field in SKILL_MAP.items():
    FIELD_TO_SKILLS[_field].append(_skill)

VALID_FIELDS = set(FIELD_TO_SKILLS.keys())
VALID_SKILLS = set(SKILL_MAP.keys())

SKILL_LIST_TEXT = "\n".join(
    f'  - "{skill}" (field: "{field}")' for skill, field in SKILL_MAP.items()
)


def normalize_field_name(value):
    """Return canonical field name if value matches a field (case-insensitive)."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    for field in VALID_FIELDS:
        if field.lower() == value.lower():
            return field
    return None


def field_for_skill(skill):
    """Return the canonical field for a skill, or None if invalid."""
    return SKILL_MAP.get(skill)


def validate_skill_field_pair(skill, field):
    """True if skill is valid and field matches the map."""
    if skill not in SKILL_MAP:
        return False
    return SKILL_MAP[skill] == field
