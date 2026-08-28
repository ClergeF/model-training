"""
Skill -> field mapping for Model 2 (Technical Skill Categorization Classifier).

The classifier predicts skill only; field is derived from this map after prediction
so skill and field cannot mismatch.
"""

SKILL_TO_FIELD = {
    "Web Development": "Applications",
    "App Development": "Applications",
    "Robotics": "Robotics",
    "Data Curation": "Data",
    "Data Collection": "Data",
    "Game Design": "Game Development",
    "Game Coding": "Game Development",
    "Game Concept": "Game Development",
    "Animation": "Game Development",
    "MCP Server": "Artificial Intelligence",
    "Prompt Engineering": "Artificial Intelligence",
    "Workflow Building": "Artificial Intelligence",
    "Model Training": "Artificial Intelligence",
    "System Architecture": "Artificial Intelligence",
    "Mathematics": "Soft Skills",
    "Knowledge Articulation": "Soft Skills",
    "Problem Solving": "Soft Skills",
    "Teamwork": "Soft Skills",
    "Group Leadership": "Soft Skills",
    "Sound Design": "Production",
    "Game Production": "Production",
    "API Creation": "Production",
    "App Deployment": "Production",
    "3D Character Modeling": "Digital Art",
}
