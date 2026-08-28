"""
Scenario templates for V2 transcript-first generation.

Scenarios describe meeting intent and natural story count — not predetermined
section titles or time slots.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioTemplate:
    key: str
    title: str
    target_story_count: int
    duration_minutes_range: tuple[int, int]
    speaker_count_range: tuple[int, int]
    narrative_hints: str


# Distribution for 20-example test run: 6/4/4/3/3
STORY_COUNT_QUOTAS: dict[int, int] = {1: 6, 2: 4, 3: 4, 4: 3, 5: 3}


SCENARIO_TEMPLATES: list[ScenarioTemplate] = [
    # --- 1 story ---
    ScenarioTemplate(
        key="single_ml_project",
        title="Single machine-learning project deep dive",
        target_story_count=1,
        duration_minutes_range=(15, 25),
        speaker_count_range=(3, 5),
        narrative_hints=(
            "One continuous project discussion. Subtopics may include dataset selection, "
            "model choice, evaluation metrics, and train/test split — but these are parts "
            "of ONE story, not separate stories."
        ),
    ),
    ScenarioTemplate(
        key="single_community_event",
        title="Community event planning session",
        target_story_count=1,
        duration_minutes_range=(18, 28),
        speaker_count_range=(4, 6),
        narrative_hints=(
            "Planning one community event. Discussion may cover basketball, food vendors, "
            "registration, and sponsors — all subtopics of the same event, one story."
        ),
    ),
    ScenarioTemplate(
        key="single_video_project",
        title="Video editing project troubleshooting",
        target_story_count=1,
        duration_minutes_range=(12, 22),
        speaker_count_range=(2, 4),
        narrative_hints=(
            "One video project workflow: editing, noise cleanup, captions, export. "
            "Progression inside one story."
        ),
    ),
    ScenarioTemplate(
        key="single_website_launch",
        title="Website launch sprint",
        target_story_count=1,
        duration_minutes_range=(20, 30),
        speaker_count_range=(3, 5),
        narrative_hints=(
            "One website launch effort: design fixes, Stripe testing, hosting, deployment. "
            "Single coherent objective."
        ),
    ),
    ScenarioTemplate(
        key="single_game_dev",
        title="Game mechanics iteration session",
        target_story_count=1,
        duration_minutes_range=(15, 25),
        speaker_count_range=(3, 5),
        narrative_hints=(
            "One game dev session iterating on mechanics, physics, and playtesting feedback."
        ),
    ),
    ScenarioTemplate(
        key="single_3d_art",
        title="3D character rigging workshop",
        target_story_count=1,
        duration_minutes_range=(18, 28),
        speaker_count_range=(2, 4),
        narrative_hints=(
            "One 3D art session focused on rigging, weight painting, and animation tests."
        ),
    ),
    # --- 2 stories ---
    ScenarioTemplate(
        key="client_work_then_robotics",
        title="Website client work, then unrelated robotics club",
        target_story_count=2,
        duration_minutes_range=(20, 30),
        speaker_count_range=(4, 6),
        narrative_hints=(
            "First: website client feedback and fixes. Clear shift: unrelated robotics club "
            "discussion with different objective."
        ),
    ),
    ScenarioTemplate(
        key="project_demo_then_field_trip",
        title="Game programming demo, then field trip logistics",
        target_story_count=2,
        duration_minutes_range=(18, 28),
        speaker_count_range=(4, 6),
        narrative_hints=(
            "First: game programming project demo and Q&A. Second: upcoming field trip "
            "planning — genuinely different topic."
        ),
    ),
    ScenarioTemplate(
        key="ai_chatbot_then_personal_issue",
        title="AI chatbot project, then student personal issue",
        target_story_count=2,
        duration_minutes_range=(20, 30),
        speaker_count_range=(3, 5),
        narrative_hints=(
            "First: AI chatbot implementation discussion. Second: unrelated personal/student "
            "issue that interrupts and changes the meeting focus."
        ),
    ),
    ScenarioTemplate(
        key="art_critique_then_logistics",
        title="Art portfolio critique, then program logistics",
        target_story_count=2,
        duration_minutes_range=(22, 32),
        speaker_count_range=(4, 6),
        narrative_hints=(
            "First: detailed art portfolio critique. Second: program logistics and scheduling."
        ),
    ),
    # --- 3 stories ---
    ScenarioTemplate(
        key="three_topic_standup",
        title="Three distinct project standups",
        target_story_count=3,
        duration_minutes_range=(25, 35),
        speaker_count_range=(4, 7),
        narrative_hints=(
            "Three genuinely separate project updates: e.g. mobile app, mural project, "
            "and podcast episode — each with its own objective."
        ),
    ),
    ScenarioTemplate(
        key="demo_help_planning",
        title="Demo, technical help, planning",
        target_story_count=3,
        duration_minutes_range=(28, 38),
        speaker_count_range=(4, 6),
        narrative_hints=(
            "Story 1: student demo. Story 2: technical troubleshooting for another student. "
            "Story 3: planning next week's deliverables."
        ),
    ),
    ScenarioTemplate(
        key="design_code_deploy",
        title="Design review, coding session, deployment",
        target_story_count=3,
        duration_minutes_range=(30, 40),
        speaker_count_range=(3, 5),
        narrative_hints=(
            "Three distinct phases with clear topic changes: UI design review, backend coding, "
            "deployment discussion."
        ),
    ),
    ScenarioTemplate(
        key="mentor_feedback_three",
        title="Three separate mentor feedback blocks",
        target_story_count=3,
        duration_minutes_range=(25, 35),
        speaker_count_range=(4, 6),
        narrative_hints=(
            "Mentor gives feedback on three unrelated student projects in sequence."
        ),
    ),
    # --- 4 stories ---
    ScenarioTemplate(
        key="four_student_demos",
        title="Four short student demos",
        target_story_count=4,
        duration_minutes_range=(30, 40),
        speaker_count_range=(5, 8),
        narrative_hints=(
            "Four distinct student project demos, each on a different topic."
        ),
    ),
    ScenarioTemplate(
        key="workshop_four_modules",
        title="Workshop with four modules",
        target_story_count=4,
        duration_minutes_range=(35, 45),
        speaker_count_range=(4, 7),
        narrative_hints=(
            "Workshop covering four separate topics: intro to shaders, particle systems, "
            "lighting, and post-processing — each a distinct teaching segment."
        ),
    ),
    ScenarioTemplate(
        key="planning_four_tracks",
        title="Planning session for four program tracks",
        target_story_count=4,
        duration_minutes_range=(32, 42),
        speaker_count_range=(5, 7),
        narrative_hints=(
            "Discussion of four program tracks: 2D art, 3D art, game dev, and AI — "
            "each as a separate planning block."
        ),
    ),
    # --- 5+ stories ---
    ScenarioTemplate(
        key="five_quick_updates",
        title="Five quick project updates",
        target_story_count=5,
        duration_minutes_range=(35, 45),
        speaker_count_range=(6, 9),
        narrative_hints=(
            "Five brief but distinct project updates from different students."
        ),
    ),
    ScenarioTemplate(
        key="six_topic_roundtable",
        title="Six-topic roundtable",
        target_story_count=6,
        duration_minutes_range=(40, 50),
        speaker_count_range=(6, 9),
        narrative_hints=(
            "Roundtable touching six genuinely separate topics: hackathon prep, internship "
            "applications, equipment requests, showcase date, mentor availability, and "
            "community outreach."
        ),
    ),
    ScenarioTemplate(
        key="five_demos_plus_qa",
        title="Five demos with Q&A blocks",
        target_story_count=5,
        duration_minutes_range=(38, 48),
        speaker_count_range=(6, 8),
        narrative_hints=(
            "Five student demos, each followed by a short Q&A — five distinct stories."
        ),
    ),
]


def build_story_count_schedule(count: int, seed: int = 42) -> list[int]:
    """Build a shuffled list of target story counts for `count` examples."""
    if count <= 20:
        schedule: list[int] = []
        for story_count, quota in STORY_COUNT_QUOTAS.items():
            schedule.extend([story_count] * quota)
        schedule = schedule[:count]
        while len(schedule) < count:
            schedule.append(random.choice([1, 2, 3, 4, 5]))
    else:
        weights = [(1, 30), (2, 20), (3, 20), (4, 15), (5, 15)]
        population = [w[0] for w in weights]
        probs = [w[1] / 100 for w in weights]
        schedule = random.choices(population, weights=probs, k=count)

    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


def pick_scenario(target_story_count: int, rng: random.Random | None = None) -> ScenarioTemplate:
    rng = rng or random
    candidates = [s for s in SCENARIO_TEMPLATES if s.target_story_count == target_story_count]
    if not candidates:
        if target_story_count >= 5:
            candidates = [s for s in SCENARIO_TEMPLATES if s.target_story_count >= 5]
        else:
            candidates = SCENARIO_TEMPLATES
    return rng.choice(candidates)


def scenario_duration_minutes(scenario: ScenarioTemplate, rng: random.Random | None = None) -> int:
    rng = rng or random
    low, high = scenario.duration_minutes_range
    return rng.randint(low, high)
