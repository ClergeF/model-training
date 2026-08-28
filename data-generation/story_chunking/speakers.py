"""
Speaker name selection for synthetic meetings.

Each meeting gets a unique cast so training data is not tied to a fixed
set of real names. Names are mostly procedurally generated from first/last
pools; a small chance of picking from legacy-style pools for variety.
"""

from __future__ import annotations

import random

# Expanded pools — fictional / diverse names, not tied to one real roster.
FIRST_NAMES = [
    "Alex", "Jordan", "Sam", "Taylor", "Morgan", "Riley", "Casey", "Avery",
    "Quinn", "Reese", "Dakota", "Skyler", "Jamie", "Devon", "Kai", "Noah",
    "Mia", "Elena", "Marcus", "Priya", "Diego", "Aaliyah", "Omar", "Lin",
    "Jasmine", "Ethan", "Sofia", "Andre", "Nina", "Leo", "Zara", "Chris",
    "Maria", "Tyler", "Aiden", "Brianna", "Malik", "Hannah", "Isaac", "Luna",
]

LAST_NAMES = [
    "Rivera", "Chen", "Patel", "Johnson", "Williams", "Garcia", "Kim", "Nguyen",
    "Brown", "Davis", "Martinez", "Lee", "Wilson", "Anderson", "Thomas", "Jackson",
    "White", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Hall", "Young",
    "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker",
    "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell",
]

# Occasional mentor/student-style labels (low weight so casts stay varied).
MENTOR_STYLE_NAMES = [
    "Coach Williams",
    "Ms. Thompson",
    "Mr. Okonkwo",
    "Dr. Hayes",
    "Facilitator Kim",
]

STUDENT_STYLE_NAMES = [
    "Student A",
    "Student B",
]


def _unique_full_name(used: set[str]) -> str:
    for _ in range(50):
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        if name not in used:
            used.add(name)
            return name
    # Fallback if collision lottery fails (very unlikely).
    suffix = random.randint(100, 999)
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)} {suffix}"
    used.add(name)
    return name


def choose_speakers() -> list[str]:
    """
    Pick a unique cast for one meeting: 1-2 mentor-like leads + 3-6 participants.
    """
    used: set[str] = set()
    speakers: list[str] = []

    mentor_count = random.randint(1, 2)
    student_count = random.randint(3, 6)

    for _ in range(mentor_count):
        if random.random() < 0.25:
            pool = MENTOR_STYLE_NAMES + [
                f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            ]
            candidates = [n for n in pool if n not in used]
            if candidates:
                name = random.choice(candidates)
            else:
                name = _unique_full_name(used)
        else:
            name = _unique_full_name(used)
        used.add(name)
        speakers.append(name)

    for _ in range(student_count):
        if random.random() < 0.1 and STUDENT_STYLE_NAMES:
            candidates = [n for n in STUDENT_STYLE_NAMES if n not in used]
            name = random.choice(candidates) if candidates else _unique_full_name(used)
        else:
            name = _unique_full_name(used)
        used.add(name)
        speakers.append(name)

    random.shuffle(speakers)
    return speakers
