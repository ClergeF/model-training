"""Provisional input schema and authoritative output schema (spec v1.3)."""

from __future__ import annotations

DETECTION_STATUSES = frozenset({"CONFIRMED", "POTENTIAL", "NOT_A_MOMENT"})

MOMENT_TYPES = frozenset(
    {
        "WELLNESS_CHECK",
        "INFO_SHARING",
        "INFO_REQUEST",
        "FOLLOW_UP",
        "SUPPORT",
        "MENTORSHIP",
        "MONETARY",
        "TIME",
        "DONATION",
        "SERVICE",
        "OTHER",
    }
)

CATEGORIES = frozenset(
    {
        "Outreach Notes",
        "Family & History",
        "Business & Finance",
        "Community & Health",
        "Faith & Spirituality",
        "Education & Innovation",
    }
)

INPUT_SOURCES = frozenset({"synthetic", "SYNTHETIC", "MANUAL", "API", "IMPORT"})

FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "moment_count",
        "is_moment",
        "moments",
        "confidence",
        "expected_output",
        "detectionStatus",
        "detection_status",
        "speaker_id",
        "participant_id",
        "category_id",
        "predicted_category",
        "sharing_option",
    }
)

MOMENT_OBJECT_KEYS = frozenset(
    {
        "contributor",
        "beneficiary",
        "summary",
        "momentType",
        "category",
        "themes",
        "tone",
    }
)

OUTPUT_TOP_KEYS = frozenset(
    {
        "originalText",
        "source",
        "detectionStatus",
        "momentCount",
        "moments",
        "missingInformation",
    }
)
