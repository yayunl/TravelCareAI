from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

CareLevel = Literal["emergency_immediate", "er_hospital", "clinic", "pharmacy_self_care"]


@dataclass
class RuleTriageResult:
    care_level: CareLevel
    emergency: bool
    matched_rules: list[str]
    rationale: list[str]


_RED_FLAG_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "possible_stroke",
        re.compile(
            r"\b(face droop|facial droop|slurred speech|slurred|one side weak|"
            r"one-sided weakness|sudden confusion|sudden severe headache|FAST)\b",
            re.I,
        ),
        "Possible stroke or neurological emergency — seek emergency care immediately (use your local emergency number).",
    ),
    (
        "chest_pain_severe",
        re.compile(
            r"\b(chest pain|crushing chest|pain radiat|short of breath|can't breathe|cannot breathe)\b",
            re.I,
        ),
        "Chest pain or severe breathing difficulty — treat as emergency until evaluated (call local emergency services or go to the nearest ER).",
    ),
    (
        "severe_bleeding",
        re.compile(
            r"\b(uncontrolled bleed|severe bleed|vomit blood|coughing blood|"
            r"passing out|fainted|loss of consciousness|unconscious)\b",
            re.I,
        ),
        "Severe bleeding or altered consciousness — contact local emergency services immediately.",
    ),
    (
        "anaphylaxis_hint",
        re.compile(
            r"\b(throat closing|can't swallow|lip swell|tongue swell|"
            r"anaphylaxis|after epipen|used epipen)\b",
            re.I,
        ),
        "Possible severe allergic reaction — seek emergency care (local emergency number or nearest ER).",
    ),
    (
        "suicide_self_harm",
        re.compile(
            r"\b(suicid|kill myself|self[- ]harm|want to die)\b",
            re.I,
        ),
        "Crisis safety: if in immediate danger, contact local emergency services or crisis lines for your area, or go to the nearest ER.",
    ),
]

_MODERATE_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "high_fever",
        re.compile(r"\b(39|40|41)\s*°?c|high fever|febrile|fever.+\b(102|103|104)\b", re.I),
        "High fever — same-day medical assessment is often appropriate; pharmacy alone may be insufficient.",
    ),
    (
        "severe_abdominal",
        re.compile(r"\b(severe stomach pain|severe abdominal|appendicitis)\b", re.I),
        "Severe abdominal pain — urgent in-person evaluation recommended.",
    ),
]


def rule_triage(user_text: str) -> RuleTriageResult:
    """Deterministic keyword / pattern layer. Not a clinical device."""
    text = user_text.strip()
    matched: list[str] = []
    rationale: list[str] = []

    for rule_id, pattern, msg in _RED_FLAG_PATTERNS:
        if pattern.search(text):
            matched.append(rule_id)
            rationale.append(msg)

    if matched:
        return RuleTriageResult(
            care_level="emergency_immediate",
            emergency=True,
            matched_rules=matched,
            rationale=rationale,
        )

    for rule_id, pattern, msg in _MODERATE_PATTERNS:
        if pattern.search(text):
            matched.append(rule_id)
            rationale.append(msg)

    if matched:
        return RuleTriageResult(
            care_level="er_hospital",
            emergency=False,
            matched_rules=matched,
            rationale=rationale,
        )

    low = re.compile(
        r"\b(sore throat|runny nose|cough|mild cold|rash itch|minor cut|band-?aid|"
        r"allergy sneez|hay fever|mild headache)\b",
        re.I,
    )
    if low.search(text):
        return RuleTriageResult(
            care_level="pharmacy_self_care",
            emergency=False,
            matched_rules=["low_acuity_lexicon"],
            rationale=[
                "Mild symptoms described — pharmacy or self-care may be reasonable if symptoms stay mild and no red flags appear.",
            ],
        )

    return RuleTriageResult(
        care_level="clinic",
        emergency=False,
        matched_rules=["default_non_emergency"],
        rationale=[
            "No urgent keyword pattern matched this description (the app uses a short safety screen, not a diagnosis). "
            "If symptoms persist, worsen, or worry you, consider an in-person clinic or primary-care visit.",
        ],
    )
