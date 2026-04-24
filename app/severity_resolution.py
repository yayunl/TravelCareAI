"""Merge rule-based triage with optional LLM severity_assessment."""

from __future__ import annotations

import logging
from typing import Any

from app.triage import CareLevel, RuleTriageResult

logger = logging.getLogger(__name__)

_VALID_LEVELS: frozenset[str] = frozenset(
    {"emergency_immediate", "er_hospital", "clinic", "pharmacy_self_care"}
)

_ORDER: tuple[str, ...] = ("pharmacy_self_care", "clinic", "er_hospital", "emergency_immediate")


def _level_rank(level: str) -> int:
    try:
        return _ORDER.index(level)
    except ValueError:
        return -1


def extract_severity_assessment(llm_out: dict[str, Any]) -> dict[str, Any] | None:
    """Return a cleaned disagreement payload, or None if model agrees / field unusable."""
    raw = llm_out.get("severity_assessment")
    if not isinstance(raw, dict):
        return None
    agrees = raw.get("agrees_with_server_triage")
    if agrees is not False:
        return None

    level_raw = raw.get("suggested_care_level")
    if not isinstance(level_raw, str):
        return None
    suggested_level = level_raw.strip()
    if suggested_level not in _VALID_LEVELS:
        return None

    se = raw.get("suggested_emergency")
    suggested_emergency: bool | None
    if isinstance(se, bool):
        suggested_emergency = se
    else:
        suggested_emergency = None

    rationale = raw.get("rationale_for_adjustment")
    rationale_s = str(rationale).strip()[:800] if rationale is not None else ""
    if len(rationale_s) < 8:
        return None

    return {
        "agrees_with_server_triage": False,
        "suggested_care_level": suggested_level,
        "suggested_emergency": suggested_emergency,
        "rationale_for_adjustment": rationale_s,
    }


def _effective_emergency(level: str, suggested_emergency: bool | None) -> bool:
    if level == "emergency_immediate":
        return True
    if isinstance(suggested_emergency, bool):
        return suggested_emergency
    return level in ("er_hospital", "emergency_immediate")


def merge_effective_severity(rules: RuleTriageResult, llm_out: dict[str, Any] | None) -> dict[str, Any]:
    """
    Build API fields: effective care_level / emergency plus audit trail.

    Guardrail: if rules already flagged an emergency, the model cannot lower
    severity via severity_assessment (rejected with a reason string).
    """
    rule_care: CareLevel = rules.care_level
    rule_emerg = rules.emergency
    rule_snap = {
        "care_level": rule_care,
        "emergency": rule_emerg,
        "matched_rules": list(rules.matched_rules),
        "rationale": list(rules.rationale),
    }
    base_out: dict[str, Any] = {
        "care_level": rule_care,
        "emergency": rule_emerg,
        "severity_source": "rules",
        "rule_triage": rule_snap,
    }
    if not llm_out or not isinstance(llm_out, dict):
        return base_out

    sa = extract_severity_assessment(llm_out)
    if not sa:
        return base_out

    suggested_level = str(sa["suggested_care_level"])
    suggested_emergency = sa.get("suggested_emergency")

    if rule_emerg or rule_care == "emergency_immediate":
        new_rank = _level_rank(suggested_level)
        eff_emerg = _effective_emergency(suggested_level, suggested_emergency)
        if new_rank < _level_rank("er_hospital") or not eff_emerg:
            logger.info(
                "llm severity override rejected: rules had emergency care_level=%s emergency=%s",
                rule_care,
                rule_emerg,
            )
            return {
                **base_out,
                "llm_severity_override_rejected": True,
                "llm_severity_override_reject_reason": (
                    "Rule triage flagged an emergency-level pattern; the model cannot lower "
                    "severity using severity_assessment. Effective values remain from rules."
                ),
            }

    eff_level = suggested_level
    eff_emerg = _effective_emergency(suggested_level, suggested_emergency)
    logger.info(
        "severity merged from LLM: rule care=%s -> effective=%s emergency=%s",
        rule_care,
        eff_level,
        eff_emerg,
    )
    return {
        "care_level": eff_level,
        "emergency": eff_emerg,
        "severity_source": "llm_adjusted",
        "rule_triage": rule_snap,
    }
