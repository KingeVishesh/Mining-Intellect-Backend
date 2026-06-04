"""Shared diagnostics for blind gold Parallel backtests.

These helpers are intentionally deterministic and source-light: they never
fetch new data and they never inspect target MRE fields except when comparing a
finished prediction against the held-out truth supplied by the caller.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def evidence_quality_score(evidence: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a compact pre-MRE evidence score from 0-100.

    The score is not a source of truth; it is a routing signal for whether the
    blind model should trust target drilling, geometry, or analog-only fallback.
    """
    if not isinstance(evidence, dict) or evidence.get("redacted"):
        return {"score": 0, "grade": "none", "signals": []}

    signals: List[str] = []
    score = 0
    if evidence.get("queried_pre_mre_cutoff") or evidence.get("source_date"):
        score += 15
        signals.append("dated_pre_mre")
    if evidence.get("source_url"):
        score += 10
        signals.append("source_url")
    confidence = str(evidence.get("confidence") or "").lower()
    if confidence == "high":
        score += 15
        signals.append("high_confidence")
    elif confidence == "medium":
        score += 10
        signals.append("medium_confidence")
    elif confidence == "low":
        score += 3
        signals.append("low_confidence")
    if _as_float(evidence.get("total_meters_drilled")):
        score += 20
        signals.append("meters")
    if _as_float(evidence.get("total_holes")):
        score += 10
        signals.append("holes")
    if _as_float(evidence.get("weighted_grade_g_t")) or _as_float(evidence.get("average_intercept_grade_g_t")):
        score += 20
        signals.append("grade_proxy")
    if any(
        _as_float(evidence.get(k))
        for k in ("strike_length_m", "down_dip_extent_m", "avg_true_width_m", "drilled_area_km2")
    ):
        score += 20
        signals.append("geometry")
    if evidence.get("best_intercepts"):
        score += 5
        signals.append("intercepts")

    score = min(100, score)
    if score >= 70:
        grade = "high"
    elif score >= 40:
        grade = "medium"
    elif score > 0:
        grade = "low"
    else:
        grade = "none"
    return {"score": score, "grade": grade, "signals": signals}


def extract_local_guards(model: Optional[Dict[str, Any]]) -> List[str]:
    methodology = (model or {}).get("methodology") or {}
    notes = str(methodology.get("notes") or "")
    return re.findall(r"local_guard=([a-zA-Z0-9_]+)", notes)


def classify_failure(
    *,
    errors: Dict[str, Optional[float]],
    project: Optional[Dict[str, Any]] = None,
    model: Optional[Dict[str, Any]] = None,
    evidence_score: Optional[Dict[str, Any]] = None,
    threshold: float = 0.05,
    leak_detected: bool = False,
) -> Dict[str, Any]:
    """Classify the dominant miss mode and reusable lesson.

    `errors` are fractions, e.g. +0.20 means predicted 20% too high.
    """
    if leak_detected:
        return {
            "class": "blind_leakage",
            "lesson": "Reject any blind output that references target MRE/resource-anchor language.",
            "severity": "critical",
        }

    core = [errors.get(k) for k in ("tonnage", "grade", "contained")]
    if all(v is not None and not math.isinf(v) and abs(v) <= threshold for v in core):
        return {
            "class": "core_pass",
            "lesson": "Current blind routing matched the MRE holdout within threshold.",
            "severity": "none",
        }

    tonnage = errors.get("tonnage")
    grade = errors.get("grade")
    contained = errors.get("contained")
    project = project or {}
    ev_grade = (evidence_score or {}).get("grade") or "none"
    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    mining = str(project.get("mining_method_class") or project.get("mining_method") or "").lower()
    pattern = str(project.get("mineralization_pattern") or "").lower()
    guards = extract_local_guards(model)

    if tonnage is not None and tonnage <= -threshold and grade is not None and abs(grade) <= threshold:
        if "irgs" in subtype or "intrusion" in subtype:
            klass = "under_tonnage_large_low_grade_irgs"
            lesson = "Large low-grade IRGS/bulk targets need pre-MRE scale evidence or a large-system analog prior."
        elif "orogenic" in subtype or "vein" in pattern:
            klass = "under_tonnage_orogenic_scale_floor"
            lesson = "Orogenic targets with good grade proxy but sparse geometry need a district-scale tonnage floor."
        else:
            klass = "under_tonnage_grade_ok"
            lesson = "Grade proxy is acceptable; improve target scale evidence or tonnage prior."
        return {"class": klass, "lesson": lesson, "severity": "high", "guards": guards, "evidence": ev_grade}

    if tonnage is not None and tonnage >= threshold and grade is not None and abs(grade) <= threshold:
        return {
            "class": "over_tonnage_grade_ok",
            "lesson": "Analog cohort scale is too mature or too broad; tighten scale/stage filters before modelling.",
            "severity": "high",
            "guards": guards,
            "evidence": ev_grade,
        }

    if grade is not None and abs(grade) > threshold and contained is not None and abs(contained) <= threshold:
        return {
            "class": "grade_tonnage_tradeoff",
            "lesson": "Contained metal is close but grade/tonnage decomposition is wrong; prefer target grade proxy when source-backed.",
            "severity": "medium",
            "guards": guards,
            "evidence": ev_grade,
        }

    if "open_pit_selective" in mining and grade is not None and grade > threshold:
        return {
            "class": "open_pit_selective_grade_inflation",
            "lesson": "Do not let underground/high-grade vein analogs dominate low-grade open-pit-selective targets.",
            "severity": "high",
            "guards": guards,
            "evidence": ev_grade,
        }

    return {
        "class": "mixed_residual",
        "lesson": "Miss needs project-specific evidence enrichment and analog audit.",
        "severity": "medium",
        "guards": guards,
        "evidence": ev_grade,
    }


def leaderboard_row(
    *,
    project_name: str,
    errors: Dict[str, Optional[float]],
    passed: bool,
    failure: Dict[str, Any],
    guards: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "project": project_name,
        "pass": passed,
        "tonnage_error_pct": None if errors.get("tonnage") is None else round(errors["tonnage"] * 100, 3),
        "grade_error_pct": None if errors.get("grade") is None else round(errors["grade"] * 100, 3),
        "contained_error_pct": None if errors.get("contained") is None else round(errors["contained"] * 100, 3),
        "failure_class": failure.get("class"),
        "lesson": failure.get("lesson"),
        "local_guards": guards or [],
    }
