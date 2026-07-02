"""Shared diagnostics for blind gold Parallel backtests.

These helpers are intentionally deterministic and source-light: they never
fetch new data and they never inspect target MRE fields except when comparing a
finished prediction against the held-out truth supplied by the caller.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


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
    if _as_float(evidence.get("tailings_inventory_tonnage_mt")) or (
        _as_float(evidence.get("tailings_inventory_min_mt"))
        and _as_float(evidence.get("tailings_inventory_max_mt"))
    ):
        score += 20
        signals.append("tailings_inventory")
    if _as_float(evidence.get("tailings_grade_g_t")):
        score += 10
        signals.append("tailings_grade")
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


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _finite_positive(value: Any) -> Optional[float]:
    parsed = _as_float(value)
    return parsed if parsed and parsed > 0 else None


def _same_or_unknown(a: Any, b: Any) -> Optional[bool]:
    left = _norm(a)
    right = _norm(b)
    if not left or not right:
        return None
    return left == right


def _ratio_within(a: Any, b: Any, max_ratio: float) -> Optional[bool]:
    left = _finite_positive(a)
    right = _finite_positive(b)
    if not left or not right:
        return None
    return max(left, right) / min(left, right) <= max_ratio


def _tonnage_band(value: Any) -> Optional[str]:
    tonnage = _finite_positive(value)
    if not tonnage:
        return None
    if tonnage < 1:
        return "sub_1mt"
    if tonnage < 5:
        return "1_5mt"
    if tonnage < 25:
        return "5_25mt"
    if tonnage < 100:
        return "25_100mt"
    if tonnage < 300:
        return "100_300mt"
    return "300mt_plus"


def _project_grade_proxy(project: Dict[str, Any]) -> Optional[float]:
    direct = _finite_positive(project.get("pre_mre_grade_proxy") or project.get("blind_grade_proxy"))
    if direct:
        return direct
    evidence = project.get("drilling_evidence")
    if isinstance(evidence, dict) and not evidence.get("redacted"):
        direct = _finite_positive(
            evidence.get("weighted_grade_g_t")
            or evidence.get("average_intercept_grade_g_t")
        )
        if direct:
            return direct
        intercept_grades = [
            _finite_positive(item.get("grade_g_t") or item.get("grade_gpt"))
            for item in (evidence.get("best_intercepts") or [])
            if isinstance(item, dict)
        ]
        intercept_grades = [grade for grade in intercept_grades if grade]
        if intercept_grades:
            intercept_grades.sort()
            mid = len(intercept_grades) // 2
            median = intercept_grades[mid] if len(intercept_grades) % 2 else (
                intercept_grades[mid - 1] + intercept_grades[mid]
            ) / 2.0
            return median * 0.5
    return _finite_positive(project.get("grade_value"))


def _project_tonnage_proxy(project: Dict[str, Any]) -> Optional[float]:
    return _finite_positive(project.get("pre_mre_tonnage_proxy") or project.get("blind_tonnage_proxy") or project.get("tonnage_mt"))


def analog_quality_score(
    *,
    project: Optional[Dict[str, Any]],
    analogs: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Grade the supplied blind analog cohort for audit/release gating.

    This deliberately stays deterministic and source-light. It does not decide
    the model output; it tells the backtest artifact whether a numeric pass was
    supported by a coherent analog cohort or mostly rescued by guardrails.
    """
    project = project or {}
    rows = [row for row in (analogs or []) if isinstance(row, dict)]
    if not rows:
        return {
            "grade": "reject",
            "score": 0,
            "analog_count": 0,
            "source_backed_count": 0,
            "flags": ["no_analogs"],
            "metrics": {},
        }

    score = 0
    flags: List[str] = []
    metrics: Dict[str, Any] = {}

    count = len(rows)
    source_backed = sum(1 for row in rows if row.get("source_url") or row.get("data_source"))
    resource_backed = sum(
        1
        for row in rows
        if _finite_positive(row.get("tonnage_mt")) and _finite_positive(row.get("grade_value"))
    )
    subtype_checks = [
        _same_or_unknown(project.get("deposit_subtype"), row.get("deposit_subtype") or row.get("analog_deposit_subtype"))
        for row in rows
    ]
    belt_checks = [
        _same_or_unknown(project.get("tectonic_belt"), row.get("tectonic_belt") or row.get("analog_tectonic_belt"))
        for row in rows
    ]
    mining_checks = [
        _same_or_unknown(project.get("mining_method_class") or project.get("mining_method"), row.get("mining_method_class") or row.get("mining_method"))
        for row in rows
    ]
    project_grade = _project_grade_proxy(project)
    project_tonnage = _project_tonnage_proxy(project)
    grade_band_checks = [
        _ratio_within(project_grade, row.get("grade_value"), 3.0)
        for row in rows
    ]
    tonnage_band_checks = [
        _ratio_within(project_tonnage, row.get("tonnage_mt"), 10.0)
        for row in rows
    ]
    analog_tonnages = [
        tonnage for tonnage in (_finite_positive(row.get("tonnage_mt")) for row in rows) if tonnage
    ]
    analog_tonnage_bands = [band for band in (_tonnage_band(value) for value in analog_tonnages) if band]

    def rate(values: List[Optional[bool]]) -> Optional[float]:
        evaluated = [value for value in values if value is not None]
        if not evaluated:
            return None
        return sum(1 for value in evaluated if value) / len(evaluated)

    rates = {
        "subtype_match_rate": rate(subtype_checks),
        "belt_match_rate": rate(belt_checks),
        "mining_method_match_rate": rate(mining_checks),
        "grade_band_match_rate": rate(grade_band_checks),
        "tonnage_band_match_rate": rate(tonnage_band_checks),
    }
    metrics.update(rates)
    if analog_tonnage_bands:
        metrics["analog_tonnage_band_counts"] = dict(Counter(analog_tonnage_bands))

    if count >= 5:
        score += 20
    elif count >= 3:
        score += 14
    else:
        score += 5
        flags.append("thin_analog_count")

    resource_rate = resource_backed / count
    source_rate = source_backed / count
    score += round(20 * resource_rate)
    score += round(15 * source_rate)
    if resource_rate < 0.75:
        flags.append("weak_resource_backing")
    if source_rate < 0.5:
        flags.append("weak_source_backing")

    unknown_core_flags = {
        "subtype_match_rate_unknown",
        "belt_match_rate_unknown",
        "mining_method_match_rate_unknown",
    }
    for key, value in rates.items():
        if value is None:
            flags.append(f"{key}_unknown")
            continue
        if value >= 0.75:
            score += 10
        elif value >= 0.5:
            score += 5
        else:
            flags.append(f"low_{key}")

    unknown_core_count = sum(1 for flag in flags if flag in unknown_core_flags)
    score -= 6 * unknown_core_count
    if len(set(analog_tonnage_bands)) >= 4:
        flags.append("broad_tonnage_band_cohort")
        score -= 8
    if len(analog_tonnages) >= 3 and max(analog_tonnages) / min(analog_tonnages) > 50:
        flags.append("extreme_tonnage_spread")
        score -= 10
    score = max(0, min(100, int(score)))
    if score >= 75 and not any(flag.startswith("low_") for flag in flags) and unknown_core_count == 0:
        grade = "high"
    elif score >= 55:
        grade = "medium"
    elif score >= 35:
        grade = "low"
    else:
        grade = "reject"

    return {
        "grade": grade,
        "score": score,
        "analog_count": count,
        "source_backed_count": source_backed,
        "resource_backed_count": resource_backed,
        "flags": flags,
        "metrics": metrics,
    }


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
