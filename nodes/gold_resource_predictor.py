"""Deterministic gold-only resource predictor.

This module is the hard-edged calculator for the v2 gold rebuild:

* no target MRE leakage
* no post-cutoff evidence
* no weak analog fallback
* no estimate when the evidence package cannot support one

Parallel can still be used upstream to discover truth, evidence, and analog
rows. This module only consumes already-stored facts and produces either a
prediction or `no_prediction` with explicit reasons.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from nodes import geo_taxonomy
from schemas.gold_resource_predictor import (
    GoldAnalogCandidate,
    GoldEvidenceFact,
    GoldMreTruthRecord,
    GoldProjectRecord,
)


PREDICTOR_VERSION = "gold_resource_predictor_v2.0"
TROY_OZ_PER_MT_GPT = 32150.7466
MIN_CLEAN_ANALOGS = 3
MIN_SPLIT_ANALOGS = 3
EVIDENCE_WEIGHT = 0.80
ANALOG_WEIGHT = 0.20

_TAINT_RE = re.compile(
    r"\b("
    r"mre|mineral resource|resource estimate|technical report|ni[- ]?43[- ]?101|"
    r"jorc|sk[- ]?1300|measured and indicated|inferred resource"
    r")\b",
    re.IGNORECASE,
)
_PRE_MRE_DRILLING_CONTEXT_RE = re.compile(
    r"\b("
    r"ahead of|prior to|before|pre[- ]?resource|pre[- ]?mre|towards?|supporting"
    r")\b.{0,80}\b(maiden\s+)?(mineral\s+)?resource\b|"
    r"\bresource drilling\b",
    re.IGNORECASE,
)
_PRE_MRE_EXPLORATION_TARGET_RE = re.compile(
    r"\b(exploration target|resource definition)\b",
    re.IGNORECASE,
)
_HARD_RESOURCE_DISCLOSURE_RE = re.compile(
    r"\b("
    r"technical report|resource estimate|mineral resource estimate|"
    r"measured and indicated|inferred resource|ni[- ]?43[- ]?101|jorc|sk[- ]?1300"
    r")\b",
    re.IGNORECASE,
)

_LOW_QUALITY_RESOURCE_STANDARDS = {
    "historical",
    "internal",
    "exploration_target",
}

_GRADE_FACT_TYPES = (
    "weighted_grade_gpt",
    "grade_proxy_gpt",
    "average_intercept_grade_gpt",
    "head_grade_gpt",
)

_DIRECT_TONNAGE_FACT_TYPES = (
    "geometry_tonnage_mt",
    "tonnage_proxy_mt",
    "tailings_inventory_tonnage_mt",
)

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def contained_gold_oz(tonnage_mt: Optional[float], grade_gpt: Optional[float]) -> Optional[float]:
    if tonnage_mt is None or grade_gpt is None:
        return None
    if tonnage_mt < 0 or grade_gpt < 0:
        return None
    return tonnage_mt * grade_gpt * TROY_OZ_PER_MT_GPT


def pct_error(predicted: Optional[float], actual: Optional[float]) -> Optional[float]:
    if predicted is None or actual is None:
        return None
    if actual == 0:
        return math.inf if predicted != 0 else 0.0
    return (predicted - actual) / actual


def make_input_hash(payload: Dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def tonnage_band(tonnage_mt: Optional[float]) -> Optional[str]:
    if tonnage_mt is None or tonnage_mt <= 0:
        return None
    if tonnage_mt < 1:
        return "sub_1mt"
    if tonnage_mt < 5:
        return "1_5mt"
    if tonnage_mt < 25:
        return "5_25mt"
    if tonnage_mt < 100:
        return "25_100mt"
    if tonnage_mt < 300:
        return "100_300mt"
    return "300mt_plus"


def grade_band(grade_gpt: Optional[float]) -> Optional[str]:
    if grade_gpt is None or grade_gpt <= 0:
        return None
    if grade_gpt < 0.5:
        return "sub_0_5gpt"
    if grade_gpt < 1.0:
        return "0_5_1gpt"
    if grade_gpt < 2.0:
        return "1_2gpt"
    if grade_gpt < 5.0:
        return "2_5gpt"
    return "5gpt_plus"


def _as_project(project: GoldProjectRecord | Dict[str, Any]) -> GoldProjectRecord:
    return project if isinstance(project, GoldProjectRecord) else GoldProjectRecord(**project)


def _as_evidence(fact: GoldEvidenceFact | Dict[str, Any]) -> GoldEvidenceFact:
    return fact if isinstance(fact, GoldEvidenceFact) else GoldEvidenceFact(**fact)


def _as_analog(candidate: GoldAnalogCandidate | Dict[str, Any]) -> GoldAnalogCandidate:
    return candidate if isinstance(candidate, GoldAnalogCandidate) else GoldAnalogCandidate(**candidate)


def _as_truth(truth: GoldMreTruthRecord | Dict[str, Any]) -> GoldMreTruthRecord:
    return truth if isinstance(truth, GoldMreTruthRecord) else GoldMreTruthRecord(**truth)


def _compact_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json", exclude_none=True)
    return dict(model)


def _source_blob(*parts: Optional[str]) -> str:
    return " ".join(part or "" for part in parts).strip()


def _has_hard_resource_disclosure(blob: str) -> bool:
    if _PRE_MRE_EXPLORATION_TARGET_RE.search(blob):
        blob = re.sub(r"\bjorc\b", "", blob, flags=re.IGNORECASE)
    return bool(_HARD_RESOURCE_DISCLOSURE_RE.search(blob))


def evidence_is_mre_tainted(fact: GoldEvidenceFact | Dict[str, Any]) -> bool:
    fact_model = _as_evidence(fact)
    if fact_model.is_mre_tainted:
        return True
    title_blob = _source_blob(
        fact_model.source_url,
        fact_model.source_title,
        fact_model.source_document_type,
    )
    normalized_title_blob = re.sub(r"[-_/]+", " ", title_blob)
    if (
        fact_model.source_date is not None
        and fact_model.source_date < fact_model.cutoff_date
        and (
            _PRE_MRE_DRILLING_CONTEXT_RE.search(normalized_title_blob)
            or _PRE_MRE_EXPLORATION_TARGET_RE.search(normalized_title_blob)
        )
        and not _has_hard_resource_disclosure(normalized_title_blob)
    ):
        return False
    blob = _source_blob(
        fact_model.source_url,
        fact_model.source_title,
        fact_model.source_document_type,
        fact_model.value_text,
        json.dumps(fact_model.fact_payload, sort_keys=True, default=str),
    )
    blob = re.sub(r"[-_/]+", " ", blob)
    return bool(_TAINT_RE.search(blob))


def validate_pre_mre_evidence(
    fact: GoldEvidenceFact | Dict[str, Any],
) -> Tuple[bool, List[str]]:
    fact_model = _as_evidence(fact)
    reasons: List[str] = []

    if fact_model.evidence_status == "rejected":
        reasons.append(fact_model.rejection_reason or "evidence_row_already_rejected")
    if fact_model.source_date is None:
        reasons.append("missing_source_date")
    elif fact_model.source_date >= fact_model.cutoff_date:
        reasons.append("source_not_before_mre_cutoff")
    if evidence_is_mre_tainted(fact_model):
        reasons.append("mre_tainted_source")
    if fact_model.value_num is None and not fact_model.value_text:
        reasons.append("empty_evidence_value")
    if fact_model.confidence == "low" and fact_model.fact_type not in {
        "total_drill_meters",
        "drill_holes",
        "weighted_grade_gpt",
        "grade_proxy_gpt",
        "geometry_tonnage_mt",
        "tailings_inventory_tonnage_mt",
    }:
        reasons.append("low_confidence_weak_fact")

    return not reasons, reasons


def split_evidence(
    evidence_facts: Iterable[GoldEvidenceFact | Dict[str, Any]],
) -> Tuple[List[GoldEvidenceFact], List[Dict[str, Any]]]:
    accepted: List[GoldEvidenceFact] = []
    rejected: List[Dict[str, Any]] = []
    for raw in evidence_facts:
        fact = _as_evidence(raw)
        ok, reasons = validate_pre_mre_evidence(fact)
        if ok:
            accepted.append(fact)
        else:
            rejected.append({
                "fact_type": fact.fact_type,
                "source_url": fact.source_url,
                "source_date": fact.source_date.isoformat() if fact.source_date else None,
                "reasons": reasons,
            })
    return accepted, rejected


def _best_numeric_fact(facts: Sequence[GoldEvidenceFact], fact_types: Sequence[str]) -> Optional[GoldEvidenceFact]:
    candidates = [
        fact for fact in facts
        if fact.fact_type in fact_types and fact.value_num is not None and fact.value_num > 0
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda fact: (
            _CONFIDENCE_RANK.get(fact.confidence, 0),
            fact.source_date or date.min,
            fact.value_num or 0,
        ),
        reverse=True,
    )[0]


def _numeric_fact_value(facts: Sequence[GoldEvidenceFact], fact_type: str) -> Optional[float]:
    fact = _best_numeric_fact(facts, (fact_type,))
    return float(fact.value_num) if fact and fact.value_num is not None else None


def evidence_tonnage_proxy(facts: Sequence[GoldEvidenceFact]) -> Tuple[Optional[float], Dict[str, Any]]:
    direct = _best_numeric_fact(facts, _DIRECT_TONNAGE_FACT_TYPES)
    if direct and direct.value_num is not None:
        return float(direct.value_num), {
            "method": "direct_tonnage_fact",
            "fact_type": direct.fact_type,
            "source_url": direct.source_url,
        }

    strike_m = _numeric_fact_value(facts, "strike_length_m")
    depth_m = _numeric_fact_value(facts, "down_dip_extent_m")
    width_m = _numeric_fact_value(facts, "avg_true_width_m")
    density_t_m3 = _numeric_fact_value(facts, "bulk_density_t_m3")
    continuity_factor = _numeric_fact_value(facts, "mineralized_continuity_factor")
    if all(v is not None and v > 0 for v in (strike_m, depth_m, width_m, density_t_m3, continuity_factor)):
        continuity_factor = min(float(continuity_factor), 1.0)
        tonnage_mt = (
            float(strike_m)
            * float(depth_m)
            * float(width_m)
            * float(density_t_m3)
            * continuity_factor
            / 1_000_000.0
        )
        return tonnage_mt, {
            "method": "strict_geometry_volume",
            "strike_length_m": strike_m,
            "down_dip_extent_m": depth_m,
            "avg_true_width_m": width_m,
            "bulk_density_t_m3": density_t_m3,
            "mineralized_continuity_factor": continuity_factor,
        }

    return None, {
        "method": "no_tonnage_proxy",
        "missing_required_fact_types": [
            name for name, value in (
                ("geometry_tonnage_mt", None),
                ("strike_length_m", strike_m),
                ("down_dip_extent_m", depth_m),
                ("avg_true_width_m", width_m),
                ("bulk_density_t_m3", density_t_m3),
                ("mineralized_continuity_factor", continuity_factor),
            )
            if value is None
        ],
    }


def evidence_grade_proxy(facts: Sequence[GoldEvidenceFact]) -> Tuple[Optional[float], Dict[str, Any]]:
    fact = _best_numeric_fact(facts, _GRADE_FACT_TYPES)
    if fact and fact.value_num is not None:
        return float(fact.value_num), {
            "method": "grade_fact",
            "fact_type": fact.fact_type,
            "source_url": fact.source_url,
        }
    return None, {
        "method": "no_grade_proxy",
        "missing_required_fact_types": list(_GRADE_FACT_TYPES),
    }


def _resource_standard_ok(standard: Optional[str]) -> bool:
    if not standard:
        return False
    return standard.strip().lower() not in _LOW_QUALITY_RESOURCE_STANDARDS


def _same_or_compatible_belt(target: str, candidate: str) -> bool:
    return target == candidate or geo_taxonomy.belt_compatible(target, candidate)


def gate_gold_analog(
    project: GoldProjectRecord | Dict[str, Any],
    candidate: GoldAnalogCandidate | Dict[str, Any],
    *,
    cutoff_date: date,
    target_tonnage_mt: float,
    target_grade_gpt: float,
) -> Dict[str, Any]:
    project_model = _as_project(project)
    analog = _as_analog(candidate)
    reasons: List[str] = []
    rules: List[Dict[str, Any]] = []

    def require(rule: str, passed: bool, reason: str) -> None:
        rules.append({"rule": rule, "passed": passed, "reason": reason})
        if not passed:
            reasons.append(reason)

    require("source_url", bool(analog.source_url), "missing_source_url")
    require("source_date", analog.source_date is not None, "missing_source_date")
    if analog.source_date is not None:
        require("pre_target_cutoff", analog.source_date < cutoff_date, "analog_source_not_before_target_cutoff")

    require("resource_standard", _resource_standard_ok(analog.resource_standard), "weak_or_missing_resource_standard")
    require(
        "resource_numbers",
        bool(analog.total_tonnage_mt and analog.total_tonnage_mt > 0 and analog.total_grade_gpt and analog.total_grade_gpt > 0),
        "missing_resource_tonnage_or_grade",
    )

    require("target_subtype_present", bool(project_model.deposit_subtype), "target_missing_deposit_subtype")
    require("candidate_subtype_present", bool(analog.candidate_deposit_subtype), "analog_missing_deposit_subtype")
    if project_model.deposit_subtype and analog.candidate_deposit_subtype:
        require(
            "deposit_subtype_exact",
            project_model.deposit_subtype == analog.candidate_deposit_subtype,
            "deposit_subtype_mismatch",
        )

    require("target_belt_present", bool(project_model.tectonic_belt), "target_missing_tectonic_belt")
    require("candidate_belt_present", bool(analog.candidate_tectonic_belt), "analog_missing_tectonic_belt")
    if project_model.tectonic_belt and analog.candidate_tectonic_belt:
        require(
            "tectonic_belt_compatible",
            _same_or_compatible_belt(project_model.tectonic_belt, analog.candidate_tectonic_belt),
            "tectonic_belt_incompatible",
        )

    require("target_mining_method_present", bool(project_model.mining_method_class), "target_missing_mining_method")
    require("candidate_mining_method_present", bool(analog.candidate_mining_method_class), "analog_missing_mining_method")
    if project_model.mining_method_class and analog.candidate_mining_method_class:
        require(
            "mining_method_compatible",
            geo_taxonomy.mining_method_compatible(
                project_model.mining_method_class,
                analog.candidate_mining_method_class,
            ),
            "mining_method_incompatible",
        )

    if project_model.project_stage_class and analog.candidate_project_stage_class:
        require(
            "stage_compatible",
            geo_taxonomy.stage_compatible(
                project_model.project_stage_class,
                analog.candidate_project_stage_class,
            ),
            "project_stage_incompatible",
        )
    else:
        require("stage_metadata_present", False, "missing_project_stage_class")

    candidate_tonnage_band = tonnage_band(analog.total_tonnage_mt)
    target_tonnage_band = tonnage_band(target_tonnage_mt)
    require(
        "tonnage_band_compatible",
        candidate_tonnage_band == target_tonnage_band,
        "tonnage_band_mismatch",
    )

    candidate_grade_band = grade_band(analog.total_grade_gpt)
    target_grade_band = grade_band(target_grade_gpt)
    require(
        "grade_band_compatible",
        candidate_grade_band == target_grade_band,
        "grade_band_mismatch",
    )

    return {
        "candidate_project_name": analog.candidate_project_name,
        "analog_candidate_id": analog.id,
        "decision": "accepted" if not reasons else "rejected",
        "rejection_reasons": reasons,
        "decision_rules": rules,
        "tonnage_band": candidate_tonnage_band,
        "grade_band": candidate_grade_band,
    }


def clean_analog_cohort(
    project: GoldProjectRecord | Dict[str, Any],
    analog_candidates: Iterable[GoldAnalogCandidate | Dict[str, Any]],
    *,
    cutoff_date: date,
    target_tonnage_mt: float,
    target_grade_gpt: float,
) -> Tuple[List[GoldAnalogCandidate], List[Dict[str, Any]]]:
    accepted: List[GoldAnalogCandidate] = []
    decisions: List[Dict[str, Any]] = []
    for raw in analog_candidates:
        analog = _as_analog(raw)
        decision = gate_gold_analog(
            project,
            analog,
            cutoff_date=cutoff_date,
            target_tonnage_mt=target_tonnage_mt,
            target_grade_gpt=target_grade_gpt,
        )
        decisions.append(decision)
        if decision["decision"] == "accepted":
            accepted.append(analog)
    return accepted, decisions


def _geometric_blend(evidence_value: float, analog_value: float) -> float:
    return math.exp(
        EVIDENCE_WEIGHT * math.log(evidence_value)
        + ANALOG_WEIGHT * math.log(analog_value)
    )


def _positive_values(values: Iterable[Optional[float]]) -> List[float]:
    return [float(v) for v in values if v is not None and v > 0]


def _split_ready_analogs(analogs: Sequence[GoldAnalogCandidate]) -> List[GoldAnalogCandidate]:
    ready = []
    for analog in analogs:
        if all(
            value is not None and value > 0
            for value in (
                analog.mi_tonnage_mt,
                analog.mi_grade_gpt,
                analog.inferred_tonnage_mt,
                analog.inferred_grade_gpt,
            )
        ):
            total = (analog.mi_tonnage_mt or 0) + (analog.inferred_tonnage_mt or 0)
            if total > 0:
                ready.append(analog)
    return ready


def _no_prediction(
    *,
    reasons: List[str],
    project: GoldProjectRecord,
    cutoff_date: date,
    evidence_facts: Sequence[GoldEvidenceFact],
    analog_candidates: Sequence[GoldAnalogCandidate],
    evidence_rejections: Sequence[Dict[str, Any]],
    analog_decisions: Sequence[Dict[str, Any]],
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "project": _compact_model(project),
        "cutoff_date": cutoff_date.isoformat(),
        "evidence": [_compact_model(fact) for fact in evidence_facts],
        "analogs": [_compact_model(analog) for analog in analog_candidates],
        "predictor_version": PREDICTOR_VERSION,
    }
    return {
        "run_status": "no_prediction",
        "predictor_version": PREDICTOR_VERSION,
        "input_hash": make_input_hash(payload),
        "cutoff_date": cutoff_date.isoformat(),
        "no_prediction_reasons": sorted(set(reasons)),
        "evidence_rejections": list(evidence_rejections),
        "analog_decisions": list(analog_decisions),
        "calculator_trace": trace,
    }


def predict_gold_resource(
    project: GoldProjectRecord | Dict[str, Any],
    evidence_facts: Iterable[GoldEvidenceFact | Dict[str, Any]],
    analog_candidates: Iterable[GoldAnalogCandidate | Dict[str, Any]],
    *,
    cutoff_date: date,
) -> Dict[str, Any]:
    project_model = _as_project(project)
    all_evidence = [_as_evidence(fact) for fact in evidence_facts]
    all_analogs = [_as_analog(analog) for analog in analog_candidates]

    accepted_evidence, evidence_rejections = split_evidence(all_evidence)
    tonnage_mt, tonnage_trace = evidence_tonnage_proxy(accepted_evidence)
    grade_gpt, grade_trace = evidence_grade_proxy(accepted_evidence)
    trace: Dict[str, Any] = {
        "evidence_tonnage": tonnage_trace,
        "evidence_grade": grade_trace,
        "accepted_evidence_count": len(accepted_evidence),
        "rejected_evidence_count": len(evidence_rejections),
        "accepted_evidence": [
            {
                "id": fact.id,
                "fact_type": fact.fact_type,
                "source_url": fact.source_url,
                "source_date": fact.source_date.isoformat() if fact.source_date else None,
            }
            for fact in accepted_evidence
        ],
    }

    early_reasons: List[str] = []
    if tonnage_mt is None:
        early_reasons.append("insufficient_pre_mre_tonnage_evidence")
    if grade_gpt is None:
        early_reasons.append("insufficient_pre_mre_grade_evidence")
    if early_reasons:
        return _no_prediction(
            reasons=early_reasons,
            project=project_model,
            cutoff_date=cutoff_date,
            evidence_facts=all_evidence,
            analog_candidates=all_analogs,
            evidence_rejections=evidence_rejections,
            analog_decisions=[],
            trace=trace,
        )

    assert tonnage_mt is not None
    assert grade_gpt is not None

    accepted_analogs, analog_decisions = clean_analog_cohort(
        project_model,
        all_analogs,
        cutoff_date=cutoff_date,
        target_tonnage_mt=tonnage_mt,
        target_grade_gpt=grade_gpt,
    )
    trace["accepted_analog_count"] = len(accepted_analogs)
    trace["rejected_analog_count"] = len(analog_decisions) - len(accepted_analogs)

    reasons = []
    if len(accepted_analogs) < MIN_CLEAN_ANALOGS:
        reasons.append("insufficient_clean_analog_cohort")

    split_ready = _split_ready_analogs(accepted_analogs)
    trace["split_ready_analog_count"] = len(split_ready)
    if len(split_ready) < MIN_SPLIT_ANALOGS:
        reasons.append("insufficient_split_ready_analog_cohort")

    if reasons:
        return _no_prediction(
            reasons=reasons,
            project=project_model,
            cutoff_date=cutoff_date,
            evidence_facts=all_evidence,
            analog_candidates=all_analogs,
            evidence_rejections=evidence_rejections,
            analog_decisions=analog_decisions,
            trace=trace,
        )

    analog_tonnage = median(_positive_values(analog.total_tonnage_mt for analog in accepted_analogs))
    analog_grade = median(_positive_values(analog.total_grade_gpt for analog in accepted_analogs))
    predicted_tonnage = _geometric_blend(tonnage_mt, analog_tonnage)
    predicted_grade = _geometric_blend(grade_gpt, analog_grade)
    predicted_contained = contained_gold_oz(predicted_tonnage, predicted_grade)

    split_ratios = [
        float(analog.mi_tonnage_mt) / (float(analog.mi_tonnage_mt) + float(analog.inferred_tonnage_mt))
        for analog in split_ready
    ]
    mi_ratio = median(split_ratios)
    inferred_ratio = 1.0 - mi_ratio

    mi_grade_factors = [
        float(analog.mi_grade_gpt) / float(analog.total_grade_gpt)
        for analog in split_ready
        if analog.total_grade_gpt and analog.total_grade_gpt > 0
    ]
    inferred_grade_factors = [
        float(analog.inferred_grade_gpt) / float(analog.total_grade_gpt)
        for analog in split_ready
        if analog.total_grade_gpt and analog.total_grade_gpt > 0
    ]
    mi_grade_factor = median(mi_grade_factors)
    inferred_grade_factor = median(inferred_grade_factors)

    trace.update({
        "analog_branch": {
            "median_tonnage_mt": analog_tonnage,
            "median_grade_gpt": analog_grade,
        },
        "blend": {
            "evidence_weight": EVIDENCE_WEIGHT,
            "analog_weight": ANALOG_WEIGHT,
            "method": "log_space_geometric_blend",
        },
        "split_model": {
            "mi_tonnage_ratio": mi_ratio,
            "inferred_tonnage_ratio": inferred_ratio,
            "mi_grade_factor": mi_grade_factor,
            "inferred_grade_factor": inferred_grade_factor,
        },
    })

    payload = {
        "project": _compact_model(project_model),
        "cutoff_date": cutoff_date.isoformat(),
        "evidence": [_compact_model(fact) for fact in all_evidence],
        "analogs": [_compact_model(analog) for analog in all_analogs],
        "predictor_version": PREDICTOR_VERSION,
    }
    return {
        "run_status": "predicted",
        "predictor_version": PREDICTOR_VERSION,
        "input_hash": make_input_hash(payload),
        "cutoff_date": cutoff_date.isoformat(),
        "predicted_total_tonnage_mt": predicted_tonnage,
        "predicted_total_grade_gpt": predicted_grade,
        "predicted_total_contained_oz": predicted_contained,
        "predicted_mi_tonnage_mt": predicted_tonnage * mi_ratio,
        "predicted_mi_grade_gpt": predicted_grade * mi_grade_factor,
        "predicted_inferred_tonnage_mt": predicted_tonnage * inferred_ratio,
        "predicted_inferred_grade_gpt": predicted_grade * inferred_grade_factor,
        "no_prediction_reasons": [],
        "evidence_rejections": evidence_rejections,
        "analog_decisions": analog_decisions,
        "calculator_trace": trace,
    }


def _truth_total_tonnage(truth: GoldMreTruthRecord) -> Optional[float]:
    if truth.total_tonnage_mt is not None:
        return truth.total_tonnage_mt
    if truth.mi_tonnage_mt is None and truth.inferred_tonnage_mt is None:
        return None
    return float(truth.mi_tonnage_mt or 0) + float(truth.inferred_tonnage_mt or 0)


def _truth_total_grade(truth: GoldMreTruthRecord) -> Optional[float]:
    if truth.total_grade_gpt is not None:
        return truth.total_grade_gpt
    total_tonnage = _truth_total_tonnage(truth)
    if not total_tonnage or total_tonnage <= 0:
        return None
    mi_tonnage = float(truth.mi_tonnage_mt or 0)
    inferred_tonnage = float(truth.inferred_tonnage_mt or 0)
    if truth.mi_grade_gpt is None and truth.inferred_grade_gpt is None:
        return None
    return (
        mi_tonnage * float(truth.mi_grade_gpt or 0)
        + inferred_tonnage * float(truth.inferred_grade_gpt or 0)
    ) / total_tonnage


def _truth_total_contained(truth: GoldMreTruthRecord) -> Optional[float]:
    if truth.total_contained_oz is not None:
        return truth.total_contained_oz
    return contained_gold_oz(_truth_total_tonnage(truth), _truth_total_grade(truth))


def _within_threshold(value: Optional[float], threshold: float) -> bool:
    return value is not None and not math.isinf(value) and abs(value) <= threshold


def score_gold_prediction(
    prediction: Dict[str, Any],
    truth: GoldMreTruthRecord | Dict[str, Any],
    *,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    truth_model = _as_truth(truth)
    if prediction.get("run_status") != "predicted":
        return {
            "core_pass": False,
            "split_pass": False,
            "production_like_pass": False,
            "failure_class": "no_prediction",
            "failure_reason": ", ".join(prediction.get("no_prediction_reasons") or ["no_prediction"]),
            "score_payload": {"prediction_status": prediction.get("run_status")},
        }

    errors = {
        "tonnage": pct_error(prediction.get("predicted_total_tonnage_mt"), _truth_total_tonnage(truth_model)),
        "grade": pct_error(prediction.get("predicted_total_grade_gpt"), _truth_total_grade(truth_model)),
        "contained": pct_error(prediction.get("predicted_total_contained_oz"), _truth_total_contained(truth_model)),
        "mi_tonnage": pct_error(prediction.get("predicted_mi_tonnage_mt"), truth_model.mi_tonnage_mt),
        "mi_grade": pct_error(prediction.get("predicted_mi_grade_gpt"), truth_model.mi_grade_gpt),
        "inferred_tonnage": pct_error(prediction.get("predicted_inferred_tonnage_mt"), truth_model.inferred_tonnage_mt),
        "inferred_grade": pct_error(prediction.get("predicted_inferred_grade_gpt"), truth_model.inferred_grade_gpt),
    }
    core_pass = all(_within_threshold(errors[key], threshold) for key in ("tonnage", "grade", "contained"))
    split_pass = all(
        _within_threshold(errors[key], threshold)
        for key in ("mi_tonnage", "mi_grade", "inferred_tonnage", "inferred_grade")
    )
    production_like_pass = core_pass and split_pass

    failure_class = None
    failure_reason = None
    if not core_pass:
        worst_key = max(
            ("tonnage", "grade", "contained"),
            key=lambda key: abs(errors[key]) if errors[key] is not None and not math.isinf(errors[key]) else math.inf,
        )
        failure_class = f"{worst_key}_outside_threshold"
        failure_reason = f"{worst_key} error exceeded +/-{threshold * 100:.1f}%"
    elif not split_pass:
        failure_class = "split_outside_threshold"
        failure_reason = f"M&I or inferred split exceeded +/-{threshold * 100:.1f}%"

    return {
        "core_pass": core_pass,
        "split_pass": split_pass,
        "production_like_pass": production_like_pass,
        "threshold_pct": threshold * 100,
        "tonnage_error_pct": None if errors["tonnage"] is None else errors["tonnage"] * 100,
        "grade_error_pct": None if errors["grade"] is None else errors["grade"] * 100,
        "contained_error_pct": None if errors["contained"] is None else errors["contained"] * 100,
        "mi_tonnage_error_pct": None if errors["mi_tonnage"] is None else errors["mi_tonnage"] * 100,
        "mi_grade_error_pct": None if errors["mi_grade"] is None else errors["mi_grade"] * 100,
        "inferred_tonnage_error_pct": None if errors["inferred_tonnage"] is None else errors["inferred_tonnage"] * 100,
        "inferred_grade_error_pct": None if errors["inferred_grade"] is None else errors["inferred_grade"] * 100,
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "score_payload": {"errors": errors},
    }
