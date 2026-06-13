"""Typed records for the gold-only resource predictor.

These models describe the new `gold_*` database surface. They are deliberately
strict around the fields used by the deterministic calculator: a row can be
missing optional context, but a usable truth/evidence/analog fact must carry its
source chronology and enough numeric data to audit it.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


Confidence = Literal["high", "medium", "low"]
Decision = Literal["accepted", "rejected"]
EvidenceStatus = Literal["accepted", "rejected"]
RunStatus = Literal["predicted", "no_prediction", "failed"]


class GoldProjectRecord(BaseModel):
    id: Optional[str] = None
    external_key: Optional[str] = None
    company_name: Optional[str] = None
    project_name: str
    material: Literal["gold"] = "gold"
    country: Optional[str] = None
    region: Optional[str] = None
    district: Optional[str] = None
    deposit_family: Optional[str] = None
    deposit_subtype: Optional[str] = None
    tectonic_belt: Optional[str] = None
    mineralization_mode: Optional[str] = None
    mineralization_pattern: Optional[str] = None
    host_rock_class: Optional[str] = None
    mining_method_class: Optional[str] = None
    project_stage_class: Optional[str] = None
    recovery_method: Optional[str] = None
    source_payload: Dict[str, Any] = Field(default_factory=dict)


class GoldMreTruthRecord(BaseModel):
    id: Optional[str] = None
    project_id: str
    truth_status: Literal["validated", "uncertain", "rejected"] = "validated"
    effective_date: Optional[date] = None
    publication_date: date
    cutoff_date: Optional[date] = None
    source_url: str
    source_title: Optional[str] = None
    source_publisher: Optional[str] = None
    resource_standard: Optional[str] = None
    mi_tonnage_mt: Optional[float] = None
    mi_grade_gpt: Optional[float] = None
    inferred_tonnage_mt: Optional[float] = None
    inferred_grade_gpt: Optional[float] = None
    total_tonnage_mt: Optional[float] = None
    total_grade_gpt: Optional[float] = None
    total_contained_oz: Optional[float] = None
    raw_parallel_output: Dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "mi_tonnage_mt",
        "mi_grade_gpt",
        "inferred_tonnage_mt",
        "inferred_grade_gpt",
        "total_tonnage_mt",
        "total_grade_gpt",
        "total_contained_oz",
    )
    @classmethod
    def _non_negative(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("resource values must be non-negative")
        return value


class GoldEvidenceFact(BaseModel):
    id: Optional[str] = None
    project_id: Optional[str] = None
    mre_truth_id: Optional[str] = None
    cutoff_date: date
    source_url: str
    source_date: Optional[date] = None
    source_title: Optional[str] = None
    source_document_type: Optional[str] = None
    evidence_status: EvidenceStatus = "accepted"
    rejection_reason: Optional[str] = None
    fact_type: str
    value_num: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None
    confidence: Confidence = "medium"
    is_mre_tainted: bool = False
    fact_payload: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("value_num")
    @classmethod
    def _value_num_finite(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("numeric evidence values must be non-negative")
        return value


class GoldAnalogCandidate(BaseModel):
    id: Optional[str] = None
    target_project_id: Optional[str] = None
    candidate_project_name: str
    candidate_company_name: Optional[str] = None
    candidate_country: Optional[str] = None
    candidate_region: Optional[str] = None
    candidate_district: Optional[str] = None
    candidate_deposit_family: Optional[str] = None
    candidate_deposit_subtype: Optional[str] = None
    candidate_tectonic_belt: Optional[str] = None
    candidate_mineralization_mode: Optional[str] = None
    candidate_mineralization_pattern: Optional[str] = None
    candidate_host_rock_class: Optional[str] = None
    candidate_mining_method_class: Optional[str] = None
    candidate_project_stage_class: Optional[str] = None
    candidate_recovery_method: Optional[str] = None
    source_url: str
    source_date: Optional[date] = None
    resource_standard: Optional[str] = None
    total_tonnage_mt: Optional[float] = None
    total_grade_gpt: Optional[float] = None
    total_contained_oz: Optional[float] = None
    mi_tonnage_mt: Optional[float] = None
    mi_grade_gpt: Optional[float] = None
    inferred_tonnage_mt: Optional[float] = None
    inferred_grade_gpt: Optional[float] = None
    drill_meters: Optional[float] = None
    drill_holes: Optional[int] = None
    best_intercepts: List[Dict[str, Any]] = Field(default_factory=list)
    geometry_payload: Dict[str, Any] = Field(default_factory=dict)
    raw_parallel_output: Dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "total_tonnage_mt",
        "total_grade_gpt",
        "total_contained_oz",
        "mi_tonnage_mt",
        "mi_grade_gpt",
        "inferred_tonnage_mt",
        "inferred_grade_gpt",
        "drill_meters",
    )
    @classmethod
    def _analog_values_non_negative(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("analog numeric values must be non-negative")
        return value


class GoldAnalogDecision(BaseModel):
    id: Optional[str] = None
    target_project_id: Optional[str] = None
    analog_candidate_id: Optional[str] = None
    decision: Decision
    decision_rules: List[Dict[str, Any]] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    accepted_at: Optional[datetime] = None


class GoldPredictionRunRecord(BaseModel):
    id: Optional[str] = None
    project_id: Optional[str] = None
    mre_truth_id: Optional[str] = None
    run_mode: Literal["blind_no_mre", "cached_replay", "production"] = "blind_no_mre"
    run_status: RunStatus
    input_hash: str
    cutoff_date: date
    evidence_fact_ids: List[str] = Field(default_factory=list)
    analog_candidate_ids: List[str] = Field(default_factory=list)
    analog_decision_ids: List[str] = Field(default_factory=list)
    no_prediction_reasons: List[str] = Field(default_factory=list)
    predicted_total_tonnage_mt: Optional[float] = None
    predicted_total_grade_gpt: Optional[float] = None
    predicted_total_contained_oz: Optional[float] = None
    predicted_mi_tonnage_mt: Optional[float] = None
    predicted_mi_grade_gpt: Optional[float] = None
    predicted_inferred_tonnage_mt: Optional[float] = None
    predicted_inferred_grade_gpt: Optional[float] = None
    predictor_version: str
    calculator_trace: Dict[str, Any] = Field(default_factory=dict)


class GoldPredictionScoreRecord(BaseModel):
    prediction_run_id: str
    mre_truth_id: str
    threshold_pct: float = 5.0
    core_pass: bool = False
    split_pass: bool = False
    production_like_pass: bool = False
    tonnage_error_pct: Optional[float] = None
    grade_error_pct: Optional[float] = None
    contained_error_pct: Optional[float] = None
    mi_tonnage_error_pct: Optional[float] = None
    mi_grade_error_pct: Optional[float] = None
    inferred_tonnage_error_pct: Optional[float] = None
    inferred_grade_error_pct: Optional[float] = None
    failure_class: Optional[str] = None
    failure_reason: Optional[str] = None
    score_payload: Dict[str, Any] = Field(default_factory=dict)
