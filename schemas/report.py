"""
Mining Intellect — Canonical Report Schema (Pydantic)

This is the contract between backend and frontend.
Frontend always reads from this exact structure.
Never change field names without coordinating with frontend.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ComparisonTableRow(BaseModel):
    model: str                      # "MI Model (Pre-MRE)" | "MI Model (Post-MRE)" | "Official MRE"
    mi_tonnage_kt: float
    mi_grade_pct: float
    mi_contained_mlb: float
    inferred_tonnage_kt: float
    inferred_grade_pct: float
    inferred_contained_mlb: float
    total_tonnage_kt: float
    total_grade_pct: float
    total_contained_mlb: float
    description: str
    conviction_pct: Optional[float] = None
    conviction_tier: Optional[str] = None    # e.g. "PRE-3" or "POST-2"
    conviction_label: Optional[str] = None  # e.g. "Developing" or "Resource-Stage"


class ResourceEstimates(BaseModel):
    comparison_table: List[ComparisonTableRow]
    independent_analysis: dict      # { confidence_pct, key_factors: List[str] }
    updated_analysis: dict          # { confidence_pct, key_factors: List[str] }
    compliance_summary: List[str]


class MiningReport(BaseModel):
    # ── Core (always present) ──────────────────────────────────────────────────
    metadata: dict                  # { project_name, material, generated_at, report_type }
    executive_summary: dict         # { summary_text, overall_assessment, key_takeaway }
    project_overview: dict          # { project_summary, key_characteristics, official_mre_summary, drilling_data_summary }
    resource_estimates: ResourceEstimates
    actionable_recommendations: List[dict]
    lessons_summary: dict           # { total_lessons_applied, high_confidence_lessons }
    key_uncertainties_and_strengths: dict

    # ── Extended sections (all default on, can be excluded via sections config) ─
    analogs_comparison: Optional[List[dict]] = None
    # Each item: { name, tonnage_mt, grade_value, grade_unit, deposit_type, country,
    #              similarity_score, source, source_url }

    sensitivity_analysis: Optional[dict] = None
    # { cutoff_table: [...], price_table: [...], recovery_table: [...], scenario_table: [...] }

    risk_matrix: Optional[List[dict]] = None
    # Each item: { risk_factor, probability, impact, mitigation }

    exploration_strategy: Optional[List[dict]] = None
    # Each item: { activity, cost_estimate, timeline, priority }

    economic_assumptions: Optional[dict] = None
    # { cueq_formula, recoveries: {cu, au, ag}, cutoff_grades: {base, high_grade},
    #   metal_prices: {cu_per_lb, au_per_oz, ag_per_oz}, cost_per_tonne }

    key_terms: Optional[List[dict]] = None
    # Each item: { term, definition }

    acquisition_analysis: Optional[dict] = None
    # { junior: { verdict, score_summary, items: [...] },
    #   mid_tier: { verdict, score_summary, items: [...] },
    #   major: { verdict, score_summary, items: [...] },
    #   comparable_transactions: [...] }

    # ── Deep-dive sections (LLM-generated, added by generate_extended_sections_node) ─
    geological_framework: Optional[dict] = None
    # { regional_setting, deposit_characteristics, mineralization_description,
    #   structural_complexity, geological_continuity, logistics_and_infrastructure,
    #   mineral_zones: [{zone_name, description, grade_range}] }

    drilling_and_sampling: Optional[dict] = None
    # { drillhole_strategy, total_holes_estimated, assay_qa_qc,
    #   xrf_geochemical_notes, cost_efficiency_notes, data_quality_assessment }

    drilling_efficiency_metrics: Optional[dict] = None
    # { narrative, metrics_table: [{metric, project_value, peer_range, assessment}],
    #   shareholder_dilution_efficiency, cost_per_meter_vs_peers }

    geophysical_integration: Optional[dict] = None
    # { survey_types_recommended: [{survey_type, rationale, priority}],
    #   continuity_thresholds, validation_triggers, existing_data_notes }

    geostatistical_modeling: Optional[dict] = None
    # { variography_narrative, variogram_parameters: [{zone, nugget, sill, range_major_m, range_minor_m, anisotropy_ratio}],
    #   grade_capping_method, extension_ranges, byproduct_modeling, estimation_method }

    validation_and_qc: Optional[dict] = None
    # { check_assay_protocol, monte_carlo_summary,
    #   p10_tonnage_kt, p90_tonnage_kt, p10_grade, p90_grade,
    #   statistical_reconciliation, audit_trail_notes }

    conclusion: Optional[dict] = None
    # { conclusion_text, headline_finding, next_milestone, investment_readiness }

    appendices: Optional[dict] = None
    # { input_weighting_table: [{analog_name, weight_pct, key_rationale}],
    #   variogram_parameters_table: [{zone, nugget, sill, range_major_m, range_minor_m}],
    #   drilling_summary_table: [{hole_type, count, avg_depth_m, purpose}],
    #   references: [str] }
