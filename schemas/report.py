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
    model: str                      # "Model 1 (Independent)" | "Model 2 (Updated)" | "Official MRE"
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
