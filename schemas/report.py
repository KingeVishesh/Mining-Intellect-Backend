"""
Mining Intellect — Canonical Report Schema (Pydantic)

This is the contract between backend and frontend.
Frontend always reads from this exact structure.
Never change field names without coordinating with frontend.
"""
from __future__ import annotations
from typing import Any, List, Optional
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
    metadata: dict                  # { project_name, material, generated_at, report_type }
    executive_summary: dict         # { summary_text, overall_assessment, key_takeaway }
    project_overview: dict          # { project_summary, key_characteristics, official_mre_summary, drilling_data_summary }
    resource_estimates: ResourceEstimates
    actionable_recommendations: List[dict]
    lessons_summary: dict           # { total_lessons_applied, high_confidence_lessons }
    key_uncertainties_and_strengths: dict
