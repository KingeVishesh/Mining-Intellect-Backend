"""
Model Builder — builds resource estimates (Model 1 and Model 2).

Model 1 (Independent): based entirely on analogs + rules, ignoring any official MRE.
Model 2 (Updated):     reconciles Model 1 with the official MRE if one is available.

LLM is used only as a last-resort sanity check and for narrative explanation.
All core calculations are deterministic.
"""
from __future__ import annotations
import json
import logging
import math
from typing import Dict, List, Optional

from nodes.llm_factory import get_llm

logger = logging.getLogger(__name__)


def _weighted_average(values: List[float], weights: List[float]) -> float:
    """Compute a weighted average, returning 0 if total weight is 0."""
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _contained_metal(tonnage_kt: float, grade_pct: float, material: str) -> float:
    """
    Calculate contained metal.
    For base metals (%, lb): tonnage_kt * 1000t/kt * grade_pct/100 * 2204.62 lb/t / 1e6 = Mlb
    For gold/silver (g/t, oz): tonnage_kt * 1000 * grade_g_t / 31.1035 / 1e6 = Moz
    Returns in Mlb (base metals) or Moz (precious metals).
    """
    precious = {"gold", "silver", "platinum", "palladium"}
    if material.lower() in precious:
        # g/t -> Moz
        return (tonnage_kt * 1000 * grade_pct) / 31.1035 / 1e6
    else:
        # % -> Mlb
        return (tonnage_kt * 1000 * (grade_pct / 100) * 2204.62) / 1e6


def build_model_1(
    analogs: List[Dict],
    project: Dict,
    rule_effects: Dict,
) -> Dict:
    """
    Build Model 1 (Independent estimate).
    Uses analog-weighted average + rule multipliers.
    """
    material = project.get("material", "unknown")

    # Filter analogs that have both tonnage and grade
    valid = [
        a for a in analogs
        if a.get("tonnage_mt") is not None and a.get("grade_value") is not None
    ]

    if not valid:
        logger.warning("[Model1] No valid analogs with tonnage+grade — using minimal defaults")
        return _minimal_model(project, material, "Model 1 (Independent)")

    # Use similarity score as weight (default 50 if missing)
    weights = [float(a.get("similarity_score", 50)) for a in valid]
    tonnages_kt = [float(a["tonnage_mt"]) * 1000 for a in valid]  # convert Mt -> kt
    grades = [float(a["grade_value"]) for a in valid]

    # Split into M&I (70%) and inferred (30%) based on analog averages
    base_total_kt = _weighted_average(tonnages_kt, weights)
    base_grade = _weighted_average(grades, weights)
    base_mi_kt = base_total_kt * 0.70
    base_inferred_kt = base_total_kt * 0.30

    # Apply rule multipliers
    adj = rule_effects or {}
    t_mult = float(adj.get("tonnage_multiplier", 1.0))
    g_mult = float(adj.get("grade_multiplier", 1.0))
    conf_delta = float(adj.get("confidence_delta", 0.0))

    mi_kt = base_mi_kt * t_mult
    inferred_kt = base_inferred_kt * t_mult
    grade = base_grade * g_mult

    # Conviction: based on analog count, similarity scores, and rule confidence
    avg_similarity = _weighted_average(weights, [1.0] * len(weights))
    analog_confidence = min(100.0, (len(valid) / 5) * 40 + avg_similarity * 0.4)
    conviction = max(0.0, min(100.0, analog_confidence + conf_delta))

    return {
        "model": "Model 1 (Independent)",
        "mi_tonnage_kt": round(mi_kt, 2),
        "mi_grade_pct": round(grade, 4),
        "mi_contained_mlb": round(_contained_metal(mi_kt, grade, material), 3),
        "inferred_tonnage_kt": round(inferred_kt, 2),
        "inferred_grade_pct": round(grade * 0.95, 4),  # inferred slightly lower confidence
        "inferred_contained_mlb": round(_contained_metal(inferred_kt, grade * 0.95, material), 3),
        "total_tonnage_kt": round(mi_kt + inferred_kt, 2),
        "total_grade_pct": round(grade, 4),
        "total_contained_mlb": round(_contained_metal(mi_kt + inferred_kt, grade, material), 3),
        "description": f"Independent estimate using {len(valid)} analog project(s) and {len(adj.get('rules_applied', []))} rules.",
        "conviction_pct": round(conviction, 1),
        "analogs_used": [a.get("name", "unknown") for a in valid],
        "rules_applied": adj.get("rules_applied", []),
    }


def build_model_2(
    model_1: Dict,
    project: Dict,
    official_mre: Optional[Dict],
) -> Optional[Dict]:
    """
    Build Model 2 (Updated estimate).
    Reconciles Model 1 with the official MRE using an 80/20 blend.
    Returns None if no official MRE is available.
    """
    if not official_mre:
        return None

    material = project.get("material", "unknown")
    official_tonnage_kt = float(official_mre.get("tonnage_mt", 0)) * 1000
    official_grade = float(official_mre.get("grade_value", 0))

    if official_tonnage_kt == 0 or official_grade == 0:
        return None

    m1_tonnage = model_1["total_tonnage_kt"]
    m1_grade = model_1["total_grade_pct"]

    # 80% official MRE, 20% Model 1
    blended_tonnage = 0.8 * official_tonnage_kt + 0.2 * m1_tonnage
    blended_grade = 0.8 * official_grade + 0.2 * m1_grade

    mi_kt = blended_tonnage * 0.65
    inferred_kt = blended_tonnage * 0.35

    # Model 2 conviction is higher because we have official data
    conviction = min(100.0, model_1["conviction_pct"] * 0.3 + 65.0)

    return {
        "model": "Model 2 (Updated)",
        "mi_tonnage_kt": round(mi_kt, 2),
        "mi_grade_pct": round(blended_grade, 4),
        "mi_contained_mlb": round(_contained_metal(mi_kt, blended_grade, material), 3),
        "inferred_tonnage_kt": round(inferred_kt, 2),
        "inferred_grade_pct": round(blended_grade * 0.95, 4),
        "inferred_contained_mlb": round(_contained_metal(inferred_kt, blended_grade * 0.95, material), 3),
        "total_tonnage_kt": round(blended_tonnage, 2),
        "total_grade_pct": round(blended_grade, 4),
        "total_contained_mlb": round(_contained_metal(blended_tonnage, blended_grade, material), 3),
        "description": "Updated estimate reconciling independent model with official MRE (80/20 blend).",
        "conviction_pct": round(conviction, 1),
        "analogs_used": model_1.get("analogs_used", []),
        "rules_applied": model_1.get("rules_applied", []),
    }


def build_official_mre_row(project: Dict) -> Optional[Dict]:
    """
    If the project has official MRE data (tonnage_mt + grade_value), return a row for
    the comparison table labelled 'Official MRE'.
    """
    material = project.get("material", "unknown")
    tonnage_mt = project.get("tonnage_mt")
    grade = project.get("grade_value")
    if not tonnage_mt or not grade:
        return None
    total_kt = float(tonnage_mt) * 1000
    mi_kt = total_kt * 0.65
    inferred_kt = total_kt * 0.35
    return {
        "model": "Official MRE",
        "mi_tonnage_kt": round(mi_kt, 2),
        "mi_grade_pct": float(grade),
        "mi_contained_mlb": round(_contained_metal(mi_kt, float(grade), material), 3),
        "inferred_tonnage_kt": round(inferred_kt, 2),
        "inferred_grade_pct": round(float(grade) * 0.95, 4),
        "inferred_contained_mlb": round(_contained_metal(inferred_kt, float(grade) * 0.95, material), 3),
        "total_tonnage_kt": round(total_kt, 2),
        "total_grade_pct": float(grade),
        "total_contained_mlb": round(_contained_metal(total_kt, float(grade), material), 3),
        "description": f"Official MRE from project data ({project.get('resource_category', 'M+I+Inf')}).",
        "conviction_pct": 95.0,
        "analogs_used": [],
        "rules_applied": [],
    }


def _minimal_model(project: Dict, material: str, label: str) -> Dict:
    """Return a zeroed-out model when no analogs are available."""
    return {
        "model": label,
        "mi_tonnage_kt": 0.0,
        "mi_grade_pct": 0.0,
        "mi_contained_mlb": 0.0,
        "inferred_tonnage_kt": 0.0,
        "inferred_grade_pct": 0.0,
        "inferred_contained_mlb": 0.0,
        "total_tonnage_kt": 0.0,
        "total_grade_pct": 0.0,
        "total_contained_mlb": 0.0,
        "description": "Insufficient analog data to build estimate.",
        "conviction_pct": 0.0,
        "analogs_used": [],
        "rules_applied": [],
    }


def generate_report_narrative(
    project: Dict,
    model_1: Dict,
    model_2: Optional[Dict],
    analogs: List[Dict],
    activated_rules: List[Dict],
) -> Dict:
    """
    Use the LLM to generate the narrative sections of the report.
    All numbers come from the deterministic models above — LLM only writes prose.
    """
    llm = get_llm(temperature=0.2)

    has_mre = project.get("tonnage_mt") and project.get("grade_value")
    model_summary = json.dumps(model_1, indent=2)
    if model_2:
        model_summary += "\n\nModel 2:\n" + json.dumps(model_2, indent=2)

    prompt = f"""You are a senior mining analyst writing a resource estimation report.

PROJECT: {project.get('name')} — {project.get('material')}
Stage: {project.get('project_stage', 'Unknown')}
Location: {project.get('country', 'Unknown')}, {project.get('region', '')}
Deposit Type: {project.get('deposit_type', 'Unknown')}
Official MRE: {"Yes — " + str(project.get('tonnage_mt')) + "Mt @ " + str(project.get('grade_value')) + " " + str(project.get('grade_unit')) if has_mre else "Not available"}

RESOURCE MODELS:
{model_summary}

ANALOGS USED: {[a.get('name') for a in analogs[:5]]}
RULES APPLIED: {len(activated_rules)}

Write the following sections as valid JSON:
{{
  "executive_summary": {{
    "summary_text": "2-3 paragraph executive summary",
    "overall_assessment": "Positive | Cautious | Negative",
    "key_takeaway": "One sentence key takeaway"
  }},
  "project_overview": {{
    "project_summary": "1-2 paragraph project description",
    "key_characteristics": ["characteristic 1", "characteristic 2", ...],
    "official_mre_summary": "1 paragraph on official MRE or null",
    "drilling_data_summary": "1 paragraph on drilling or null"
  }},
  "actionable_recommendations": [
    {{"recommendation": "...", "priority": "High|Medium|Low", "rationale": "..."}}
  ],
  "key_uncertainties_and_strengths": {{
    "strengths": ["strength 1", ...],
    "uncertainties": ["uncertainty 1", ...]
  }}
}}

Return ONLY the JSON object. No markdown. No other text.
"""
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        logger.error(f"[Report] Narrative generation error: {e}")
        return {
            "executive_summary": {
                "summary_text": "Report generation encountered an error.",
                "overall_assessment": "Cautious",
                "key_takeaway": "Manual review required.",
            },
            "project_overview": {
                "project_summary": "",
                "key_characteristics": [],
                "official_mre_summary": None,
                "drilling_data_summary": None,
            },
            "actionable_recommendations": [],
            "key_uncertainties_and_strengths": {"strengths": [], "uncertainties": []},
        }
