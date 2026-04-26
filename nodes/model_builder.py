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


def compute_sensitivity_analysis(model_1: Dict, project: Dict) -> Dict:
    """
    Compute sensitivity tables programmatically from model numbers.
    No LLM needed — pure arithmetic based on industry-standard approximations.
    """
    base_tonnage = model_1.get("total_tonnage_kt", 0)
    base_grade   = model_1.get("total_grade_pct", 0)
    base_metal   = model_1.get("total_contained_mlb", 0)
    material     = project.get("material", "unknown")
    grade_unit   = project.get("grade_unit", "%")

    # Typical IOCG/porphyry cut-off grade sensitivity: lower cut-off → more tonnage at lower grade
    cutoff_steps = [
        (-0.30, +0.15, -0.04),  # cut-off -30% → tonnage +15%, grade -4%
        (-0.20, +0.10, -0.03),
        (-0.10, +0.05, -0.01),
        (0.00,   0.00,  0.00),  # base
        (+0.10, -0.05, +0.02),
        (+0.20, -0.10, +0.03),
        (+0.30, -0.15, +0.05),
    ]
    cutoff_table = []
    for co_delta, t_delta, g_delta in cutoff_steps:
        label = "Base" if co_delta == 0 else f"{'+' if co_delta > 0 else ''}{int(co_delta*100)}%"
        t = base_tonnage * (1 + t_delta)
        g = base_grade * (1 + g_delta)
        m = base_metal * (1 + t_delta + g_delta)
        cutoff_table.append({
            "cut_off_label": label,
            "tonnage_kt": round(t, 0),
            "grade": round(g, 4),
            "grade_unit": grade_unit,
            "contained_metal": round(max(0, m), 3),
            "metal_unit": "Mlb" if material.lower() not in {"gold","silver","platinum","palladium"} else "Moz",
        })

    # Metal price sensitivity
    price_steps = [-0.30, -0.20, -0.10, 0.0, +0.10, +0.20, +0.30]
    price_table = []
    for p_delta in price_steps:
        label = "Base" if p_delta == 0 else f"{'+' if p_delta > 0 else ''}{int(p_delta*100)}%"
        # ±10% price → ~±3% tonnage (lower price = higher cut-off = less tonnage)
        t_delta = -p_delta * 0.3
        m_delta = p_delta * 0.7  # metal value tracks price more directly
        t = base_tonnage * (1 + t_delta)
        m = base_metal * (1 + m_delta)
        price_table.append({
            "price_label": label,
            "price_delta_pct": round(p_delta * 100, 0),
            "tonnage_kt": round(t, 0),
            "contained_metal": round(max(0, m), 3),
            "metal_unit": "Mlb" if material.lower() not in {"gold","silver","platinum","palladium"} else "Moz",
        })

    # Recovery sensitivity (±10%)
    recovery_steps = [-0.10, 0.0, +0.10]
    recovery_table = []
    for r_delta in recovery_steps:
        label = "Base" if r_delta == 0 else f"{'+' if r_delta > 0 else ''}{int(r_delta*100)}%"
        m = base_metal * (1 + r_delta)
        recovery_table.append({
            "recovery_label": label,
            "recovery_delta_pct": round(r_delta * 100, 0),
            "contained_metal": round(max(0, m), 3),
            "metal_unit": "Mlb" if material.lower() not in {"gold","silver","platinum","palladium"} else "Moz",
        })

    # Combined scenarios
    scenario_table = [
        {
            "scenario": "Best Case",
            "cut_off": "-30%",
            "metal_price": "+20%",
            "recovery": "+10%",
            "tonnage_kt": round(base_tonnage * 1.15, 0),
            "contained_metal": round(base_metal * 1.35, 3),
        },
        {
            "scenario": "Base Case",
            "cut_off": "Base",
            "metal_price": "Base",
            "recovery": "Base",
            "tonnage_kt": round(base_tonnage, 0),
            "contained_metal": round(base_metal, 3),
        },
        {
            "scenario": "Worst Case",
            "cut_off": "+30%",
            "metal_price": "-20%",
            "recovery": "-10%",
            "tonnage_kt": round(base_tonnage * 0.85, 0),
            "contained_metal": round(base_metal * 0.65, 3),
        },
    ]

    return {
        "cutoff_table":   cutoff_table,
        "price_table":    price_table,
        "recovery_table": recovery_table,
        "scenario_table": scenario_table,
    }


def generate_report_narrative(
    project: Dict,
    model_1: Dict,
    model_2: Optional[Dict],
    analogs: List[Dict],
    activated_rules: List[Dict],
    sections: Optional[List[str]] = None,
) -> Dict:
    """
    Use the LLM to generate all narrative sections of the report.
    All numbers come from the deterministic models above — LLM only writes prose.
    If sections is None, all sections are generated.
    """
    llm = get_llm(temperature=0.2)

    # Default: all sections
    all_sections = {
        "executive_summary", "project_overview", "actionable_recommendations",
        "key_uncertainties_and_strengths", "risk_matrix", "exploration_strategy",
        "key_terms", "economic_assumptions", "acquisition_analysis",
    }
    active = set(sections) if sections else all_sections

    has_mre = project.get("tonnage_mt") and project.get("grade_value")
    material = project.get("material", "Unknown")
    grade_unit = project.get("grade_unit", "%")
    model_summary = json.dumps(model_1, indent=2)
    if model_2:
        model_summary += "\n\nModel 2:\n" + json.dumps(model_2, indent=2)

    analogs_summary = json.dumps([
        {k: a.get(k) for k in ("name","tonnage_mt","grade_value","grade_unit","deposit_type","country","similarity_score")}
        for a in analogs[:8]
    ], indent=2)

    prompt = f"""You are a senior mining analyst writing a detailed resource estimation report.

PROJECT: {project.get('name')} — {material}
Stage: {project.get('project_stage', 'Unknown')}
Location: {project.get('country', 'Unknown')}{', ' + project.get('region','') if project.get('region') else ''}
Deposit Type: {project.get('deposit_type', 'Unknown')}
Official MRE: {"Yes — " + str(project.get('tonnage_mt')) + "Mt @ " + str(project.get('grade_value')) + " " + str(grade_unit) if has_mre else "Not available"}

RESOURCE MODELS:
{model_summary}

ANALOGS USED ({len(analogs)} total, top 8 shown):
{analogs_summary}

RULES APPLIED: {len(activated_rules)}

Return a single valid JSON object with ALL of the following keys.
Be specific, technical, and professional. Use actual project data in every section.

{{
  "executive_summary": {{
    "summary_text": "3 detailed paragraphs: (1) project overview + resource estimates with actual numbers, (2) methodology and analog comparison, (3) key risks and upside potential",
    "overall_assessment": "Positive | Cautious | Negative",
    "key_takeaway": "One crisp sentence with the most important insight"
  }},
  "project_overview": {{
    "project_summary": "2 paragraphs covering location, deposit type, host rocks, mineralization style, and current exploration stage",
    "key_characteristics": ["specific characteristic with numbers", "..."],
    "official_mre_summary": "1 paragraph on official MRE data, or null if none",
    "drilling_data_summary": "1 paragraph on drilling history and data quality, or null"
  }},
  "actionable_recommendations": [
    {{"recommendation": "specific action", "priority": "High|Medium|Low", "rationale": "why this matters with supporting data", "estimated_cost": "e.g. $X million or N/A", "timeline": "e.g. 6-12 months"}}
  ],
  "key_uncertainties_and_strengths": {{
    "strengths": ["specific strength with evidence", "..."],
    "uncertainties": ["specific uncertainty with impact", "..."]
  }},
  "risk_matrix": [
    {{"risk_factor": "name", "probability": "High (80%) | Moderate (50%) | Low (20%)", "impact": "High | Moderate | Low", "mitigation": "specific mitigation strategy"}}
  ],
  "exploration_strategy": [
    {{"activity": "specific activity", "cost_estimate": "e.g. US$X million", "timeline": "e.g. 6-12 months", "priority": "High|Medium|Low", "expected_outcome": "what success looks like"}}
  ],
  "key_terms": [
    {{"term": "technical term", "definition": "plain-English definition relevant to this project"}}
  ],
  "economic_assumptions": {{
    "cueq_formula": "CuEq formula as a string e.g. CuEq% = (Cu% x Cu_price x Cu_recovery + ...)",
    "metal_prices": {{"primary_metal": "{material}", "primary_price": "price used", "other_metals": []}},
    "recoveries": {{"primary_pct": 88, "notes": "assumed flotation recovery"}},
    "cutoff_grade": "e.g. 0.2% CuEq",
    "block_model_size": "25x25x10m",
    "cost_per_tonne": "estimated exploration cost per tonne of resource"
  }},
  "acquisition_analysis": {{
    "junior": {{
      "verdict": "Not suitable | Potentially suitable | Well-suited",
      "score_summary": "brief reason",
      "items": [{{"criterion": "...", "status": "green|amber|red", "comment": "..."}}]
    }},
    "mid_tier": {{
      "verdict": "Not suitable | Potentially suitable | Well-suited",
      "score_summary": "brief reason",
      "items": [{{"criterion": "...", "status": "green|amber|red", "comment": "..."}}]
    }},
    "major": {{
      "verdict": "Not suitable | Potentially suitable | Well-suited",
      "score_summary": "brief reason",
      "items": [{{"criterion": "...", "status": "green|amber|red", "comment": "..."}}]
    }}
  }}
}}

Rules:
- Include exactly 5 risk_matrix items
- Include exactly 4 exploration_strategy items
- Include exactly 8 key_terms specific to this deposit type and material
- Include exactly 4 actionable_recommendations
- Include exactly 3 items per acquisition tier checklist
- Return ONLY the JSON. No markdown fences, no explanation, no trailing text.
"""
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        # Find JSON boundaries
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        return json.loads(content)
    except Exception as e:
        logger.error(f"[Report] Narrative generation error: {e}")
        return _fallback_narrative(project)


def _fallback_narrative(project: Dict) -> Dict:
    name = project.get("name", "Unknown Project")
    material = project.get("material", "Unknown")
    return {
        "executive_summary": {
            "summary_text": f"This report presents a resource modeling assessment for {name}, a {material} project. The analysis uses analog-based methodology and available project data.",
            "overall_assessment": "Cautious",
            "key_takeaway": "Preliminary assessment — further data collection recommended.",
        },
        "project_overview": {
            "project_summary": f"{name} is a {material} project located in {project.get('country', 'unknown location')}.",
            "key_characteristics": [f"Material: {material}", f"Stage: {project.get('project_stage', 'Unknown')}"],
            "official_mre_summary": None,
            "drilling_data_summary": None,
        },
        "actionable_recommendations": [
            {"recommendation": "Conduct additional drilling", "priority": "High",
             "rationale": "Increase data density to improve resource confidence.",
             "estimated_cost": "N/A", "timeline": "6-12 months"},
        ],
        "key_uncertainties_and_strengths": {
            "strengths": ["Favorable jurisdiction", "Known deposit type"],
            "uncertainties": ["Limited drilling data", "Sparse analog comparison"],
        },
        "risk_matrix": [
            {"risk_factor": "Data sparsity", "probability": "High (80%)", "impact": "Moderate", "mitigation": "Additional drilling program"},
            {"risk_factor": "Commodity price volatility", "probability": "Moderate (50%)", "impact": "High", "mitigation": "Sensitivity analysis and hedging"},
            {"risk_factor": "Permitting delays", "probability": "Moderate (50%)", "impact": "Moderate", "mitigation": "Early engagement with regulators"},
            {"risk_factor": "Geological uncertainty", "probability": "Moderate (50%)", "impact": "High", "mitigation": "Geophysical surveys and ML modeling"},
            {"risk_factor": "Infrastructure requirements", "probability": "Low (20%)", "impact": "Moderate", "mitigation": "Feasibility study for infrastructure"},
        ],
        "exploration_strategy": [
            {"activity": "Infill drilling program", "cost_estimate": "TBD", "timeline": "6-12 months", "priority": "High", "expected_outcome": "Upgrade resource classification"},
            {"activity": "Geophysical surveys", "cost_estimate": "TBD", "timeline": "3-6 months", "priority": "High", "expected_outcome": "Define exploration targets"},
            {"activity": "Metallurgical testwork", "cost_estimate": "TBD", "timeline": "6-9 months", "priority": "Medium", "expected_outcome": "Confirm recovery assumptions"},
            {"activity": "Environmental baseline study", "cost_estimate": "TBD", "timeline": "12-18 months", "priority": "Medium", "expected_outcome": "Support permitting process"},
        ],
        "key_terms": [
            {"term": "Inferred Resource", "definition": "Mineral resource with lowest confidence — sufficient data to imply but not verify continuity."},
            {"term": "M&I Resource", "definition": "Measured and Indicated resources — higher confidence than Inferred."},
            {"term": "Grade", "definition": "Concentration of the target mineral expressed as % or g/t."},
            {"term": "Tonnage", "definition": "Total mass of mineralized rock in the resource estimate."},
            {"term": "Cut-off Grade", "definition": "Minimum grade below which material is not economic to mine."},
            {"term": "Analog Project", "definition": "A comparable deposit used to calibrate the resource model."},
            {"term": "Conviction", "definition": "MI's internal confidence rating in the resource estimate (0–100%)."},
            {"term": "NI 43-101", "definition": "Canadian regulatory standard for reporting mineral resources — this report does not comply."},
        ],
        "economic_assumptions": {
            "cueq_formula": "Based on primary metal value and standard industry recoveries",
            "metal_prices": {"primary_metal": material, "primary_price": "Market rate", "other_metals": []},
            "recoveries": {"primary_pct": 88, "notes": "Assumed standard flotation recovery"},
            "cutoff_grade": "0.2% equivalent",
            "block_model_size": "25x25x10m",
            "cost_per_tonne": "~$1.50/t equivalent",
        },
        "acquisition_analysis": {
            "junior": {"verdict": "Not suitable", "score_summary": "Insufficient data for junior assessment",
                       "items": [{"criterion": "Resource size", "status": "amber", "comment": "Pending further data"}]},
            "mid_tier": {"verdict": "Potentially suitable", "score_summary": "Depends on final resource size",
                         "items": [{"criterion": "Resource size", "status": "amber", "comment": "Monitor as resource grows"}]},
            "major": {"verdict": "Potentially suitable", "score_summary": "Favorable jurisdiction",
                      "items": [{"criterion": "Jurisdiction", "status": "green", "comment": "Tier-1 jurisdiction favorable"}]},
        },
    }
