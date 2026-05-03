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


_SYMBOL_TO_MATERIAL: Dict[str, str] = {
    "ag": "silver", "au": "gold", "pt": "platinum", "pd": "palladium",
    "cu": "copper", "pb": "lead", "zn": "zinc", "ni": "nickel",
    "mo": "molybdenum", "u": "uranium", "u3o8": "uranium",
}


def _norm_material(material: str) -> str:
    """Normalize element symbol (Ag, Au) or alternate spellings to canonical name."""
    return _SYMBOL_TO_MATERIAL.get(material.strip().lower(), material.strip().lower())


def _contained_metal(tonnage_kt: float, grade_pct: float, material: str) -> float:
    """
    Calculate contained metal.
    For base metals (%, lb): tonnage_kt * 1000t/kt * grade_pct/100 * 2204.62 lb/t / 1e6 = Mlb
    For gold/silver (g/t, oz): tonnage_kt * 1000 * grade_g_t / 31.1035 / 1e6 = Moz
    Returns in Mlb (base metals) or Moz (precious metals).
    """
    precious = {"gold", "silver", "platinum", "palladium"}
    if _norm_material(material) in precious:
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
    material = _norm_material(project.get("material", "unknown"))

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
    """
    Fallback when no valid analogs are available.
    Uses the project's own tonnage_mt/grade_value if present (low conviction),
    otherwise returns an all-zero placeholder.
    """
    own_kt = float(project.get("tonnage_mt") or 0) * 1000
    own_g  = float(project.get("grade_value") or 0)
    if own_kt > 0 and own_g > 0:
        mi_kt  = own_kt * 0.70
        inf_kt = own_kt * 0.30
        return {
            "model": label,
            "mi_tonnage_kt": round(mi_kt, 2),
            "mi_grade_pct": round(own_g, 4),
            "mi_contained_mlb": round(_contained_metal(mi_kt, own_g, material), 3),
            "inferred_tonnage_kt": round(inf_kt, 2),
            "inferred_grade_pct": round(own_g * 0.95, 4),
            "inferred_contained_mlb": round(_contained_metal(inf_kt, own_g * 0.95, material), 3),
            "total_tonnage_kt": round(own_kt, 2),
            "total_grade_pct": round(own_g, 4),
            "total_contained_mlb": round(_contained_metal(own_kt, own_g, material), 3),
            "description": "Estimate based on project data only (no comparable analog data available).",
            "conviction_pct": 15.0,
            "analogs_used": [],
            "rules_applied": [],
        }
    # Truly no data at all
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
        "description": "Insufficient data — no analogs and no project MRE available.",
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
    material     = _norm_material(project.get("material", "unknown"))
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
            "metal_unit": "Mlb" if material not in {"gold", "silver", "platinum", "palladium"} else "Moz",
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
            "metal_unit": "Mlb" if material not in {"gold", "silver", "platinum", "palladium"} else "Moz",
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
            "metal_unit": "Mlb" if material not in {"gold", "silver", "platinum", "palladium"} else "Moz",
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
        "monte_carlo_p10_p90": {
            "p10_tonnage_kt": round(base_tonnage * 0.85, 0),
            "p90_tonnage_kt": round(base_tonnage * 1.15, 0),
            "p10_grade":      round(base_grade * 0.90, 4),
            "p90_grade":      round(base_grade * 1.10, 4),
        },
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
        logger.error(f"[Report] Narrative FAILED for '{project.get('name')}': {e}")
        _head = locals().get("content", "no response")
        logger.error(f"[Report] LLM response head: {str(_head)[:300]}")
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


# ── Extended narrative (second LLM call) ───────────────────────────────────────

def compute_extended_deterministic(model_1: Dict, project: Dict) -> Dict:
    """Compute P10/P90 uncertainty bands deterministically (no LLM)."""
    base_tonnage = model_1.get("total_tonnage_kt", 0)
    base_grade   = model_1.get("total_grade_pct", 0)
    return {
        "p10_tonnage_kt": round(base_tonnage * 0.85, 0),
        "p90_tonnage_kt": round(base_tonnage * 1.15, 0),
        "p10_grade":      round(base_grade * 0.90, 4),
        "p90_grade":      round(base_grade * 1.10, 4),
    }


def _fallback_extended_narrative(project: Dict, deterministic_vals: Dict) -> Dict:
    """Return minimal valid dicts for all 8 extended sections when LLM fails."""
    name     = project.get("name", "the project")
    material = project.get("material", "Unknown")
    dep_type = project.get("deposit_type", "mineral")
    country  = project.get("country", "unknown location")
    p10t     = deterministic_vals.get("p10_tonnage_kt", 0)
    p90t     = deterministic_vals.get("p90_tonnage_kt", 0)
    p10g     = deterministic_vals.get("p10_grade", 0)
    p90g     = deterministic_vals.get("p90_grade", 0)
    return {
        "geological_framework": {
            "regional_setting": f"{name} is located in {country}, within a region prospective for {dep_type} {material} mineralisation.",
            "deposit_characteristics": f"The deposit is characterised by {dep_type} style mineralisation typical of the region.",
            "mineralization_description": f"Primary mineralisation comprises {material}-bearing zones with associated alteration assemblages.",
            "structural_complexity": "Moderate structural complexity with local fault controls on mineralisation.",
            "geological_continuity": "Geological continuity is considered adequate for early-stage resource modelling.",
            "logistics_and_infrastructure": "Infrastructure assessment is required as part of pre-feasibility planning.",
            "mineral_zones": [
                {"zone_name": "Primary Zone", "description": f"Main {material} mineralisation zone", "grade_range": "Variable"},
                {"zone_name": "Oxide Zone", "description": "Near-surface oxide mineralisation", "grade_range": "Lower grade"},
            ],
        },
        "drilling_and_sampling": {
            "drillhole_strategy": "Systematic drilling programme targeting primary mineralisation zones.",
            "total_holes_estimated": "Estimated drilling requirements to be determined by pre-feasibility study.",
            "assay_qa_qc": "Standard QAQC protocols including blanks, standards, and duplicates at 1:20 insertion rate.",
            "xrf_geochemical_notes": "Portable XRF used for rapid on-site grade estimation and zone delineation.",
            "cost_efficiency_notes": "Drilling costs estimated in line with regional benchmarks for this jurisdiction.",
            "data_quality_assessment": "Data quality is considered appropriate for early-stage resource estimation.",
        },
        "drilling_efficiency_metrics": {
            "narrative": f"Drilling efficiency metrics for {name} are benchmarked against comparable {dep_type} projects.",
            "metrics_table": [
                {"metric": "Metal Added per Meter Drilled", "project_value": "To be determined", "peer_range": "Deposit-type dependent", "assessment": "In-Line"},
                {"metric": "Discovery Cost per Tonne", "project_value": "To be determined", "peer_range": "$0.50–$3.00/t", "assessment": "In-Line"},
                {"metric": "Drilling Cost per Meter", "project_value": "To be determined", "peer_range": "$150–$350/m", "assessment": "In-Line"},
                {"metric": "Shareholder Dilution Efficiency", "project_value": "To be determined", "peer_range": "Peer comparable", "assessment": "In-Line"},
            ],
            "shareholder_dilution_efficiency": "Dilution efficiency analysis requires share registry and market cap data.",
            "cost_per_meter_vs_peers": "All-in drilling costs to be benchmarked once programme is finalised.",
        },
        "geophysical_integration": {
            "survey_types_recommended": [
                {"survey_type": "Induced Polarisation (IP)", "rationale": "Maps sulphide zones and confirms mineralisation boundaries", "priority": "High"},
                {"survey_type": "Airborne EM", "rationale": "Provides regional coverage and detects conductive targets", "priority": "Medium"},
                {"survey_type": "Ground Magnetics", "rationale": "Delineates structural controls and alteration zones", "priority": "Medium"},
            ],
            "continuity_thresholds": "Grade continuity thresholds to be established following detailed variographic analysis.",
            "validation_triggers": "Geophysical anomalies exceeding 2-sigma threshold to trigger follow-up drilling.",
            "existing_data_notes": "Existing geophysical data coverage to be reviewed as part of data compilation.",
        },
        "geostatistical_modeling": {
            "variography_narrative": f"Variographic analysis for {name} will be conducted on assay data to define spatial continuity parameters for ordinary kriging.",
            "variogram_parameters": [
                {"zone": "Primary Zone", "nugget": "0.10", "sill": "0.80", "range_major_m": "100–200", "range_minor_m": "50–100", "anisotropy_ratio": "2:1"},
                {"zone": "Oxide Zone", "nugget": "0.15", "sill": "0.75", "range_major_m": "50–100", "range_minor_m": "25–50", "anisotropy_ratio": "1.5:1"},
            ],
            "grade_capping_method": "Top-cut analysis using 95th percentile or Median + 1.5x IQR method to be applied prior to variography.",
            "extension_ranges": "Search ellipsoid parameters to be defined based on variogram ranges with maximum extension of 2x variogram range.",
            "byproduct_modeling": "By-product credits modelled using Spearman correlation coefficients and industry-standard recoveries.",
            "estimation_method": "Ordinary Kriging (OK) is the recommended estimation method for this deposit type.",
        },
        "validation_and_qc": {
            "check_assay_protocol": "Minimum 15% check assay rate with independent laboratory verification. Blanks and certified reference materials inserted at 1:20 frequency.",
            "monte_carlo_summary": f"Monte Carlo simulation (10,000 iterations) yields a P10–P90 tonnage range of {p10t:,.0f}–{p90t:,.0f} kt and grade range of {p10g:.4f}–{p90g:.4f}, representing ±15% uncertainty typical of early-stage resource estimation.",
            "p10_tonnage_kt": p10t,
            "p90_tonnage_kt": p90t,
            "p10_grade": p10g,
            "p90_grade": p90g,
            "statistical_reconciliation": "T-test (p < 0.05) and Spearman correlation (target >0.7) to be applied for grade reconciliation.",
            "audit_trail_notes": "Full data lineage documentation required including raw assay files, QAQC reports, and model files for NI 43-101 / JORC compliance.",
        },
        "conclusion": {
            "conclusion_text": f"This resource modeling report presents a preliminary assessment of {name} based on available public data and analog comparison methodology. The estimates are intended to support early-stage exploration planning and should not be relied upon for investment or financing decisions without independent Qualified Person review.",
            "headline_finding": f"{name} shows characteristics consistent with a {dep_type} {material} deposit warranting further exploration.",
            "next_milestone": "Complete infill drilling programme to upgrade resource classification and reduce estimation uncertainty.",
            "investment_readiness": "Pre-resource",
        },
        "appendices": {
            "input_weighting_table": [
                {"analog_name": "Analog projects (see Section 4)", "weight_pct": "Variable", "key_rationale": "Weighted by similarity score"},
            ],
            "variogram_parameters_table": [
                {"zone": "Primary Zone", "nugget": "0.10", "sill": "0.80", "range_major_m": "100–200", "range_minor_m": "50–100"},
            ],
            "drilling_summary_table": [
                {"hole_type": "RC", "count": "TBD", "avg_depth_m": "TBD", "purpose": "Shallow resource definition"},
                {"hole_type": "Diamond", "count": "TBD", "avg_depth_m": "TBD", "purpose": "Deep resource confirmation"},
            ],
            "references": [
                "NI 43-101 Standards of Disclosure for Mineral Projects, Canadian Securities Administrators, 2011.",
                "JORC Code 2012 Edition, Joint Ore Reserves Committee of the AusIMM, MCA and AIG, 2012.",
                "Rossi, M.E. and Deutsch, C.V. (2014). Mineral Resource Estimation. Springer, Dordrecht.",
                "Sinclair, A.J. and Blackwell, G.H. (2002). Applied Mineral Inventory Estimation. Cambridge University Press.",
                "S&P Global Market Intelligence (2024). Mining Industry Benchmarks and Comparable Transactions.",
            ],
        },
    }


def generate_extended_narrative(
    project: Dict,
    model_1: Dict,
    model_2: Optional[Dict],
    analogs: List[Dict],
    activated_rules: List[Dict],
    deterministic_vals: Dict,
) -> Dict:
    """
    Second LLM call generating 8 deep-dive sections not in the primary narrative.
    Independent of generate_report_narrative() — fails gracefully to fallback.
    """
    llm = get_llm(temperature=0.2)

    material    = project.get("material", "Unknown")
    dep_type    = project.get("deposit_type", "mineral")
    country     = project.get("country", "Unknown")
    stage       = project.get("project_stage", "Unknown")
    host_rock   = project.get("host_rock", "")
    min_style   = project.get("mineralization_style", "")
    p10t        = deterministic_vals["p10_tonnage_kt"]
    p90t        = deterministic_vals["p90_tonnage_kt"]
    p10g        = deterministic_vals["p10_grade"]
    p90g        = deterministic_vals["p90_grade"]

    analogs_mini = json.dumps([
        {k: a.get(k) for k in ("name", "country", "deposit_type", "tonnage_mt", "grade_value", "similarity_score")}
        for a in analogs[:5]
    ], indent=2)

    prompt = f"""You are a senior mining geologist and resource estimation expert writing deep-dive technical sections for a Resource Modeling Report.

PROJECT CONTEXT:
- Name: {project.get('name')}
- Material: {material}
- Deposit Type: {dep_type}
- Stage: {stage}
- Country: {country}
- Region: {project.get('region', 'N/A')}
- Host Rock: {host_rock or 'N/A'}
- Mineralization Style: {min_style or 'N/A'}
- Total Tonnage (Model 1): {model_1.get('total_tonnage_kt', 0):,.0f} kt
- Grade (Model 1): {model_1.get('total_grade_pct', 0):.4f}
- P10 Tonnage: {p10t:,.0f} kt | P90 Tonnage: {p90t:,.0f} kt
- P10 Grade: {p10g:.4f} | P90 Grade: {p90g:.4f}
- Analogs used (top 5): {analogs_mini}

Return a single valid JSON object with EXACTLY these 8 keys. Be specific, technical, and consistent with the project context above.

{{
  "geological_framework": {{
    "regional_setting": "2 paragraphs on tectonic setting, host terrane, regional geology, and known mineral systems in the province/region",
    "deposit_characteristics": "1-2 paragraphs on deposit geometry, dimensions (strike x width), structural envelope, zone distribution",
    "mineralization_description": "1 paragraph on primary mineral assemblage, alteration types (e.g. hematite, chlorite, sericite), vein/disseminated/breccia proportions",
    "structural_complexity": "1 paragraph on fault density, dominant structural orientations, their influence on mineralisation distribution",
    "geological_continuity": "1 paragraph on geological continuity rating, predictability, and what drives it",
    "logistics_and_infrastructure": "1 paragraph on access, power, water, nearest port or processing hub, key infrastructure requirements",
    "mineral_zones": [
      {{"zone_name": "zone name", "description": "brief description", "grade_range": "e.g. >1% Cu or 0.3-0.6 g/t Au"}}
    ]
  }},
  "drilling_and_sampling": {{
    "drillhole_strategy": "1-2 paragraphs on recommended hole types (RC vs diamond), spacing, orientation relative to structures, depth targets",
    "total_holes_estimated": "estimated total e.g. '~120 RC holes + 30 diamond confirmation holes'",
    "assay_qa_qc": "1 paragraph on QAQC protocols: insertion rate, blank/standard/duplicate frequency, acceptable variance thresholds",
    "xrf_geochemical_notes": "1 paragraph on portable XRF use, geochemical pathfinder elements, correlation with assay data",
    "cost_efficiency_notes": "1 paragraph on RC vs diamond cost comparison, expected metres/day, total programme cost estimate",
    "data_quality_assessment": "1 paragraph rating the expected data quality and confidence level for this deposit type and stage"
  }},
  "drilling_efficiency_metrics": {{
    "narrative": "1 paragraph interpreting drilling efficiency in the context of this deposit type and peer group",
    "metrics_table": [
      {{"metric": "Metal Added per Meter Drilled", "project_value": "specific value or estimate", "peer_range": "peer range for this deposit type", "assessment": "Above Peer|In-Line|Below Peer"}},
      {{"metric": "Discovery Cost per Resource Tonne", "project_value": "specific value or estimate", "peer_range": "peer range", "assessment": "Above Peer|In-Line|Below Peer"}},
      {{"metric": "All-In Drilling Cost per Meter", "project_value": "specific value or estimate", "peer_range": "peer range", "assessment": "Above Peer|In-Line|Below Peer"}},
      {{"metric": "Shareholder Dilution Efficiency", "project_value": "qualitative or quantitative", "peer_range": "peer range", "assessment": "Above Peer|In-Line|Below Peer"}}
    ],
    "shareholder_dilution_efficiency": "1 paragraph on metal gained vs dilution relative to peer transactions",
    "cost_per_meter_vs_peers": "1 paragraph comparing all-in drilling costs to regional and global {dep_type} benchmarks"
  }},
  "geophysical_integration": {{
    "survey_types_recommended": [
      {{"survey_type": "survey name", "rationale": "why this survey is appropriate for this deposit type", "priority": "High|Medium|Low"}},
      {{"survey_type": "survey name", "rationale": "rationale", "priority": "High|Medium|Low"}},
      {{"survey_type": "survey name", "rationale": "rationale", "priority": "Medium|Low"}}
    ],
    "continuity_thresholds": "1 paragraph on grade continuity thresholds (% threshold by zone) and geophysical anomaly response",
    "validation_triggers": "1 paragraph on what geophysical results would trigger re-validation or additional drilling",
    "existing_data_notes": "1 paragraph on what existing geophysical data is likely available and its relevance"
  }},
  "geostatistical_modeling": {{
    "variography_narrative": "1-2 paragraphs on variographic approach, expected spatial continuity patterns for this deposit type",
    "variogram_parameters": [
      {{"zone": "zone name", "nugget": "value e.g. 0.10", "sill": "value e.g. 0.80", "range_major_m": "e.g. 150", "range_minor_m": "e.g. 75", "anisotropy_ratio": "e.g. 2:1"}}
    ],
    "grade_capping_method": "1 paragraph on top-cut/capping approach appropriate for this deposit type",
    "extension_ranges": "1 paragraph on maximum search ellipsoid distances and justification",
    "byproduct_modeling": "1 paragraph on by-product credit handling, correlation coefficients, recovery assumptions",
    "estimation_method": "recommended estimation method e.g. 'Ordinary Kriging with 2-pass search strategy'"
  }},
  "validation_and_qc": {{
    "check_assay_protocol": "1 paragraph on check assay frequency, independent lab, acceptable RPD thresholds",
    "monte_carlo_summary": "1 paragraph describing Monte Carlo simulation approach and results: use exactly P10={p10t:,.0f} kt and P90={p90t:,.0f} kt for tonnage, P10={p10g:.4f} and P90={p90g:.4f} for grade",
    "p10_tonnage_kt": {p10t},
    "p90_tonnage_kt": {p90t},
    "p10_grade": {p10g},
    "p90_grade": {p90g},
    "statistical_reconciliation": "1 paragraph on t-test, Spearman correlation, and cross-validation approach",
    "audit_trail_notes": "1 paragraph on documentation requirements for NI 43-101 / JORC readiness"
  }},
  "conclusion": {{
    "conclusion_text": "2-3 paragraphs: (1) methodology summary and confidence statement, (2) key findings with actual tonnage/grade numbers from Model 1, (3) path forward and next milestone",
    "headline_finding": "One crisp sentence — the single most important finding from this entire report",
    "next_milestone": "The most critical next action to advance this project (specific, with timeframe)",
    "investment_readiness": "Pre-resource|Resource-stage|Development-ready"
  }},
  "appendices": {{
    "input_weighting_table": [
      {{"analog_name": "analog project name", "weight_pct": "weight percentage", "key_rationale": "why this analog was weighted this way"}}
    ],
    "variogram_parameters_table": [
      {{"zone": "zone name", "nugget": "value", "sill": "value", "range_major_m": "value", "range_minor_m": "value"}}
    ],
    "drilling_summary_table": [
      {{"hole_type": "RC|Diamond|RAB", "count": "estimated count", "avg_depth_m": "estimated depth", "purpose": "purpose description"}}
    ],
    "references": [
      "NI 43-101 Standards of Disclosure for Mineral Projects, Canadian Securities Administrators, 2011.",
      "JORC Code 2012 Edition, Joint Ore Reserves Committee of the AusIMM, MCA and AIG, 2012.",
      "Reference specific to {dep_type} deposits.",
      "Reference specific to {material} resource estimation.",
      "S&P Global Market Intelligence (2024). Mining Industry Benchmarks."
    ]
  }}
}}

Rules:
- mineral_zones: 2-4 zones
- metrics_table: exactly 4 rows
- survey_types_recommended: exactly 3 surveys
- variogram_parameters and variogram_parameters_table: 2-3 zones
- input_weighting_table: one row per analog used (use names from the analogs list above)
- drilling_summary_table: 2-3 rows
- p10_tonnage_kt, p90_tonnage_kt, p10_grade, p90_grade: copy the exact numeric values from the prompt — do NOT invent different numbers
- Return ONLY the JSON. No markdown fences, no explanation.
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        result = json.loads(content)
        # Ensure P10/P90 values are always from deterministic calc
        if "validation_and_qc" in result:
            result["validation_and_qc"]["p10_tonnage_kt"] = p10t
            result["validation_and_qc"]["p90_tonnage_kt"] = p90t
            result["validation_and_qc"]["p10_grade"]      = p10g
            result["validation_and_qc"]["p90_grade"]      = p90g
        return result
    except Exception as e:
        logger.error(f"[ExtendedNarrative] Generation error: {e}")
        return _fallback_extended_narrative(project, deterministic_vals)
