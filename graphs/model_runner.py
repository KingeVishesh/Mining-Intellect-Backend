"""
Graph: model_runner

Standalone pipeline that builds Model 1 (+ Model 2 when an official MRE
exists) WITHOUT generating a PDF or saving a reports row. Used by the
/projects-back "Build Models" button so the user can iterate on model
output without paying the cost of report generation.

Flow:
  load_project_and_analogs
    → check_analogs_present
        ↳ END if no analogs
    → load_rules → activate_rules
    → build_model_1 → build_model_2 → save_model_run → END

Persistence:
  - INSERT row into model_runs for each Model produced (history).
  - UPDATE projects with the latest Model values so the table view reflects them.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from nodes import supabase_ops, model_builder
from graphs.report_generator import (
    load_project_and_analogs_node,
    load_rules_node,
    activate_rules_node,
    build_model_1_node,
    build_model_2_node,
)

logger = logging.getLogger(__name__)


class ModelRunnerState(TypedDict, total=False):
    # Input
    project_id: str

    # Loaded
    project: Optional[Dict]
    analogs: List[Dict]
    all_rules: List[Dict]
    activated_rules: List[Dict]
    rule_effects: Dict

    # Models
    model_1: Optional[Dict]
    model_2: Optional[Dict]
    official_mre_row: Optional[Dict]

    # Output
    saved: bool
    error: Optional[str]


def check_analogs_present_node(state: ModelRunnerState) -> ModelRunnerState:
    """Block model building when there are no analogs to score against."""
    if state.get("error"):
        return {}
    analogs = state.get("analogs") or []
    if not analogs:
        msg = "Cannot run models: project has no analogs. Run analog finder first."
        logger.warning(f"[model_runner] {msg}")
        return {"error": msg}
    return {}


def _route_after_check(state: ModelRunnerState) -> str:
    return END if state.get("error") else "load_rules"


def _split_mi_by_stage(stage: Optional[str]) -> tuple[float, float]:
    """Within the Measured+Indicated bucket the model produces, return
    (measured_fraction, indicated_fraction). Drill density goes up with
    stage maturity, so later stages have more Measured.
    """
    s = (stage or "").lower()
    if any(k in s for k in ("production", "producing", "operating", "operation", "construction")):
        return (0.70, 0.30)
    if any(k in s for k in ("bankable", "definitive feasibility", " dfs", " bfs")) or (
        "feasibility" in s and "pre" not in s and "prefeasibility" not in s
    ):
        return (0.50, 0.50)
    if any(k in s for k in ("pre-feasibility", "prefeasibility", " pfs")):
        return (0.30, 0.70)
    if any(k in s for k in ("pea", "scoping", "preliminary economic", "preliminary assessment")):
        return (0.15, 0.85)
    if "exploration" in s or not s:
        return (0.00, 1.00)
    return (0.20, 0.80)


def _round(x: Optional[float], digits: int = 4) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), digits)
    except (TypeError, ValueError):
        return None


def _fields_from_model(project: Dict, model: Dict, is_post_mre: bool) -> Dict:
    """Translate a Model 1 / Model 2 output dict into the 12 columns we persist.

    Per category (Measured / Indicated / Inferred): tonnage_mt, grade,
    contained. Plus totals (tonnage_mt, grade_value, total_contained).
    Conviction is the tier code ("PRE-1".."PRE-5" / "POST-1".."POST-5");
    conviction_tier carries the full human label.

    The underlying model already splits its output into M+I vs Inferred
    (industry standard). We split the M+I bucket between Measured and
    Indicated using stage-keyed ratios — production-stage projects skew
    Measured-heavy, exploration-stage projects skew Indicated.
    """
    mi_kt   = float(model.get("mi_tonnage_kt") or 0)
    mi_g    = model.get("mi_grade_pct")
    mi_cont = model.get("mi_contained_mlb")
    inf_kt  = float(model.get("inferred_tonnage_kt") or 0)
    inf_g   = model.get("inferred_grade_pct")
    inf_c   = model.get("inferred_contained_mlb")
    tot_kt  = float(model.get("total_tonnage_kt") or 0)
    tot_g   = model.get("total_grade_pct")
    tot_c   = model.get("total_contained_mlb")

    m_frac, i_frac = _split_mi_by_stage(project.get("project_stage"))

    measured_mt  = _round((mi_kt * m_frac) / 1000.0) if mi_kt else None
    indicated_mt = _round((mi_kt * i_frac) / 1000.0) if mi_kt else None
    inferred_mt  = _round(inf_kt / 1000.0) if inf_kt else None
    total_mt     = _round(tot_kt / 1000.0) if tot_kt else None

    measured_contained  = _round((float(mi_cont) * m_frac), 3) if mi_cont is not None else None
    indicated_contained = _round((float(mi_cont) * i_frac), 3) if mi_cont is not None else None
    inferred_contained  = _round(inf_c, 3) if inf_c is not None else None
    total_contained     = _round(tot_c, 3) if tot_c is not None else None

    conviction_num = float(model.get("conviction_pct") or 0)
    if is_post_mre:
        tier_code, tier_label = model_builder._compute_post_tier(conviction_num, project)
    else:
        tier_code, tier_label = model_builder._compute_pre_tier(conviction_num)

    return {
        # Per-category tonnage (existing column names — column is "Tonnage (Mt)")
        "measured_resource_mt":  measured_mt,
        "indicated_resource_mt": indicated_mt,
        "inferred_resource_mt":  inferred_mt,
        # Per-category grade (same units as project.grade_unit, e.g. "g/t Au")
        "measured_grade":        _round(mi_g),
        "indicated_grade":       _round(mi_g),
        "inferred_grade":        _round(inf_g),
        # Per-category contained metal (Moz for precious metals, Mlb otherwise)
        "measured_contained":    measured_contained,
        "indicated_contained":   indicated_contained,
        "inferred_contained":    inferred_contained,
        # Totals
        "tonnage_mt":            total_mt,
        "grade_value":           _round(tot_g),
        "total_contained":       total_contained,
        # Conviction
        "conviction_score":      tier_code,
        "conviction_tier":       f"{tier_code}: {tier_label}",
    }


def save_model_run_node(
    state: ModelRunnerState,
    config: Optional[Dict] = None,
) -> ModelRunnerState:
    """Persist Model 1 (+ Model 2 if built) to model_runs and overwrite projects."""
    if state.get("error"):
        return {}

    project = state.get("project") or {}
    project_id = state["project_id"]
    cfg = (config or {}).get("configurable") or {}
    thread_id = cfg.get("thread_id")
    run_id = cfg.get("run_id")

    model_1 = state.get("model_1")
    model_2 = state.get("model_2")

    latest_fields: Optional[Dict] = None

    if model_1:
        fields_1 = _fields_from_model(project, model_1, is_post_mre=False)
        supabase_ops.save_model_run(
            project_id=project_id,
            model_type="model_1",
            fields=fields_1,
            model_output_json=model_1,
            thread_id=thread_id,
            run_id=run_id,
        )
        latest_fields = fields_1

    if model_2:
        fields_2 = _fields_from_model(project, model_2, is_post_mre=True)
        supabase_ops.save_model_run(
            project_id=project_id,
            model_type="model_2",
            fields=fields_2,
            model_output_json=model_2,
            thread_id=thread_id,
            run_id=run_id,
        )
        latest_fields = fields_2  # Model 2 wins as "latest" when it exists

    if latest_fields:
        supabase_ops.update_project_latest_model(project_id, latest_fields)

    return {"saved": True, "error": None}


# ── Graph ──────────────────────────────────────────────────────────────────────

builder = StateGraph(ModelRunnerState)
builder.add_node("load_project_and_analogs", load_project_and_analogs_node)
builder.add_node("check_analogs_present", check_analogs_present_node)
builder.add_node("load_rules", load_rules_node)
builder.add_node("activate_rules", activate_rules_node)
builder.add_node("build_model_1", build_model_1_node)
builder.add_node("build_model_2", build_model_2_node)
builder.add_node("save_model_run", save_model_run_node)

builder.set_entry_point("load_project_and_analogs")
builder.add_edge("load_project_and_analogs", "check_analogs_present")
builder.add_conditional_edges("check_analogs_present", _route_after_check, {
    "load_rules": "load_rules",
    END: END,
})
builder.add_edge("load_rules", "activate_rules")
builder.add_edge("activate_rules", "build_model_1")
builder.add_edge("build_model_1", "build_model_2")
builder.add_edge("build_model_2", "save_model_run")
builder.add_edge("save_model_run", END)

graph = builder.compile()
