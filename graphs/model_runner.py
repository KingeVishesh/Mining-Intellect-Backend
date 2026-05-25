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


def _round(x: Optional[float], digits: int = 4) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), digits)
    except (TypeError, ValueError):
        return None


def _fields_from_model(project: Dict, model: Dict, is_post_mre: bool) -> Dict:
    """Translate a Model 1 / Model 2 output dict into the 9 columns we persist.

    Industry resource statements bundle Measured + Indicated together as
    "M&I" — the model itself doesn't split them, and re-splitting them
    via a stage heuristic added noise without information. We now store
    M&I as a single triple (tonnage, grade, contained) alongside the
    Inferred triple and the totals.

    Conviction is the tier code ("PRE-1".."PRE-5" / "POST-1".."POST-5");
    conviction_tier carries the full human label.
    """
    mi_kt   = float(model.get("mi_tonnage_kt") or 0)
    mi_g    = model.get("mi_grade_pct")
    mi_cont = model.get("mi_contained_mlb")
    inf_kt  = float(model.get("inferred_tonnage_kt") or 0)
    inf_g   = model.get("inferred_grade_pct")
    inf_c   = model.get("inferred_contained_mlb")

    mi_mt       = _round(mi_kt / 1000.0) if mi_kt else None
    inferred_mt = _round(inf_kt / 1000.0) if inf_kt else None

    # Totals are recomputed from the per-category values so the table is
    # internally consistent: total tonnage = M&I + Inferred, total contained
    # = M&I + Inferred, and avg grade is tonnage-weighted so it equals
    # total_contained / total_tonnage in the right units.
    total_kt = mi_kt + inf_kt
    total_mt = _round(total_kt / 1000.0) if total_kt else None

    mi_cont_f  = float(mi_cont) if mi_cont is not None else 0.0
    inf_cont_f = float(inf_c)   if inf_c   is not None else 0.0
    total_contained = (
        _round(mi_cont_f + inf_cont_f, 3)
        if (mi_cont is not None or inf_c is not None) else None
    )

    mi_g_f  = float(mi_g) if mi_g is not None else 0.0
    inf_g_f = float(inf_g) if inf_g is not None else 0.0
    avg_grade = (
        _round((mi_g_f * mi_kt + inf_g_f * inf_kt) / total_kt)
        if total_kt > 0 and (mi_g is not None or inf_g is not None) else None
    )

    conviction_num = float(model.get("conviction_pct") or 0)
    if is_post_mre:
        tier_code, tier_label = model_builder._compute_post_tier(conviction_num, project)
    else:
        tier_code, tier_label = model_builder._compute_pre_tier(conviction_num)

    return {
        # Measured + Indicated combined (M&I)
        "mi_tonnage_mt":         mi_mt,
        "mi_grade":              _round(mi_g),
        "mi_contained":          _round(mi_cont, 3),
        # Inferred
        "inferred_resource_mt":  inferred_mt,
        "inferred_grade":        _round(inf_g),
        "inferred_contained":    _round(inf_c, 3),
        # Totals — derived as sums / weighted average for arithmetic consistency
        "tonnage_mt":            total_mt,
        "grade_value":           avg_grade,
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
