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


def _category_bucket(project: Dict) -> str:
    """Map the project's reported resource_category to a column key.
    Defaults to 'inferred' (most conservative) when category is unknown.
    """
    cat = (project.get("resource_category") or "").strip().lower()
    if "measured" in cat:
        return "measured_resource_mt"
    if "indicated" in cat:
        return "indicated_resource_mt"
    return "inferred_resource_mt"


def _fields_from_model(project: Dict, model: Dict, is_post_mre: bool) -> Dict:
    """
    Translate a Model 1 or Model 2 output dict into the 7 columns we persist.

    Current model builder produces a total — we assign that total to whichever
    M/I/I bucket matches project.resource_category (default Inferred). tonnage_mt
    becomes the sum (which equals the single bucket), grade_value is the model's
    grade, conviction_score the conviction.
    """
    total_kt = float(model.get("total_tonnage_kt") or 0)
    total_mt = round(total_kt / 1000.0, 4) if total_kt else None
    grade = model.get("total_grade_pct")
    conviction = float(model.get("conviction_pct") or 0)

    bucket = _category_bucket(project)
    cats = {
        "measured_resource_mt": None,
        "indicated_resource_mt": None,
        "inferred_resource_mt": None,
    }
    cats[bucket] = total_mt

    if is_post_mre:
        tier_code, tier_label = model_builder._compute_post_tier(conviction, project)
    else:
        tier_code, tier_label = model_builder._compute_pre_tier(conviction)

    return {
        **cats,
        "tonnage_mt": total_mt,
        "grade_value": grade,
        "conviction_score": conviction,
        "conviction_tier": f"{tier_code}: {tier_label}",
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
