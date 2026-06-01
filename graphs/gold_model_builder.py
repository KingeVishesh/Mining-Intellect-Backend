"""
Graph: gold_model_builder

A single-model pipeline that outsources gold-project M&I / Inferred
estimation to Parallel.ai's deep-research agent. The agent is given the
FULL project + analog context (drilling evidence, MRE breakdowns, etc.) so
it doesn't have to re-discover facts the rest of the pipeline already
extracted. It learns drilling -> MRE conversion ratios from the analog
cohort and applies them to the target's drilling profile.

Flow:
  load_project_and_analogs
    → check_analogs_present
        ↳ END if no analogs
    → fetch_drilling_evidence      (reuse model_runner helper)
    → fetch_inferred_evidence      (reuse model_runner helper)
    → call_parallel_gold_model     (the big Parallel.ai call)
    → save_model_run               (persist + overwrite latest projects.* fields)
    → END

Inputs:
  project_id            : str
  use_mre               : bool (default True) — when False, Parallel ignores
                          the project's published MRE and produces a blind
                          pre-MRE estimate (for backtesting).
  fetch_recent_drill_holes : bool — force-refresh cached drilling evidence.

Persistence:
  - INSERT row into model_runs with model_type = "parallel_pre_mre" or
    "parallel_post_mre". model_output_json carries the raw Parallel response
    so methodology + analog weights + sources are fully auditable.
  - UPDATE projects with the latest M&I / Inferred / total values so the
    /projects-back table reflects them.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from nodes import supabase_ops
from nodes.parallel_gold_model import parallel_gold_model_node
from graphs.report_generator import load_project_and_analogs_node
from graphs.model_runner import (
    check_analogs_present_node,
    fetch_drilling_evidence_node,
    fetch_inferred_evidence_node,
    _round,
    _contained_native_unit,
)

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class GoldModelBuilderState(TypedDict, total=False):
    # Input
    project_id: str
    # When False, Parallel pretends the official MRE doesn't exist. Default
    # True — incorporate the published MRE when present (post-MRE estimate).
    use_mre: bool
    # Forward to fetch_drilling_evidence_node so cached evidence can be
    # bypassed when the caller wants a fresh extraction.
    fetch_recent_drill_holes: bool

    # Loaded
    project: Optional[Dict]
    analogs: List[Dict]

    # Parallel output
    parallel_model: Optional[Dict]

    # Persistence
    saved: bool
    error: Optional[str]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _route_after_check(state: GoldModelBuilderState) -> str:
    return END if state.get("error") else "fetch_drilling_evidence"


def _route_after_parallel(state: GoldModelBuilderState) -> str:
    return END if state.get("error") else "save_model_run"


_GOLD_MATERIAL = "gold"


def _fields_from_parallel(project: Dict, parallel_out: Dict) -> Dict[str, Any]:
    """Translate Parallel's JSON into the column shape `save_model_run`
    expects. Mirrors model_runner._fields_from_model.
    """
    material = project.get("material") or _GOLD_MATERIAL

    mi_block = parallel_out.get("m_and_i") or {}
    inf_block = parallel_out.get("inferred") or {}

    mi_mt = mi_block.get("tonnage_mt")
    mi_g = mi_block.get("grade_gpt")
    inf_mt = inf_block.get("tonnage_mt")
    inf_g = inf_block.get("grade_gpt")

    mi_kt = float(mi_mt) * 1000.0 if mi_mt is not None else 0.0
    inf_kt = float(inf_mt) * 1000.0 if inf_mt is not None else 0.0

    # contained_moz reported by Parallel is in Moz; we persist contained in
    # the material's native unit (troy oz for gold). Convert Moz -> oz when
    # present; otherwise derive from tonnage × grade for consistency.
    mi_contained_oz = (
        float(mi_block["contained_moz"]) * 1_000_000.0
        if mi_block.get("contained_moz") is not None
        else _contained_native_unit(mi_kt, mi_g, material)
    )
    inf_contained_oz = (
        float(inf_block["contained_moz"]) * 1_000_000.0
        if inf_block.get("contained_moz") is not None
        else _contained_native_unit(inf_kt, inf_g, material)
    )

    total_kt = mi_kt + inf_kt
    total_mt = _round(total_kt / 1000.0) if total_kt else None
    total_contained = None
    if mi_contained_oz is not None or inf_contained_oz is not None:
        total_contained = _round((mi_contained_oz or 0.0) + (inf_contained_oz or 0.0), 3)

    mi_g_f = float(mi_g) if mi_g is not None else 0.0
    inf_g_f = float(inf_g) if inf_g is not None else 0.0
    avg_grade = (
        _round((mi_g_f * mi_kt + inf_g_f * inf_kt) / total_kt)
        if total_kt > 0 and (mi_g is not None or inf_g is not None)
        else None
    )

    conviction = (parallel_out.get("conviction") or {})
    conviction_level = conviction.get("level") or ""
    tier_code = f"PARALLEL-{conviction_level.upper()}" if conviction_level else "PARALLEL-UNKNOWN"
    tier_label = conviction.get("rationale") or "Parallel.ai deep-research estimate"

    return {
        # M&I
        "mi_tonnage_mt":       _round(mi_mt),
        "mi_grade":            _round(mi_g),
        "mi_contained":        _round(mi_contained_oz, 3),
        # Inferred
        "inferred_resource_mt": _round(inf_mt),
        "inferred_grade":       _round(inf_g),
        "inferred_contained":   _round(inf_contained_oz, 3),
        # Totals (derived for arithmetic consistency)
        "tonnage_mt":          total_mt,
        "grade_value":         avg_grade,
        "total_contained":     total_contained,
        # Conviction
        "conviction_score":    tier_code,
        "conviction_tier":     f"{tier_code}: {tier_label}",
        # Percentile / CV columns: Parallel doesn't produce them yet — leave null.
        "p10_tonnage_mt": None, "p50_tonnage_mt": None, "p90_tonnage_mt": None,
        "p10_grade": None, "p50_grade": None, "p90_grade": None,
        "p10_contained": None, "p50_contained": None, "p90_contained": None,
        "cv_contained": None,
        # Audit trail — the full Parallel response is also saved into
        # model_output_json below; this column gets the analogs-used trace
        # so downstream calibration can attribute residual error per analog.
        "signal_contributions_json": {
            "source": "parallel.ai",
            "anchor_used": parallel_out.get("anchor_used"),
            "methodology": parallel_out.get("methodology"),
            "analogs_used": parallel_out.get("analogs_used"),
            "analogs_rejected": parallel_out.get("analogs_rejected"),
            "sources": parallel_out.get("sources"),
        },
    }


# ── Nodes ────────────────────────────────────────────────────────────────────

def save_model_run_node(
    state: GoldModelBuilderState,
    config: Optional[Dict] = None,
) -> GoldModelBuilderState:
    """Persist the Parallel-produced model to model_runs + projects."""
    if state.get("error"):
        return {}

    project = state.get("project") or {}
    project_id = state["project_id"]
    parallel_out = state.get("parallel_model")
    if not parallel_out:
        return {"saved": False, "error": "No Parallel output to persist"}

    use_mre = bool(state.get("use_mre", True))
    model_type = "parallel_post_mre" if use_mre else "parallel_pre_mre"

    cfg = (config or {}).get("configurable") or {}
    thread_id = cfg.get("thread_id")
    run_id = cfg.get("run_id")

    fields = _fields_from_parallel(project, parallel_out)
    supabase_ops.save_model_run(
        project_id=project_id,
        model_type=model_type,
        fields=fields,
        model_output_json=parallel_out,
        thread_id=thread_id,
        run_id=run_id,
    )
    supabase_ops.update_project_latest_model(project_id, fields)
    return {"saved": True, "error": None}


# ── Graph wiring ─────────────────────────────────────────────────────────────

builder = StateGraph(GoldModelBuilderState)
builder.add_node("load_project_and_analogs", load_project_and_analogs_node)
builder.add_node("check_analogs_present", check_analogs_present_node)
builder.add_node("fetch_drilling_evidence", fetch_drilling_evidence_node)
builder.add_node("fetch_inferred_evidence", fetch_inferred_evidence_node)
builder.add_node("call_parallel_gold_model", parallel_gold_model_node)
builder.add_node("save_model_run", save_model_run_node)

builder.set_entry_point("load_project_and_analogs")
builder.add_edge("load_project_and_analogs", "check_analogs_present")
builder.add_conditional_edges("check_analogs_present", _route_after_check, {
    "fetch_drilling_evidence": "fetch_drilling_evidence",
    END: END,
})
builder.add_edge("fetch_drilling_evidence", "fetch_inferred_evidence")
builder.add_edge("fetch_inferred_evidence", "call_parallel_gold_model")
builder.add_conditional_edges("call_parallel_gold_model", _route_after_parallel, {
    "save_model_run": "save_model_run",
    END: END,
})
builder.add_edge("save_model_run", END)

graph = builder.compile()
