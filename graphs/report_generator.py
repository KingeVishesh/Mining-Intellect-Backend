"""
Graph 3: report_generator

Flow:
  load_project_and_analogs → load_rules → activate_rules
  → build_model_1 → build_model_2 (if official MRE exists)
  → INTERRUPT(human_review_model) → generate_report → save_report → END

Input:  { project_id }
Output: Full MiningReport JSON saved to Supabase reports table
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from nodes import supabase_ops, rules_engine, model_builder
from schemas.report import MiningReport, ResourceEstimates, ComparisonTableRow

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class ReportState(TypedDict, total=False):
    # Input
    project_id: str

    # Intermediate
    project: Optional[Dict]
    analogs: List[Dict]
    all_rules: List[Dict]
    activated_rules: List[Dict]
    rule_effects: Dict

    # Models
    model_1: Optional[Dict]
    model_2: Optional[Dict]
    official_mre_row: Optional[Dict]

    # Human review
    human_approved: bool
    human_model_edits: Dict

    # Report
    report_json: Optional[Dict]
    report_id: Optional[str]

    # Output
    saved: bool
    error: Optional[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def load_project_and_analogs_node(state: ReportState) -> ReportState:
    """Load project and its approved analogs from Supabase."""
    project_id = state["project_id"]

    project = supabase_ops.get_project(project_id)
    if not project:
        return {"error": f"Project {project_id} not found"}

    analogs = supabase_ops.get_analogs(project_id)
    logger.info(f"[load] Project: {project.get('name')} | Analogs: {len(analogs)}")
    return {"project": project, "analogs": analogs, "error": None}


def load_rules_node(state: ReportState) -> ReportState:
    """Load compiled rules for the project's material."""
    if state.get("error"):
        return {}
    material = state["project"].get("material", "")
    all_rules = rules_engine.load_rules(material)
    return {"all_rules": all_rules}


def activate_rules_node(state: ReportState) -> ReportState:
    """LLM selects the rules relevant to this project."""
    if state.get("error"):
        return {}
    project = state["project"]
    all_rules = state.get("all_rules", [])
    activated = rules_engine.activate_rules(project, all_rules)
    return {"activated_rules": activated}


def build_model_1_node(state: ReportState) -> ReportState:
    """Build Model 1 (Independent) using analogs + rules."""
    if state.get("error"):
        return {}

    project = state["project"]
    analogs = state.get("analogs", [])
    activated_rules = state.get("activated_rules", [])

    # Compute rule multipliers
    # Use project tonnage/grade as base if available, otherwise analogs provide the base
    base_tonnage = float(project.get("tonnage_mt") or 0) * 1000  # Mt -> kt
    base_grade = float(project.get("grade_value") or 0)

    rule_effects = rules_engine.apply_rule_multipliers(
        base_tonnage=base_tonnage or 1000,  # fallback so multipliers have a base
        base_grade=base_grade or 1.0,
        activated_rules=activated_rules,
    )

    model_1 = model_builder.build_model_1(analogs, project, rule_effects)
    official_mre_row = model_builder.build_official_mre_row(project)

    return {"model_1": model_1, "rule_effects": rule_effects, "official_mre_row": official_mre_row}


def build_model_2_node(state: ReportState) -> ReportState:
    """Build Model 2 (Updated) if official MRE exists."""
    if state.get("error"):
        return {}

    project = state["project"]
    model_1 = state.get("model_1")
    if not model_1:
        return {}

    official_mre = None
    if project.get("tonnage_mt") and project.get("grade_value"):
        official_mre = {
            "tonnage_mt": project["tonnage_mt"],
            "grade_value": project["grade_value"],
        }

    model_2 = model_builder.build_model_2(model_1, project, official_mre)
    logger.info(f"[model2] built={'yes' if model_2 else 'no (no official MRE)'}")
    return {"model_2": model_2}


def human_review_model_node(state: ReportState) -> ReportState:
    """
    Runs after the interrupt_before pause is resumed.
    The frontend updated state with human_approved + human_model_edits before triggering this run.
    """
    approved = state.get("human_approved", False)
    edits = state.get("human_model_edits", {})
    return {"human_approved": approved, "human_model_edits": edits}


def generate_report_node(state: ReportState) -> ReportState:
    """Generate the full MiningReport JSON using models + LLM narrative."""
    if not state.get("human_approved"):
        logger.info("[generate_report] Human rejected — not generating")
        return {"report_json": None}

    project = state["project"]
    analogs = state.get("analogs", [])
    activated_rules = state.get("activated_rules", [])
    model_1 = state.get("model_1", {})
    model_2 = state.get("model_2")
    official_mre_row = state.get("official_mre_row")
    human_edits = state.get("human_model_edits", {})

    # Apply any human corrections to model 1
    if human_edits.get("model_1"):
        model_1.update(human_edits["model_1"])
    if human_edits.get("model_2") and model_2:
        model_2.update(human_edits["model_2"])

    # Build comparison table
    comparison_table = [model_1]
    if model_2:
        comparison_table.append(model_2)
    if official_mre_row:
        comparison_table.append(official_mre_row)

    # Generate LLM narrative
    narrative = model_builder.generate_report_narrative(
        project, model_1, model_2, analogs, activated_rules
    )

    # Assemble final report
    report = MiningReport(
        metadata={
            "project_name": project.get("name"),
            "material": project.get("material"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "full",
        },
        executive_summary=narrative.get("executive_summary", {}),
        project_overview=narrative.get("project_overview", {}),
        resource_estimates=ResourceEstimates(
            comparison_table=[ComparisonTableRow(**row) for row in comparison_table],
            independent_analysis={
                "confidence_pct": model_1.get("conviction_pct", 0),
                "key_factors": model_1.get("analogs_used", [])[:3],
            },
            updated_analysis={
                "confidence_pct": (model_2 or {}).get("conviction_pct", 0),
                "key_factors": [(model_2 or {}).get("description", "No updated model")],
            },
            compliance_summary=[
                "Estimates are internal MI models — NOT NI 43-101 or JORC compliant.",
                "For investment decisions, rely only on officially filed technical reports.",
            ],
        ),
        actionable_recommendations=narrative.get("actionable_recommendations", []),
        lessons_summary={
            "total_lessons_applied": len(activated_rules),
            "high_confidence_lessons": sum(
                1 for r in activated_rules if (r.get("weight") or 0) >= 0.7
            ),
        },
        key_uncertainties_and_strengths=narrative.get("key_uncertainties_and_strengths", {}),
    )

    report_json = report.model_dump()
    logger.info(f"[generate_report] Report assembled for {project.get('name')}")
    return {"report_json": report_json}


def save_report_node(state: ReportState) -> ReportState:
    """Save the report to Supabase."""
    report_json = state.get("report_json")
    if not report_json:
        return {"saved": False}

    project = state["project"]
    meta = {
        "report_type": "full",
        "material": project.get("material"),
        "deposit_type": project.get("deposit_type"),
    }

    try:
        report_id = supabase_ops.save_report(state["project_id"], report_json, meta)

        # Update project to mark models as built
        supabase_ops.upsert_project({
            "id": state["project_id"],
            "has_model_1": state.get("model_1") is not None,
            "has_model_2": state.get("model_2") is not None,
        })

        logger.info(f"[save_report] Saved report {report_id}")
        return {"report_id": report_id, "saved": True, "error": None}
    except Exception as e:
        logger.error(f"[save_report] Error: {e}")
        return {"saved": False, "error": str(e)}


# ── Graph ─────────────────────────────────────────────────────────────────────

def should_continue(state: ReportState) -> str:
    return END if state.get("error") else "load_rules"


builder = StateGraph(ReportState)

builder.add_node("load_project_and_analogs", load_project_and_analogs_node)
builder.add_node("load_rules", load_rules_node)
builder.add_node("activate_rules", activate_rules_node)
builder.add_node("build_model_1", build_model_1_node)
builder.add_node("build_model_2", build_model_2_node)
builder.add_node("human_review_model", human_review_model_node)
builder.add_node("generate_report", generate_report_node)
builder.add_node("save_report", save_report_node)

builder.set_entry_point("load_project_and_analogs")
builder.add_conditional_edges(
    "load_project_and_analogs",
    should_continue,
    {"load_rules": "load_rules", END: END},
)
builder.add_edge("load_rules", "activate_rules")
builder.add_edge("activate_rules", "build_model_1")
builder.add_edge("build_model_1", "build_model_2")
builder.add_edge("build_model_2", "human_review_model")
builder.add_edge("human_review_model", "generate_report")
builder.add_edge("generate_report", "save_report")
builder.add_edge("save_report", END)

graph = builder.compile(interrupt_before=["human_review_model"])
