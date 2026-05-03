"""
Graph: pipeline_orchestrator

Chains project_research → analog_finder → report_generator via LangGraph Cloud API calls.
Each child graph runs independently and can still be triggered separately.

Interrupt points (interrupt_before):
  "start_analogs"            → paused after research child started; frontend polls research child
  "load_analogs_for_review"  → paused after analogs child started; frontend polls analogs child
  "analog_review"            → paused for human to approve/reject analogs
  "finalize"                 → paused after report child started; frontend polls report child
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, TypedDict

import requests
from langgraph.graph import StateGraph, END

from config import settings
from nodes import supabase_ops

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    project_id: str
    project_name: str
    material: str
    company: str
    stage: str
    research_thread_id: Optional[str]
    research_run_id: Optional[str]
    analogs_thread_id: Optional[str]
    analogs_run_id: Optional[str]
    report_thread_id: Optional[str]
    report_run_id: Optional[str]
    analogs_for_review: Optional[List[Dict]]
    approved_analogs: Optional[List[Dict]]
    rejected_analogs: Optional[List[Dict]]
    report_id: Optional[str]
    error: Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lg_headers() -> Dict:
    # LANGGRAPH_API_KEY if explicitly set; fall back to LANGCHAIN_API_KEY which
    # LangGraph Cloud injects automatically into every deployed graph's environment.
    key = settings.langgraph_api_key or settings.langchain_api_key or ""
    return {
        "X-Api-Key": key,
        "Content-Type": "application/json",
    }


def _start_child_graph(assistant_id: str, input_data: Dict) -> tuple:
    """Create a LangGraph thread and start a run. Returns (thread_id, run_id)."""
    base = settings.langgraph_base_url
    headers = _lg_headers()

    thread_res = requests.post(f"{base}/threads", headers=headers, json={}, timeout=30)
    thread_res.raise_for_status()
    thread_id = thread_res.json()["thread_id"]

    run_res = requests.post(
        f"{base}/threads/{thread_id}/runs",
        headers=headers,
        json={"assistant_id": assistant_id, "input": input_data},
        timeout=30,
    )
    run_res.raise_for_status()
    run_id = run_res.json()["run_id"]

    logger.info(f"[orchestrator] Started {assistant_id}: thread={thread_id}, run={run_id}")
    return thread_id, run_id


# ── Nodes ─────────────────────────────────────────────────────────────────────

def init_pipeline_node(state: PipelineState) -> PipelineState:
    """Start the project_research child graph."""
    project_id = state["project_id"]
    try:
        thread_id, run_id = _start_child_graph("project_research", {
            "project_id": project_id,
            "project_name": state.get("project_name", ""),
            "material": state.get("material", ""),
            "company": state.get("company") or state.get("project_name", ""),
        })
        supabase_ops.save_pipeline_state(
            project_id=project_id,
            orchestrator_stage="research_running",
            research_thread_id=thread_id,
        )
        return {
            "stage": "research_running",
            "research_thread_id": thread_id,
            "research_run_id": run_id,
        }
    except Exception as e:
        logger.error(f"[init_pipeline] Error: {e}")
        return {"error": str(e), "stage": "error"}


def start_analogs_node(state: PipelineState) -> PipelineState:
    """Start the analog_finder child graph (called after research child is complete)."""
    if state.get("error"):
        return {}
    project_id = state["project_id"]
    try:
        thread_id, run_id = _start_child_graph("analog_finder", {"project_id": project_id})
        supabase_ops.save_pipeline_state(
            project_id=project_id,
            orchestrator_stage="analogs_running",
            analogs_thread_id=thread_id,
        )
        return {
            "stage": "analogs_running",
            "analogs_thread_id": thread_id,
            "analogs_run_id": run_id,
        }
    except Exception as e:
        logger.error(f"[start_analogs] Error: {e}")
        return {"error": str(e), "stage": "error"}


def load_analogs_for_review_node(state: PipelineState) -> PipelineState:
    """Read scored analogs from DB and surface them for human review."""
    if state.get("error"):
        return {}
    analogs = supabase_ops.get_analogs(state["project_id"])
    logger.info(f"[load_analogs] {len(analogs)} analogs ready for review")
    return {"stage": "analogs_review", "analogs_for_review": analogs}


def analog_review_node(state: PipelineState) -> PipelineState:
    """
    Process human analog review.
    approved_analogs / rejected_analogs are set by the human via the resume call.
    Falls back to approving all analogs if the human provided no edits.
    """
    approved = state.get("approved_analogs") or state.get("analogs_for_review", [])
    rejected = state.get("rejected_analogs") or []

    # Write the approved list back to DB so report_generator picks it up
    supabase_ops.save_approved_analogs(state["project_id"], approved)

    logger.info(f"[analog_review] Approved: {len(approved)}, Rejected: {len(rejected)}")
    return {
        "approved_analogs": approved,
        "rejected_analogs": rejected,
        "stage": "report_ready",
    }


def start_report_node(state: PipelineState) -> PipelineState:
    """Start the report_generator child graph."""
    if state.get("error"):
        return {}
    project_id = state["project_id"]
    try:
        thread_id, run_id = _start_child_graph("report_generator", {"project_id": project_id})
        supabase_ops.save_pipeline_state(
            project_id=project_id,
            orchestrator_stage="report_running",
            report_thread_id=thread_id,
        )
        return {
            "stage": "report_running",
            "report_thread_id": thread_id,
            "report_run_id": run_id,
        }
    except Exception as e:
        logger.error(f"[start_report] Error: {e}")
        return {"error": str(e), "stage": "error"}


def finalize_node(state: PipelineState) -> PipelineState:
    """Save report_analogs tracking data and mark pipeline complete."""
    if state.get("error"):
        return {}

    project_id = state["project_id"]
    approved = state.get("approved_analogs", [])
    rejected = state.get("rejected_analogs", [])

    report = supabase_ops.get_report(project_id)
    report_id = report["id"] if report else None

    if report_id and (approved or rejected):
        supabase_ops.save_report_analogs(
            report_id=report_id,
            project_id=project_id,
            approved=approved,
            rejected=rejected,
        )
        logger.info(
            f"[finalize] Saved analog tracking: {len(approved)} approved, {len(rejected)} rejected"
        )

    supabase_ops.save_pipeline_state(project_id=project_id, orchestrator_stage="complete")
    return {"stage": "complete", "report_id": report_id}


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(PipelineState)

builder.add_node("init_pipeline", init_pipeline_node)
builder.add_node("start_analogs", start_analogs_node)
builder.add_node("load_analogs_for_review", load_analogs_for_review_node)
builder.add_node("analog_review", analog_review_node)
builder.add_node("start_report", start_report_node)
builder.add_node("finalize", finalize_node)

builder.set_entry_point("init_pipeline")
builder.add_edge("init_pipeline", "start_analogs")
builder.add_edge("start_analogs", "load_analogs_for_review")
builder.add_edge("load_analogs_for_review", "analog_review")
builder.add_edge("analog_review", "start_report")
builder.add_edge("start_report", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile(
    interrupt_before=[
        "start_analogs",
        "load_analogs_for_review",
        "finalize",
    ]
)
