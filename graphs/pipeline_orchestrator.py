"""
Graph: pipeline_orchestrator

Chains project_research → analog_finder → report_generator end-to-end.
Each node starts a child graph via the LangGraph Cloud API and then BLOCKS,
polling the child run every 20 s until it reaches a terminal status.

No interrupt_before — the pipeline is fully automatic. No frontend resume calls
are needed. The stage is written to Supabase at the start of each node so the
frontend can track progress even while LangGraph state is mid-checkpoint.
"""
from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, TypedDict

import requests
from langgraph.graph import StateGraph, END

from config import settings
from nodes import supabase_ops

logger = logging.getLogger(__name__)

# Max time (seconds) to wait for a single child run to finish.
_CHILD_TIMEOUT = 3600  # 60 min — generous for large projects
_POLL_INTERVAL = 20    # seconds between status polls


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
    key = settings.langgraph_api_key or settings.langchain_api_key or ""
    return {"X-Api-Key": key, "Content-Type": "application/json"}


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

    logger.info(f"[orchestrator] Started {assistant_id}: thread={thread_id} run={run_id}")
    return thread_id, run_id


def _wait_for_run(thread_id: str, run_id: str, label: str = "child") -> str:
    """
    Poll a child run every _POLL_INTERVAL seconds until it reaches a terminal status.
    Returns the final status string: "success", "error", or "interrupted".
    Raises TimeoutError if _CHILD_TIMEOUT is exceeded.
    """
    base = settings.langgraph_base_url
    headers = _lg_headers()
    elapsed = 0

    while elapsed < _CHILD_TIMEOUT:
        try:
            res = requests.get(
                f"{base}/threads/{thread_id}/runs/{run_id}",
                headers=headers,
                timeout=30,
            )
            if res.ok:
                status = res.json().get("status", "unknown")
                logger.info(f"[orchestrator] {label} status={status} elapsed={elapsed}s")
                if status in ("success", "error", "interrupted"):
                    return status
        except Exception as exc:
            logger.warning(f"[orchestrator] poll error for {label}: {exc}")

        time.sleep(_POLL_INTERVAL)
        elapsed += _POLL_INTERVAL

    raise TimeoutError(f"{label} timed out after {_CHILD_TIMEOUT}s (thread={thread_id})")


# ── Nodes ─────────────────────────────────────────────────────────────────────

def init_pipeline_node(state: PipelineState) -> PipelineState:
    """Start project_research child, block until it finishes."""
    project_id = state["project_id"]
    try:
        thread_id, run_id = _start_child_graph("project_research", {
            "project_id": project_id,
            "project_name": state.get("project_name", ""),
            "material": state.get("material", ""),
            "company": state.get("company") or state.get("project_name", ""),
        })
        # Write to DB immediately so frontend sees progress before LangGraph checkpoints
        supabase_ops.save_pipeline_state(
            project_id=project_id,
            orchestrator_stage="research_running",
            research_thread_id=thread_id,
        )

        final_status = _wait_for_run(thread_id, run_id, label="project_research")
        if final_status == "error":
            supabase_ops.save_pipeline_state(project_id=project_id, orchestrator_stage="error")
            return {"error": "project_research child failed", "stage": "error",
                    "research_thread_id": thread_id, "research_run_id": run_id}

        return {
            "stage": "research_running",
            "research_thread_id": thread_id,
            "research_run_id": run_id,
        }
    except Exception as e:
        logger.error(f"[init_pipeline] Error: {e}")
        supabase_ops.save_pipeline_state(project_id=project_id, orchestrator_stage="error")
        return {"error": str(e), "stage": "error"}


def start_analogs_node(state: PipelineState) -> PipelineState:
    """Start analog_finder child, block until it finishes."""
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

        final_status = _wait_for_run(thread_id, run_id, label="analog_finder")
        if final_status == "error":
            supabase_ops.save_pipeline_state(project_id=project_id, orchestrator_stage="error")
            return {"error": "analog_finder child failed", "stage": "error",
                    "analogs_thread_id": thread_id, "analogs_run_id": run_id}

        return {
            "stage": "analogs_running",
            "analogs_thread_id": thread_id,
            "analogs_run_id": run_id,
        }
    except Exception as e:
        logger.error(f"[start_analogs] Error: {e}")
        supabase_ops.save_pipeline_state(project_id=project_id, orchestrator_stage="error")
        return {"error": str(e), "stage": "error"}


def load_analogs_for_review_node(state: PipelineState) -> PipelineState:
    """Read scored analogs from DB (no human review — auto-approve all)."""
    if state.get("error"):
        return {}
    analogs = supabase_ops.get_analogs(state["project_id"])
    logger.info(f"[load_analogs] {len(analogs)} analogs loaded")
    return {"analogs_for_review": analogs}


def analog_review_node(state: PipelineState) -> PipelineState:
    """Auto-approve all analogs (no human interrupt in pipeline mode)."""
    approved = state.get("approved_analogs") or state.get("analogs_for_review") or []
    rejected = state.get("rejected_analogs") or []
    supabase_ops.save_approved_analogs(state["project_id"], approved)
    logger.info(f"[analog_review] Auto-approved {len(approved)} analogs")
    return {"approved_analogs": approved, "rejected_analogs": rejected}


def start_report_node(state: PipelineState) -> PipelineState:
    """Start report_generator child, block until it finishes."""
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

        final_status = _wait_for_run(thread_id, run_id, label="report_generator")
        if final_status == "error":
            supabase_ops.save_pipeline_state(project_id=project_id, orchestrator_stage="error")
            return {"error": "report_generator child failed", "stage": "error",
                    "report_thread_id": thread_id, "report_run_id": run_id}

        return {
            "stage": "report_running",
            "report_thread_id": thread_id,
            "report_run_id": run_id,
        }
    except Exception as e:
        logger.error(f"[start_report] Error: {e}")
        supabase_ops.save_pipeline_state(project_id=project_id, orchestrator_stage="error")
        return {"error": str(e), "stage": "error"}


def finalize_node(state: PipelineState) -> PipelineState:
    """Save analog tracking data and mark pipeline complete."""
    if state.get("error"):
        return {}

    project_id = state["project_id"]
    approved = state.get("approved_analogs") or []
    rejected = state.get("rejected_analogs") or []

    report = supabase_ops.get_report(project_id)
    report_id = report["id"] if report else None

    if report_id and (approved or rejected):
        supabase_ops.save_report_analogs(
            report_id=report_id,
            project_id=project_id,
            approved=approved,
            rejected=rejected,
        )
        logger.info(f"[finalize] Saved {len(approved)} approved + {len(rejected)} rejected analogs")

    supabase_ops.save_pipeline_state(project_id=project_id, orchestrator_stage="complete")
    logger.info(f"[finalize] Pipeline complete for project {project_id}")
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

# No interrupt_before — fully automatic end-to-end pipeline.
graph = builder.compile()
