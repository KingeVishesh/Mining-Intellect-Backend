"""
Pipeline Orchestrator State
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    # Input
    project_id: str
    project_name: str
    material: str
    company: str

    # Pipeline stage tracking
    stage: str  # research_running | analogs_running | analogs_review | report_running | complete | error

    # Child graph thread / run tracking
    research_thread_id: Optional[str]
    research_run_id: Optional[str]
    analogs_thread_id: Optional[str]
    analogs_run_id: Optional[str]
    report_thread_id: Optional[str]
    report_run_id: Optional[str]

    # Analog review
    analogs_for_review: Optional[List[Dict]]   # populated before analog_review interrupt
    approved_analogs: Optional[List[Dict]]     # set by human on resume
    rejected_analogs: Optional[List[Dict]]     # set by human on resume

    # Final results
    report_id: Optional[str]
    error: Optional[str]
