"""
Graph 2: analog_finder

Flow:
  load_project → db_analog_search → exa_analog_search → score_analogs
              → INTERRUPT(human_review) → save_analogs → END

Input:  { project_id }
Output: Approved analogs saved to Supabase (workflow_states.analogs_json)
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from nodes import exa_search, field_extractor, supabase_ops

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class AnalogState(TypedDict, total=False):
    # Input
    project_id: str

    # Intermediate
    project: Optional[Dict]
    db_analogs: List[Dict]
    exa_analogs: List[Dict]
    all_candidates: List[Dict]
    scored_analogs: List[Dict]

    # Human review
    human_approved: bool
    approved_analogs: List[Dict]

    # Output
    saved: bool
    error: Optional[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def load_project_node(state: AnalogState) -> AnalogState:
    """Fetch project data from Supabase."""
    project_id = state["project_id"]
    project = supabase_ops.get_project(project_id)
    if not project:
        return {"error": f"Project {project_id} not found in Supabase"}
    logger.info(f"[load_project] Loaded: {project.get('name')} ({project.get('material')})")
    return {"project": project, "error": None}


def db_analog_search_node(state: AnalogState) -> AnalogState:
    """Find similar projects in the Supabase database."""
    if state.get("error"):
        return {}

    project = state["project"]
    material = project.get("material")
    deposit_type = project.get("deposit_type")
    tonnage_mt = project.get("tonnage_mt")
    grade_value = project.get("grade_value")

    # Search with a ±5x tonnage range
    min_t = (tonnage_mt / 5) if tonnage_mt else None
    max_t = (tonnage_mt * 5) if tonnage_mt else None
    min_g = (grade_value / 3) if grade_value else None
    max_g = (grade_value * 3) if grade_value else None

    db_results = supabase_ops.search_projects_by_criteria(
        material=material,
        deposit_type=deposit_type,
        min_tonnage=min_t,
        max_tonnage=max_t,
        min_grade=min_g,
        max_grade=max_g,
        limit=15,
    )

    # Exclude the project itself
    db_analogs = [
        {**r, "source": "db", "similarity_score": 50, "similarity_reasons": []}
        for r in db_results
        if r.get("id") != state["project_id"]
    ]

    logger.info(f"[db_search] Found {len(db_analogs)} DB analogs")
    return {"db_analogs": db_analogs}


def exa_analog_search_node(state: AnalogState) -> AnalogState:
    """Find comparable projects via Exa."""
    if state.get("error"):
        return {}

    project = state["project"]
    material = project.get("material", "")
    deposit_type = project.get("deposit_type", "")
    grade_value = project.get("grade_value")
    grade_unit = project.get("grade_unit")
    tonnage_mt = project.get("tonnage_mt")
    country = project.get("country")

    text, sources = exa_search.search_analog_projects(
        material=material,
        deposit_type=deposit_type,
        grade_value=grade_value,
        grade_unit=grade_unit,
        tonnage_mt=tonnage_mt,
        country=country,
    )

    exa_analogs = []
    if text:
        raw = field_extractor.extract_analog_projects(text, material, sources)
        for i, a in enumerate(raw):
            exa_analogs.append({
                "name": a.get("name", f"Unknown project {i}"),
                "material": material,
                "deposit_type": a.get("deposit_type"),
                "tonnage_mt": a.get("tonnage_mt"),
                "grade_value": a.get("grade_value"),
                "grade_unit": a.get("grade_unit"),
                "country": a.get("country"),
                "project_stage": a.get("project_stage"),
                "mining_method": a.get("mining_method"),
                "source": "exa",
                "source_url": a.get("source_url") or (sources[i] if i < len(sources) else None),
                "similarity_score": 50,
                "similarity_reasons": [],
                "approved": False,
            })

    logger.info(f"[exa_search] Found {len(exa_analogs)} Exa analogs")
    return {"exa_analogs": exa_analogs}


def _filter_analog_candidates(
    candidates: list,
    target_material: str,
    target_tonnage: float,
) -> list:
    """Hard-filter candidates before LLM scoring to remove obviously wrong analogs."""
    out = []
    for c in candidates:
        # Same commodity required
        if (c.get("material") or "").lower() != (target_material or "").lower():
            continue
        # Tonnage within 10x (guards against 1 Mt project vs 5 000 Mt project)
        c_tonnage = c.get("tonnage_mt") or 0
        if c_tonnage > 0 and target_tonnage > 0:
            ratio = max(c_tonnage, target_tonnage) / min(c_tonnage, target_tonnage)
            if ratio > 10:
                continue
        out.append(c)
    return out


def score_analogs_node(state: AnalogState) -> AnalogState:
    """Combine DB + Exa analogs, validate, score with LLM, take top 4."""
    if state.get("error"):
        return {}

    project = state["project"]
    db_analogs = state.get("db_analogs", [])
    exa_analogs = state.get("exa_analogs", [])
    all_candidates = db_analogs + exa_analogs

    if not all_candidates:
        return {"all_candidates": [], "scored_analogs": []}

    # Hard-filter before expensive LLM scoring
    target_material = project.get("material", "")
    target_tonnage = project.get("tonnage_mt") or 0
    all_candidates = _filter_analog_candidates(all_candidates, target_material, target_tonnage)
    logger.info(f"[score] {len(all_candidates)} candidates after validation filter")

    if not all_candidates:
        return {"all_candidates": [], "scored_analogs": []}

    # Build compact project summary for scoring
    target_summary = {
        "name": project.get("name"),
        "material": project.get("material"),
        "deposit_type": project.get("deposit_type"),
        "tonnage_mt": project.get("tonnage_mt"),
        "grade_value": project.get("grade_value"),
        "grade_unit": project.get("grade_unit"),
        "project_stage": project.get("project_stage"),
        "mining_method": project.get("mining_method"),
        "country": project.get("country"),
    }

    scored = field_extractor.score_analogs(target_summary, all_candidates)
    top_4 = sorted(scored, key=lambda x: x.get("similarity_score", 0), reverse=True)[:4]

    logger.info(f"[score] Top analog: {top_4[0].get('name') if top_4 else 'none'}")
    return {"all_candidates": all_candidates, "scored_analogs": top_4}


def human_review_analog_node(state: AnalogState) -> AnalogState:
    """Auto-approve all scored analogs — no human interrupt."""
    approved_analogs = state.get("scored_analogs", [])
    return {"human_approved": True, "approved_analogs": approved_analogs}


def save_analogs_node(state: AnalogState) -> AnalogState:
    """Save approved analogs to Supabase."""
    if not state.get("human_approved"):
        logger.info("[save_analogs] Human rejected — not saving")
        return {"saved": False}

    analogs = state.get("approved_analogs", [])
    for a in analogs:
        a["approved"] = True

    try:
        supabase_ops.save_analogs(state["project_id"], analogs)
        logger.info(f"[save_analogs] Saved {len(analogs)} analogs for project {state['project_id']}")
        return {"saved": True, "error": None}
    except Exception as e:
        logger.error(f"[save_analogs] Error: {e}")
        return {"saved": False, "error": str(e)}


# ── Graph ─────────────────────────────────────────────────────────────────────

def should_continue(state: AnalogState) -> str:
    return END if state.get("error") else "exa_analog_search"


builder = StateGraph(AnalogState)

builder.add_node("load_project", load_project_node)
builder.add_node("db_analog_search", db_analog_search_node)
builder.add_node("exa_analog_search", exa_analog_search_node)
builder.add_node("score_analogs", score_analogs_node)
builder.add_node("human_review", human_review_analog_node)
builder.add_node("save_analogs", save_analogs_node)

builder.set_entry_point("load_project")
builder.add_edge("load_project", "db_analog_search")
builder.add_conditional_edges(
    "db_analog_search",
    should_continue,
    {"exa_analog_search": "exa_analog_search", END: END},
)
builder.add_edge("exa_analog_search", "score_analogs")
builder.add_edge("score_analogs", "human_review")
builder.add_edge("human_review", "save_analogs")
builder.add_edge("save_analogs", END)

graph = builder.compile()
