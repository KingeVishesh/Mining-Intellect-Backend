"""
Graph: geocode_project

Lightweight graph that resolves GPS coordinates for a single mining project.
Uses Exa Answer API first, falls back to Nominatim if Exa returns nothing.

Flow:
  load_project → exa_answer → (conditional) nominatim? → save_coords → END

No human review, no interrupt. Fully automated.
Input:  {"project_id": "<uuid>"}
"""
from __future__ import annotations
import logging
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from nodes import exa_answer, geocoder, supabase_ops

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────

class GeocodeState(TypedDict, total=False):
    # Input
    project_id: str
    # Loaded from DB
    project_name: str
    company: str
    region: str
    country: str
    _existing_data_sources: Optional[dict]
    # Outputs
    latitude: Optional[float]
    longitude: Optional[float]
    source_url: Optional[str]
    method: Optional[str]   # "exa_answer" | "nominatim" | "failed"
    saved: bool
    error: Optional[str]


# ── Nodes ──────────────────────────────────────────────────────────────────────

def load_project(state: GeocodeState) -> GeocodeState:
    """Fetch project row from Supabase and populate state."""
    project_id = state.get("project_id")
    if not project_id:
        return {**state, "error": "project_id not provided", "method": "failed"}

    row = supabase_ops.get_project(project_id)
    if row is None:
        return {**state, "error": f"project {project_id} not found", "method": "failed"}

    return {
        **state,
        "project_name": row.get("name", ""),
        "company": row.get("company_name", ""),
        "region": row.get("region", ""),
        "country": row.get("country", ""),
        "_existing_data_sources": row.get("data_sources"),
    }


def exa_answer_node(state: GeocodeState) -> GeocodeState:
    """Call Exa Answer API for GPS coordinates."""
    if state.get("error"):
        return state

    lat, lng, source_url = exa_answer.ask_coords(
        project_name=state.get("project_name", ""),
        company=state.get("company", ""),
        region=state.get("region", ""),
        country=state.get("country", ""),
    )

    if lat is not None and lng is not None:
        return {**state, "latitude": lat, "longitude": lng, "source_url": source_url, "method": "exa_answer"}

    return {**state, "latitude": None, "longitude": None}


def nominatim_node(state: GeocodeState) -> GeocodeState:
    """Fallback: geocode via Nominatim using region + country."""
    region = state.get("region", "")
    country = state.get("country", "")
    location_name = ", ".join(p for p in [region, country] if p and p.strip())

    lat, lng = geocoder.geocode(location_name)

    if lat is not None and lng is not None:
        return {**state, "latitude": lat, "longitude": lng, "source_url": None, "method": "nominatim"}

    return {**state, "method": "failed"}


def save_coords_node(state: GeocodeState) -> GeocodeState:
    """Write lat/lng to Supabase if found."""
    if state.get("error") or state.get("method") == "failed":
        logger.warning(
            f"[geocode_project] No coords for '{state.get('project_name')}' "
            f"({state.get('project_id')}): {state.get('error', 'all methods failed')}"
        )
        return {**state, "saved": False}

    lat = state.get("latitude")
    lng = state.get("longitude")
    if lat is None or lng is None:
        return {**state, "method": "failed", "saved": False}

    saved = supabase_ops.save_coords(
        project_id=state["project_id"],
        latitude=lat,
        longitude=lng,
        method=state.get("method", "unknown"),
        source_url=state.get("source_url"),
        existing_data_sources=state.get("_existing_data_sources"),
    )
    return {**state, "saved": saved}


# ── Conditional edge ───────────────────────────────────────────────────────────

def _after_exa(state: GeocodeState) -> str:
    """Go to save_coords if Exa found coords, otherwise try Nominatim."""
    if state.get("error") or state.get("method") == "failed":
        return "save_coords"
    if state.get("latitude") is not None:
        return "save_coords"
    return "nominatim"


# ── Graph ──────────────────────────────────────────────────────────────────────

builder = StateGraph(GeocodeState)

builder.add_node("load_project", load_project)
builder.add_node("exa_answer", exa_answer_node)
builder.add_node("nominatim", nominatim_node)
builder.add_node("save_coords", save_coords_node)

builder.set_entry_point("load_project")
builder.add_edge("load_project", "exa_answer")
builder.add_conditional_edges("exa_answer", _after_exa, {
    "save_coords": "save_coords",
    "nominatim": "nominatim",
})
builder.add_edge("nominatim", "save_coords")
builder.add_edge("save_coords", END)

graph = builder.compile()
