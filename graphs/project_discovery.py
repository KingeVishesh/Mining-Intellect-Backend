"""
Graph 4: project_discovery (scheduled background graph)

Flow:
  exa_discover → filter_new → extract_basic_fields → save_draft → END

Runs on a cron schedule (e.g. daily).
No human review required — saves projects as status='draft' for admin to promote.

Input:  { materials: List[str] }   e.g. ["uranium", "copper", "lithium"]
Output: New draft project rows in Supabase
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4

from langgraph.graph import StateGraph, END

from nodes import exa_search, field_extractor, supabase_ops

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class DiscoveryState(TypedDict, total=False):
    # Input
    materials: List[str]

    # Intermediate
    discovered: List[Dict]       # raw projects from Exa
    filtered: List[Dict]         # only genuinely new ones (not in DB)
    extracted: List[Dict]        # with basic fields parsed

    # Output
    saved_count: int
    error: Optional[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def exa_discover_node(state: DiscoveryState) -> DiscoveryState:
    """Search Exa for newly announced mining projects across all target materials."""
    materials = state.get("materials") or ["uranium", "copper", "lithium", "gold", "nickel"]
    all_discovered = []

    for material in materials:
        logger.info(f"[discover] Searching for new {material} projects...")
        text, sources = exa_search.discover_new_projects(material)
        if not text:
            continue
        raw = field_extractor.extract_new_projects(text, material)
        for p in raw:
            p["material"] = material
            p["_sources"] = sources
        all_discovered.extend(raw)
        logger.info(f"[discover] {len(raw)} {material} projects found")

    logger.info(f"[discover] Total discovered: {len(all_discovered)}")
    return {"discovered": all_discovered, "error": None}


def filter_new_node(state: DiscoveryState) -> DiscoveryState:
    """
    Filter out projects already in the database.
    Uses fuzzy name matching against existing project names.
    """
    if state.get("error"):
        return {}

    discovered = state.get("discovered", [])
    if not discovered:
        return {"filtered": []}

    # Fetch all existing project names from Supabase
    try:
        res = supabase_ops.get_client().table("projects").select("name, material, company_name").execute()
        existing = [
            (r["name"].lower().strip(), r.get("material", "").lower(), (r.get("company_name") or "").lower())
            for r in (res.data or [])
        ]
    except Exception as e:
        logger.error(f"[filter] Could not fetch existing projects: {e}")
        existing = []

    filtered = []
    for p in discovered:
        name = (p.get("name") or "").lower().strip()
        material = (p.get("material") or "").lower()
        if not name:
            continue
        # Skip if name+material already in DB (exact or partial name match)
        already_exists = any(
            (name in ex_name or ex_name in name) and ex_mat == material
            for ex_name, ex_mat, _ in existing
        )
        if not already_exists:
            filtered.append(p)

    logger.info(f"[filter] {len(filtered)} genuinely new projects (of {len(discovered)})")
    return {"filtered": filtered}


def extract_basic_fields_node(state: DiscoveryState) -> DiscoveryState:
    """
    Normalize extracted project stubs into the Supabase project schema.
    Only minimal fields are set — full enrichment happens later via project_research graph.
    """
    if state.get("error"):
        return {}

    filtered = state.get("filtered", [])
    extracted = []

    for p in filtered:
        company_name = p.get("company_name")
        company_id = None
        if company_name:
            try:
                company_id = supabase_ops.upsert_company(company_name)
            except Exception as e:
                logger.warning(f"[extract] Could not upsert company '{company_name}': {e}")

        extracted.append({
            "id": str(uuid4()),
            "name": p.get("name", "Unknown Project"),
            "material": p.get("material", "unknown"),
            "company_name": company_name,
            "company_id": company_id,
            "country": p.get("country"),
            "project_stage": p.get("project_stage"),
            "description": p.get("description"),
            "status": "draft",
            "enrichment_status": "pending",
            "data_sources": {"discovery_sources": p.get("_sources", [])},
        })

    return {"extracted": extracted}


def save_draft_node(state: DiscoveryState) -> DiscoveryState:
    """Save new draft projects to Supabase."""
    if state.get("error"):
        return {}

    extracted = state.get("extracted", [])
    saved = 0

    for project in extracted:
        try:
            supabase_ops.upsert_project(project)
            saved += 1
            logger.info(f"[save_draft] Saved: {project['name']} ({project['material']})")
        except Exception as e:
            logger.error(f"[save_draft] Error saving {project.get('name')}: {e}")

    logger.info(f"[save_draft] {saved}/{len(extracted)} new projects saved as drafts")
    return {"saved_count": saved}


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(DiscoveryState)

builder.add_node("exa_discover", exa_discover_node)
builder.add_node("filter_new", filter_new_node)
builder.add_node("extract_basic_fields", extract_basic_fields_node)
builder.add_node("save_draft", save_draft_node)

builder.set_entry_point("exa_discover")
builder.add_edge("exa_discover", "filter_new")
builder.add_edge("filter_new", "extract_basic_fields")
builder.add_edge("extract_basic_fields", "save_draft")
builder.add_edge("save_draft", END)

graph = builder.compile()
