"""
Graph 1: project_research

Flow:
  load_context → exa_search → extract_fields → geocode → validate
              → save_to_supabase → END

Input:  { project_name, material, project_id, company }
Output: Populated project record saved to Supabase

No human-in-the-loop. The graph saves extracted fields directly.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from nodes import exa_search, field_extractor, geo_taxonomy, geocoder, supabase_ops
from nodes.rules_engine import sanitize_deposit_type

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict, total=False):
    # Input
    project_id: str
    project_name: str
    material: str
    company: str

    # Intermediate
    existing_project: Optional[Dict]
    exa_text: str
    exa_sources: List[str]
    extracted_fields: Dict
    field_statuses: Dict
    validation_errors: List[str]

    # Output
    saved: bool
    error: Optional[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def load_context(state: ResearchState) -> ResearchState:
    """Load existing project from Supabase if it exists."""
    project_id = state["project_id"]
    existing = supabase_ops.get_project(project_id)
    logger.info(f"[load_context] project_id={project_id} existing={'yes' if existing else 'no'}")
    return {"existing_project": existing}


def exa_search_node(state: ResearchState) -> ResearchState:
    """Search Exa for project data."""
    project_name = state["project_name"]
    company = state.get("company") or project_name
    material = state["material"]

    logger.info(f"[exa_search] Searching for: {company} - {project_name} ({material})")
    text, sources = exa_search.search_project_data(project_name, company, material)

    if not text:
        return {"exa_text": "", "exa_sources": [], "error": "Exa search returned no content"}
    return {"exa_text": text, "exa_sources": sources, "error": None}


def extract_fields_node(state: ResearchState) -> ResearchState:
    """Use Grok to extract structured fields from Exa text."""
    if state.get("error"):
        return {}

    project_name = state["project_name"]
    company = state.get("company") or project_name
    material = state["material"]
    text = state["exa_text"]

    fields = field_extractor.extract_fields(text, project_name, company, material)
    clean_fields, statuses = field_extractor.judge_fields(text, fields, project_name, company, material)

    # Retry for search_miss fields
    search_miss = [f for f, s in statuses.items() if s == "search_miss"]
    if search_miss:
        retry_text, retry_sources = exa_search.search_missing_fields(
            project_name, company, material, search_miss
        )
        if retry_text:
            retry_fields = field_extractor.extract_fields(retry_text, project_name, company, material)
            retry_clean, retry_statuses = field_extractor.judge_fields(
                retry_text, retry_fields, project_name, company, material, judge_only=search_miss
            )
            for f in search_miss:
                if retry_clean.get(f) is not None and clean_fields.get(f) is None:
                    clean_fields[f] = retry_clean[f]
                    statuses[f] = "found_on_retry"

    return {"extracted_fields": clean_fields, "field_statuses": statuses}


def derive_geological_profile_node(state: ResearchState) -> ResearchState:
    """Run deterministic taxonomy detectors on whatever Grok DID extract, then
    backfill structured columns the LLM left empty.

    This is NOT a wildcard fallback — it uses the controlled vocabulary in
    nodes.geo_taxonomy to map freeform text (mineralization_style, host_rock,
    district, region, country) onto exact slugs. If the detectors come up
    empty too, the field stays null and the analog finder will correctly
    refuse to score the project. The point is that when the source DOES
    say e.g. "Carlin-style sediment-hosted disseminated gold" but Grok
    forgot to copy that into deposit_type, the detector recovers it.

    Side effect: when deposit_type is null but detect_subtype produces a
    confident slug, we synthesize a human-readable deposit_type from the
    style/subtype so downstream rule-matching (Pass 1/2) has a string to
    work with.
    """
    if state.get("error"):
        return {}
    fields = dict(state.get("extracted_fields") or {})
    if not fields:
        return {}

    clean_dep = sanitize_deposit_type(fields.get("deposit_type"))
    if clean_dep and clean_dep != fields.get("deposit_type"):
        fields["deposit_type"] = clean_dep

    style = fields.get("mineralization_style")
    alt = fields.get("alteration_signature")
    district = fields.get("district") or fields.get("location_name")
    country = fields.get("country")
    region = fields.get("region")
    host = fields.get("host_rock")
    mining = fields.get("mining_method")
    processing = fields.get("processing_method")

    inferred: dict[str, str] = {}

    # Subtype
    if not fields.get("deposit_subtype"):
        sub = geo_taxonomy.detect_subtype(clean_dep, style, alt, district)
        if sub:
            inferred["deposit_subtype"] = sub

    # Pattern
    if not fields.get("mineralization_pattern"):
        pat = geo_taxonomy.detect_pattern(style, mining, processing, clean_dep)
        if pat:
            inferred["mineralization_pattern"] = pat

    # Mode
    if not fields.get("mineralization_mode"):
        mode = geo_taxonomy.detect_mode(processing, style, district, clean_dep)
        if mode:
            inferred["mineralization_mode"] = mode

    # Tectonic belt
    if not fields.get("tectonic_belt"):
        belt = geo_taxonomy.detect_belt(country, region, district)
        if belt:
            inferred["tectonic_belt"] = belt

    # Metal suite
    if not fields.get("metal_suite"):
        suite = geo_taxonomy.detect_metal_suite(
            fields.get("material"), fields.get("by_product_commodities"),
            district, clean_dep,
        )
        if suite:
            inferred["metal_suite"] = suite

    # Alteration
    if not fields.get("alteration_signature"):
        a = geo_taxonomy.detect_alteration_signature(None, district, clean_dep)
        if a:
            inferred["alteration_signature"] = a

    # Recovery method
    if not fields.get("recovery_method"):
        rec = geo_taxonomy.detect_recovery_method(processing, district, clean_dep)
        if rec:
            inferred["recovery_method"] = rec

    # Host rock class
    if not fields.get("host_rock_class"):
        hc = geo_taxonomy.detect_host_class(host, clean_dep, style)
        if hc:
            inferred["host_rock_class"] = hc

    # Stage class
    if not fields.get("project_stage_class"):
        sc = geo_taxonomy.detect_stage_class(
            fields.get("project_stage"), None, district,
        )
        if sc:
            inferred["project_stage_class"] = sc

    # Mining method class
    if not fields.get("mining_method_class"):
        mc = geo_taxonomy.detect_mining_method_class(mining, processing, district)
        if mc:
            inferred["mining_method_class"] = mc

    # Synthesize deposit_type from subtype when Grok left it blank. The
    # rule engine's Pass 1/2 ILIKE matching needs a non-empty string; the
    # subtype slug, humanized, is the safest derivation.
    if not fields.get("deposit_type") and inferred.get("deposit_subtype"):
        humanized = inferred["deposit_subtype"].replace("_", " ")
        inferred["deposit_type"] = humanized
        logger.info(
            f"[derive] synthesized deposit_type='{humanized}' from "
            f"detected subtype={inferred['deposit_subtype']!r}"
        )

    if not inferred:
        return {}

    logger.info(f"[derive] backfilled {len(inferred)} structured fields: "
                f"{list(inferred.keys())}")
    fields.update(inferred)

    # Record which fields came from the deterministic post-pass so the
    # audit trail can distinguish LLM-extracted from taxonomy-derived
    # values when debugging.
    statuses = dict(state.get("field_statuses") or {})
    for k in inferred:
        statuses.setdefault(k, "derived_post_extraction")
    return {"extracted_fields": fields, "field_statuses": statuses}


def geocode_node(state: ResearchState) -> ResearchState:
    """Geocode lat/lng if not already extracted."""
    if state.get("error"):
        return {}

    fields = state.get("extracted_fields", {})
    if fields.get("latitude") and fields.get("longitude"):
        return {}  # already have coords

    location_name = fields.get("location_name") or (
        f"{fields.get('region', '')} {fields.get('country', '')}".strip()
    )
    if location_name:
        lat, lng = geocoder.geocode(location_name)
        if lat and lng:
            updated = dict(fields)
            updated["latitude"] = lat
            updated["longitude"] = lng
            return {"extracted_fields": updated}
    return {}


def validate_node(state: ResearchState) -> ResearchState:
    """Check for required fields and build validation_errors list."""
    if state.get("error"):
        return {}

    fields = state.get("extracted_fields", {})
    errors = []

    required = ["country", "deposit_type", "project_stage"]
    for f in required:
        if not fields.get(f):
            errors.append(f"Missing required field: {f}")

    if not fields.get("latitude") or not fields.get("longitude"):
        errors.append("Missing lat/lng (geocoding failed or location not found)")

    # Tonnage is only required when a formal resource study exists (PEA or later).
    # Early-stage exploration projects legitimately have no resource estimate yet.
    stage = (fields.get("project_stage") or "").lower()
    study_stages = {"pea", "pfs", "feasibility", "construction", "production"}
    if stage in study_stages and fields.get("tonnage_mt") is None:
        errors.append("No resource tonnage found (required for PEA+ stage projects)")

    logger.info(f"[validate] {len(errors)} validation issues")
    return {"validation_errors": errors}


def save_to_supabase_node(state: ResearchState) -> ResearchState:
    """Save extracted project fields to Supabase. No human gate.

    Null / empty extracted values are dropped so a re-research run never
    wipes good data that the LLM happens to miss on this pass. A field is
    only written when the extractor produced a positive value.
    """
    if state.get("error"):
        logger.info(f"[save] Upstream error — not saving: {state['error']}")
        return {"saved": False}

    raw_fields = state.get("extracted_fields") or {}
    fields = {k: v for k, v in raw_fields.items()
              if v is not None and v != "" and v != [] and v != {}}

    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": state["project_id"],
        "name": state["project_name"],
        "material": state["material"],
        **fields,
        "enrichment_status": "complete",
        "field_statuses": state.get("field_statuses", {}),
        "data_sources": {"exa_sources": state.get("exa_sources", [])},
        "last_verified_at": now,
        "updated_at": now,
    }

    try:
        # Use extracted company_name; fall back to the caller-supplied company input
        company_name = row.get("company_name") or state.get("company")
        if company_name:
            row["company_name"] = company_name
            row["company_id"] = supabase_ops.upsert_company(company_name)
        supabase_ops.upsert_project(row)
        logger.info(f"[save] Project {state['project_id']} saved to Supabase")
        return {"saved": True, "error": None}
    except Exception as e:
        logger.error(f"[save] Supabase write error: {e}")
        return {"saved": False, "error": str(e)}


# ── Graph ─────────────────────────────────────────────────────────────────────

def should_continue(state: ResearchState) -> str:
    if state.get("error"):
        return END
    return "extract_fields"


builder = StateGraph(ResearchState)

builder.add_node("load_context", load_context)
builder.add_node("exa_search", exa_search_node)
builder.add_node("extract_fields", extract_fields_node)
builder.add_node("derive_geological_profile", derive_geological_profile_node)
builder.add_node("geocode", geocode_node)
builder.add_node("validate", validate_node)
builder.add_node("save_to_supabase", save_to_supabase_node)

builder.set_entry_point("load_context")
builder.add_edge("load_context", "exa_search")
builder.add_conditional_edges("exa_search", should_continue, {"extract_fields": "extract_fields", END: END})
builder.add_edge("extract_fields", "derive_geological_profile")
builder.add_edge("derive_geological_profile", "geocode")
builder.add_edge("geocode", "validate")
builder.add_edge("validate", "save_to_supabase")
builder.add_edge("save_to_supabase", END)

graph = builder.compile()
