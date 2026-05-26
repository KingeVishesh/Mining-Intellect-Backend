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

    effective_subtype = (
        inferred.get("deposit_subtype") or fields.get("deposit_subtype")
    )

    # Grok probe: fire WHENEVER deposit_type is null and we have any
    # location signal. Don't gate on effective_subtype — the probe gives
    # richer deposit_type text than the slug-synthesis fallback ("orogenic
    # vein-hosted gold (Newfoundland Appalachian belt)" beats "orogenic
    # general"). The probe is also where projects with only mining_method
    # / processing_method get rescued (Hammerdown, Goldfields).
    if (not fields.get("deposit_type")
            and (fields.get("country") or fields.get("region") or fields.get("district"))):
        from nodes.field_extractor import probe_deposit_type
        project_name = state.get("project_name") or ""
        material = state.get("material") or fields.get("material") or "gold"
        location_bits = [b for b in (fields.get("district"), fields.get("region"),
                                       fields.get("country")) if b]
        # company_name disambiguates ambiguous targets ("Goldfields" alone
        # matches dozens of projects worldwide; "Fortune Bay Corp -
        # Goldfields" pinpoints the Saskatchewan one).
        company_name = state.get("company") or fields.get("company_name")
        probe = probe_deposit_type(
            project_name, material, " / ".join(location_bits),
            company_name=company_name,
        )
        if probe:
            dep_text = probe.get("deposit_type")
            sub_slug = probe.get("deposit_subtype")
            pat_slug = probe.get("mineralization_pattern")
            if dep_text:
                inferred["deposit_type"] = dep_text
                logger.info(
                    f"[derive] Grok probe filled deposit_type={dep_text!r}"
                )
            if sub_slug and not effective_subtype:
                inferred["deposit_subtype"] = sub_slug
                effective_subtype = sub_slug
            if pat_slug and not (
                inferred.get("mineralization_pattern")
                or fields.get("mineralization_pattern")
            ):
                inferred["mineralization_pattern"] = pat_slug

    # Fallback synthesis: if the probe failed or didn't return deposit_type
    # but we DO have a subtype slug, synthesize a human-readable deposit_type
    # from it so rule routing (Pass 1/2 ILIKE) has a string to match on.
    # Runs after the probe so a real description beats a slug humanization.
    if not fields.get("deposit_type") and "deposit_type" not in inferred and effective_subtype:
        humanized = effective_subtype.replace("_", " ")
        inferred["deposit_type"] = humanized
        logger.info(
            f"[derive] synthesized deposit_type='{humanized}' from "
            f"subtype={effective_subtype!r}"
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


# Country keyword index — substring matches in location_name are signals
# of the project's actual country. Used by validate_node to catch field
# crosstalk like Latin Metals Crosby whose region was "Jujuy Province"
# (Argentina) but location_name was "Crosby County, Texas, USA" — two
# different real-world places that got merged into one record.
#
# Keep this list ASCII-lowercase. The validator lowercases location_name
# before matching.
_COUNTRY_KEYWORDS: dict[str, list[str]] = {
    "argentina": ["argentina", "salta", "jujuy", "san juan", "mendoza",
                   "catamarca", "santa cruz argentina", "patagonia argentina"],
    "australia": ["australia", "nsw", "new south wales", "queensland",
                   "western australia", "south australia", "tasmania",
                   "northern territory"],
    "brazil":    ["brazil", "brasil", "minas gerais", "pará", "para state",
                   "tapajós", "carajás", "bahia"],
    "canada":    ["canada", "ontario", "quebec", "british columbia",
                   "alberta", "saskatchewan", "manitoba", "yukon",
                   "newfoundland", "labrador", "nova scotia", "nunavut"],
    "chile":     ["chile", "atacama", "antofagasta", "coquimbo"],
    "china":     ["china", "yunnan", "guizhou", "xinjiang"],
    "colombia":  ["colombia"],
    "côte d'ivoire": ["côte d'ivoire", "cote d'ivoire", "ivory coast",
                       "abidjan", "yamoussoukro", "bouaflé"],
    "ecuador":   ["ecuador", "imbabura", "el oro province"],
    "finland":   ["finland", "lapland", "kuusamo", "kittilä"],
    "ghana":     ["ghana", "ashanti", "kumasi", "obuasi"],
    "guyana":    ["guyana", "cuyuni", "mazaruni", "co-operative republic of guyana"],
    "guinea":    ["guinea conakry", "guinea bissau", "republic of guinea"],
    "indonesia": ["indonesia", "sumatra", "kalimantan", "papua indonesia"],
    "liberia":   ["liberia", "sinoe county", "monrovia"],
    "mali":      ["mali", "kayes", "kéniéba", "kenieba", "loulo"],
    "mexico":    ["mexico", "sonora", "chihuahua", "durango", "zacatecas"],
    "new zealand": ["new zealand", "reefton", "otago"],
    "papua new guinea": ["papua new guinea", "png", "porgera", "lihir"],
    "peru":      ["peru", "lima", "cajamarca", "yanacocha"],
    "philippines": ["philippines", "luzon", "mindanao"],
    "saudi arabia": ["saudi arabia", "jeddah"],
    "senegal":   ["senegal", "kédougou", "kedougou", "sabodala"],
    "south africa": ["south africa", "bushveld", "witwatersrand"],
    "suriname":  ["suriname", "brokopondo"],
    "sweden":    ["sweden", "skellefte", "kiruna"],
    "tanzania":  ["tanzania", "geita", "lake victoria"],
    "turkey":    ["turkey", "anatolia"],
    "uk":        ["united kingdom", "scotland", "england", "wales", "cornwall"],
    "usa":       ["usa", "united states", "u.s.a", "nevada", "arizona",
                   "alaska", "idaho", "montana", "colorado", "utah",
                   "wyoming", "california", "new mexico", "north dakota",
                   "south dakota", "texas"],
    "venezuela": ["venezuela"],
    "zambia":    ["zambia", "lufilian", "copperbelt zambia"],
    "zimbabwe":  ["zimbabwe", "great dyke"],
    "burkina faso": ["burkina faso", "houndé", "hounde"],
    "drc":       ["drc", "dr congo", "democratic republic of"],
}


def _country_in_text(text: str) -> set[str]:
    """Return the set of country slugs whose keywords appear in `text`.
    Used to detect when location_name implies a country different from
    what's stored in the `country` field."""
    if not text:
        return set()
    t = text.lower()
    hits: set[str] = set()
    for slug, kws in _COUNTRY_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                hits.add(slug)
                break
    return hits


def _detect_country_conflict(fields: dict) -> Optional[str]:
    """
    Return a human-readable conflict description when `country` and
    `location_name` (or `region`) clearly disagree about which country
    the project sits in. Returns None when no conflict.

    Conservative — only fires when location_name names a SPECIFIC
    country different from the stored one. Generic mentions like
    "South America" don't trigger.
    """
    stored_country = (fields.get("country") or "").strip().lower()
    if not stored_country:
        return None
    location_hits = _country_in_text(fields.get("location_name") or "")
    region_hits = _country_in_text(fields.get("region") or "")
    all_hits = location_hits | region_hits
    if not all_hits:
        return None
    # Map stored_country onto its slug
    stored_slug: Optional[str] = None
    for slug, kws in _COUNTRY_KEYWORDS.items():
        if stored_country in kws or stored_country == slug:
            stored_slug = slug
            break
    if stored_slug is None:
        return None
    # Conflict only when location names a DIFFERENT country AND the stored
    # country isn't also mentioned (e.g., "near Argentine border, Chile"
    # legitimately mentions both).
    if stored_slug in all_hits:
        return None
    if all_hits:
        wrong = sorted(all_hits)
        return (
            f"country={fields.get('country')!r} but location_name "
            f"({fields.get('location_name')!r}) / region "
            f"({fields.get('region')!r}) name a different country: "
            f"{wrong}. Likely Exa returned text from the wrong project; "
            f"re-research with disambiguating context."
        )
    return None


def validate_node(state: ResearchState) -> ResearchState:
    """Check for required fields, country-consistency, and build
    validation_errors list. On a country conflict (Crosby-style crosstalk),
    we additionally CLEAR the conflicting fields so the bad data isn't
    persisted — the project is left thin instead of poisoned."""
    if state.get("error"):
        return {}

    fields = state.get("extracted_fields", {})
    errors = []

    # Cross-field country consistency check — fires before required-field
    # check so cleared fields aren't double-reported as missing.
    conflict = _detect_country_conflict(fields)
    if conflict:
        errors.append(f"Country conflict: {conflict}")
        logger.warning(f"[validate] {conflict}")
        # Clear the location-related fields that are suspect. Keep country
        # (the inbound seed value, generally trustworthy) but drop the
        # downstream fields that may have been extracted from the wrong
        # project's text. The next run can re-extract cleanly.
        fields = dict(fields)
        for k in ("location_name", "district", "region", "latitude",
                   "longitude", "mineralization_style", "host_rock",
                   "deposit_type", "deposit_subtype"):
            if fields.get(k):
                fields[k] = None

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

    # ── Plausibility checks ─────────────────────────────────────────────
    # Catch obvious Grok decimal-misplacement or unit confusion before
    # the absurd value enters the cascade and corrupts a model run.
    # When tonnage × grade implies a deposit too large to be plausible
    # for the stated stage, clear the suspect numeric values (keep the
    # geological identity).
    tonnage = fields.get("tonnage_mt")
    grade = fields.get("grade_value")
    grade_unit = (fields.get("grade_unit") or "").lower()
    material = (fields.get("material") or "").lower()
    if tonnage and grade:
        try:
            t, g = float(tonnage), float(grade)
            # Gold bulk deposits rarely exceed 5 g/t; refractory UG up to 25.
            # Anything > 50 g/t at >5 Mt tonnage is almost certainly wrong
            # (decimal misplaced or grade reported in oz/t mistakenly).
            if material == "gold" and "g/t" in grade_unit:
                if g > 50 and t > 5:
                    errors.append(
                        f"Implausible gold grade {g} g/t @ {t} Mt — likely "
                        f"unit confusion (oz/t vs g/t) or decimal misplaced. "
                        f"Clearing grade for re-research."
                    )
                    fields = dict(fields)
                    fields["grade_value"] = None
                    fields["grade_unit"] = None
                # Gold contained > 200 Moz puts the project in Witwatersrand
                # territory; no early-stage explorer has this.
                contained_moz = (t * g / 31.1035)  # Mt × g/t → Moz approx
                if contained_moz > 200 and stage in ("exploration", "pea", "pfs"):
                    errors.append(
                        f"Implausible contained gold {contained_moz:.0f} Moz "
                        f"for stage={stage!r}. Clearing tonnage/grade."
                    )
                    fields = dict(fields)
                    fields["tonnage_mt"] = None
                    fields["grade_value"] = None
            # Copper percent grades > 20% are exceptional; > 50% is impossible
            # at any meaningful tonnage (would be native copper specimens).
            if material == "copper" and "%" in grade_unit:
                if g > 20 and t > 0.5:
                    errors.append(
                        f"Implausible copper grade {g}% @ {t} Mt — "
                        f"likely unit confusion. Clearing grade."
                    )
                    fields = dict(fields)
                    fields["grade_value"] = None
                    fields["grade_unit"] = None
        except (TypeError, ValueError):
            pass  # numeric parse failed elsewhere; not a plausibility issue

    logger.info(f"[validate] {len(errors)} validation issues")
    return {"extracted_fields": fields, "validation_errors": errors}


def save_to_supabase_node(state: ResearchState, config: Optional[Dict] = None) -> ResearchState:
    """Save extracted project fields to Supabase. No human gate.

    Null / empty extracted values are dropped so a re-research run never
    wipes good data that the LLM happens to miss on this pass. A field is
    only written when the extractor produced a positive value.

    MRE versioning: when the extracted fields contain MRE totals or a
    breakdown, hand them to `save_mre_run_if_changed` so each genuine
    MRE update gets its own row in `mre_runs` (the chart reads from
    that table for the historical MRE line). The latest values are
    mirrored to projects.* via `update_project_mre_mirror` so existing
    queries that read tonnage_mt / grade_value / mre_mi_* keep working.
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

        # ── MRE versioning ──────────────────────────────────────────
        # Build the mre_runs payload from whatever the extractor pulled
        # (totals + optional M&I/Inferred split). Skip the save when no
        # MRE-related fields are present at all — that's a project that
        # doesn't have a published resource estimate yet.
        mre_payload = {
            "total_tonnage_mt":     fields.get("tonnage_mt"),
            "total_grade":          fields.get("grade_value"),
            "grade_unit":           fields.get("grade_unit"),
            "resource_category":    fields.get("resource_category"),
            "effective_date":       fields.get("resource_effective_date"),
            "mi_tonnage_mt":        fields.get("mre_mi_tonnage_mt"),
            "mi_grade":             fields.get("mre_mi_grade"),
            "mi_contained":         fields.get("mre_mi_contained"),
            "inferred_tonnage_mt":  fields.get("mre_inferred_tonnage_mt"),
            "inferred_grade":       fields.get("mre_inferred_grade"),
            "inferred_contained":   fields.get("mre_inferred_contained"),
            "source":               "project_research",
            "source_url":           (state.get("exa_sources") or [None])[0],
        }
        if any(v is not None for k, v in mre_payload.items()
               if k in ("total_tonnage_mt", "total_grade",
                        "mi_tonnage_mt", "inferred_tonnage_mt")):
            cfg = (config or {}).get("configurable") or {}
            supabase_ops.save_mre_run_if_changed(
                state["project_id"], mre_payload,
                thread_id=cfg.get("thread_id"), run_id=cfg.get("run_id"),
            )
            supabase_ops.update_project_mre_mirror(state["project_id"], mre_payload)

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
