"""
Exa Search Node — all Exa API calls in one place.

Three call types:
  1. project_research   — deep search for MRE + economics data for a named project
  2. analog_search      — search for comparable deposits
  3. discovery          — discover new mining projects (scheduled)
"""
from __future__ import annotations
import logging
import requests
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

EXA_API_URL = "https://api.exa.ai/search"


def _post(payload: dict, timeout: int = 180) -> Optional[dict]:
    """POST to Exa and return the parsed JSON, or None on failure."""
    headers = {
        "x-api-key": settings.exa_api_key,
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(EXA_API_URL, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        logger.error("[Exa] Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"[Exa] Request error: {e}")
        return None

    if resp.status_code != 200:
        logger.error(f"[Exa] HTTP {resp.status_code}: {resp.text[:300]}")
        return None

    data = resp.json()
    cost = data.get("costDollars", {}).get("total", 0)
    logger.info(f"[Exa] HTTP 200 | cost=${cost:.4f}")
    return data


def _extract_sources(data: dict) -> list[str]:
    sources = []
    for citation in data.get("output", {}).get("grounding", []):
        for c in citation.get("citations", []):
            url = c.get("url", "")
            if url and url not in sources:
                sources.append(url)
    for r in data.get("results", []):
        url = r.get("url", "")
        if url and url not in sources:
            sources.append(url)
    return sources


# ── 1. Project Research ───────────────────────────────────────────────────────

def search_project_data(
    project_name: str,
    company: str,
    material: str,
) -> tuple[str, list[str]]:
    """
    Deep Exa call: find MRE + economics for a specific mining project.
    Returns (synthesised_text, source_urls).
    """
    query = (
        f"What are the official mineral resource estimates and economic study results "
        f"for {company}'s {project_name} {material} project? "
        f"I need from the most recent NI 43-101, JORC report, feasibility study, PFS, "
        f"PEA or company press release: "
        f"total resource tonnes (measured, indicated, inferred separately) and grade, "
        f"contained metal quantity (e.g. Moz gold, Mlbs copper), "
        f"deposit type, host rock type, mineralisation style, "
        f"by-product metals in the resource, "
        f"mining method, processing method, metallurgical recovery, final saleable product, "
        f"mine life, annual production rate, initial CAPEX, OPEX per unit, "
        f"NPV with discount rate, IRR, payback period, "
        f"project location (country, region, and geological district), "
        f"current project stage, project ownership percentage, "
        f"planned construction or mine start year, "
        f"primary energy or power source, site elevation in metres, "
        f"climate and terrain description, permitting milestones achieved."
    )
    payload = {
        "query": query,
        "type": "deep",
        "systemPrompt": (
            "Prefer official primary sources: company press releases, NI 43-101 or JORC "
            "technical reports, feasibility or pre-feasibility studies, and investor "
            "presentations. Use the most recent study available. "
            "Report exact numbers with units exactly as stated. "
            "Do not round, estimate, or infer. "
            "If a value is not explicitly stated, say Not found."
        ),
        "outputSchema": {
            "type": "text",
            "description": (
                "List each value with its unit, the source document name, and year. "
                "Group resource figures by category (measured / indicated / inferred). "
                "If a value was not found, say Not found."
            ),
        },
    }
    data = _post(payload, timeout=180)
    if not data:
        return "", []
    return data.get("output", {}).get("content", ""), _extract_sources(data)


def search_missing_fields(
    project_name: str,
    company: str,
    material: str,
    missing_fields: list[str],
) -> tuple[str, list[str]]:
    """Targeted retry for specific missing fields."""
    field_labels = {
        # Existing fields
        "country": "country where the project is located",
        "region": "region or state/province within the country",
        "company_name": "mining company that owns or operates the project",
        "commodity": "primary commodity or mineral being mined",
        "deposit_type": "deposit type classification (e.g. VMS, porphyry copper, orogenic gold)",
        "project_stage": "current project stage (Exploration, PEA, PFS, Feasibility, Construction, Production)",
        "tonnage_mt": "total mineral resource in million tonnes (Mt)",
        "grade_value": "average resource grade value",
        "grade_unit": "grade unit (g/t Au, % Cu, % U3O8)",
        "resource_category": "NI 43-101 resource category (Measured, Indicated, Inferred, M+I+I)",
        "mining_method": "mining method (open pit / underground / ISR)",
        "processing_method": "processing method (heap leach / flotation / CIL mill)",
        "recovery_rate": "metallurgical or mill recovery percentage",
        "mine_life_years": "projected mine life in years",
        "depth_meters": "deposit depth in metres",
        "width_meters": "orebody width or thickness in metres",
        "strike_length_meters": "strike length in metres",
        "npv_usd_millions": "after-tax NPV in USD or CAD",
        "capex_usd_millions": "initial capital cost (CAPEX) in USD or CAD",
        "irr_percent": "after-tax IRR percentage",
        "opex_per_unit": "operating cost per unit",
        "payback_years": "payback period in years or months",
        "production_rate_per_year": "annual production rate",
        "latitude": "decimal latitude of the project site",
        "longitude": "decimal longitude of the project site",
        "location_name": "human-readable location description",
        # Extended fields
        "host_rock": "host rock type hosting the deposit (e.g. granite, limestone, greenstone)",
        "mineralization_style": "mineralisation style (e.g. epithermal vein, porphyry, VMS, IOCG)",
        "resource_size_value": "contained metal quantity as a number (e.g. 3.2 for '3.2 Moz gold', 400 for '400 Mlbs copper')",
        "resource_size_unit": "unit for contained metal quantity (Moz, Mlbs, kt Cu)",
        "by_product_commodities": "by-product metals listed in resource tables alongside the primary metal",
        "final_product": "final saleable product form (doré bars, copper concentrate, uranium oxide U3O8)",
        "ownership_type": "project ownership structure (100% owned, 50% JV with X, optioned)",
        "district": "geological or administrative district name within the broader region",
        "start_year": "planned mine start or construction start year (integer)",
        "end_year": "planned mine end or closure year (integer)",
        "energy_source": "primary power or energy source (grid power, diesel generators, LNG, solar)",
        "climate_terrain": "climate and terrain description of the project site (Arctic tundra, tropical, high-altitude)",
        "permitting_status": "permitting milestones already achieved (Environmental Assessment approval, Mining licence, Federal permits)",
        "elevation_meters": "project site elevation above sea level in metres",
    }
    needed = ", ".join(field_labels.get(f, f) for f in missing_fields)
    query = (
        f"For {company}'s {project_name} {material} project, what are the following values "
        f"from their most recent technical study (feasibility study, PFS, PEA, NI 43-101, "
        f"or company announcement): {needed}? "
        f"Include the exact numbers with units and the report name."
    )
    payload = {
        "query": query,
        "type": "deep",
        "systemPrompt": (
            "Only use official company documents. Report exact numbers as stated. "
            "Do not estimate or infer. If not found, say Not found."
        ),
        "outputSchema": {
            "type": "text",
            "description": f"Report only these specific values: {needed}.",
        },
    }
    data = _post(payload, timeout=120)
    if not data:
        return "", []
    return data.get("output", {}).get("content", ""), _extract_sources(data)


# ── 2. Analog Search ──────────────────────────────────────────────────────────

def search_analog_projects(
    material: str,
    deposit_type: str,
    grade_value: Optional[float] = None,
    grade_unit: Optional[str] = None,
    tonnage_mt: Optional[float] = None,
    country: Optional[str] = None,
) -> tuple[str, list[str]]:
    """
    Find comparable mining projects via Exa.
    Returns (synthesised_text, source_urls).
    """
    grade_str = f"{grade_value} {grade_unit}" if grade_value and grade_unit else ""
    tonnage_str = f"{tonnage_mt}Mt" if tonnage_mt else ""
    location_str = f"in {country}" if country else "globally"

    query = (
        f"What are examples of comparable {material} mining projects with {deposit_type} "
        f"deposit type {location_str}? "
        f"I need projects similar to one with {grade_str} grade and {tonnage_str} resource. "
        f"List 5-10 analog projects with their: project name, company, country, "
        f"deposit type, resource size in tonnes, grade, project stage, and NI 43-101 or JORC reference."
    )
    payload = {
        "query": query,
        "type": "deep",
        "systemPrompt": (
            "Focus on NI 43-101 and JORC compliant resource estimates. "
            "Prefer projects at a similar or more advanced stage. "
            "List distinct projects with their exact resource figures."
        ),
        "outputSchema": {
            "type": "text",
            "description": (
                "For each comparable project list: project name, company name, country, "
                "deposit type, total resource tonnage (Mt), grade and unit, project stage, "
                "and the technical report reference."
            ),
        },
    }
    data = _post(payload, timeout=120)
    if not data:
        return "", []
    return data.get("output", {}).get("content", ""), _extract_sources(data)


# ── 3. Project Discovery ──────────────────────────────────────────────────────

def discover_new_projects(material: str) -> tuple[str, list[str]]:
    """
    Find recently announced mining projects for the given material.
    Used by the scheduled project_discovery graph.
    """
    query = (
        f"What new {material} mining exploration or development projects have been announced "
        f"or had resource estimates published in the past 6 months? "
        f"List project names, companies, countries, and any initial resource data."
    )
    payload = {
        "query": query,
        "type": "deep",
        "systemPrompt": (
            "Focus on recent NI 43-101 or JORC announcements, press releases, and technical reports. "
            "Prefer newly published or updated resource estimates."
        ),
        "outputSchema": {
            "type": "text",
            "description": (
                "List each project with: project name, company, country, material, "
                "announced resource or stage, and source URL."
            ),
        },
    }
    data = _post(payload, timeout=120)
    if not data:
        return "", []
    return data.get("output", {}).get("content", ""), _extract_sources(data)
