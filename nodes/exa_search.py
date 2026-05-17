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

# Sub-trend → human-readable hint for prompt construction. Used when the
# target project resolves to a specific sub-trend within a tectonic belt
# (e.g., Cortez Trend within great_basin_carlin). Including the sub-trend
# in the Exa query biases the search toward in-trend canonical analogs
# instead of generic same-belt projects. Example: a Cortez-Trend Red Hill
# target without this hint returned Lookout Mountain / Long Canyon /
# Archimedes / Pan Mine — all great_basin_carlin but off-Cortez. With the
# hint, the query asks specifically for "Cortez Trend (Lander/Eureka
# County) projects" which surfaces Goldrush, Cortez Hills, Pipeline.
_SUB_TREND_HINTS: dict = {
    "cortez_trend": (
        "specifically on the Cortez Trend (Lander/Eureka County, Nevada — "
        "Goldrush, Cortez Hills, Pipeline, Robertson corridor)"
    ),
    "carlin_trend": (
        "specifically on the Carlin Trend (Eureka/Elko County, Nevada — "
        "Goldstrike-Betze, Meikle, Leeville, Genesis, Rodeo)"
    ),
    "getchell_trend": (
        "specifically on the Getchell Trend (Humboldt County, Nevada — "
        "Turquoise Ridge, Twin Creeks, Pinson)"
    ),
    "battle_mountain_eureka": (
        "specifically along the Battle Mountain-Eureka Trend (Nevada — "
        "Marigold, Phoenix, Lone Tree, Ruby Hill, Lookout Mountain)"
    ),
    "pequop_long_canyon": (
        "specifically in the Pequop/Long Canyon district (northeast Nevada)"
    ),
    "walker_lane_au": (
        "specifically in the Walker Lane gold belt (Nevada — Round Mountain, "
        "Manhattan, Tonopah, Paradise Peak)"
    ),
    "oquirrh_black_pine": (
        "specifically in the Oquirrh / Black Pine district "
        "(southern Idaho / northern Utah Carlin extension)"
    ),
    # Abitibi sub-camps
    "cadillac_break_valdor": (
        "specifically along the Cadillac-Larder Lake Fault Zone / Val-d'Or "
        "camp in the Abitibi (Quebec — Sigma-Lamaque, Lamaque, Beaufor, "
        "Chimo, Goldex, Bourlamaque, Canadian Malartic underground, Kiena)"
    ),
    "bousquet_camp": (
        "specifically in the Bousquet-Doyon camp (Quebec Abitibi — "
        "LaRonde, Westwood, Doyon, Bousquet — Au-rich VMS-overprint vein systems)"
    ),
    "casa_berardi_camp": (
        "specifically along the Casa Berardi Deformation Zone "
        "(Quebec Abitibi — Casa Berardi, BIF-stockwork orogenic gold)"
    ),
    "kirkland_lake_camp": (
        "specifically in the Kirkland Lake gold camp (Ontario Abitibi — "
        "Macassa, Kerr-Addison, Young-Davidson, syenite-hosted high-grade veins)"
    ),
    "rouyn_noranda_camp": (
        "specifically in the Rouyn-Noranda camp (Quebec Abitibi — "
        "Horne 5, Quemont, Au-rich VMS systems)"
    ),
    "timmins_camp": (
        "specifically in the Timmins / Porcupine camp (Ontario Abitibi — "
        "Hollinger, Dome, McIntyre, Hoyle Pond, Pamour)"
    ),
    "detour_trend": (
        "specifically along the Detour / Swayze sub-province (Ontario "
        "Abitibi — Detour Lake, Côté Gold, large bulk OP orogenic gold)"
    ),
    "hemlo_camp": (
        "specifically in the Hemlo camp (Ontario Abitibi — "
        "Williams, David Bell, Golden Giant)"
    ),
    "joutel_camp": (
        "specifically in the Joutel-Matagami sub-belt (Quebec Abitibi — "
        "Eagle-Telbel, BIF-hosted vein gold)"
    ),
    "red_lake_camp": (
        "specifically in the Red Lake gold camp (Ontario Abitibi — "
        "Red Lake mine complex, Campbell, Madsen, Cochenour — high-grade vein shoots)"
    ),
}


# Belt → human-readable hint for prompt construction
_BELT_HINTS: dict = {
    "bc_quesnel_stikine":   "in British Columbia's Quesnel/Stikine terrane (Golden Triangle, Iskut, Babine, Toodoggone)",
    "yukon_tintina":        "in the Yukon or Alaskan Tintina belt",
    "abitibi":              "in the Abitibi greenstone belt (Ontario/Quebec)",
    "newfoundland_appalachian": "in the Newfoundland-Appalachian belt",
    "laramide_southwest":   "in the Laramide southwest belt (Arizona, New Mexico, Sonora)",
    "great_basin_carlin":   "in the Great Basin / Carlin Trend (Nevada)",
    "andean":               "in the Andean copper belt (Chile, Peru, Argentina)",
    "brazilian_shield":     "in the Brazilian Shield (Carajás, Minas Gerais)",
    "central_african_copperbelt": "in the Central African Copperbelt (Zambia, DRC)",
    "bushveld":             "in the Bushveld Complex (South Africa)",
    "west_african_birimian": "in the West African Birimian belt",
    "tanzania_archean":     "in the Tanzanian Archean greenstone belt",
    "lachlan":              "in the Lachlan Fold Belt (NSW, Australia — Cadia)",
    "yilgarn":              "in the Yilgarn Craton (Western Australia)",
    "fennoscandian":        "in the Fennoscandian / Baltic Shield (Finland, Sweden, Norway)",
    "central_asian_orogenic": "in the Central Asian Orogenic Belt (Kazakhstan, Mongolia, Russia)",
    "indonesia_philippines_arc": "in the Indonesia / Philippines island-arc system",
    "new_caledonia_laterite": "in the New Caledonia laterite belt",
    "iberian_pyrite":       "in the Iberian Pyrite Belt (Portugal, Spain)",
}


def search_analog_projects(
    material: str,
    deposit_type: Optional[str],
    project_name: str = "",
    analog_rule: Optional[dict] = None,
    grade_value: Optional[float] = None,
    grade_unit: Optional[str] = None,
    tonnage_mt: Optional[float] = None,
    country: Optional[str] = None,
    host_rock: Optional[str] = None,
    mineralization_style: Optional[str] = None,
    target_profile: Optional[dict] = None,
) -> tuple[str, list[str]]:
    """
    Find comparable mining projects via Exa using a profile-driven targeted query.

    When `target_profile` is supplied, the query is built around the project's
    deposit sub-type, mineralization mode, tectonic belt, metal suite, and
    recovery method — the same dimensions used by the cascading match. This
    keeps Exa results aligned with the filters that will run on them.

    When target_profile is None, falls back to the legacy generic query.
    Returns (synthesised_text, source_urls).
    """
    deposit_str = deposit_type or material
    exclude_str = f"Do not include {project_name} itself." if project_name else ""

    # Profile-driven query when we have a full geological identity
    if target_profile and target_profile.get("deposit_subtype"):
        subtype = target_profile["deposit_subtype"].replace("_", " ")
        belt = target_profile.get("tectonic_belt")
        belt_phrase = _BELT_HINTS.get(belt, "") if belt else ""
        # Sub-trend is the geological neighborhood within the belt (e.g.
        # Cortez Trend within great_basin_carlin). When known, it narrows
        # the Exa query toward in-trend canonicals — fixes the Red Hill
        # case where a generic "Carlin Trend Nevada" query missed
        # Goldrush/Cortez Hills/Pipeline.
        sub_trend = target_profile.get("sub_trend")
        sub_trend_phrase = _SUB_TREND_HINTS.get(sub_trend, "") if sub_trend else ""
        mode = (target_profile.get("mineralization_mode") or "").replace("_", " ")
        recovery = (target_profile.get("recovery_method") or "").replace("_", " ")
        suite = (target_profile.get("metal_suite") or "").replace("_", " ").upper()

        # Identity sentence: what we're looking for in geological terms.
        # Sub-trend goes IMMEDIATELY AFTER belt so Exa reads "Carlin in
        # Great Basin specifically on the Cortez Trend ..." — the
        # sub-trend hint reinforces and narrows the belt hint rather
        # than competing with it.
        identity_parts = [f"{subtype} {material} projects"]
        if belt_phrase:
            identity_parts.append(belt_phrase)
        if sub_trend_phrase:
            identity_parts.append(sub_trend_phrase)
        if mode:
            identity_parts.append(f"with {mode} mineralization")
        if recovery:
            identity_parts.append(f"processed via {recovery}")
        if suite:
            identity_parts.append(f"({suite} metal suite)")
        identity = " ".join(identity_parts)

        # Pull up to 3 "Exclude X" rule criteria to instruct Exa to skip those
        excluded_examples: list[str] = []
        for c in (analog_rule or {}).get("excluded_subtypes") or []:
            excluded_examples.append(c.replace("_", " "))
            if len(excluded_examples) >= 3:
                break
        exclusion_clause = (
            f" Do NOT include {', '.join(excluded_examples)} projects."
            if excluded_examples else ""
        )

        grade_min = (analog_rule or {}).get("grade_min")
        grade_max = (analog_rule or {}).get("grade_max")
        rule_grade_unit = (analog_rule or {}).get("grade_unit") or grade_unit or ""
        grade_hint = (f" Grade range {grade_min}–{grade_max} {rule_grade_unit}."
                      if grade_min and grade_max else "")

        query = (
            f"Find 5-8 {identity} with confirmed NI 43-101 or JORC resource estimates."
            f"{grade_hint}{exclusion_clause}"
            f" {exclude_str}"
            f" For each project provide: project name, company, country and region, "
            f"deposit type and sub-type, host rock type, mineralization style and mode "
            f"(primary sulfide vs supergene oxide), tectonic belt or geological district, "
            f"alteration assemblage, primary metal suite, processing method "
            f"(flotation / heap leach / ISCR / SX-EW / CIL), total resource tonnage (Mt), "
            f"grade and unit, resource category, project stage, and technical report reference."
        ).strip()

    else:
        # Fallback: legacy generic query when no profile available
        location_hint = f"Prefer projects in or near {country}." if country else ""
        geo_parts: list[str] = []
        if mineralization_style:
            geo_parts.append(f"with {mineralization_style} mineralization")
        if host_rock:
            geo_parts.append(f"hosted in {host_rock}")
        geo_criteria: list[str] = []
        for c in (analog_rule or {}).get("analog_criteria") or []:
            if not c.lower().startswith("exclude") and len(geo_criteria) < 3:
                geo_criteria.append(c)
        if geo_criteria:
            geo_parts.append("(" + "; ".join(geo_criteria) + ")")
        geo_description = " ".join(geo_parts)

        grade_min = (analog_rule or {}).get("grade_min")
        grade_max = (analog_rule or {}).get("grade_max")
        rule_grade_unit = (analog_rule or {}).get("grade_unit") or grade_unit or ""
        if grade_min and grade_max:
            grade_hint = f"Grade range {grade_min}–{grade_max} {rule_grade_unit}.".strip()
        elif grade_value and grade_unit:
            grade_hint = f"Grade approximately {grade_value} {grade_unit}."
        else:
            grade_hint = ""

        query = (
            f"Find 5-8 {deposit_str} {material} deposits "
            f"{geo_description + ' ' if geo_description else ''}"
            f"with confirmed NI 43-101 or JORC resource estimates. "
            f"{grade_hint + ' ' if grade_hint else ''}"
            f"{location_hint + ' ' if location_hint else ''}"
            f"{exclude_str + ' ' if exclude_str else ''}"
            f"For each project provide: project name, company, country, deposit type, "
            f"host rock type, mineralization style, geological district or province, "
            f"total resource tonnage (Mt), grade and unit, resource category "
            f"(Measured/Indicated/Inferred), project stage, and technical report reference."
        )

    payload = {
        "query": query,
        "type": "deep",
        "systemPrompt": (
            "You are a mining industry geologist. For each project, prioritize describing "
            "the geological characteristics: deposit type AND finer sub-type (e.g. alkalic "
            "porphyry vs Laramide porphyry vs IOCG oxide), host rock type, mineralization "
            "mode (primary sulfide vs supergene oxide), tectonic belt, alteration "
            "assemblage, and processing method. Resource figures must come from NI 43-101 "
            "or JORC compliant technical reports — do not include exploration targets "
            "without a resource estimate. Report exact figures with units."
        ),
        "outputSchema": {
            "type": "text",
            "description": (
                "For each comparable project list: project name, company, country, region, "
                "deposit type and sub-type, host rock, mineralization style and mode, "
                "tectonic belt, alteration assemblage, processing method, metal suite, "
                "total resource tonnage (Mt), grade and unit, resource category, "
                "project stage, and the technical report reference."
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
