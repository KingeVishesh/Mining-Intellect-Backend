"""
Field Extractor — LLM converts Exa narrative text into structured DB fields.
Uses a two-pass approach:
  Pass 1: Grok extracts all fields from source text
  Pass 2: Grok judge verifies extractions against source (accept / reject / not_applicable / search_miss)
"""
from __future__ import annotations
import json
import logging
from typing import Optional

import requests
from config import settings

logger = logging.getLogger(__name__)

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-3"

# CAD -> USD fallback rate
_CAD_USD_FALLBACK = 0.73

# All fields this extractor targets
TARGET_FIELDS = [
    "country", "region", "company_name", "commodity",
    "deposit_type", "project_stage",
    "tonnage_mt", "grade_value", "grade_unit", "resource_category",
    "mining_method", "processing_method", "recovery_rate",
    "mine_life_years", "depth_meters", "width_meters", "strike_length_meters",
    "npv_usd_millions", "capex_usd_millions",
    "irr_percent", "opex_per_unit", "payback_years", "production_rate_per_year",
    "latitude", "longitude", "location_name",
    # Extended fields
    "host_rock", "mineralization_style",
    "resource_size_value", "resource_size_unit",
    "by_product_commodities",
    "final_product", "ownership_type", "district",
    "start_year", "end_year",
    "energy_source", "climate_terrain",
    "permitting_status",
    "elevation_meters",
    # Geological profile (used by analog_finder cascading match)
    "deposit_subtype", "mineralization_mode", "tectonic_belt",
    "metal_suite", "alteration_signature", "recovery_method",
]


def _grok(messages: list, timeout: int = 60) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {settings.grok_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROK_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        logger.error(f"[Grok] Request error: {e}")
        return None
    if resp.status_code != 200:
        logger.error(f"[Grok] HTTP {resp.status_code}: {resp.text[:300]}")
        return None
    return resp.json()["choices"][0]["message"]["content"]


def extract_fields(
    source_text: str,
    project_name: str,
    company: str,
    material: str,
    cad_usd_rate: float = _CAD_USD_FALLBACK,
) -> dict:
    """
    Pass 1: Extract DB fields from Exa narrative text.
    Returns dict with all TARGET_FIELDS (nulls for missing).
    """
    prompt = f"""You are a data extraction tool for mining finance.
Convert the mining project summary below into a flat JSON object.

Rules:
1. Use null (never 0 or empty string) when a value is not found or not applicable.

CRITICAL — Resource Estimate (highest-priority fields):
2. tonnage_mt: Total mineral resource in MILLION tonnes (Mt). Search for phrases like:
   "Indicated resource of X Mt", "total resource of X million tonnes",
   "Measured+Indicated+Inferred of X Mt", "X,000 kt" (divide kt by 1000).
   Sum all categories if only individual categories are stated.
   If source gives kt (thousand tonnes), DIVIDE by 1000 to get Mt.
   IMPORTANT: Prefer the TOTAL resource over M&I alone if available.
3. grade_value: Average resource grade as a NUMBER only (no units). grade_unit = the unit
   string e.g. "% U3O8", "g/t Au", "g/t Ag". Look for phrases like "grading X g/t",
   "averaging X% Cu", "at a grade of X". Extract the number matching the grade_unit.

Other fields:
4. npv_usd_millions = after-tax NPV in USD millions. If CAD, multiply by {cad_usd_rate:.4f}.
5. capex_usd_millions = initial CAPEX in USD millions. If CAD, multiply by {cad_usd_rate:.4f}.
6. recovery_rate = metallurgical recovery as 0-100 number. null if not stated.
7. project_stage must be one of: Exploration, PEA, PFS, Feasibility, Construction, Production.
8. irr_percent = after-tax IRR as a number (e.g. 52.4). null if not found.
9. opex_per_unit = operating cost per unit in USD. null if not found.
10. payback_years = payback period in years. Convert months to years if needed.
11. production_rate_per_year = annual production rate as a number.
12. latitude/longitude = decimal degrees if explicitly stated. null otherwise.
13. location_name = human-readable location (e.g. "Northern Ontario, Canada").
14. host_rock: the rock type that hosts the deposit (e.g. "granite", "limestone",
    "greenstone", "schist"). null if not stated.
15. mineralization_style: the mineralisation style or deposit sub-type (e.g. "epithermal vein",
    "porphyry", "VMS", "IOCG", "skarn", "MVT", "orogenic", "sediment-hosted"). null if not stated.
16. resource_size_value: contained metal quantity as a NUMBER only (e.g. 3.2 for "3.2 Moz gold",
    400 for "400 Mlbs copper"). Prefer the total resource. null if not stated.
    resource_size_unit: the unit string for the contained metal (e.g. "Moz", "Mlbs", "kt Cu").
    null if resource_size_value is null.
17. by_product_commodities: JSON array of by-product metal names mentioned alongside the primary
    resource (e.g. ["Silver", "Molybdenum"]). Use [] if none are mentioned. NOT null — always an array.
18. final_product: the planned saleable product form (e.g. "doré bars", "copper concentrate",
    "nickel sulphate", "uranium oxide (U3O8)", "iron ore pellets"). null if not stated.
19. ownership_type: the project ownership structure as stated in the text
    (e.g. "100% owned", "50% JV with Company X", "optioned from Vendor Y"). null if not stated.
20. district: geological district or sub-regional name within the broader region
    (e.g. "Abitibi Greenstone Belt", "Atacama Region", "Pilbara Craton"). null if not stated.
21. start_year: planned or actual mine start or construction start year as an INTEGER
    (e.g. 2026). Found in project timelines, FS construction schedules. null if not stated.
22. end_year: planned mine closure year as an INTEGER. If not stated but start_year and
    mine_life_years are both known, compute start_year + mine_life_years. null otherwise.
23. energy_source: primary energy or power source for the operation
    (e.g. "grid power", "diesel generators", "LNG", "solar + diesel hybrid"). null if not stated.
24. climate_terrain: brief climate and terrain description of the project site
    (e.g. "Arctic tundra", "Tropical rainforest", "High-altitude Andes", "Semi-arid desert",
    "Boreal forest"). null if not stated.
25. permitting_status: JSON array of permitting milestones already achieved
    (e.g. ["Environmental Impact Assessment approved", "Mining licence granted",
    "Federal permits received"]). Use [] if none are mentioned. NOT null — always an array.
26. elevation_meters: project site elevation above sea level in METRES as a number. null if not stated.

Geological profile fields (CRITICAL for analog matching — use ONLY values from the controlled vocabulary):

27. deposit_subtype: finer-grained classification than deposit_type. Use ONE of:
    "alkalic_porphyry"      — alkalic Cu-Au porphyries (BC Quesnel/Stikine: Mt. Milligan, Mt. Polley, Cadia)
    "calc_alkalic_porphyry" — typical Cu-Mo porphyries
    "laramide_porphyry"     — Arizona/Sonora/Chile calc-alkaline Laramide-age porphyries
    "high_sulfidation_lithocap_porphyry"
    "iocg_oxide"            — IOCG with oxide blanket
    "iocg_sulfide"          — IOCG primary sulfide
    "iocg_hybrid"
    "oxide_iscr_supergene_blanket" — supergene Cu oxide blanket processed via in-situ recovery (Florence, Van Dyke)
    "low_sulfidation_epithermal" / "high_sulfidation_epithermal" / "intermediate_sulfidation_epithermal"
    "greenstone_orogenic" / "turbidite_orogenic" / "bif_hosted_orogenic" / "orogenic_general"
    "sedex" / "kupferschiefer_style" / "manto_cu" / "crd" / "mvt" / "redbed_cu" / "sediment_hosted_general"
    "vms_general" / "carlin_general"
    "cu_au_skarn" / "fe_skarn" / "zn_pb_skarn" / "w_mo_skarn" / "skarn_general"
    "merensky_reef" / "ug2_reef" / "platreef"
    "limonite_laterite" / "saprolite_laterite" / "laterite_general"
    "komatiite_hosted" / "conduit_hosted" / "magmatic_sulphide_general"
    "bif_general"
    null if you genuinely cannot tell.

28. mineralization_mode: ONE of:
    "primary_sulfide"      — primary sulfide ore (chalcopyrite/bornite/pyrite dominant)
    "supergene_oxide"      — weathered oxide ore (chrysocolla/malachite/atacamite)
    "mixed_oxide_sulfide"  — transition zone
    "refractory_sulfide"   — refractory Au in sulfide
    "free_milling_oxide"   — free-milling oxide gold
    "placer"
    null if not determinable.

29. tectonic_belt: mineralization province slug. ONE of:
    "bc_quesnel_stikine"   — BC Canada interior alkalic arc
    "yukon_tintina"
    "abitibi"              — Ontario/Quebec greenstone
    "newfoundland_appalachian"
    "laramide_southwest"   — Arizona/New Mexico/Sonora
    "great_basin_carlin"   — Nevada
    "andean"               — Chile/Peru/Argentina/Ecuador/Bolivia/Colombia
    "brazilian_shield" / "central_african_copperbelt" / "bushveld" / "west_african_birimian"
    "tanzania_archean" / "lachlan" / "yilgarn" / "fennoscandian" / "central_asian_orogenic"
    "indonesia_philippines_arc" / "new_caledonia_laterite" / "iberian_pyrite"
    null if not determinable.

30. metal_suite: characteristic metal grouping. ONE of:
    "cu_au" / "cu_mo" / "cu_au_co_sc" (with cobalt and/or scandium byproducts)
    "cu_ag" / "cu_zn_pb"
    "au_only" / "au_ag" / "ag_pb_zn"
    "ni_cu_pge" / "ni_co"
    "pt_pd_rh" / "u_only" / "fe_only" / "li_only" / "ree_only"
    null if not determinable.

31. alteration_signature: ONE of:
    "potassic_phyllic"       — porphyry core
    "potassic_propylitic"
    "sodic_calcic"           — IOCG / alkalic intrusion-related
    "hematite_specularite"   — IOCG oxidized
    "argillic_advanced_argillic" — high-sulfidation epithermal
    "sericitic_quartz" / "skarn_calc_silicate" / "silicification_decalcified"
    "chlorite_carbonate" / "supergene_oxidation" / "lateritic_weathering"
    null if not determinable.

32. recovery_method: primary metallurgical recovery route. ONE of:
    "flotation" / "heap_leach" / "iscr" / "sx_ew" / "cn_leach" / "cil_cip"
    "gravity" / "smelting" / "atmospheric_leach" / "hpal"
    null if not stated.

Output ONLY this JSON object, no other text:

{{
  "country": string | null,
  "region": string | null,
  "company_name": string | null,
  "commodity": string | null,
  "deposit_type": string | null,
  "project_stage": string | null,
  "tonnage_mt": number | null,
  "grade_value": number | null,
  "grade_unit": string | null,
  "resource_category": string | null,
  "mining_method": string | null,
  "processing_method": string | null,
  "recovery_rate": number | null,
  "mine_life_years": number | null,
  "depth_meters": number | null,
  "width_meters": number | null,
  "strike_length_meters": number | null,
  "npv_usd_millions": number | null,
  "capex_usd_millions": number | null,
  "irr_percent": number | null,
  "opex_per_unit": number | null,
  "payback_years": number | null,
  "production_rate_per_year": number | null,
  "latitude": number | null,
  "longitude": number | null,
  "location_name": string | null,
  "host_rock": string | null,
  "mineralization_style": string | null,
  "resource_size_value": number | null,
  "resource_size_unit": string | null,
  "by_product_commodities": array,
  "final_product": string | null,
  "ownership_type": string | null,
  "district": string | null,
  "start_year": integer | null,
  "end_year": integer | null,
  "energy_source": string | null,
  "climate_terrain": string | null,
  "permitting_status": array,
  "elevation_meters": number | null,
  "deposit_subtype": string | null,
  "mineralization_mode": string | null,
  "tectonic_belt": string | null,
  "metal_suite": string | null,
  "alteration_signature": string | null,
  "recovery_method": string | null
}}

Project context: {company} - {project_name} ({material})

SOURCE TEXT:
{source_text}
"""
    raw = _grok([{"role": "user", "content": prompt}])
    if not raw:
        return _fill_geological_profile({f: None for f in TARGET_FIELDS}, material)
    try:
        parsed = json.loads(raw)
        clean = {k: v for k, v in parsed.items() if k in TARGET_FIELDS}
        for f in TARGET_FIELDS:
            clean.setdefault(f, None)
        clean = _fill_geological_profile(clean, material)
        found = sum(1 for v in clean.values() if v is not None)
        logger.info(f"[Extract] {found}/{len(TARGET_FIELDS)} fields extracted")
        return clean
    except json.JSONDecodeError as e:
        logger.error(f"[Extract] JSON error: {e}")
        return _fill_geological_profile({f: None for f in TARGET_FIELDS}, material)


# Controlled-vocab validation maps — Grok occasionally returns values outside
# the vocabulary even when prompted. Validate against the actual taxonomy slugs.
_VALID_SUBTYPES: frozenset[str] = frozenset({
    "alkalic_porphyry", "calc_alkalic_porphyry", "laramide_porphyry",
    "high_sulfidation_lithocap_porphyry",
    "iocg_oxide", "iocg_sulfide", "iocg_hybrid", "oxide_iscr_supergene_blanket",
    "low_sulfidation_epithermal", "high_sulfidation_epithermal",
    "intermediate_sulfidation_epithermal",
    "greenstone_orogenic", "turbidite_orogenic", "bif_hosted_orogenic", "orogenic_general",
    "sedex", "kupferschiefer_style", "manto_cu", "crd", "mvt", "redbed_cu",
    "sediment_hosted_general", "vms_general", "carlin_general",
    "cu_au_skarn", "fe_skarn", "zn_pb_skarn", "w_mo_skarn", "skarn_general",
    "merensky_reef", "ug2_reef", "platreef",
    "limonite_laterite", "saprolite_laterite", "laterite_general",
    "komatiite_hosted", "conduit_hosted", "magmatic_sulphide_general",
    "bif_general",
})


def _validate(value, allowed: frozenset) -> Optional[str]:
    """Return value if it's in the allowed set (case-insensitive), else None."""
    if not value or not isinstance(value, str):
        return None
    norm = value.strip().lower().replace(" ", "_").replace("-", "_")
    return norm if norm in allowed else None


def _fill_geological_profile(fields: dict, material: str) -> dict:
    """
    Validate the 6 geological profile fields against controlled vocabularies, then
    fill any nulls using nodes/geo_taxonomy heuristic detectors over the freeform
    text already extracted. Existing freeform deposit_type, mineralization_style,
    processing_method, district, region, country drive the inference.
    """
    from nodes import geo_taxonomy

    # Validate Grok output against vocabularies — fall back to heuristic on miss
    fields["deposit_subtype"] = _validate(fields.get("deposit_subtype"), _VALID_SUBTYPES)
    fields["mineralization_mode"] = _validate(
        fields.get("mineralization_mode"), frozenset(geo_taxonomy.MINERALIZATION_MODES)
    )
    fields["tectonic_belt"] = _validate(
        fields.get("tectonic_belt"), frozenset(geo_taxonomy.TECTONIC_BELTS.keys())
    )
    fields["metal_suite"] = _validate(fields.get("metal_suite"), frozenset(geo_taxonomy.METAL_SUITES))
    fields["alteration_signature"] = _validate(
        fields.get("alteration_signature"), frozenset(geo_taxonomy.ALTERATION_SIGNATURES)
    )
    fields["recovery_method"] = _validate(
        fields.get("recovery_method"), frozenset(geo_taxonomy.RECOVERY_METHODS)
    )

    # Heuristic fallback for any still-null fields
    if fields["deposit_subtype"] is None:
        fields["deposit_subtype"] = geo_taxonomy.detect_subtype(
            fields.get("deposit_type"), fields.get("mineralization_style"),
            fields.get("alteration_signature"), fields.get("location_name"),
        )
    if fields["mineralization_mode"] is None:
        fields["mineralization_mode"] = geo_taxonomy.detect_mode(
            fields.get("processing_method"), fields.get("mineralization_style"),
            fields.get("location_name"), fields.get("deposit_type"),
        )
    if fields["tectonic_belt"] is None:
        fields["tectonic_belt"] = geo_taxonomy.detect_belt(
            fields.get("country"), fields.get("region"), fields.get("district"),
        )
    if fields["metal_suite"] is None:
        byproducts_text = ", ".join(fields.get("by_product_commodities") or [])
        fields["metal_suite"] = geo_taxonomy.detect_metal_suite(
            material, byproducts_text, fields.get("location_name"), fields.get("deposit_type"),
        )
    if fields["alteration_signature"] is None:
        fields["alteration_signature"] = geo_taxonomy.detect_alteration_signature(
            None, fields.get("location_name"), fields.get("deposit_type"),
        )
    if fields["recovery_method"] is None:
        fields["recovery_method"] = geo_taxonomy.detect_recovery_method(
            fields.get("processing_method"), fields.get("location_name"),
            fields.get("deposit_type"),
        )
    return fields


def judge_fields(
    source_text: str,
    db_fields: dict,
    project_name: str,
    company: str,
    material: str,
    judge_only: Optional[list] = None,
) -> tuple[dict, dict]:
    """
    Pass 2: LLM judge verifies each extracted field against the source text.
    Returns (cleaned_fields, field_statuses).

    Verdicts:
      accept          — value supported by source text
      reject          — contradicts source or implausible (field set to null)
      not_applicable  — field doesn't apply at this project stage
      search_miss     — should exist but wasn't found (flag for retry)
    """
    fields_to_judge = judge_only or TARGET_FIELDS
    extracted_summary = {f: db_fields.get(f) for f in fields_to_judge}

    prompt = f"""You are a fact-checking agent for mining project data extraction.

PROJECT: {company} - {project_name} ({material})

EXTRACTED VALUES (some may be wrong or hallucinated):
{json.dumps(extracted_summary, indent=2)}

SOURCE TEXT (treat as ground truth):
{source_text[:12000]}

For each field return one of:
- "accept"         → value is explicitly stated or clearly derivable from the source text
- "reject"         → value is not supported by source text, looks wrong, or is physically implausible
- "not_applicable" → this field does not apply to this project (e.g. NPV for exploration-stage)
- "search_miss"    → field is null but SHOULD exist for a project at this stage

Rules:
1. For null fields: decide between not_applicable vs search_miss based on project stage.
2. For non-null fields: accept if consistent with source. Reject if it contradicts source.
3. Economic fields are search_miss only if the project has completed a PEA/PFS/FS.

Return ONLY this JSON:
{{
  "field_name": {{"verdict": "accept|reject|not_applicable|search_miss", "reason": "brief (reject only)"}},
  ...
}}
"""
    raw = _grok([{"role": "user", "content": prompt}])

    cleaned = dict(db_fields)
    statuses = {}

    if not raw:
        # Fallback: accept all non-null, mark nulls as search_miss
        for f in TARGET_FIELDS:
            statuses[f] = "found" if db_fields.get(f) is not None else "search_miss"
        return cleaned, statuses

    try:
        verdicts = json.loads(raw)
    except json.JSONDecodeError:
        for f in TARGET_FIELDS:
            statuses[f] = "found" if db_fields.get(f) is not None else "search_miss"
        return cleaned, statuses

    rejected = 0
    for field in fields_to_judge:
        entry = verdicts.get(field, {})
        verdict = entry.get("verdict", "accept") if isinstance(entry, dict) else "accept"
        reason = entry.get("reason", "") if isinstance(entry, dict) else ""

        if verdict == "reject":
            if cleaned.get(field) is not None:
                logger.warning(f"[Judge] Rejected {field}={cleaned[field]} — {reason}")
                cleaned[field] = None
                rejected += 1
            statuses[field] = "search_miss"
        elif verdict == "not_applicable":
            cleaned[field] = None
            statuses[field] = "not_applicable"
        elif verdict == "search_miss":
            statuses[field] = "search_miss"
        else:
            statuses[field] = "found" if cleaned.get(field) is not None else "search_miss"

    # Fields not in judge set keep existing status
    for field in TARGET_FIELDS:
        if field not in statuses:
            statuses[field] = "found" if db_fields.get(field) is not None else "not_found"

    logger.info(
        f"[Judge] accepted={sum(1 for s in statuses.values() if s=='found')}, "
        f"rejected={rejected}, "
        f"search_miss={sum(1 for s in statuses.values() if s=='search_miss')}, "
        f"not_applicable={sum(1 for s in statuses.values() if s=='not_applicable')}"
    )
    return cleaned, statuses


def extract_analog_projects(
    source_text: str,
    material: str,
    source_urls: list[str],
) -> list[dict]:
    """
    Extract a list of analog projects from Exa analog-search text.
    Returns a list of dicts matching the AnalogProject schema.
    """
    prompt = f"""Extract a list of mining project analogs from the text below.
Expected material type: {material}

For each project extract:
{{
  "name": string,
  "company": string | null,
  "country": string | null,
  "commodity": string | null,            (primary commodity as a single lowercase word: "silver", "gold", "copper", "nickel")
  "deposit_type": string | null,         (e.g. "porphyry copper-gold", "orogenic gold", "CRD silver-lead-zinc", "VMS")
  "host_rock": string | null,            (rock type hosting the deposit, e.g. "limestone", "granite", "greenstone schist", "rhyolite")
  "mineralization_style": string | null, (e.g. "epithermal vein", "porphyry disseminated Cu-Au", "CRD manto silver", "orogenic shear-hosted")
  "district": string | null,             (geological district or province, e.g. "Abitibi Greenstone Belt", "Central Andes porphyry belt", "Chihuahua Mexico")
  "tonnage_mt": number | null,           (in million tonnes)
  "grade_value": number | null,          (numeric value only, e.g. 350 for "350 g/t Ag")
  "grade_unit": string | null,           (unit string e.g. "g/t Ag", "% Cu", "% Ni")
  "project_stage": string | null,
  "mining_method": string | null,
  "processing_method": string | null,    (e.g. "flotation", "heap leach", "ISCR", "CIL")
  "region": string | null,               (sub-national region or state — e.g. "British Columbia", "Arizona")
  "deposit_subtype": string | null,      (controlled vocab: alkalic_porphyry, laramide_porphyry, iocg_oxide, oxide_iscr_supergene_blanket, etc. — see field doc)
  "mineralization_mode": string | null,  (primary_sulfide | supergene_oxide | mixed_oxide_sulfide | refractory_sulfide | free_milling_oxide | placer)
  "tectonic_belt": string | null,        (bc_quesnel_stikine | andean | laramide_southwest | lachlan | etc.)
  "metal_suite": string | null,          (cu_au | cu_mo | cu_au_co_sc | ni_cu_pge | au_only | etc.)
  "alteration_signature": string | null, (potassic_phyllic | sodic_calcic | hematite_specularite | argillic_advanced_argillic | etc.)
  "recovery_method": string | null,      (flotation | heap_leach | iscr | sx_ew | cn_leach | cil_cip | gravity | hpal)
  "source_url": string | null
}}

IMPORTANT: Extract "commodity" from the text — do not assume it is {material}.
IMPORTANT: host_rock, mineralization_style, deposit_subtype, mineralization_mode, tectonic_belt, and recovery_method are critical for analog cascading-match quality — extract them whenever derivable from the text.
For BC alkalic Cu-Au porphyries (Quesnel/Stikine: Mt. Milligan, Mt. Polley, Copper Mountain, Red Chris, Cadia) use deposit_subtype="alkalic_porphyry", tectonic_belt="bc_quesnel_stikine", mineralization_mode="primary_sulfide", recovery_method="flotation".
For Arizona oxide ISCR projects (Florence, Van Dyke) use deposit_subtype="oxide_iscr_supergene_blanket", tectonic_belt="laramide_southwest", mineralization_mode="supergene_oxide", recovery_method="iscr".
For Chile IOCG oxide (Marimaca) use deposit_subtype="iocg_oxide", tectonic_belt="andean", mineralization_mode="supergene_oxide", recovery_method="heap_leach".
If the text describes a gold project, set commodity to "gold" even if the expected material is "{material}".

Return ONLY a JSON array of project objects. No other text.

SOURCE TEXT:
{source_text}
"""
    raw = _grok([{"role": "user", "content": prompt}])
    if not raw:
        return []
    try:
        data = json.loads(raw)
        # Handle both array and {"projects": [...]} shapes
        analogs: list[dict] = []
        if isinstance(data, list):
            analogs = data
        elif isinstance(data, dict):
            for key in ("projects", "analogs", "results"):
                if isinstance(data.get(key), list):
                    analogs = data[key]
                    break
        # Validate + fill geological profile fields on each analog using its own
        # extracted commodity (not the parent search material).
        for a in analogs:
            if not isinstance(a, dict):
                continue
            mat = (a.get("commodity") or material or "").strip().lower()
            _fill_analog_profile(a, mat)
        return analogs
    except json.JSONDecodeError:
        pass
    return []


def _fill_analog_profile(analog: dict, material: str) -> None:
    """In-place validate + heuristic-fill the 6 geological profile fields on an analog dict."""
    from nodes import geo_taxonomy

    analog["deposit_subtype"] = _validate(analog.get("deposit_subtype"), _VALID_SUBTYPES)
    analog["mineralization_mode"] = _validate(
        analog.get("mineralization_mode"), frozenset(geo_taxonomy.MINERALIZATION_MODES)
    )
    analog["tectonic_belt"] = _validate(
        analog.get("tectonic_belt"), frozenset(geo_taxonomy.TECTONIC_BELTS.keys())
    )
    analog["metal_suite"] = _validate(analog.get("metal_suite"), frozenset(geo_taxonomy.METAL_SUITES))
    analog["alteration_signature"] = _validate(
        analog.get("alteration_signature"), frozenset(geo_taxonomy.ALTERATION_SIGNATURES)
    )
    analog["recovery_method"] = _validate(
        analog.get("recovery_method"), frozenset(geo_taxonomy.RECOVERY_METHODS)
    )

    if analog["deposit_subtype"] is None:
        analog["deposit_subtype"] = geo_taxonomy.detect_subtype(
            analog.get("deposit_type"), analog.get("mineralization_style"),
            analog.get("alteration_signature"), analog.get("district"),
        )
    if analog["mineralization_mode"] is None:
        analog["mineralization_mode"] = geo_taxonomy.detect_mode(
            analog.get("processing_method"), analog.get("mineralization_style"),
            analog.get("district"), analog.get("deposit_type"),
        )
    if analog["tectonic_belt"] is None:
        analog["tectonic_belt"] = geo_taxonomy.detect_belt(
            analog.get("country"), analog.get("region"), analog.get("district"),
        )
    if analog["metal_suite"] is None:
        analog["metal_suite"] = geo_taxonomy.detect_metal_suite(
            material, None, analog.get("district"), analog.get("deposit_type"),
        )
    if analog["alteration_signature"] is None:
        analog["alteration_signature"] = geo_taxonomy.detect_alteration_signature(
            None, analog.get("district"), analog.get("deposit_type"),
        )
    if analog["recovery_method"] is None:
        analog["recovery_method"] = geo_taxonomy.detect_recovery_method(
            analog.get("processing_method"), analog.get("district"), analog.get("deposit_type"),
        )


def score_analogs(
    target_project: dict,
    candidates: list[dict],
    commodity_criteria: list[str] | None = None,
) -> list[dict]:
    """
    LLM scores each candidate analog for relevance to the target project (0-100).
    Returns candidates with similarity_score and similarity_reasons added.
    commodity_criteria: optional list of domain-specific criteria derived from compiled rules.
    """
    if not candidates:
        return []

    criteria_block = ""
    if commodity_criteria:
        criteria_block = "\nCOMMODITY-SPECIFIC ANALOG CRITERIA (from compiled rules):\n" + \
                         "\n".join(f"- {c}" for c in commodity_criteria) + "\n"

    prompt = f"""You are a mining geology expert. Score each candidate analog for similarity
to the TARGET project using the rubric below.

SCORING RUBRIC:
- 85-100: Excellent — same deposit type, grade within 1.5×, tonnage within 2×
- 65-84:  Good — same deposit type OR grade/tonnage within 5×
- 45-64:  Acceptable — same commodity, different/unknown deposit type or missing data
- <45:    Poor — significant geological or scale mismatch, do not use
{criteria_block}
TARGET PROJECT:
{json.dumps(target_project, indent=2)}

CANDIDATES:
{json.dumps(candidates, indent=2)}

For each candidate return:
{{
  "name": string,
  "similarity_score": number (0-100),
  "similarity_reasons": ["reason 1", "reason 2"]
}}

Return ONLY a JSON array. No other text.
"""
    raw = _grok([{"role": "user", "content": prompt}], timeout=90)
    if not raw:
        return [{**c, "similarity_score": 50, "similarity_reasons": []} for c in candidates]
    try:
        scored = json.loads(raw)
        if isinstance(scored, list):
            # Merge scores back onto candidates
            score_map = {s["name"]: s for s in scored if isinstance(s, dict)}
            result = []
            for c in candidates:
                s = score_map.get(c.get("name", ""), {})
                result.append({
                    **c,
                    "similarity_score": s.get("similarity_score", 50),
                    "similarity_reasons": s.get("similarity_reasons", []),
                })
            return sorted(result, key=lambda x: x["similarity_score"], reverse=True)
    except json.JSONDecodeError:
        pass
    return candidates


def extract_new_projects(source_text: str, material: str) -> list[dict]:
    """
    Extract newly discovered project stubs from project_discovery Exa text.
    """
    prompt = f"""Extract a list of newly announced mining projects from the text below.
Material: {material}

For each project extract:
{{
  "name": string,
  "company_name": string | null,
  "country": string | null,
  "material": "{material}",
  "project_stage": string | null,
  "description": string | null
}}

Return ONLY a JSON array. No other text.

SOURCE TEXT:
{source_text}
"""
    raw = _grok([{"role": "user", "content": prompt}])
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("projects", "results"):
                if isinstance(data.get(key), list):
                    return data[key]
    except json.JSONDecodeError:
        pass
    return []
