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
  "location_name": string | null
}}

Project context: {company} - {project_name} ({material})

SOURCE TEXT:
{source_text}
"""
    raw = _grok([{"role": "user", "content": prompt}])
    if not raw:
        return {f: None for f in TARGET_FIELDS}
    try:
        parsed = json.loads(raw)
        clean = {k: v for k, v in parsed.items() if k in TARGET_FIELDS}
        for f in TARGET_FIELDS:
            clean.setdefault(f, None)
        found = sum(1 for v in clean.values() if v is not None)
        logger.info(f"[Extract] {found}/{len(TARGET_FIELDS)} fields extracted")
        return clean
    except json.JSONDecodeError as e:
        logger.error(f"[Extract] JSON error: {e}")
        return {f: None for f in TARGET_FIELDS}


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
{source_text[:6000]}

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
Material type: {material}

For each project extract:
{{
  "name": string,
  "company": string,
  "country": string | null,
  "deposit_type": string | null,
  "tonnage_mt": number | null,   (in million tonnes)
  "grade_value": number | null,
  "grade_unit": string | null,
  "project_stage": string | null,
  "mining_method": string | null,
  "source_url": string | null
}}

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
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("projects", "analogs", "results"):
                if isinstance(data.get(key), list):
                    return data[key]
    except json.JSONDecodeError:
        pass
    return []


def score_analogs(
    target_project: dict,
    candidates: list[dict],
) -> list[dict]:
    """
    LLM scores each candidate analog for relevance to the target project (0-100).
    Returns candidates with similarity_score and similarity_reasons added.
    """
    if not candidates:
        return []

    prompt = f"""You are a mining geology expert. Score each candidate project for similarity
to the TARGET project. Consider: deposit type, material, grade, tonnage, mining method, location.

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
