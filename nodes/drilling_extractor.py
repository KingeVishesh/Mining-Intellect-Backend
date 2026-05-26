"""Drilling-evidence extractor.

For each project (and on demand, each analog), this module pulls
structured drilling metadata from public sources via Exa's Answer API.
The result is stored as the `drilling_evidence` JSONB column on the
target row; downstream Model 1 reads it as a tonnage signal via the
analog-derived tonnage-per-meter ratio.

Shape returned by `extract_drilling_evidence()`:

    {
      "total_holes": int | None,           # cumulative holes drilled
      "total_meters_drilled": float | None,
      "drilled_area_km2": float | None,    # mineralized footprint defined by drilling
      "best_intercepts": [
        {"hole_id": str, "from_m": float, "to_m": float,
         "interval_m": float, "grade_g_t": float, "source_url": str}
      ],
      "weighted_grade_g_t": float | None,  # length-weighted across reported intercepts
      "qa_qc_present": bool,
      "source": "exa",
      "extracted_at": ISO 8601 timestamp,
      "source_url": str | None,            # primary source cited by Exa
      "confidence": "high" | "medium" | "low",
    }

The Exa Answer API does the heavy lifting (web search + structured
extraction) when given an output_schema. We give it a tight schema and a
domain-specific system prompt — same pattern used in `nodes/exa_answer.py`
for coordinate lookup. No additional LLM call needed.
"""
from __future__ import annotations
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

from config import settings

logger = logging.getLogger(__name__)

EXA_ANSWER_URL = "https://api.exa.ai/answer"

# Drilling data older than this is considered stale — model_runner will
# refetch on the next run unless the cached version is still recent enough.
DEFAULT_MAX_AGE_DAYS = 7


def should_refetch(
    drilling_evidence: Optional[Dict],
    fetched_at: Optional[str],
    force: bool = False,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> bool:
    """Return True when fresh drilling data should be fetched.

    Forced (caller passed `fetch_recent_drill_holes=true` to the
    LangGraph input) → always refetch. Missing data → always refetch.
    Otherwise refetch only when the stored timestamp is older than
    `max_age_days`.
    """
    if force:
        return True
    if not drilling_evidence:
        return True
    if not fetched_at:
        return True
    try:
        ts = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return True
    age = datetime.now(timezone.utc) - ts
    return age > timedelta(days=max_age_days)


def extract_drilling_evidence(
    project_name: str,
    material: str,
    country: Optional[str] = None,
    region: Optional[str] = None,
    deposit_type: Optional[str] = None,
    company: Optional[str] = None,
) -> Optional[Dict]:
    """Query Exa Answer for structured drilling metadata.

    Returns the JSONB payload (without `extracted_at` — caller sets that
    when persisting), or None on any failure. Errors are logged but never
    raised so that a missing-drilling-data project still runs Model 1.
    """
    api_key = settings.exa_api_key
    if not api_key:
        logger.warning("[DrillingExtractor] EXA_API_KEY not set, skipping fetch")
        return None

    loc_parts = [p for p in (region, country) if p and p.strip()]
    location = ", ".join(loc_parts) if loc_parts else ""
    owner_clause = f"operated by {company} " if company else ""
    deposit_clause = f"({deposit_type})" if deposit_type else ""

    query = (
        f"Drilling program data for the {project_name} {material} mining project "
        f"{owner_clause}in {location} {deposit_clause}. "
        f"How many drill holes have been completed in total? How many total meters "
        f"have been drilled to date? Over what surface area in square kilometers? "
        f"List 1–5 of the best drilling intercepts reported: hole ID, depth from-to, "
        f"interval thickness in meters, gold or copper or silver grade in g/t or %, "
        f"and source URL. Has QA/QC protocols been disclosed?"
    )

    payload = {
        "query": query,
        "system_prompt": (
            "You are a mining-industry analyst pulling drilling-program statistics "
            "from public company disclosures (press releases, technical reports, "
            "NI 43-101 / JORC filings). Return numeric values exactly as cited; "
            "do not estimate. If a metric is not publicly reported, return null "
            "for that field — do not invent a value. Best intercepts must be "
            "verbatim from a source you can cite."
        ),
        "output_schema": {
            "type": "object",
            "properties": {
                "total_holes": {
                    "type": ["integer", "null"],
                    "description": "Cumulative number of drill holes completed.",
                },
                "total_meters_drilled": {
                    "type": ["number", "null"],
                    "description": "Cumulative meters drilled across all programs.",
                },
                "drilled_area_km2": {
                    "type": ["number", "null"],
                    "description": "Surface footprint covered by the drilling program, in square kilometers.",
                },
                "best_intercepts": {
                    "type": "array",
                    "description": "Up to 5 best intercepts reported in disclosures.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hole_id": {"type": "string"},
                            "from_m": {"type": ["number", "null"]},
                            "to_m": {"type": ["number", "null"]},
                            "interval_m": {"type": ["number", "null"]},
                            "grade_g_t": {"type": ["number", "null"],
                                "description": "Grade as reported, typically g/t for Au/Ag or % for Cu/Zn/Pb"},
                            "source_url": {"type": ["string", "null"]},
                        },
                    },
                },
                "qa_qc_present": {
                    "type": ["boolean", "null"],
                    "description": "Whether QA/QC protocols (blanks, standards, duplicates) are disclosed.",
                },
                "source_url": {
                    "type": ["string", "null"],
                    "description": "Primary source URL cited.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "high=numbers from technical report / regulatory filing; medium=press release; low=secondary source",
                },
            },
            "required": ["total_holes", "total_meters_drilled", "confidence"],
        },
        "text": False,
    }

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    try:
        resp = requests.post(EXA_ANSWER_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        logger.warning(f"[DrillingExtractor] Request error for '{project_name}': {e}")
        return None

    if resp.status_code != 200:
        logger.warning(
            f"[DrillingExtractor] HTTP {resp.status_code} for '{project_name}': "
            f"{resp.text[:200]}"
        )
        return None

    data = resp.json()
    raw = data.get("answer")
    if raw is None:
        logger.warning(f"[DrillingExtractor] No 'answer' for '{project_name}'")
        return None

    answer = raw
    if isinstance(raw, str):
        try:
            answer = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                f"[DrillingExtractor] Could not parse answer JSON for "
                f"'{project_name}': {raw[:200]}"
            )
            return None

    # Compute length-weighted grade across reported intercepts so we have a
    # single grade signal to feed the model when individual intercepts
    # don't reconcile to one number.
    intercepts = answer.get("best_intercepts") or []
    valid_intercepts = [
        ic for ic in intercepts
        if (ic.get("interval_m") or 0) > 0 and (ic.get("grade_g_t") or 0) > 0
    ]
    weighted_grade = None
    if valid_intercepts:
        total_m = sum(ic["interval_m"] for ic in valid_intercepts)
        if total_m > 0:
            weighted_grade = sum(
                ic["interval_m"] * ic["grade_g_t"] for ic in valid_intercepts
            ) / total_m

    result = {
        "total_holes": answer.get("total_holes"),
        "total_meters_drilled": answer.get("total_meters_drilled"),
        "drilled_area_km2": answer.get("drilled_area_km2"),
        "best_intercepts": intercepts,
        "weighted_grade_g_t": weighted_grade,
        "qa_qc_present": answer.get("qa_qc_present"),
        "source": "exa",
        "source_url": answer.get("source_url"),
        "confidence": answer.get("confidence", "low"),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    # Sanity check: at least one of total_holes / total_meters_drilled must
    # be present, otherwise the extraction effectively yielded nothing
    # useful. Return None so callers can fall back to no-drilling-signal.
    if (
        result["total_holes"] is None
        and result["total_meters_drilled"] is None
        and not valid_intercepts
    ):
        logger.info(
            f"[DrillingExtractor] No drilling metrics found for '{project_name}' "
            f"(confidence={result['confidence']})"
        )
        return None

    logger.info(
        f"[DrillingExtractor] '{project_name}': "
        f"{result['total_holes']} holes, "
        f"{result['total_meters_drilled']} m drilled, "
        f"{len(valid_intercepts)} intercepts, "
        f"confidence={result['confidence']}"
    )
    return result
