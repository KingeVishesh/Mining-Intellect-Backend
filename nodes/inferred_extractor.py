"""Inferred-resource extractor.

For each analog used in a model run, this module pulls the published
M&I + Inferred breakdown from the most recent technical report (NI 43-101,
JORC, SK-1300) via Exa's Answer API. The result is cached on the
`public.analogs` row (`analog_inferred_tonnage_mt`, `analog_inferred_grade`)
so subsequent runs reuse it.

Background: `build_model_1`'s Inferred axis predicts the project's
Inferred bucket independently of the M&I axis, using the analog pool's
`inferred_tonnage_mt` / `inferred_grade` values. Until this extractor
was added, those fields existed only on backtest fixture JSON files and
the production library returned 0 for every project's Inferred prediction.

Shape returned by `extract_inferred_breakdown()`:

    {
      "inferred_tonnage_mt": float | None,    # Inferred-bucket tonnage in Mt
      "inferred_grade":      float | None,    # Inferred-bucket grade (native unit)
      "mi_tonnage_mt":       float | None,    # M&I total for cross-validation
      "mi_grade":            float | None,
      "as_of_year":          int | None,      # year of the MRE/PEA cited
      "source_url":          str | None,
      "confidence":          "high" | "medium" | "low",
      "source":              "exa",
      "extracted_at":        ISO 8601 timestamp,
    }

If the analog publishes ONLY Inferred (no M&I yet) or ONLY M&I (no
Inferred halo remaining), the corresponding field is None — that IS
valid data; the build_model_1 Inferred-axis just ignores rows where
`inferred_tonnage_mt` is None when computing its geometric mean.
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

# Inferred breakdowns change only when the operator publishes a new MRE,
# which is typically annual. A 90-day cache is comfortable.
DEFAULT_MAX_AGE_DAYS = 90


def should_refetch(
    inferred_data: Optional[Dict],
    fetched_at: Optional[str],
    force: bool = False,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> bool:
    """Return True when the analog's Inferred breakdown should be refetched.

    Missing data → always refetch. Forced (caller passed an explicit flag)
    → always refetch. Otherwise refetch only when the cached timestamp is
    older than `max_age_days`.
    """
    if force:
        return True
    if not inferred_data and not fetched_at:
        return True
    if not fetched_at:
        return True
    try:
        ts = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return True
    age = datetime.now(timezone.utc) - ts
    return age > timedelta(days=max_age_days)


def extract_inferred_breakdown(
    analog_name: str,
    material: str,
    country: Optional[str] = None,
    region: Optional[str] = None,
    deposit_type: Optional[str] = None,
) -> Optional[Dict]:
    """Query Exa Answer for the analog's most recent published M&I + Inferred
    breakdown. Returns the payload dict, or None on any failure.

    Errors are logged but never raised so that a missing-Inferred analog
    still feeds Model 1 (it just won't contribute to the Inferred-axis
    posterior — same as if its `inferred_tonnage_mt` were genuinely None).
    """
    api_key = settings.exa_api_key
    if not api_key:
        logger.warning("[InferredExtractor] EXA_API_KEY not set, skipping fetch")
        return None

    loc_parts = [p for p in (region, country) if p and p.strip()]
    location = ", ".join(loc_parts) if loc_parts else ""
    deposit_clause = f"({deposit_type})" if deposit_type else ""

    query = (
        f"What is the most recent Mineral Resource Estimate for the {analog_name} "
        f"{material} mining project in {location} {deposit_clause}? "
        f"Report the Measured + Indicated (M&I) tonnage and grade separately "
        f"from the Inferred tonnage and grade. Use the latest NI 43-101, JORC, "
        f"or SK-1300 technical report. Tonnage in millions of tonnes (Mt). "
        f"Grade in g/t for gold/silver/PGMs, or % for base metals. Cite the "
        f"source URL and the publication year."
    )

    payload = {
        "query": query,
        "system_prompt": (
            "You are a mining-industry analyst pulling Mineral Resource Estimate "
            "(MRE) data from public company disclosures and technical reports. "
            "Return tonnage and grade EXACTLY as published; do not estimate or "
            "interpolate. If a category is not reported for this deposit (e.g. "
            "an early-stage project may have only Inferred, a near-depleted "
            "producer may have only Reserves with no current Inferred), return "
            "null for that field — do NOT fabricate. Use the most recent "
            "publication only; do not blend multiple vintages."
        ),
        "output_schema": {
            "type": "object",
            "properties": {
                "mi_tonnage_mt": {
                    "type": ["number", "null"],
                    "description": "Measured + Indicated tonnage in Mt as published.",
                },
                "mi_grade": {
                    "type": ["number", "null"],
                    "description": "M&I grade in native units (g/t for Au/Ag/PGM, % for base).",
                },
                "inferred_tonnage_mt": {
                    "type": ["number", "null"],
                    "description": "Inferred tonnage in Mt as published.",
                },
                "inferred_grade": {
                    "type": ["number", "null"],
                    "description": "Inferred grade in same native units as mi_grade.",
                },
                "as_of_year": {
                    "type": ["integer", "null"],
                    "description": "Year the MRE/PEA was published.",
                },
                "source_url": {
                    "type": ["string", "null"],
                    "description": "Primary source URL (technical report, press release, or filing).",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "high=verbatim from NI 43-101 / JORC / SK-1300; medium=press release; low=secondary source",
                },
            },
            "required": ["confidence"],
        },
        "text": False,
    }

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    try:
        resp = requests.post(EXA_ANSWER_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        logger.warning(f"[InferredExtractor] Request error for '{analog_name}': {e}")
        return None

    if resp.status_code != 200:
        logger.warning(
            f"[InferredExtractor] HTTP {resp.status_code} for '{analog_name}': "
            f"{resp.text[:200]}"
        )
        return None

    data = resp.json()
    raw = data.get("answer")
    if raw is None:
        logger.warning(f"[InferredExtractor] No 'answer' for '{analog_name}'")
        return None

    answer = raw
    if isinstance(raw, str):
        try:
            answer = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                f"[InferredExtractor] Could not parse answer JSON for "
                f"'{analog_name}': {raw[:200]}"
            )
            return None

    result = {
        "mi_tonnage_mt":       answer.get("mi_tonnage_mt"),
        "mi_grade":            answer.get("mi_grade"),
        "inferred_tonnage_mt": answer.get("inferred_tonnage_mt"),
        "inferred_grade":      answer.get("inferred_grade"),
        "as_of_year":          answer.get("as_of_year"),
        "source_url":          answer.get("source_url"),
        "confidence":          answer.get("confidence", "low"),
        "source":              "exa",
        "extracted_at":        datetime.now(timezone.utc).isoformat(),
    }

    # If neither M&I nor Inferred surfaced, the extraction yielded nothing
    # useful — return None so callers can mark the row as "extraction
    # attempted, no data found" and not retry until staleness window.
    if (
        result["mi_tonnage_mt"] is None
        and result["inferred_tonnage_mt"] is None
    ):
        logger.info(
            f"[InferredExtractor] No MRE breakdown found for '{analog_name}' "
            f"(confidence={result['confidence']})"
        )
        return None

    logger.info(
        f"[InferredExtractor] '{analog_name}': "
        f"M&I {result['mi_tonnage_mt']} Mt @ {result['mi_grade']}, "
        f"Inferred {result['inferred_tonnage_mt']} Mt @ {result['inferred_grade']}, "
        f"as_of={result['as_of_year']}, confidence={result['confidence']}"
    )
    return result
