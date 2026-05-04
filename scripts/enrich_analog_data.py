"""
enrich_analog_data.py

Backfill tonnage_mt and grade_value for projects rows that have nulls.
Uses Exa Answer API with a structured output schema to extract MRE data.

Usage:
    python3 scripts/enrich_analog_data.py [--material silver] [--dry-run] [--limit 50]

Priority order (worst case first):
    silver, copper, gold, uranium — or pass --material to target one.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from typing import Optional, Tuple

import requests

# Add project root to path
sys.path.insert(0, "/Users/visheshjain/Documents/Mining-Intellect-Backend")

from config import settings
from nodes.supabase_ops import get_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXA_ANSWER_URL = "https://api.exa.ai/answer"

# Map material name to expected grade unit
_GRADE_UNITS = {
    "silver":    "g/t",
    "gold":      "g/t",
    "platinum":  "g/t",
    "palladium": "g/t",
    "copper":    "%",
    "nickel":    "%",
    "uranium":   "%",
    "iron":      "%",
    "lead":      "%",
    "zinc":      "%",
    "molybdenum":"%",
}


def _ask_exa_resource(
    project_name: str,
    material: str,
    country: Optional[str],
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Query Exa Answer API for total mineral resource estimate.
    Returns (tonnage_mt, grade_value, grade_unit) or (None, None, None).
    """
    api_key = settings.exa_api_key
    if not api_key:
        raise RuntimeError("EXA_API_KEY not set")

    grade_unit = _GRADE_UNITS.get((material or "").lower(), "g/t")
    location_str = f" in {country}" if country else ""
    query = (
        f"What is the total mineral resource estimate (tonnage in Mt and {material} grade "
        f"in {grade_unit}) for the {project_name} mining project{location_str}? "
        f"Include all resource categories: Measured, Indicated, and Inferred combined."
    )

    payload = {
        "query": query,
        "system_prompt": (
            "You are a mining industry analyst specializing in resource estimation. "
            "Extract the TOTAL mineral resource tonnage in million tonnes (Mt) and the "
            f"average grade in {grade_unit} from technical reports, press releases, or "
            "NI 43-101 filings. If only individual categories are available, sum them. "
            "Return null for fields you cannot find."
        ),
        "output_schema": {
            "type": "object",
            "properties": {
                "tonnage_mt": {
                    "type": ["number", "null"],
                    "description": f"Total mineral resource in million tonnes (Mt). E.g. 150.5 for 150.5 Mt",
                },
                "grade_value": {
                    "type": ["number", "null"],
                    "description": f"Average resource grade in {grade_unit}. E.g. 150 for 150 g/t Ag",
                },
                "grade_unit": {
                    "type": "string",
                    "description": f"Grade unit, typically {grade_unit}",
                },
                "source_note": {
                    "type": "string",
                    "description": "Brief note about the data source/date",
                },
            },
            "required": ["tonnage_mt", "grade_value", "grade_unit"],
        },
        "text": False,
    }

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(EXA_ANSWER_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        logger.warning(f"  Request error for '{project_name}': {e}")
        return None, None, None

    if resp.status_code != 200:
        logger.warning(f"  HTTP {resp.status_code} for '{project_name}': {resp.text[:200]}")
        return None, None, None

    data = resp.json()
    raw_answer = data.get("answer")
    if raw_answer is None:
        logger.warning(f"  No 'answer' field for '{project_name}'")
        return None, None, None

    if isinstance(raw_answer, str):
        try:
            answer = json.loads(raw_answer)
        except json.JSONDecodeError:
            logger.warning(f"  Could not parse answer for '{project_name}': {raw_answer[:200]}")
            return None, None, None
    else:
        answer = raw_answer

    tonnage = answer.get("tonnage_mt")
    grade = answer.get("grade_value")
    unit = answer.get("grade_unit") or grade_unit
    note = answer.get("source_note", "")

    if tonnage is None and grade is None:
        logger.info(f"  No resource data found for '{project_name}'")
        return None, None, None

    # Sanity checks
    if tonnage is not None and (tonnage <= 0 or tonnage > 100_000):
        logger.warning(f"  Suspicious tonnage={tonnage} for '{project_name}' — discarding")
        tonnage = None
    if grade is not None and grade <= 0:
        logger.warning(f"  Suspicious grade={grade} for '{project_name}' — discarding")
        grade = None

    logger.info(
        f"  Found: {tonnage} Mt @ {grade} {unit}"
        + (f" [{note}]" if note else "")
    )
    return tonnage, grade, unit


def fetch_empty_projects(material: Optional[str], limit: int) -> list:
    """Fetch projects with null tonnage or grade."""
    q = get_client().table("projects").select("id,name,material,country,tonnage_mt,grade_value")
    if material:
        q = q.eq("material", material)
    # Get rows where both fields are null (worst case — no data at all)
    res = q.is_("tonnage_mt", "null").limit(limit).execute()
    return res.data or []


def enrich_project(row: dict, dry_run: bool) -> bool:
    """Query Exa for resource data and update the DB row. Returns True on success."""
    project_id = row["id"]
    name = row.get("name", "Unknown")
    material = row.get("material", "")
    country = row.get("country")

    logger.info(f"Processing: {name} ({material}, {country or 'unknown country'})")

    tonnage, grade, unit = _ask_exa_resource(name, material, country)

    if tonnage is None and grade is None:
        return False

    if dry_run:
        logger.info(f"  [DRY RUN] Would update: tonnage_mt={tonnage}, grade_value={grade}, grade_unit={unit}")
        return True

    update = {"id": project_id}
    if tonnage is not None:
        update["tonnage_mt"] = tonnage
    if grade is not None:
        update["grade_value"] = grade
        update["grade_unit"] = unit

    try:
        get_client().table("projects").update(update).eq("id", project_id).execute()
        logger.info(f"  Updated DB: {update}")
        return True
    except Exception as e:
        logger.error(f"  DB update failed for {project_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Backfill analog tonnage/grade from Exa")
    parser.add_argument("--material", help="Target material (e.g. silver, copper). All if omitted.")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to process (default 50)")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing to DB")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between API calls (default 1.5)")
    args = parser.parse_args()

    material_label = args.material or "all materials"
    logger.info(f"Starting enrichment: {material_label}, limit={args.limit}, dry_run={args.dry_run}")

    rows = fetch_empty_projects(args.material, args.limit)
    logger.info(f"Found {len(rows)} projects with null tonnage_mt")

    if not rows:
        logger.info("Nothing to enrich.")
        return

    success = 0
    failed = 0
    for i, row in enumerate(rows):
        ok = enrich_project(row, args.dry_run)
        if ok:
            success += 1
        else:
            failed += 1

        if i < len(rows) - 1:
            time.sleep(args.delay)

    logger.info(f"\nDone — {success} enriched, {failed} no data found, {len(rows)} total")


if __name__ == "__main__":
    main()
