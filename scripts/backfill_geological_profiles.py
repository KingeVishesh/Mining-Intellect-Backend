"""
Backfill the 6 geological profile columns on existing `projects` and
`report_analogs` rows using nodes/geo_taxonomy heuristic detectors over the
existing freeform deposit_type, mineralization_style, processing_method,
host_rock, district, region, country, location_name fields.

Designed to be re-runnable — only fills columns that are currently null. Rows
that produce no inferable values after detection are logged but left null.

Usage:
    python3 scripts/backfill_geological_profiles.py            # both tables, dry-run
    python3 scripts/backfill_geological_profiles.py --apply    # write changes
    python3 scripts/backfill_geological_profiles.py --table projects --apply
    python3 scripts/backfill_geological_profiles.py --table report_analogs --apply
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List, Optional

sys.path.insert(0, "/Users/visheshjain/Documents/Mining-Intellect-Backend")

from nodes import geo_taxonomy
from nodes.supabase_ops import get_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROFILE_COLUMNS = [
    "deposit_subtype", "mineralization_mode", "tectonic_belt",
    "metal_suite", "alteration_signature", "recovery_method",
]


def _derive_profile_for_project(row: Dict) -> Dict[str, Optional[str]]:
    """Detect the 6 profile values from a `projects` row using existing fields."""
    derived: Dict[str, Optional[str]] = {}
    if not row.get("deposit_subtype"):
        derived["deposit_subtype"] = geo_taxonomy.detect_subtype(
            row.get("deposit_type"), row.get("mineralization_style"),
            row.get("alteration_signature"), row.get("district") or row.get("location_name"),
        )
    if not row.get("mineralization_mode"):
        derived["mineralization_mode"] = geo_taxonomy.detect_mode(
            row.get("processing_method"), row.get("mineralization_style"),
            row.get("district") or row.get("location_name"), row.get("deposit_type"),
        )
    if not row.get("tectonic_belt"):
        derived["tectonic_belt"] = geo_taxonomy.detect_belt(
            row.get("country"), row.get("region"), row.get("district"),
        )
    if not row.get("metal_suite"):
        derived["metal_suite"] = geo_taxonomy.detect_metal_suite(
            row.get("material"),
            ", ".join(row.get("by_product_commodities") or []) if isinstance(row.get("by_product_commodities"), list) else None,
            row.get("location_name") or row.get("district"),
            row.get("deposit_type"),
        )
    if not row.get("alteration_signature"):
        derived["alteration_signature"] = geo_taxonomy.detect_alteration_signature(
            None, row.get("district") or row.get("location_name"), row.get("deposit_type"),
        )
    if not row.get("recovery_method"):
        derived["recovery_method"] = geo_taxonomy.detect_recovery_method(
            row.get("processing_method"), row.get("location_name"), row.get("deposit_type"),
        )
    return {k: v for k, v in derived.items() if v is not None}


def _derive_profile_for_analog(row: Dict) -> Dict[str, Optional[str]]:
    """Detect the 6 profile values from a `report_analogs` row."""
    derived: Dict[str, Optional[str]] = {}
    if not row.get("analog_deposit_subtype"):
        sub = geo_taxonomy.detect_subtype(
            row.get("analog_deposit_type"), row.get("analog_mineralization_style"),
            row.get("analog_alteration_signature"), row.get("analog_district"),
        )
        if sub is not None:
            derived["analog_deposit_subtype"] = sub
    if not row.get("analog_mineralization_mode"):
        mode = geo_taxonomy.detect_mode(
            None, row.get("analog_mineralization_style"),
            row.get("analog_district"), row.get("analog_deposit_type"),
        )
        if mode is not None:
            derived["analog_mineralization_mode"] = mode
    if not row.get("analog_tectonic_belt"):
        belt = geo_taxonomy.detect_belt(
            row.get("analog_country"), None, row.get("analog_district"),
        )
        if belt is not None:
            derived["analog_tectonic_belt"] = belt
    if not row.get("analog_metal_suite"):
        suite = geo_taxonomy.detect_metal_suite(
            row.get("analog_material"), None,
            row.get("analog_district"), row.get("analog_deposit_type"),
        )
        if suite is not None:
            derived["analog_metal_suite"] = suite
    if not row.get("analog_alteration_signature"):
        alt = geo_taxonomy.detect_alteration_signature(
            None, row.get("analog_district"), row.get("analog_deposit_type"),
        )
        if alt is not None:
            derived["analog_alteration_signature"] = alt
    if not row.get("analog_recovery_method"):
        rec = geo_taxonomy.detect_recovery_method(
            None, row.get("analog_district"), row.get("analog_deposit_type"),
        )
        if rec is not None:
            derived["analog_recovery_method"] = rec
    return derived


def _fetch_paginated(table: str, select: str, page_size: int = 500) -> List[Dict]:
    rows: List[Dict] = []
    offset = 0
    client = get_client()
    while True:
        res = client.table(table).select(select).range(offset, offset + page_size - 1).execute()
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def backfill_projects(apply: bool) -> None:
    client = get_client()
    select = (
        "id,material,deposit_type,mineralization_style,host_rock,processing_method,"
        "country,region,district,location_name,alteration_signature,"
        "by_product_commodities,"
        "deposit_subtype,mineralization_mode,tectonic_belt,metal_suite,recovery_method"
    )
    rows = _fetch_paginated("projects", select)
    logger.info(f"[projects] fetched {len(rows)} rows")

    updated = 0
    no_change = 0
    for row in rows:
        derived = _derive_profile_for_project(row)
        if not derived:
            no_change += 1
            continue
        logger.info(f"[projects] {row['id']}: {derived}")
        if apply:
            client.table("projects").update(derived).eq("id", row["id"]).execute()
        updated += 1
    logger.info(
        f"[projects] {'APPLIED' if apply else 'DRY-RUN'}: "
        f"{updated} rows derived, {no_change} unchanged"
    )


def backfill_report_analogs(apply: bool) -> None:
    client = get_client()
    select = (
        "id,analog_name,analog_material,analog_deposit_type,analog_mineralization_style,"
        "analog_host_rock,analog_country,analog_district,analog_alteration_signature,"
        "analog_deposit_subtype,analog_mineralization_mode,analog_tectonic_belt,"
        "analog_metal_suite,analog_recovery_method"
    )
    rows = _fetch_paginated("report_analogs", select)
    logger.info(f"[report_analogs] fetched {len(rows)} rows")

    updated = 0
    no_change = 0
    for row in rows:
        derived = _derive_profile_for_analog(row)
        if not derived:
            no_change += 1
            continue
        if apply:
            client.table("report_analogs").update(derived).eq("id", row["id"]).execute()
        updated += 1
        if updated % 50 == 0:
            logger.info(f"[report_analogs] {updated} updates so far...")
    logger.info(
        f"[report_analogs] {'APPLIED' if apply else 'DRY-RUN'}: "
        f"{updated} rows derived, {no_change} unchanged"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument("--table", choices=["projects", "report_analogs", "both"],
                        default="both")
    args = parser.parse_args()

    if args.table in ("projects", "both"):
        backfill_projects(apply=args.apply)
    if args.table in ("report_analogs", "both"):
        backfill_report_analogs(apply=args.apply)


if __name__ == "__main__":
    main()
