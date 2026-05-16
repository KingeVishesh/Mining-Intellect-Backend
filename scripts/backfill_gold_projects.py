"""
One-shot bulk backfill for the 80 gold projects that returned 0 analogs
in the May-17 audit. For each project, re-run the heuristic detect_*
functions on whatever freeform text is already on the row (deposit_type,
mineralization_style, host_rock, country, region, district) and write the
detected slugs back to the structured columns.

Idempotent: only fills NULL columns; never overwrites an existing slug.
Re-runnable.

Usage:
    python3 scripts/backfill_gold_projects.py            # dry-run
    python3 scripts/backfill_gold_projects.py --apply    # write changes
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nodes import geo_taxonomy
from nodes.rules_engine import sanitize_deposit_type
from nodes.supabase_ops import get_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_PROFILE_FIELDS = (
    "deposit_subtype", "mineralization_pattern", "mineralization_mode",
    "tectonic_belt", "metal_suite", "alteration_signature",
    "recovery_method", "host_rock_class", "project_stage_class",
    "mining_method_class",
)


def derive_for_project(row: dict) -> dict[str, str]:
    """Return a dict of fields to UPDATE (only nulls that we can fill)."""
    # Sanitize deposit_type ({Epithermal} → Epithermal) so detect_* see clean text
    clean_dep = sanitize_deposit_type(row.get("deposit_type"))
    derived: dict[str, str] = {}

    # If the sanitizer normalized the deposit_type, write that back too
    if clean_dep and clean_dep != row.get("deposit_type"):
        derived["deposit_type"] = clean_dep

    # Subtype
    if not row.get("deposit_subtype"):
        sub = geo_taxonomy.detect_subtype(
            clean_dep, row.get("mineralization_style"),
            row.get("alteration_signature"), row.get("district") or row.get("location_name"),
        )
        if sub:
            derived["deposit_subtype"] = sub

    # Pattern
    if not row.get("mineralization_pattern"):
        pat = geo_taxonomy.detect_pattern(
            row.get("mineralization_style"), row.get("mining_method"),
            row.get("processing_method"), clean_dep,
        )
        if pat:
            derived["mineralization_pattern"] = pat

    # Mode
    if not row.get("mineralization_mode"):
        mode = geo_taxonomy.detect_mode(
            row.get("processing_method"), row.get("mineralization_style"),
            row.get("district") or row.get("location_name"), clean_dep,
        )
        if mode:
            derived["mineralization_mode"] = mode

    # Belt
    if not row.get("tectonic_belt"):
        belt = geo_taxonomy.detect_belt(
            row.get("country"), row.get("region"), row.get("district"),
        )
        if belt:
            derived["tectonic_belt"] = belt

    # Metal suite — Gold defaults to au_only when no by-products mentioned
    if not row.get("metal_suite"):
        suite = geo_taxonomy.detect_metal_suite(
            row.get("material"), None, row.get("district"), clean_dep,
        )
        if suite:
            derived["metal_suite"] = suite

    # Alteration
    if not row.get("alteration_signature"):
        alt = geo_taxonomy.detect_alteration_signature(
            None, row.get("district") or row.get("location_name"), clean_dep,
        )
        if alt:
            derived["alteration_signature"] = alt

    # Recovery method
    if not row.get("recovery_method"):
        rec = geo_taxonomy.detect_recovery_method(
            row.get("processing_method"), row.get("location_name"), clean_dep,
        )
        if rec:
            derived["recovery_method"] = rec

    # Host rock class
    if not row.get("host_rock_class"):
        host = geo_taxonomy.detect_host_class(
            row.get("host_rock"), clean_dep, row.get("mineralization_style"),
        )
        if host:
            derived["host_rock_class"] = host

    # Stage class
    if not row.get("project_stage_class"):
        stage = geo_taxonomy.detect_stage_class(
            row.get("project_stage"), None, row.get("location_name"),
        )
        if stage:
            derived["project_stage_class"] = stage

    # Mining method class
    if not row.get("mining_method_class"):
        mm = geo_taxonomy.detect_mining_method_class(
            row.get("mining_method"), row.get("processing_method"),
            row.get("location_name"),
        )
        if mm:
            derived["mining_method_class"] = mm

    return derived


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                          help="Write changes (default: dry-run)")
    parser.add_argument("--material", default="gold",
                          help="Filter by material (default: gold)")
    args = parser.parse_args()

    client = get_client()
    rows = client.table("projects").select("*").ilike("material", args.material).execute().data or []
    logger.info(f"Loaded {len(rows)} {args.material} projects")

    updated = 0
    no_change = 0
    derive_counts: dict[str, int] = {}
    for r in rows:
        derived = derive_for_project(r)
        if not derived:
            no_change += 1
            continue
        for k in derived:
            derive_counts[k] = derive_counts.get(k, 0) + 1
        logger.info(f"  {r['name'][:50]:<50} → {list(derived.keys())}")
        if args.apply:
            client.table("projects").update(derived).eq("id", r["id"]).execute()
        updated += 1

    action = "APPLIED" if args.apply else "DRY-RUN"
    logger.info(
        f"\n[{action}] {updated} rows updated, {no_change} unchanged.\n"
        f"Fields filled (cumulative):\n  "
        + "\n  ".join(f"{k:25} {v}" for k, v in sorted(
            derive_counts.items(), key=lambda x: -x[1]
        ))
    )


if __name__ == "__main__":
    main()
