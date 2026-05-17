"""
Bulk-re-research script for projects with thin structured metadata.

Finds gold (or material-specified) projects whose deposit_type is null or
empty and re-runs the project_research graph against them. The updated
graph applies:
  - the synthesis bug fix in derive_geological_profile_node
  - the Grok deposit_type probe fallback
  - the cross-field country-conflict detector
  - the new sub-trend taxonomy

Reports before / after counts for each structured field so you can see
what got recovered. Idempotent: never overwrites a non-null field with
null (project_research handles that downstream).

Usage:
    python3 scripts/refresh_thin_projects.py            # dry-run
    python3 scripts/refresh_thin_projects.py --apply    # actually re-research
    python3 scripts/refresh_thin_projects.py --apply --limit 5
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphs.project_research import graph as research_graph
from nodes.supabase_ops import get_client, get_project

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Fields we measure recovery on. If any of these flips from null to a
# non-null value, that's a "recovery" credit.
_RECOVERY_FIELDS = [
    "deposit_type", "deposit_subtype", "mineralization_pattern",
    "mineralization_mode", "tectonic_belt", "metal_suite",
    "host_rock_class", "project_stage_class", "mining_method_class",
]


def _list_thin_projects(material: str, limit: int | None = None) -> List[Dict[str, Any]]:
    """Find projects of `material` with empty deposit_type."""
    q = (
        get_client()
        .table("projects")
        .select("id,name,country,region,district,deposit_type,deposit_subtype,"
                "company_name")
        .ilike("material", material)
        .or_("deposit_type.is.null,deposit_type.eq.")
        .order("name")
    )
    if limit:
        q = q.limit(limit)
    res = q.execute()
    return res.data or []


def _snapshot(row: Dict[str, Any]) -> Dict[str, Any]:
    """Capture just the fields we're trying to recover, for before/after diff."""
    return {f: row.get(f) for f in _RECOVERY_FIELDS}


def _diff_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Return the field-level recoveries: which fields newly populated."""
    recovered: Dict[str, Any] = {}
    for f in _RECOVERY_FIELDS:
        if not before.get(f) and after.get(f):
            recovered[f] = after.get(f)
    return recovered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="gold")
    parser.add_argument("--limit", type=int, default=None,
                          help="Process only first N matches")
    parser.add_argument("--apply", action="store_true",
                          help="Actually run the graph (default: dry-run list only)")
    parser.add_argument("--rate-limit-sleep", type=float, default=1.0)
    args = parser.parse_args()

    projects = _list_thin_projects(args.material, args.limit)
    print(f"[refresh] {len(projects)} {args.material} project(s) with empty deposit_type")
    for p in projects:
        print(f"  - {p['name'][:60]}   "
              f"(country={p.get('country')!r}, region={p.get('region')!r})")

    if not args.apply:
        print()
        print("[refresh] DRY-RUN — pass --apply to actually re-research")
        return 0

    print()
    print("[refresh] APPLYING — re-running project_research graph for each...")
    print()

    summary = {"total": len(projects), "recovered_deposit_type": 0,
                "recovered_any_field": 0, "no_change": 0, "errors": 0}
    field_recovery: Dict[str, int] = {f: 0 for f in _RECOVERY_FIELDS}
    recoveries: List[Dict[str, Any]] = []

    for i, p in enumerate(projects, 1):
        name = p["name"]
        print(f"  [{i:2}/{len(projects)}] {name[:60]}", flush=True)
        before = _snapshot(p)
        try:
            result = research_graph.invoke({
                "project_id": p["id"],
                "project_name": name,
                "material": args.material,
                "company": p.get("company_name") or "",
            })
        except Exception as e:
            logger.error(f"    ! exception: {e}")
            summary["errors"] += 1
            time.sleep(args.rate_limit_sleep)
            continue

        # Reload from DB to see what actually got saved
        try:
            after_row = get_project(p["id"]) or {}
        except Exception:
            after_row = {}
        after = _snapshot(after_row)
        recovered = _diff_snapshots(before, after)

        if recovered:
            summary["recovered_any_field"] += 1
            if "deposit_type" in recovered:
                summary["recovered_deposit_type"] += 1
            for f in recovered:
                field_recovery[f] += 1
            print(f"      ✓ recovered: {list(recovered.keys())}")
            recoveries.append({"name": name, "recovered": recovered})
        else:
            summary["no_change"] += 1
            print(f"      · no change")

        time.sleep(args.rate_limit_sleep)

    print()
    print("== Summary ==")
    for k, v in summary.items():
        print(f"  {k:30s} {v}")
    print()
    print("Field-level recovery counts:")
    for f, c in sorted(field_recovery.items(), key=lambda x: -x[1]):
        print(f"  {f:30s} {c}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
