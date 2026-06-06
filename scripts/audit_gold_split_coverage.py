#!/usr/bin/env python3
"""Audit gold-project M&I/Inferred split coverage.

This is read-only. It gives the backtest/model-fill work a concrete queue:

* projects with official M&I/Inferred truth fields for strict blind backtests
* projects with current model split fields populated
* projects that likely need official split extraction from an existing MRE
* projects that need a blind model run because no total resource is present

Examples:
    python3 scripts/audit_gold_split_coverage.py
    python3 scripts/audit_gold_split_coverage.py --status needs_model_run
    python3 scripts/audit_gold_split_coverage.py --json-out artifacts/gold_split_coverage.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from nodes import supabase_ops  # noqa: E402


OFFICIAL_SPLIT_FIELDS = (
    "mre_mi_tonnage_mt",
    "mre_mi_grade",
    "mre_inferred_tonnage_mt",
    "mre_inferred_grade",
)
MODEL_SPLIT_FIELDS = (
    "mi_tonnage_mt",
    "mi_grade",
    "inferred_resource_mt",
    "inferred_grade",
)
TOTAL_RESOURCE_FIELDS = ("tonnage_mt", "grade_value")

PROJECT_SELECT = ",".join((
    "id",
    "name",
    "material",
    "country",
    "region",
    "district",
    "deposit_type",
    "deposit_subtype",
    "tectonic_belt",
    "tonnage_mt",
    "grade_value",
    "grade_unit",
    "total_contained",
    "resource_category",
    "resource_compliance_standard",
    "resource_vintage_year",
    "mre_mi_tonnage_mt",
    "mre_mi_grade",
    "mre_inferred_tonnage_mt",
    "mre_inferred_grade",
    "mi_tonnage_mt",
    "mi_grade",
    "inferred_resource_mt",
    "inferred_grade",
    "updated_at",
))


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _has_all(row: Dict[str, Any], fields: Iterable[str]) -> bool:
    return all(_present(row.get(field)) for field in fields)


def _missing(row: Dict[str, Any], fields: Iterable[str]) -> List[str]:
    return [field for field in fields if not _present(row.get(field))]


def classify_gold_project(row: Dict[str, Any]) -> Dict[str, Any]:
    has_official_split = _has_all(row, OFFICIAL_SPLIT_FIELDS)
    has_model_split = _has_all(row, MODEL_SPLIT_FIELDS)
    has_total_resource = _has_all(row, TOTAL_RESOURCE_FIELDS)
    has_any_split = has_official_split or has_model_split

    if has_official_split:
        primary_status = "official_split_ready"
        next_action = "use_for_backtest"
    elif has_model_split:
        primary_status = "model_split_ready"
        next_action = "review_model_output"
    elif has_total_resource:
        primary_status = "needs_official_split_extraction"
        next_action = "run_backfill_gold_mre_truth"
    else:
        primary_status = "needs_model_run"
        next_action = "run_blind_model_after_accuracy_gate"

    return {
        "project_id": row.get("id"),
        "name": row.get("name"),
        "primary_status": primary_status,
        "next_action": next_action,
        "has_official_split": has_official_split,
        "has_model_split": has_model_split,
        "has_any_split": has_any_split,
        "has_total_resource": has_total_resource,
        "missing_official_split_fields": _missing(row, OFFICIAL_SPLIT_FIELDS),
        "missing_model_split_fields": _missing(row, MODEL_SPLIT_FIELDS),
        "missing_total_resource_fields": _missing(row, TOTAL_RESOURCE_FIELDS),
        "tonnage_mt": row.get("tonnage_mt"),
        "grade_value": row.get("grade_value"),
        "deposit_subtype": row.get("deposit_subtype"),
        "tectonic_belt": row.get("tectonic_belt"),
        "resource_category": row.get("resource_category"),
        "resource_vintage_year": row.get("resource_vintage_year"),
    }


def fetch_gold_projects() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    client = supabase_ops.get_client()
    while True:
        res = (
            client.table("projects")
            .select(PROJECT_SELECT)
            .ilike("material", "gold")
            .order("name")
            .range(offset, offset + 999)
            .execute()
        )
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    classified = [classify_gold_project(row) for row in rows]
    summary = {
        "gold_projects": len(classified),
        "official_split_ready": sum(1 for row in classified if row["has_official_split"]),
        "model_split_ready": sum(1 for row in classified if row["has_model_split"]),
        "any_split_ready": sum(1 for row in classified if row["has_any_split"]),
        "total_resource_ready": sum(1 for row in classified if row["has_total_resource"]),
        "missing_any_split": sum(1 for row in classified if not row["has_any_split"]),
        "needs_official_split_extraction": sum(
            1 for row in classified if row["primary_status"] == "needs_official_split_extraction"
        ),
        "needs_model_run": sum(1 for row in classified if row["primary_status"] == "needs_model_run"),
    }
    return {"summary": summary, "projects": classified}


def _print_summary(payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    print("Gold split coverage")
    for key in (
        "gold_projects",
        "official_split_ready",
        "model_split_ready",
        "any_split_ready",
        "total_resource_ready",
        "missing_any_split",
        "needs_official_split_extraction",
        "needs_model_run",
    ):
        print(f"  {key:34s} {summary[key]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        choices=(
            "official_split_ready",
            "model_split_ready",
            "needs_official_split_extraction",
            "needs_model_run",
            "missing_any_split",
        ),
        default=None,
        help="Print only projects matching this status.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    payload = summarize(fetch_gold_projects())
    _print_summary(payload)

    projects = payload["projects"]
    if args.status:
        if args.status == "missing_any_split":
            projects = [row for row in projects if not row["has_any_split"]]
        elif args.status == "model_split_ready":
            projects = [row for row in projects if row["has_model_split"]]
        else:
            projects = [row for row in projects if row["primary_status"] == args.status]
        if args.limit:
            projects = projects[: args.limit]
        print()
        print(f"{args.status}: {len(projects)} shown")
        for row in projects:
            print(
                f"{row['project_id']} | {row['name']} | "
                f"T={row['tonnage_mt']} G={row['grade_value']} | {row['next_action']}"
            )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"json_out: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
