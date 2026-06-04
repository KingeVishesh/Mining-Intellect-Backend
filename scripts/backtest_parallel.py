#!/usr/bin/env python3
"""Backtest saved Parallel.ai model runs against published MRE truth.

This script does not call Parallel. It evaluates rows already saved in
`model_runs` with model_type `parallel_pre_mre` and/or `parallel_post_mre`
against the official MRE mirror fields on `projects`:

    mre_mi_tonnage_mt, mre_mi_grade,
    mre_inferred_tonnage_mt, mre_inferred_grade

Use this for a fast accuracy check after running `gold_model_builder` on one
or more projects.

Examples:
    python3 scripts/backtest_parallel.py
    python3 scripts/backtest_parallel.py --model-type parallel_pre_mre
    python3 scripts/backtest_parallel.py --all-runs --threshold 0.10
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from nodes.supabase_ops import get_client  # noqa: E402
from nodes.parallel_gold_model import _blind_result_mentions_mre_anchor  # noqa: E402


TROY_OZ_PER_TONNE = 32150.7466
DEFAULT_MODEL_TYPES = ("parallel_pre_mre", "parallel_post_mre")


def pct_err(predicted: Optional[float], actual: Optional[float]) -> Optional[float]:
    if predicted is None or actual is None:
        return None
    if actual == 0:
        return math.inf if predicted != 0 else 0.0
    return (predicted - actual) / actual


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.1f}%"


def contained_oz(tonnage_mt: Optional[float], grade_gpt: Optional[float]) -> Optional[float]:
    if tonnage_mt is None or grade_gpt is None:
        return None
    return tonnage_mt * grade_gpt * TROY_OZ_PER_TONNE


def official_truth(project: Dict[str, Any]) -> Dict[str, Optional[float]]:
    mi_t = project.get("mre_mi_tonnage_mt")
    mi_g = project.get("mre_mi_grade")
    inf_t = project.get("mre_inferred_tonnage_mt")
    inf_g = project.get("mre_inferred_grade")

    total_t = None
    if mi_t is not None or inf_t is not None:
        total_t = float(mi_t or 0.0) + float(inf_t or 0.0)

    mi_oz = contained_oz(float(mi_t), float(mi_g)) if mi_t is not None and mi_g is not None else None
    inf_oz = contained_oz(float(inf_t), float(inf_g)) if inf_t is not None and inf_g is not None else None
    total_oz = None
    if mi_oz is not None or inf_oz is not None:
        total_oz = float(mi_oz or 0.0) + float(inf_oz or 0.0)

    total_g = None
    if total_t and total_t > 0 and mi_t is not None and mi_g is not None:
        total_g = ((float(mi_t) * float(mi_g)) + (float(inf_t or 0.0) * float(inf_g or 0.0))) / total_t

    return {
        "total_tonnage_mt": total_t,
        "total_grade_gpt": total_g,
        "total_contained_oz": total_oz,
        "mi_tonnage_mt": float(mi_t) if mi_t is not None else None,
        "mi_grade_gpt": float(mi_g) if mi_g is not None else None,
        "inferred_tonnage_mt": float(inf_t) if inf_t is not None else None,
        "inferred_grade_gpt": float(inf_g) if inf_g is not None else None,
    }


def select_runs(model_types: Iterable[str], latest_only: bool) -> List[Dict[str, Any]]:
    client = get_client()
    res = (
        client.table("model_runs")
        .select(
            "id,project_id,run_at,model_type,status,"
            "tonnage_mt,grade_value,total_contained,"
            "mi_tonnage_mt,mi_grade,mi_contained,"
            "inferred_resource_mt,inferred_grade,inferred_contained,"
            "model_output_json,signal_contributions_json"
        )
        .in_("model_type", list(model_types))
        .eq("status", "complete")
        .order("run_at", desc=True)
        .execute()
    )
    rows = res.data or []
    if not latest_only:
        return rows

    seen = set()
    latest = []
    for row in rows:
        key = (row.get("project_id"), row.get("model_type"))
        if key in seen:
            continue
        seen.add(key)
        latest.append(row)
    return latest


def fetch_projects(project_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    ids = sorted({pid for pid in project_ids if pid})
    if not ids:
        return {}
    res = (
        get_client().table("projects")
        .select(
            "id,name,material,"
            "mre_mi_tonnage_mt,mre_mi_grade,mre_mi_contained,"
            "mre_inferred_tonnage_mt,mre_inferred_grade,mre_inferred_contained"
        )
        .in_("id", ids)
        .execute()
    )
    return {row["id"]: row for row in (res.data or [])}


def run_backtest(model_types: Iterable[str], latest_only: bool, threshold: float) -> int:
    runs = select_runs(model_types, latest_only=latest_only)
    projects = fetch_projects(row["project_id"] for row in runs)

    evaluated = []
    skipped = []
    for row in runs:
        project = projects.get(row["project_id"]) or {}
        truth = official_truth(project)
        if truth["total_tonnage_mt"] is None or truth["total_grade_gpt"] is None:
            skipped.append((row, project))
            continue

        pred_contained = row.get("total_contained")
        if pred_contained is not None and pred_contained < 10000:
            # Some old non-Parallel rows stored contained as Moz/t. Parallel
            # rows are already oz, but this guard makes accidental inclusion
            # obvious in the display instead of silently comparing wrong units.
            pred_contained = float(pred_contained) * 1_000_000.0

        errors = {
            "tonnage": pct_err(row.get("tonnage_mt"), truth["total_tonnage_mt"]),
            "grade": pct_err(row.get("grade_value"), truth["total_grade_gpt"]),
            "contained": pct_err(pred_contained, truth["total_contained_oz"]),
            "mi_tonnage": pct_err(row.get("mi_tonnage_mt"), truth["mi_tonnage_mt"]),
            "mi_grade": pct_err(row.get("mi_grade"), truth["mi_grade_gpt"]),
            "inferred_tonnage": pct_err(row.get("inferred_resource_mt"), truth["inferred_tonnage_mt"]),
            "inferred_grade": pct_err(row.get("inferred_grade"), truth["inferred_grade_gpt"]),
        }
        leak_detected = (
            row.get("model_type") == "parallel_pre_mre"
            and _blind_result_mentions_mre_anchor({
                "anchor_used": (row.get("model_output_json") or {}).get("anchor_used"),
                "methodology": (row.get("model_output_json") or {}).get("methodology"),
                "conviction": (row.get("model_output_json") or {}).get("conviction"),
                "analogs_used": (row.get("model_output_json") or {}).get("analogs_used"),
                "analogs_rejected": (row.get("model_output_json") or {}).get("analogs_rejected"),
                "signal_contributions_json": row.get("signal_contributions_json"),
            })
        )
        pass_core = (
            not leak_detected
            and all(
            errors[k] is not None
            and not math.isinf(errors[k])
            and abs(errors[k]) <= threshold
            for k in ("tonnage", "grade", "contained")
            )
        )
        evaluated.append((row, project, truth, errors, pass_core, leak_detected))

    print(f"[parallel-backtest] rows evaluated: {len(evaluated)}")
    print(f"[parallel-backtest] rows skipped without MRE truth: {len(skipped)}")
    print(f"[parallel-backtest] threshold: ±{threshold * 100:.0f}%")
    print()

    for row, project, truth, errors, pass_core, leak_detected in evaluated:
        print("=" * 92)
        print(f"{project.get('name')}  |  {row['model_type']}  |  {row.get('run_at')}")
        print("-" * 92)
        print(f"{'Metric':24s} {'Predicted':>14s} {'Official MRE':>14s} {'Error':>10s}")
        print(f"{'Total tonnage (Mt)':24s} {row.get('tonnage_mt') or 0:14.3f} {truth['total_tonnage_mt'] or 0:14.3f} {fmt_pct(errors['tonnage']):>10s}")
        print(f"{'Total grade (g/t)':24s} {row.get('grade_value') or 0:14.3f} {truth['total_grade_gpt'] or 0:14.3f} {fmt_pct(errors['grade']):>10s}")
        print(f"{'Contained Au (oz)':24s} {row.get('total_contained') or 0:14.0f} {truth['total_contained_oz'] or 0:14.0f} {fmt_pct(errors['contained']):>10s}")
        print(f"{'M&I tonnage (Mt)':24s} {row.get('mi_tonnage_mt') or 0:14.3f} {truth['mi_tonnage_mt'] or 0:14.3f} {fmt_pct(errors['mi_tonnage']):>10s}")
        print(f"{'M&I grade (g/t)':24s} {row.get('mi_grade') or 0:14.3f} {truth['mi_grade_gpt'] or 0:14.3f} {fmt_pct(errors['mi_grade']):>10s}")
        print(f"{'Inferred tonnage (Mt)':24s} {row.get('inferred_resource_mt') or 0:14.3f} {truth['inferred_tonnage_mt'] or 0:14.3f} {fmt_pct(errors['inferred_tonnage']):>10s}")
        print(f"{'Inferred grade (g/t)':24s} {row.get('inferred_grade') or 0:14.3f} {truth['inferred_grade_gpt'] or 0:14.3f} {fmt_pct(errors['inferred_grade']):>10s}")
        if leak_detected:
            print("Blind leakage: target MRE/resource-anchor language detected")
        print(f"Core pass: {'PASS' if pass_core else 'FAIL'}")

    if evaluated:
        print()
        print("=" * 92)
        by_type: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Optional[float]], bool, bool]]] = defaultdict(list)
        for item in evaluated:
            by_type[item[0]["model_type"]].append(item)
        for model_type, items in sorted(by_type.items()):
            pass_count = sum(1 for *_rest, passed, _leak in items if passed)
            leak_count = sum(1 for *_rest, _passed, leak in items if leak)
            print(f"{model_type}: {pass_count}/{len(items)} pass")
            if leak_count:
                print(f"  rejected blind leakage: {leak_count}")
            for key in ("tonnage", "grade", "contained"):
                vals = [
                    abs(item[3][key])
                    for item in items
                    if item[3][key] is not None and not math.isinf(item[3][key])
                ]
                if vals:
                    print(f"  MAPE {key:9s}: {sum(vals) / len(vals) * 100:.1f}%")

    return 0 if evaluated else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-type",
        action="append",
        choices=DEFAULT_MODEL_TYPES,
        help="Model type to evaluate. Repeat to include multiple. Defaults to both Parallel types.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Evaluate every saved run. Default evaluates latest run per project/model_type.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Core pass threshold as a fraction. Default 0.05.",
    )
    args = parser.parse_args()

    model_types = tuple(args.model_type or DEFAULT_MODEL_TYPES)
    return run_backtest(model_types, latest_only=not args.all_runs, threshold=args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
