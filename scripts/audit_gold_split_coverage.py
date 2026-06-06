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
import glob
import json
import re
import sys
from itertools import zip_longest
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


def _project_name_key(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def project_name_to_id_map(rows: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    return {
        _project_name_key(row.get("name")): str(row.get("id"))
        for row in rows
        if row.get("id") and row.get("name")
    }


def _blank_backtest_status() -> Dict[str, Any]:
    return {
        "backtest_status": "not_backtest_eligible",
        "backtest_requested_count": 0,
        "backtest_evaluated_count": 0,
        "backtest_pass_count": 0,
        "backtest_miss_count": 0,
        "backtest_error_count": 0,
        "backtest_last_result": None,
        "backtest_last_error_class": None,
        "backtest_last_batch_id": None,
        "backtest_last_artifact": None,
    }


def _history_for(history: Dict[str, Any], project_id: Any) -> Dict[str, Any]:
    if not project_id:
        return {}
    return history.get(str(project_id), {})


def _backtest_status_for_project(
    row: Dict[str, Any],
    *,
    has_official_split: bool,
    history: Dict[str, Any],
) -> Dict[str, Any]:
    status = _blank_backtest_status()
    if not has_official_split:
        return status

    status["backtest_status"] = "ready_untested"
    project_history = _history_for(history, row.get("id"))
    if not project_history:
        return status

    status.update({
        "backtest_requested_count": project_history.get("requested_count", 0),
        "backtest_evaluated_count": project_history.get("evaluated_count", 0),
        "backtest_pass_count": project_history.get("pass_count", 0),
        "backtest_miss_count": project_history.get("miss_count", 0),
        "backtest_error_count": project_history.get("error_count", 0),
        "backtest_last_result": project_history.get("last_result"),
        "backtest_last_error_class": project_history.get("last_error_class"),
        "backtest_last_batch_id": project_history.get("last_batch_id"),
        "backtest_last_artifact": project_history.get("last_artifact"),
    })
    last_result = project_history.get("last_result")
    last_error_class = str(project_history.get("last_error_class") or "")
    if last_result == "pass":
        status["backtest_status"] = "validated_pass"
    elif last_result == "miss":
        status["backtest_status"] = "needs_accuracy_review"
    elif last_result == "error" and last_error_class.startswith("parallel_quota"):
        status["backtest_status"] = "retry_after_quota"
    elif last_result == "error":
        status["backtest_status"] = "retry_after_error"
    elif project_history.get("requested_count"):
        status["backtest_status"] = "requested_not_evaluated"
    return status


def classify_gold_project(
    row: Dict[str, Any],
    *,
    backtest_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    has_official_split = _has_all(row, OFFICIAL_SPLIT_FIELDS)
    has_model_split = _has_all(row, MODEL_SPLIT_FIELDS)
    has_total_resource = _has_all(row, TOTAL_RESOURCE_FIELDS)
    has_any_split = has_official_split or has_model_split
    backtest_history = backtest_history or {}

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

    classified = {
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
    classified.update(_backtest_status_for_project(
        row,
        has_official_split=has_official_split,
        history=backtest_history,
    ))
    return classified


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


def _leaderboard_name_map(payload: Dict[str, Any]) -> Dict[str, str]:
    target_selection = payload.get("target_selection") or {}
    project_ids = target_selection.get("project_ids") or []
    project_names = target_selection.get("project_names") or []
    return {
        _project_name_key(name): str(project_id)
        for project_id, name in zip_longest(project_ids, project_names)
        if project_id and name
    }


def _project_id_from_artifact_row(
    row: Dict[str, Any],
    name_map: Dict[str, str],
    project_name_to_id: Dict[str, str],
) -> Optional[str]:
    project_id = row.get("project_id")
    if project_id:
        return str(project_id)
    project_name = row.get("project")
    if project_name:
        key = _project_name_key(project_name)
        return name_map.get(key) or project_name_to_id.get(key)
    return None


def _touch_history(
    history: Dict[str, Dict[str, Any]],
    project_id: str,
    *,
    project_name: Optional[str],
    artifact: str,
    batch_id: Optional[str],
) -> Dict[str, Any]:
    row = history.setdefault(project_id, {
        "project_id": project_id,
        "project_name": project_name,
        "requested_count": 0,
        "evaluated_count": 0,
        "pass_count": 0,
        "miss_count": 0,
        "error_count": 0,
        "last_result": None,
        "last_error_class": None,
        "last_batch_id": None,
        "last_artifact": None,
    })
    if project_name and not row.get("project_name"):
        row["project_name"] = project_name
    row["last_batch_id"] = batch_id
    row["last_artifact"] = artifact
    return row


def load_backtest_history(
    paths: Iterable[Path],
    *,
    project_name_to_id: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    history: Dict[str, Dict[str, Any]] = {}
    project_name_to_id = project_name_to_id or {}
    for path in paths:
        artifact = str(path)
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        batch_id = payload.get("batch_id")
        target_selection = payload.get("target_selection") or {}
        project_ids = target_selection.get("project_ids") or []
        project_names = target_selection.get("project_names") or []
        name_map = _leaderboard_name_map(payload)

        for project_id, project_name in zip_longest(project_ids, project_names):
            if not project_id:
                continue
            row = _touch_history(
                history,
                str(project_id),
                project_name=str(project_name) if project_name else None,
                artifact=artifact,
                batch_id=batch_id,
            )
            row["requested_count"] += 1

        for result in payload.get("leaderboard") or []:
            project_id = _project_id_from_artifact_row(result, name_map, project_name_to_id)
            if not project_id:
                continue
            row = _touch_history(
                history,
                project_id,
                project_name=result.get("project"),
                artifact=artifact,
                batch_id=batch_id,
            )
            row["evaluated_count"] += 1
            if result.get("pass") is True:
                row["pass_count"] += 1
                row["last_result"] = "pass"
            else:
                row["miss_count"] += 1
                row["last_result"] = "miss"
            row["last_error_class"] = None

        for error in payload.get("errors") or []:
            project_id = _project_id_from_artifact_row(error, name_map, project_name_to_id)
            if not project_id:
                continue
            row = _touch_history(
                history,
                project_id,
                project_name=error.get("project"),
                artifact=artifact,
                batch_id=batch_id,
            )
            row["error_count"] += 1
            row["last_result"] = "error"
            row["last_error_class"] = error.get("error_class")

    return history


def resolve_artifact_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if not path.exists() or not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return sorted(paths, key=lambda path: (path.stat().st_mtime, str(path)))


def summarize(
    rows: List[Dict[str, Any]],
    *,
    backtest_history: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    classified = [
        classify_gold_project(row, backtest_history=backtest_history)
        for row in rows
    ]
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
        "backtest_ready_untested": sum(
            1 for row in classified if row["backtest_status"] == "ready_untested"
        ),
        "backtest_retry_after_quota": sum(
            1 for row in classified if row["backtest_status"] == "retry_after_quota"
        ),
        "backtest_retry_after_error": sum(
            1 for row in classified if row["backtest_status"] == "retry_after_error"
        ),
        "backtest_needs_accuracy_review": sum(
            1 for row in classified if row["backtest_status"] == "needs_accuracy_review"
        ),
        "backtest_validated_pass": sum(
            1 for row in classified if row["backtest_status"] == "validated_pass"
        ),
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
        "backtest_ready_untested",
        "backtest_retry_after_quota",
        "backtest_retry_after_error",
        "backtest_needs_accuracy_review",
        "backtest_validated_pass",
    ):
        print(f"  {key:34s} {summary[key]}")


def _print_queue_report(payload: Dict[str, Any]) -> None:
    print()
    print("Gold fill/backtest queue")
    queue_order = (
        ("retry_after_quota", "resume quota-blocked blind backtests"),
        ("retry_after_error", "retry blind backtests after non-quota errors"),
        ("needs_accuracy_review", "fix model/analog lessons from misses"),
        ("ready_untested", "strict blind backtest candidates"),
        ("needs_official_split_extraction", "extract official M&I/Inferred truth"),
        ("needs_model_run", "run blind model after accuracy gate"),
    )
    for status, label in queue_order:
        if status in {"needs_official_split_extraction", "needs_model_run"}:
            rows = [row for row in payload["projects"] if row["primary_status"] == status]
        else:
            rows = [row for row in payload["projects"] if row["backtest_status"] == status]
        print(f"  {status:32s} {len(rows):4d}  {label}")
        for row in rows[:10]:
            suffix = ""
            if row.get("backtest_last_batch_id"):
                suffix = f" | last_batch={row['backtest_last_batch_id']}"
            print(f"    - {row['project_id']} | {row['name']}{suffix}")


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
    parser.add_argument(
        "--backtest-artifacts",
        action="append",
        default=[],
        help="Leaderboard artifact path or glob. Repeatable.",
    )
    parser.add_argument(
        "--backtest-status",
        choices=(
            "ready_untested",
            "validated_pass",
            "needs_accuracy_review",
            "retry_after_quota",
            "retry_after_error",
            "requested_not_evaluated",
        ),
        default=None,
        help="Print only projects matching this backtest queue status.",
    )
    parser.add_argument(
        "--queue-report",
        action="store_true",
        help="Print the all-project fill/backtest queues.",
    )
    args = parser.parse_args()

    gold_projects = fetch_gold_projects()
    artifact_paths = resolve_artifact_paths(args.backtest_artifacts)
    backtest_history = (
        load_backtest_history(
            artifact_paths,
            project_name_to_id=project_name_to_id_map(gold_projects),
        )
        if artifact_paths
        else None
    )
    payload = summarize(gold_projects, backtest_history=backtest_history)
    if artifact_paths:
        payload["backtest_artifacts"] = [str(path) for path in artifact_paths]
        payload["backtest_history_project_count"] = len(backtest_history or {})
    _print_summary(payload)
    if artifact_paths:
        print(f"  backtest_artifacts_loaded          {len(artifact_paths)}")
        print(f"  backtest_history_projects         {len(backtest_history or {})}")
    if args.queue_report:
        _print_queue_report(payload)

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
    if args.backtest_status:
        projects = [row for row in payload["projects"] if row["backtest_status"] == args.backtest_status]
        if args.limit:
            projects = projects[: args.limit]
        print()
        print(f"{args.backtest_status}: {len(projects)} shown")
        for row in projects:
            print(
                f"{row['project_id']} | {row['name']} | "
                f"last={row['backtest_last_result']} "
                f"error={row['backtest_last_error_class']} "
                f"batch={row['backtest_last_batch_id']}"
            )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"json_out: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
