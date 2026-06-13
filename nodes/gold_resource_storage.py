"""Storage helpers for the gold-only predictor tables.

All functions target the new `gold_*` schema. They do not read or write legacy
`projects`, `analogs`, or `model_runs` rows.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from nodes.supabase_ops import get_client


GOLD_TABLES = {
    "projects": "gold_projects",
    "mre_truths": "gold_mre_truths",
    "pre_mre_evidence": "gold_pre_mre_evidence",
    "analog_candidates": "gold_analog_candidates",
    "analog_decisions": "gold_analog_decisions",
    "prediction_runs": "gold_prediction_runs",
    "prediction_scores": "gold_prediction_scores",
    "backtest_batches": "gold_backtest_batches",
    "parallel_cache": "gold_parallel_cache",
}


def _dump(row: Any) -> Dict[str, Any]:
    if hasattr(row, "model_dump"):
        return row.model_dump(mode="json", exclude_none=True)
    payload = {k: v for k, v in dict(row).items() if v is not None}
    return json.loads(json.dumps(payload, default=str))


def _first(data: Any) -> Dict[str, Any]:
    if isinstance(data, list):
        return data[0] if data else {}
    return data or {}


def upsert_gold_project(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = _dump(row)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    res = get_client().table(GOLD_TABLES["projects"]).upsert(payload).execute()
    return _first(res.data)


def insert_gold_mre_truth(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = _dump(row)
    payload.pop("cutoff_date", None)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    res = get_client().table(GOLD_TABLES["mre_truths"]).insert(payload).execute()
    return _first(res.data)


def upsert_gold_mre_truth(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = _dump(row)
    payload.pop("cutoff_date", None)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    existing_res = (
        get_client()
        .table(GOLD_TABLES["mre_truths"])
        .select("id")
        .eq("project_id", payload["project_id"])
        .eq("truth_status", payload.get("truth_status", "validated"))
        .maybe_single()
        .execute()
    )
    existing = existing_res.data if existing_res is not None else None
    if existing and existing.get("id"):
        res = (
            get_client()
            .table(GOLD_TABLES["mre_truths"])
            .update(payload)
            .eq("id", existing["id"])
            .execute()
        )
    else:
        res = get_client().table(GOLD_TABLES["mre_truths"]).insert(payload).execute()
    return _first(res.data)


def insert_gold_pre_mre_evidence(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = [_dump(row) for row in rows]
    if not payload:
        return []
    res = get_client().table(GOLD_TABLES["pre_mre_evidence"]).insert(payload).execute()
    return res.data or []


def upsert_gold_pre_mre_evidence(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = [_dump(row) for row in rows]
    if not payload:
        return []
    res = (
        get_client()
        .table(GOLD_TABLES["pre_mre_evidence"])
        .upsert(payload, on_conflict="id")
        .execute()
    )
    return res.data or []


def insert_gold_analog_candidates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = [_dump(row) for row in rows]
    if not payload:
        return []
    res = get_client().table(GOLD_TABLES["analog_candidates"]).insert(payload).execute()
    return res.data or []


def upsert_gold_analog_candidates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = [_dump(row) for row in rows]
    if not payload:
        return []
    res = (
        get_client()
        .table(GOLD_TABLES["analog_candidates"])
        .upsert(payload, on_conflict="id")
        .execute()
    )
    return res.data or []


def insert_gold_analog_decisions(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = [_dump(row) for row in rows]
    if not payload:
        return []
    res = get_client().table(GOLD_TABLES["analog_decisions"]).insert(payload).execute()
    return res.data or []


def upsert_gold_analog_decisions(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = [_dump(row) for row in rows]
    if not payload:
        return []
    res = (
        get_client()
        .table(GOLD_TABLES["analog_decisions"])
        .upsert(payload, on_conflict="id")
        .execute()
    )
    return res.data or []


def create_gold_backtest_batch(row: Dict[str, Any]) -> Dict[str, Any]:
    res = (
        get_client()
        .table(GOLD_TABLES["backtest_batches"])
        .upsert(_dump(row), on_conflict="run_label")
        .execute()
    )
    return _first(res.data)


def update_gold_backtest_batch(batch_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    res = (
        get_client()
        .table(GOLD_TABLES["backtest_batches"])
        .update(_dump(patch))
        .eq("id", batch_id)
        .execute()
    )
    return _first(res.data)


def insert_gold_prediction_run(row: Dict[str, Any]) -> Dict[str, Any]:
    res = get_client().table(GOLD_TABLES["prediction_runs"]).insert(_dump(row)).execute()
    return _first(res.data)


def upsert_gold_prediction_run(row: Dict[str, Any]) -> Dict[str, Any]:
    res = (
        get_client()
        .table(GOLD_TABLES["prediction_runs"])
        .upsert(_dump(row), on_conflict="project_id,run_mode,input_hash")
        .execute()
    )
    return _first(res.data)


def insert_gold_prediction_score(row: Dict[str, Any]) -> Dict[str, Any]:
    res = get_client().table(GOLD_TABLES["prediction_scores"]).insert(_dump(row)).execute()
    return _first(res.data)


def upsert_gold_prediction_score(row: Dict[str, Any]) -> Dict[str, Any]:
    res = (
        get_client()
        .table(GOLD_TABLES["prediction_scores"])
        .upsert(_dump(row), on_conflict="prediction_run_id")
        .execute()
    )
    return _first(res.data)


def get_parallel_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    res = (
        get_client()
        .table(GOLD_TABLES["parallel_cache"])
        .select("*")
        .eq("cache_key", cache_key)
        .maybe_single()
        .execute()
    )
    return res.data if res is not None else None


def upsert_parallel_cache(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = _dump(row)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    res = (
        get_client()
        .table(GOLD_TABLES["parallel_cache"])
        .upsert(payload, on_conflict="cache_key")
        .execute()
    )
    return _first(res.data)


def truth_cutoff_date(truth: Optional[Dict[str, Any]]) -> Optional[str]:
    if not truth:
        return None
    for key in ("cutoff_date", "effective_date", "publication_date"):
        value = truth.get(key)
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if value:
            return str(value)
    return None


def load_gold_case_bundle(project_id: str) -> Dict[str, Any]:
    client = get_client()
    project_res = (
        client.table(GOLD_TABLES["projects"])
        .select("*")
        .eq("id", project_id)
        .maybe_single()
        .execute()
    )
    project = project_res.data if project_res is not None else None
    truth_res = (
        client.table(GOLD_TABLES["mre_truths"])
        .select("*")
        .eq("project_id", project_id)
        .eq("truth_status", "validated")
        .maybe_single()
        .execute()
    )
    truth = truth_res.data if truth_res is not None else None
    cutoff_date = truth_cutoff_date(truth)

    all_evidence_query = (
        client.table(GOLD_TABLES["pre_mre_evidence"])
        .select("*")
        .eq("project_id", project_id)
    )
    if cutoff_date:
        all_evidence_query = all_evidence_query.eq("cutoff_date", cutoff_date)
    all_evidence = all_evidence_query.execute().data or []
    evidence = [
        row for row in all_evidence
        if row.get("evidence_status") == "accepted"
    ]
    rejected_evidence = [
        row for row in all_evidence
        if row.get("evidence_status") == "rejected"
    ]

    analogs = (
        client.table(GOLD_TABLES["analog_candidates"])
        .select("*")
        .eq("target_project_id", project_id)
        .execute()
        .data
        or []
    )
    decisions = (
        client.table(GOLD_TABLES["analog_decisions"])
        .select("*")
        .eq("target_project_id", project_id)
        .execute()
        .data
        or []
    )
    return {
        "project": project,
        "truth": truth,
        "evidence": evidence,
        "all_evidence": all_evidence,
        "rejected_evidence": rejected_evidence,
        "analog_candidates": analogs,
        "analog_decisions": decisions,
    }
