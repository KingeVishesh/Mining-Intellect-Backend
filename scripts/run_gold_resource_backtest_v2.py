#!/usr/bin/env python3
"""Populate and replay the DB-backed gold resource predictor v2 loop.

The runner is deliberately strict:

* gold projects only
* validated first-MRE truth only
* pre-MRE evidence only
* deterministic analog accept/reject decisions
* no numeric prediction when evidence is insufficient

Parallel research is optional and cached in ``gold_parallel_cache``. The default
path replays what is already in the database before spending on new research.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from config import settings  # noqa: E402
from nodes import geo_taxonomy, supabase_ops  # noqa: E402
from nodes.gold_resource_predictor import (  # noqa: E402
    PREDICTOR_VERSION,
    clean_analog_cohort,
    contained_gold_oz,
    evidence_grade_proxy,
    evidence_tonnage_proxy,
    predict_gold_resource,
    score_gold_prediction,
    split_evidence,
)
from nodes.gold_resource_storage import (  # noqa: E402
    GOLD_TABLES,
    create_gold_backtest_batch,
    get_parallel_cache,
    load_gold_case_bundle,
    update_gold_backtest_batch,
    upsert_gold_analog_candidates,
    upsert_gold_analog_decisions,
    upsert_gold_mre_truth,
    upsert_gold_prediction_run,
    upsert_gold_prediction_score,
    upsert_gold_pre_mre_evidence,
    upsert_gold_project,
    upsert_parallel_cache,
)
from nodes.parallel_gold_model import _run_parallel_task  # noqa: E402


LOGGER = logging.getLogger("gold_resource_backtest_v2")
DATE_RE = re.compile(r"\b(19|20)\d{2}(?:[-/](?:0?[1-9]|1[0-2])(?:[-/](?:0?[1-9]|[12]\d|3[01]))?)?\b")
UPDATED_MRE_RE = re.compile(r"\b(updated|update|supersedes|latest|revised|revision)\b", re.IGNORECASE)
POST_MRE_STUDY_RE = re.compile(r"\b(pea|pfs|pre[- ]?feasibility|feasibility|fs|mine[- ]?plan)\b", re.IGNORECASE)
UUID_NS = uuid.uuid5(uuid.NAMESPACE_URL, "mining-intellect-gold-resource-predictor-v2")
LEGACY_PROJECT_SELECT = (
    "id,name,company_name,material,country,region,district,location_name,"
    "latitude,longitude,deposit_type,deposit_subtype,tectonic_belt,"
    "mineralization_mode,mineralization_pattern,host_rock_class,"
    "mining_method_class,project_stage_class,recovery_method,"
    "mre_mi_tonnage_mt,mre_mi_grade,mre_mi_contained,"
    "mre_inferred_tonnage_mt,mre_inferred_grade,mre_inferred_contained,"
    "resource_compliance_standard,resource_vintage_year,drilling_evidence"
)


def stable_uuid(*parts: Any) -> str:
    blob = "|".join("" if part is None else str(part) for part in parts)
    return str(uuid.uuid5(UUID_NS, blob))


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def parse_loose_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, dict):
        for key in ("publication_date", "effective_date", "as_of_date", "source_date", "fetched_at"):
            parsed = parse_loose_date(value.get(key))
            if parsed:
                return parsed
        return None
    match = DATE_RE.search(str(value))
    if not match:
        return None
    parts = match.group(0).replace("/", "-").split("-")
    try:
        return date(
            int(parts[0]),
            int(parts[1]) if len(parts) > 1 else 1,
            int(parts[2]) if len(parts) > 2 else 1,
        )
    except ValueError:
        return None


def parse_year_end(value: Any) -> Optional[date]:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if 1900 <= year <= 2100:
        return date(year, 12, 31)
    return None


def positive_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def weighted_grade(
    mi_tonnage: Optional[float],
    mi_grade: Optional[float],
    inferred_tonnage: Optional[float],
    inferred_grade: Optional[float],
) -> Optional[float]:
    mi_t = positive_float(mi_tonnage) or 0.0
    inf_t = positive_float(inferred_tonnage) or 0.0
    total = mi_t + inf_t
    if total <= 0:
        return None
    return (mi_t * float(mi_grade or 0.0) + inf_t * float(inferred_grade or 0.0)) / total


def full_split_values(project: Dict[str, Any], run: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, float]]:
    source = run or project
    mapping = {
        "mi_tonnage_mt": source.get("mi_tonnage_mt", project.get("mre_mi_tonnage_mt")),
        "mi_grade_gpt": source.get("mi_grade", project.get("mre_mi_grade")),
        "inferred_tonnage_mt": source.get("inferred_tonnage_mt", project.get("mre_inferred_tonnage_mt")),
        "inferred_grade_gpt": source.get("inferred_grade", project.get("mre_inferred_grade")),
    }
    parsed = {key: positive_float(value) for key, value in mapping.items()}
    if any(value is None for value in parsed.values()):
        return None
    total_tonnage = parsed["mi_tonnage_mt"] + parsed["inferred_tonnage_mt"]
    total_grade = weighted_grade(
        parsed["mi_tonnage_mt"],
        parsed["mi_grade_gpt"],
        parsed["inferred_tonnage_mt"],
        parsed["inferred_grade_gpt"],
    )
    if total_grade is None:
        return None
    return {
        **parsed,
        "total_tonnage_mt": total_tonnage,
        "total_grade_gpt": total_grade,
        "mi_contained_oz": contained_gold_oz(parsed["mi_tonnage_mt"], parsed["mi_grade_gpt"]),
        "inferred_contained_oz": contained_gold_oz(parsed["inferred_tonnage_mt"], parsed["inferred_grade_gpt"]),
        "total_contained_oz": contained_gold_oz(total_tonnage, total_grade),
    }


def build_gold_project_row(project: Dict[str, Any], *, data_status: str = "candidate", exclusion_reason: Optional[str] = None) -> Dict[str, Any]:
    safe_payload = {
        "legacy_project_id": project.get("id"),
        "legacy_name": project.get("name"),
        "location_name": project.get("location_name"),
        "deposit_type": project.get("deposit_type"),
    }
    return {
        "id": project["id"],
        "external_key": f"legacy_project:{project['id']}",
        "company_name": project.get("company_name"),
        "project_name": project.get("name") or "Unknown gold project",
        "material": "gold",
        "country": project.get("country"),
        "region": project.get("region"),
        "district": project.get("district"),
        "latitude": project.get("latitude"),
        "longitude": project.get("longitude"),
        "deposit_family": project.get("deposit_type"),
        "deposit_subtype": project.get("deposit_subtype"),
        "tectonic_belt": project.get("tectonic_belt") or geo_taxonomy.detect_belt_from_row(project),
        "mineralization_mode": project.get("mineralization_mode"),
        "mineralization_pattern": project.get("mineralization_pattern"),
        "host_rock_class": project.get("host_rock_class"),
        "mining_method_class": project.get("mining_method_class"),
        "project_stage_class": project.get("project_stage_class"),
        "recovery_method": project.get("recovery_method"),
        "data_status": data_status,
        "exclusion_reason": exclusion_reason,
        "source_payload": safe_payload,
    }


def ensure_project_for_parallel_cache(project: Dict[str, Any], *, data_status: str) -> None:
    upsert_gold_project(build_gold_project_row(project, data_status=data_status))


def truth_run_rejection_reasons(run: Dict[str, Any], effective: Optional[date]) -> List[str]:
    reasons: List[str] = []
    source_url = str(run.get("source_url") or "").strip()
    source_text = " ".join(
        str(value or "")
        for value in (
            run.get("source_url"),
            run.get("source"),
            run.get("notes"),
        )
    )
    if not source_url:
        reasons.append("missing_mre_source_url")
    if effective is None:
        reasons.append("missing_mre_effective_date")
    elif effective.month == 1 and effective.day == 1:
        reasons.append("year_start_placeholder_mre_date")
    elif effective.month == 12 and effective.day == 31:
        reasons.append("year_end_placeholder_mre_date")
    if UPDATED_MRE_RE.search(source_text):
        reasons.append("non_first_or_updated_mre_source")
    if POST_MRE_STUDY_RE.search(source_text) and not re.search(r"\bmre\b|\bmineral resource\b", source_text, re.IGNORECASE):
        reasons.append("post_mre_study_source")
    return reasons


def build_truth_row(project: Dict[str, Any], runs: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    candidates: List[Tuple[date, Dict[str, Any], Dict[str, float]]] = []
    rejected_reasons: Counter[str] = Counter()
    for run in runs:
        effective = parse_loose_date(run.get("effective_date"))
        reasons = truth_run_rejection_reasons(run, effective)
        if reasons:
            rejected_reasons.update(reasons)
            continue
        values = full_split_values(project, run)
        if values:
            candidates.append((effective, run, values))
        else:
            rejected_reasons["missing_full_mre_split"] += 1

    if not candidates:
        if rejected_reasons:
            detail = ",".join(f"{reason}:{count}" for reason, count in sorted(rejected_reasons.items()))
            return None, f"no_validated_first_mre_with_full_split_and_date:{detail}"
        return None, "no_validated_first_mre_with_full_split_and_date"

    effective, run, values = sorted(candidates, key=lambda item: item[0])[0]
    source_url = run.get("source_url") or f"legacy:mre_runs:{run.get('id') or project['id']}"
    return {
        "project_id": project["id"],
        "truth_status": "validated",
        "effective_date": effective,
        "publication_date": effective,
        "source_url": source_url,
        "source_title": f"First validated MRE for {project.get('name')}",
        "source_publisher": run.get("source"),
        "source_document_type": "mre_truth",
        "resource_standard": project.get("resource_compliance_standard"),
        "mi_tonnage_mt": values["mi_tonnage_mt"],
        "mi_grade_gpt": values["mi_grade_gpt"],
        "inferred_tonnage_mt": values["inferred_tonnage_mt"],
        "inferred_grade_gpt": values["inferred_grade_gpt"],
        "total_tonnage_mt": values["total_tonnage_mt"],
        "total_grade_gpt": values["total_grade_gpt"],
        "mi_contained_oz": values["mi_contained_oz"],
        "inferred_contained_oz": values["inferred_contained_oz"],
        "total_contained_oz": values["total_contained_oz"],
        "validation_notes": run.get("notes"),
        "raw_parallel_output": {"source": "legacy_mre_runs", "legacy_mre_run": json_safe(run)},
    }, None


def parallel_truth_prompt(project: Dict[str, Any], legacy_runs: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    safe_project = {
        key: project.get(key)
        for key in (
            "name", "company_name", "country", "region", "district",
            "deposit_type", "deposit_subtype", "tectonic_belt",
            "resource_compliance_standard",
        )
    }
    weak_legacy_runs = [
        {
            "effective_date": run.get("effective_date"),
            "source": run.get("source"),
            "source_url": run.get("source_url"),
            "notes": run.get("notes"),
        }
        for run in legacy_runs[:5]
    ]
    prompt = f"""
You are a mining MRE truth auditor.

TARGET GOLD PROJECT:
{json.dumps(safe_project, indent=2, sort_keys=True, default=str)}

WEAK LEGACY MRE ROWS TO AUDIT, NOT TRUST:
{json.dumps(weak_legacy_runs, indent=2, sort_keys=True, default=str)}

Find the first publicly disclosed mineral resource estimate (MRE) for the
target project. It must be gold-focused and must contain M&I and Inferred
tonnage and grade split values. Do not return an updated/revised/latest MRE,
PEA/PFS/FS/Mine Plan resource table, or a source that only references a prior
MRE. If the first MRE cannot be validated, return status "no_validated_first_mre".
Include rejected sources you inspected and why they were rejected.
""".strip()
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["validated", "no_validated_first_mre"]},
            "project_name": {"type": ["string", "null"]},
            "effective_date": {"type": ["string", "null"]},
            "publication_date": {"type": ["string", "null"]},
            "source_url": {"type": ["string", "null"]},
            "source_title": {"type": ["string", "null"]},
            "source_publisher": {"type": ["string", "null"]},
            "resource_standard": {"type": ["string", "null"]},
            "mi_tonnage_mt": {"type": ["number", "null"]},
            "mi_grade_gpt": {"type": ["number", "null"]},
            "inferred_tonnage_mt": {"type": ["number", "null"]},
            "inferred_grade_gpt": {"type": ["number", "null"]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "validation_notes": {"type": ["string", "null"]},
            "rejected_sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_url": {"type": ["string", "null"]},
                        "source_date": {"type": ["string", "null"]},
                        "source_title": {"type": ["string", "null"]},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
        "required": ["status", "source_url", "publication_date", "mi_tonnage_mt", "mi_grade_gpt", "inferred_tonnage_mt", "inferred_grade_gpt", "confidence"],
    }
    return prompt, schema


def truth_row_from_parallel(project: Dict[str, Any], response: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(response, dict):
        return None, "parallel_truth_invalid_response"
    if response.get("status") != "validated":
        return None, str(response.get("status") or "parallel_truth_not_validated")
    if normalize_evidence_confidence(response.get("confidence")) == "low":
        return None, "parallel_truth_low_confidence"

    effective = parse_loose_date(response.get("effective_date")) or parse_loose_date(response.get("publication_date"))
    publication = parse_loose_date(response.get("publication_date")) or effective
    synthetic_run = {
        "id": response.get("parallel_task_id"),
        "effective_date": effective,
        "source": response.get("source_publisher") or "parallel_mre_truth",
        "source_url": response.get("source_url"),
        "notes": response.get("validation_notes"),
        "mi_tonnage_mt": response.get("mi_tonnage_mt"),
        "mi_grade": response.get("mi_grade_gpt"),
        "inferred_tonnage_mt": response.get("inferred_tonnage_mt"),
        "inferred_grade": response.get("inferred_grade_gpt"),
    }
    reasons = truth_run_rejection_reasons(synthetic_run, effective)
    if reasons:
        return None, "parallel_truth_rejected:" + ",".join(sorted(set(reasons)))
    values = full_split_values(project, synthetic_run)
    if not values:
        return None, "parallel_truth_missing_full_mre_split"
    if publication is None:
        return None, "parallel_truth_missing_publication_date"

    return {
        "project_id": project["id"],
        "truth_status": "validated",
        "effective_date": effective,
        "publication_date": publication,
        "source_url": str(response["source_url"]),
        "source_title": response.get("source_title") or f"First validated MRE for {project.get('name')}",
        "source_publisher": response.get("source_publisher") or "parallel_mre_truth",
        "source_document_type": "mre_truth",
        "resource_standard": response.get("resource_standard") or project.get("resource_compliance_standard"),
        "mi_tonnage_mt": values["mi_tonnage_mt"],
        "mi_grade_gpt": values["mi_grade_gpt"],
        "inferred_tonnage_mt": values["inferred_tonnage_mt"],
        "inferred_grade_gpt": values["inferred_grade_gpt"],
        "total_tonnage_mt": values["total_tonnage_mt"],
        "total_grade_gpt": values["total_grade_gpt"],
        "mi_contained_oz": values["mi_contained_oz"],
        "inferred_contained_oz": values["inferred_contained_oz"],
        "total_contained_oz": values["total_contained_oz"],
        "validation_notes": response.get("validation_notes"),
        "raw_parallel_output": {"source": "parallel_mre_truth", "response": json_safe(response)},
    }, None


def latest_intercept_source_date(evidence: Dict[str, Any]) -> Optional[date]:
    dates = []
    for item in evidence.get("best_intercepts") or []:
        if isinstance(item, dict):
            parsed = parse_loose_date(item.get("source_date") or item.get("source_url"))
            if parsed:
                dates.append(parsed)
    return max(dates) if dates else None


def evidence_source_date(evidence: Dict[str, Any]) -> Optional[date]:
    return parse_loose_date(evidence.get("source_date")) or latest_intercept_source_date(evidence)


def evidence_source_url(evidence: Dict[str, Any], project_id: str) -> str:
    if evidence.get("source_url"):
        return str(evidence["source_url"])
    for item in evidence.get("best_intercepts") or []:
        if isinstance(item, dict) and item.get("source_url"):
            return str(item["source_url"])
    return f"legacy:projects.drilling_evidence:{project_id}"


def normalize_evidence_confidence(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
    elif isinstance(value, dict):
        normalized = str(value.get("level") or value.get("confidence") or value.get("rating") or "").strip().lower()
    else:
        normalized = ""
    return normalized if normalized in {"high", "medium", "low"} else "medium"


def evidence_fact_row(
    *,
    project_id: str,
    truth_id: Optional[str],
    cutoff_date: date,
    evidence: Dict[str, Any],
    fact_type: str,
    value_num: Optional[float] = None,
    value_text: Optional[str] = None,
    unit: Optional[str] = None,
    raw_parallel_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_url = evidence_source_url(evidence, project_id)
    parsed_source_date = evidence_source_date(evidence)
    source_title = evidence.get("source_title") or evidence.get("report_title")
    confidence = normalize_evidence_confidence(evidence.get("confidence"))
    row = {
        "id": stable_uuid("evidence", project_id, truth_id, cutoff_date.isoformat(), fact_type, source_url, parsed_source_date, value_num, value_text),
        "project_id": project_id,
        "mre_truth_id": truth_id,
        "cutoff_date": cutoff_date,
        "source_url": source_url,
        "source_title": source_title,
        "source_date": parsed_source_date,
        "source_document_type": evidence.get("source") or evidence.get("source_document_type"),
        "evidence_status": "accepted",
        "fact_type": fact_type,
        "value_num": value_num,
        "value_text": value_text,
        "unit": unit,
        "confidence": confidence,
        "is_mre_tainted": False,
        "fact_payload": json_safe(evidence),
        "raw_parallel_output": json_safe(raw_parallel_output or {}),
    }
    from nodes.gold_resource_predictor import validate_pre_mre_evidence

    ok, reasons = validate_pre_mre_evidence(row)
    if not ok:
        row["evidence_status"] = "rejected"
        row["rejection_reason"] = ";".join(reasons)
        # The live schema's pre-cutoff check still applies to rejected rows.
        # Preserve the actual rejected date in the payload while keeping the row storable.
        if parsed_source_date and parsed_source_date >= cutoff_date:
            row["fact_payload"] = {**row["fact_payload"], "rejected_source_date": parsed_source_date.isoformat()}
            row["source_date"] = None
    return row


def evidence_rows_from_payload(
    *,
    project_id: str,
    truth_id: Optional[str],
    cutoff_date: date,
    evidence: Optional[Dict[str, Any]],
    raw_parallel_output: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(evidence, dict) or not evidence:
        return []

    specs = (
        ("total_drill_meters", evidence.get("total_meters_drilled"), "m"),
        ("drill_holes", evidence.get("total_holes"), "holes"),
        ("weighted_grade_gpt", evidence.get("weighted_grade_g_t"), "g/t"),
        ("grade_proxy_gpt", evidence.get("grade_proxy_g_t"), "g/t"),
        ("average_intercept_grade_gpt", evidence.get("average_intercept_grade_g_t"), "g/t"),
        ("geometry_tonnage_mt", evidence.get("geometry_tonnage_mt"), "Mt"),
        ("tailings_inventory_tonnage_mt", evidence.get("tailings_inventory_tonnage_mt"), "Mt"),
        ("strike_length_m", evidence.get("strike_length_m"), "m"),
        ("down_dip_extent_m", evidence.get("down_dip_extent_m"), "m"),
        ("avg_true_width_m", evidence.get("avg_true_width_m"), "m"),
        ("bulk_density_t_m3", evidence.get("bulk_density_t_m3"), "t/m3"),
        ("mineralized_continuity_factor", evidence.get("mineralized_continuity_factor"), "ratio"),
    )
    rows = [
        evidence_fact_row(
            project_id=project_id,
            truth_id=truth_id,
            cutoff_date=cutoff_date,
            evidence=evidence,
            fact_type=fact_type,
            value_num=positive_float(value),
            unit=unit,
            raw_parallel_output=raw_parallel_output,
        )
        for fact_type, value, unit in specs
        if positive_float(value) is not None
    ]
    if evidence.get("best_intercepts"):
        rows.append(
            evidence_fact_row(
                project_id=project_id,
                truth_id=truth_id,
                cutoff_date=cutoff_date,
                evidence=evidence,
                fact_type="best_intercepts",
                value_text=json.dumps(evidence.get("best_intercepts"), sort_keys=True, default=str),
                raw_parallel_output=raw_parallel_output,
            )
        )
    if not rows:
        rows.append(
            evidence_fact_row(
                project_id=project_id,
                truth_id=truth_id,
                cutoff_date=cutoff_date,
                evidence=evidence,
                fact_type="unusable_evidence_payload",
                value_text=evidence.get("notes") or "No numeric pre-MRE model facts extracted.",
                raw_parallel_output=raw_parallel_output,
            )
        )
    return rows


def analog_source_date(analog: Dict[str, Any]) -> Optional[date]:
    return (
        parse_loose_date(analog.get("source_date"))
        or parse_year_end(analog.get("resource_vintage_year") or analog.get("analog_resource_vintage_year"))
        or parse_loose_date(analog.get("source_url"))
    )


def analog_name(analog: Dict[str, Any]) -> str:
    return str(analog.get("name") or analog.get("analog_name") or analog.get("candidate_project_name") or "").strip()


def analog_stage_class(analog: Dict[str, Any]) -> Optional[str]:
    value = analog.get("project_stage_class") or analog.get("analog_project_stage_class")
    if value in geo_taxonomy.ALL_STAGE_SLUGS:
        return value
    return geo_taxonomy.detect_stage_class(
        project_stage=str(value or ""),
        description=str(analog.get("notes") or ""),
    )


def analog_candidate_row(project_id: str, analog: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = analog_name(analog)
    if not name:
        return None
    total_tonnage = positive_float(analog.get("tonnage_mt") or analog.get("analog_tonnage_mt") or analog.get("total_tonnage_mt"))
    total_grade = positive_float(analog.get("grade_value") or analog.get("analog_grade_value") or analog.get("total_grade_gpt"))
    inferred_tonnage = positive_float(analog.get("inferred_tonnage_mt") or analog.get("analog_inferred_tonnage_mt"))
    inferred_grade = positive_float(analog.get("inferred_grade") or analog.get("analog_inferred_grade"))
    mi_tonnage = positive_float(analog.get("mi_tonnage_mt"))
    mi_grade = positive_float(analog.get("mi_grade") or analog.get("mi_grade_gpt"))
    if not mi_tonnage and total_tonnage and inferred_tonnage and total_tonnage > inferred_tonnage:
        mi_tonnage = total_tonnage - inferred_tonnage
    if not mi_grade and total_tonnage and total_grade and inferred_tonnage and inferred_grade and mi_tonnage:
        mi_grade = ((total_tonnage * total_grade) - (inferred_tonnage * inferred_grade)) / mi_tonnage
        if mi_grade <= 0:
            mi_grade = None
    source_url = str(analog.get("source_url") or f"legacy:analogs:{name}")
    return {
        "id": stable_uuid("analog", project_id, name.lower(), source_url),
        "target_project_id": project_id,
        "candidate_project_name": name,
        "candidate_company_name": analog.get("company_name") or analog.get("analog_company_name"),
        "candidate_country": analog.get("country") or analog.get("analog_country"),
        "candidate_region": analog.get("region") or analog.get("analog_region"),
        "candidate_district": analog.get("district") or analog.get("analog_district"),
        "candidate_deposit_family": analog.get("deposit_type") or analog.get("analog_deposit_type"),
        "candidate_deposit_subtype": analog.get("deposit_subtype") or analog.get("analog_deposit_subtype"),
        "candidate_tectonic_belt": analog.get("tectonic_belt") or analog.get("analog_tectonic_belt"),
        "candidate_mineralization_mode": analog.get("mineralization_mode") or analog.get("analog_mineralization_mode"),
        "candidate_mineralization_pattern": analog.get("mineralization_pattern") or analog.get("analog_mineralization_pattern"),
        "candidate_host_rock_class": analog.get("host_rock_class") or analog.get("analog_host_rock_class"),
        "candidate_mining_method_class": analog.get("mining_method_class") or analog.get("analog_mining_method_class"),
        "candidate_project_stage_class": analog_stage_class(analog),
        "candidate_recovery_method": analog.get("recovery_method") or analog.get("analog_recovery_method"),
        "source_url": source_url,
        "source_date": analog_source_date(analog),
        "source_title": analog.get("source_title"),
        "resource_standard": analog.get("resource_compliance_standard") or analog.get("analog_resource_compliance_standard"),
        "total_tonnage_mt": total_tonnage,
        "total_grade_gpt": total_grade,
        "total_contained_oz": contained_gold_oz(total_tonnage, total_grade),
        "mi_tonnage_mt": mi_tonnage,
        "mi_grade_gpt": mi_grade,
        "inferred_tonnage_mt": inferred_tonnage,
        "inferred_grade_gpt": inferred_grade,
        "drill_meters": positive_float(analog.get("total_meters_drilled") or analog.get("drill_meters")),
        "drill_holes": int(analog["drill_holes"]) if str(analog.get("drill_holes") or "").isdigit() else None,
        "best_intercepts": analog.get("best_intercepts") or [],
        "geometry_payload": {
            key: analog.get(key)
            for key in ("strike_length_m", "down_dip_extent_m", "avg_true_width_m", "drilled_area_km2")
            if analog.get(key) is not None
        },
        "raw_parallel_output": json_safe(analog.get("raw_parallel_output") or {}),
    }


def merge_analogs(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        name = analog_name(row).lower()
        if not name or name in seen:
            continue
        seen.add(name)
        merged.append(row)
    return merged


def load_legacy_analogs(project: Dict[str, Any], *, limit: int = 80) -> List[Dict[str, Any]]:
    project_id = project["id"]
    rows = list(supabase_ops.get_analogs(project_id) or [])
    deposit_subtype = project.get("deposit_subtype")
    deposit_type = project.get("deposit_type")
    target_belt = project.get("tectonic_belt") or geo_taxonomy.detect_belt_from_row(project)
    if deposit_subtype or deposit_type:
        try:
            rows.extend(
                supabase_ops.get_approved_analogs(
                    material="gold",
                    deposit_type=deposit_type,
                    deposit_subtype=deposit_subtype,
                    target_tectonic_belt=target_belt,
                    limit=limit,
                )
            )
        except Exception:
            LOGGER.exception("failed to load approved analog library for %s", project.get("name"))
    return merge_analogs(rows)


def decision_rows_for_candidates(
    project_row: Dict[str, Any],
    evidence_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    *,
    cutoff_date: date,
) -> List[Dict[str, Any]]:
    accepted_evidence, _ = split_evidence(row for row in evidence_rows if row.get("evidence_status") == "accepted")
    tonnage, tonnage_trace = evidence_tonnage_proxy(accepted_evidence)
    grade, grade_trace = evidence_grade_proxy(accepted_evidence)
    decisions: List[Dict[str, Any]] = []
    if tonnage is None or grade is None:
        reasons = []
        if tonnage is None:
            reasons.append("target_missing_pre_mre_tonnage_proxy")
            reasons.extend(str(reason) for reason in tonnage_trace.get("quality_reasons") or [] if reason)
        if grade is None:
            reasons.append("target_missing_pre_mre_grade_proxy")
            reasons.extend(str(reason) for reason in grade_trace.get("quality_reasons") or [] if reason)
        reasons = sorted(set(reasons))
        for candidate in candidate_rows:
            decisions.append({
                "id": stable_uuid("analog_decision", candidate["id"]),
                "target_project_id": project_row["id"],
                "analog_candidate_id": candidate["id"],
                "decision": "rejected",
                "decision_rules": [],
                "rejection_reasons": reasons,
            })
        return decisions

    _accepted, raw_decisions = clean_analog_cohort(
        project_row,
        candidate_rows,
        cutoff_date=cutoff_date,
        target_tonnage_mt=tonnage,
        target_grade_gpt=grade,
    )
    for decision in raw_decisions:
        candidate_id = decision.get("analog_candidate_id")
        decisions.append({
            "id": stable_uuid("analog_decision", candidate_id),
            "target_project_id": project_row["id"],
            "analog_candidate_id": candidate_id,
            "decision": decision["decision"],
            "decision_rules": decision.get("decision_rules") or [],
            "rejection_reasons": decision.get("rejection_reasons") or [],
            "accepted_at": datetime.now(timezone.utc).isoformat() if decision["decision"] == "accepted" else None,
        })
    return decisions


def cache_key_for(task_kind: str, request_payload: Dict[str, Any]) -> str:
    payload = json.dumps(request_payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(f"{task_kind}:{payload}".encode("utf-8")).hexdigest()


def comparable_parallel_request_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in (payload or {}).items()
        if key != "predictor_version"
    }


def find_reusable_parallel_cache(
    *,
    task_kind: str,
    project_id: str,
    cutoff_date: Optional[date],
    request_payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    target_payload = comparable_parallel_request_payload(request_payload)
    query = (
        supabase_ops.get_client()
        .table(GOLD_TABLES["parallel_cache"])
        .select("*")
        .eq("task_kind", task_kind)
        .eq("project_id", project_id)
        .eq("response_status", "complete")
        .order("created_at", desc=True)
        .limit(20)
    )
    if cutoff_date is None:
        query = query.is_("cutoff_date", "null")
    else:
        query = query.eq("cutoff_date", cutoff_date.isoformat())
    rows = query.execute().data or []
    for row in rows:
        if comparable_parallel_request_payload(row.get("request_payload") or {}) == target_payload:
            return row
    return None


def provider_task_id_from_error(error: str) -> Optional[str]:
    match = re.search(r"\brun_id=([A-Za-z0-9_-]+)", error or "")
    return match.group(1) if match else None


def run_parallel_cached(
    *,
    task_kind: str,
    project_id: str,
    cutoff_date: Optional[date],
    prompt: str,
    output_schema: Dict[str, Any],
    save: bool,
    allow_paid: bool = True,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    request_payload = {
        "prompt": prompt,
        "output_schema": output_schema,
        "processor": settings.parallel_processor,
        "predictor_version": PREDICTOR_VERSION,
    }
    key = cache_key_for(task_kind, request_payload)
    cached = get_parallel_cache(key)
    if cached and cached.get("response_status") == "complete":
        return cached.get("response_payload"), False
    reusable = find_reusable_parallel_cache(
        task_kind=task_kind,
        project_id=project_id,
        cutoff_date=cutoff_date,
        request_payload=request_payload,
    )
    if reusable:
        return reusable.get("response_payload"), False
    if not allow_paid:
        return None, False
    try:
        response = _run_parallel_task(prompt=prompt, output_schema=output_schema)
    except Exception as exc:
        provider_error = str(exc)
        if save:
            upsert_parallel_cache({
                "task_kind": task_kind,
                "cache_key": key,
                "project_id": project_id,
                "cutoff_date": cutoff_date,
                "request_payload": request_payload,
                "response_payload": {},
                "response_status": "failed",
                "provider_error": provider_error,
                "provider_task_id": provider_task_id_from_error(provider_error),
            })
        return None, True
    if save and response is not None:
        upsert_parallel_cache({
            "task_kind": task_kind,
            "cache_key": key,
            "project_id": project_id,
            "cutoff_date": cutoff_date,
            "request_payload": request_payload,
            "response_payload": response,
            "response_status": "complete",
        })
    return response, True


def parallel_evidence_prompt(project: Dict[str, Any], cutoff_date: date) -> Tuple[str, Dict[str, Any]]:
    safe_project = {
        key: project.get(key)
        for key in (
            "name", "company_name", "country", "region", "district",
            "deposit_type", "deposit_subtype", "tectonic_belt",
            "mining_method_class", "project_stage_class", "recovery_method",
        )
    }
    prompt = f"""
You are a mining backtest evidence auditor.

TARGET GOLD PROJECT:
{json.dumps(safe_project, indent=2, sort_keys=True, default=str)}

HARD CUTOFF: use only public information published before {cutoff_date.isoformat()}.

Find pre-MRE evidence only. Do not use or quote target MRE/resource tonnes,
grade, ounces, categories, resource tables, PEA/PFS/FS resource summaries, or
technical reports dated on or after the cutoff. Return null for unavailable
facts. If a pre-MRE source gives an explicit exploration geometry tonnage or
inventory tonnage estimate that is not an MRE/resource estimate, put it in
geometry_tonnage_mt and explain the basis in notes. If that same non-MRE source
gives an explicit grade range or grade proxy, return its midpoint in
grade_proxy_g_t and explain the basis in notes. Include rejected sources you
inspected and why they were rejected.
""".strip()
    schema = {
        "type": "object",
        "properties": {
            "total_holes": {"type": ["integer", "null"]},
            "total_meters_drilled": {"type": ["number", "null"]},
            "weighted_grade_g_t": {"type": ["number", "null"]},
            "grade_proxy_g_t": {"type": ["number", "null"]},
            "average_intercept_grade_g_t": {"type": ["number", "null"]},
            "geometry_tonnage_mt": {"type": ["number", "null"]},
            "tailings_inventory_tonnage_mt": {"type": ["number", "null"]},
            "strike_length_m": {"type": ["number", "null"]},
            "down_dip_extent_m": {"type": ["number", "null"]},
            "avg_true_width_m": {"type": ["number", "null"]},
            "bulk_density_t_m3": {"type": ["number", "null"]},
            "mineralized_continuity_factor": {"type": ["number", "null"]},
            "source_url": {"type": ["string", "null"]},
            "source_date": {"type": ["string", "null"]},
            "source_title": {"type": ["string", "null"]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "notes": {"type": ["string", "null"]},
            "rejected_sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_url": {"type": ["string", "null"]},
                        "source_date": {"type": ["string", "null"]},
                        "source_title": {"type": ["string", "null"]},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
        "required": ["total_holes", "total_meters_drilled", "weighted_grade_g_t", "confidence"],
    }
    return prompt, schema


def parallel_analog_prompt(
    project: Dict[str, Any],
    cutoff_date: date,
    *,
    target_tonnage_mt: float,
    target_grade_gpt: float,
) -> Tuple[str, Dict[str, Any]]:
    safe_project = {
        key: project.get(key)
        for key in (
            "name", "company_name", "country", "region", "district",
            "deposit_type", "deposit_subtype", "tectonic_belt",
            "host_rock_class", "mining_method_class", "project_stage_class",
            "recovery_method",
        )
    }
    prompt = f"""
You are a mining analog research auditor for a strict blind gold backtest.

TARGET GOLD PROJECT:
{json.dumps(safe_project, indent=2, sort_keys=True, default=str)}

BLIND TARGET PRE-MRE SCALE EVIDENCE:
- tonnage proxy: {target_tonnage_mt} Mt
- grade proxy: {target_grade_gpt} g/t Au

HARD CUTOFF: use only public analog information published before {cutoff_date.isoformat()}.

Find gold analog candidates that can be deterministically gated later. Do not use
target MRE/resource information. Prefer analogs with source URLs, source dates,
resource standard, M&I and inferred split tonnes/grades, deposit subtype, tectonic
belt, mining method class, project stage class, and comparable tonnage/grade band.
Return only analogs with enough cited data to audit; omit weak candidates rather
than filling fields from memory.
""".strip()
    analog_schema = {
        "type": "object",
        "properties": {
            "analog_name": {"type": "string"},
            "country": {"type": ["string", "null"]},
            "deposit_subtype": {"type": ["string", "null"]},
            "tectonic_belt": {"type": ["string", "null"]},
            "mining_method_class": {"type": ["string", "null"]},
            "project_stage_class": {"type": ["string", "null"]},
            "source_url": {"type": "string"},
            "source_date": {"type": "string"},
            "resource_compliance_standard": {"type": "string"},
            "total_tonnage_mt": {"type": "number"},
            "total_grade_gpt": {"type": "number"},
            "mi_tonnage_mt": {"type": ["number", "null"]},
            "mi_grade_gpt": {"type": ["number", "null"]},
            "inferred_tonnage_mt": {"type": ["number", "null"]},
            "inferred_grade": {"type": ["number", "null"]},
            "notes": {"type": ["string", "null"]},
        },
        "required": [
            "analog_name", "source_url", "source_date",
            "resource_compliance_standard", "total_tonnage_mt", "total_grade_gpt",
        ],
    }
    schema = {
        "type": "object",
        "properties": {
            "analogs": {
                "type": "array",
                "items": analog_schema,
            },
            "notes": {"type": ["string", "null"]},
        },
        "required": ["analogs"],
    }
    return prompt, schema


def fetch_legacy_truth_projects(limit: Optional[int] = None, project_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    client = supabase_ops.get_client()
    while True:
        query = (
            client.table("projects")
            .select(LEGACY_PROJECT_SELECT)
            .ilike("material", "gold")
            .not_.is_("mre_mi_tonnage_mt", "null")
            .not_.is_("mre_mi_grade", "null")
            .not_.is_("mre_inferred_tonnage_mt", "null")
            .not_.is_("mre_inferred_grade", "null")
        )
        if project_ids:
            query = query.in_("id", list(project_ids))
        upper = offset + 999
        if limit is not None:
            upper = min(upper, max(0, limit - len(rows)) + offset - 1)
        if upper < offset:
            break
        res = query.order("name").range(offset, upper).execute()
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < (upper - offset + 1):
            break
        if limit is not None and len(rows) >= limit:
            break
        offset += 1000
    return rows[:limit] if limit is not None else rows


def fetch_mre_runs(project_ids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    by_project: Dict[str, List[Dict[str, Any]]] = {project_id: [] for project_id in project_ids}
    client = supabase_ops.get_client()
    for idx in range(0, len(project_ids), 100):
        batch = list(project_ids[idx : idx + 100])
        if not batch:
            continue
        rows = client.table("mre_runs").select("*").in_("project_id", batch).execute().data or []
        for row in rows:
            by_project.setdefault(row["project_id"], []).append(row)
    return by_project


def fetch_validated_gold_truths(project_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    by_project: Dict[str, Dict[str, Any]] = {}
    client = supabase_ops.get_client()
    for idx in range(0, len(project_ids), 100):
        batch = list(project_ids[idx : idx + 100])
        if not batch:
            continue
        rows = (
            client.table(GOLD_TABLES["mre_truths"])
            .select("*")
            .in_("project_id", batch)
            .eq("truth_status", "validated")
            .execute()
            .data
            or []
        )
        for row in rows:
            project_id = row.get("project_id")
            if project_id and project_id not in by_project:
                by_project[project_id] = row
    return by_project


def gold_table_counts() -> Dict[str, int]:
    client = supabase_ops.get_client()
    counts: Dict[str, int] = {}
    for table in GOLD_TABLES.values():
        res = client.table(table).select("id", count="exact").limit(1).execute()
        counts[table] = int(res.count or 0)
    return counts


def prediction_run_row(
    *,
    project_id: str,
    truth_id: str,
    batch_id: Optional[str],
    prediction: Dict[str, Any],
    decision_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    accepted_decisions = [
        decision for decision in prediction.get("analog_decisions") or []
        if decision.get("decision") == "accepted"
    ]
    accepted_candidate_ids = [
        decision.get("analog_candidate_id")
        for decision in accepted_decisions
        if decision.get("analog_candidate_id")
    ]
    return {
        "project_id": project_id,
        "mre_truth_id": truth_id,
        "backtest_batch_id": batch_id,
        "run_mode": "blind_no_mre",
        "run_status": prediction["run_status"],
        "input_hash": prediction["input_hash"],
        "cutoff_date": prediction["cutoff_date"],
        "evidence_fact_ids": [
            item.get("id")
            for item in prediction.get("calculator_trace", {}).get("accepted_evidence", [])
            if item.get("id")
        ],
        "analog_candidate_ids": accepted_candidate_ids,
        "analog_decision_ids": [row["id"] for row in decision_rows if row.get("id")],
        "no_prediction_reasons": prediction.get("no_prediction_reasons") or [],
        "predicted_total_tonnage_mt": prediction.get("predicted_total_tonnage_mt"),
        "predicted_total_grade_gpt": prediction.get("predicted_total_grade_gpt"),
        "predicted_total_contained_oz": prediction.get("predicted_total_contained_oz"),
        "predicted_mi_tonnage_mt": prediction.get("predicted_mi_tonnage_mt"),
        "predicted_mi_grade_gpt": prediction.get("predicted_mi_grade_gpt"),
        "predicted_inferred_tonnage_mt": prediction.get("predicted_inferred_tonnage_mt"),
        "predicted_inferred_grade_gpt": prediction.get("predicted_inferred_grade_gpt"),
        "predictor_version": prediction["predictor_version"],
        "calculator_trace": prediction.get("calculator_trace") or {},
    }


def score_row(run_id: str, truth_id: str, score: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction_run_id": run_id,
        "mre_truth_id": truth_id,
        **score,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.processor:
        settings.parallel_processor = args.processor
    if args.poll_timeout_s is not None:
        import nodes.parallel_gold_model as parallel_gold_model

        parallel_gold_model._POLL_TIMEOUT_S = max(15, int(args.poll_timeout_s))

    before_counts = gold_table_counts()
    project_ids = args.project_id or None
    legacy_projects = fetch_legacy_truth_projects(limit=args.limit, project_ids=project_ids)
    validated_gold_truths = fetch_validated_gold_truths([project["id"] for project in legacy_projects])
    mre_runs_by_project = fetch_mre_runs([project["id"] for project in legacy_projects])

    run_label = args.run_label or f"gold_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    batch = None
    if not args.no_save:
        batch = create_gold_backtest_batch({
            "run_label": run_label,
            "batch_status": "running",
            "requested_count": len(legacy_projects),
            "input_selector": {
                "project_ids": args.project_id,
                "limit": args.limit,
                "research_missing_truth": args.research_missing_truth,
                "max_parallel_truth_projects": args.max_parallel_truth_projects,
                "research_missing_evidence": args.research_missing_evidence,
                "max_parallel_projects": args.max_parallel_projects,
                "research_missing_analogs": args.research_missing_analogs,
                "max_parallel_analog_projects": args.max_parallel_analog_projects,
            },
            "started_at": datetime.now(timezone.utc).isoformat(),
        })

    results: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    parallel_truth_spend_count = 0
    parallel_spend_count = 0
    parallel_analog_spend_count = 0

    for project in legacy_projects:
        truth = validated_gold_truths.get(project["id"])
        exclusion_reason = None
        if not truth:
            truth, exclusion_reason = build_truth_row(project, mre_runs_by_project.get(project["id"], []))
        if exclusion_reason or not truth:
            if args.research_missing_truth:
                prompt, schema = parallel_truth_prompt(project, mre_runs_by_project.get(project["id"], []))
                try:
                    if not args.no_save:
                        ensure_project_for_parallel_cache(project, data_status="candidate")
                    response, paid_call = run_parallel_cached(
                        task_kind="mre_truth",
                        project_id=project["id"],
                        cutoff_date=None,
                        prompt=prompt,
                        output_schema=schema,
                        save=not args.no_save,
                        allow_paid=parallel_truth_spend_count < args.max_parallel_truth_projects,
                    )
                    if paid_call:
                        parallel_truth_spend_count += 1
                except Exception as exc:
                    LOGGER.warning("Parallel MRE truth research failed for %s: %s", project.get("name"), exc)
                    response = None
                if isinstance(response, dict):
                    repaired_truth, repair_reason = truth_row_from_parallel(project, response)
                    if repaired_truth:
                        truth = repaired_truth
                        exclusion_reason = None
                    else:
                        exclusion_reason = repair_reason or exclusion_reason

        if exclusion_reason or not truth:
            excluded.append({
                "project_id": project["id"],
                "project_name": project.get("name"),
                "reason": exclusion_reason or "truth_builder_failed",
            })
            if not args.no_save:
                upsert_gold_project(build_gold_project_row(project, data_status="excluded", exclusion_reason=exclusion_reason))
            continue

        cutoff = parse_loose_date(truth.get("effective_date") or truth.get("publication_date"))
        if not cutoff:
            excluded.append({
                "project_id": project["id"],
                "project_name": project.get("name"),
                "reason": "truth_missing_cutoff_date",
            })
            continue

        project_row = build_gold_project_row(project, data_status="truth_validated")
        evidence_rows = evidence_rows_from_payload(
            project_id=project["id"],
            truth_id=None,
            cutoff_date=cutoff,
            evidence=project.get("drilling_evidence"),
        )
        accepted_evidence, _rejected = split_evidence(row for row in evidence_rows if row.get("evidence_status") == "accepted")
        tonnage, _ = evidence_tonnage_proxy(accepted_evidence)
        grade, _ = evidence_grade_proxy(accepted_evidence)
        needs_parallel_evidence = args.research_missing_evidence and (tonnage is None or grade is None)
        if needs_parallel_evidence:
            prompt, schema = parallel_evidence_prompt(project, cutoff)
            try:
                if not args.no_save:
                    ensure_project_for_parallel_cache(project, data_status="candidate")
                response, paid_call = run_parallel_cached(
                    task_kind="pre_mre_evidence",
                    project_id=project["id"],
                    cutoff_date=cutoff,
                    prompt=prompt,
                    output_schema=schema,
                    save=not args.no_save,
                    allow_paid=parallel_spend_count < args.max_parallel_projects,
                )
                if paid_call:
                    parallel_spend_count += 1
            except Exception as exc:
                LOGGER.warning("Parallel evidence research failed for %s: %s", project.get("name"), exc)
                response = None
            if isinstance(response, dict):
                evidence_rows.extend(
                    evidence_rows_from_payload(
                        project_id=project["id"],
                        truth_id=None,
                        cutoff_date=cutoff,
                        evidence=response,
                        raw_parallel_output=response,
                    )
                )
                for rejected in response.get("rejected_sources") or []:
                    if isinstance(rejected, dict):
                        evidence_rows.append(
                            evidence_fact_row(
                                project_id=project["id"],
                                truth_id=None,
                                cutoff_date=cutoff,
                                evidence={
                                    "source_url": rejected.get("source_url"),
                                    "source_date": rejected.get("source_date"),
                                    "source_title": rejected.get("source_title"),
                                    "source": "parallel_rejected_source",
                                    "confidence": "low",
                                    "notes": rejected.get("reason"),
                                },
                                fact_type="rejected_source",
                                value_text=rejected.get("reason"),
                                raw_parallel_output=response,
                            )
                        )

        analogs = load_legacy_analogs(project)
        analog_rows = [
            row for row in (analog_candidate_row(project["id"], analog) for analog in analogs)
            if row is not None
        ]
        accepted_evidence, _ = split_evidence(row for row in evidence_rows if row.get("evidence_status") == "accepted")
        target_tonnage, _ = evidence_tonnage_proxy(accepted_evidence)
        target_grade, _ = evidence_grade_proxy(accepted_evidence)
        if args.research_missing_analogs and target_tonnage is not None and target_grade is not None:
            clean_rows, _ = clean_analog_cohort(
                project_row,
                analog_rows,
                cutoff_date=cutoff,
                target_tonnage_mt=target_tonnage,
                target_grade_gpt=target_grade,
            )
            split_ready_rows = [
                row for row in clean_rows
                if all(
                    positive_float(row.get(field)) is not None
                    for field in ("mi_tonnage_mt", "mi_grade_gpt", "inferred_tonnage_mt", "inferred_grade_gpt")
                )
            ]
            if len(clean_rows) < 3 or len(split_ready_rows) < 3:
                prompt, schema = parallel_analog_prompt(
                    project,
                    cutoff,
                    target_tonnage_mt=target_tonnage,
                    target_grade_gpt=target_grade,
                )
                try:
                    if not args.no_save:
                        ensure_project_for_parallel_cache(project, data_status="candidate")
                    response, paid_call = run_parallel_cached(
                        task_kind="analog_research",
                        project_id=project["id"],
                        cutoff_date=cutoff,
                        prompt=prompt,
                        output_schema=schema,
                        save=not args.no_save,
                        allow_paid=parallel_analog_spend_count < args.max_parallel_analog_projects,
                    )
                    if paid_call:
                        parallel_analog_spend_count += 1
                except Exception as exc:
                    LOGGER.warning("Parallel analog research failed for %s: %s", project.get("name"), exc)
                    response = None
                if isinstance(response, dict):
                    researched_analogs = []
                    for analog in response.get("analogs") or []:
                        if isinstance(analog, dict):
                            researched_analogs.append({**analog, "raw_parallel_output": response})
                    if researched_analogs:
                        analogs = merge_analogs([*analogs, *researched_analogs])
                        analog_rows = [
                            row for row in (analog_candidate_row(project["id"], analog) for analog in analogs)
                            if row is not None
                        ]
        decision_rows = decision_rows_for_candidates(project_row, evidence_rows, analog_rows, cutoff_date=cutoff)

        truth_row = truth
        if not args.no_save:
            upsert_gold_project(project_row)
            saved_truth = upsert_gold_mre_truth(truth)
            truth_id = saved_truth.get("id")
            if truth_id:
                truth_row = {**truth, **saved_truth}
                evidence_rows = [{**row, "mre_truth_id": truth_id} for row in evidence_rows]
            saved_evidence = upsert_gold_pre_mre_evidence(evidence_rows)
            saved_analogs = upsert_gold_analog_candidates(analog_rows)
            saved_decisions = upsert_gold_analog_decisions(decision_rows)
            bundle = load_gold_case_bundle(project["id"])
            project_for_prediction = bundle["project"]
            truth_for_prediction = bundle["truth"]
            evidence_for_prediction = bundle["evidence"]
            analogs_for_prediction = bundle["analog_candidates"]
        else:
            truth_id = truth.get("id") or stable_uuid("truth", project["id"], cutoff)
            saved_evidence = evidence_rows
            saved_analogs = analog_rows
            saved_decisions = decision_rows
            project_for_prediction = project_row
            truth_for_prediction = {**truth_row, "id": truth_id, "cutoff_date": cutoff}
            evidence_for_prediction = [row for row in evidence_rows if row.get("evidence_status") == "accepted"]
            analogs_for_prediction = analog_rows

        prediction = predict_gold_resource(
            project_for_prediction,
            evidence_for_prediction,
            analogs_for_prediction,
            cutoff_date=cutoff,
        )
        score = score_gold_prediction(prediction, truth_for_prediction, threshold=args.threshold)
        saved_run = {}
        saved_score = {}
        if not args.no_save:
            saved_run = upsert_gold_prediction_run(
                prediction_run_row(
                    project_id=project["id"],
                    truth_id=truth_for_prediction["id"],
                    batch_id=batch.get("id") if batch else None,
                    prediction=prediction,
                    decision_rows=saved_decisions,
                )
            )
            if saved_run.get("id"):
                saved_score = upsert_gold_prediction_score(score_row(saved_run["id"], truth_for_prediction["id"], score))

        results.append({
            "project_id": project["id"],
            "project_name": project.get("name"),
            "truth_id": truth_for_prediction.get("id") if isinstance(truth_for_prediction, dict) else truth_id,
            "run_status": prediction["run_status"],
            "no_prediction_reasons": prediction.get("no_prediction_reasons") or [],
            "core_pass": score["core_pass"],
            "split_pass": score["split_pass"],
            "production_like_pass": score["production_like_pass"],
            "failure_class": score.get("failure_class"),
            "failure_reason": score.get("failure_reason"),
            "metrics": {
                key: score.get(key)
                for key in (
                    "tonnage_error_pct",
                    "grade_error_pct",
                    "contained_error_pct",
                    "mi_tonnage_error_pct",
                    "mi_grade_error_pct",
                    "inferred_tonnage_error_pct",
                    "inferred_grade_error_pct",
                )
            },
            "db_rows": {
                "evidence": len(saved_evidence),
                "analog_candidates": len(saved_analogs),
                "analog_decisions": len(saved_decisions),
                "prediction_run_id": saved_run.get("id"),
                "prediction_score_id": saved_score.get("id"),
            },
        })

    predicted_count = sum(1 for row in results if row["run_status"] == "predicted")
    no_prediction_count = sum(1 for row in results if row["run_status"] == "no_prediction")
    core_pass_count = sum(1 for row in results if row["core_pass"])
    split_pass_count = sum(1 for row in results if row["split_pass"])
    production_like_pass_count = sum(1 for row in results if row["production_like_pass"])
    failure_count = sum(1 for row in results if row["run_status"] == "predicted" and not row["core_pass"])

    if batch and not args.no_save:
        update_gold_backtest_batch(batch["id"], {
            "batch_status": "complete",
            "evaluated_count": len(results),
            "pass_count": production_like_pass_count,
            "no_prediction_count": no_prediction_count,
            "failure_count": failure_count,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

    after_counts = gold_table_counts()
    summary = {
        "run_label": run_label,
        "projects_found": len(legacy_projects),
        "truth_validated": len(results),
        "excluded": len(excluded),
        "predicted": predicted_count,
        "no_prediction": no_prediction_count,
        "core_pass": core_pass_count,
        "split_pass": split_pass_count,
        "production_like_pass": production_like_pass_count,
        "parallel_research_calls": parallel_truth_spend_count + parallel_spend_count + parallel_analog_spend_count,
        "parallel_truth_research_calls": parallel_truth_spend_count,
        "parallel_evidence_research_calls": parallel_spend_count,
        "parallel_analog_research_calls": parallel_analog_spend_count,
        "no_prediction_reasons": dict(Counter(reason for row in results for reason in row["no_prediction_reasons"])),
        "pass_project_names_metrics": [
            {
                "project": row["project_name"],
                "metrics": row["metrics"],
            }
            for row in results
            if row["production_like_pass"]
        ],
        "failures_reasons": [
            {
                "project": row["project_name"],
                "status": row["run_status"],
                "failure_class": row.get("failure_class"),
                "failure_reason": row.get("failure_reason"),
                "no_prediction_reasons": row["no_prediction_reasons"],
            }
            for row in results
            if not row["production_like_pass"]
        ],
        "excluded_reasons": excluded,
        "db_rows_populated_delta": {
            table: after_counts.get(table, 0) - before_counts.get(table, 0)
            for table in sorted(after_counts)
        },
        "db_rows_total": after_counts,
        "batch_id": batch.get("id") if batch else None,
        "results": results,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", action="append", default=[], help="Legacy/gold project UUID to run. Repeatable.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected truth-backed gold projects.")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--no-save", action="store_true", help="Do not write gold_* rows.")
    parser.add_argument("--research-missing-truth", action="store_true", help="Use cached/Parallel research for projects without validated first-MRE truth.")
    parser.add_argument("--max-parallel-truth-projects", type=int, default=0, help="Maximum paid Parallel MRE truth calls for this run.")
    parser.add_argument("--research-missing-evidence", action="store_true", help="Use cached/Parallel research when target evidence is insufficient.")
    parser.add_argument("--max-parallel-projects", type=int, default=0, help="Maximum paid Parallel evidence calls for this run.")
    parser.add_argument("--research-missing-analogs", action="store_true", help="Use cached/Parallel research when the clean analog cohort is insufficient.")
    parser.add_argument("--max-parallel-analog-projects", type=int, default=0, help="Maximum paid Parallel analog research calls for this run.")
    parser.add_argument("--processor", default=None, help="Override PARALLEL_PROCESSOR.")
    parser.add_argument("--poll-timeout-s", type=int, default=None, help="Override Parallel poll timeout.")
    parser.add_argument("--json-out", default=None, help="Optional path for the machine-readable summary.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = run(args)
    text = json.dumps(summary, indent=2, sort_keys=True, default=str)
    print(text)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
