#!/usr/bin/env python3
"""Run blind Parallel.ai gold-model backtests in parallel.

This launches `nodes.parallel_gold_model.parallel_gold_model_node` with
use_mre=False for a batch of MRE-backed gold projects, persists each completed
run to `model_runs`, and prints a 95% reconciliation table against the
published MRE mirror fields on `projects`.

Examples:
    python3 scripts/run_parallel_gold_backtest.py --fixture-first 10 --workers 5
    python3 scripts/run_parallel_gold_backtest.py --project-id <uuid> --workers 1
    python3 scripts/run_parallel_gold_backtest.py --no-save --fixture-first 3
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from nodes import geo_taxonomy, supabase_ops  # noqa: E402
from nodes import parallel_gold_model as parallel_gold_model_module  # noqa: E402
from nodes.parallel_gold_model import parallel_gold_model_node  # noqa: E402
from scripts.backtest_parallel import official_truth, pct_err, fmt_pct  # noqa: E402
from scripts.gold_backtest_diagnostics import (  # noqa: E402
    classify_failure,
    evidence_quality_score,
    extract_local_guards,
    leaderboard_row,
)
from config import settings  # noqa: E402


TROY_OZ_PER_TONNE = 32150.7466
DEFAULT_FIXTURES = (
    "aurmac", "cadillac", "doyle", "fenn_gib", "goldfields",
    "hammerdown", "opinaca", "p2_gold", "red_hill", "rhosgobel",
)
_SUPABASE_LOCK = RLock()
_DATE_RE = re.compile(r"\b(19|20)\d{2}(?:[-/](?:0?[1-9]|1[0-2])(?:[-/](?:0?[1-9]|[12]\d|3[01]))?)?\b")


def _parse_loose_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, dict):
        for key in ("as_of_date", "effective_date", "report_date", "source_date"):
            parsed = _parse_loose_date(value.get(key))
            if parsed:
                return parsed
        return None
    match = _DATE_RE.search(str(value))
    if not match:
        return None
    parts = match.group(0).replace("/", "-").split("-")
    try:
        return date(
            int(parts[0]),
            int(parts[1]) if len(parts) > 1 else 12,
            int(parts[2]) if len(parts) > 2 else 31,
        )
    except ValueError:
        return None


def _evidence_is_pre_cutoff(evidence: Optional[Dict[str, Any]], cutoff: Optional[date]) -> bool:
    if not evidence or not cutoff:
        return bool(evidence)
    if _evidence_mentions_target_mre(evidence):
        return False
    if evidence.get("queried_pre_mre_cutoff") == cutoff.isoformat():
        return True
    # Prefer the publication/source date over report_cutoff_date. The latter
    # is often our synthetic "search before this MRE cutoff" marker, not the
    # date the evidence itself became public.
    source_date = _parse_loose_date(evidence.get("source_date") or evidence.get("source_url"))
    if source_date:
        return source_date < cutoff
    source_date = _parse_loose_date(evidence.get("report_cutoff_date") or evidence.get("extracted_at"))
    return bool(source_date and source_date < cutoff)


def _evidence_score_value(evidence: Optional[Dict[str, Any]]) -> int:
    return int((evidence_quality_score(evidence) or {}).get("score") or 0)


_EVIDENCE_MRE_MARKERS = (
    "maiden resource",
    "mineral resource",
    "mineral resource estimate",
    "resource update",
    "resource estimate",
    "updated resource",
    "updated mre",
    "mre technical report",
    "technical report",
    "ni 43-101",
    "jorc",
)


def _evidence_mentions_target_mre(evidence: Dict[str, Any]) -> bool:
    """True when cached evidence is actually sourced from target MRE material."""
    text = " ".join(
        str(evidence.get(k) or "")
        for k in (
            "source_url", "source_title", "source_name", "notes", "summary",
            "report_title", "report_type",
        )
    ).lower()
    text = re.sub(r"[_\\/-]+", " ", text)
    return any(marker in text for marker in _EVIDENCE_MRE_MARKERS)


def _extract_pre_mre_target_evidence(project: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract target drilling/geometry from sources before the target MRE.

    This is intentionally backtest-specific: it asks Exa Answer for public
    drill facts available before the cutoff and refuses to use the MRE itself.
    """
    if not settings.exa_api_key:
        logging.warning("[pre-mre-evidence] EXA_API_KEY not configured")
        return None
    cutoff = _parse_loose_date(project.get("mre_date") or project.get("mre_data_source"))
    cutoff_s = cutoff.isoformat() if cutoff else "the first published mineral resource estimate"
    loc = ", ".join(
        p for p in (
            project.get("region"),
            project.get("state_or_province"),
            project.get("country"),
        )
        if p
    )
    query = (
        f"For the {project.get('name')} gold project {loc}, find ONLY public "
        f"company disclosures published before {cutoff_s}. Extract cumulative "
        f"drilling meters/holes completed before the first MRE, representative "
        f"pre-MRE average or weighted assay grade, mineralized strike length, "
        f"depth/down-dip extent, true width, and source URLs. Do not use or "
        f"quote the MRE resource tonnes, grade, or ounces."
    )
    payload = {
        "query": query,
        "system_prompt": (
            "You are a mining backtest data auditor. Use only drill releases, "
            "presentations, exchange filings, and technical summaries published "
            "before the cutoff. Do not use the target's MRE numbers. If a value "
            "is not available pre-cutoff, return null rather than estimating."
        ),
        "output_schema": {
            "type": "object",
            "properties": {
                "total_holes": {"type": ["integer", "null"]},
                "total_meters_drilled": {"type": ["number", "null"]},
                "weighted_grade_g_t": {"type": ["number", "null"]},
                "strike_length_m": {"type": ["number", "null"]},
                "down_dip_extent_m": {"type": ["number", "null"]},
                "avg_true_width_m": {"type": ["number", "null"]},
                "drilled_area_km2": {"type": ["number", "null"]},
                "best_intercepts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hole_id": {"type": ["string", "null"]},
                            "interval_m": {"type": ["number", "null"]},
                            "grade_g_t": {"type": ["number", "null"]},
                            "source_url": {"type": ["string", "null"]},
                            "source_date": {"type": ["string", "null"]},
                        },
                    },
                },
                "report_cutoff_date": {"type": ["string", "null"]},
                "source_url": {"type": ["string", "null"]},
                "source_date": {"type": ["string", "null"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "notes": {"type": ["string", "null"]},
            },
            "required": ["total_holes", "total_meters_drilled", "weighted_grade_g_t", "confidence"],
        },
        "text": False,
    }
    try:
        resp = requests.post(
            "https://api.exa.ai/answer",
            headers={"x-api-key": settings.exa_api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logging.warning("[pre-mre-evidence] Exa request failed for %s: %s", project.get("name"), exc)
        return None
    raw = (resp.json() or {}).get("answer")
    try:
        answer = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        logging.warning("[pre-mre-evidence] Could not parse answer for %s", project.get("name"))
        return None
    intercepts = answer.get("best_intercepts") or []
    if isinstance(intercepts, dict) and isinstance(intercepts.get("value"), list):
        intercepts = intercepts["value"]
    evidence = {
        "total_holes": answer.get("total_holes"),
        "total_meters_drilled": answer.get("total_meters_drilled"),
        "weighted_grade_g_t": answer.get("weighted_grade_g_t"),
        "strike_length_m": answer.get("strike_length_m"),
        "down_dip_extent_m": answer.get("down_dip_extent_m"),
        "avg_true_width_m": answer.get("avg_true_width_m"),
        "drilled_area_km2": answer.get("drilled_area_km2"),
        "best_intercepts": intercepts if isinstance(intercepts, list) else [],
        "qa_qc_present": None,
        "source": "exa_pre_mre_target_evidence",
        "source_url": answer.get("source_url"),
        "source_date": answer.get("source_date"),
        "report_cutoff_date": answer.get("report_cutoff_date") or (cutoff.isoformat() if cutoff else None),
        "confidence": answer.get("confidence", "low"),
        "notes": answer.get("notes"),
        "queried_pre_mre_cutoff": cutoff.isoformat() if cutoff else None,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    if (
        evidence.get("total_meters_drilled") is None
        and evidence.get("weighted_grade_g_t") is None
        and evidence.get("strike_length_m") is None
    ):
        return None
    if cutoff and not _evidence_is_pre_cutoff(evidence, cutoff):
        logging.warning(
            "[pre-mre-evidence] rejected post-cutoff evidence for %s: cutoff=%s evidence_date=%s",
            project.get("name"),
            cutoff,
            evidence.get("report_cutoff_date") or evidence.get("source_date"),
        )
        return None
    return evidence


def _round(value: Any, ndigits: int = 3) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _contained_oz(tonnage_mt: Optional[float], grade_gpt: Optional[float]) -> Optional[float]:
    if tonnage_mt is None or grade_gpt is None:
        return None
    return float(tonnage_mt) * float(grade_gpt) * TROY_OZ_PER_TONNE


def _fields_from_parallel(
    project: Dict[str, Any],
    parallel_out: Dict[str, Any],
    *,
    batch_id: Optional[str] = None,
    evidence_score: Optional[Dict[str, Any]] = None,
    failure_class: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mi = parallel_out.get("m_and_i") or {}
    inf = parallel_out.get("inferred") or {}

    mi_mt = mi.get("tonnage_mt")
    mi_g = mi.get("grade_gpt")
    inf_mt = inf.get("tonnage_mt")
    inf_g = inf.get("grade_gpt")

    mi_oz = (
        float(mi["contained_moz"]) * 1_000_000.0
        if mi.get("contained_moz") is not None
        else _contained_oz(mi_mt, mi_g)
    )
    inf_oz = (
        float(inf["contained_moz"]) * 1_000_000.0
        if inf.get("contained_moz") is not None
        else _contained_oz(inf_mt, inf_g)
    )

    total_mt = None
    if mi_mt is not None or inf_mt is not None:
        total_mt = float(mi_mt or 0.0) + float(inf_mt or 0.0)

    avg_grade = None
    if total_mt and total_mt > 0:
        avg_grade = (
            float(mi_mt or 0.0) * float(mi_g or 0.0)
            + float(inf_mt or 0.0) * float(inf_g or 0.0)
        ) / total_mt

    total_oz = None
    if mi_oz is not None or inf_oz is not None:
        total_oz = float(mi_oz or 0.0) + float(inf_oz or 0.0)

    conviction = parallel_out.get("conviction") or {}
    level = conviction.get("level") or "unknown"

    return {
        "mi_tonnage_mt": _round(mi_mt),
        "mi_grade": _round(mi_g),
        "mi_contained": _round(mi_oz, 3),
        "inferred_resource_mt": _round(inf_mt),
        "inferred_grade": _round(inf_g),
        "inferred_contained": _round(inf_oz, 3),
        "tonnage_mt": _round(total_mt),
        "grade_value": _round(avg_grade),
        "total_contained": _round(total_oz, 3),
        "conviction_score": f"PARALLEL-{level.upper()}",
        "conviction_tier": conviction.get("rationale") or "",
        "p10_tonnage_mt": None,
        "p50_tonnage_mt": None,
        "p90_tonnage_mt": None,
        "p10_grade": None,
        "p50_grade": None,
        "p90_grade": None,
        "p10_contained": None,
        "p50_contained": None,
        "p90_contained": None,
        "cv_contained": None,
        "signal_contributions_json": {
            "source": "parallel.ai",
            "runner": "scripts/run_parallel_gold_backtest.py",
            "batch_id": batch_id,
            "use_mre": False,
            "anchor_used": parallel_out.get("anchor_used"),
            "methodology": parallel_out.get("methodology"),
            "analogs_used": parallel_out.get("analogs_used"),
            "analogs_rejected": parallel_out.get("analogs_rejected"),
            "sources": parallel_out.get("sources"),
            "evidence_quality": evidence_score,
            "failure_class": failure_class,
            "local_guards": extract_local_guards(parallel_out),
        },
    }


def _has_full_truth(row: Dict[str, Any]) -> bool:
    return all(
        row.get(k) is not None
        for k in (
            "mre_mi_tonnage_mt", "mre_mi_grade",
            "mre_inferred_tonnage_mt", "mre_inferred_grade",
        )
    )


def _fixture_targets(names: Iterable[str], *, allow_no_truth: bool) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for name in names:
        path = ROOT / "tests" / "fixtures" / "backtest" / f"{name}.json"
        data = json.loads(path.read_text())
        if not allow_no_truth and not _has_full_truth(data):
            print(f"[parallel-gold-backtest] skip fixture {name}: no full MRE truth")
            continue
        targets.append({"project_id": data["id"], "fixture": data, "fixture_name": name})
    return targets


def _fetch_db_truth_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        res = (
            supabase_ops.get_client()
            .table("projects")
            .select(
                "id,name,mre_mi_tonnage_mt,mre_mi_grade,"
                "mre_inferred_tonnage_mt,mre_inferred_grade"
            )
            .ilike("material", "gold")
            .not_.is_("mre_mi_tonnage_mt", "null")
            .not_.is_("mre_mi_grade", "null")
            .not_.is_("mre_inferred_tonnage_mt", "null")
            .not_.is_("mre_inferred_grade", "null")
            .range(offset, offset + 999)
            .execute()
        )
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return rows


def _select_truth_target_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    limit: int,
    exclude_project_ids: Iterable[str] = (),
    random_seed: Optional[str] = None,
    randomize: bool = False,
) -> List[Dict[str, Any]]:
    excluded = {pid for pid in exclude_project_ids if pid}
    eligible = [
        row
        for row in rows
        if row.get("id") and row["id"] not in excluded and _has_full_truth(row)
    ]
    if randomize:
        eligible = sorted(eligible, key=lambda row: ((row.get("name") or "").lower(), row["id"]))
        random.Random(random_seed).shuffle(eligible)
    return eligible[:limit]


def _db_truth_targets(
    limit: int,
    *,
    exclude_project_ids: Iterable[str] = (),
    random_seed: Optional[str] = None,
    randomize: bool = False,
) -> List[Dict[str, Any]]:
    selected = _select_truth_target_rows(
        _fetch_db_truth_rows(),
        limit=limit,
        exclude_project_ids=exclude_project_ids,
        random_seed=random_seed,
        randomize=randomize,
    )
    return [
        {"project_id": row["id"], "fixture": None, "fixture_name": None}
        for row in selected
    ]


def _run_project_ids(run_ids: Iterable[str]) -> Set[str]:
    ids = [run_id for run_id in run_ids if run_id]
    if not ids:
        return set()
    res = (
        supabase_ops.get_client()
        .table("model_runs")
        .select("project_id,run_id")
        .in_("run_id", ids)
        .execute()
    )
    return {row["project_id"] for row in (res.data or []) if row.get("project_id")}


def _blind_cutoff_from_mre_run(latest_mre: Dict[str, Any]) -> Optional[str]:
    effective = latest_mre.get("effective_date")
    if not effective:
        return None
    source = str(latest_mre.get("source") or "")
    if source == "exa_2pass_mre_truth_backfill":
        parsed = _parse_loose_date(effective)
        if parsed and parsed.month == 12 and parsed.day == 31:
            return f"{parsed.year}-01-01"
    return effective


def _analog_key(analog: Dict[str, Any]) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(analog.get("name") or analog.get("analog_name") or "").strip().lower(),
    )


def _positive_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return f if math.isfinite(f) and f > 0 else None
    except (TypeError, ValueError):
        return None


def _gold_library_filters(project: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Infer the same minimal gold routing fields Analog Finder uses."""
    material = project.get("material") or "gold"
    deposit_type = project.get("deposit_type")
    deposit_subtype = project.get("deposit_subtype")
    target_belt = project.get("tectonic_belt")
    material_key = str(material or "").strip().lower()
    blob = " ".join(
        str(project.get(k) or "")
        for k in (
            "tectonic_belt", "district", "region", "location_name",
            "mining_method", "mining_method_class", "processing_method",
            "recovery_method",
        )
    ).lower()
    if (
        material_key in {"gold", "au"}
        and not deposit_type
        and not deposit_subtype
        and target_belt in geo_taxonomy.BELT_COMPATIBILITY_GROUPS.get("archean_greenstone", frozenset())
    ):
        deposit_type = "orogenic gold"
        deposit_subtype = "orogenic_general"
    if (
        material_key in {"gold", "au"}
        and not deposit_subtype
        and (target_belt == "guiana_shield" or "shear" in str(deposit_type or "").lower())
    ):
        deposit_type = "orogenic gold"
        deposit_subtype = "orogenic_general"
    if material_key in {"gold", "au"} and str(deposit_subtype or "").lower() == "carlin_general":
        target_belt = "great_basin_carlin"
    if (
        material_key in {"gold", "au"}
        and not deposit_subtype
        and "near-surface" in str(deposit_type or "").lower()
        and (target_belt == "yukon_tintina" or "yukon" in blob or "tintina" in blob)
    ):
        deposit_type = "intrusion-related gold"
        deposit_subtype = "irgs_general"
    if (
        material_key in {"gold", "au"}
        and not deposit_type
        and not deposit_subtype
        and (target_belt == "andean" or "andean" in blob or "maricunga" in blob)
        and ("heap" in blob or "open pit" in blob or "open-pit" in blob or "heap_leach_pad" in blob)
    ):
        deposit_type = "epithermal-HS"
        deposit_subtype = "high_sulfidation_epithermal"
    return {
        "material": material,
        "deposit_type": deposit_type,
        "deposit_subtype": deposit_subtype,
        "target_tectonic_belt": target_belt,
    }


def _resource_variant_key(analog: Dict[str, Any]) -> Optional[tuple]:
    tonnage = _positive_float(analog.get("tonnage_mt"))
    grade = _positive_float(analog.get("grade_value"))
    if not tonnage or not grade:
        return None
    subtype = str(analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
    family = "orogenic" if "orogenic" in subtype else subtype
    return (family, round(tonnage, 1), round(grade, 1))


def _blind_library_analog_is_compatible(project: Dict[str, Any], analog: Dict[str, Any]) -> bool:
    """Production-like guard for approved-library rows used in blind backtests."""
    material = str(project.get("material") or "").strip().lower()
    if material not in {"gold", "au"}:
        return True
    tonnage = _positive_float(analog.get("tonnage_mt"))
    grade = _positive_float(analog.get("grade_value"))
    if not tonnage or not grade:
        return False

    target_mining = str(
        project.get("mining_method_class") or project.get("mining_method") or ""
    ).strip().lower()
    analog_mining = str(
        analog.get("mining_method_class") or analog.get("mining_method") or ""
    ).strip().lower()
    if target_mining == "underground_vein":
        if analog_mining and not geo_taxonomy.mining_method_compatible(target_mining, analog_mining):
            return False
        if not analog_mining and grade < 2.0:
            return False
        if not analog_mining and tonnage >= 50:
            return False
    elif target_mining and analog_mining and not geo_taxonomy.mining_method_compatible(target_mining, analog_mining):
        return False
    return True


def _merge_library_analogs(
    project: Dict[str, Any],
    supplied: Sequence[Dict[str, Any]],
    library: Sequence[Dict[str, Any]],
    *,
    max_count: int = 8,
) -> List[Dict[str, Any]]:
    cutoff = _parse_loose_date(project.get("mre_date") or project.get("mre_data_source"))
    cleaned = parallel_gold_model_module._clean_blind_analogs(
        project,
        [*supplied, *library],
        cutoff,
    )
    merged: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for analog in cleaned:
        key = _analog_key(analog)
        if not key or key in seen:
            continue
        if not _blind_library_analog_is_compatible(project, analog):
            continue
        variant_key = _resource_variant_key(analog)
        if variant_key:
            variant_marker = f"resource::{variant_key!r}"
            if variant_marker in seen:
                continue
            seen.add(variant_marker)
        if key in seen:
            continue
        seen.add(key)
        merged.append(analog)
        if len(merged) >= max_count:
            break
    return merged


def _supplement_with_library_analogs(
    project: Dict[str, Any],
    analogs: Sequence[Dict[str, Any]],
    *,
    min_count: int = 3,
    max_count: int = 8,
) -> List[Dict[str, Any]]:
    if len(analogs) >= min_count:
        return list(analogs)
    filters = _gold_library_filters(project)
    material = filters["material"] or "gold"
    deposit_type = filters["deposit_type"]
    deposit_subtype = filters["deposit_subtype"]
    if not deposit_type and not deposit_subtype:
        return list(analogs)
    try:
        library = supabase_ops.get_approved_analogs(
            material=material,
            deposit_type=deposit_type,
            deposit_subtype=deposit_subtype,
            target_tectonic_belt=filters["target_tectonic_belt"],
            limit=50,
        )
    except Exception:
        logging.exception(
            "[parallel-gold-backtest] failed loading approved analog library for %s",
            project.get("name"),
        )
        return list(analogs)
    merged = _merge_library_analogs(project, analogs, library, max_count=max_count)
    if len(merged) < min_count and filters["target_tectonic_belt"]:
        try:
            broad_library = supabase_ops.get_approved_analogs(
                material=material,
                deposit_type=deposit_type,
                deposit_subtype=deposit_subtype,
                target_tectonic_belt=None,
                limit=50,
            )
        except Exception:
            logging.exception(
                "[parallel-gold-backtest] failed loading broad approved analog library for %s",
                project.get("name"),
            )
        else:
            merged = _merge_library_analogs(project, merged, broad_library, max_count=max_count)
    if len(merged) > len(analogs):
        logging.info(
            "[parallel-gold-backtest] seeded %s approved-library analog(s) for %s",
            len(merged) - len(analogs),
            project.get("name"),
        )
    return merged


def _merge_fixture_truth(project: Dict[str, Any], fixture: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not fixture:
        merged = dict(project)
    else:
        merged = dict(project)
        for key in (
            "mre_mi_tonnage_mt", "mre_mi_grade", "mre_mi_contained",
            "mre_inferred_tonnage_mt", "mre_inferred_grade",
            "mre_inferred_contained", "mre_data_source",
        ):
            if merged.get(key) is None and fixture.get(key) is not None:
                merged[key] = fixture[key]

    with _SUPABASE_LOCK:
        latest_mre = supabase_ops.get_latest_mre_run(project["id"])
    if latest_mre:
        blind_cutoff = _blind_cutoff_from_mre_run(latest_mre)
        merged["mre_data_source"] = {
            "as_of_date": blind_cutoff,
            "reported_effective_date": latest_mre.get("effective_date"),
            "source_url": latest_mre.get("source_url"),
            "source_doc": latest_mre.get("source"),
            "notes": latest_mre.get("notes"),
        }
        merged["mre_date"] = blind_cutoff
        merged["mre_source_url"] = latest_mre.get("source_url")
    return merged


def _run_one(
    project_id: str,
    *,
    fixture: Optional[Dict[str, Any]],
    save: bool,
    find_analogs: bool,
    refresh_target_evidence: bool,
    batch_id: str,
    threshold: float,
) -> Dict[str, Any]:
    with _SUPABASE_LOCK:
        project = supabase_ops.get_project(project_id)
    if not project:
        return {"project_id": project_id, "error": "project not found"}
    project_for_model = _merge_fixture_truth(project, fixture)
    cutoff = _parse_loose_date(project_for_model.get("mre_date") or project_for_model.get("mre_data_source"))
    cached_evidence = project_for_model.get("drilling_evidence")
    cached_pre_cutoff = cached_evidence if _evidence_is_pre_cutoff(cached_evidence, cutoff) else None
    if refresh_target_evidence or not _evidence_is_pre_cutoff(cached_evidence, cutoff):
        evidence = _extract_pre_mre_target_evidence(project_for_model)
        if evidence:
            save_evidence = True
            if cached_pre_cutoff:
                cached_score = _evidence_score_value(cached_pre_cutoff)
                fresh_score = _evidence_score_value(evidence)
                if cached_score > fresh_score:
                    logging.info(
                        "[pre-mre-evidence] keeping richer cached evidence for %s "
                        "(cached_score=%s fresh_score=%s)",
                        project_for_model.get("name"),
                        cached_score,
                        fresh_score,
                    )
                    evidence = cached_pre_cutoff
                    save_evidence = False
            project_for_model = {**project_for_model, "drilling_evidence": evidence}
            for field in ("strike_length_m", "down_dip_extent_m", "avg_true_width_m"):
                if project_for_model.get(field) is None and evidence.get(field) is not None:
                    project_for_model[field] = evidence.get(field)
            if save_evidence and save and project.get("id") and _evidence_is_pre_cutoff(evidence, cutoff):
                with _SUPABASE_LOCK:
                    supabase_ops.save_project_drilling_evidence(project_id, evidence)

    with _SUPABASE_LOCK:
        analogs = supabase_ops.get_analogs(project_id)
        analogs = _supplement_with_library_analogs(project_for_model, analogs)
    state = {
        "project_id": project_id,
        "project": project_for_model,
        "analogs": analogs,
        "use_mre": False,
        "find_analogs": find_analogs,
    }
    out = parallel_gold_model_node(state)
    if out.get("error"):
        return {"project_id": project_id, "project_name": project.get("name"), "error": out["error"]}

    model = out.get("parallel_model") or {}
    evidence_score = evidence_quality_score(project_for_model.get("drilling_evidence"))
    provisional_fields = _fields_from_parallel(
        project_for_model,
        model,
        batch_id=batch_id,
        evidence_score=evidence_score,
    )
    truth = official_truth(project_for_model)
    provisional_errors = {
        "tonnage": pct_err(provisional_fields.get("tonnage_mt"), truth["total_tonnage_mt"]),
        "grade": pct_err(provisional_fields.get("grade_value"), truth["total_grade_gpt"]),
        "contained": pct_err(provisional_fields.get("total_contained"), truth["total_contained_oz"]),
        "mi_tonnage": pct_err(provisional_fields.get("mi_tonnage_mt"), truth["mi_tonnage_mt"]),
        "mi_grade": pct_err(provisional_fields.get("mi_grade"), truth["mi_grade_gpt"]),
        "inferred_tonnage": pct_err(provisional_fields.get("inferred_resource_mt"), truth["inferred_tonnage_mt"]),
        "inferred_grade": pct_err(provisional_fields.get("inferred_grade"), truth["inferred_grade_gpt"]),
    }
    failure = classify_failure(
        errors=provisional_errors,
        project=project_for_model,
        model=model,
        evidence_score=evidence_score,
        threshold=threshold,
    )
    fields = _fields_from_parallel(
        project_for_model,
        model,
        batch_id=batch_id,
        evidence_score=evidence_score,
        failure_class=failure,
    )
    if save:
        with _SUPABASE_LOCK:
            supabase_ops.save_model_run(
                project_id=project_id,
                model_type="parallel_pre_mre",
                fields=fields,
                model_output_json=model,
                run_id=batch_id,
            )

    errors = {
        "tonnage": pct_err(fields.get("tonnage_mt"), truth["total_tonnage_mt"]),
        "grade": pct_err(fields.get("grade_value"), truth["total_grade_gpt"]),
        "contained": pct_err(fields.get("total_contained"), truth["total_contained_oz"]),
        "mi_tonnage": pct_err(fields.get("mi_tonnage_mt"), truth["mi_tonnage_mt"]),
        "mi_grade": pct_err(fields.get("mi_grade"), truth["mi_grade_gpt"]),
        "inferred_tonnage": pct_err(fields.get("inferred_resource_mt"), truth["inferred_tonnage_mt"]),
        "inferred_grade": pct_err(fields.get("inferred_grade"), truth["inferred_grade_gpt"]),
    }
    return {
        "project_id": project_id,
        "project_name": project.get("name"),
        "fields": fields,
        "truth": truth,
        "errors": errors,
        "model": model,
        "batch_id": batch_id,
        "evidence_quality": evidence_score,
        "failure_class": failure,
        "local_guards": extract_local_guards(model),
    }


def _core_pass(errors: Dict[str, Optional[float]], threshold: float) -> bool:
    return all(
        errors[k] is not None
        and not math.isinf(errors[k])
        and abs(errors[k]) <= threshold
        for k in ("tonnage", "grade", "contained")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", action="append", default=[])
    parser.add_argument("--fixture", action="append", choices=DEFAULT_FIXTURES, default=[])
    parser.add_argument("--fixture-first", type=int, default=0)
    parser.add_argument(
        "--random-targets",
        type=int,
        default=0,
        help="Select this many random truth-backed gold projects from Supabase.",
    )
    parser.add_argument(
        "--random-seed",
        default=None,
        help="Seed for --random-targets so production can rerun the same holdout.",
    )
    parser.add_argument(
        "--exclude-project-id",
        action="append",
        default=[],
        help="Project ID to exclude from Supabase target selection. Repeatable.",
    )
    parser.add_argument(
        "--exclude-run-id",
        action="append",
        default=[],
        help="Exclude all project IDs already present in this model_runs.run_id. Repeatable.",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--min-pass-count",
        type=int,
        default=5,
        help="Exit successfully only when at least this many projects pass. Default 5.",
    )
    parser.add_argument("--find-analogs", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--refresh-target-evidence",
        action="store_true",
        help="Fetch pre-MRE target drilling/geometry evidence before running Parallel.",
    )
    parser.add_argument(
        "--allow-no-truth",
        action="store_true",
        help="Allow exploratory runs on projects lacking full MRE truth. Defaults false.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="Print selected target IDs and exit without launching Parallel.",
    )
    parser.add_argument(
        "--processor",
        default=None,
        help="Override PARALLEL_PROCESSOR for this run, e.g. ultra. Defaults to config.",
    )
    parser.add_argument(
        "--poll-timeout-s",
        type=int,
        default=None,
        help="Override local Parallel poll timeout in seconds for backtest runs.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Run ID to write to model_runs.run_id. Defaults to a generated gold_blind_<uuid>.",
    )
    parser.add_argument(
        "--leaderboard-json",
        default=None,
        help="Optional path to write a machine-readable pass/fail leaderboard with failure lessons.",
    )
    args = parser.parse_args()
    if args.processor:
        settings.parallel_processor = args.processor
    if args.poll_timeout_s is not None:
        parallel_gold_model_module._POLL_TIMEOUT_S = max(15, int(args.poll_timeout_s))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    excluded_project_ids = set(args.exclude_project_id)
    excluded_project_ids.update(_run_project_ids(args.exclude_run_id))

    targets = [{"project_id": pid, "fixture": None, "fixture_name": None} for pid in args.project_id]
    if args.fixture:
        targets.extend(_fixture_targets(args.fixture, allow_no_truth=args.allow_no_truth))
    if args.fixture_first:
        targets.extend(
            _fixture_targets(
                DEFAULT_FIXTURES[: args.fixture_first],
                allow_no_truth=args.allow_no_truth,
            )
        )
    if args.random_targets:
        already_selected = {target["project_id"] for target in targets}
        targets.extend(
            _db_truth_targets(
                limit=args.random_targets,
                exclude_project_ids=excluded_project_ids | already_selected,
                random_seed=args.random_seed,
                randomize=True,
            )
        )
    if not targets:
        targets.extend(_db_truth_targets(limit=10, exclude_project_ids=excluded_project_ids))
    deduped: Dict[str, Dict[str, Any]] = {}
    for target in targets:
        deduped[target["project_id"]] = target
    targets = list(deduped.values())
    if len(targets) < 10:
        print(
            f"[parallel-gold-backtest] warning: only {len(targets)} "
            "truth-backed target(s) selected; cannot prove 5/10 95% matches "
            "until more projects have full MRE truth fields.",
            flush=True,
        )
    if args.list_targets:
        if excluded_project_ids:
            print(f"# excluded_project_ids={len(excluded_project_ids)}")
        if args.random_targets:
            print(f"# random_targets={args.random_targets} random_seed={args.random_seed}")
        for target in targets:
            project = supabase_ops.get_project(target["project_id"])
            print(f"{target['project_id']} | {project.get('name') if project else '?'}")
        return 0

    batch_id = args.batch_id or f"gold_blind_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    print(
        f"[parallel-gold-backtest] launching {len(targets)} blind pre-MRE run(s) "
        f"with workers={args.workers}, save={not args.no_save}, "
        f"find_analogs={args.find_analogs}, batch_id={batch_id}"
    )

    results: List[Dict[str, Any]] = []
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = [
        pool.submit(
            _run_one,
            target["project_id"],
            fixture=target.get("fixture"),
            save=not args.no_save,
            find_analogs=args.find_analogs,
            refresh_target_evidence=args.refresh_target_evidence,
            batch_id=batch_id,
            threshold=args.threshold,
        )
        for target in targets
    ]
    try:
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                logging.exception("[parallel-gold-backtest] worker failed")
                result = {"project_id": "unknown", "error": str(exc)}
            results.append(result)
            name = result.get("project_name") or result.get("project_id")
            if result.get("error"):
                print(f"  FAIL  {name}: {result['error']}", flush=True)
            else:
                passed = _core_pass(result["errors"], args.threshold)
                print(
                    f"  {'PASS' if passed else 'MISS'}  {name}: "
                    f"T {fmt_pct(result['errors']['tonnage'])}, "
                    f"G {fmt_pct(result['errors']['grade'])}, "
                    f"Au {fmt_pct(result['errors']['contained'])}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[parallel-gold-backtest] interrupted; cancelling queued futures", flush=True)
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)

    print()
    print(f"{'Project':54s} {'Pass':>5s} {'T err':>9s} {'G err':>9s} {'Au err':>9s}")
    print("-" * 86)
    pass_count = 0
    leaderboard: List[Dict[str, Any]] = []
    for result in sorted(results, key=lambda r: r.get("project_name") or ""):
        if result.get("error"):
            print(f"{(result.get('project_name') or result.get('project_id'))[:54]:54s} {'ERR':>5s}")
            continue
        passed = _core_pass(result["errors"], args.threshold)
        pass_count += int(passed)
        failure = result.get("failure_class") or {}
        leaderboard.append(
            leaderboard_row(
                project_name=result["project_name"],
                errors=result["errors"],
                passed=passed,
                failure=failure,
                guards=result.get("local_guards"),
            )
        )
        print(
            f"{result['project_name'][:54]:54s} "
            f"{'YES' if passed else 'NO':>5s} "
            f"{fmt_pct(result['errors']['tonnage']):>9s} "
            f"{fmt_pct(result['errors']['grade']):>9s} "
            f"{fmt_pct(result['errors']['contained']):>9s}"
        )
        if not passed:
            print(f"{'':54s} {'':>5s} lesson: {failure.get('class')} — {failure.get('lesson')}")

    print("-" * 86)
    print(f"95% core matches: {pass_count}/{sum(1 for r in results if not r.get('error'))}")
    print(f"batch_id: {batch_id}")
    if args.leaderboard_json:
        payload = {
            "batch_id": batch_id,
            "threshold": args.threshold,
            "min_pass_count": args.min_pass_count,
            "pass_count": pass_count,
            "evaluated_count": sum(1 for r in results if not r.get("error")),
            "target_selection": {
                "random_targets": args.random_targets,
                "random_seed": args.random_seed,
                "excluded_project_ids": sorted(excluded_project_ids),
                "excluded_run_ids": args.exclude_run_id,
                "project_ids": [target["project_id"] for target in targets],
            },
            "leaderboard": leaderboard,
        }
        out_path = Path(args.leaderboard_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"leaderboard_json: {out_path}")
    return 0 if pass_count >= args.min_pass_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
