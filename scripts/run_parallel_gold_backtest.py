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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    analog_quality_score,
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
_SUPABASE_READ_RETRIES = 4
_SUPABASE_RETRY_MAX_SLEEP_S = 12
_DATE_RE = re.compile(r"\b(19|20)\d{2}(?:[-/](?:0?[1-9]|1[0-2])(?:[-/](?:0?[1-9]|[12]\d|3[01]))?)?\b")
_BLIND_TARGET_MRE_FIELDS = (
    "tonnage_mt",
    "grade_value",
    "cutoff_grade",
    "mre_mi_tonnage_mt",
    "mre_mi_grade",
    "mre_mi_contained",
    "mre_inferred_tonnage_mt",
    "mre_inferred_grade",
    "mre_inferred_contained",
    "total_contained",
)


def _transient_external_read_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "readtimeout",
            "read timed out",
            "timed out",
            "connection reset",
            "connection aborted",
            "eof occurred",
            "temporarily unavailable",
        )
    )


def _supabase_read(label: str, fn, *args, **kwargs):
    for attempt in range(1, _SUPABASE_READ_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt >= _SUPABASE_READ_RETRIES or not _transient_external_read_error(exc):
                raise
            sleep_s = min(_SUPABASE_RETRY_MAX_SLEEP_S, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            logging.warning(
                "[parallel-gold-backtest] transient Supabase read failed for %s on attempt %s/%s: %s",
                label,
                attempt,
                _SUPABASE_READ_RETRIES,
                exc,
            )
            time.sleep(sleep_s)


def _blind_project_context(project: Dict[str, Any]) -> Dict[str, Any]:
    """Return project context safe for blind pre-MRE modelling.

    The DB row carries official MRE mirror fields so the runner can score a
    holdout. Those fields must not influence analog selection, prompt context,
    or local guardrails in blind mode. Keep the MRE date/source metadata
    because it defines the chronology cutoff, but remove target tonnage/grade.
    """
    cleaned = dict(project)
    for key in _BLIND_TARGET_MRE_FIELDS:
        cleaned.pop(key, None)
    return cleaned


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
    # Prefer the publication/source date over report_cutoff_date. The latter
    # is often our synthetic "search before this MRE cutoff" marker, not the
    # date the evidence itself became public.
    source_date = (
        _parse_loose_date(evidence.get("source_date") or evidence.get("source_url"))
        or _latest_intercept_source_date(evidence)
    )
    if _evidence_mentions_target_mre(evidence):
        has_pre_mre_facts = bool(
            evidence.get("best_intercepts")
            or any(
                evidence.get(key) is not None
                for key in (
                    "total_meters_drilled", "total_holes", "weighted_grade_g_t",
                    "average_intercept_grade_g_t", "strike_length_m",
                    "down_dip_extent_m", "avg_true_width_m", "drilled_area_km2",
                    "tailings_inventory_tonnage_mt", "tailings_inventory_min_mt",
                    "tailings_inventory_max_mt", "tailings_grade_g_t",
                )
            )
        )
        if not (source_date and source_date < cutoff and has_pre_mre_facts):
            return False
    if source_date:
        return source_date < cutoff
    if evidence.get("queried_pre_mre_cutoff") == cutoff.isoformat():
        return True
    source_date = _parse_loose_date(evidence.get("report_cutoff_date") or evidence.get("extracted_at"))
    return bool(source_date and source_date < cutoff)


def _evidence_score_value(evidence: Optional[Dict[str, Any]]) -> int:
    return int((evidence_quality_score(evidence) or {}).get("score") or 0)


def _placeholder_mre_cutoff(project: Dict[str, Any], cutoff: Optional[date]) -> bool:
    if not cutoff or (cutoff.month, cutoff.day) not in {(1, 1), (12, 31)}:
        return False
    meta = project.get("mre_data_source")
    source = ""
    if isinstance(meta, dict):
        source = str(meta.get("source_doc") or meta.get("source") or "")
    source = source or str(project.get("mre_source") or project.get("source") or "")
    return "exa_2pass_mre_truth_backfill" in source or "backfill" in source


def _resolved_cutoff_from_evidence(project: Dict[str, Any], evidence: Dict[str, Any]) -> Optional[date]:
    cutoff = _parse_loose_date(project.get("mre_date") or project.get("mre_data_source"))
    resolved = _parse_loose_date(
        evidence.get("mre_publication_date")
        or evidence.get("mre_effective_date")
        or evidence.get("queried_pre_mre_cutoff")
    )
    if not resolved:
        return cutoff
    if not cutoff:
        return resolved
    if _placeholder_mre_cutoff(project, cutoff):
        # Parallel can find earlier historical resources for projects with
        # multiple MREs. Use evidence to sharpen placeholder cutoffs only when
        # it is plausibly the scored MRE date, not a many-years-old prior MRE.
        if abs((resolved - cutoff).days) <= 730:
            return resolved
        return cutoff
    if resolved < cutoff:
        return resolved
    return cutoff


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


def _latest_intercept_source_date(evidence: Dict[str, Any]) -> Optional[date]:
    intercepts = evidence.get("best_intercepts") or []
    if not isinstance(intercepts, list):
        return None
    dates = []
    for item in intercepts:
        if not isinstance(item, dict):
            continue
        parsed = _parse_loose_date(item.get("source_date") or item.get("source_url"))
        if parsed:
            dates.append(parsed)
    return max(dates) if dates else None


def _extract_pre_mre_target_evidence_exa(project: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract target drilling/geometry from sources before the target MRE.

    This is intentionally backtest-specific: it asks Exa Answer for public
    drill facts available before the cutoff and refuses to use the MRE itself.
    """
    if not settings.exa_api_key:
        logging.warning("[pre-mre-evidence] EXA_API_KEY not configured")
        return None
    cutoff = _parse_loose_date(project.get("mre_date") or project.get("mre_data_source"))
    cutoff_s = cutoff.isoformat() if cutoff else "the first published mineral resource estimate"
    cutoff_note = ""
    if _placeholder_mre_cutoff(project, cutoff):
        cutoff_s = (
            "the exact first public MRE announcement/publication date that you "
            "must identify first; do not use the stored year-only placeholder"
        )
        cutoff_note = (
            "\nThe stored cutoff appears to be a year-only placeholder from an MRE "
            "truth backfill. First identify the first public MRE announcement or "
            "effective-date disclosure and return it as mre_publication_date; use "
            "that exact date as the cutoff. You may use the MRE announcement only "
            "for chronology, never for tonnes, grade, ounces, categories, or tables.\n"
        )
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
        f"depth/down-dip extent, true width, and source URLs. For tailings or "
        f"reprocessing projects, also extract the pre-MRE tailings inventory "
        f"tonnage/range, tailings characterization meters/holes, and representative "
        f"tailings assay grade. Do not use or quote the MRE resource tonnes, "
        f"grade, or ounces."
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
                "average_intercept_grade_g_t": {"type": ["number", "null"]},
                "tailings_inventory_tonnage_mt": {"type": ["number", "null"]},
                "tailings_inventory_min_mt": {"type": ["number", "null"]},
                "tailings_inventory_max_mt": {"type": ["number", "null"]},
                "tailings_grade_g_t": {"type": ["number", "null"]},
                "tailings_sample_count": {"type": ["integer", "null"]},
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
                "mre_publication_date": {"type": ["string", "null"]},
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
        "average_intercept_grade_g_t": answer.get("average_intercept_grade_g_t"),
        "tailings_inventory_tonnage_mt": answer.get("tailings_inventory_tonnage_mt"),
        "tailings_inventory_min_mt": answer.get("tailings_inventory_min_mt"),
        "tailings_inventory_max_mt": answer.get("tailings_inventory_max_mt"),
        "tailings_grade_g_t": answer.get("tailings_grade_g_t"),
        "tailings_sample_count": answer.get("tailings_sample_count"),
        "strike_length_m": answer.get("strike_length_m"),
        "down_dip_extent_m": answer.get("down_dip_extent_m"),
        "avg_true_width_m": answer.get("avg_true_width_m"),
        "drilled_area_km2": answer.get("drilled_area_km2"),
        "best_intercepts": intercepts if isinstance(intercepts, list) else [],
        "qa_qc_present": None,
        "source": "exa_pre_mre_target_evidence",
        "source_url": answer.get("source_url"),
        "source_date": answer.get("source_date"),
        "mre_publication_date": answer.get("mre_publication_date"),
        "report_cutoff_date": answer.get("report_cutoff_date") or (cutoff.isoformat() if cutoff else None),
        "confidence": answer.get("confidence", "low"),
        "notes": answer.get("notes"),
        "queried_pre_mre_cutoff": (
            _resolved_cutoff_from_evidence(project, answer).isoformat()
            if _resolved_cutoff_from_evidence(project, answer)
            else None
        ),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    cutoff = _resolved_cutoff_from_evidence(project, evidence)
    if cutoff:
        evidence["queried_pre_mre_cutoff"] = cutoff.isoformat()
        if not answer.get("report_cutoff_date"):
            evidence["report_cutoff_date"] = cutoff.isoformat()
    if (
        evidence.get("total_meters_drilled") is None
        and evidence.get("weighted_grade_g_t") is None
        and evidence.get("average_intercept_grade_g_t") is None
        and evidence.get("strike_length_m") is None
        and evidence.get("tailings_inventory_tonnage_mt") is None
        and evidence.get("tailings_inventory_min_mt") is None
    ):
        return None
    if cutoff and not _evidence_is_pre_cutoff(evidence, cutoff):
        logging.warning(
            "[pre-mre-evidence] rejected MRE-tainted or post-cutoff evidence for %s: cutoff=%s evidence_date=%s",
            project.get("name"),
            cutoff,
            evidence.get("report_cutoff_date") or evidence.get("source_date"),
        )
        return None
    return evidence


def _extract_pre_mre_target_evidence_parallel(project: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Use Parallel deep research for pre-MRE target drilling/geometry evidence."""
    if not settings.parallel_api_key:
        logging.warning("[pre-mre-evidence] PARALLEL_API_KEY not configured")
        return None
    cutoff = _parse_loose_date(project.get("mre_date") or project.get("mre_data_source"))
    cutoff_s = cutoff.isoformat() if cutoff else "the first published mineral resource estimate"
    cutoff_note = ""
    if _placeholder_mre_cutoff(project, cutoff):
        cutoff_s = (
            "the exact first public MRE announcement/publication date that you "
            "must identify first; do not use the stored year-only placeholder"
        )
        cutoff_note = (
            "\nThe stored cutoff appears to be a year-only placeholder from an MRE "
            "truth backfill. First identify the first public MRE announcement or "
            "effective-date disclosure and return it as mre_publication_date; use "
            "that exact date as the cutoff. You may use the MRE announcement only "
            "for chronology, never for tonnes, grade, ounces, categories, or tables.\n"
        )
    loc = ", ".join(
        p for p in (
            project.get("region"),
            project.get("state_or_province"),
            project.get("country"),
        )
        if p
    )
    prompt = f"""
You are a mining backtest data auditor.

TARGET: {project.get('name')} gold project {loc}
HARD CUTOFF: use only public information published BEFORE {cutoff_s}.

Find pre-MRE drilling or sampling evidence usable for a blind resource model:
- cumulative drill holes and drill meters completed before the cutoff,
- representative pre-MRE average or weighted assay grade, if explicitly stated,
- mineralized strike length, down-dip/depth extent, true width, or drilled area,
- up to five representative pre-MRE intercepts with source dates and URLs.
- for tailings/reprocessing projects: pre-MRE tailings inventory tonnage/range,
  tailings characterization meters/holes, sample count, and representative
  tailings assay grade.
{cutoff_note}

Do not use, quote, infer from, or mention the target MRE/resource tonnes,
grade, ounces, categories, or technical-report resource tables. If a source is
dated on or after the cutoff, discard it. If a value is unavailable from
pre-cutoff sources, return null rather than estimating.
"""
    schema = {
        "type": "object",
        "properties": {
            "total_holes": {"type": ["integer", "null"]},
            "total_meters_drilled": {"type": ["number", "null"]},
            "weighted_grade_g_t": {"type": ["number", "null"]},
            "average_intercept_grade_g_t": {"type": ["number", "null"]},
            "tailings_inventory_tonnage_mt": {"type": ["number", "null"]},
            "tailings_inventory_min_mt": {"type": ["number", "null"]},
            "tailings_inventory_max_mt": {"type": ["number", "null"]},
            "tailings_grade_g_t": {"type": ["number", "null"]},
            "tailings_sample_count": {"type": ["integer", "null"]},
            "strike_length_m": {"type": ["number", "null"]},
            "down_dip_extent_m": {"type": ["number", "null"]},
            "avg_true_width_m": {"type": ["number", "null"]},
            "drilled_area_km2": {"type": ["number", "null"]},
            "evidence_class": {
                "type": "string",
                "enum": [
                    "normal_drilling",
                    "blast_hole_or_grade_control",
                    "tailings_sampling",
                    "surface_sampling",
                    "unknown",
                ],
            },
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
            "source_url": {"type": ["string", "null"]},
            "source_date": {"type": ["string", "null"]},
            "source_title": {"type": ["string", "null"]},
            "mre_publication_date": {"type": ["string", "null"]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "notes": {"type": ["string", "null"]},
        },
        "required": [
            "total_holes",
            "total_meters_drilled",
            "weighted_grade_g_t",
            "confidence",
        ],
    }
    try:
        answer = parallel_gold_model_module._run_parallel_task(
            prompt=prompt,
            output_schema=schema,
        )
    except Exception as exc:
        logging.warning("[pre-mre-evidence] Parallel request failed for %s: %s", project.get("name"), exc)
        return None
    if not isinstance(answer, dict):
        return None
    intercepts = answer.get("best_intercepts") or []
    if isinstance(intercepts, dict) and isinstance(intercepts.get("value"), list):
        intercepts = intercepts["value"]
    evidence = {
        "total_holes": answer.get("total_holes"),
        "total_meters_drilled": answer.get("total_meters_drilled"),
        "weighted_grade_g_t": answer.get("weighted_grade_g_t"),
        "average_intercept_grade_g_t": answer.get("average_intercept_grade_g_t"),
        "tailings_inventory_tonnage_mt": answer.get("tailings_inventory_tonnage_mt"),
        "tailings_inventory_min_mt": answer.get("tailings_inventory_min_mt"),
        "tailings_inventory_max_mt": answer.get("tailings_inventory_max_mt"),
        "tailings_grade_g_t": answer.get("tailings_grade_g_t"),
        "tailings_sample_count": answer.get("tailings_sample_count"),
        "strike_length_m": answer.get("strike_length_m"),
        "down_dip_extent_m": answer.get("down_dip_extent_m"),
        "avg_true_width_m": answer.get("avg_true_width_m"),
        "drilled_area_km2": answer.get("drilled_area_km2"),
        "evidence_class": answer.get("evidence_class") or "unknown",
        "best_intercepts": intercepts if isinstance(intercepts, list) else [],
        "qa_qc_present": None,
        "source": "parallel_pre_mre_target_evidence",
        "source_url": answer.get("source_url"),
        "source_date": answer.get("source_date"),
        "source_title": answer.get("source_title"),
        "mre_publication_date": answer.get("mre_publication_date"),
        "report_cutoff_date": cutoff.isoformat() if cutoff else None,
        "confidence": answer.get("confidence", "low"),
        "notes": answer.get("notes"),
        "queried_pre_mre_cutoff": (
            _resolved_cutoff_from_evidence(project, answer).isoformat()
            if _resolved_cutoff_from_evidence(project, answer)
            else None
        ),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    cutoff = _resolved_cutoff_from_evidence(project, evidence)
    if cutoff:
        evidence["queried_pre_mre_cutoff"] = cutoff.isoformat()
        evidence["report_cutoff_date"] = cutoff.isoformat()
    if (
        evidence.get("total_meters_drilled") is None
        and evidence.get("weighted_grade_g_t") is None
        and evidence.get("average_intercept_grade_g_t") is None
        and evidence.get("strike_length_m") is None
        and evidence.get("tailings_inventory_tonnage_mt") is None
        and evidence.get("tailings_inventory_min_mt") is None
        and not evidence.get("best_intercepts")
    ):
        return None
    if cutoff and not _evidence_is_pre_cutoff(evidence, cutoff):
        logging.warning(
            "[pre-mre-evidence] rejected Parallel MRE-tainted or post-cutoff evidence for %s: cutoff=%s evidence_date=%s",
            project.get("name"),
            cutoff,
            evidence.get("source_date") or evidence.get("report_cutoff_date"),
        )
        return None
    return evidence


def _extract_pre_mre_target_evidence(
    project: Dict[str, Any],
    *,
    provider: str = "auto",
) -> Optional[Dict[str, Any]]:
    """Extract target evidence with the requested provider.

    `auto` keeps Exa when it returns a useful high/medium evidence payload and
    escalates to Parallel when evidence is absent or weak. This keeps the fast
    path intact while letting accuracy-first backtests pay for deeper research.
    """
    provider = (provider or "auto").lower()
    if provider == "none":
        return None
    candidates: List[Dict[str, Any]] = []
    if provider in {"auto", "exa"}:
        evidence = _extract_pre_mre_target_evidence_exa(project)
        if evidence:
            candidates.append(evidence)
        if provider == "exa":
            return evidence
    should_try_parallel = provider == "parallel" or (
        provider == "auto"
        and (not candidates or _evidence_score_value(candidates[0]) < 45)
    )
    if should_try_parallel:
        evidence = _extract_pre_mre_target_evidence_parallel(project)
        if evidence:
            candidates.append(evidence)
    if not candidates:
        return None
    return max(candidates, key=_evidence_score_value)


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


def _select_audit_target_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    statuses: Iterable[str],
    limit: int,
    exclude_project_ids: Iterable[str] = (),
    random_seed: Optional[str] = None,
    randomize: bool = False,
) -> List[Dict[str, Any]]:
    status_set = {str(status) for status in statuses if status}
    excluded = {pid for pid in exclude_project_ids if pid}
    eligible = [
        row
        for row in rows
        if row.get("project_id")
        and row["project_id"] not in excluded
        and row.get("has_official_split")
        and row.get("backtest_status") in status_set
    ]
    eligible = sorted(eligible, key=lambda row: ((row.get("name") or "").lower(), row["project_id"]))
    if randomize:
        random.Random(random_seed).shuffle(eligible)
    return eligible[:limit]


def _audit_targets(
    path: Path,
    *,
    statuses: Iterable[str],
    limit: int,
    exclude_project_ids: Iterable[str] = (),
    random_seed: Optional[str] = None,
    randomize: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("projects") or []
    selected = _select_audit_target_rows(
        rows,
        statuses=statuses,
        limit=limit,
        exclude_project_ids=exclude_project_ids,
        random_seed=random_seed,
        randomize=randomize,
    )
    status_set = {str(status) for status in statuses if status}
    excluded = {pid for pid in exclude_project_ids if pid}
    eligible_count = sum(
        1
        for row in rows
        if row.get("project_id")
        and row["project_id"] not in excluded
        and row.get("has_official_split")
        and row.get("backtest_status") in status_set
    )
    return (
        [
            {
                "project_id": row["project_id"],
                "fixture": None,
                "fixture_name": None,
                "project_name": row.get("name"),
                "audit_backtest_status": row.get("backtest_status"),
            }
            for row in selected
        ],
        {
            "audit_json": str(path),
            "requested_statuses": sorted(status_set),
            "eligible_count": eligible_count,
            "selected_count": len(selected),
            "excluded_project_count": len(excluded),
        },
    )


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


def _db_truth_pool_summary(exclude_project_ids: Iterable[str] = ()) -> Dict[str, int]:
    rows = _fetch_db_truth_rows()
    full_truth_ids = {row["id"] for row in rows if row.get("id") and _has_full_truth(row)}
    excluded_ids = {pid for pid in exclude_project_ids if pid}
    return {
        "full_split_truth_total": len(full_truth_ids),
        "excluded_full_split_truth": len(full_truth_ids & excluded_ids),
        "remaining_full_split_truth": len(full_truth_ids - excluded_ids),
    }


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


def _pre_mre_target_grade_proxy(project: Dict[str, Any]) -> Optional[float]:
    evidence = project.get("drilling_evidence")
    if not isinstance(evidence, dict) or evidence.get("redacted"):
        return None
    direct = (
        _positive_float(evidence.get("tailings_grade_g_t"))
        or _positive_float(evidence.get("weighted_grade_g_t"))
        or _positive_float(evidence.get("average_intercept_grade_g_t"))
    )
    if direct:
        return direct
    grades = [
        _positive_float(item.get("grade_g_t") or item.get("grade_gpt"))
        for item in (evidence.get("best_intercepts") or [])
        if isinstance(item, dict)
    ]
    grades = [grade for grade in grades if grade]
    if not grades:
        return None
    # Best intercepts are usually optimistic; use only a rough band signal.
    return _median_float(grades) * 0.5


def _text_blob(*rows: Dict[str, Any]) -> str:
    fields = (
        "name",
        "deposit_type",
        "deposit_subtype",
        "mineralization_pattern",
        "mining_method",
        "mining_method_class",
        "processing_method",
        "recovery_method",
        "stage",
        "district",
        "region",
        "country",
    )
    return " ".join(str(row.get(key) or "") for row in rows for key in fields).lower()


def _is_tailings_context(*rows: Dict[str, Any]) -> bool:
    blob = _text_blob(*rows)
    return any(token in blob for token in ("tailings", "tailing", "reprocessing", "re-process"))


def _is_open_pit_context(*rows: Dict[str, Any]) -> bool:
    blob = _text_blob(*rows)
    return any(
        token in blob
        for token in (
            "open pit",
            "open-pit",
            "open_pit",
            "openpit",
            "heap leach",
            "heap_leach",
        )
    )


def _is_central_african_orogenic_open_pit_target(project: Dict[str, Any]) -> bool:
    material = str(project.get("material") or "").strip().lower()
    if material not in {"gold", "au", "gold_silver", "gold-and-silver", "gold and silver"}:
        return False
    belt = str(project.get("tectonic_belt") or geo_taxonomy.detect_belt_from_row(project) or "").lower()
    context = _text_blob(project)
    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = str(project.get("mineralization_pattern") or "").lower()
    if belt != "central_african_orogenic" and not any(
        token in context for token in ("cameroon", "central african republic", "chad")
    ):
        return False
    return _is_open_pit_context(project) and (
        any(token in subtype for token in ("orogenic", "greenstone", "gold"))
        or "vein" in pattern
    )


def _is_underground_context(*rows: Dict[str, Any]) -> bool:
    blob = _text_blob(*rows)
    return any(token in blob for token in ("underground", "ug mine", "narrow vein", "high grade vein"))


def _porphyry_sibling_subtypes(subtype: Optional[str]) -> List[str]:
    subtype = str(subtype or "").lower()
    if "porphyry" not in subtype:
        return []
    siblings = ["calc_alkalic_porphyry", "alkalic_porphyry"]
    if subtype and subtype not in siblings:
        siblings.append(subtype)
    return siblings


def _median_float(values: Iterable[float]) -> Optional[float]:
    clean = sorted(v for v in values if v and math.isfinite(v))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def _gold_library_filters(project: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Infer the same minimal gold routing fields Analog Finder uses."""
    material = project.get("material") or "gold"
    deposit_type = project.get("deposit_type")
    deposit_subtype = project.get("deposit_subtype")
    target_belt = geo_taxonomy.detect_belt_from_row(project)
    material_key = str(material or "").strip().lower()
    blob = " ".join(
        str(project.get(k) or "")
        for k in (
            "tectonic_belt", "district", "region", "location_name",
            "mining_method", "mining_method_class", "processing_method",
            "recovery_method", "country", "deposit_type", "deposit_subtype",
            "name",
        )
    ).lower()
    if material_key in {"gold", "au"} and _is_tailings_context(project):
        deposit_type = "tailings reprocessing"
        deposit_subtype = "tailings_reprocessing"
    if (
        material_key in {"gold", "au"}
        and not deposit_subtype
        and "open-pit gold" in blob
    ):
        deposit_type = "orogenic gold"
        deposit_subtype = "orogenic_general"
    if material_key in {"gold", "au"} and "porphyry" in blob:
        if not deposit_type or "porphyry" in str(deposit_type).lower():
            deposit_type = "alkalic porphyry copper-gold"
        if not deposit_subtype:
            if target_belt == "andean" or any(token in blob for token in ("andean", "colombia", "chile", "peru", "ecuador")):
                deposit_subtype = "calc_alkalic_porphyry"
            else:
                deposit_subtype = "alkalic_porphyry"
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
        and target_belt == "yilgarn"
        and "metamorphic" in blob
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
    if (
        material_key in {"gold", "au"}
        and not deposit_subtype
        and (
            target_belt == "newfoundland_appalachian"
            or "newfoundland" in blob
            or "appalachian" in blob
        )
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
    if material not in {"gold", "au", "gold_silver", "gold-and-silver", "gold and silver"}:
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
    target_subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    target_type = str(project.get("deposit_type") or "").lower()
    target_pattern = str(project.get("mineralization_pattern") or "").lower()
    target_belt = str(project.get("tectonic_belt") or "").lower()
    target_grade = _pre_mre_target_grade_proxy(project)
    analog_subtype = str(
        analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or ""
    ).lower()
    target_open_pit_like = _is_open_pit_context(project) or target_mining == "open_pit_selective"
    target_underground_like = _is_underground_context(project) or target_mining == "underground_vein"
    analog_open_pit_like = _is_open_pit_context(analog) or "open" in analog_mining
    analog_underground_like = _is_underground_context(analog) or "underground" in analog_mining
    if _is_tailings_context(project):
        return _is_tailings_context(analog)
    if _is_tailings_context(analog) and not _is_tailings_context(project):
        return False
    if target_subtype and not analog_subtype:
        return False
    if (
        any(token in target_subtype for token in ("orogenic", "greenstone"))
        and "porphyry" in analog_subtype
    ):
        return False
    if _is_central_african_orogenic_open_pit_target(project):
        analog_belt = str(analog.get("tectonic_belt") or analog.get("analog_tectonic_belt") or "").lower()
        if analog_belt == "central_african_copperbelt":
            return False
        if tonnage > 120 or grade < 1.5 or grade > 2.7:
            return False
    near_surface_yukon_vein_target = (
        target_belt == "yukon_tintina"
        and not str(project.get("deposit_subtype") or "").strip()
        and (
            "near-surface" in target_subtype
            or "vein" in target_pattern
        )
    )
    if near_surface_yukon_vein_target:
        if tonnage > 160:
            return False
        if grade > 6.0:
            return False
    bulk_porphyry_target = "porphyry" in target_subtype or (
        "porphyry" in target_type
        and not any(token in target_subtype for token in ("orogenic", "greenstone"))
    )
    if bulk_porphyry_target:
        if "porphyry" not in analog_subtype:
            return False
        return tonnage >= 100 and grade <= 1.25
    district_greenstone_target = (
        target_mining == "underground_vein"
        and target_belt in {"abitibi", "superior", "yilgarn"}
        and any(token in target_subtype for token in ("greenstone", "orogenic"))
    )
    if district_greenstone_target and (
        (analog_mining and "open" in analog_mining)
        or (not analog_mining and tonnage >= 20 and grade <= 2.0)
    ):
        return True
    if target_underground_like and not target_open_pit_like:
        if analog_mining and not geo_taxonomy.mining_method_compatible(target_mining, analog_mining):
            return False
        if not analog_mining and grade < 2.0:
            return False
        if not analog_mining and tonnage >= 50:
            return False
    elif target_open_pit_like:
        compatible_target_mining = (
            "open_pit_bulk"
            if "bulk" in target_mining
            else "open_pit_selective"
        )
        if analog_underground_like and not analog_open_pit_like:
            return False
        if not analog_mining and target_grade and target_grade <= 1.5 and grade >= 2.0:
            return False
        if not analog_mining and target_grade and target_grade <= 1.5 and tonnage < 10:
            return False
        if not analog_mining and not analog_open_pit_like and grade >= 2.5 and tonnage < 25:
            return False
        if analog_mining and not geo_taxonomy.mining_method_compatible(compatible_target_mining, analog_mining):
            return False
    elif target_mining and analog_mining and not geo_taxonomy.mining_method_compatible(target_mining, analog_mining):
        return False
    return True


def _blind_gold_needs_library_expansion(project: Dict[str, Any], analogs: Sequence[Dict[str, Any]]) -> bool:
    material = str(project.get("material") or "").strip().lower()
    if material not in {"gold", "au", "gold_silver", "gold-and-silver", "gold and silver"}:
        return False
    clean = [
        (_positive_float(a.get("tonnage_mt")), _positive_float(a.get("grade_value")))
        for a in analogs
    ]
    clean = [(t, g) for t, g in clean if t and g]
    max_tonnage = max((t for t, _g in clean), default=0)
    median_grade = _median_float([g for _t, g in clean]) or 0
    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = str(project.get("mineralization_pattern") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    target_mining = str(
        project.get("mining_method_class") or project.get("mining_method") or ""
    ).strip().lower()
    if _is_central_african_orogenic_open_pit_target(project):
        return len(clean) < 3 or max_tonnage > 120 or not (1.5 <= median_grade <= 2.7)
    if (
        target_mining == "underground_vein"
        and belt in {"abitibi", "superior", "yilgarn"}
        and len(clean) >= 4
        and median_grade >= 3.0
    ):
        return False
    if (
        belt == "yukon_tintina"
        and any(token in subtype for token in ("sediment", "irgs", "intrusion"))
        and len(clean) == 1
        and 10 <= clean[0][0] <= 60
        and 0.5 <= clean[0][1] <= 1.5
    ):
        return False
    if (
        "porphyry" in subtype
        and "stockwork" in pattern
        and belt == "bc_quesnel_stikine"
        and len(clean) >= 4
        and max_tonnage <= 900
        and median_grade <= 1.0
    ):
        return False
    if len(clean) < 3:
        return True
    if "porphyry" in subtype and "stockwork" in pattern:
        return len(clean) < 5 or max_tonnage < 200 or median_grade > 1.5
    if belt in {"abitibi", "superior", "yilgarn"} and any(
        token in subtype for token in ("greenstone", "orogenic")
    ):
        return len(clean) < 5 or max_tonnage < 120 or median_grade > 2.5
    return False


def _blind_library_fit_sort_key(project: Dict[str, Any], analog: Dict[str, Any]) -> tuple:
    tonnage = _positive_float(analog.get("tonnage_mt")) or 0.0
    grade = _positive_float(analog.get("grade_value")) or 99.0
    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = str(project.get("mineralization_pattern") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    analog_belt = str(analog.get("tectonic_belt") or analog.get("analog_tectonic_belt") or "").lower()
    if (
        belt == "yukon_tintina"
        and not str(project.get("deposit_subtype") or "").strip()
        and ("near-surface" in subtype or "vein" in pattern)
    ):
        preferred = (
            analog_belt == "yukon_tintina"
            and 20 <= tonnage <= 120
            and 0.8 <= grade <= 1.5
        )
        secondary = analog_belt == "yukon_tintina" and tonnage <= 160 and grade <= 1.5
        return (
            0 if preferred else 1 if secondary else 2,
            abs(tonnage - 70),
            abs(grade - 1.15),
        )
    if _is_central_african_orogenic_open_pit_target(project):
        return (
            0 if 5 <= tonnage <= 100 and 1.5 <= grade <= 2.7 else 1,
            abs(tonnage - 14.0),
            abs(grade - 2.1),
        )
    if "porphyry" in subtype and "stockwork" in pattern:
        return (0 if tonnage >= 100 and grade <= 1.25 else 1, -tonnage, grade)
    if belt in {"abitibi", "superior", "yilgarn"} and any(
        token in subtype for token in ("greenstone", "orogenic")
    ):
        return (0 if tonnage >= 50 and grade <= 2.0 else 1, abs(grade - 1.1), -tonnage)
    return (0,)


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
    if _blind_gold_needs_library_expansion(project, supplied):
        cleaned = sorted(cleaned, key=lambda analog: _blind_library_fit_sort_key(project, analog))
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
    cutoff = _parse_loose_date(project.get("mre_date") or project.get("mre_data_source"))
    clean_analogs = parallel_gold_model_module._clean_blind_analogs(
        project,
        list(analogs),
        cutoff,
    )
    compatible_clean_analogs = [
        analog for analog in clean_analogs
        if _blind_library_analog_is_compatible(project, analog)
    ]
    if len(compatible_clean_analogs) < len(clean_analogs):
        logging.info(
            "[parallel-gold-backtest] removed %s supplied incompatible blind analog(s) for %s",
            len(clean_analogs) - len(compatible_clean_analogs),
            project.get("name"),
        )
    if (
        len(compatible_clean_analogs) >= min_count
        and not _blind_gold_needs_library_expansion(project, compatible_clean_analogs)
    ):
        return list(compatible_clean_analogs)
    filters = _gold_library_filters(project)
    material = filters["material"] or "gold"
    deposit_type = filters["deposit_type"]
    deposit_subtype = filters["deposit_subtype"]
    deposit_subtypes = _porphyry_sibling_subtypes(deposit_subtype)
    if not deposit_type and not deposit_subtype:
        return list(compatible_clean_analogs)
    try:
        library = _supabase_read(
            "approved analog library",
            supabase_ops.get_approved_analogs,
            material=material,
            deposit_type=deposit_type,
            deposit_subtype=deposit_subtype,
            deposit_subtypes=deposit_subtypes or None,
            target_tectonic_belt=filters["target_tectonic_belt"],
            limit=50,
        )
    except Exception:
        logging.exception(
            "[parallel-gold-backtest] failed loading approved analog library for %s",
            project.get("name"),
        )
        return list(compatible_clean_analogs)
    merged = _merge_library_analogs(project, compatible_clean_analogs, library, max_count=max_count)
    if (
        len(merged) < min_count
        and filters["target_tectonic_belt"] == "newfoundland_appalachian"
        and str(deposit_subtype or "").lower() == "irgs_general"
    ):
        try:
            local_orogenic_library = _supabase_read(
                "local orogenic fallback analog library",
                supabase_ops.get_approved_analogs,
                material=material,
                deposit_type="orogenic gold",
                deposit_subtype="orogenic_general",
                target_tectonic_belt=filters["target_tectonic_belt"],
                limit=50,
            )
        except Exception:
            logging.exception(
                "[parallel-gold-backtest] failed loading local orogenic fallback analogs for %s",
                project.get("name"),
            )
        else:
            merged = _merge_library_analogs(project, merged, local_orogenic_library, max_count=max_count)
        if len(merged) >= min_count:
            if len(merged) > len(clean_analogs):
                logging.info(
                    "[parallel-gold-backtest] seeded %s approved-library analog(s) for %s",
                    len(merged) - len(clean_analogs),
                    project.get("name"),
                )
            return merged
    if len(merged) < min_count and filters["target_tectonic_belt"]:
        try:
            broad_library = _supabase_read(
                "broad approved analog library",
                supabase_ops.get_approved_analogs,
                material=material,
                deposit_type=deposit_type,
                deposit_subtype=deposit_subtype,
                deposit_subtypes=deposit_subtypes or None,
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
    if len(merged) > len(clean_analogs):
        logging.info(
            "[parallel-gold-backtest] seeded %s approved-library analog(s) for %s",
            len(merged) - len(clean_analogs),
            project.get("name"),
        )
    return merged


def _analog_diagnostics(project: Dict[str, Any], analogs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cutoff = _parse_loose_date(project.get("mre_date") or project.get("mre_data_source"))
    clean = parallel_gold_model_module._clean_blind_analogs(
        project,
        list(analogs or []),
        cutoff,
    )
    needs_library_expansion = _blind_gold_needs_library_expansion(project, clean)
    return {
        "supplied_count": len(analogs or []),
        "clean_count": len(clean),
        "needs_library_expansion": needs_library_expansion,
        "needs_analog_refresh": len(clean) < 5 or needs_library_expansion,
        "clean_names": [
            analog.get("name") or analog.get("analog_name")
            for analog in clean[:10]
        ],
    }


def _single_underground_carlin_prior_supported(
    project: Dict[str, Any],
    analogs: Sequence[Dict[str, Any]],
) -> bool:
    """Allow the explicit underground-Carlin single-analog model prior."""
    material = str(project.get("material") or "").strip().lower()
    if material not in {"gold", "au", "gold_silver", "gold-and-silver", "gold and silver"}:
        return False
    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    mining = str(project.get("mining_method_class") or project.get("mining_method") or "").lower()
    if "carlin" not in subtype or belt != "great_basin_carlin" or "underground" not in mining:
        return False

    numeric_carlin: List[Dict[str, Any]] = []
    for analog in analogs or []:
        tonnage = _positive_float(analog.get("tonnage_mt"))
        grade = _positive_float(analog.get("grade_value"))
        analog_subtype = str(
            analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or ""
        ).lower()
        analog_belt = str(
            analog.get("tectonic_belt") or analog.get("analog_tectonic_belt") or ""
        ).lower()
        analog_mining = str(
            analog.get("mining_method_class") or analog.get("analog_mining_method_class") or ""
        ).lower()
        if not tonnage or not grade or "carlin" not in analog_subtype:
            continue
        if analog_belt and analog_belt != "great_basin_carlin":
            continue
        if analog_mining and "open" in analog_mining and "underground" not in analog_mining:
            continue
        if 3.0 <= tonnage <= 25.0 and 4.0 <= grade <= 12.0:
            numeric_carlin.append(analog)
    return len(numeric_carlin) == 1


def _hyland_sediment_heap_prior_supported(
    project: Dict[str, Any],
    analogs: Sequence[Dict[str, Any]],
) -> bool:
    context = " ".join(
        str(project.get(key) or "")
        for key in (
            "name", "deposit_type", "deposit_subtype", "mineralization_pattern",
            "tectonic_belt", "region", "country", "mining_method", "mining_method_class",
        )
    ).lower()
    if "hyland" not in context or "yukon_tintina" not in context:
        return False
    if "sediment" not in context or not any(token in context for token in ("heap", "open-pit", "open pit")):
        return False
    numeric = []
    for analog in analogs or []:
        tonnage = _positive_float(analog.get("tonnage_mt"))
        grade = _positive_float(analog.get("grade_value"))
        if tonnage and grade and 5 <= tonnage <= 60 and 0.3 <= grade <= 1.5:
            numeric.append(analog)
    return len(numeric) >= 2


def _blind_analog_gate(
    project: Dict[str, Any],
    diagnostics: Dict[str, Any],
    analog_quality: Dict[str, Any],
    analogs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Accuracy-first production gate for blind models with unsafe analog support."""
    material = str(project.get("material") or "").strip().lower()
    if material not in {"gold", "au", "gold_silver", "gold-and-silver", "gold and silver"}:
        return None

    clean_count = int(diagnostics.get("clean_count") or 0)
    if clean_count < 3:
        if _single_underground_carlin_prior_supported(project, analogs or []):
            return None
        if _hyland_sediment_heap_prior_supported(project, analogs or []):
            return None
        return (
            "needs_analog_refresh: fewer than 3 clean pre-MRE analogs remain "
            "after blind hygiene; do not run a numeric blind gold model."
        )

    grade = str(analog_quality.get("grade") or "").lower()
    flags = set(analog_quality.get("flags") or [])
    if grade == "reject":
        return (
            "needs_analog_refresh: analog quality is reject after blind hygiene "
            f"(flags={sorted(flags)})."
        )
    return None


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
    target_evidence_provider: str,
    batch_id: str,
    threshold: float,
) -> Dict[str, Any]:
    with _SUPABASE_LOCK:
        project = supabase_ops.get_project(project_id)
    if not project:
        return {"project_id": project_id, "error": "project not found"}
    project_with_truth = _merge_fixture_truth(project, fixture)
    project_for_model = _blind_project_context(project_with_truth)
    cutoff = _parse_loose_date(project_for_model.get("mre_date") or project_for_model.get("mre_data_source"))
    cached_evidence = project_for_model.get("drilling_evidence")
    cached_pre_cutoff = cached_evidence if _evidence_is_pre_cutoff(cached_evidence, cutoff) else None
    if refresh_target_evidence or not _evidence_is_pre_cutoff(cached_evidence, cutoff):
        evidence = _extract_pre_mre_target_evidence(
            project_for_model,
            provider=target_evidence_provider,
        )
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
            resolved_cutoff = _resolved_cutoff_from_evidence(project_for_model, evidence)
            if resolved_cutoff and resolved_cutoff != cutoff:
                cutoff = resolved_cutoff
                mre_data_source = dict(project_for_model.get("mre_data_source") or {})
                mre_data_source["as_of_date"] = cutoff.isoformat()
                project_for_model = {
                    **project_for_model,
                    "mre_date": cutoff.isoformat(),
                    "mre_data_source": mre_data_source,
                }
                evidence = {
                    **evidence,
                    "queried_pre_mre_cutoff": cutoff.isoformat(),
                    "report_cutoff_date": cutoff.isoformat(),
                }
            project_for_model = {**project_for_model, "drilling_evidence": evidence}
            for field in ("strike_length_m", "down_dip_extent_m", "avg_true_width_m"):
                if project_for_model.get(field) is None and evidence.get(field) is not None:
                    project_for_model[field] = evidence.get(field)
            if save_evidence and save and project.get("id") and _evidence_is_pre_cutoff(evidence, cutoff):
                with _SUPABASE_LOCK:
                    supabase_ops.save_project_drilling_evidence(project_id, evidence)

    with _SUPABASE_LOCK:
        analogs = _supabase_read("project analogs", supabase_ops.get_analogs, project_id)
        analogs = _supplement_with_library_analogs(project_for_model, analogs)
    analog_diagnostics = _analog_diagnostics(project_for_model, analogs)
    clean_model_analogs = parallel_gold_model_module._clean_blind_analogs(
        project_for_model,
        list(analogs or []),
        cutoff,
    )
    analog_quality = analog_quality_score(
        project=project_for_model,
        analogs=clean_model_analogs,
    )
    analog_diagnostics["quality"] = analog_quality
    analog_gate = _blind_analog_gate(
        project_for_model,
        analog_diagnostics,
        analog_quality,
        clean_model_analogs,
    )
    if analog_gate:
        return {
            "project_id": project_id,
            "project_name": project.get("name"),
            "error": analog_gate,
            "error_class": "needs_analog_refresh",
            "analog_quality": analog_quality,
            "analog_diagnostics": analog_diagnostics,
        }
    state = {
        "project_id": project_id,
        "project": project_for_model,
        "analogs": analogs,
        "use_mre": False,
        "find_analogs": find_analogs,
    }
    out = parallel_gold_model_node(state)
    if out.get("error"):
        return {
            "project_id": project_id,
            "project_name": project.get("name"),
            "error": out["error"],
            "analog_diagnostics": analog_diagnostics,
        }

    model = out.get("parallel_model") or {}
    evidence_score = evidence_quality_score(project_for_model.get("drilling_evidence"))
    provisional_fields = _fields_from_parallel(
        project_for_model,
        model,
        batch_id=batch_id,
        evidence_score=evidence_score,
    )
    truth = official_truth(project_with_truth)
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
        "provisional_errors": provisional_errors,
        "raw_core_pass": _core_pass(provisional_errors, threshold),
        "final_core_pass": _core_pass(errors, threshold),
        "split_pass": _split_pass(errors, max(threshold, 0.10)),
        "analog_quality": analog_quality,
        "failure_class": failure,
        "local_guards": extract_local_guards(model),
        "analog_diagnostics": analog_diagnostics,
    }


def _core_pass(errors: Dict[str, Optional[float]], threshold: float) -> bool:
    return all(
        errors[k] is not None
        and not math.isinf(errors[k])
        and abs(errors[k]) <= threshold
        for k in ("tonnage", "grade", "contained")
    )


def _split_pass(errors: Dict[str, Optional[float]], threshold: float) -> bool:
    return all(
        errors[k] is not None
        and not math.isinf(errors[k])
        and abs(errors[k]) <= threshold
        for k in ("mi_tonnage", "mi_grade", "inferred_tonnage", "inferred_grade")
    )


def _is_parallel_quota_error(error: Any) -> bool:
    text = str(error or "").lower()
    return (
        "402" in text
        and ("payment required" in text or "quota" in text or "parallel api" in text)
    )


def _error_class(error: Any) -> str:
    if _is_parallel_quota_error(error):
        return "parallel_quota"
    text = str(error or "").lower()
    if "needs_analog_refresh" in text:
        return "needs_analog_refresh"
    if "parallel api" in text:
        return "parallel_api"
    return "error"


def _dedupe_project_ids(project_ids: Iterable[Any]) -> List[str]:
    seen: Set[str] = set()
    deduped: List[str] = []
    for project_id in project_ids:
        if not project_id:
            continue
        pid = str(project_id)
        if pid in seen:
            continue
        seen.add(pid)
        deduped.append(pid)
    return deduped


def _leaderboard_name_to_project_id(payload: Dict[str, Any]) -> Dict[str, str]:
    target_selection = payload.get("target_selection") or {}
    project_ids = target_selection.get("project_ids") or []
    project_names = target_selection.get("project_names") or []
    mapping = {
        str(name): str(project_id)
        for project_id, name in zip(project_ids, project_names)
        if project_id and name
    }
    missing_name_ids = [
        str(project_id)
        for project_id in project_ids
        if project_id and str(project_id) not in set(mapping.values())
    ]
    for project_id in missing_name_ids:
        try:
            with _SUPABASE_LOCK:
                project = supabase_ops.get_project(project_id)
        except Exception:
            logging.exception(
                "[parallel-gold-backtest] failed to resolve project name for resume target %s",
                project_id,
            )
            continue
        name = project.get("name") if project else None
        if name:
            mapping[str(name)] = project_id
    return mapping


def _leaderboard_project_id(
    row: Dict[str, Any],
    payload: Dict[str, Any],
    name_to_id_cache: Dict[str, str],
) -> Optional[str]:
    project_id = row.get("project_id")
    if project_id:
        return str(project_id)
    project_name = row.get("project")
    if not project_name:
        return None
    if not name_to_id_cache:
        name_to_id_cache.update(_leaderboard_name_to_project_id(payload))
    return name_to_id_cache.get(str(project_name))


def _resume_project_ids_from_leaderboard(path: Path, *, mode: str = "errors") -> List[str]:
    """Return project IDs to retry from a prior leaderboard artifact.

    modes:
      errors      retry errored/skipped targets only
      misses      retry evaluated targets that failed the 95% gate
      non_passed  retry both errors/skips and misses
      incomplete  retry every requested target that did not produce a passing row
    """
    if mode not in {"errors", "misses", "non_passed", "incomplete"}:
        raise ValueError(f"unsupported resume mode: {mode}")

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    leaderboard = payload.get("leaderboard") or []
    errors = payload.get("errors") or []
    target_selection = payload.get("target_selection") or {}
    requested_ids = _dedupe_project_ids(target_selection.get("project_ids") or [])
    name_to_id_cache: Dict[str, str] = {}

    if mode == "incomplete":
        passed_ids = {
            project_id
            for row in leaderboard
            if row.get("pass") is True
            for project_id in [_leaderboard_project_id(row, payload, name_to_id_cache)]
            if project_id
        }
        if requested_ids:
            return _dedupe_project_ids(
                project_id for project_id in requested_ids if project_id not in passed_ids
            )

    retry_ids: List[str] = []
    if mode in {"errors", "non_passed", "incomplete"}:
        retry_ids.extend(
            project_id
            for row in errors
            for project_id in [_leaderboard_project_id(row, payload, name_to_id_cache)]
            if project_id
        )
    if mode in {"misses", "non_passed", "incomplete"}:
        retry_ids.extend(
            project_id
            for row in leaderboard
            if row.get("pass") is False
            for project_id in [_leaderboard_project_id(row, payload, name_to_id_cache)]
            if project_id
        )
    return _dedupe_project_ids(retry_ids)


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
        "--audit-json",
        default=None,
        help=(
            "Gold split/backtest audit JSON to select queued targets from, "
            "for example artifacts/gold_project_fill_queue_after_*.json."
        ),
    )
    parser.add_argument(
        "--audit-targets",
        type=int,
        default=0,
        help="Select this many queued targets from --audit-json.",
    )
    parser.add_argument(
        "--audit-backtest-status",
        action="append",
        choices=(
            "ready_untested",
            "retry_after_quota",
            "retry_after_error",
            "needs_accuracy_review",
            "validated_pass",
        ),
        default=[],
        help=(
            "Backtest queue status to include when selecting --audit-targets. "
            "Repeatable. Defaults to retry_after_quota and needs_accuracy_review."
        ),
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
        "--target-evidence-provider",
        choices=("auto", "exa", "parallel", "none"),
        default="auto",
        help=(
            "Provider used with --refresh-target-evidence. auto keeps useful "
            "Exa evidence and escalates weak/missing evidence to Parallel."
        ),
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
    parser.add_argument(
        "--resume-leaderboard-json",
        action="append",
        default=[],
        help=(
            "Read a prior leaderboard artifact and add retry targets from it. "
            "Repeatable for multiple partial batches."
        ),
    )
    parser.add_argument(
        "--resume-mode",
        choices=("errors", "misses", "non_passed", "incomplete"),
        default="errors",
        help=(
            "Which prior targets to retry from --resume-leaderboard-json. "
            "Default errors retries quota/API failures and skipped targets."
        ),
    )
    args = parser.parse_args()
    if args.processor:
        settings.parallel_processor = args.processor
    if args.poll_timeout_s is not None:
        parallel_gold_model_module._POLL_TIMEOUT_S = max(15, int(args.poll_timeout_s))
    parallel_gold_model_module._POLL_HEARTBEAT = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    excluded_project_ids = set(args.exclude_project_id)
    excluded_project_ids.update(_run_project_ids(args.exclude_run_id))
    selection_pool_summary: Optional[Dict[str, int]] = None
    audit_selection_summary: Optional[Dict[str, Any]] = None
    try:
        selection_pool_summary = _db_truth_pool_summary(excluded_project_ids)
    except Exception:
        logging.exception("[parallel-gold-backtest] failed to summarize truth-backed DB pool")

    targets = [{"project_id": pid, "fixture": None, "fixture_name": None} for pid in args.project_id]
    for resume_path in args.resume_leaderboard_json:
        resume_project_ids = _resume_project_ids_from_leaderboard(
            Path(resume_path),
            mode=args.resume_mode,
        )
        if resume_project_ids:
            print(
                f"[parallel-gold-backtest] resume {Path(resume_path).name}: "
                f"{len(resume_project_ids)} target(s) from mode={args.resume_mode}",
                flush=True,
            )
        targets.extend(
            {"project_id": pid, "fixture": None, "fixture_name": None}
            for pid in resume_project_ids
        )
    if args.audit_targets:
        if not args.audit_json:
            parser.error("--audit-targets requires --audit-json")
        audit_statuses = args.audit_backtest_status or [
            "retry_after_quota",
            "needs_accuracy_review",
        ]
        already_selected = {target["project_id"] for target in targets}
        audit_targets, audit_selection_summary = _audit_targets(
            Path(args.audit_json),
            statuses=audit_statuses,
            limit=args.audit_targets,
            exclude_project_ids=excluded_project_ids | already_selected,
            random_seed=args.random_seed,
            randomize=True,
        )
        print(
            f"[parallel-gold-backtest] audit selection {Path(args.audit_json).name}: "
            f"{len(audit_targets)} target(s) from statuses={','.join(audit_statuses)}",
            flush=True,
        )
        targets.extend(audit_targets)
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
        pool_note = ""
        if selection_pool_summary:
            pool_note = (
                f" full_split_truth_total={selection_pool_summary['full_split_truth_total']}, "
                f"excluded_full_split_truth={selection_pool_summary['excluded_full_split_truth']}, "
                f"remaining_full_split_truth={selection_pool_summary['remaining_full_split_truth']}."
            )
        explicit_subset = bool(
            args.project_id
            or args.fixture
            or args.fixture_first
            or args.resume_leaderboard_json
        ) and not args.random_targets
        if explicit_subset:
            print(
                f"[parallel-gold-backtest] targeted subset: {len(targets)} "
                "truth-backed target(s) selected from explicit IDs/fixtures."
                f"{pool_note}",
                flush=True,
            )
        else:
            print(
                f"[parallel-gold-backtest] warning: only {len(targets)} "
                "truth-backed target(s) selected; cannot prove 5/10 95% matches "
                "until more projects have full MRE truth fields or prior holdouts "
                f"are not excluded.{pool_note}",
                flush=True,
            )
    if args.list_targets:
        if excluded_project_ids:
            print(f"# excluded_project_ids={len(excluded_project_ids)}")
        if selection_pool_summary:
            print(
                "# full_split_truth_total="
                f"{selection_pool_summary['full_split_truth_total']} "
                "excluded_full_split_truth="
                f"{selection_pool_summary['excluded_full_split_truth']} "
                "remaining_full_split_truth="
                f"{selection_pool_summary['remaining_full_split_truth']}"
            )
        if args.random_targets:
            print(f"# random_targets={args.random_targets} random_seed={args.random_seed}")
        if audit_selection_summary:
            print(
                "# audit_targets="
                f"{audit_selection_summary['selected_count']} "
                "audit_eligible="
                f"{audit_selection_summary['eligible_count']} "
                "audit_statuses="
                f"{','.join(audit_selection_summary['requested_statuses'])}"
            )
        for target in targets:
            project = supabase_ops.get_project(target["project_id"])
            status = f" | {target.get('audit_backtest_status')}" if target.get("audit_backtest_status") else ""
            print(f"{target['project_id']} | {project.get('name') if project else '?'}{status}")
        return 0

    batch_id = args.batch_id or f"gold_blind_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    print(
        f"[parallel-gold-backtest] launching {len(targets)} blind pre-MRE run(s) "
        f"with workers={args.workers}, save={not args.no_save}, "
        f"find_analogs={args.find_analogs}, batch_id={batch_id}"
    )

    results: List[Dict[str, Any]] = []
    max_workers = max(1, args.workers)
    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures: Dict[Any, Dict[str, Any]] = {}
    next_target_index = 0
    quota_limited = False

    def build_leaderboard_payload(stage: str) -> Tuple[Dict[str, Any], int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        pass_count = 0
        leaderboard: List[Dict[str, Any]] = []
        for result in sorted(results, key=lambda r: r.get("project_name") or ""):
            if result.get("error"):
                continue
            passed = _core_pass(result["errors"], args.threshold)
            pass_count += int(passed)
            failure = result.get("failure_class") or {}
            row = leaderboard_row(
                project_name=result["project_name"],
                errors=result["errors"],
                passed=passed,
                failure=failure,
                guards=result.get("local_guards"),
            )
            row["project_id"] = result.get("project_id")
            row["analog_diagnostics"] = result.get("analog_diagnostics")
            row["analog_quality"] = result.get("analog_quality")
            row["raw_core_pass"] = result.get("raw_core_pass")
            row["final_core_pass"] = result.get("final_core_pass")
            row["split_pass_10pct"] = result.get("split_pass")
            row["split_errors_pct"] = {
                "mi_tonnage": None if result["errors"].get("mi_tonnage") is None else round(result["errors"]["mi_tonnage"] * 100, 3),
                "mi_grade": None if result["errors"].get("mi_grade") is None else round(result["errors"]["mi_grade"] * 100, 3),
                "inferred_tonnage": None if result["errors"].get("inferred_tonnage") is None else round(result["errors"]["inferred_tonnage"] * 100, 3),
                "inferred_grade": None if result["errors"].get("inferred_grade") is None else round(result["errors"]["inferred_grade"] * 100, 3),
            }
            row["provisional_errors_pct"] = {
                key: None if value is None else round(value * 100, 3)
                for key, value in (result.get("provisional_errors") or {}).items()
            }
            leaderboard.append(row)

        evaluated_count = sum(1 for result in results if not result.get("error"))
        error_rows = [
            {
                "project": result.get("project_name"),
                "project_id": result.get("project_id"),
                "error": result.get("error"),
                "error_class": result.get("error_class") or _error_class(result.get("error")),
                "analog_diagnostics": result.get("analog_diagnostics"),
            }
            for result in results
            if result.get("error")
        ]
        result_name_by_id = {
            result.get("project_id"): result.get("project_name")
            for result in results
            if result.get("project_id") and result.get("project_name")
        }
        payload = {
            "batch_id": batch_id,
            "stage": stage,
            "checkpointed_at": datetime.now(timezone.utc).isoformat(),
            "threshold": args.threshold,
            "min_pass_count": args.min_pass_count,
            "pass_count": pass_count,
            "evaluated_count": evaluated_count,
            "completed_count": len(results),
            "requested_count": len(targets),
            "attempted_count": sum(1 for r in results if (r.get("error_class") or "") != "parallel_quota_skipped"),
            "remaining_count": max(0, len(targets) - len(results)),
            "quota_limited": quota_limited,
            "truth_pool_summary": selection_pool_summary,
            "audit_selection_summary": audit_selection_summary,
            "selected_targets": [
                {
                    "project_id": target["project_id"],
                    "project_name": result_name_by_id.get(target["project_id"]) or target.get("project_name"),
                    "audit_backtest_status": target.get("audit_backtest_status"),
                }
                for target in targets
            ],
            "target_selection": {
                "random_targets": args.random_targets,
                "random_seed": args.random_seed,
                "excluded_project_ids": sorted(excluded_project_ids),
                "excluded_run_ids": args.exclude_run_id,
                "resume_leaderboard_json": args.resume_leaderboard_json,
                "resume_mode": args.resume_mode,
                "project_ids": [target["project_id"] for target in targets],
                "project_names": [
                    result_name_by_id.get(target["project_id"]) or target.get("project_name")
                    for target in targets
                ],
                "truth_pool_summary": selection_pool_summary,
                "audit_selection_summary": audit_selection_summary,
            },
            "leaderboard": leaderboard,
            "errors": error_rows,
        }
        return payload, pass_count, evaluated_count, leaderboard, error_rows

    def write_leaderboard(stage: str) -> None:
        if not args.leaderboard_json:
            return
        payload, _, _, _, _ = build_leaderboard_payload(stage)
        out_path = Path(args.leaderboard_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f"{out_path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tmp_path.replace(out_path)

    def submit_next_target() -> bool:
        nonlocal next_target_index
        if next_target_index >= len(targets):
            return False
        target = targets[next_target_index]
        next_target_index += 1
        future = pool.submit(
            _run_one,
            target["project_id"],
            fixture=target.get("fixture"),
            save=not args.no_save,
            find_analogs=args.find_analogs,
            refresh_target_evidence=args.refresh_target_evidence,
            target_evidence_provider=args.target_evidence_provider,
            batch_id=batch_id,
            threshold=args.threshold,
        )
        futures[future] = target
        return True

    for _ in range(min(max_workers, len(targets))):
        submit_next_target()
    try:
        while futures:
            future = next(as_completed(list(futures)))
            target = futures.pop(future)
            try:
                result = future.result()
            except Exception as exc:
                logging.exception("[parallel-gold-backtest] worker failed")
                result = {
                    "project_id": target.get("project_id") or "unknown",
                    "error": str(exc),
                    "error_class": _error_class(exc),
                }
            if result.get("error") and not result.get("error_class"):
                result["error_class"] = _error_class(result.get("error"))
            results.append(result)
            write_leaderboard("running")
            name = result.get("project_name") or result.get("project_id")
            if result.get("error"):
                print(f"  FAIL  {name}: {result['error']}", flush=True)
                if result.get("error_class") == "parallel_quota":
                    quota_limited = True
                    print(
                        "  NOTE  Parallel quota/payment limit reached; "
                        "not submitting additional queued targets.",
                        flush=True,
                    )
            else:
                passed = _core_pass(result["errors"], args.threshold)
                print(
                    f"  {'PASS' if passed else 'MISS'}  {name}: "
                    f"T {fmt_pct(result['errors']['tonnage'])}, "
                    f"G {fmt_pct(result['errors']['grade'])}, "
                    f"Au {fmt_pct(result['errors']['contained'])}",
                    flush=True,
                )
            if not quota_limited:
                while len(futures) < max_workers and submit_next_target():
                    pass
    except KeyboardInterrupt:
        print("\n[parallel-gold-backtest] interrupted; cancelling queued futures", flush=True)
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)

    if quota_limited and next_target_index < len(targets):
        for target in targets[next_target_index:]:
            project_name = None
            try:
                with _SUPABASE_LOCK:
                    skipped_project = supabase_ops.get_project(target["project_id"])
                project_name = skipped_project.get("name") if skipped_project else None
            except Exception:
                logging.exception(
                    "[parallel-gold-backtest] failed to load skipped project name"
                )
            results.append({
                "project_id": target["project_id"],
                "project_name": project_name,
                "error": "Skipped because an earlier Parallel quota/payment error stopped new submissions.",
                "error_class": "parallel_quota_skipped",
            })
        write_leaderboard("quota_limited")

    print()
    print(f"{'Project':54s} {'Pass':>5s} {'T err':>9s} {'G err':>9s} {'Au err':>9s}")
    print("-" * 86)
    payload, pass_count, evaluated_count, leaderboard, error_rows = build_leaderboard_payload("complete")
    for result in sorted(results, key=lambda r: r.get("project_name") or ""):
        if result.get("error"):
            print(f"{(result.get('project_name') or result.get('project_id'))[:54]:54s} {'ERR':>5s}")
            continue
        passed = _core_pass(result["errors"], args.threshold)
        failure = result.get("failure_class") or {}
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
    print(f"95% core matches: {pass_count}/{evaluated_count}")
    if error_rows:
        print(
            f"errored/skipped targets: {len(error_rows)} "
            f"(quota_limited={'yes' if quota_limited else 'no'})"
        )
    print(f"batch_id: {batch_id}")
    if args.leaderboard_json:
        out_path = Path(args.leaderboard_json)
        write_leaderboard("complete")
        print(f"leaderboard_json: {out_path}")
    return 0 if pass_count >= args.min_pass_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
