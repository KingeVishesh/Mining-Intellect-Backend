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
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from nodes import supabase_ops  # noqa: E402
from nodes import parallel_gold_model as parallel_gold_model_module  # noqa: E402
from nodes.parallel_gold_model import parallel_gold_model_node  # noqa: E402
from scripts.backtest_parallel import official_truth, pct_err, fmt_pct  # noqa: E402
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


_EVIDENCE_MRE_MARKERS = (
    "mineral resource estimate",
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


def _fields_from_parallel(project: Dict[str, Any], parallel_out: Dict[str, Any]) -> Dict[str, Any]:
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
            "use_mre": False,
            "anchor_used": parallel_out.get("anchor_used"),
            "methodology": parallel_out.get("methodology"),
            "analogs_used": parallel_out.get("analogs_used"),
            "analogs_rejected": parallel_out.get("analogs_rejected"),
            "sources": parallel_out.get("sources"),
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


def _db_truth_targets(limit: int) -> List[Dict[str, Any]]:
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
        .limit(limit)
        .execute()
    )
    return [
        {"project_id": row["id"], "fixture": None, "fixture_name": None}
        for row in (res.data or [])
    ]


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
        merged["mre_data_source"] = {
            "as_of_date": latest_mre.get("effective_date"),
            "source_url": latest_mre.get("source_url"),
            "source_doc": latest_mre.get("source"),
            "notes": latest_mre.get("notes"),
        }
        merged["mre_date"] = latest_mre.get("effective_date")
        merged["mre_source_url"] = latest_mre.get("source_url")
    return merged


def _run_one(
    project_id: str,
    *,
    fixture: Optional[Dict[str, Any]],
    save: bool,
    find_analogs: bool,
    refresh_target_evidence: bool,
) -> Dict[str, Any]:
    with _SUPABASE_LOCK:
        project = supabase_ops.get_project(project_id)
    if not project:
        return {"project_id": project_id, "error": "project not found"}
    project_for_model = _merge_fixture_truth(project, fixture)
    cutoff = _parse_loose_date(project_for_model.get("mre_date") or project_for_model.get("mre_data_source"))
    cached_evidence = project_for_model.get("drilling_evidence")
    if refresh_target_evidence or not _evidence_is_pre_cutoff(cached_evidence, cutoff):
        evidence = _extract_pre_mre_target_evidence(project_for_model)
        if evidence:
            project_for_model = {**project_for_model, "drilling_evidence": evidence}
            for field in ("strike_length_m", "down_dip_extent_m", "avg_true_width_m"):
                if project_for_model.get(field) is None and evidence.get(field) is not None:
                    project_for_model[field] = evidence.get(field)
            if save and project.get("id") and _evidence_is_pre_cutoff(evidence, cutoff):
                with _SUPABASE_LOCK:
                    supabase_ops.save_project_drilling_evidence(project_id, evidence)

    with _SUPABASE_LOCK:
        analogs = supabase_ops.get_analogs(project_id)
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
    fields = _fields_from_parallel(project_for_model, model)
    if save:
        with _SUPABASE_LOCK:
            supabase_ops.save_model_run(
                project_id=project_id,
                model_type="parallel_pre_mre",
                fields=fields,
                model_output_json=model,
            )

    truth = official_truth(project_for_model)
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
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.05)
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
    args = parser.parse_args()
    if args.processor:
        settings.parallel_processor = args.processor
    if args.poll_timeout_s is not None:
        parallel_gold_model_module._POLL_TIMEOUT_S = max(15, int(args.poll_timeout_s))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
    if not targets:
        targets.extend(_db_truth_targets(limit=10))
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
        for target in targets:
            project = supabase_ops.get_project(target["project_id"])
            print(f"{target['project_id']} | {project.get('name') if project else '?'}")
        return 0

    print(
        f"[parallel-gold-backtest] launching {len(targets)} blind pre-MRE run(s) "
        f"with workers={args.workers}, save={not args.no_save}, "
        f"find_analogs={args.find_analogs}"
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
    for result in sorted(results, key=lambda r: r.get("project_name") or ""):
        if result.get("error"):
            print(f"{(result.get('project_name') or result.get('project_id'))[:54]:54s} {'ERR':>5s}")
            continue
        passed = _core_pass(result["errors"], args.threshold)
        pass_count += int(passed)
        print(
            f"{result['project_name'][:54]:54s} "
            f"{'YES' if passed else 'NO':>5s} "
            f"{fmt_pct(result['errors']['tonnage']):>9s} "
            f"{fmt_pct(result['errors']['grade']):>9s} "
            f"{fmt_pct(result['errors']['contained']):>9s}"
        )

    print("-" * 86)
    print(f"95% core matches: {pass_count}/{sum(1 for r in results if not r.get('error'))}")
    return 0 if pass_count >= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
