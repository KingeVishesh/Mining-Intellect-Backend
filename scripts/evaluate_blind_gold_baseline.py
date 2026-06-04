#!/usr/bin/env python3
"""Fast blind gold baseline using only pre-MRE-safe target evidence + analogs.

This diagnostic does not call Parallel and does not write to Supabase. It is a
cheap way to test whether the evidence/analog layer can support a blind MRE
estimate before spending time on a Parallel run.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from graphs.analog_finder import _is_self_analog  # noqa: E402
from nodes import supabase_ops  # noqa: E402
from nodes.parallel_gold_model import (  # noqa: E402
    _as_float,
    _blind_scale_cap_mt,
    _evidence_mentions_target_mre,
    _median,
    _weak_geometry_only_evidence,
)
from scripts.backtest_parallel import official_truth, pct_err, fmt_pct  # noqa: E402
from scripts.run_parallel_gold_backtest import _db_truth_targets, _merge_fixture_truth  # noqa: E402

TROY_OZ_PER_TONNE = 32150.7466


def _contained(tonnage_mt: float, grade_gpt: float) -> float:
    return tonnage_mt * grade_gpt * TROY_OZ_PER_TONNE


def _clean_evidence(project: Dict[str, Any]) -> Dict[str, Any]:
    evidence = project.get("drilling_evidence")
    if not isinstance(evidence, dict):
        return {}
    if (
        evidence.get("redacted")
        or _evidence_mentions_target_mre(evidence)
        or _weak_geometry_only_evidence(evidence)
    ):
        return {}
    return evidence


def _clean_analogs(project: Dict[str, Any], analogs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    project_name = project.get("name") or ""
    company = project.get("company_name") or ""
    for analog in analogs:
        name = analog.get("name") or analog.get("analog_name") or ""
        if _is_self_analog(project_name, name, project_company=company):
            continue
        tonnage = _as_float(analog.get("tonnage_mt") or analog.get("analog_tonnage_mt"))
        grade = _as_float(analog.get("grade_value") or analog.get("analog_grade_value"))
        if not tonnage or not grade:
            continue
        copied = dict(analog)
        copied["tonnage_mt"] = tonnage
        copied["grade_value"] = grade
        out.append(copied)
    return out


def _geometry_tonnage(project: Dict[str, Any], evidence: Dict[str, Any]) -> Optional[float]:
    evidence_has_envelope = any(
        evidence.get(k) is not None
        for k in ("strike_length_m", "down_dip_extent_m", "avg_true_width_m", "drilled_area_km2")
    )
    strike = _as_float(
        evidence.get("strike_length_m")
        or project.get("strike_length_m")
        or project.get("strike_length_meters")
    )
    depth = _as_float(
        evidence.get("down_dip_extent_m")
        or project.get("down_dip_extent_m")
        or project.get("depth_meters")
    )
    width = _as_float(
        evidence.get("avg_true_width_m")
        or project.get("avg_true_width_m")
        or project.get("width_meters")
    )
    area = _as_float(evidence.get("drilled_area_km2"))
    if area and depth:
        return area * 1_000_000 * depth * 2.7 * 0.08 / 1_000_000
    if not strike:
        return None
    if not evidence_has_envelope and strike > 20_000:
        return None
    pattern = (project.get("mineralization_pattern") or "").lower()
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    if not width:
        if "vein" in pattern or "underground" in mining:
            width = 4.0
        elif "bulk" in pattern or "open" in mining:
            width = 25.0
        else:
            width = 10.0
    elif not evidence_has_envelope:
        width = min(width, 80.0)
    if not depth:
        depth = 150.0
    continuity = 0.18 if ("vein" in pattern or "underground" in mining) else 0.12
    return strike * depth * width * 2.7 * continuity / 1_000_000


def estimate(project: Dict[str, Any], analogs: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence = _clean_evidence(project)
    clean_analogs = _clean_analogs(project, analogs)
    analog_tonnage = _median([a["tonnage_mt"] for a in clean_analogs])
    analog_grade = _median([a["grade_value"] for a in clean_analogs])
    cap = _blind_scale_cap_mt({**project, "drilling_evidence": evidence}, clean_analogs)
    geom = _geometry_tonnage(project, evidence)

    candidates = [v for v in (geom, analog_tonnage) if v and v > 0]
    total_t = _median(candidates) or 1.0
    if cap:
        total_t = min(total_t, cap)
    if geom and analog_tonnage:
        # Let geometry constrain large analog medians but avoid throwing away
        # all analog scale information for partially defined footprints.
        total_t = min(total_t, max(geom * 2.0, geom + 2.0))
    total_t = max(0.001, total_t)

    grade = (
        _as_float(evidence.get("weighted_grade_g_t"))
        or _as_float(evidence.get("average_intercept_grade_g_t"))
        or analog_grade
        or 1.0
    )

    meters = _as_float(evidence.get("total_meters_drilled"))
    holes = _as_float(evidence.get("total_holes"))
    if meters and meters >= 75_000:
        mi_share = 0.55
    elif meters and meters >= 20_000:
        mi_share = 0.40
    elif holes and holes >= 150:
        mi_share = 0.35
    else:
        mi_share = 0.25
    mi_t = total_t * mi_share
    inf_t = total_t - mi_t
    return {
        "total_tonnage_mt": total_t,
        "total_grade_gpt": grade,
        "total_contained_oz": _contained(total_t, grade),
        "mi_tonnage_mt": mi_t,
        "mi_grade_gpt": grade,
        "inferred_tonnage_mt": inf_t,
        "inferred_grade_gpt": grade,
        "analog_count": len(clean_analogs),
        "cap_mt": cap,
        "geometry_mt": geom,
        "evidence_keys": sorted(k for k, v in evidence.items() if v not in (None, "", [])),
    }


def run(project_ids: List[str], threshold: float) -> int:
    if not project_ids:
        project_ids = [t["project_id"] for t in _db_truth_targets(limit=10)]
    rows = []
    for pid in project_ids:
        project = supabase_ops.get_project(pid)
        if not project:
            continue
        project = _merge_fixture_truth(project, None)
        analogs = supabase_ops.get_analogs(pid)
        pred = estimate(project, analogs)
        truth = official_truth(project)
        errors = {
            "tonnage": pct_err(pred["total_tonnage_mt"], truth["total_tonnage_mt"]),
            "grade": pct_err(pred["total_grade_gpt"], truth["total_grade_gpt"]),
            "contained": pct_err(pred["total_contained_oz"], truth["total_contained_oz"]),
        }
        passed = all(
            errors[k] is not None and not math.isinf(errors[k]) and abs(errors[k]) <= threshold
            for k in ("tonnage", "grade", "contained")
        )
        rows.append((project, pred, truth, errors, passed))

    for project, pred, truth, errors, passed in rows:
        print("=" * 96)
        print(project.get("name"))
        print(f"analog_count={pred['analog_count']} cap={pred['cap_mt']} geometry={pred['geometry_mt']} evidence={pred['evidence_keys']}")
        print(f"Total Mt    pred={pred['total_tonnage_mt']:.3f} truth={truth['total_tonnage_mt']:.3f} err={fmt_pct(errors['tonnage'])}")
        print(f"Total grade pred={pred['total_grade_gpt']:.3f} truth={truth['total_grade_gpt']:.3f} err={fmt_pct(errors['grade'])}")
        print(f"Contained   pred={pred['total_contained_oz']:.0f} truth={truth['total_contained_oz']:.0f} err={fmt_pct(errors['contained'])}")
        print(f"Core pass: {'PASS' if passed else 'FAIL'}")
    print("=" * 96)
    print(f"baseline: {sum(1 for *_, passed in rows if passed)}/{len(rows)} pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    return run(args.project_id, args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
