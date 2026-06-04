#!/usr/bin/env python3
"""Backfill gold-project MRE truth fields for blind-model validation.

The gold Parallel backtest can only count projects with full official MRE
truth:

    mre_mi_tonnage_mt, mre_mi_grade,
    mre_inferred_tonnage_mt, mre_inferred_grade

This script uses the existing two-pass Exa M&I/Inferred extractor, validates
the extracted split against any known total resource on the project/fixture,
and optionally writes the result through `mre_runs` + the projects-table mirror.

Examples:
    python3 scripts/backfill_gold_mre_truth.py --fixtures
    python3 scripts/backfill_gold_mre_truth.py --fixtures --apply
    python3 scripts/backfill_gold_mre_truth.py --project-id <uuid> --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from nodes import inferred_extractor, supabase_ops  # noqa: E402


TROY_OZ_PER_TONNE = 32150.7466
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "backtest"


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _fixture_rows(names: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    paths = sorted(FIXTURE_DIR.glob("*.json"))
    if names:
        wanted = {n.removesuffix(".json") for n in names}
        paths = [p for p in paths if p.stem in wanted]
    return [json.loads(p.read_text()) | {"_fixture": p.stem} for p in paths]


def _db_rows(project_ids: Iterable[str]) -> List[Dict[str, Any]]:
    rows = []
    for pid in project_ids:
        project = supabase_ops.get_project(pid)
        if not project:
            logger.warning("Project not found: %s", pid)
            continue
        rows.append(project)
    return rows


def _has_full_truth(row: Dict[str, Any]) -> bool:
    return all(
        row.get(k) is not None
        for k in (
            "mre_mi_tonnage_mt", "mre_mi_grade",
            "mre_inferred_tonnage_mt", "mre_inferred_grade",
        )
    )


def _contained(tonnage_mt: Optional[float], grade_gpt: Optional[float]) -> Optional[float]:
    if tonnage_mt is None or grade_gpt is None:
        return None
    return float(tonnage_mt) * float(grade_gpt) * TROY_OZ_PER_TONNE


def _weighted_total(mi_t: float, mi_g: float, inf_t: float, inf_g: float) -> Dict[str, float]:
    total_t = float(mi_t) + float(inf_t)
    total_g = ((float(mi_t) * float(mi_g)) + (float(inf_t) * float(inf_g))) / total_t
    return {
        "total_tonnage_mt": total_t,
        "total_grade": total_g,
        "total_contained": _contained(total_t, total_g) or 0.0,
    }


def _rel_err(pred: Optional[float], actual: Optional[float]) -> Optional[float]:
    if pred is None or actual is None:
        return None
    if actual == 0:
        return math.inf if pred else 0.0
    return (float(pred) - float(actual)) / float(actual)


def _validate_against_known_total(
    row: Dict[str, Any],
    extracted: Dict[str, Any],
    *,
    allow_total_mismatch: bool = False,
) -> tuple[bool, str]:
    required = ("mi_tonnage_mt", "mi_grade", "inferred_tonnage_mt", "inferred_grade")
    missing = [k for k in required if extracted.get(k) is None]
    if missing:
        return False, f"missing required split fields: {', '.join(missing)}"

    totals = _weighted_total(
        extracted["mi_tonnage_mt"], extracted["mi_grade"],
        extracted["inferred_tonnage_mt"], extracted["inferred_grade"],
    )
    if totals["total_tonnage_mt"] <= 0 or totals["total_tonnage_mt"] > 1000:
        return False, f"extracted total tonnage {totals['total_tonnage_mt']:.1f} Mt is outside gold sanity bounds"
    if totals["total_grade"] <= 0 or totals["total_grade"] > 50:
        return False, f"extracted weighted grade {totals['total_grade']:.2f} g/t is outside gold sanity bounds"

    known_t = row.get("tonnage_mt")
    known_g = row.get("grade_value")
    if known_t is None or known_g is None:
        return True, "no known total to cross-check"

    t_err = _rel_err(totals["total_tonnage_mt"], known_t)
    g_err = _rel_err(totals["total_grade"], known_g)
    # Some rows have rounded or stale totals. Keep the gate strict enough to
    # catch category mixups but loose enough for rounding/reporting variants.
    if allow_total_mismatch:
        return True, (
            f"accepted despite local total mismatch: "
            f"T {t_err*100:.1f}%, G {g_err*100:.1f}%"
        )
    if t_err is not None and abs(t_err) > 0.10:
        return False, f"split total tonnage differs from known total by {t_err*100:.1f}%"
    if g_err is not None and abs(g_err) > 0.15:
        return False, f"split weighted grade differs from known total by {g_err*100:.1f}%"
    return True, f"total cross-check ok: T {t_err*100:.1f}%, G {g_err*100:.1f}%"


def _build_mre_payload(row: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
    totals = _weighted_total(
        extracted["mi_tonnage_mt"], extracted["mi_grade"],
        extracted["inferred_tonnage_mt"], extracted["inferred_grade"],
    )
    as_of = extracted.get("as_of_year")
    effective_date = f"{int(as_of)}-12-31" if as_of else None
    return {
        "total_tonnage_mt": round(totals["total_tonnage_mt"], 4),
        "total_grade": round(totals["total_grade"], 5),
        "total_contained": round(totals["total_contained"], 3),
        "grade_unit": row.get("grade_unit") or "g/t Au",
        "resource_category": row.get("resource_category") or "Measured + Indicated + Inferred",
        "effective_date": effective_date,
        "mi_tonnage_mt": round(float(extracted["mi_tonnage_mt"]), 4),
        "mi_grade": round(float(extracted["mi_grade"]), 5),
        "mi_contained": round(_contained(extracted["mi_tonnage_mt"], extracted["mi_grade"]) or 0.0, 3),
        "inferred_tonnage_mt": round(float(extracted["inferred_tonnage_mt"]), 4),
        "inferred_grade": round(float(extracted["inferred_grade"]), 5),
        "inferred_contained": round(
            _contained(extracted["inferred_tonnage_mt"], extracted["inferred_grade"]) or 0.0,
            3,
        ),
        "source": "exa_2pass_mre_truth_backfill",
        "source_url": extracted.get("source_url"),
        "notes": json.dumps(
            {
                "publisher": extracted.get("publisher"),
                "confidence": extracted.get("confidence"),
                "cross_validation": extracted.get("cross_validation"),
            },
            default=str,
        ),
    }


def _extract_one(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return inferred_extractor.extract_inferred_breakdown(
        analog_name=row.get("name") or "",
        material=row.get("material") or "gold",
        country=row.get("country"),
        region=row.get("region"),
        deposit_type=row.get("deposit_type"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", action="store_true", help="Use backtest fixture project IDs.")
    parser.add_argument("--fixture", action="append", default=[], help="Specific fixture stem.")
    parser.add_argument("--project-id", action="append", default=[], help="Specific Supabase project ID.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="Write verified truth to Supabase.")
    parser.add_argument("--force", action="store_true", help="Refetch even when full truth already exists.")
    parser.add_argument(
        "--allow-total-mismatch",
        action="store_true",
        help="Accept sane extracted split truth even when local total fields are stale/conflicting.",
    )
    parser.add_argument("--json-out", default=None, help="Optional path for extraction audit JSON.")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    if args.fixtures or args.fixture:
        rows.extend(_fixture_rows(args.fixture or None))
    rows.extend(_db_rows(args.project_id))
    if not rows:
        rows.extend(_fixture_rows(None))
    if args.limit:
        rows = rows[: args.limit]

    audit: List[Dict[str, Any]] = []
    applied = 0
    usable = 0
    for row in rows:
        name = row.get("name") or row.get("_fixture") or row.get("id")
        project_id = row.get("id")
        if _has_full_truth(row) and not args.force:
            logger.info("SKIP already truth-backed: %s", name)
            continue

        logger.info("EXTRACT %s", name)
        extracted = _extract_one(row)
        if not extracted:
            audit.append({"name": name, "project_id": project_id, "status": "failed"})
            logger.warning("  failed: no extraction")
            continue

        ok, reason = _validate_against_known_total(
            row,
            extracted,
            allow_total_mismatch=args.allow_total_mismatch,
        )
        status = "usable" if ok else "rejected"
        audit.append({
            "name": name,
            "project_id": project_id,
            "fixture": row.get("_fixture"),
            "status": status,
            "reason": reason,
            "extracted": extracted,
        })
        logger.info("  %s: %s", status.upper(), reason)
        if not ok:
            continue
        usable += 1

        if args.apply and project_id:
            payload = _build_mre_payload(row, extracted)
            supabase_ops.save_mre_run_if_changed(project_id, payload)
            supabase_ops.update_project_mre_mirror(project_id, payload)
            applied += 1
            logger.info("  APPLIED truth to %s", project_id)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(audit, indent=2, default=str))
        logger.info("Wrote audit JSON: %s", args.json_out)

    logger.info("Done: usable=%s applied=%s total_checked=%s", usable, applied, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
