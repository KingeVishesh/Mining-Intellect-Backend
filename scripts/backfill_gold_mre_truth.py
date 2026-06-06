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
    python3 scripts/backfill_gold_mre_truth.py --db-gold --list-candidates
    python3 scripts/backfill_gold_mre_truth.py --db-gold --limit 20 --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
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
TRUTH_FIELDS = (
    "mre_mi_tonnage_mt", "mre_mi_grade",
    "mre_inferred_tonnage_mt", "mre_inferred_grade",
)
PROJECT_SELECT = ",".join((
    "id", "name", "material", "country", "region", "district",
    "deposit_type", "deposit_subtype", "tonnage_mt",
    "grade_value", "grade_unit", "total_contained", "resource_category",
    "resource_compliance_standard", "resource_vintage_year",
    "mre_mi_tonnage_mt", "mre_mi_grade", "mre_mi_contained",
    "mre_inferred_tonnage_mt", "mre_inferred_grade",
    "mre_inferred_contained", "updated_at",
))


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


def _fetch_gold_project_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        res = (
            supabase_ops.get_client()
            .table("projects")
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


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_full_truth(row: Dict[str, Any]) -> bool:
    return all(row.get(k) is not None for k in TRUTH_FIELDS)


def _missing_truth_fields(row: Dict[str, Any]) -> List[str]:
    return [field for field in TRUTH_FIELDS if row.get(field) is None]


def _resource_category_kind(row: Dict[str, Any]) -> str:
    category = re.sub(r"\s+", " ", str(row.get("resource_category") or "").strip().lower())
    if not category:
        return "unknown"
    has_inferred = "inferred" in category
    has_mi = any(token in category for token in ("measured", "indicated", "m&i", "m and i"))
    if has_inferred and has_mi:
        return "mi_and_inferred"
    if has_inferred:
        return "inferred_only"
    if has_mi:
        return "mi_only"
    return "unknown"


def _candidate_priority(row: Dict[str, Any]) -> tuple:
    category_rank = {
        "mi_and_inferred": 0,
        "unknown": 20,
        "mi_only": 40,
        "inferred_only": 50,
    }[_resource_category_kind(row)]
    compliance_rank = 0 if row.get("resource_compliance_standard") else 1
    vintage = _as_float(row.get("resource_vintage_year")) or 0
    return (
        category_rank,
        compliance_rank,
        -vintage,
        (row.get("name") or "").lower(),
        row.get("id") or "",
    )


def _has_valid_known_total(row: Dict[str, Any]) -> bool:
    tonnage = _as_float(row.get("tonnage_mt"))
    grade = _as_float(row.get("grade_value"))
    if tonnage is None or grade is None:
        return False
    return 0 < tonnage <= 1000 and 0 < grade <= 50


def _candidate_status(
    row: Dict[str, Any],
    *,
    require_known_total: bool = True,
    include_full_truth: bool = False,
) -> tuple[bool, str]:
    material = str(row.get("material") or "").lower()
    if "gold" not in material:
        return False, "not a gold row"
    if _has_full_truth(row) and not include_full_truth:
        return False, "already has full M&I/Inferred truth"
    if require_known_total and not _has_valid_known_total(row):
        return False, "missing or invalid known total tonnage/grade"
    missing = _missing_truth_fields(row)
    if missing:
        return True, f"missing truth fields: {', '.join(missing)}"
    return True, "full truth included by --force"


def _db_gold_rows(
    *,
    limit: Optional[int] = None,
    require_known_total: bool = True,
    include_full_truth: bool = False,
    randomize: bool = False,
    random_seed: Optional[str] = None,
    prioritize_candidates: bool = True,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    for row in _fetch_gold_project_rows():
        ok, reason = _candidate_status(
            row,
            require_known_total=require_known_total,
            include_full_truth=include_full_truth,
        )
        if not ok:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        candidates.append(row)
    if randomize:
        candidates = sorted(candidates, key=lambda row: ((row.get("name") or "").lower(), row["id"]))
        random.Random(random_seed).shuffle(candidates)
    elif prioritize_candidates:
        candidates = sorted(candidates, key=_candidate_priority)
    if limit:
        candidates = candidates[:limit]
    logger.info(
        "DB gold candidates selected=%s randomize=%s seed=%s prioritize=%s skipped=%s",
        len(candidates),
        randomize,
        random_seed,
        prioritize_candidates,
        json.dumps(skipped, sort_keys=True),
    )
    return candidates


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


def _normalise_extracted_tonnage_units(
    row: Dict[str, Any],
    extracted: Dict[str, Any],
) -> tuple[Dict[str, Any], Optional[str]]:
    """Rescale obvious kt/t tonnage answers before validation.

    Exa sometimes returns table values in kt or tonnes while filling our
    `*_tonnage_mt` schema. Only correct when both split buckets exist and
    the combined split is near an exact 1,000x or 1,000,000x multiple of
    the known project total.
    """
    known_t = _as_float(row.get("tonnage_mt"))
    mi_t = _as_float(extracted.get("mi_tonnage_mt"))
    inf_t = _as_float(extracted.get("inferred_tonnage_mt"))
    if known_t is None or known_t <= 0 or mi_t is None or inf_t is None:
        return extracted, None

    total_t = mi_t + inf_t
    if total_t <= 0:
        return extracted, None

    ratio = total_t / known_t
    for factor, label in ((1000.0, "kt"), (1_000_000.0, "tonnes")):
        scaled_total = total_t / factor
        ratio_matches_known_total = abs(ratio - factor) / factor <= 0.05
        obvious_raw_unit = ratio > (factor * 0.10) and 0 < scaled_total <= 1000
        if not (ratio_matches_known_total or obvious_raw_unit):
            continue
        normalised = dict(extracted)
        normalised["mi_tonnage_mt"] = mi_t / factor
        normalised["inferred_tonnage_mt"] = inf_t / factor
        normalised["unit_normalization"] = {
            "from": label,
            "factor_to_mt": factor,
            "known_total_mt": known_t,
            "raw_total": total_t,
        }
        return (
            normalised,
            f"normalised extracted tonnage from {label} to Mt",
        )
    return extracted, None


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


def _reviewable_total_mismatch(row: Dict[str, Any], extracted: Dict[str, Any], reason: str) -> bool:
    if "differs from known total" not in reason:
        return False
    relaxed_ok, _ = _validate_against_known_total(row, extracted, allow_total_mismatch=True)
    return relaxed_ok


def _build_mre_payload(row: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
    totals = _weighted_total(
        extracted["mi_tonnage_mt"], extracted["mi_grade"],
        extracted["inferred_tonnage_mt"], extracted["inferred_grade"],
    )
    as_of = extracted.get("as_of_year")
    # The extractor often only knows the MRE/report year. Use Jan 1 as the
    # conservative blind-backtest cutoff so same-year post-MRE disclosures
    # are not accidentally allowed as "pre-MRE" evidence.
    effective_date = f"{int(as_of)}-01-01" if as_of else None
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
    parser.add_argument(
        "--db-gold",
        action="store_true",
        help="Scan Supabase gold projects missing full M&I/Inferred MRE truth.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--random-candidates",
        action="store_true",
        help="Shuffle DB gold extraction candidates with --random-seed before applying --limit.",
    )
    parser.add_argument(
        "--random-seed",
        default=None,
        help="Seed for --random-candidates so the same backfill batch can be reproduced.",
    )
    parser.add_argument(
        "--no-prioritize-candidates",
        action="store_true",
        help="Keep alphabetical DB candidate order instead of ranking likely full split disclosures first.",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print candidate rows and exit without calling Exa or writing data.",
    )
    parser.add_argument("--apply", action="store_true", help="Write verified truth to Supabase.")
    parser.add_argument("--force", action="store_true", help="Refetch even when full truth already exists.")
    parser.add_argument(
        "--allow-no-known-total",
        action="store_true",
        help="Allow DB gold rows without a known total cross-check into extraction candidates.",
    )
    parser.add_argument(
        "--allow-total-mismatch",
        action="store_true",
        help="Accept sane extracted split truth even when local total fields are stale/conflicting.",
    )
    parser.add_argument(
        "--target-usable",
        type=int,
        default=None,
        help="Stop extraction after this many rows pass validation.",
    )
    parser.add_argument("--json-out", default=None, help="Optional path for extraction audit JSON.")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    if args.fixtures or args.fixture:
        rows.extend(_fixture_rows(args.fixture or None))
    rows.extend(_db_rows(args.project_id))
    if args.db_gold:
        rows.extend(
            _db_gold_rows(
                limit=args.limit,
                require_known_total=not args.allow_no_known_total,
                include_full_truth=args.force,
                randomize=args.random_candidates,
                random_seed=args.random_seed,
                prioritize_candidates=not args.no_prioritize_candidates,
            )
        )
    if not rows:
        rows.extend(_fixture_rows(None))
    if args.limit and not args.db_gold:
        rows = rows[: args.limit]

    if args.list_candidates:
        audit = []
        for row in rows:
            ok, reason = _candidate_status(
                row,
                require_known_total=not args.allow_no_known_total,
                include_full_truth=args.force,
            )
            audit.append({
                "name": row.get("name") or row.get("_fixture") or row.get("id"),
                "project_id": row.get("id"),
                "status": "candidate" if ok else "skipped",
                "reason": reason,
                "tonnage_mt": row.get("tonnage_mt"),
                "grade_value": row.get("grade_value"),
                "resource_category": row.get("resource_category"),
                "resource_category_kind": _resource_category_kind(row),
                "resource_compliance_standard": row.get("resource_compliance_standard"),
                "resource_vintage_year": row.get("resource_vintage_year"),
                "priority": _candidate_priority(row),
                "missing_truth_fields": _missing_truth_fields(row),
            })
        for item in audit:
            print(
                f"{item['status']:9s} | {item['project_id']} | "
                f"{item['name']} | T={item['tonnage_mt']} G={item['grade_value']} | "
                f"{item['resource_category_kind']} | {item['reason']}"
            )
        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(audit, indent=2, default=str))
            logger.info("Wrote candidate JSON: %s", out_path)
        return 0

    audit: List[Dict[str, Any]] = []
    applied = 0
    usable = 0
    checked = 0
    for row in rows:
        name = row.get("name") or row.get("_fixture") or row.get("id")
        project_id = row.get("id")
        if _has_full_truth(row) and not args.force:
            logger.info("SKIP already truth-backed: %s", name)
            continue

        checked += 1
        logger.info("EXTRACT %s", name)
        extracted = _extract_one(row)
        if not extracted:
            audit.append({"name": name, "project_id": project_id, "status": "failed"})
            logger.warning("  failed: no extraction")
            continue

        extracted, unit_note = _normalise_extracted_tonnage_units(row, extracted)
        ok, reason = _validate_against_known_total(
            row,
            extracted,
            allow_total_mismatch=args.allow_total_mismatch,
        )
        if unit_note:
            reason = f"{unit_note}; {reason}"
        status = "usable" if ok else "rejected"
        if not ok and _reviewable_total_mismatch(row, extracted, reason):
            status = "review"
            relaxed_ok, relaxed_reason = _validate_against_known_total(
                row,
                extracted,
                allow_total_mismatch=True,
            )
            if relaxed_ok:
                reason = f"reviewable total mismatch: {reason}; {relaxed_reason}"
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
        if args.target_usable and usable >= args.target_usable:
            logger.info("Reached target usable rows: %s", args.target_usable)
            break

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(audit, indent=2, default=str))
        logger.info("Wrote audit JSON: %s", out_path)

    logger.info(
        "Done: usable=%s applied=%s total_checked=%s selected_rows=%s",
        usable,
        applied,
        checked,
        len(rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
