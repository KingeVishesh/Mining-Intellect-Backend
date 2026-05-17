"""
Corpus-level eval harness for the analog finder.

Runs the cascade against every project of a given material (default: gold),
LIBRARY-ONLY (no Exa — Exa is non-deterministic and noisy for regression
testing), and records the top-4 picks per project. Compares against a
checked-in baseline JSON; reports diffs and quality metrics.

Two modes:
    python3 scripts/eval_gold_corpus.py --bless
        Runs and OVERWRITES the baseline. Use after a deliberate change
        you want to bless as the new known-good state.

    python3 scripts/eval_gold_corpus.py
        Runs and DIFFS against the existing baseline. Exits non-zero
        when any project regressed (lost a baseline analog or gained a
        new one with sub_trend mismatch). Use in CI / pre-commit.

Other flags:
    --material gold          (default; could extend to copper/silver later)
    --limit N                Process only first N projects (smoke test)
    --sample N               Random sample of N
    --report-only            Don't update baseline, just print
    --baseline PATH          Override default baseline path
"""
from __future__ import annotations
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphs.analog_finder import (
    load_project_and_rule_node,
    build_target_profile_node,
    library_search_node,
    combine_filter_score_node,
)
from nodes.supabase_ops import get_client

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default baseline location — checked into git under tests/fixtures.
DEFAULT_BASELINE = (
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "gold_corpus_baseline.json"
)


def _list_projects(material: str) -> List[Dict[str, Any]]:
    client = get_client()
    res = (
        client.table("projects")
        .select("id,name,deposit_type,deposit_subtype,tectonic_belt")
        .ilike("material", material)
        .order("name")
        .execute()
    )
    return res.data or []


def _run_one(project_id: str) -> Dict[str, Any]:
    """Run the cascade against one project, library-only. Returns a
    snapshot dict with the fields we care about for regression checks."""
    state: Dict[str, Any] = {"project_id": project_id}
    state.update(load_project_and_rule_node(state))
    if state.get("error"):
        return {
            "project_id": project_id,
            "error": state["error"],
            "rule_id": None,
            "top_4": [],
            "library_count": 0,
            "low_confidence": True,
            "sub_trend": None,
        }
    state.update(build_target_profile_node(state))
    state.update(library_search_node(state))
    state["exa_analogs"] = []  # library-only for determinism
    state.update(combine_filter_score_node(state))

    project = state.get("project") or {}
    profile = state.get("target_profile") or {}
    scored = state.get("scored_analogs") or []

    return {
        "project_id": project_id,
        "project_name": project.get("name"),
        "rule_id": (state.get("analog_rule") or {}).get("rule_id"),
        "tectonic_belt": profile.get("tectonic_belt"),
        "sub_trend": profile.get("sub_trend"),
        "deposit_subtype": profile.get("deposit_subtype"),
        "mineralization_pattern": profile.get("mineralization_pattern"),
        "library_count": len(state.get("library_analogs") or []),
        "low_confidence": state.get("low_confidence", False),
        "top_4": [
            {
                "name": a.get("name"),
                "source": a.get("source"),
                "tectonic_belt": a.get("tectonic_belt"),
                "district": a.get("district"),
                "similarity_score": a.get("similarity_score"),
            }
            for a in scored[:4]
        ],
    }


def _diff_run(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
) -> List[str]:
    """Return human-readable diff lines, or empty list if no regression.

    Regression definition (deliberately strict; relax with judgment):
      - rule_id changed
      - low_confidence flipped True (regression) — flipping to False is FINE
      - sub_trend changed
      - top_4 set of names differs (added/removed)
    """
    diffs: List[str] = []
    if baseline.get("rule_id") != current.get("rule_id"):
        diffs.append(
            f"  rule_id: {baseline.get('rule_id')!r} → {current.get('rule_id')!r}"
        )
    if not baseline.get("low_confidence") and current.get("low_confidence"):
        diffs.append("  low_confidence flipped to True (regression)")
    if baseline.get("sub_trend") != current.get("sub_trend"):
        diffs.append(
            f"  sub_trend: {baseline.get('sub_trend')!r} → {current.get('sub_trend')!r}"
        )
    base_names = [a["name"] for a in (baseline.get("top_4") or [])]
    cur_names = [a["name"] for a in (current.get("top_4") or [])]
    if base_names != cur_names:
        removed = [n for n in base_names if n not in cur_names]
        added = [n for n in cur_names if n not in base_names]
        if removed or added:
            diffs.append(
                f"  top_4 changed: -{removed} +{added}"
            )
    return diffs


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    routed = sum(1 for r in rows if r.get("rule_id"))
    low_conf = sum(1 for r in rows if r.get("low_confidence"))
    has_sub = sum(1 for r in rows if r.get("sub_trend"))
    has_belt = sum(1 for r in rows if r.get("tectonic_belt"))
    avg_top = sum(len(r.get("top_4") or []) for r in rows) / max(total, 1)
    has_4 = sum(1 for r in rows if len(r.get("top_4") or []) == 4)
    return {
        "total": total,
        "routed_to_rule": routed,
        "low_confidence": low_conf,
        "has_tectonic_belt": has_belt,
        "has_sub_trend": has_sub,
        "average_top_count": round(avg_top, 2),
        "full_top_4": has_4,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="gold")
    parser.add_argument("--limit", type=int, default=None,
                          help="Process only first N projects")
    parser.add_argument("--sample", type=int, default=None,
                          help="Random sample of N projects (uses --seed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bless", action="store_true",
                          help="Overwrite the baseline with the current run")
    parser.add_argument("--report-only", action="store_true",
                          help="Don't touch baseline; just print results")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                          help=f"Baseline JSON path (default: {DEFAULT_BASELINE})")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    projects = _list_projects(args.material)
    if args.limit:
        projects = projects[: args.limit]
    if args.sample:
        random.Random(args.seed).shuffle(projects)
        projects = projects[: args.sample]
    print(f"[eval] {len(projects)} {args.material} project(s) to run "
          f"(library-only, no Exa)")

    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(projects, 1):
        if not args.quiet:
            print(f"  [{i:3}/{len(projects)}] {p['name'][:60]}", flush=True)
        try:
            snap = _run_one(p["id"])
            rows.append(snap)
        except Exception as e:
            logger.warning(f"  ! {p['name']}: {e}")
            rows.append({
                "project_id": p["id"],
                "project_name": p["name"],
                "error": str(e),
                "rule_id": None,
                "top_4": [],
                "low_confidence": True,
                "sub_trend": None,
            })

    rows.sort(key=lambda r: r.get("project_name") or "")

    summary = _summarize(rows)
    print()
    print("== Summary ==")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print()

    baseline_path = Path(args.baseline)

    if args.bless:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2,
                        sort_keys=False, default=str)
        )
        print(f"[eval] BLESSED — wrote {len(rows)} rows to {baseline_path}")
        return 0

    if args.report_only or not baseline_path.exists():
        if not baseline_path.exists():
            print(f"[eval] No baseline at {baseline_path}. "
                  f"Run with --bless to create one.")
        return 0

    # Diff against baseline
    baseline = json.loads(baseline_path.read_text())
    base_by_id = {r["project_id"]: r for r in baseline.get("rows", [])}

    regressions: List[Dict[str, Any]] = []
    for current in rows:
        b = base_by_id.get(current["project_id"])
        if not b:
            continue  # new project; not a regression
        d = _diff_run(b, current)
        if d:
            regressions.append({
                "name": current.get("project_name"),
                "diffs": d,
            })

    if not regressions:
        print(f"[eval] OK — no regressions vs {baseline_path.name}")
        return 0

    print(f"[eval] REGRESSIONS in {len(regressions)} project(s):")
    for r in regressions[:30]:
        print(f"\n  {r['name']}")
        for d in r["diffs"]:
            print(d)
    if len(regressions) > 30:
        print(f"\n  ... and {len(regressions) - 30} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
