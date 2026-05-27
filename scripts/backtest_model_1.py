#!/usr/bin/env python3
"""Backtest harness for Model 1.

For each project that has an official MRE (tonnage_mt + grade_value
populated), we run `build_model_1` on a copy of the project with those
two fields blanked — simulating the state of the data BEFORE the MRE
was published — and compare the prediction against the actual MRE.

The goal is to track how close Model 1 gets without help from the MRE.
The threshold for "accurate" defaults to ≤5% error on total tonnage,
grade, and contained metal. M&I-specific comparison is harder because
most projects in the DB don't store the M&I/Inferred split separately
— `projects.tonnage_mt` is the total resource of whatever
`resource_category` says. We therefore compare total numbers; if those
agree to within 5%, the M&I figures produced by Model 1 are correctly
scaled regardless of the deposit-aware split.

Run:
    python3 scripts/backtest_model_1.py
    python3 scripts/backtest_model_1.py --project fenn_gib
    python3 scripts/backtest_model_1.py --threshold 0.05 --verbose

Fixtures live in `tests/fixtures/backtest/*.json` so the test reproduces
without any network calls. A fixture file is the JSON dump of one row
from `projects` plus the `analogs` array embedded.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make `nodes.*` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Avoid initialising the real Supabase client / LLM clients just to import
# build_model_1 — those modules read env vars at import time. Stub them.
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "stub")
os.environ.setdefault("SUPABASE_ANON_KEY", "stub")
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://stub")
os.environ.setdefault("XAI_API_KEY", "stub")
os.environ.setdefault("EXA_API_KEY", "stub")
os.environ.setdefault("GROK_API_KEY", "stub")
os.environ.setdefault("LANGSMITH_API_KEY", "stub")

# Stub the Supabase rule loader so we don't need a live DB connection. Mirrors
# the autouse fixture in tests/conftest.py: rules are loaded from the same
# seed script that populates the live DB, so the backtest exercises the real
# rule contents without a network call.
from nodes import supabase_ops, rules_engine  # noqa: E402
from scripts.seed_analog_rules import (  # noqa: E402
    ANALOG_SELECTION_RULES, CONFIDENCE_RULES,
)
_ALL_RULES = (
    [{**r, "rule_type": "analog_selection", "active": r.get("active", True)}
     for r in ANALOG_SELECTION_RULES]
    + [{**r, "rule_type": "confidence_adjustment", "active": r.get("active", True)}
       for r in CONFIDENCE_RULES]
)
def _stub_compiled_rules(material, rule_type=None):
    keys = supabase_ops._MATERIAL_TO_RULES_KEYS.get(
        material.strip().lower(), [material.strip().lower()],
    )
    out = [r for r in _ALL_RULES if r.get("source_material") in keys
                                   and r.get("active", True)]
    if rule_type:
        out = [r for r in out if r.get("rule_type") == rule_type]
    return out
supabase_ops.get_compiled_rules = _stub_compiled_rules
rules_engine.get_compiled_rules = _stub_compiled_rules

from nodes.model_builder import build_model_1, _norm_material  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "backtest"


# ── Comparison ────────────────────────────────────────────────────────────────

_PRECIOUS = {"gold", "silver", "platinum", "palladium"}
_TROY_OZ_PER_TONNE = 32150.7466


def _contained_native(tonnage_mt: float, grade: float, material: str) -> float:
    """Contained metal in the material's industry-reporting unit — same
    convention model_builder._contained_t_from_mt uses (oz for precious,
    tonnes for base). Kept local so we can compute the actual MRE's
    contained without importing private helpers."""
    if tonnage_mt is None or grade is None or tonnage_mt <= 0 or grade <= 0:
        return 0.0
    if _norm_material(material) in _PRECIOUS:
        return tonnage_mt * grade * _TROY_OZ_PER_TONNE
    return tonnage_mt * grade * 10000.0


# Back-compat alias for any external scripts that import _contained_t.
_contained_t = _contained_native


def _pct_err(predicted: float, actual: float) -> float:
    """Signed percent error: (predicted - actual) / actual."""
    if actual == 0:
        return float("inf") if predicted != 0 else 0.0
    return (predicted - actual) / actual


def _fmt_pct(p: float) -> str:
    sign = "+" if p > 0 else ""
    return f"{sign}{p*100:.1f}%"


def _color(pct: float, threshold: float) -> str:
    """Green if within threshold, yellow if within 2× threshold, else red."""
    a = abs(pct)
    if a <= threshold:
        return "\033[32m"  # green
    if a <= threshold * 2:
        return "\033[33m"  # yellow
    return "\033[31m"      # red


# ── Backtest one project ─────────────────────────────────────────────────────

def backtest_one(
    fixture: Dict,
    threshold: float = 0.05,
    verbose: bool = False,
) -> Dict:
    """Run Model 1 with the MRE blanked and compare to ground truth.

    Returns a dict of error metrics suitable for aggregating across many
    projects.
    """
    name = fixture["name"]
    material = fixture["material"]
    actual_total_mt = float(fixture["tonnage_mt"])
    actual_grade = float(fixture["grade_value"])
    actual_contained_t = _contained_native(actual_total_mt, actual_grade, material)
    analogs = fixture.get("analogs", []) or []

    # Blank the ground truth so Model 1 can't peek at it. Everything else
    # on the project is data that was knowable BEFORE the MRE was published
    # (deposit type, geometry, location, stage, etc.).
    project = {**fixture, "tonnage_mt": None, "grade_value": None}

    # use_mre=False for backtest: the MRE IS what we're trying to predict.
    # Even though we already blank tonnage_mt/grade_value on the project
    # dict, the explicit flag makes intent clear and survives if a
    # future fixture format starts retaining the MRE elsewhere.
    out = build_model_1(analogs, project, {}, use_mre=False)

    pred_total_mt = out["p50_total_tonnage_mt"]
    pred_grade    = out["p50_grade"]
    pred_contained_t = out["p50_contained_t"]
    pred_mi_mt    = out["mi_tonnage_kt"] / 1000.0
    pred_inf_mt   = out["inferred_tonnage_kt"] / 1000.0
    p10_T = out["p10_total_tonnage_mt"]
    p90_T = out["p90_total_tonnage_mt"]
    p10_G = out["p10_grade"]
    p90_G = out["p90_grade"]
    cv = out.get("cv_contained", 0.0) or 0.0
    tier = out.get("conviction_tier", "—")
    sig = out.get("signal_contributions") or {}
    split = sig.get("split") or {"mi_frac": 0.70, "inf_frac": 0.30,
                                  "source": "fallback"}
    stage_prior = sig.get("stage_prior") or {"mu_logT": 0.0,
                                              "sigma_logT": 0.0,
                                              "source": "—"}
    analog_sig = sig.get("analog") or {
        "mu_logT": 0.0, "sigma_logT": 0.0,
        "mu_logG": 0.0, "sigma_logG": 0.0,
        "rho": 0.0, "n_analogs": 0, "n_pool": 0, "n_eff": 0.0,
    }

    err_T = _pct_err(pred_total_mt, actual_total_mt)
    err_G = _pct_err(pred_grade, actual_grade)
    err_C = _pct_err(pred_contained_t, actual_contained_t)

    pass_T = abs(err_T) <= threshold
    pass_G = abs(err_G) <= threshold
    pass_C = abs(err_C) <= threshold
    overall_pass = pass_T and pass_G and pass_C

    print()
    print("=" * 78)
    print(f"Project: {name}")
    print(f"  Deposit:  {fixture.get('deposit_type')}  "
          f"({fixture.get('mineralization_pattern') or '—'})")
    print(f"  Stage:    {fixture.get('project_stage')}  "
          f"({len(analogs)} analogs)")
    print(f"  Category: {fixture.get('resource_category') or '—'}")
    print("-" * 78)
    print(f"  {'':22s}  {'Predicted':>14s}  {'Actual':>14s}  {'Error':>10s}")
    contained_unit = "oz" if _norm_material(material) in _PRECIOUS else "t"
    for label, pred, actual, err, ok in (
        ("Total tonnage (Mt)", pred_total_mt, actual_total_mt, err_T, pass_T),
        (f"Grade ({fixture.get('grade_unit') or '–'})", pred_grade, actual_grade, err_G, pass_G),
        (f"Contained ({contained_unit})", pred_contained_t, actual_contained_t, err_C, pass_C),
    ):
        c = _color(err, threshold)
        flag = "PASS" if ok else "FAIL"
        print(f"  {label:22s}  {pred:14.3f}  {actual:14.3f}  "
              f"{c}{_fmt_pct(err):>10s} {flag}\033[0m")
    print("-" * 78)
    print(f"  P10–P90 tonnage: {p10_T:.2f} – {p90_T:.2f} Mt  "
          f"(actual {actual_total_mt:.2f} {'inside' if p10_T <= actual_total_mt <= p90_T else 'OUTSIDE'})")
    print(f"  P10–P90 grade:   {p10_G:.3f} – {p90_G:.3f}  "
          f"(actual {actual_grade:.3f} {'inside' if p10_G <= actual_grade <= p90_G else 'OUTSIDE'})")
    print(f"  CV(contained) = {cv:.2f}  →  {tier}")
    print(f"  Split: {int(split['mi_frac']*100)}/{int(split['inf_frac']*100)} M&I/Inf  "
          f"→ M&I {pred_mi_mt:.1f} Mt, Inf {pred_inf_mt:.1f} Mt")

    if verbose:
        print("-" * 78)
        print("  Signal trace:")
        print(f"    analog  μ_logT={analog_sig['mu_logT']:.3f} σ_logT={analog_sig['sigma_logT']:.3f} "
              f"μ_logG={analog_sig['mu_logG']:.3f} σ_logG={analog_sig['sigma_logG']:.3f} "
              f"ρ={analog_sig['rho']:.2f}  n={analog_sig['n_analogs']}/{analog_sig['n_pool']}  "
              f"N_eff={analog_sig['n_eff']:.2f}")
        print(f"    L151    μ_logT={stage_prior['mu_logT']:.3f} σ_logT={stage_prior['sigma_logT']:.3f}  "
              f"(median ≈ {math.exp(stage_prior['mu_logT']):.1f} Mt)")
        print(f"  Analogs (name, tonnage_mt, grade, sim_score):")
        for a in analogs:
            print(f"    {a.get('name','?'):40s}  "
                  f"{a.get('tonnage_mt',0):8.2f}   {a.get('grade_value',0):6.3f}  "
                  f"score={a.get('similarity_score',0):.1f}")

    flag = "PASS" if overall_pass else "FAIL"
    print(f"  Overall: {flag} (threshold ±{threshold*100:.0f}%)")

    return {
        "name": name,
        "actual_total_mt": actual_total_mt,
        "actual_grade": actual_grade,
        "actual_contained_t": actual_contained_t,
        "pred_total_mt": pred_total_mt,
        "pred_grade": pred_grade,
        "pred_contained_t": pred_contained_t,
        "err_T": err_T, "err_G": err_G, "err_C": err_C,
        "pass": overall_pass,
        "cv": cv, "tier": tier,
        "inside_p10_p90": (p10_T <= actual_total_mt <= p90_T) and
                          (p10_G <= actual_grade <= p90_G),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", help="Run only this fixture (basename without .json)")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="Pass threshold as a fraction, default 0.05 (5%%)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print signal trace and analog list per project")
    args = ap.parse_args()

    fixtures = sorted(FIXTURES_DIR.glob("*.json"))
    if args.project:
        fixtures = [FIXTURES_DIR / f"{args.project}.json"]
        if not fixtures[0].exists():
            print(f"No fixture: {fixtures[0]}", file=sys.stderr)
            sys.exit(2)

    results: List[Dict] = []
    for f in fixtures:
        with open(f) as fp:
            fixture = json.load(fp)
        results.append(backtest_one(fixture, threshold=args.threshold,
                                    verbose=args.verbose))

    print()
    print("=" * 78)
    print(f"Summary — {sum(1 for r in results if r['pass'])} / {len(results)} pass "
          f"at ±{args.threshold*100:.0f}%")
    print("=" * 78)
    if results:
        mape_T = sum(abs(r["err_T"]) for r in results) / len(results)
        mape_G = sum(abs(r["err_G"]) for r in results) / len(results)
        mape_C = sum(abs(r["err_C"]) for r in results) / len(results)
        coverage = sum(1 for r in results if r["inside_p10_p90"]) / len(results)
        print(f"  MAPE tonnage:    {mape_T*100:.1f}%")
        print(f"  MAPE grade:      {mape_G*100:.1f}%")
        print(f"  MAPE contained:  {mape_C*100:.1f}%")
        print(f"  P10–P90 coverage: {coverage*100:.0f}%  (target 80%)")

    sys.exit(0 if all(r["pass"] for r in results) else 1)


if __name__ == "__main__":
    main()
