"""
Golden tests — every analog_selection rule must correctly classify a canonical
positive set (must_pick) and a canonical negative set (must_drop).

Run: `python -m pytest tests/test_analog_rules.py -v`

A regression breaks one of these tests immediately. CI should block merging
until every golden case passes. To add a new rule, append to GOLDEN_CASES in
tests/fixtures/golden_analogs.py — no test code changes needed.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from schemas.analog_rule import AnalogRule
from scripts.seed_analog_rules import ANALOG_SELECTION_RULES
from graphs.analog_finder import _build_profile, _cascading_match
from tests.fixtures.golden_analogs import GOLDEN_CASES


# ── Schema-level guards ─────────────────────────────────────────────────────

def test_all_rules_validate_through_schema():
    """Every rule in seed_analog_rules.py must construct cleanly via AnalogRule.

    This is the build-time check: a typo in a slug, an unknown lesson ID, or
    an unknown commodity becomes a test failure here. The same check runs at
    `seed_analog_rules` import, so a deploy can't even start; this duplicates
    it for explicit pytest visibility.
    """
    for raw in ANALOG_SELECTION_RULES:
        AnalogRule(**raw)


def test_every_rule_has_lessons():
    """Rules without applies_lessons are decorative; flag them so authors fix it."""
    missing = [
        r["rule_id"] for r in ANALOG_SELECTION_RULES
        if not r.get("applies_lessons")
    ]
    assert not missing, f"Rules missing applies_lessons: {missing}"


# ── Per-rule golden cases (must_pick / must_drop) ───────────────────────────


def _find_rule(rule_id: str) -> dict:
    for r in ANALOG_SELECTION_RULES:
        if r["rule_id"] == rule_id:
            return r
    raise AssertionError(f"Rule {rule_id!r} not found in ANALOG_SELECTION_RULES")


def _apply_rule_then_cascade(
    rule: dict, target_profile: dict, candidate: dict,
) -> tuple[bool, str | None, list[str]]:
    """Return (passes, dropped_at, reasons) for a candidate against a rule.

    Mirrors the production logic in `combine_filter_score_node` but isolated
    so tests don't need Supabase or LangGraph.
    """
    cand_profile = _build_profile(candidate)
    excluded_subtypes = set(rule.get("excluded_subtypes") or [])
    excluded_modes = set(rule.get("excluded_modes") or [])
    excluded_recovery = set(rule.get("excluded_recovery") or [])
    required_subtypes = set(rule.get("required_subtypes") or [])

    if cand_profile["deposit_subtype"] and cand_profile["deposit_subtype"] in excluded_subtypes:
        return False, "rule_subtype", [f"excluded subtype: {cand_profile['deposit_subtype']}"]
    if cand_profile["mineralization_mode"] and cand_profile["mineralization_mode"] in excluded_modes:
        return False, "rule_mode", [f"excluded mode: {cand_profile['mineralization_mode']}"]
    if cand_profile["recovery_method"] and cand_profile["recovery_method"] in excluded_recovery:
        return False, "rule_recovery", [f"excluded recovery: {cand_profile['recovery_method']}"]
    if required_subtypes and cand_profile["deposit_subtype"]:
        if cand_profile["deposit_subtype"] not in required_subtypes:
            return False, "rule_required_subtype", [
                f"{cand_profile['deposit_subtype']} not in {sorted(required_subtypes)}"
            ]
    if required_subtypes and not cand_profile["deposit_subtype"]:
        has_text = bool(
            (candidate.get("deposit_type") or "").strip()
            or (candidate.get("mineralization_style") or "").strip()
        )
        if not has_text:
            return False, "unenriched", ["no subtype, no deposit_type, no min_style"]

    passes, _pts, _m, _e, reasons, dropped_at = _cascading_match(
        target_profile, cand_profile, rule,
    )
    return passes, dropped_at, reasons


@pytest.mark.parametrize(
    "case",
    GOLDEN_CASES,
    ids=lambda c: c["name"],
)
def test_golden_case_must_pick_must_drop(case: dict):
    rule = _find_rule(case["rule_id"])
    target_profile = _build_profile(case["target"])

    # must_pick — every entry has to pass the full cascade
    pick_failures: list[str] = []
    for candidate in case["must_pick"]:
        passes, dropped_at, reasons = _apply_rule_then_cascade(
            rule, target_profile, candidate,
        )
        if not passes:
            pick_failures.append(
                f"{candidate['name']}: expected PASS, got DROP at {dropped_at} "
                f"({reasons[0] if reasons else '?'})"
            )
    assert not pick_failures, (
        "must_pick candidates wrongly dropped:\n  " + "\n  ".join(pick_failures)
    )

    # must_drop — every entry has to be dropped at one of the expected levels
    drop_failures: list[str] = []
    for candidate, allowed_levels in case["must_drop"]:
        passes, dropped_at, reasons = _apply_rule_then_cascade(
            rule, target_profile, candidate,
        )
        if passes:
            drop_failures.append(
                f"{candidate['name']}: expected DROP, got PASS"
            )
        elif allowed_levels and dropped_at not in allowed_levels:
            drop_failures.append(
                f"{candidate['name']}: dropped at {dropped_at}, expected one of {allowed_levels}"
            )
    assert not drop_failures, (
        "must_drop candidates wrongly passed or dropped at wrong level:\n  "
        + "\n  ".join(drop_failures)
    )


# ── Bootstrap rules-hash determinism guard ──────────────────────────────────


def test_bootstrap_hash_is_deterministic():
    """Two consecutive hash computations on the same code must match."""
    from nodes.bootstrap import _compute_rules_hash
    h1, n1 = _compute_rules_hash()
    h2, n2 = _compute_rules_hash()
    assert h1 == h2
    assert n1 == n2 > 0


# ── Vocabulary single-source-of-truth guard ─────────────────────────────────


def test_no_orphan_subtype_slugs_in_rules():
    """Every required/excluded subtype in every rule must exist in the taxonomy."""
    from nodes.geo_taxonomy import ALL_SUBTYPE_SLUGS
    orphans = set()
    for r in ANALOG_SELECTION_RULES:
        for k in ("required_subtypes", "excluded_subtypes"):
            for slug in r.get(k, []):
                if slug not in ALL_SUBTYPE_SLUGS:
                    orphans.add((r["rule_id"], k, slug))
    assert not orphans, f"Orphan subtype slugs: {orphans}"


def test_field_extractor_uses_taxonomy_directly():
    """Make sure field_extractor doesn't re-define _VALID_SUBTYPES (drift risk)."""
    import nodes.field_extractor as fe
    assert not hasattr(fe, "_VALID_SUBTYPES"), (
        "_VALID_SUBTYPES re-introduced — vocabulary must come from "
        "nodes/geo_taxonomy.py to prevent drift"
    )
