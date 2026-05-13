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
    excluded_patterns = set(rule.get("excluded_patterns") or [])
    excluded_host_classes = set(rule.get("excluded_host_classes") or [])
    required_subtypes = set(rule.get("required_subtypes") or [])
    required_patterns = set(rule.get("required_patterns") or [])
    required_host_classes = set(rule.get("required_host_classes") or [])

    if cand_profile["deposit_subtype"] and cand_profile["deposit_subtype"] in excluded_subtypes:
        return False, "rule_subtype", [f"excluded subtype: {cand_profile['deposit_subtype']}"]
    if cand_profile["mineralization_mode"] and cand_profile["mineralization_mode"] in excluded_modes:
        return False, "rule_mode", [f"excluded mode: {cand_profile['mineralization_mode']}"]
    if cand_profile["recovery_method"] and cand_profile["recovery_method"] in excluded_recovery:
        return False, "rule_recovery", [f"excluded recovery: {cand_profile['recovery_method']}"]
    if cand_profile["mineralization_pattern"] and cand_profile["mineralization_pattern"] in excluded_patterns:
        return False, "rule_pattern", [f"excluded pattern: {cand_profile['mineralization_pattern']}"]
    if required_patterns and cand_profile["mineralization_pattern"]:
        if cand_profile["mineralization_pattern"] not in required_patterns:
            return False, "rule_required_pattern", [
                f"{cand_profile['mineralization_pattern']} not in {sorted(required_patterns)}"
            ]
    if cand_profile["host_rock_class"] and cand_profile["host_rock_class"] in excluded_host_classes:
        return False, "rule_host_class", [f"excluded host: {cand_profile['host_rock_class']}"]
    if required_host_classes and cand_profile["host_rock_class"]:
        if cand_profile["host_rock_class"] not in required_host_classes:
            return False, "rule_required_host_class", [
                f"{cand_profile['host_rock_class']} not in {sorted(required_host_classes)}"
            ]
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


# ── Profile-strength gate ──────────────────────────────────────────────────


def test_profile_strength_gate_refuses_unenriched_target():
    """
    When the target project has <3 of 6 geological dimensions populated AND
    the rule pins required_subtypes, the cascade must refuse to score and
    return low_confidence=True with a profile_warning. This is the Hat
    Copper "all geological columns null" failure mode — surface it to the
    user instead of returning garbage analogs.
    """
    from graphs.analog_finder import combine_filter_score_node

    sparse_target = {
        "name": "Bare Project", "material": "copper",
        # All geological dims null — simulates an unenriched DB row
        "deposit_type": None, "country": None, "region": None, "district": None,
    }
    rule = _find_rule("analog_sel_copper_porphyry_alkalic")
    state = {
        "project": sparse_target,
        "analog_rule": rule,
        # No target_profile pre-built — combine_filter_score_node will derive
        # an empty one from the sparse project dict.
        "library_analogs": [],
        "exa_analogs": [],
    }
    result = combine_filter_score_node(state)
    assert result["low_confidence"] is True
    assert "geological enrichment" in result["profile_warning"]
    assert result["scored_analogs"] == []


def test_profile_strength_gate_runs_when_target_has_data():
    """A well-enriched target should NOT trigger the strength gate."""
    from graphs.analog_finder import combine_filter_score_node
    from tests.fixtures.golden_analogs import HAT_TARGET, MT_MILLIGAN

    rule = _find_rule("analog_sel_copper_porphyry_alkalic")
    state = {
        "project": HAT_TARGET,
        "analog_rule": rule,
        "library_analogs": [MT_MILLIGAN],
        "exa_analogs": [],
    }
    result = combine_filter_score_node(state)
    # Mt. Milligan should pass — gate doesn't block a well-enriched target
    assert any(a["name"] == "Mt. Milligan" for a in result["scored_analogs"])
    # No profile warning because strength is 5+/6
    assert "profile_warning" not in result or not result.get("profile_warning")


# ── Audit event emission ───────────────────────────────────────────────────


def test_audit_events_emitted_for_every_candidate():
    """Every candidate considered must produce one audit event."""
    from graphs.analog_finder import combine_filter_score_node
    from tests.fixtures.golden_analogs import (
        HAT_TARGET, MT_MILLIGAN, MARIMACA, KAMOA_KAKULA,
    )

    rule = _find_rule("analog_sel_copper_porphyry_alkalic")
    state = {
        "project": HAT_TARGET,
        "analog_rule": rule,
        "library_analogs": [MT_MILLIGAN, MARIMACA, KAMOA_KAKULA],
        "exa_analogs": [],
    }
    result = combine_filter_score_node(state)
    events = result["audit_events"]
    assert len(events) == 3, f"expected 1 event per candidate, got {len(events)}"

    by_name = {e["candidate_name"]: e for e in events}
    assert by_name["Mt. Milligan"]["decision"] == "PASS"
    assert by_name["Marimaca"]["decision"] == "DROP"
    assert by_name["Kamoa-Kakula"]["decision"] == "DROP"

    # Every event must carry the rule_id and at least one resolved lesson
    for e in events:
        assert e["rule_id"] == "analog_sel_copper_porphyry_alkalic"
        assert isinstance(e["lessons"], list)
        assert any(l.get("text") for l in e["lessons"])
        assert e["detected_profile"]
        assert e["reason"]
