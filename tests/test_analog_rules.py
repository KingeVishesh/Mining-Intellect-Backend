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


def test_self_analog_drops_name_containment():
    from graphs.analog_finder import _is_self_analog

    assert _is_self_analog(
        "Cape Ray Shear Zone",
        "Cape Ray Gold Project",
    )


def test_sediment_hosted_intrusion_related_allowed_in_irgs_rule():
    rule = _find_rule("analog_sel_gold_irgs")

    assert "sediment_hosted_general" in rule["required_subtypes"]


def test_sediment_hosted_intrusion_related_uses_intrusion_family():
    profile = _build_profile({
        "name": "Hyland",
        "material": "gold",
        "deposit_type": "sediment hosted intrusion related",
        "deposit_subtype": "sediment_hosted_general",
    })

    assert profile["deposit_type_family"] == "intrusion_related"


def test_missing_taxonomy_yilgarn_gold_routes_to_orogenic_vein():
    from graphs.analog_finder import _derive_rule_inputs

    material, deposit_type, subtype, pattern = _derive_rule_inputs({
        "name": "Spargoville",
        "material": "Gold",
        "tectonic_belt": "yilgarn",
        "mining_method_class": "open_pit_selective",
        "tonnage_mt": 3.0,
        "grade_value": 1.4,
    })

    assert material == "Gold"
    assert deposit_type == "orogenic gold"
    assert subtype == "orogenic_general"
    assert pattern == "vein_hosted"


def test_missing_taxonomy_yilgarn_gold_builds_orogenic_profile():
    profile = _build_profile({
        "name": "Spargoville",
        "material": "Gold",
        "tectonic_belt": "yilgarn",
        "mining_method_class": "open_pit_selective",
        "tonnage_mt": 3.0,
        "grade_value": 1.4,
    })

    assert profile["deposit_type_family"] == "orogenic"
    assert profile["deposit_subtype"] == "orogenic_general"
    assert profile["mineralization_pattern"] == "vein_hosted"
    assert profile["tectonic_belt"] == "yilgarn"


def test_irgs_sibling_subtypes_soft_pass_oxidation_variants():
    rule = _find_rule("analog_sel_gold_irgs")
    target = _build_profile({
        "name": "Hyland",
        "material": "gold",
        "deposit_type": "sediment hosted intrusion related",
        "deposit_subtype": "sediment_hosted_general",
        "mineralization_mode": "supergene_oxide",
        "mineralization_pattern": "disseminated_bulk",
        "tectonic_belt": "yukon_tintina",
        "mining_method_class": "heap_leach_pad",
        "project_stage_class": "pea",
        "tonnage_mt": 100.0,
        "grade_value": 0.65,
        "grade_unit": "g/t Au",
    })
    candidate = _build_profile({
        "name": "Fort Knox",
        "material": "gold",
        "deposit_type": "intrusion-related gold system",
        "deposit_subtype": "irgs_general",
        "mineralization_mode": "primary_sulfide",
        "mineralization_pattern": "stockwork",
        "tectonic_belt": "yukon_tintina",
        "mining_method_class": "open_pit_bulk",
        "project_stage_class": "feasibility",
        "tonnage_mt": 160.0,
        "grade_value": 0.45,
        "grade_unit": "g/t Au",
    })

    passes, _pts, _matched, _evaluated, reasons, dropped_at = _cascading_match(
        target, candidate, rule,
    )

    assert passes, reasons
    assert dropped_at is None
    assert any("mode soft-pass" in reason for reason in reasons)


def test_irgs_drops_mature_mine_scale_for_mid_scale_target():
    rule = _find_rule("analog_sel_gold_irgs")
    target = _build_profile({
        "name": "Hyland",
        "material": "gold",
        "deposit_type": "sediment hosted intrusion related",
        "deposit_subtype": "sediment_hosted_general",
        "mineralization_mode": "supergene_oxide",
        "mineralization_pattern": "disseminated_bulk",
        "tectonic_belt": "yukon_tintina",
        "mining_method_class": "heap_leach_pad",
        "project_stage_class": "pea",
        "tonnage_mt": 15.2,
        "grade_value": 0.935,
        "grade_unit": "g/t Au",
    })
    eagle = _build_profile({
        "name": "Eagle Gold",
        "material": "gold",
        "deposit_type": "intrusion-related gold system",
        "deposit_subtype": "irgs_general",
        "mineralization_mode": "free_milling_oxide",
        "mineralization_pattern": "stockwork",
        "tectonic_belt": "yukon_tintina",
        "mining_method_class": "open_pit_bulk",
        "project_stage_class": "feasibility",
        "tonnage_mt": 145.0,
        "grade_value": 0.65,
        "grade_unit": "g/t Au",
    })

    passes, _pts, _matched, _evaluated, reasons, dropped_at = _cascading_match(
        target, eagle, rule,
    )

    assert not passes
    assert dropped_at == "L5.5"
    assert any("scale mismatch" in reason for reason in reasons)


def test_exa_search_node_can_be_skipped():
    from graphs.analog_finder import exa_search_node

    assert exa_search_node({"skip_exa": True, "project": {}}) == {"exa_analogs": []}


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


def test_profile_strength_gate_drops_to_relaxed_mode_when_unenriched():
    """
    Ruoppa regression guard: when the target's profile strength is below
    the rule's min_profile_strength, the cascade drops to RELAXED MODE
    instead of refusing to score. It returns low_confidence=True + a
    profile_warning, but DOES still try to score candidates with the
    rule's required_* filters disabled (exclusions still apply).
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
        "library_analogs": [],
        "exa_analogs": [],
    }
    result = combine_filter_score_node(state)
    # Empty candidate pool → still zero analogs, but the IMPORTANT guard is
    # that low_confidence is flagged and a warning is surfaced — not silent
    # failure. With no candidates supplied we can't assert any are returned;
    # with candidates, the relaxed mode would return them (see the dedicated
    # test_thinly_enriched_project_still_gets_analogs_in_relaxed_mode test).
    assert result["low_confidence"] is True
    # When the strict pre-gate fires it emits a RELAXED_MODE warning. The
    # warning text mentions either "geological enrichment" or "relaxed mode"
    # — both phrasings have appeared as we iterated the message.
    if result.get("profile_warning"):
        w = result["profile_warning"].lower()
        assert "enrichment" in w or "relaxed mode" in w or "missing" in w


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


def test_no_rule_returns_low_confidence_with_warning():
    """When get_analog_rule returns None, the cascade refuses to score and
    returns a profile_warning instead of running family-only matching."""
    from graphs.analog_finder import combine_filter_score_node
    state = {
        "project": {"name": "Mystery", "material": "vanadium",
                     "deposit_type": "shale-hosted vanadium"},
        "analog_rule": None,
        "target_profile": {"material": "vanadium"},
        "library_analogs": [{"name": "Some V Analog", "material": "vanadium"}],
        "exa_analogs": [],
    }
    result = combine_filter_score_node(state)
    assert result["low_confidence"] is True
    assert result["scored_analogs"] == []
    assert "research is incomplete" in result["profile_warning"]


def test_no_rule_when_deposit_type_and_subtype_both_missing():
    """Strict contract: get_analog_rule returns None when the project has
    neither deposit_type nor deposit_subtype. Previously fell through to a
    generic_fallback rule that produced wrong analogs for ~30% of gold
    projects (Cartier-Cadillac surfaced this in 2026-05)."""
    from nodes.rules_engine import get_analog_rule
    assert get_analog_rule("gold", None, None, None) is None
    assert get_analog_rule("gold", "", "", "") is None


def test_no_generic_fallback_rule_exists():
    """The generic_fallback rule was removed; it must not return from any
    routing call. If any commodity needs a fallback in the future, route
    None instead — the cascade refuses to score and surfaces the gap."""
    from scripts.seed_analog_rules import ANALOG_SELECTION_RULES
    fb = [r for r in ANALOG_SELECTION_RULES
           if r.get("rule_id", "").endswith("_generic_fallback")]
    assert not fb, f"generic_fallback rules must be removed: {[r['rule_id'] for r in fb]}"


def test_load_node_auto_runs_research_when_rule_missing(monkeypatch):
    """When the project lacks deposit_type AND deposit_subtype, the load
    node should auto-invoke project_research, reload the project, and
    return a rule on the second pass — no manual re-research required."""
    from graphs import analog_finder
    from nodes import supabase_ops

    project_id = "test-auto-research-id"
    thin = {
        "id": project_id, "name": "Thin Gold Project", "material": "gold",
        "deposit_type": None, "deposit_subtype": None,
        "mineralization_pattern": None,
    }
    enriched = {
        **thin,
        "deposit_type": "orogenic gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "mining_method_class": "underground_vein",
        "tectonic_belt": "abitibi", "country": "Canada",
    }
    call_count = {"get_project": 0, "research_invoke": 0}

    def fake_get_project(pid):
        call_count["get_project"] += 1
        return thin if call_count["get_project"] == 1 else enriched

    class FakeResearchGraph:
        def invoke(self, _input):
            call_count["research_invoke"] += 1
            return {"saved": True, "error": None}

    monkeypatch.setattr(supabase_ops, "get_project", fake_get_project)
    # _trigger_project_research imports lazily — patch the underlying module
    import graphs.project_research as project_research_mod
    monkeypatch.setattr(project_research_mod, "graph", FakeResearchGraph())

    result = analog_finder.load_project_and_rule_node({"project_id": project_id})

    assert call_count["research_invoke"] == 1, "project_research must be invoked once"
    assert call_count["get_project"] == 2,    "project must be reloaded after research"
    assert result["analog_rule"] is not None, "rule lookup should succeed on second pass"
    assert result["research_attempted"] is True, "sentinel must be set to prevent looping"


def test_load_node_does_not_auto_research_when_subtype_already_present(monkeypatch):
    """Auto-research must NOT fire when deposit_subtype is set — that means
    enrichment ran. If no rule maps to it, that's a taxonomy gap, not a
    data gap, and another research pass won't help."""
    from graphs import analog_finder
    from nodes import supabase_ops

    project_id = "test-no-auto-research"
    project = {
        "id": project_id, "name": "Vanadium Project", "material": "vanadium",
        "deposit_type": "shale-hosted vanadium",
        "deposit_subtype": None,  # truly unknown sub-type
    }
    invoke_count = {"n": 0}

    monkeypatch.setattr(supabase_ops, "get_project", lambda pid: project)
    import graphs.project_research as project_research_mod

    class FakeResearchGraph:
        def invoke(self, _input):
            invoke_count["n"] += 1
            return {"saved": True}

    monkeypatch.setattr(project_research_mod, "graph", FakeResearchGraph())

    result = analog_finder.load_project_and_rule_node({"project_id": project_id})

    # deposit_type is populated → auto-research path must NOT fire even if
    # subtype is None and no vanadium rule exists.
    assert invoke_count["n"] == 0
    assert result["analog_rule"] is None


def test_belt_hard_filter_drops_cross_belt_orogenic_gold():
    """Cartier-Cadillac (Abitibi) must NOT pair with Brucejack (BC
    Quesnel-Stikine) even though both are vein-hosted greenstone-orogenic
    underground gold mines. They belong to different belt-compatibility
    groups (Archean greenstone vs Cordilleran arc)."""
    from graphs.analog_finder import _build_profile, _cascading_match
    cartier = _build_profile({
        "name": "Cartier-Cadillac", "material": "gold",
        "deposit_type": "Gold-bearing structures",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "tectonic_belt": "abitibi",
        "mining_method_class": "underground_vein",
        "country": "Canada", "region": "Quebec", "district": "Abitibi",
    })
    brucejack = _build_profile({
        "name": "Brucejack", "material": "gold",
        "deposit_type": "orogenic vein-hosted gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "tectonic_belt": "bc_quesnel_stikine",
        "mining_method_class": "underground_vein",
        "country": "Canada", "region": "British Columbia",
        "tonnage_mt": 16.0, "grade_value": 8.4, "grade_unit": "g/t Au",
    })
    rule = _find_rule("analog_sel_gold_orogenic_vein")
    passes, _, _, _, reasons, dropped = _cascading_match(cartier, brucejack, rule)
    assert not passes
    assert dropped == "L2.5"
    assert any("belt incompatible" in r for r in reasons)


def test_belt_hard_filter_drops_guiana_shield_for_abitibi():
    """An Abitibi target should not draw Guiana Shield analogs (Aurora /
    Toroparu / Rosebel). Different compatibility group: Archean greenstone
    Superior craton vs Birimian-equivalent Guiana shield."""
    from graphs.analog_finder import _build_profile, _cascading_match
    abitibi = _build_profile({
        "name": "Some Abitibi Vein Au", "material": "gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "tectonic_belt": "abitibi",
    })
    guiana = _build_profile({
        "name": "Toroparu", "material": "gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "tectonic_belt": "guiana_shield",
    })
    rule = _find_rule("analog_sel_gold_orogenic_vein")
    passes, _, _, _, _, dropped = _cascading_match(abitibi, guiana, rule)
    assert not passes
    assert dropped == "L2.5"


def test_belt_hard_filter_allows_in_group_match():
    """Abitibi and Yilgarn are both Archean greenstone — they SHOULD pass
    L2.5. Granny Smith / Sunrise Dam / etc. are legitimate analogs for an
    Abitibi target when the library lacks in-belt matches."""
    from graphs.analog_finder import _build_profile, _cascading_match
    abitibi = _build_profile({
        "name": "Cartier-Cadillac", "material": "gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "tectonic_belt": "abitibi",
        "tonnage_mt": 45.0,
    })
    yilgarn = _build_profile({
        "name": "Sunrise Dam", "material": "gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "tectonic_belt": "yilgarn",
        "tonnage_mt": 50.0, "grade_value": 2.5, "grade_unit": "g/t Au",
    })
    rule = _find_rule("analog_sel_gold_orogenic_vein")
    passes, _, _, _, _, dropped = _cascading_match(abitibi, yilgarn, rule)
    assert passes, f"Yilgarn analog should pass L2.5 for Abitibi target; dropped={dropped}"


def test_sub_trend_detection_cortez_from_eureka_simpson_park():
    """Red Hill is in Simpson Park Mountains, Eureka County, Nevada. The
    Cortez Trend extends through this area — detect_sub_trend should
    return 'cortez_trend', not get hijacked by the 'eureka county'
    keyword on battle_mountain_eureka."""
    from nodes.geo_taxonomy import detect_sub_trend
    assert detect_sub_trend(
        None, "Nevada",
        "Northern Simpson Park Mountains, Eureka County, Nevada, USA",
    ) == "cortez_trend"


def test_sub_trend_detection_distinguishes_trends():
    """SUB_TRENDS keyword routing covers each major Carlin sub-trend."""
    from nodes.geo_taxonomy import detect_sub_trend
    assert detect_sub_trend("Cortez Hills District",
                              "Lander County, Nevada") == "cortez_trend"
    assert detect_sub_trend("Carlin Trend",
                              "Elko County, Nevada") == "carlin_trend"
    assert detect_sub_trend("Getchell Trend",
                              "Humboldt County, Nevada") == "getchell_trend"
    assert detect_sub_trend("Battle Mountain District",
                              "Lander County, Nevada") == "battle_mountain_eureka"
    assert detect_sub_trend("Ruby Hill / Eureka District",
                              "Eureka County, Nevada") == "battle_mountain_eureka"
    assert detect_sub_trend("Pequop Mountains",
                              "Elko County, Nevada") == "pequop_long_canyon"
    assert detect_sub_trend("Round Mountain",
                              "Nye County, Nevada") == "walker_lane_au"
    assert detect_sub_trend("Black Pine",
                              "Cassia County, Idaho") == "oquirrh_black_pine"
    assert detect_sub_trend("Ottawa", "Canada") is None


def test_sub_trend_detection_tanzania_lake_victoria():
    """Buckreef + Geita + Bulyanhulu + North Mara are all in the Tanzania
    Lake Victoria Goldfields. Without this sub-trend, Tanzanian orogenic
    targets pulled Abitibi-greenstone analogs (same belt-compatibility
    group, but cross-craton) via L6.5 + 0 — no in-country boost.
    With this sub-trend, Tanzanian analogs get the +40 L6.5 bonus."""
    from nodes.geo_taxonomy import detect_sub_trend
    assert detect_sub_trend(
        None, None, None, "Buckreef Gold Project",
    ) == "tanzania_lake_victoria"
    assert detect_sub_trend(
        "Geita Greenstone Belt", "Geita Region", "Tanzania",
    ) == "tanzania_lake_victoria"
    assert detect_sub_trend(
        None, "Mara Region", "North Mara mining district, Tanzania",
    ) == "tanzania_lake_victoria"
    assert detect_sub_trend(
        "Lake Victoria Goldfields", None, None,
    ) == "tanzania_lake_victoria"


def test_sub_trend_detection_asankrangwa_distinct_from_ashanti():
    """Abore is in Asankrangwa belt (granite-hosted, Ghana) and was
    being routed to ashanti_belt (sediment-hosted) because the previous
    keywords lumped 'asankrangwa' under ashanti_belt. Split fixes this
    so the granite-hosted style cohort (Chirano, Edikan, Essakane) gets
    surfaced via the Asankrangwa Exa hint."""
    from nodes.geo_taxonomy import detect_sub_trend
    assert detect_sub_trend(
        "Asankrangwa gold belt", "Ashanti Region",
        "Asankrangwa belt, Ghana",
    ) == "asankrangwa_belt"
    assert detect_sub_trend(
        None, "Ashanti Region", "Abore project, Ghana",
    ) == "asankrangwa_belt"
    assert detect_sub_trend(
        None, "Ghana", "Obotan / Nkran district",
    ) == "asankrangwa_belt"
    # Ashanti belt proper — politicial "Ashanti Region" alone should NOT
    # claim it; needs "ashanti belt" phrasing or a specific mine.
    assert detect_sub_trend(
        None, "Ashanti Region", "Obuasi gold mine area",
    ) == "ashanti_belt"
    assert detect_sub_trend(
        None, None, "Ashanti gold belt classic shear",
    ) == "ashanti_belt"
    # Bare political "Ashanti Region" with no other context → None
    # (preferable to wrong-belt routing). Adjusted from the previous
    # bare-keyword false positive.
    assert detect_sub_trend(
        None, "Ashanti Region", None,
    ) is None


def test_sub_trend_detection_james_bay_eeyou_istchee():
    """Opinaca, Cheechoo, Éléonore — all James Bay / Eeyou Istchee
    sub-province. Without this sub-trend a Targa Opinaca-style target
    was pulling Tintina RIRGS analogs (Eagle, Valley, Golden Summit)
    despite being in northern Quebec."""
    from nodes.geo_taxonomy import detect_sub_trend
    assert detect_sub_trend(
        None, "Quebec",
        "Opinaca subprovince, James Bay, Quebec, Canada",
    ) == "james_bay_eeyou_istchee"
    assert detect_sub_trend(
        "La Grande Subprovince", "Quebec",
        "James Bay region, Quebec",
    ) == "james_bay_eeyou_istchee"
    assert detect_sub_trend(
        "Eastmain", "Quebec", None,
    ) == "james_bay_eeyou_istchee"


def test_sub_trend_semihard_filter_keeps_top4_in_trend():
    """Buckreef audit (2026-05-22): when a target has a sub-trend AND
    >=3 in-sub-trend candidates pass the cascade, top-4 must contain
    ONLY in-sub-trend candidates — cross-belt-group same-archean
    candidates (Canadian Malartic UG, Westwood) must be diverted to
    NEAR_MISS, not backfill top-4."""
    from graphs.analog_finder import combine_filter_score_node
    target_profile = {
        "material": "gold",
        "deposit_type_family": "orogenic",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "mineralization_mode": "primary_sulfide",
        "tectonic_belt": "tanzania_archean",
        "sub_trend": "tanzania_lake_victoria",
        "metal_suite": "au_only",
        "mining_method_class": "underground_vein",
        "host_rock_class": None,
        "project_stage_class": None,
        "alteration_signature": None,
        "recovery_method": None,
        "resource_category_class": None,
        "resource_compliance_standard": None,
        "resource_vintage_year": None,
        "grade_value": 2.8, "grade_unit": "g/t Au", "tonnage_mt": 25.0,
        "country": "tanzania", "district": "Lake Victoria Goldfields",
        "host_rock": "", "mineralization_style": "",
        "source_url": None, "company_name": "", "project_id": "test-buckreef",
    }
    # 4 Tanzanian candidates (in-trend) + 2 cross-belt-group Abitibi
    library = [
        # In-trend Tanzanian candidates — all should be in top 4
        {"name": "Bulyanhulu", "material": "gold",
         "deposit_subtype": "greenstone_orogenic",
         "mineralization_pattern": "vein_hosted",
         "tectonic_belt": "tanzania_archean",
         "mining_method_class": "underground_vein",
         "district": "Sukumaland Greenstone Belt, Tanzania",
         "tonnage_mt": 30, "grade_value": 8.0, "grade_unit": "g/t Au",
         "source": "library"},
        {"name": "Geita", "material": "gold",
         "deposit_subtype": "greenstone_orogenic",
         "mineralization_pattern": "vein_hosted",
         "tectonic_belt": "tanzania_archean",
         "mining_method_class": "underground_vein",
         "district": "Geita greenstone belt, Tanzania",
         "tonnage_mt": 100, "grade_value": 3.5, "grade_unit": "g/t Au",
         "source": "library"},
        {"name": "North Mara", "material": "gold",
         "deposit_subtype": "greenstone_orogenic",
         "mineralization_pattern": "vein_hosted",
         "tectonic_belt": "tanzania_archean",
         "mining_method_class": "underground_vein",
         "district": "Musoma-Mara Greenstone Belt, Tanzania",
         "tonnage_mt": 40, "grade_value": 3.0, "grade_unit": "g/t Au",
         "source": "library"},
        {"name": "Nyanzaga", "material": "gold",
         "deposit_subtype": "greenstone_orogenic",
         "mineralization_pattern": "vein_hosted",
         "tectonic_belt": "tanzania_archean",
         "mining_method_class": "underground_vein",
         "district": "Sukumaland Greenstone Belt, Tanzania",
         "tonnage_mt": 60, "grade_value": 3.0, "grade_unit": "g/t Au",
         "source": "library"},
        # Cross-belt-group Abitibi candidates — must be filtered out of top-4
        {"name": "Westwood Mine", "material": "gold",
         "deposit_subtype": "greenstone_orogenic",
         "mineralization_pattern": "vein_hosted",
         "tectonic_belt": "abitibi",
         "mining_method_class": "underground_vein",
         "district": "Bousquet Camp Quebec",
         "tonnage_mt": 22, "grade_value": 5.4, "grade_unit": "g/t Au",
         "source": "library"},
        {"name": "Canadian Malartic - Odyssey UG", "material": "gold",
         "deposit_subtype": "greenstone_orogenic",
         "mineralization_pattern": "vein_hosted",
         "tectonic_belt": "abitibi",
         "mining_method_class": "underground_vein",
         "district": "Malartic Quebec",
         "tonnage_mt": 110, "grade_value": 2.5, "grade_unit": "g/t Au",
         "source": "library"},
    ]
    state = {
        "project": {"id": "test-buckreef", "name": "Test Buckreef",
                     "material": "gold"},
        "analog_rule": _find_rule("analog_sel_gold_orogenic_vein"),
        "target_profile": target_profile,
        "library_analogs": library,
        "exa_analogs": [],
    }
    result = combine_filter_score_node(state)
    top_names = [a.get("name") for a in result["scored_analogs"]]
    cross_belt_names = {"Westwood Mine", "Canadian Malartic - Odyssey UG"}
    leak = cross_belt_names.intersection(top_names)
    assert not leak, (
        f"Cross-belt-group Abitibi candidates leaked into top-4: {leak}. "
        f"Top: {top_names}"
    )
    # The 4 Tanzanian candidates should all be in top 4
    tanzanian = {"Bulyanhulu", "Geita", "North Mara", "Nyanzaga"}
    assert tanzanian.issubset(set(top_names)), (
        f"Tanzanian in-trend candidates missing from top-4. "
        f"Top: {top_names}"
    )
    # And there should be NEAR_MISS / SUB_TREND_FILTERED events for the dropped ones
    audit = result.get("audit_events") or []
    filtered_events = [e for e in audit if e.get("level") == "SUB_TREND_FILTERED"]
    filtered_names = {e.get("candidate_name") for e in filtered_events}
    assert cross_belt_names.issubset(filtered_names), (
        f"Cross-belt candidates should appear in SUB_TREND_FILTERED audit "
        f"events. Filtered: {filtered_names}"
    )


def test_sub_trend_semihard_filter_falls_back_when_thin():
    """When fewer than 3 in-sub-trend candidates pass the cascade, the
    semi-hard filter should NOT engage — fall back to allowing cross-
    belt-group same-archean candidates to backfill top-4. This protects
    obscure-geology projects from returning empty cohorts."""
    from graphs.analog_finder import combine_filter_score_node
    target_profile = {
        "material": "gold",
        "deposit_type_family": "orogenic",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "mineralization_mode": "primary_sulfide",
        "tectonic_belt": "tanzania_archean",
        "sub_trend": "tanzania_lake_victoria",
        "metal_suite": "au_only",
        "mining_method_class": "underground_vein",
        "host_rock_class": None,
        "project_stage_class": None,
        "alteration_signature": None,
        "recovery_method": None,
        "resource_category_class": None,
        "resource_compliance_standard": None,
        "resource_vintage_year": None,
        "grade_value": 2.8, "grade_unit": "g/t Au", "tonnage_mt": 25.0,
        "country": "tanzania", "district": "Lake Victoria Goldfields",
        "host_rock": "", "mineralization_style": "",
        "source_url": None, "company_name": "", "project_id": "test-thin",
    }
    library = [
        # Only 2 in-trend — below the threshold of 3
        {"name": "Bulyanhulu", "material": "gold",
         "deposit_subtype": "greenstone_orogenic",
         "mineralization_pattern": "vein_hosted",
         "tectonic_belt": "tanzania_archean",
         "mining_method_class": "underground_vein",
         "district": "Sukumaland Greenstone Belt, Tanzania",
         "tonnage_mt": 30, "grade_value": 8.0, "grade_unit": "g/t Au",
         "source": "library"},
        {"name": "North Mara", "material": "gold",
         "deposit_subtype": "greenstone_orogenic",
         "mineralization_pattern": "vein_hosted",
         "tectonic_belt": "tanzania_archean",
         "mining_method_class": "underground_vein",
         "district": "Musoma-Mara Greenstone Belt, Tanzania",
         "tonnage_mt": 40, "grade_value": 3.0, "grade_unit": "g/t Au",
         "source": "library"},
        # Cross-belt-group Abitibi candidates — should backfill since
        # in-trend coverage is thin
        {"name": "Westwood Mine", "material": "gold",
         "deposit_subtype": "greenstone_orogenic",
         "mineralization_pattern": "vein_hosted",
         "tectonic_belt": "abitibi",
         "mining_method_class": "underground_vein",
         "district": "Bousquet Camp Quebec",
         "tonnage_mt": 22, "grade_value": 5.4, "grade_unit": "g/t Au",
         "source": "library"},
        {"name": "Canadian Malartic - Odyssey UG", "material": "gold",
         "deposit_subtype": "greenstone_orogenic",
         "mineralization_pattern": "vein_hosted",
         "tectonic_belt": "abitibi",
         "mining_method_class": "underground_vein",
         "district": "Malartic Quebec",
         "tonnage_mt": 110, "grade_value": 2.5, "grade_unit": "g/t Au",
         "source": "library"},
    ]
    state = {
        "project": {"id": "test-thin", "name": "Test Thin",
                     "material": "gold"},
        "analog_rule": _find_rule("analog_sel_gold_orogenic_vein"),
        "target_profile": target_profile,
        "library_analogs": library,
        "exa_analogs": [],
    }
    result = combine_filter_score_node(state)
    top_names = [a.get("name") for a in result["scored_analogs"]]
    # With only 2 in-trend, semi-hard filter should NOT engage.
    # Top-4 may include cross-belt-group backfills.
    assert len(top_names) >= 3, (
        f"Thin-trend case should still produce ≥3 picks via backfill. "
        f"Got {len(top_names)}: {top_names}"
    )


def test_irgs_subtype_detected_from_bare_intrusion_related():
    """Targa Opinaca's deposit_type was the string 'Intrusion Related'
    (no hyphen, no 'gold' suffix). The previous detect_subtype only
    matched 'intrusion-related gold' with the gold word, so Opinaca
    fell through to no-subtype and the cascade routed to fallback."""
    from nodes.geo_taxonomy import detect_subtype
    assert detect_subtype("Intrusion Related", None, None, None) == "irgs_general"
    assert detect_subtype("intrusion related", None, None, None) == "irgs_general"
    assert detect_subtype("Intrusion-Related", None, None, None) == "irgs_general"


def test_sub_trend_detection_normalizes_curly_apostrophe():
    """Cartier-Cadillac's location_name in the DB is "Val-d’Or, ..."
    with a curly apostrophe (U+2019), not the straight ASCII one. Without
    Unicode normalization the keyword 'val-d\\'or' silently missed —
    target.sub_trend was None and the L6.5 ranking bonus never fired.
    The audit revealed this; the fix normalizes curly quotes to ASCII
    before substring matching."""
    from nodes.geo_taxonomy import detect_sub_trend
    # Curly apostrophe (U+2019)
    assert detect_sub_trend(
        None, None, "Val-d’Or, Abitibi, Quebec, Canada",
    ) == "cadillac_break_valdor"
    # Straight ASCII apostrophe — must still work
    assert detect_sub_trend(
        None, None, "Val-d'Or, Abitibi, Quebec, Canada",
    ) == "cadillac_break_valdor"


def test_sub_trend_detection_distinguishes_abitibi_camps():
    """Abitibi sub-camps route distinctly so Cartier-Cadillac (Val-d'Or
    on the Cadillac Break) doesn't pair with Bousquet VMS-overprint
    veins or Casa Berardi BIF-stockwork as if they were the same camp."""
    from nodes.geo_taxonomy import detect_sub_trend
    assert detect_sub_trend(
        "Abitibi", "Quebec",
        "Val-d'Or, Abitibi, Quebec, Canada",
    ) == "cadillac_break_valdor"
    assert detect_sub_trend(
        "Val d'Or Camp Quebec", None, None,
    ) == "cadillac_break_valdor"
    assert detect_sub_trend(
        "Bousquet Camp Quebec", None, None,
    ) == "bousquet_camp"
    assert detect_sub_trend(
        "Bousquet Doyon Camp Quebec", None, None,
    ) == "bousquet_camp"
    assert detect_sub_trend(
        "Casa Berardi Quebec", None, None,
    ) == "casa_berardi_camp"
    assert detect_sub_trend(
        "Kirkland Lake Ontario", None, None,
    ) == "kirkland_lake_camp"
    assert detect_sub_trend(
        "Red Lake greenstone belt Ontario", None, None,
    ) == "red_lake_camp"
    assert detect_sub_trend(
        "Detour Lake Abitibi Ontario", None, None,
    ) == "detour_trend"
    assert detect_sub_trend(
        "Hemlo greenstone belt Ontario", None, None,
    ) == "hemlo_camp"
    assert detect_sub_trend(
        "Timmins Porcupine Ontario", None, None,
    ) == "timmins_camp"
    assert detect_sub_trend(
        "Malartic Quebec", None, None,
    ) == "cadillac_break_valdor"


def test_cartier_cadillac_in_camp_analog_outranks_other_abitibi_camps():
    """Cartier-Cadillac (Cadillac Break, Val-d'Or) target. A Lamaque-
    style in-camp candidate must outrank a Bousquet-camp Westwood-style
    candidate of identical cascade quality. This is the Cadillac audit
    fix: VMS-overprint Bousquet veins are different geology from
    Cadillac-Break shear-hosted veins despite being in the same belt."""
    from graphs.analog_finder import _build_profile, _cascading_match
    cartier = _build_profile({
        "name": "Cartier-Cadillac", "material": "gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "host_rock_class": "volcanic_mafic",
        "tectonic_belt": "abitibi",
        "metal_suite": "au_only",
        "mining_method_class": "underground_vein",
        "country": "Canada", "region": "Quebec",
        "district": "Abitibi",
        "location_name": "Val-d'Or, Abitibi, Quebec, Canada",
        "tonnage_mt": 45.0,
    })
    lamaque_in_camp = _build_profile({
        "name": "Lamaque Complex", "material": "gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "host_rock_class": "volcanic_mafic",
        "tectonic_belt": "abitibi",
        "metal_suite": "au_only",
        "mining_method_class": "underground_vein",
        "country": "Canada", "region": "Quebec",
        "district": "Val d'Or Camp Quebec",
        "tonnage_mt": 30.0, "grade_value": 6.0, "grade_unit": "g/t Au",
    })
    westwood_off_camp = _build_profile({
        "name": "Westwood Mine", "material": "gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "host_rock_class": "volcanic_mafic",
        "tectonic_belt": "abitibi",
        "metal_suite": "au_only",
        "mining_method_class": "underground_vein",
        "country": "Canada", "region": "Quebec",
        "district": "Bousquet Camp Quebec",
        "tonnage_mt": 22.0, "grade_value": 5.4, "grade_unit": "g/t Au",
    })
    rule = _find_rule("analog_sel_gold_orogenic_vein")

    l_pass, l_rank, _, _, _, l_drop = _cascading_match(cartier, lamaque_in_camp, rule)
    w_pass, w_rank, _, _, _, w_drop = _cascading_match(cartier, westwood_off_camp, rule)

    assert l_pass, f"Lamaque must pass for Cartier (dropped at {l_drop})"
    assert w_pass, f"Westwood must pass for Cartier (dropped at {w_drop})"
    assert l_rank > w_rank, (
        f"In-camp Lamaque (rank={l_rank}) must outrank off-camp Westwood "
        f"(rank={w_rank}) for a Cadillac-Break target — sub-camp +25 should "
        f"clearly elevate same-camp analogs."
    )


def test_orogenic_vein_tonnage_tolerance_allows_camp_scale_variation():
    """Cartier-Cadillac (45 Mt) vs Red Lake (8 Mt) — both in-belt Archean
    greenstone orogenic vein gold, but in different sub-camps and very
    different scale. 5.6× tonnage ratio used to drop Red Lake at L5.5.
    Loosened to 10× post-audit so industry-canonical Abitibi-scale
    variations all pass."""
    from graphs.analog_finder import _build_profile, _cascading_match
    target = _build_profile({
        "material": "gold", "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "tectonic_belt": "abitibi",
        "mining_method_class": "underground_vein",
        "tonnage_mt": 45.0,
    })
    red_lake = _build_profile({
        "name": "Red Lake", "material": "gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "tectonic_belt": "abitibi",
        "mining_method_class": "underground_vein",
        "tonnage_mt": 8.0, "grade_value": 13.0, "grade_unit": "g/t Au",
    })
    rule = _find_rule("analog_sel_gold_orogenic_vein")
    passes, _, _, _, _, dropped = _cascading_match(target, red_lake, rule)
    assert passes, f"Red Lake (8 Mt) must pass for 45 Mt target; dropped at {dropped}"


def test_exa_query_includes_cadillac_break_hint():
    """Verify that the Cadillac Break hint names Sigma-Lamaque, Beaufor,
    Chimo, Goldex — Exa's anchor points for in-camp canonicals."""
    from nodes.exa_search import _SUB_TREND_HINTS
    hint = _SUB_TREND_HINTS["cadillac_break_valdor"]
    assert "Cadillac" in hint
    assert "Val-d'Or" in hint
    assert "Sigma-Lamaque" in hint
    assert "Beaufor" in hint
    assert "Chimo" in hint


def test_l6_5_sub_trend_bonus_lifts_in_trend_above_out_of_trend():
    """For a Cortez-Trend target, an in-trend Carlin analog must rank
    above an out-of-trend Carlin analog of otherwise equal quality —
    that's the Red Hill fix: Goldrush (Cortez) ranks above Lookout
    Mountain (Battle Mountain-Eureka) for a Red Hill (Cortez) target."""
    from graphs.analog_finder import _build_profile, _cascading_match
    red_hill = _build_profile({
        "name": "Red Hill", "material": "gold",
        "deposit_type": "Sediment-hosted Carlin-style",
        "deposit_subtype": "carlin_general",
        "mineralization_pattern": "disseminated_bulk",
        "host_rock_class": "carbonate_sediment",
        "tectonic_belt": "great_basin_carlin",
        "metal_suite": "au_only",
        "country": "USA", "region": "Nevada",
        "location_name": "Northern Simpson Park Mountains, Eureka County, Nevada",
        "tonnage_mt": 253.0, "grade_value": 0.51, "grade_unit": "g/t Au",
        "project_stage_class": "exploration",
    })
    cortez_in_trend = _build_profile({
        "name": "Cortez Hills", "material": "gold",
        "deposit_type": "Carlin-style sediment-hosted",
        "deposit_subtype": "carlin_general",
        "mineralization_pattern": "disseminated_bulk",
        "host_rock_class": "carbonate_sediment",
        "tectonic_belt": "great_basin_carlin",
        "metal_suite": "au_only",
        "mining_method_class": "open_pit_bulk",
        "country": "USA", "region": "Nevada",
        "location_name": "Cortez Hills District, Lander County, Nevada",
        "tonnage_mt": 90.0, "grade_value": 3.0, "grade_unit": "g/t Au",
    })
    lookout_off_trend = _build_profile({
        "name": "Lookout Mountain", "material": "gold",
        "deposit_type": "Carlin-style sediment-hosted",
        "deposit_subtype": "carlin_general",
        "mineralization_pattern": "disseminated_bulk",
        "host_rock_class": "carbonate_sediment",
        "tectonic_belt": "great_basin_carlin",
        "metal_suite": "au_only",
        "mining_method_class": "open_pit_bulk",
        "country": "USA", "region": "Nevada",
        "location_name": "Eureka District, Eureka County, Nevada",
        "tonnage_mt": 90.0, "grade_value": 0.5, "grade_unit": "g/t Au",
    })
    rule = _find_rule("analog_sel_gold_carlin_super_large")

    c_pass, c_rank, _, _, _, c_drop = _cascading_match(red_hill, cortez_in_trend, rule)
    l_pass, l_rank, _, _, _, l_drop = _cascading_match(red_hill, lookout_off_trend, rule)

    assert c_pass, f"Cortez Hills must pass the cascade for Red Hill (dropped at {c_drop})"
    assert l_pass, f"Lookout Mountain must pass the cascade for Red Hill (dropped at {l_drop})"
    assert c_rank > l_rank, (
        f"In-trend Cortez Hills (rank={c_rank}) should outrank off-trend "
        f"Lookout Mountain (rank={l_rank}) for Cortez-Trend Red Hill target"
    )


def test_carlin_grade_tolerance_passes_in_trend_higher_grade():
    """Goldstrike-style 4 g/t in-trend Carlin must pass for a low-grade
    bulk Carlin target (0.5 g/t). Pre-fix: 4× ratio dropped it. Post-fix:
    10× ratio admits geologically valid in-trend canonicals at any of the
    grade tiers Carlin deposits span (0.3 – 14 g/t)."""
    from graphs.analog_finder import _build_profile, _cascading_match
    target = _build_profile({
        "material": "gold", "deposit_subtype": "carlin_general",
        "mineralization_pattern": "disseminated_bulk",
        "tectonic_belt": "great_basin_carlin",
        "tonnage_mt": 253.0, "grade_value": 0.51, "grade_unit": "g/t Au",
    })
    cand = _build_profile({
        "name": "Goldstrike-style", "material": "gold",
        "deposit_subtype": "carlin_general",
        "mineralization_pattern": "disseminated_bulk",
        "tectonic_belt": "great_basin_carlin",
        "tonnage_mt": 180.0, "grade_value": 4.0, "grade_unit": "g/t Au",
    })
    rule = _find_rule("analog_sel_gold_carlin_super_large")
    passes, _, _, _, _, dropped = _cascading_match(target, cand, rule)
    assert passes, f"Goldstrike-style 4 g/t Carlin must pass; dropped at {dropped}"


def test_exa_query_includes_sub_trend_hint_for_cortez_target():
    """Verify that the Exa query builder injects the sub-trend hint when
    a target resolves to a sub-trend. This is the root fix for Red Hill —
    the system asks Exa for Cortez Trend canonicals instead of generic
    Carlin projects."""
    from nodes.exa_search import _SUB_TREND_HINTS
    assert "Cortez Trend" in _SUB_TREND_HINTS["cortez_trend"]
    assert "Goldrush" in _SUB_TREND_HINTS["cortez_trend"]
    assert "Cortez Hills" in _SUB_TREND_HINTS["cortez_trend"]
    assert "Pipeline" in _SUB_TREND_HINTS["cortez_trend"]


def test_belt_hard_filter_skips_when_candidate_belt_unknown():
    """If the candidate has no tectonic_belt detected, L2.5 should NOT drop
    it — the rest of the cascade still applies and the lack of a belt
    becomes a rank penalty at L6 only. We don't punish poorly enriched
    library entries here."""
    from graphs.analog_finder import _build_profile, _cascading_match
    target = _build_profile({
        "name": "Target", "material": "gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "tectonic_belt": "abitibi",
    })
    cand = _build_profile({
        "name": "Mystery Au", "material": "gold",
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
        "tectonic_belt": None,
    })
    rule = _find_rule("analog_sel_gold_orogenic_vein")
    passes, _, _, _, _, dropped = _cascading_match(target, cand, rule)
    assert passes, f"Unknown-belt candidate should not be dropped at L2.5; dropped={dropped}"


def test_rule_priority_routes_to_most_specific():
    """alkalic_porphyry + stockwork pattern should route to the alkalic rule
    (priority 200) ahead of the generic copper porphyry (priority 100)."""
    from nodes.rules_engine import get_analog_rule
    rule = get_analog_rule("copper", "alkalic porphyry copper-gold",
                            "alkalic_porphyry", "stockwork")
    assert rule is not None
    assert rule["rule_id"] == "analog_sel_copper_porphyry_alkalic"


def test_mining_method_hard_filter_drops_ug_vein_for_op_carlin():
    """A Carlin super-large target must drop an underground-vein analog at L4.8."""
    from graphs.analog_finder import _build_profile, _cascading_match
    from tests.fixtures.golden_analogs import BLACK_PINE_TARGET
    rule = _find_rule("analog_sel_gold_carlin_super_large")
    target = _build_profile(BLACK_PINE_TARGET)
    ug_vein_cand = _build_profile({
        "name": "Some UG Vein Au", "material": "gold",
        "deposit_subtype": "carlin_general",  # bypass subtype
        "mineralization_pattern": "disseminated_bulk",  # bypass pattern
        "mining_method_class": "underground_vein",
        "tonnage_mt": 600.0, "grade_value": 0.3, "grade_unit": "g/t Au",
    })
    passes, _, _, _, reasons, dropped = _cascading_match(target, ug_vein_cand, rule)
    assert not passes
    # Could be L4.8 (cascade), or rule_mining_method (rule-driven)
    # Both are correct outcomes — we hit a mining-method gate
    assert dropped in ("L4.8", "rule_mining_method", "rule_required_mining_method")


def test_vintage_filter_drops_historical_resource():
    """Pre-2010 vintage should be dropped at L4.95 when the rule sets min_resource_year."""
    from graphs.analog_finder import _build_profile, _cascading_match
    from tests.fixtures.golden_analogs import BLACK_PINE_TARGET
    rule = _find_rule("analog_sel_gold_carlin_super_large")
    target = _build_profile(BLACK_PINE_TARGET)
    old_cand = _build_profile({
        "name": "1985 Historical Carlin", "material": "gold",
        "deposit_subtype": "carlin_general",
        "mineralization_pattern": "disseminated_bulk",
        "mining_method_class": "open_pit_bulk",
        "resource_compliance_standard": "ni_43_101",
        "resource_vintage_year": 1985,
        "tonnage_mt": 500.0, "grade_value": 0.3, "grade_unit": "g/t Au",
    })
    passes, _, _, _, reasons, dropped = _cascading_match(target, old_cand, rule)
    assert not passes
    assert dropped == "L4.95"


def test_compliance_filter_drops_press_release():
    """A press-release-grade resource should never pass."""
    from graphs.analog_finder import _build_profile, _cascading_match
    from tests.fixtures.golden_analogs import BLACK_PINE_TARGET
    rule = _find_rule("analog_sel_gold_carlin_super_large")
    target = _build_profile(BLACK_PINE_TARGET)
    pr_cand = _build_profile({
        "name": "Press-Release Resource", "material": "gold",
        "deposit_subtype": "carlin_general",
        "mineralization_pattern": "disseminated_bulk",
        "mining_method_class": "open_pit_bulk",
        "resource_compliance_standard": "press_release",
        "resource_vintage_year": 2020,
        "tonnage_mt": 400.0, "grade_value": 0.3, "grade_unit": "g/t Au",
    })
    passes, _, _, _, _, dropped = _cascading_match(target, pr_cand, rule)
    assert not passes
    assert dropped == "L4.95"


def test_grade_tolerance_drops_wildly_off_grade():
    """An 8 g/t Au analog (28×) for a 0.3 g/t Au super-large Carlin target
    drops at L5.6 grade mismatch."""
    from graphs.analog_finder import _build_profile, _cascading_match
    from tests.fixtures.golden_analogs import BLACK_PINE_TARGET
    rule = _find_rule("analog_sel_gold_carlin_super_large")
    target = _build_profile(BLACK_PINE_TARGET)
    high_g_cand = _build_profile({
        "name": "High-Grade Carlin", "material": "gold",
        "deposit_subtype": "carlin_general",
        "mineralization_pattern": "disseminated_bulk",
        "mining_method_class": "open_pit_bulk",
        "resource_compliance_standard": "ni_43_101",
        "resource_vintage_year": 2020,
        "tonnage_mt": 500.0, "grade_value": 8.0, "grade_unit": "g/t Au",
    })
    passes, _, _, _, _, dropped = _cascading_match(target, high_g_cand, rule)
    assert not passes
    assert dropped == "L5.6"


def test_hallucination_guard_drops_sourceless_exa():
    """Exa-sourced candidate with no source_url AND no tonnage/grade should
    be flagged as suspected_hallucination in audit and dropped pre-cascade."""
    from graphs.analog_finder import combine_filter_score_node
    from tests.fixtures.golden_analogs import HAT_TARGET
    rule = _find_rule("analog_sel_copper_porphyry_alkalic")
    state = {
        "project": HAT_TARGET, "project_id": "test-id",
        "analog_rule": rule,
        "target_profile": None,  # let the node derive it
        "library_analogs": [],
        "exa_analogs": [{
            "name": "Phantom Copper Project", "material": "copper",
            "source": "exa",
            # No URL, no tonnage, no grade → suspected hallucination
        }],
    }
    result = combine_filter_score_node(state)
    halluc_events = [e for e in result["audit_events"]
                     if e["level"] == "suspected_hallucination"]
    assert len(halluc_events) == 1
    assert halluc_events[0]["candidate_name"] == "Phantom Copper Project"


def test_library_search_accepts_sibling_subtypes():
    """True North regression: orogenic-vein rule lists multiple acceptable
    subtypes (greenstone, turbidite, bif_hosted, orogenic_general).
    The library filter must return ALL of them, not only an exact match on
    the target's specific subtype — otherwise turbidite-hosted Fosterville
    is wrongly dropped for a greenstone-hosted target."""
    import inspect
    from nodes.supabase_ops import get_approved_analogs
    sig = inspect.signature(get_approved_analogs)
    assert "deposit_subtypes" in sig.parameters
    src = inspect.getsource(get_approved_analogs)
    # Must use `.in_()` for multi-slug case so siblings pass
    assert ".in_(" in src and "analog_deposit_subtype" in src


def test_library_search_routes_by_subtype_not_deposit_type_string():
    """Black Pine regression: get_approved_analogs used ILIKE with the full
    multi-word deposit_type string, which never matched analogs whose freeform
    deposit_type differed slightly (Carlin-type vs Carlin-style, etc.).
    Library matching MUST use the controlled-vocab deposit_subtype slug."""
    import inspect
    from nodes.supabase_ops import get_approved_analogs
    src = inspect.getsource(get_approved_analogs)
    assert "deposit_subtype" in inspect.signature(get_approved_analogs).parameters
    assert "analog_deposit_subtype" in src, (
        "Library search must filter by analog_deposit_subtype exact match — "
        "freeform ILIKE on full deposit_type doesn't survive text variations"
    )


def test_stage_compatibility_allows_late_stage_analogs_for_early_target():
    """Black Pine regression: exploration target should accept production-stage
    analogs (Marigold etc.) — that's the gold standard for analog-based
    resource modeling. The OPPOSITE direction (early analog for late target)
    is the one that's restrictive."""
    from nodes.geo_taxonomy import stage_compatible
    # Late-stage analog for early target: ALLOWED
    assert stage_compatible("exploration", "production")
    assert stage_compatible("exploration", "feasibility")
    assert stage_compatible("resource_inferred", "production")
    assert stage_compatible("pea", "feasibility")
    # Same stage: ALLOWED
    assert stage_compatible("production", "production")
    # Early analog for late target: BLOCKED
    assert not stage_compatible("production", "exploration")
    assert not stage_compatible("feasibility", "resource_inferred")


def test_carlin_priority_over_sediment_hosted_cu():
    """Black Pine regression: 'Carlin-style sediment-hosted disseminated gold'
    must classify as carlin_general, not sediment_hosted_general. The phrase
    'sediment-hosted' is standard Carlin terminology in NI 43-101 reports."""
    from nodes.geo_taxonomy import detect_subtype
    assert detect_subtype("Carlin-style sediment-hosted disseminated gold") == "carlin_general"
    assert detect_subtype("Carlin-type sediment-hosted gold",
                            mineralization_style="disseminated invisible gold") == "carlin_general"
    # Pure sediment-hosted Cu (no Carlin keyword) still routes correctly
    assert detect_subtype("sediment-hosted stratiform copper",
                            mineralization_style="redbed Cu") == "redbed_cu"


def test_great_basin_belt_includes_idaho():
    """Black Pine sits on Oquirrh Formation in southern Idaho — geologically
    part of the Great Basin Carlin host stratigraphy. The belt must include
    Idaho (and Oquirrh district term) or the cascade can't route Carlin
    projects there. Utah is genuinely ambiguous (Bingham Laramide vs Great
    Basin) so we don't assert a single answer for Utah-alone."""
    from nodes.geo_taxonomy import detect_belt
    assert detect_belt("USA", "Idaho", "Oquirrh Formation") == "great_basin_carlin"
    assert detect_belt("USA", "Nevada", "Carlin Trend") == "great_basin_carlin"
    # Utah with Oquirrh district resolves to great_basin_carlin
    assert detect_belt("USA", "Utah", "Oquirrh Formation") == "great_basin_carlin"


def test_mode_compatible_substring_check():
    """Regression: `"mixed" in (target, candidate)` is tuple membership and
    never fires for `mixed_oxide_sulfide`. The check must use substring."""
    from nodes.geo_taxonomy import mode_compatible
    # mixed_oxide_sulfide must be compatible with pure sulfide or pure oxide
    assert mode_compatible("supergene_oxide", "mixed_oxide_sulfide")
    assert mode_compatible("mixed_oxide_sulfide", "supergene_oxide")
    assert mode_compatible("primary_sulfide", "mixed_oxide_sulfide")
    assert mode_compatible("mixed_oxide_sulfide", "primary_sulfide")
    # Pure-sulfide vs pure-oxide is still incompatible (sanity)
    assert not mode_compatible("primary_sulfide", "supergene_oxide")
    assert not mode_compatible("supergene_oxide", "primary_sulfide")


def test_finnish_lapland_routes_to_fennoscandian():
    """Ruoppa-style Finnish Lapland gold project must detect Fennoscandian belt
    so the orogenic-vein cascade has enough profile strength to score."""
    from nodes.geo_taxonomy import detect_belt
    assert detect_belt("Finland", "Lapland", "Central Lapland Greenstone Belt") == "fennoscandian"
    assert detect_belt("Finland", "Kittilä", None) == "fennoscandian"
    # Country-only fallback for Finland (no region match) — still fennoscandian
    assert detect_belt("Finland", None, "Unknown District") == "fennoscandian"


def test_thinly_enriched_project_still_gets_analogs_in_relaxed_mode():
    """Ruoppa regression: a thinly-enriched orogenic gold target with profile
    strength below the rule's min must still return analogs — degrading to
    relaxed mode with low_confidence, not silent 0."""
    from graphs.analog_finder import combine_filter_score_node, _build_profile
    rule = _find_rule("analog_sel_gold_orogenic_vein")
    # Force a target with strength below rule.min_profile_strength=5 by
    # dropping the belt detection (use an unrecognised country)
    sparse_target = {
        "id": "test", "name": "Sparse Orogenic Test", "material": "gold",
        "deposit_type": "Orogenic", "country": "Madagascar",  # not in our taxonomy
        "deposit_subtype": "greenstone_orogenic",
        "mineralization_pattern": "vein_hosted",
    }
    profile = _build_profile(sparse_target)
    candidate = {
        "name": "Detour Lake", "material": "gold", "source": "library",
        "deposit_type": "orogenic shear-hosted",
        "mineralization_style": "shear-hosted sulphide vein",
        "country": "Canada", "district": "Abitibi",
        "processing_method": "CIL", "tonnage_mt": 350, "grade_value": 1.0,
        "grade_unit": "g/t Au", "source_url": "http://example.com",
    }
    result = combine_filter_score_node({
        "project_id": "test", "project": sparse_target,
        "analog_rule": rule, "target_profile": profile,
        "library_analogs": [candidate], "exa_analogs": [],
    })
    # Either we return analogs in relaxed mode, OR the target's strength was
    # high enough to run strict — both are acceptable, but EMPTY is not.
    assert len(result["scored_analogs"]) >= 1, (
        f"Sparse target produced zero analogs — regression. "
        f"scored_analogs={result['scored_analogs']}, "
        f"low_confidence={result.get('low_confidence')}, "
        f"warning={result.get('profile_warning')}"
    )


def test_self_analog_by_project_id():
    """Same project_id on both sides should be detected even with different names."""
    from graphs.analog_finder import _is_self_analog
    assert _is_self_analog("Hat Copper", "Doubleview Hat Project",
                             project_id="abc-123", candidate_project_id="abc-123")
    assert not _is_self_analog("Hat Copper", "Mt. Milligan",
                                 project_id="abc-123", candidate_project_id="xyz-789")


def test_self_analog_by_distinctive_project_token():
    """Same-property project variants should not leak into the analog cohort."""
    from graphs.analog_finder import _is_self_analog

    assert _is_self_analog(
        "Banyan Gold - AurMac Gold Project",
        "AurMac Project (Powerline/Airstrip)",
    )
    assert not _is_self_analog(
        "Cadillac Gold Project",
        "Canadian Malartic - Odyssey UG",
    )


def test_low_grade_open_pit_gold_uses_bulk_pattern():
    """Large low-grade open-pit gold should not be modeled as narrow vein-hosted."""
    from graphs.analog_finder import _build_profile

    profile = _build_profile({
        "name": "Mandilla Gold Project",
        "material": "Gold",
        "deposit_type": "orogenic gold",
        "deposit_subtype": "orogenic_general",
        "mining_method": "Open-pit",
        "tonnage_mt": 41.5,
        "grade_value": 1.1,
    })

    assert profile["mineralization_pattern"] == "disseminated_bulk"


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
