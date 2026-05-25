"""Pytest conftest — stub the Supabase rule loader so tests don't hit the network.

Two tests (`test_no_rule_when_deposit_type_and_subtype_both_missing` and
`test_rule_priority_routes_to_most_specific`) call `get_analog_rule` directly,
which in turn calls `nodes.supabase_ops.get_compiled_rules`. That hits the
PostgREST endpoint at SUPABASE_URL, which in CI is the localhost stub set
by tests.yml — connection refused.

We replace the loader with an in-memory version sourced from the same seed
script that populates the DB in production, so the tests exercise the real
rule contents without a network call. Applied autouse so every test gets it.
"""
from __future__ import annotations
import pytest
from typing import Dict, List, Optional


@pytest.fixture(autouse=True)
def _stub_supabase_rules_loader(monkeypatch):
    from nodes import supabase_ops, rules_engine
    from scripts.seed_analog_rules import (
        ANALOG_SELECTION_RULES,
        CONFIDENCE_RULES,
    )

    # Tag each seed row with its rule_type — the DB version of this is set
    # at write time by build_rows() in seed_analog_rules.main().
    analog = [
        {**r, "rule_type": "analog_selection", "active": r.get("active", True)}
        for r in ANALOG_SELECTION_RULES
    ]
    confidence = [
        {**r, "rule_type": "confidence_adjustment", "active": r.get("active", True)}
        for r in CONFIDENCE_RULES
    ]
    all_rules = analog + confidence

    def fake_get_compiled_rules(
        material: str,
        rule_type: Optional[str] = None,
    ) -> List[Dict]:
        keys = supabase_ops._MATERIAL_TO_RULES_KEYS.get(
            material.strip().lower(),
            [material.strip().lower()],
        )
        out = [r for r in all_rules if r.get("source_material") in keys and r.get("active", True)]
        if rule_type:
            out = [r for r in out if r.get("rule_type") == rule_type]
        return out

    # rules_engine imported get_compiled_rules at module load — patch BOTH
    # bindings so callers don't slip through.
    monkeypatch.setattr(supabase_ops, "get_compiled_rules", fake_get_compiled_rules)
    monkeypatch.setattr(rules_engine, "get_compiled_rules", fake_get_compiled_rules)
