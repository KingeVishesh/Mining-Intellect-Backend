import pytest

from nodes import project_intelligence, supabase_ops
from nodes.parallel_gold_model import _rule_guided_output_schema


def _valid_prediction(rule_hash: str) -> dict:
    block = {
        "tonnage_mt": 10.0,
        "grade_gpt": 1.2,
        "contained_moz": 0.386,
        "tonnage_range_mt": {"p10": 5.0, "p50": 10.0, "p90": 20.0},
        "grade_range_gpt": {"p10": 0.8, "p50": 1.2, "p90": 1.8},
        "contained_range_moz": {"p10": 0.13, "p50": 0.386, "p90": 1.1},
    }
    return {
        "m_and_i": dict(block),
        "inferred": dict(block),
        "anchor_used": "drill_transformation",
        "rule_pack_hash": rule_hash,
        "rule_applications": [
            {"rule": "scale", "applied_to": "tonnage", "effect": "range", "rationale": "test"}
        ],
        "units": {"tonnage": "Mt", "grade": "g/t", "contained": "Moz"},
        "sources_used": [
            {
                "role": "target evidence",
                "used_for": ["scale"],
                "title": "Release",
                "summary": "Pre-MRE drilling release",
                "confidence": "high",
            }
        ],
        "sources_rejected": [],
        "conviction": {"level": "medium", "rationale": "test"},
        "methodology": {"branch": "drill", "top_cut_gpt": 30, "reference_cutoff_gpt": 0.4, "notes": "test"},
        "analogs_used": [],
        "analogs_rejected": [],
    }


def test_blind_cache_key_excludes_mre_truth_fields():
    project = {
        "id": "p1",
        "name": "Blind Gold",
        "material": "Gold",
        "deposit_type": "orogenic",
        "tonnage_mt": 40,
        "grade_value": 1.1,
        "mre_mi_tonnage_mt": 30,
        "mre_inferred_tonnage_mt": 10,
        "drilling_evidence": {"source_url": "https://example.com/pre-mre", "source_date": "2020-01-01"},
    }
    changed_truth = {**project, "tonnage_mt": 400, "mre_mi_tonnage_mt": 300}
    blind_a, _ = project_intelligence.build_intelligence_cache_key(project, [], use_mre=False)
    blind_b, _ = project_intelligence.build_intelligence_cache_key(changed_truth, [], use_mre=False)
    post_a, _ = project_intelligence.build_intelligence_cache_key(project, [], use_mre=True)
    post_b, _ = project_intelligence.build_intelligence_cache_key(changed_truth, [], use_mre=True)
    assert blind_a == blind_b
    assert post_a != post_b


def test_normalize_intelligence_requires_sources_and_hashes_rule_pack():
    result = {
        "project_dossier": {"scale": "district"},
        "deposit_classification": {"archetype": "yilgarn_orogenic_open_pit"},
        "evidence_inventory": [],
        "evidence_gaps": ["drill meters"],
        "analog_logic": {"selected": []},
        "rule_pack": {
            "commodity": "gold",
            "archetype": "yilgarn_orogenic_open_pit",
            "rules": [],
            "scale_logic": {},
            "grade_logic": {},
            "contained_logic": {},
            "mi_inferred_split_logic": {},
            "uncertainty_logic": {},
        },
        "sources_used": [
            {
                "role": "target",
                "used_for": ["classification"],
                "title": "Release",
                "summary": "source",
                "confidence": "high",
            }
        ],
    }
    normalized = project_intelligence.normalize_intelligence_output(
        result,
        project={"material": "Gold"},
        use_mre=False,
    )
    assert normalized["rule_pack_hash"] == project_intelligence.hash_json(result["rule_pack"])
    assert normalized["mode"] == "blind_pre_mre"

    result_without_sources = {**result, "sources_used": []}
    with pytest.raises(ValueError, match="no material sources"):
        project_intelligence.normalize_intelligence_output(
            result_without_sources,
            project={"material": "Gold"},
            use_mre=False,
        )


def test_intelligence_schema_has_no_empty_object_properties():
    def walk(schema, path="root"):
        if not isinstance(schema, dict):
            return
        if schema.get("type") == "object":
            assert schema.get("properties"), f"empty object properties at {path}"
        for key, value in schema.items():
            if isinstance(value, dict):
                walk(value, f"{path}.{key}")
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    walk(item, f"{path}.{key}[{idx}]")

    walk(project_intelligence._intelligence_schema())


def test_new_parallel_schemas_avoid_unsupported_keywords():
    unsupported = {"minLength"}

    def walk(schema, path="root"):
        if not isinstance(schema, dict):
            return
        assert not (unsupported & set(schema)), f"unsupported schema keyword at {path}"
        if schema.get("type") == "object":
            assert schema.get("properties"), f"empty object properties at {path}"
        for key, value in schema.items():
            if isinstance(value, dict):
                walk(value, f"{path}.{key}")
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    walk(item, f"{path}.{key}[{idx}]")

    walk(project_intelligence._intelligence_schema(), "intelligence")
    walk(_rule_guided_output_schema(use_mre=False), "rule_guided_prediction")


def test_project_intelligence_node_uses_cache(monkeypatch):
    cached = {
        "id": "intel-1",
        "project_id": "p1",
        "commodity": "gold",
        "mode": "blind_pre_mre",
        "cache_key": "cache",
        "rule_pack_hash": "abc123",
        "dossier_json": {"name": "cached"},
        "classification_json": {"archetype": "cached"},
        "rule_pack_json": {"archetype": "cached"},
        "quality_json": {"evidence_gaps": []},
        "status": "complete",
    }
    monkeypatch.setattr(project_intelligence, "build_intelligence_cache_key", lambda *_args, **_kwargs: ("cache", {}))
    monkeypatch.setattr(supabase_ops, "get_cached_project_intelligence_run", lambda _cache_key: cached)
    monkeypatch.setattr(
        project_intelligence,
        "_run_parallel_task",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Parallel should not be called")),
    )
    out = project_intelligence.project_intelligence_node({
        "project_id": "p1",
        "project": {"id": "p1", "material": "Gold"},
        "analogs": [],
        "use_mre": False,
    })
    assert out["intelligence_run_id"] == "intel-1"
    assert out["rule_pack_hash"] == "abc123"
    assert out["project_intelligence"]["rule_pack"]["archetype"] == "cached"


def test_project_intelligence_refresh_bypasses_cache(monkeypatch):
    raw_intelligence = {
        "project_dossier": {"scale": "fresh"},
        "deposit_classification": {"archetype": "fresh_archetype"},
        "evidence_inventory": [],
        "evidence_gaps": [],
        "analog_logic": {"selected": [], "rejected": []},
        "rule_pack": {
            "commodity": "gold",
            "archetype": "fresh_archetype",
            "rules": [],
            "scale_logic": {},
            "grade_logic": {},
            "contained_logic": {},
            "mi_inferred_split_logic": {},
            "uncertainty_logic": {},
        },
        "sources_used": [
            {
                "role": "target",
                "used_for": ["classification"],
                "title": "Release",
                "summary": "source",
                "confidence": "high",
            }
        ],
    }
    saved_sources = {}

    monkeypatch.setattr(project_intelligence, "build_intelligence_cache_key", lambda *_args, **_kwargs: ("cache", {}))
    monkeypatch.setattr(
        supabase_ops,
        "get_cached_project_intelligence_run",
        lambda _cache_key: (_ for _ in ()).throw(AssertionError("cache should be skipped")),
    )
    monkeypatch.setattr(project_intelligence.settings, "parallel_api_key", "test-key")
    monkeypatch.setattr(project_intelligence.settings, "parallel_processor", "test-processor")
    monkeypatch.setattr(
        project_intelligence,
        "_run_parallel_task",
        lambda **_kwargs: {"result": raw_intelligence, "run_id": "parallel-1", "status": "completed"},
    )
    monkeypatch.setattr(
        supabase_ops,
        "save_project_intelligence_run",
        lambda row: {**row, "id": "intel-fresh"},
    )
    monkeypatch.setattr(
        supabase_ops,
        "save_project_intelligence_sources",
        lambda **kwargs: saved_sources.update(kwargs),
    )

    out = project_intelligence.project_intelligence_node({
        "project_id": "p1",
        "project": {"id": "p1", "material": "Gold"},
        "analogs": [],
        "use_mre": False,
        "refresh_project_intelligence": True,
    })

    assert out["intelligence_run_id"] == "intel-fresh"
    assert out["project_intelligence"]["rule_pack"]["archetype"] == "fresh_archetype"
    assert saved_sources["intelligence_run_id"] == "intel-fresh"


def test_get_cached_project_intelligence_run_skips_expired_rows(monkeypatch):
    rows = [
        {
            "id": "expired",
            "expires_at": "2000-01-01T00:00:00+00:00",
            "status": "complete",
        },
        {
            "id": "fresh",
            "expires_at": "2999-01-01T00:00:00+00:00",
            "status": "complete",
        },
    ]

    class FakeTable:
        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            return type("Response", (), {"data": rows})()

    class FakeClient:
        def table(self, table_name):
            assert table_name == "project_intelligence_runs"
            return FakeTable()

    monkeypatch.setattr(supabase_ops, "get_client", lambda: FakeClient())
    assert supabase_ops.get_cached_project_intelligence_run("cache")["id"] == "fresh"


def test_validate_rule_guided_prediction_contract():
    intelligence = {"rule_pack_hash": "a" * 64}
    valid = _valid_prediction("a" * 64)
    assert project_intelligence.validate_rule_guided_prediction(
        valid,
        intelligence=intelligence,
        use_mre=False,
    ) == []

    bad = _valid_prediction("b" * 64)
    bad["m_and_i"]["tonnage_range_mt"] = {"p10": 20, "p50": 10, "p90": 5}
    bad["sources_used"] = []
    errors = project_intelligence.validate_rule_guided_prediction(
        bad,
        intelligence=intelligence,
        use_mre=False,
    )
    assert "prediction rule_pack_hash does not match project intelligence" in errors
    assert "prediction has no material sources" in errors
    assert "m_and_i.tonnage_range_mt must satisfy p10 <= p50 <= p90" in errors

    leaked = _valid_prediction("a" * 64)
    leaked["anchor_used"] = "mre_anchored"
    errors = project_intelligence.validate_rule_guided_prediction(
        leaked,
        intelligence=intelligence,
        use_mre=False,
    )
    assert "blind/pre-MRE prediction appears to reference target MRE leakage" in errors


def test_save_project_intelligence_sources_persists_normalized_rows(monkeypatch):
    inserted = {}

    class FakeTable:
        def insert(self, rows):
            inserted["rows"] = rows
            return self

        def execute(self):
            return type("Response", (), {"data": inserted.get("rows")})()

    class FakeClient:
        def table(self, table_name):
            inserted["table"] = table_name
            return FakeTable()

    monkeypatch.setattr(supabase_ops, "get_client", lambda: FakeClient())
    supabase_ops.save_project_intelligence_sources(
        intelligence_run_id="intel-1",
        project_id="p1",
        sources=[
            {
                "role": "target",
                "used_for": ["scale"],
                "title": "Release",
                "source_date": "2024",
                "confidence": "HIGH",
                "summary": "summary",
            }
        ],
    )
    assert inserted["table"] == "project_intelligence_sources"
    assert inserted["rows"][0]["source_date"] == "2024-12-31"
    assert inserted["rows"][0]["confidence"] == "high"
