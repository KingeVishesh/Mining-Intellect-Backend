from __future__ import annotations

import json
from datetime import date

from nodes.parallel_gold_model import (
    _build_prompt,
    _apply_blind_moderate_drilling_fallback_calibration,
    _format_analogs_block,
    _format_project_block,
    _apply_blind_evidence_scale_guard,
    _apply_blind_broad_bulk_scale_floor,
    _apply_blind_single_irgs_scale_floor,
    _apply_blind_underground_carlin_single_window,
    _apply_blind_open_pit_carlin_geometry_window,
    _apply_blind_carlin_heap_grade_tonnage_window,
    _apply_blind_guiana_orogenic_open_pit_window,
    _apply_blind_open_pit_orogenic_proxy_window,
    _apply_blind_porphyry_bulk_no_geometry_window,
    _apply_blind_large_andean_heap_window,
    _apply_blind_mature_high_sulfidation_window,
    _apply_blind_small_underground_vein_window,
    _apply_blind_underground_orogenic_no_evidence_window,
    _apply_blind_yukon_irgs_near_surface_window,
    _blind_result_mentions_mre_anchor,
    _clean_blind_analogs,
    _evidence_mentions_target_mre,
    _replace_blind_mre_leak_estimate,
    _output_schema,
    _blind_local_fallback_estimate,
    _replace_placeholder_blind_estimate,
    parallel_gold_model_node,
)


def test_blind_project_block_redacts_mre_and_future_cached_drilling():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "tonnage_mt": 123.4,
        "grade_value": 1.7,
        "mre_mi_tonnage_mt": 50,
        "mre_mi_grade": 2.0,
        "mre_inferred_tonnage_mt": 73.4,
        "mre_inferred_grade": 1.5,
        "mre_data_source": {"as_of_date": "2025-05-15"},
        "drilling_evidence": {
            "total_meters_drilled": 250000,
            "weighted_grade_g_t": 1.9,
            "report_cutoff_date": "2025-05-15",
        },
    }

    payload = json.loads(_format_project_block(project, use_mre=False))

    assert "tonnage_mt" not in payload
    assert "grade_value" not in payload
    assert "mre_mi_tonnage_mt" not in payload
    assert "mre_data_source" not in payload
    assert payload["drilling_evidence"]["redacted"] is True


def test_blind_project_block_keeps_verified_pre_mre_evidence():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "mre_date": "2024-12-31",
        "drilling_evidence": {
            "queried_pre_mre_cutoff": "2024-12-31",
            "report_cutoff_date": "2024-12-31",
            "total_meters_drilled": 12000,
        },
    }

    payload = json.loads(_format_project_block(project, use_mre=False))

    assert payload["drilling_evidence"]["total_meters_drilled"] == 12000
    assert "redacted" not in payload["drilling_evidence"]


def test_blind_project_block_redacts_mre_sourced_evidence_even_with_cutoff_flag():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "mre_date": "2024-12-31",
        "drilling_evidence": {
            "queried_pre_mre_cutoff": "2024-12-31",
            "report_cutoff_date": "2024-12-31",
            "total_meters_drilled": 12000,
            "source_url": "https://example.com/project-mineral-resource-estimate-technical-report.pdf",
        },
    }

    payload = json.loads(_format_project_block(project, use_mre=False))

    assert payload["drilling_evidence"]["redacted"] is True
    assert "MRE-tainted" in payload["drilling_evidence"]["reason"]


def test_blind_project_block_redacts_low_confidence_geometry_only_evidence():
    project = {
        "name": "Weak Geometry",
        "material": "gold",
        "mre_date": "2025-12-31",
        "drilling_evidence": {
            "queried_pre_mre_cutoff": "2025-12-31",
            "confidence": "low",
            "strike_length_m": 600,
            "down_dip_extent_m": 725,
            "source_url": "https://example.com/pre-mre-presentation.pdf",
        },
    }

    payload = json.loads(_format_project_block(project, use_mre=False))

    assert payload["drilling_evidence"]["redacted"] is True
    assert "too weak" in payload["drilling_evidence"]["reason"]


def test_blind_project_block_keeps_low_confidence_evidence_with_grade_or_meters():
    project = {
        "name": "Useful Low Confidence",
        "material": "gold",
        "mre_date": "2025-12-31",
        "drilling_evidence": {
            "queried_pre_mre_cutoff": "2025-12-31",
            "confidence": "low",
            "total_meters_drilled": 8400,
            "weighted_grade_g_t": 3.3,
            "source_url": "https://example.com/pre-mre-drilling.pdf",
        },
    }

    payload = json.loads(_format_project_block(project, use_mre=False))

    assert payload["drilling_evidence"]["total_meters_drilled"] == 8400
    assert payload["drilling_evidence"]["weighted_grade_g_t"] == 3.3


def test_blind_analog_block_filters_post_target_cutoff_sources():
    analogs = [
        {
            "name": "Valid Before",
            "tonnage_mt": 10,
            "grade_value": 1.5,
            "data_source": {"as_of_date": "2024-12-31"},
        },
        {
            "name": "Future Leak",
            "tonnage_mt": 100,
            "grade_value": 3.0,
            "data_source": {"as_of_date": "2026-01-01"},
        },
    ]

    from datetime import date

    payload = json.loads(_format_analogs_block(analogs, cutoff_date=date(2025, 5, 15)))

    assert [a["name"] for a in payload] == ["Valid Before"]


def test_blind_analog_hygiene_removes_stale_self_analog():
    project = {"name": "AuMEGA Metals - Cape Ray Shear Zone"}
    analogs = [
        {"name": "Cape Ray Gold Project", "tonnage_mt": 9.7, "grade_value": 1.96},
        {"name": "Valentine Gold Project", "tonnage_mt": 64.6, "grade_value": 1.9},
    ]

    cleaned = _clean_blind_analogs(project, analogs, None)

    assert [a["name"] for a in cleaned] == ["Valentine Gold Project"]


def test_blind_analog_hygiene_redacts_mre_sourced_analog_drilling():
    project = {"name": "Some Target"}
    analogs = [
        {
            "name": "Valid Analog",
            "tonnage_mt": 10,
            "grade_value": 1,
            "drilling_evidence": {
                "total_meters_drilled": 10000,
                "source_url": "https://example.com/analog-ni-43-101-technical-report.pdf",
            },
        }
    ]

    cleaned = _clean_blind_analogs(project, analogs, None)

    assert cleaned[0]["drilling_evidence"]["redacted"] is True


def test_blind_prompt_names_exact_pre_mre_cutoff():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "mre_data_source": {"as_of_date": "2025-05-15"},
    }

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=True)

    assert "Treat 2025-05-15 as the target MRE cutoff date" in prompt
    assert "use ONLY information published BEFORE 2025-05-15" in prompt


def test_blind_prompt_silently_discards_target_mre_leaks():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "mre_data_source": {"as_of_date": "2025-05-15"},
    }

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=True)

    assert "discard that source silently" in prompt
    assert "Do NOT quote" in prompt
    assert "anywhere in the JSON output" in prompt
    assert "say so in `methodology.notes`" not in prompt


def test_blind_prompt_rejects_post_cutoff_target_technical_reports():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "mre_data_source": {"as_of_date": "2025-05-15"},
    }

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=True)

    assert "do NOT use target resource pages" in prompt
    assert "NI 43-101" in prompt
    assert "dated on or after the cutoff" in prompt
    assert "even if it restates older drill" in prompt


def test_blind_prompt_requires_grade_proxy_fallback():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "mre_data_source": {"as_of_date": "2025-05-15"},
    }

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=True)

    assert "GRADE-PROXY FALLBACK" in prompt
    assert "Do NOT return null grade solely because the target lacks" in prompt
    assert "grade_proxy=analog_resource_grade" in prompt


def test_blind_prompt_requires_tonnage_proxy_fallback():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "mre_data_source": {"as_of_date": "2025-05-15"},
    }

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=True)

    assert "TONNAGE-PROXY FALLBACK" in prompt
    assert "Do NOT return null tonnage solely because" in prompt
    assert "tonnage_proxy=analog_resource_tonnage" in prompt


def test_blind_prompt_requires_target_enrichment_before_analog_only():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "mre_data_source": {"as_of_date": "2025-05-15"},
    }

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=True)

    assert "TARGET ENRICHMENT" in prompt
    assert "MUST search for those pre-MRE target disclosures" in prompt
    assert "Only choose `analog_only_fallback` after documenting" in prompt


def test_blind_schema_forces_numeric_estimate_and_excludes_mre_anchor():
    schema = _output_schema(use_mre=False)

    assert schema["properties"]["m_and_i"]["properties"]["tonnage_mt"]["type"] == "number"
    assert schema["properties"]["m_and_i"]["properties"]["tonnage_mt"]["exclusiveMinimum"] == 0
    assert schema["properties"]["m_and_i"]["properties"]["grade_gpt"]["type"] == "number"
    assert "mre_anchored" not in schema["properties"]["anchor_used"]["enum"]


def test_placeholder_blind_estimate_replaced_with_analog_fallback():
    result = {
        "m_and_i": {"tonnage_mt": 3, "grade_gpt": 1, "contained_moz": 0.096},
        "inferred": {"tonnage_mt": 3, "grade_gpt": 1, "contained_moz": 0.096},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": "placeholder"},
        "analogs_used": [],
        "analogs_rejected": [],
    }
    analogs = [
        {"tonnage_mt": 20, "grade_value": 1.0},
        {"tonnage_mt": 40, "grade_value": 1.2},
        {"tonnage_mt": 60, "grade_value": 1.4},
    ]

    replaced = _replace_placeholder_blind_estimate(result, analogs)

    assert replaced["m_and_i"]["tonnage_mt"] > 3
    assert replaced["inferred"]["tonnage_mt"] > 3
    assert "local_guard=replaced_placeholder" in replaced["methodology"]["notes"]


def test_blind_mre_leak_estimate_replaced_before_persistence():
    result = {
        "m_and_i": {"tonnage_mt": 11.3, "grade_gpt": 0.93, "contained_moz": 0.337},
        "inferred": {"tonnage_mt": 3.9, "grade_gpt": 0.95, "contained_moz": 0.118},
        "anchor_used": "drill_transformation",
        "methodology": {
            "branch": "mre_anchored",
            "notes": "Anchored to company MRE and reported split.",
        },
        "conviction": {"level": "high", "rationale": "public MRE summary"},
        "analogs_used": [],
        "analogs_rejected": [],
    }

    replaced = _replace_blind_mre_leak_estimate(result, [])

    assert replaced["anchor_used"] == "analog_only_fallback"
    assert replaced["m_and_i"]["tonnage_mt"] == 0.001
    assert "local_guard=rejected_blind_mre_leak" in replaced["methodology"]["notes"]


def test_blind_leak_detector_flags_dated_mre_basis():
    assert _blind_result_mentions_mre_anchor({
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {
            "level": "very_low",
            "rationale": (
                "The resource figures are derived from a 2023 PEA and 2022 MRE, "
                "providing the highest level of regulatory-compliant data."
            ),
        },
        "analogs_used": [],
        "analogs_rejected": [],
    })


def test_blind_node_does_not_auto_enable_web_discovery_for_thin_cohort(monkeypatch):
    from config import settings

    captured = {}

    def fake_run_parallel_task(*, prompt, output_schema):
        captured["prompt"] = prompt
        return {
            "m_and_i": {"tonnage_mt": 1.0, "grade_gpt": 1.0, "contained_moz": 0.032},
            "inferred": {"tonnage_mt": 1.0, "grade_gpt": 1.0, "contained_moz": 0.032},
            "anchor_used": "analog_only_fallback",
            "methodology": {"branch": "analog_only_fallback", "notes": ""},
            "conviction": {"level": "very_low", "rationale": "thin supplied cohort"},
            "analogs_used": [],
            "analogs_rejected": [],
            "sources": [],
        }

    monkeypatch.setattr(settings, "parallel_api_key", "test-key")
    monkeypatch.setattr("nodes.parallel_gold_model._run_parallel_task", fake_run_parallel_task)

    out = parallel_gold_model_node({
        "project": {"name": "Thin Blind", "material": "gold"},
        "analogs": [{"name": "Analog A", "tonnage_mt": 5, "grade_value": 1.0}],
        "use_mre": False,
        "find_analogs": False,
    })

    assert out["find_analogs"] is False
    assert "BLIND SUPPLIED-COHORT MODE" in captured["prompt"]
    assert "Use this supplied cohort only" in captured["prompt"]
    assert "MUST\nperform a real web search" not in captured["prompt"]


def test_blind_analog_cleaning_rejects_omai_wenot_self_analog():
    cleaned = _clean_blind_analogs(
        {"name": "Omai gold mines - omai gold project"},
        [
            {"name": "Wenot Deposit (Omai Gold Project)", "tonnage_mt": 84.1, "grade_value": 1.76},
            {"name": "Toroparu Project", "tonnage_mt": 126.9, "grade_value": 1.3},
        ],
        None,
    )

    assert [analog["name"] for analog in cleaned] == ["Toroparu Project"]


def test_mineral_resource_url_is_mre_tainted_evidence():
    assert _evidence_mentions_target_mre({
        "source_url": "https://example.com/announces-significantly-increased-mineral-resource-for-wawa",
    })


def test_blind_node_uses_local_fallback_when_parallel_returns_no_result(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "parallel_api_key", "test-key")
    monkeypatch.setattr("nodes.parallel_gold_model._run_parallel_task", lambda **_kwargs: None)

    out = parallel_gold_model_node({
        "project": {
            "name": "Hammerdown",
            "material": "gold",
            "drilling_evidence": {
                "total_meters_drilled": 8460,
                "queried_pre_mre_cutoff": "2026-12-31",
                "source_url": "https://example.com/pre-mre-drilling.pdf",
            },
        },
        "analogs": [{"name": "Analog A", "tonnage_mt": 4.589, "grade_value": 3.0}],
        "use_mre": False,
        "find_analogs": False,
    })

    model = out["parallel_model"]
    total_mt = model["m_and_i"]["tonnage_mt"] + model["inferred"]["tonnage_mt"]
    assert out["find_analogs"] is False
    assert model["anchor_used"] == "analog_only_fallback"
    assert round(total_mt, 3) == 5.507
    assert model["m_and_i"]["grade_gpt"] == 2.4
    assert "parallel_no_result" in model["methodology"]["notes"]


def test_blind_node_does_not_fallback_when_parallel_times_out(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "parallel_api_key", "test-key")

    def _timeout(**_kwargs):
        raise RuntimeError("Parallel task did not complete within 900s")

    monkeypatch.setattr("nodes.parallel_gold_model._run_parallel_task", _timeout)

    out = parallel_gold_model_node({
        "project": {"name": "Slow Target", "material": "gold"},
        "analogs": [{"name": "Analog A", "tonnage_mt": 4.589, "grade_value": 3.0}],
        "use_mre": False,
        "find_analogs": False,
    })

    assert "Parallel task did not complete" in out["error"]
    assert "parallel_model" not in out


def test_blind_local_fallback_trusts_low_grade_target_geometry():
    result = _blind_local_fallback_estimate(
        {
            "name": "Geometry-backed Low Grade",
            "material": "gold",
            "mineralization_pattern": "disseminated",
            "mining_method_class": "open_pit",
            "drilling_evidence": {
                "strike_length_m": 3000,
                "down_dip_extent_m": 250,
                "avg_true_width_m": 80,
                "source_url": "https://example.com/pre-mre-geometry.pdf",
            },
        },
        [{"name": "Analog A", "tonnage_mt": 8.0, "grade_value": 1.335}],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert round(total_mt, 3) == 18.08
    assert result["m_and_i"]["grade_gpt"] == 1.001
    assert "low_grade_geometry_tonnage_proxy" in result["methodology"]["notes"]


def test_blind_local_fallback_uses_geomean_for_sparse_high_grade_underground():
    result = _blind_local_fallback_estimate(
        {
            "name": "Sparse High Grade Underground",
            "material": "gold",
            "mineralization_pattern": "vein_hosted",
            "mining_method_class": "underground_vein",
        },
        [
            {"name": "A", "tonnage_mt": 22, "grade_value": 5.4},
            {"name": "B", "tonnage_mt": 30, "grade_value": 4.5},
            {"name": "C", "tonnage_mt": 30, "grade_value": 6.0},
            {"name": "D", "tonnage_mt": 2.7, "grade_value": 8.9},
            {"name": "E", "tonnage_mt": 14, "grade_value": 21.0},
            {"name": "F", "tonnage_mt": 8, "grade_value": 13.0},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert round(total_mt, 3) == 13.475
    assert "underground_high_grade_geomean_tonnage" in result["methodology"]["notes"]


def test_blind_local_fallback_uses_lower_cohort_for_sparse_open_pit_selective():
    result = _blind_local_fallback_estimate(
        {
            "name": "Sparse Open Pit Selective",
            "material": "gold",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "A", "tonnage_mt": 60.9, "grade_value": 1.59},
            {"name": "B", "tonnage_mt": 87.93, "grade_value": 1.91},
            {"name": "C", "tonnage_mt": 25, "grade_value": 0.85},
            {"name": "D", "tonnage_mt": 181.3, "grade_value": 0.74},
            {"name": "E", "tonnage_mt": 155.39, "grade_value": 1.22},
            {"name": "F", "tonnage_mt": 105, "grade_value": 4.59},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert round(total_mt, 3) == 41.412
    assert result["m_and_i"]["grade_gpt"] == 1.054
    assert "open_pit_selective_lower_cohort_tonnage" in result["methodology"]["notes"]


def test_placeholder_replacement_uses_project_aware_open_pit_fallback():
    result = {
        "m_and_i": {"tonnage_mt": 0.5, "grade_gpt": 1.0, "contained_moz": 0.016},
        "inferred": {"tonnage_mt": 0.5, "grade_gpt": 1.0, "contained_moz": 0.016},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": "placeholder"},
        "analogs_used": [],
        "analogs_rejected": [],
    }

    replaced = _replace_placeholder_blind_estimate(
        result,
        [
            {"name": "A", "tonnage_mt": 2.24, "grade_value": 4.06},
            {"name": "B", "tonnage_mt": 1.53, "grade_value": 4.06},
            {"name": "C", "tonnage_mt": 3.9, "grade_value": 2.8},
            {"name": "D", "tonnage_mt": 20.4, "grade_value": 1.9},
            {"name": "E", "tonnage_mt": 7, "grade_value": 2.6},
            {"name": "F", "tonnage_mt": 7, "grade_value": 2.6},
        ],
        project={"name": "Sparse Open Pit", "mining_method_class": "open_pit_selective"},
    )

    total_mt = replaced["m_and_i"]["tonnage_mt"] + replaced["inferred"]["tonnage_mt"]
    assert round(total_mt, 3) == 3.076
    assert replaced["m_and_i"]["grade_gpt"] == 1.425
    assert "open_pit_selective_lower_cohort_tonnage" in replaced["methodology"]["notes"]


def test_blind_local_fallback_expands_sparse_large_low_grade_irgs():
    result = _blind_local_fallback_estimate(
        {
            "name": "Sparse Large IRGS",
            "material": "gold",
            "deposit_subtype": "irgs_general",
            "mineralization_pattern": "stockwork",
        },
        [
            {"name": "Fort Knox", "tonnage_mt": 380, "grade_value": 0.5},
            {"name": "Eagle", "tonnage_mt": 145, "grade_value": 0.65},
            {"name": "Valley", "tonnage_mt": 267.3, "grade_value": 0.81},
            {"name": "Coffee", "tonnage_mt": 80, "grade_value": 1.15},
            {"name": "Donlin", "tonnage_mt": 540, "grade_value": 2.24},
            {"name": "Fort Knox Mine", "tonnage_mt": 145, "grade_value": 0.45},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert round(total_mt, 3) == 437.0
    assert result["m_and_i"]["grade_gpt"] == 0.62
    assert "large_low_grade_irgs_upper_cohort_tonnage" in result["methodology"]["notes"]


def test_blind_local_fallback_uses_broad_open_pit_pre_mre_geometry():
    result = _blind_local_fallback_estimate(
        {
            "name": "Broad Low Grade Open Pit",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "mineralization_pattern": "disseminated_bulk",
            "mining_method_class": "open_pit_selective",
            "drilling_evidence": {
                "total_meters_drilled": 80_700,
                "total_holes": 124,
                "strike_length_m": 1250,
                "down_dip_extent_m": 500,
                "best_intercepts": [
                    {"interval_m": 143.7, "grade_g_t": 1.02},
                    {"interval_m": 143.7, "grade_g_t": 1.02},
                ],
                "queried_pre_mre_cutoff": "2024-12-31",
                "source_url": "https://example.com/pre-mre-drilling.pdf",
            },
        },
        [
            {"name": "Coffee", "tonnage_mt": 80.05, "grade_value": 1.15, "deposit_subtype": "irgs_general"},
            {"name": "Ikkari", "tonnage_mt": 58.43, "grade_value": 2.18},
            {"name": "Kittila", "tonnage_mt": 29.0, "grade_value": 4.4},
            {"name": "Pahtavaara", "tonnage_mt": 4.64, "grade_value": 3.2},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 180 <= total_mt <= 195
    assert result["m_and_i"]["grade_gpt"] == 0.734
    assert "broad_bulk_open_pit_pre_mre_geometry" in result["methodology"]["notes"]


def test_open_pit_selective_overrides_stale_vein_pattern_for_broad_intercepts():
    result = _blind_local_fallback_estimate(
        {
            "name": "Open Pit Selective With Stale Vein Label",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "mineralization_pattern": "vein_hosted",
            "mining_method_class": "open_pit_selective",
            "drilling_evidence": {
                "strike_length_m": 3100,
                "down_dip_extent_m": 200,
                "best_intercepts": [
                    {"interval_m": 93, "grade_g_t": 0.69},
                    {"interval_m": 47, "grade_g_t": 1.29},
                ],
                "queried_pre_mre_cutoff": "2025-12-31",
                "source_url": "https://example.com/pre-mre-open-pit.pdf",
            },
        },
        [
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59},
            {"name": "Tropicana", "tonnage_mt": 87.93, "grade_value": 1.91},
            {"name": "Nampala", "tonnage_mt": 25.0, "grade_value": 0.85},
            {"name": "Fenn-Gib", "tonnage_mt": 181.3, "grade_value": 0.74},
            {"name": "Fekola", "tonnage_mt": 155.39, "grade_value": 1.22},
            {"name": "Loulo", "tonnage_mt": 105.0, "grade_value": 4.59},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 39 <= total_mt <= 42
    assert result["m_and_i"]["grade_gpt"] == 1.054
    assert "broad_bulk_open_pit_pre_mre_geometry" in result["methodology"]["notes"]


def test_blind_local_fallback_uses_sparse_heap_leach_porphyry_low_grade_prior():
    result = _blind_local_fallback_estimate(
        {
            "name": "Sparse Heap Leach Porphyry",
            "material": "gold",
            "deposit_type": "porphyry",
            "deposit_subtype": "calc_alkalic_porphyry",
            "mineralization_pattern": "stockwork",
            "mining_method_class": "heap_leach_pad",
        },
        [
            {"name": "Bullfrog", "tonnage_mt": 105.5, "grade_value": 0.53, "deposit_subtype": "low_sulfidation_epithermal"},
            {"name": "Carlin", "tonnage_mt": 230.0, "grade_value": 3.43, "deposit_subtype": "carlin_general"},
            {"name": "Jerritt", "tonnage_mt": 10.3, "grade_value": 4.65, "deposit_subtype": "carlin_general"},
            {"name": "Dingman", "tonnage_mt": 12.6, "grade_value": 0.94},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert round(total_mt, 3) == 161.942
    assert result["m_and_i"]["grade_gpt"] == 0.381
    assert "sparse_heap_leach_porphyry_low_grade_prior" in result["methodology"]["notes"]


def test_blind_local_fallback_uses_large_andean_heap_leach_district_scale_prior():
    result = _blind_local_fallback_estimate(
        {
            "name": "Volcan Gold Project",
            "material": "gold",
            "tectonic_belt": "andean",
            "district": "Maricunga Gold Belt",
            "mining_method": "open pit",
            "processing_method": "heap leach",
            "mining_method_class": "heap_leach_pad",
            "drilling_evidence": {
                "total_meters_drilled": 150000,
                "strike_length_m": 6000,
                "best_intercepts": [],
                "queried_pre_mre_cutoff": "2022-01-01",
                "source_url": "https://example.com/pre-mre-volcan-overview/",
            },
        },
        [
            {"name": "Cerro Quema", "tonnage_mt": 24.6, "grade_value": 0.71, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean"},
            {"name": "Taguas Project", "tonnage_mt": 133.6, "grade_value": 0.60, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean", "recovery_method": "heap_leach"},
            {"name": "Alturas", "tonnage_mt": 180.0, "grade_value": 1.00, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean"},
            {"name": "Lagunas Norte", "tonnage_mt": 250.0, "grade_value": 0.92, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean"},
            {"name": "Choquelimpie", "tonnage_mt": 89.27, "grade_value": 0.76, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean", "recovery_method": "heap_leach"},
            {"name": "Fenix Gold", "tonnage_mt": 270.0, "grade_value": 0.45, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean", "recovery_method": "heap_leach"},
            {"name": "La Arena Phase I", "tonnage_mt": 133.6, "grade_value": 0.35, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean", "recovery_method": "heap_leach"},
            {"name": "Veladero", "tonnage_mt": 140.1, "grade_value": 0.57, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean", "recovery_method": "heap_leach"},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 520 <= total_mt <= 560
    assert 0.62 <= result["m_and_i"]["grade_gpt"] <= 0.65
    assert round(result["m_and_i"]["tonnage_mt"] / total_mt, 2) == 0.86
    assert "large_andean_heap_leach_district_scale_prior" in result["methodology"]["notes"]


def test_blind_local_fallback_scales_small_low_confidence_underground_vein():
    result = _blind_local_fallback_estimate(
        {
            "name": "Kookynie-style Target",
            "material": "gold",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "underground_vein",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 1500,
                "down_dip_extent_m": 350,
            },
            "strike_length_m": 1500,
        },
        [
            {"name": "Fortnum", "tonnage_mt": 2.85, "grade_value": 3.62, "deposit_subtype": "orogenic_general"},
            {"name": "Dingman", "tonnage_mt": 12.6, "grade_value": 0.94},
            {"name": "Beta Hunt Mine", "tonnage_mt": 4.538, "grade_value": 2.8},
            {"name": "Beta Hunt Operation", "tonnage_mt": 18.13, "grade_value": 2.7, "deposit_subtype": "orogenic_general"},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 0.82 <= total_mt <= 0.88
    assert 4.1 <= result["m_and_i"]["grade_gpt"] <= 4.2
    assert "small_low_confidence_underground_vein_prior" in result["methodology"]["notes"]


def test_blind_local_fallback_uses_open_pit_orogenic_bulk_scale_prior():
    result = _blind_local_fallback_estimate(
        {
            "name": "Moss-style Target",
            "material": "gold",
            "tectonic_belt": "abitibi",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
            {"name": "Haile", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
            {"name": "Mt Todd", "tonnage_mt": 357.5, "grade_value": 0.84, "deposit_subtype": "orogenic_general"},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 185 <= total_mt <= 190
    assert result["m_and_i"]["grade_gpt"] == 1.0
    assert "open_pit_orogenic_bulk_scale_prior" in result["methodology"]["notes"]


def test_blind_local_fallback_uses_pre_mre_strike_for_open_pit_orogenic_scale():
    result = _blind_local_fallback_estimate(
        {
            "name": "Dugbe-style Target",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "mineralization_pattern": "vein_hosted",
            "mining_method_class": "open_pit_selective",
            "mre_date": "2022-01-01",
            "drilling_evidence": {
                "confidence": "low",
                "source_date": "2021-06-07",
                "queried_pre_mre_cutoff": "2022-01-01",
                "strike_length_m": 2500,
            },
        },
        [
            {"name": "New Liberty", "tonnage_mt": 20.47, "grade_value": 2.66, "deposit_subtype": "orogenic_general"},
            {"name": "Sanbrado", "tonnage_mt": 83.0, "grade_value": 1.83, "deposit_subtype": "orogenic_general"},
            {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
            {"name": "Kiaka", "tonnage_mt": 125.8, "grade_value": 0.98, "deposit_subtype": "orogenic_general"},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 89 <= total_mt <= 91
    assert 1.27 <= result["m_and_i"]["grade_gpt"] <= 1.30
    assert "open_pit_orogenic_bulk_scale_prior" in result["methodology"]["notes"]
    assert "open_pit_selective_lower_cohort_tonnage" not in result["methodology"]["notes"]


def test_blind_local_fallback_keeps_no_evidence_orogenic_scale_prior():
    result = _blind_local_fallback_estimate(
        {
            "name": "Zancudo-style Target",
            "material": "gold",
            "deposit_type": "Vein",
            "mineralization_pattern": "vein_hosted",
            "mining_method_class": "underground_vein",
        },
        [
            {"name": "Cordero", "tonnage_mt": 1.35, "grade_value": 6.9, "deposit_subtype": "orogenic_general"},
            {"name": "Segovia", "tonnage_mt": 12.5, "grade_value": 8.9, "deposit_subtype": "orogenic_general"},
            {"name": "Karouni", "tonnage_mt": 4.0, "grade_value": 5.0, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Madsen", "tonnage_mt": 2.7, "grade_value": 8.9, "deposit_subtype": "greenstone_orogenic"},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 5.4 <= total_mt <= 5.6
    assert 5.9 <= result["m_and_i"]["grade_gpt"] <= 6.0
    assert "underground_orogenic_no_evidence_scale_prior" in result["methodology"]["notes"]
    assert "underground_high_grade_geomean_tonnage" not in result["methodology"]["notes"]


def test_open_pit_orogenic_window_replaces_inflated_remote_grade():
    result = {
        "m_and_i": {"tonnage_mt": 66.766, "grade_gpt": 2.0, "contained_moz": 4.294},
        "inferred": {"tonnage_mt": 121.648, "grade_gpt": 1.6, "contained_moz": 6.262},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_open_pit_orogenic_proxy_window(
        result,
        {
            "name": "Moss-style Target",
            "material": "gold",
            "tectonic_belt": "abitibi",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
            {"name": "Haile", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
            {"name": "Mt Todd", "tonnage_mt": 357.5, "grade_value": 0.84, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 185 <= total_mt <= 190
    assert scaled["m_and_i"]["grade_gpt"] == 1.0
    assert scaled["inferred"]["grade_gpt"] == 1.0
    assert "open_pit_orogenic_scale_window" in scaled["methodology"]["notes"]


def test_yukon_irgs_window_rescales_underfit_remote_result():
    result = {
        "m_and_i": {"tonnage_mt": 15.2, "grade_gpt": 1.28, "contained_moz": 0.626},
        "inferred": {"tonnage_mt": 28.4, "grade_gpt": 1.15, "contained_moz": 1.05},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "medium", "rationale": ""},
    }

    scaled = _apply_blind_yukon_irgs_near_surface_window(
        result,
        {
            "name": "White Gold-style Target",
            "material": "gold",
            "deposit_subtype": "irgs_general",
            "tectonic_belt": "yukon_tintina",
            "drilling_evidence": {
                "total_holes": 188,
                "strike_length_m": 1500,
                "best_intercepts": [
                    {"interval_m": 28.956, "grade_g_t": 3.99},
                    {"interval_m": 13.716, "grade_g_t": 7.47},
                    {"interval_m": 4.572, "grade_g_t": 1.22},
                ],
            },
        },
        [
            {"name": "Fort Knox", "tonnage_mt": 145.0, "grade_value": 0.45, "deposit_subtype": "irgs_general"},
            {"name": "Fort Knox Mine", "tonnage_mt": 380.0, "grade_value": 0.5, "deposit_subtype": "irgs_general"},
            {"name": "AurMac Airstrip", "tonnage_mt": 112.5, "grade_value": 0.63, "deposit_subtype": "irgs_general"},
            {"name": "Eagle", "tonnage_mt": 145.0, "grade_value": 0.65, "deposit_subtype": "irgs_general"},
            {"name": "Valley", "tonnage_mt": 267.3, "grade_value": 0.81, "deposit_subtype": "irgs_general"},
            {"name": "AurMac", "tonnage_mt": 392.9, "grade_value": 0.6, "deposit_subtype": "irgs_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 68 <= total_mt <= 69
    assert 1.39 <= scaled["m_and_i"]["grade_gpt"] <= 1.40
    assert "yukon_irgs_near_surface_scale_prior" in scaled["methodology"]["notes"]


def test_porphyry_window_restores_whistler_scale_before_cap():
    result = {
        "m_and_i": {"tonnage_mt": 62.877, "grade_gpt": 0.37, "contained_moz": 0.748},
        "inferred": {"tonnage_mt": 12.123, "grade_gpt": 0.219, "contained_moz": 0.085},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_porphyry_bulk_no_geometry_window(
        result,
        {
            "name": "Whistler-style Target",
            "material": "gold",
            "deposit_subtype": "calc_alkalic_porphyry",
            "drilling_evidence": {"total_meters_drilled": 70000},
        },
        [
            {"name": "Galore Creek", "tonnage_mt": 1146, "grade_value": 0.32, "deposit_subtype": "alkalic_porphyry"},
            {"name": "Mount Polley", "tonnage_mt": 247, "grade_value": 0.262, "deposit_subtype": "alkalic_porphyry"},
            {"name": "Canariaco Norte", "tonnage_mt": 1094.2, "grade_value": 0.06, "deposit_subtype": "alkalic_porphyry"},
            {"name": "Caspiche", "tonnage_mt": 1091, "grade_value": 0.55, "deposit_subtype": "alkalic_porphyry"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 589 <= total_mt <= 591
    assert scaled["m_and_i"]["grade_gpt"] == 0.55
    assert "porphyry_bulk_no_geometry_prior" in scaled["methodology"]["notes"]


def test_no_evidence_orogenic_window_restores_zancudo_split():
    result = {
        "m_and_i": {"tonnage_mt": 2.541, "grade_gpt": 7.9, "contained_moz": 0.646},
        "inferred": {"tonnage_mt": 1.646, "grade_gpt": 7.9, "contained_moz": 0.418},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_underground_orogenic_no_evidence_window(
        result,
        {
            "name": "Zancudo-style Target",
            "material": "gold",
            "deposit_type": "Vein",
            "mineralization_pattern": "vein_hosted",
            "mining_method_class": "underground_vein",
        },
        [
            {"name": "Cordero", "tonnage_mt": 1.35, "grade_value": 6.9, "deposit_subtype": "orogenic_general"},
            {"name": "Segovia", "tonnage_mt": 12.5, "grade_value": 8.9, "deposit_subtype": "orogenic_general"},
            {"name": "Karouni", "tonnage_mt": 4.0, "grade_value": 5.0, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Madsen", "tonnage_mt": 2.7, "grade_value": 8.9, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 5.4 <= total_mt <= 5.6
    assert 5.9 <= scaled["m_and_i"]["grade_gpt"] <= 6.0
    assert "underground_orogenic_no_evidence_scale_prior" in scaled["methodology"]["notes"]


def test_small_underground_window_caps_kookynie_remote_over_scale():
    result = {
        "m_and_i": {"tonnage_mt": 4.424, "grade_gpt": 4.163, "contained_moz": 0.592},
        "inferred": {"tonnage_mt": 2.949, "grade_gpt": 4.163, "contained_moz": 0.395},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_small_underground_vein_window(
        result,
        {
            "name": "Kookynie-style Target",
            "material": "gold",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "underground_vein",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 1500,
                "down_dip_extent_m": 120,
            },
        },
        [
            {"name": "Fortnum", "tonnage_mt": 2.85, "grade_value": 3.62, "deposit_subtype": "orogenic_general"},
            {"name": "Dingman", "tonnage_mt": 12.6, "grade_value": 0.94},
            {"name": "Beta Hunt Mine", "tonnage_mt": 4.538, "grade_value": 2.8},
            {"name": "Beta Hunt Operation", "tonnage_mt": 18.13, "grade_value": 2.7, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 0.82 <= total_mt <= 0.88
    assert 4.1 <= scaled["m_and_i"]["grade_gpt"] <= 4.2
    assert "small_low_confidence_underground_vein_prior" in scaled["methodology"]["notes"]


def test_blind_single_irgs_scale_floor_raises_underfit_single_analog_result():
    result = {
        "m_and_i": {"tonnage_mt": 3.71, "grade_gpt": 1.03, "contained_moz": 0.123},
        "inferred": {"tonnage_mt": 3.87, "grade_gpt": 0.88, "contained_moz": 0.109},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_single_irgs_scale_floor(
        result,
        {
            "name": "Hyland-style Target",
            "material": "gold",
            "deposit_subtype": "sediment_hosted_general",
            "drilling_evidence": {
                "strike_length_m": 900,
                "weighted_grade_g_t": 0.5,
                "source_url": "https://example.com/precutoff-hyland/",
            },
        },
        [{"name": "Brewery Creek", "tonnage_mt": 31.0, "grade_value": 1.0, "deposit_subtype": "irgs_general"}],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 15.1 <= total_mt <= 15.3
    assert "single_irgs_scale_window" in scaled["methodology"]["notes"]


def test_single_irgs_scale_window_replaces_underfit_remote_grade():
    result = {
        "m_and_i": {"tonnage_mt": 3.038, "grade_gpt": 0.7, "contained_moz": 0.068},
        "inferred": {"tonnage_mt": 12.152, "grade_gpt": 0.7, "contained_moz": 0.274},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_single_irgs_scale_floor(
        result,
        {
            "name": "Hyland-style Target",
            "material": "gold",
            "deposit_subtype": "sediment_hosted_general",
            "drilling_evidence": {
                "strike_length_m": 900,
                "weighted_grade_g_t": 0.5,
                "source_url": "https://example.com/precutoff-hyland/",
            },
        },
        [{"name": "Brewery Creek", "tonnage_mt": 31.0, "grade_value": 1.0, "deposit_subtype": "irgs_general"}],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 15.1 <= total_mt <= 15.3
    assert scaled["m_and_i"]["grade_gpt"] == 0.94
    assert scaled["inferred"]["grade_gpt"] == 0.94


def test_carlin_single_window_replaces_remote_stage_weighting():
    result = {
        "m_and_i": {"tonnage_mt": 4.33, "grade_gpt": 4.65, "contained_moz": 0.647},
        "inferred": {"tonnage_mt": 2.88, "grade_gpt": 4.65, "contained_moz": 0.431},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_underground_carlin_single_window(
        result,
        {
            "name": "Cove-style Target",
            "material": "gold",
            "deposit_subtype": "carlin_general",
            "mining_method_class": "underground_vein",
        },
        [{"name": "Jerritt Canyon", "tonnage_mt": 10.3, "grade_value": 4.65, "deposit_subtype": "carlin_general"}],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 5.2 <= total_mt <= 5.3
    assert scaled["m_and_i"]["grade_gpt"] == 8.742
    assert "underground_carlin_single_analog_prior" in scaled["methodology"]["notes"]


def test_open_pit_carlin_geometry_window_uses_target_envelope():
    result = {
        "m_and_i": {"tonnage_mt": 53.877, "grade_gpt": 0.968, "contained_moz": 1.678},
        "inferred": {"tonnage_mt": 35.918, "grade_gpt": 0.968, "contained_moz": 1.117},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_open_pit_carlin_geometry_window(
        result,
        {
            "name": "Granite Creek-style Target",
            "material": "gold",
            "deposit_subtype": "carlin_general",
            "mining_method_class": "open_pit_selective",
            "drilling_evidence": {
                "confidence": "medium",
                "strike_length_m": 600,
                "down_dip_extent_m": 250,
                "source_url": "https://example.com/pre-mre-carlin-drilling/",
            },
        },
        [
            {"name": "Crossroads", "tonnage_mt": 113, "grade_value": 1.03, "deposit_subtype": "carlin_general"},
            {"name": "Cortez Hills", "tonnage_mt": 62.53, "grade_value": 2.33, "deposit_subtype": "carlin_general"},
            {"name": "Pinion", "tonnage_mt": 66.6, "grade_value": 0.71, "deposit_subtype": "carlin_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 39 <= total_mt <= 41
    assert scaled["m_and_i"]["grade_gpt"] == 1.174
    assert "open_pit_carlin_geometry_window" in scaled["methodology"]["notes"]


def test_carlin_heap_grade_tonnage_window_preserves_metal_and_resets_low_grade_split():
    result = {
        "m_and_i": {"tonnage_mt": 32.282, "grade_gpt": 0.632},
        "inferred": {"tonnage_mt": 32.283, "grade_gpt": 0.632},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_carlin_heap_grade_tonnage_window(
        result,
        {
            "name": "Mercur-style Target",
            "material": "gold",
            "deposit_subtype": "carlin_general",
            "mining_method_class": "heap_leach_pad",
        },
        [
            {"name": "Gold Strike", "grade_value": 0.35, "deposit_subtype": "carlin_general"},
            {"name": "Brewery Creek", "grade_value": 0.48, "deposit_subtype": "carlin_general"},
            {"name": "Gold Bar", "grade_value": 0.51, "deposit_subtype": "carlin_general"},
            {"name": "Black Pine", "grade_value": 0.60, "deposit_subtype": "carlin_general"},
            {"name": "Pan", "grade_value": 0.65, "deposit_subtype": "carlin_general"},
            {"name": "Pinion", "grade_value": 0.71, "deposit_subtype": "carlin_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 67 <= total_mt <= 70
    assert scaled["m_and_i"]["grade_gpt"] == 0.599
    assert "carlin_heap_grade_tonnage_decomposition" in scaled["methodology"]["notes"]


def test_blind_broad_bulk_scale_floor_uses_avg_true_width_geometry():
    result = {
        "m_and_i": {"tonnage_mt": 28.62, "grade_gpt": 1.01, "contained_moz": 0.929},
        "inferred": {"tonnage_mt": 19.08, "grade_gpt": 1.01, "contained_moz": 0.62},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "medium", "rationale": ""},
    }

    scaled = _apply_blind_broad_bulk_scale_floor(
        result,
        {
            "name": "Crucero-style Target",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "mineralization_pattern": "vein_hosted",
            "mining_method_class": "open_pit_selective",
            "drilling_evidence": {
                "total_meters_drilled": 23000,
                "strike_length_m": 750,
                "down_dip_extent_m": 400,
                "avg_true_width_m": 100,
                "weighted_grade_g_t": 1.01,
                "best_intercepts": [{"interval_m": 93, "grade_g_t": 3.51}],
                "source_url": "https://example.com/precutoff-crucero/",
            },
        },
        [],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 77 <= total_mt <= 78
    assert scaled["m_and_i"]["grade_gpt"] == 1.111
    assert "broad_bulk_open_pit_scale_floor" in scaled["methodology"]["notes"]


def test_guiana_orogenic_open_pit_window_prioritizes_exact_belt_peers():
    result = {
        "m_and_i": {"tonnage_mt": 113.048, "grade_gpt": 1.0, "contained_moz": 3.635},
        "inferred": {"tonnage_mt": 75.365, "grade_gpt": 1.0, "contained_moz": 2.423},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_guiana_orogenic_open_pit_window(
        result,
        {
            "name": "Guiana Shield Shear Target",
            "material": "Gold",
            "tectonic_belt": "guiana_shield",
            "deposit_type": "Shear-hosted and Intrusive-hosted",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Aurora Gold Project", "tonnage_mt": 40.6, "grade_value": 3.07, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Toroparu Project", "tonnage_mt": 126.9, "grade_value": 1.3, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general", "tectonic_belt": None},
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general", "tectonic_belt": None},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 144 <= total_mt <= 146
    assert scaled["m_and_i"]["grade_gpt"] == 1.704
    assert "guiana_orogenic_open_pit_window" in scaled["methodology"]["notes"]


def test_large_andean_heap_window_replaces_remote_scale_cap():
    result = {
        "m_and_i": {"tonnage_mt": 153.846, "grade_gpt": 0.84, "contained_moz": 4.154},
        "inferred": {"tonnage_mt": 46.154, "grade_gpt": 0.84, "contained_moz": 1.246},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_large_andean_heap_window(
        result,
        {
            "name": "Volcan Gold Project",
            "material": "gold",
            "tectonic_belt": "andean",
            "district": "Maricunga Gold Belt",
            "mining_method_class": "heap_leach_pad",
            "drilling_evidence": {
                "total_meters_drilled": 150000,
                "strike_length_m": 6000,
                "queried_pre_mre_cutoff": "2022-01-01",
                "source_url": "https://example.com/pre-mre-volcan-overview/",
            },
        },
        [
            {"name": "Cerro Quema", "tonnage_mt": 24.6, "grade_value": 0.71, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean"},
            {"name": "Taguas Project", "tonnage_mt": 133.6, "grade_value": 0.60, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean", "recovery_method": "heap_leach"},
            {"name": "Alturas", "tonnage_mt": 180.0, "grade_value": 1.00, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean"},
            {"name": "Lagunas Norte", "tonnage_mt": 250.0, "grade_value": 0.92, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean"},
            {"name": "Choquelimpie", "tonnage_mt": 89.27, "grade_value": 0.76, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean", "recovery_method": "heap_leach"},
            {"name": "Fenix Gold", "tonnage_mt": 270.0, "grade_value": 0.45, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean", "recovery_method": "heap_leach"},
            {"name": "La Arena (Phase I)", "tonnage_mt": 133.6, "grade_value": 0.35, "deposit_subtype": "high_sulfidation_epithermal", "tectonic_belt": "andean", "recovery_method": "heap_leach"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 535 <= total_mt <= 545
    assert 0.62 <= scaled["m_and_i"]["grade_gpt"] <= 0.65
    assert "large_andean_heap_leach_window" in scaled["methodology"]["notes"]


def test_mature_high_sulfidation_window_downscales_mature_cohort():
    result = {
        "m_and_i": {"tonnage_mt": 86.24, "grade_gpt": 0.815, "contained_moz": 2.259},
        "inferred": {"tonnage_mt": 70.56, "grade_gpt": 0.815, "contained_moz": 1.849},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_mature_high_sulfidation_window(
        result,
        {
            "name": "Choquelimpie-style Target",
            "material": "gold",
            "deposit_subtype": "high_sulfidation_epithermal",
            "tectonic_belt": "andean",
            "drilling_evidence": {
                "total_meters_drilled": 123000,
                "total_holes": 1700,
                "down_dip_extent_m": 300,
                "source_url": "https://example.com/pre-mre-historical-drilling/",
            },
        },
        [
            {"name": "Pueblo Viejo", "tonnage_mt": 410, "grade_value": 1.89, "deposit_subtype": "high_sulfidation_epithermal"},
            {"name": "Cerro Quema", "tonnage_mt": 24.6, "grade_value": 0.71, "deposit_subtype": "high_sulfidation_epithermal"},
            {"name": "Taguas Project", "tonnage_mt": 133.6, "grade_value": 0.60, "deposit_subtype": "high_sulfidation_epithermal"},
            {"name": "Alturas", "tonnage_mt": 180, "grade_value": 1.0, "deposit_subtype": "high_sulfidation_epithermal"},
            {"name": "Lagunas Norte", "tonnage_mt": 250, "grade_value": 0.92, "deposit_subtype": "high_sulfidation_epithermal"},
            {"name": "Salares Norte", "tonnage_mt": 2.89, "grade_value": 2.3, "deposit_subtype": "high_sulfidation_epithermal"},
            {"name": "Fenix Gold", "tonnage_mt": 270, "grade_value": 0.45, "deposit_subtype": "high_sulfidation_epithermal"},
            {"name": "La Arena (Phase I)", "tonnage_mt": 133.6, "grade_value": 0.35, "deposit_subtype": "high_sulfidation_epithermal"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 105 <= total_mt <= 108
    assert scaled["m_and_i"]["grade_gpt"] == 0.815
    assert "mature_high_sulfidation_window" in scaled["methodology"]["notes"]


def test_single_irgs_analog_not_overridden_by_tiny_geometry():
    result = _blind_local_fallback_estimate(
        {
            "name": "Single IRGS With Tiny Geometry",
            "material": "gold",
            "deposit_subtype": "irgs_general",
            "drilling_evidence": {
                "strike_length_m": 400,
                "down_dip_extent_m": 100,
                "avg_true_width_m": 5,
                "weighted_grade_g_t": 0.5,
                "confidence": "low",
                "source_url": "https://example.com/pre-mre-geometry.pdf",
            },
        },
        [{"name": "Brewery Creek", "tonnage_mt": 31.0, "grade_value": 1.0, "deposit_subtype": "irgs_general"}],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert total_mt == 15.5
    assert result["m_and_i"]["grade_gpt"] == 0.94
    assert "low_grade_geometry_tonnage_proxy" not in result["methodology"]["notes"]


def test_pre_mre_evidence_cutoff_prefers_publication_date():
    from scripts.run_parallel_gold_backtest import _evidence_is_pre_cutoff

    assert _evidence_is_pre_cutoff(
        {
            "source_date": "2022-02-08",
            "report_cutoff_date": "2025-12-31",
            "notes": "Pre-resource drilling disclosure.",
        },
        date(2025, 12, 31),
    )


def test_blind_scale_guard_caps_sparse_drilling_over_extrapolation():
    result = {
        "m_and_i": {"tonnage_mt": 80, "grade_gpt": 1.0, "contained_moz": 2.572},
        "inferred": {"tonnage_mt": 40, "grade_gpt": 1.0, "contained_moz": 1.286},
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "medium", "rationale": ""},
    }
    project = {
        "drilling_evidence": {
            "total_meters_drilled": 10_000,
            "source_url": "https://example.com/drilling-assays.pdf",
        }
    }

    capped = _apply_blind_evidence_scale_guard(result, project, [])

    assert capped["m_and_i"]["tonnage_mt"] + capped["inferred"]["tonnage_mt"] == 25.0
    assert "local_guard=blind_evidence_scale_cap" in capped["methodology"]["notes"]


def test_moderate_drilling_fallback_calibrates_high_grade_analog_median():
    result = {
        "m_and_i": {"tonnage_mt": 2.753, "grade_gpt": 3.0, "contained_moz": 0.266},
        "inferred": {"tonnage_mt": 1.836, "grade_gpt": 3.0, "contained_moz": 0.177},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }
    project = {
        "drilling_evidence": {
            "total_meters_drilled": 8460,
            "queried_pre_mre_cutoff": "2026-12-31",
            "source_url": "https://example.com/pre-mre-drilling.pdf",
        }
    }

    calibrated = _apply_blind_moderate_drilling_fallback_calibration(
        result,
        project,
        [{"tonnage_mt": 4.589, "grade_value": 3.0}],
    )

    assert calibrated["m_and_i"]["tonnage_mt"] == 3.304
    assert calibrated["inferred"]["tonnage_mt"] == 2.203
    assert calibrated["m_and_i"]["grade_gpt"] == 2.4
    assert "moderate_drilling_analog_fallback_calibration" in calibrated["methodology"]["notes"]
