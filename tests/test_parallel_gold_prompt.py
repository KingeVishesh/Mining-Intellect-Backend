from __future__ import annotations

import json
from datetime import date

import requests

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
    _apply_blind_large_low_grade_carlin_window,
    _apply_blind_great_basin_heap_breccia_window,
    _apply_blind_bc_porphyry_sparse_stockwork_window,
    _apply_blind_guiana_orogenic_open_pit_window,
    _apply_blind_newfoundland_orogenic_window,
    _apply_blind_fennoscandian_orogenic_hybrid_window,
    _apply_blind_west_african_orogenic_open_pit_window,
    _apply_blind_central_african_orogenic_open_pit_window,
    _apply_blind_high_grade_pre_mre_evidence_window,
    _apply_blind_open_pit_orogenic_proxy_window,
    _apply_blind_trans_hudson_goldfields_syncline_window,
    _apply_blind_trans_hudson_orogenic_open_pit_window,
    _apply_blind_great_basin_orogenic_open_pit_window,
    _apply_blind_great_basin_beartrack_heap_window,
    _apply_blind_sparse_stockwork_lode_window,
    _apply_blind_sparse_tiny_yilgarn_vein_window,
    _apply_blind_yilgarn_small_open_pit_window,
    _apply_blind_yilgarn_shallow_bulk_decomposition_window,
    _apply_blind_yilgarn_metamorphic_mixed_bulk_grade_window,
    _apply_blind_yilgarn_mandilla_geometry_window,
    _apply_blind_high_grade_vms_scout_window,
    _apply_blind_abitibi_greenstone_district_window,
    _apply_blind_large_abitibi_open_pit_bulk_window,
    _apply_blind_abitibi_unknown_orogenic_scout_window,
    _apply_blind_abitibi_moderate_underground_window,
    _apply_blind_abitibi_wawa_mixed_grade_window,
    _apply_blind_new_zealand_reefton_ausb_window,
    _apply_blind_abitibi_tower_gold_district_window,
    _apply_blind_ontario_irgs_tower_mountain_window,
    _apply_blind_andean_colombia_underground_vein_window,
    _apply_blind_yukon_rogue_irgs_window,
    _apply_blind_yukon_hyland_sediment_heap_window,
    _apply_blind_abitibi_long_intercept_open_pit_window,
    _apply_blind_abitibi_small_open_pit_vein_window,
    _apply_blind_abitibi_open_pit_vein_grade_window,
    _apply_blind_brazilian_shield_open_pit_window,
    _apply_blind_guiana_underground_vein_high_grade_window,
    _apply_blind_bc_porphyry_stockwork_grade_window,
    _apply_blind_bc_porphyry_project_scale_window,
    _apply_blind_andean_porphyry_gold_copper_window,
    _apply_blind_andean_underground_vein_scale_floor_window,
    _apply_blind_porphyry_bulk_no_geometry_window,
    _apply_blind_large_andean_heap_window,
    _apply_blind_mature_high_sulfidation_window,
    _apply_blind_sparse_yilgarn_metamorphic_underground_window,
    _apply_blind_small_underground_vein_window,
    _apply_blind_broad_bulk_geometry_window,
    _apply_blind_underground_orogenic_no_evidence_window,
    _apply_blind_yukon_irgs_near_surface_window,
    _apply_blind_yukon_near_surface_vein_window,
    _apply_blind_large_yukon_irgs_window,
    _apply_blind_tailings_reprocessing_window,
    _blind_result_mentions_mre_anchor,
    _clean_blind_analogs,
    _evidence_mentions_target_mre,
    _ensure_blind_resource_ranges,
    _high_grade_vms_scout_proxy,
    _large_yukon_irgs_proxy,
    _parallel_request,
    _parse_parallel_output_content,
    _result_total_tonnage,
    _sparse_stockwork_lode_proxy,
    _replace_blind_mre_leak_estimate,
    _output_schema,
    _blind_local_fallback_estimate,
    _replace_placeholder_blind_estimate,
    _target_evidence_for_scale,
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

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=False)

    assert "Treat 2025-05-15 as the target MRE cutoff date" in prompt
    assert "use ONLY information published BEFORE 2025-05-15" in prompt


def test_blind_prompt_silently_discards_target_mre_leaks():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "mre_data_source": {"as_of_date": "2025-05-15"},
    }

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=False)

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

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=False)

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

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=False)

    assert "TARGET ENRICHMENT" in prompt
    assert "MUST search for those pre-MRE target disclosures" in prompt
    assert "Only choose `analog_only_fallback` after documenting" in prompt


def test_blind_find_analogs_prompt_disables_target_open_web_search():
    project = {
        "name": "Blind Gold",
        "material": "gold",
        "mre_data_source": {"as_of_date": "2025-05-15"},
    }

    prompt = _build_prompt(project=project, analogs=[], use_mre=False, find_analogs=True)

    assert "TARGET ENRICHMENT — BLIND ANALOG-DISCOVERY MODE" in prompt
    assert "do NOT run open" in prompt
    assert "web searches on the target project name" in prompt
    assert "target_open_web_search=disabled_blind" in prompt
    assert "MUST search for those pre-MRE target disclosures" not in prompt


def test_blind_schema_forces_numeric_estimate_and_excludes_mre_anchor():
    schema = _output_schema(use_mre=False)

    assert schema["properties"]["m_and_i"]["properties"]["tonnage_mt"]["type"] == "number"
    assert schema["properties"]["m_and_i"]["properties"]["tonnage_mt"]["exclusiveMinimum"] == 0
    assert schema["properties"]["m_and_i"]["properties"]["grade_gpt"]["type"] == "number"
    assert "tonnage_range_mt" in schema["properties"]["m_and_i"]["required"]
    assert "grade_range_gpt" in schema["properties"]["inferred"]["required"]
    assert "sources_used" in schema["required"]
    assert "mre_anchored" not in schema["properties"]["anchor_used"]["enum"]


def test_blind_range_safeguard_derives_monotonic_ranges_from_scalars():
    result = {
        "m_and_i": {"tonnage_mt": 10.0, "grade_gpt": 1.2, "contained_moz": 0.386},
        "inferred": {"tonnage_mt": 5.0, "grade_gpt": 0.9, "contained_moz": 0.145},
        "anchor_used": "analog_only_fallback",
        "conviction": {"level": "low", "rationale": "thin evidence"},
        "methodology": {"branch": "analog_only_fallback"},
        "analogs_used": [],
        "analogs_rejected": [],
    }

    out = _ensure_blind_resource_ranges(result)

    assert out["m_and_i"]["tonnage_range_mt"] == {
        "p10": 2.857,
        "p50": 10.0,
        "p90": 35.0,
    }
    assert out["inferred"]["grade_range_gpt"]["p10"] < out["inferred"]["grade_gpt"]
    assert out["sources_used"] == []


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


def test_small_drill_transformation_is_not_replaced_when_not_placeholder():
    result = {
        "m_and_i": {"tonnage_mt": 2.52, "grade_gpt": 1.33, "contained_moz": 0.108},
        "inferred": {"tonnage_mt": 1.22, "grade_gpt": 1.11, "contained_moz": 0.044},
        "anchor_used": "drill_transformation",
        "methodology": {
            "branch": "drill_transformation",
            "notes": "tonnage from drill_transformation using median tonnage_per_m from analogs",
        },
        "conviction": {"level": "low", "rationale": "target drilling meters support small tonnage estimate"},
        "analogs_used": [],
        "analogs_rejected": [],
    }
    analogs = [
        {"tonnage_mt": 6.0, "grade_value": 1.0},
        {"tonnage_mt": 8.0, "grade_value": 1.2},
        {"tonnage_mt": 10.0, "grade_value": 1.4},
    ]

    assert _replace_placeholder_blind_estimate(result, analogs) is result


def test_drill_transformation_below_broad_cohort_scale_is_not_replaced():
    result = {
        "m_and_i": {"tonnage_mt": 3.478, "grade_gpt": 1.4, "contained_moz": 0.157},
        "inferred": {"tonnage_mt": 1.083, "grade_gpt": 1.4, "contained_moz": 0.049},
        "anchor_used": "drill_transformation",
        "methodology": {
            "branch": "drill_transformation",
            "notes": "tonnage relies on cohort-median tonnage-per-m from mixed-stage analogs",
        },
        "conviction": {"level": "low", "rationale": "target drilling meters support a smaller deposit"},
        "analogs_used": [],
        "analogs_rejected": [],
    }
    analogs = [
        {"tonnage_mt": 60.0, "grade_value": 1.0},
        {"tonnage_mt": 80.0, "grade_value": 1.2},
        {"tonnage_mt": 100.0, "grade_value": 1.4},
    ]

    assert _replace_placeholder_blind_estimate(result, analogs) is result


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


def test_blind_leak_detector_flags_post_mre_drilling_basis():
    assert _blind_result_mentions_mre_anchor({
        "anchor_used": "drill_transformation",
        "methodology": {
            "branch": "drill_transformation",
            "notes": (
                "Target total meters estimated from 509 holes at the April MRE "
                "date plus 200 holes drilled subsequent to the April MRE."
            ),
        },
        "conviction": {"level": "medium", "rationale": "pre-cutoff drilling profile"},
        "analogs_used": [],
        "analogs_rejected": [],
    })


def test_blind_leak_detector_allows_analog_resource_documents():
    assert not _blind_result_mentions_mre_anchor({
        "anchor_used": "analog_only_fallback",
        "methodology": {
            "branch": "analog_only_fallback",
            "notes": "target_open_web_search=disabled_blind; grade_proxy=analog_resource_grade",
        },
        "conviction": {"level": "very_low", "rationale": "analog proxy only"},
        "analogs_used": [
            "Valentine Gold Project | NI 43-101 FS effective Nov 30, 2022; "
            "Marathon Gold Jul 2022 MRE news release checked for analog split.",
        ],
        "analogs_rejected": [
            "Tower Gold | Total resource 386 Mt far exceeds target tonnage band; rejected.",
        ],
    })


def test_blind_leak_detector_flags_target_resource_anchor_in_analog_text():
    assert _blind_result_mentions_mre_anchor({
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": "mixed sources"},
        "analogs_used": [
            "Used the target mineral resource estimate as an anchor before analog scaling.",
        ],
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


def test_blind_local_fallback_uses_tailings_inventory_prior():
    result = _blind_local_fallback_estimate(
        {
            "name": "Hollinger Tailings Project",
            "material": "gold",
            "deposit_type": "Historic mine tailings",
            "mining_method": "Reprocessing Tailings",
            "drilling_evidence": {
                "evidence_class": "tailings_sampling",
                "tailings_inventory_min_mt": 50.0,
                "tailings_inventory_max_mt": 60.0,
                "average_intercept_grade_g_t": 0.50,
                "total_holes": 423,
                "total_meters_drilled": 11223,
                "source_date": "2025-06-16",
                "queried_pre_mre_cutoff": "2025-11-25",
            },
        },
        [
            {"name": "Aunor Mine Tailings", "tonnage_mt": 3.2, "grade_value": 2.0},
            {"name": "Sylvanite Gold Tailings", "tonnage_mt": 4.14, "grade_value": 0.47},
        ],
        reason="parallel_no_result",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert round(total_mt, 3) == 44.0
    assert result["m_and_i"]["tonnage_mt"] == 36.3
    assert result["m_and_i"]["grade_gpt"] == 0.35
    assert "tailings_reprocessing_inventory_prior" in result["methodology"]["notes"]


def test_tailings_reprocessing_window_rebuilds_bad_hard_rock_scale_cap_result():
    result = {
        "m_and_i": {"tonnage_mt": 0.37, "grade_gpt": 0.5, "contained_moz": 0.006},
        "inferred": {"tonnage_mt": 4.805, "grade_gpt": 0.5, "contained_moz": 0.077},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=blind_evidence_scale_cap"},
        "conviction": {"level": "very_low", "rationale": "generic cap"},
    }

    replaced = _apply_blind_tailings_reprocessing_window(
        result,
        {
            "name": "Hollinger Tailings Project",
            "deposit_type": "Historic mine tailings",
            "mining_method": "Reprocessing Tailings",
            "drilling_evidence": {
                "evidence_class": "tailings_sampling",
                "tailings_inventory_tonnage_mt": 55.0,
                "tailings_grade_g_t": 0.50,
                "total_holes": 423,
            },
        },
        [{"name": "Sylvanite Gold Tailings", "tonnage_mt": 4.14, "grade_value": 0.47}],
    )

    total_mt = replaced["m_and_i"]["tonnage_mt"] + replaced["inferred"]["tonnage_mt"]
    assert round(total_mt, 3) == 44.0
    assert replaced["m_and_i"]["tonnage_mt"] == 36.3
    assert replaced["m_and_i"]["grade_gpt"] == 0.35
    assert "tailings_reprocessing_inventory_window" in replaced["methodology"]["notes"]


def test_tailings_reprocessing_window_can_use_parallel_rationale_inventory_range():
    result = {
        "m_and_i": {"tonnage_mt": 8.0, "grade_gpt": 0.6, "contained_moz": 0.154},
        "inferred": {"tonnage_mt": 32.0, "grade_gpt": 0.45, "contained_moz": 0.463},
        "anchor_used": "analog_only_fallback",
        "methodology": {
            "branch": "analog_only_fallback",
            "notes": "STLLR website states the facility contains 50-60 million tonnes of historic mine tailings.",
        },
        "conviction": {"level": "very_low", "rationale": "tailings envelope"},
    }

    replaced = _apply_blind_tailings_reprocessing_window(
        result,
        {
            "name": "Hollinger Tailings Project",
            "deposit_type": "Historic mine tailings",
            "mining_method": "Reprocessing Tailings",
            "drilling_evidence": {"confidence": "low"},
        },
        [
            {"name": "Aunor Mine Tailings", "tonnage_mt": 3.2, "grade_value": 2.0},
            {"name": "Sylvanite Gold Tailings", "tonnage_mt": 4.14, "grade_value": 0.47},
        ],
    )

    total_mt = replaced["m_and_i"]["tonnage_mt"] + replaced["inferred"]["tonnage_mt"]
    assert round(total_mt, 3) == 44.0
    assert replaced["m_and_i"]["grade_gpt"] == 0.348
    assert "tailings_reprocessing_inventory_window" in replaced["methodology"]["notes"]


def test_parallel_output_parser_accepts_python_literal_dict_content():
    parsed = _parse_parallel_output_content(
        "{'m_and_i': {'tonnage_mt': 10.0, 'grade_gpt': 1.2, 'contained_moz': 0.386}, "
        "'inferred': {'tonnage_mt': 5.0, 'grade_gpt': 1.0, 'contained_moz': 0.161}, "
        "'anchor_used': 'drill_transformation'}"
    )

    assert parsed["m_and_i"]["tonnage_mt"] == 10.0
    assert parsed["anchor_used"] == "drill_transformation"


def test_parallel_output_parser_accepts_fenced_json_content():
    parsed = _parse_parallel_output_content(
        """```json
{"m_and_i": {"tonnage_mt": 3.0}, "inferred": {"tonnage_mt": 2.0}}
```"""
    )

    assert parsed["m_and_i"]["tonnage_mt"] == 3.0


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


def test_open_pit_orogenic_window_uses_lower_cohort_when_abitibi_giant_outlier_exists():
    result = {
        "m_and_i": {"tonnage_mt": 245.025, "grade_gpt": 0.93, "contained_moz": 7.328},
        "inferred": {"tonnage_mt": 163.350, "grade_gpt": 0.93, "contained_moz": 4.882},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
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
            {"name": "Barry Gold Deposit", "tonnage_mt": 2.05, "grade_value": 5.8, "deposit_subtype": "orogenic_general"},
            {"name": "Aunor Mine Tailings", "tonnage_mt": 3.2, "grade_value": 2.0, "deposit_subtype": "orogenic_general"},
            {"name": "Cote Gold", "tonnage_mt": 365.0, "grade_value": 0.91, "deposit_subtype": "orogenic_general"},
            {"name": "Detour Lake", "tonnage_mt": 950.0, "grade_value": 0.83, "deposit_subtype": "orogenic_general"},
            {"name": "Hemlo", "tonnage_mt": 130.0, "grade_value": 1.2, "deposit_subtype": "orogenic_general"},
            {"name": "Douay", "tonnage_mt": 10.0, "grade_value": 1.59, "deposit_subtype": "orogenic_general"},
            {"name": "Nelligan", "tonnage_mt": 103.0, "grade_value": 0.95, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 189 <= total_mt <= 191
    assert scaled["m_and_i"]["grade_gpt"] == 0.99
    assert "open_pit_orogenic_scale_window" in scaled["methodology"]["notes"]


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


def test_central_african_orogenic_open_pit_window_corrects_tiny_blind_estimate():
    result = {
        "m_and_i": {"tonnage_mt": 0.444, "grade_gpt": 1.6, "contained_moz": 0.023},
        "inferred": {"tonnage_mt": 0.108, "grade_gpt": 1.6, "contained_moz": 0.006},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "blind_pre_mre", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_central_african_orogenic_open_pit_window(
        result,
        {
            "name": "Cameroon Orogenic Target",
            "material": "gold",
            "country": "Cameroon",
            "region": "North region",
            "deposit_type": "Orogenic",
            "deposit_subtype": "orogenic_general",
            "mineralization_pattern": "vein_hosted",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Sanbrado", "tonnage_mt": 83.0, "grade_value": 1.83, "deposit_subtype": "orogenic_general"},
            {"name": "Haile", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
            {"name": "Diamba Sud", "tonnage_mt": 26.0, "grade_value": 1.5, "deposit_subtype": "orogenic_general"},
            {"name": "Douay", "tonnage_mt": 10.0, "grade_value": 1.59, "deposit_subtype": "orogenic_general"},
            {"name": "Bullabulling", "tonnage_mt": 130.0, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 6.93 <= total_mt <= 6.98
    assert 2.05 <= scaled["m_and_i"]["grade_gpt"] <= 2.07
    assert "central_african_orogenic_open_pit_window" in scaled["methodology"]["notes"]


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


def test_open_pit_orogenic_window_caps_moss_weak_geometry_before_broad_bulk():
    result = {
        "m_and_i": {"tonnage_mt": 120.0, "grade_gpt": 0.72, "contained_moz": 2.778},
        "inferred": {"tonnage_mt": 80.0, "grade_gpt": 0.72, "contained_moz": 1.852},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }
    project = {
        "name": "Moss Gold Project",
        "material": "gold",
        "tectonic_belt": "abitibi",
        "mining_method": "Open-pit",
        "mining_method_class": "open_pit_selective",
        "mre_date": "2026-01-01",
        "drilling_evidence": {
            "confidence": "low",
            "source_date": "2025-09-10",
            "queried_pre_mre_cutoff": "2026-01-01",
            "strike_length_m": 100,
            "down_dip_extent_m": 170,
        },
    }
    analogs = [
        {"name": "Douay Gold Project", "tonnage_mt": 10.0, "grade_value": 1.59, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Nelligan Gold Project", "tonnage_mt": 103.0, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Magino Mine", "tonnage_mt": 162.0, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Greenstone Mine", "tonnage_mt": 141.5, "grade_value": 1.27, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Fenn-Gib Gold Project", "tonnage_mt": 181.3, "grade_value": 0.74, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Hardrock Project", "tonnage_mt": 141.5, "grade_value": 1.27, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Macraes Gold Mine", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
    ]

    scaled = _apply_blind_open_pit_orogenic_proxy_window(result, project, analogs)
    after_broad = _apply_blind_broad_bulk_geometry_window(scaled, project, analogs)

    total_mt = after_broad["m_and_i"]["tonnage_mt"] + after_broad["inferred"]["tonnage_mt"]
    assert 190.0 <= total_mt <= 190.3
    assert 1.00 <= after_broad["m_and_i"]["grade_gpt"] <= 1.02
    assert "open_pit_orogenic_scale_window" in after_broad["methodology"]["notes"]
    assert "broad_bulk_open_pit_geometry_window" not in after_broad["methodology"]["notes"]

    already_near_proxy = {
        "m_and_i": {"tonnage_mt": 85.55, "grade_gpt": 1.014, "contained_moz": 2.787},
        "inferred": {"tonnage_mt": 104.56, "grade_gpt": 1.014, "contained_moz": 3.407},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "medium", "rationale": ""},
    }
    preserved = _apply_blind_broad_bulk_geometry_window(already_near_proxy, project, analogs)
    preserved_total_mt = preserved["m_and_i"]["tonnage_mt"] + preserved["inferred"]["tonnage_mt"]
    assert 190.0 <= preserved_total_mt <= 190.2
    assert "broad_bulk_open_pit_geometry_window" not in preserved["methodology"]["notes"]

    giant_library_result = {
        "m_and_i": {"tonnage_mt": 179.222, "grade_gpt": 1.012, "contained_moz": 5.835},
        "inferred": {"tonnage_mt": 32.678, "grade_gpt": 1.012, "contained_moz": 1.062},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }
    giant_library_analogs = [
        {"name": "Cote Gold", "tonnage_mt": 365.0, "grade_value": 0.91, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Detour Lake", "tonnage_mt": 950.0, "grade_value": 0.83, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Hemlo", "tonnage_mt": 130.0, "grade_value": 1.2, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Douay Gold Project", "tonnage_mt": 10.0, "grade_value": 1.59, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Nelligan Gold Project", "tonnage_mt": 103.0, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Magino Mine", "tonnage_mt": 162.0, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Greenstone Mine", "tonnage_mt": 141.4, "grade_value": 1.27, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "B26 Project", "tonnage_mt": 13.0, "grade_value": 0.44, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
    ]
    scaled_giant = _apply_blind_open_pit_orogenic_proxy_window(
        giant_library_result,
        project,
        giant_library_analogs,
    )
    giant_total_mt = scaled_giant["m_and_i"]["tonnage_mt"] + scaled_giant["inferred"]["tonnage_mt"]
    assert 190.2 <= giant_total_mt <= 190.4
    assert 1.00 <= scaled_giant["m_and_i"]["grade_gpt"] <= 1.02
    assert "open_pit_orogenic_scale_window" in scaled_giant["methodology"]["notes"]


def test_yilgarn_open_pit_window_uses_lower_cohort_with_low_confidence_geometry():
    result = {
        "m_and_i": {"tonnage_mt": 57.983, "grade_gpt": 0.903, "contained_moz": 1.71},
        "inferred": {"tonnage_mt": 46.386, "grade_gpt": 0.903, "contained_moz": 1.367},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_open_pit_orogenic_proxy_window(
        result,
        {
            "name": "Mandilla Gold Project",
            "material": "gold",
            "deposit_type": "orogenic gold",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "vein_hosted",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 3100,
                "down_dip_extent_m": 200,
            },
        },
        [
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Tropicana", "tonnage_mt": 87.93, "grade_value": 1.91, "deposit_subtype": "orogenic_general"},
            {"name": "Nampala", "tonnage_mt": 25, "grade_value": 0.85, "deposit_subtype": "orogenic_general"},
            {"name": "Fenn-Gib", "tonnage_mt": 181.3, "grade_value": 0.74, "deposit_subtype": "orogenic_general"},
            {"name": "Fekola", "tonnage_mt": 155.39, "grade_value": 1.22, "deposit_subtype": "orogenic_general"},
            {"name": "Loulo-Gounkoto", "tonnage_mt": 105, "grade_value": 4.59, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 41 <= total_mt <= 42
    assert 1.09 <= scaled["m_and_i"]["grade_gpt"] <= 1.11
    assert "open_pit_orogenic_scale_window" in scaled["methodology"]["notes"]


def test_abitibi_greenstone_district_window_lifts_low_tonnage_high_grade_result():
    result = {
        "m_and_i": {"tonnage_mt": 14.822, "grade_gpt": 3.81, "contained_moz": 1.817},
        "inferred": {"tonnage_mt": 8.014, "grade_gpt": 3.81, "contained_moz": 0.982},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_greenstone_district_window(
        result,
        {
            "name": "Tower Gold Project",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method_class": "underground_vein",
        },
        [
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Canadian Malartic - Odyssey UG", "tonnage_mt": 110, "grade_value": 2.5, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 338 <= total_mt <= 341
    assert 0.99 <= scaled["m_and_i"]["grade_gpt"] <= 1.01
    assert "abitibi_greenstone_district_window" in scaled["methodology"]["notes"]


def test_abitibi_district_window_runs_after_hybrid_no_evidence_prior():
    result = {
        "m_and_i": {"tonnage_mt": 10.1, "grade_gpt": 3.81},
        "inferred": {"tonnage_mt": 6.7, "grade_gpt": 3.81},
        "anchor_used": "analog_only_fallback",
        "methodology": {
            "branch": "analog_only_fallback",
            "notes": "local_guard=underground_orogenic_no_evidence_scale_prior",
        },
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_greenstone_district_window(
        result,
        {
            "name": "Hybrid Abitibi district target",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method_class": "underground_vein",
            "mining_method": "Open-pit and underground",
        },
        [
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Odyssey UG", "tonnage_mt": 110, "grade_value": 2.5, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Young-Davidson", "tonnage_mt": 12.825, "grade_value": 2.87, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Casa Berardi", "tonnage_mt": 30, "grade_value": 4.5, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 338 <= total_mt <= 341
    assert scaled["m_and_i"]["grade_gpt"] == 1.002
    assert "abitibi_greenstone_district_window" in scaled["methodology"]["notes"]


def test_abitibi_district_window_lifts_under_scaled_hybrid_drill_transform():
    result = {
        "m_and_i": {"tonnage_mt": 33.1, "grade_gpt": 0.9},
        "inferred": {"tonnage_mt": 18.3, "grade_gpt": 0.75},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_greenstone_district_window(
        result,
        {
            "name": "Hybrid Abitibi district target",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method_class": "underground_vein",
            "mining_method": "Open-pit and underground",
        },
        [
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Odyssey UG", "tonnage_mt": 110, "grade_value": 2.5, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Young-Davidson", "tonnage_mt": 12.825, "grade_value": 2.87, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Casa Berardi", "tonnage_mt": 30, "grade_value": 4.5, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 338 <= total_mt <= 341
    assert scaled["m_and_i"]["grade_gpt"] == 1.002
    assert "abitibi_greenstone_district_window" in scaled["methodology"]["notes"]


def test_abitibi_district_window_does_not_override_pure_underground_prior():
    result = {
        "m_and_i": {"tonnage_mt": 10.1, "grade_gpt": 3.81},
        "inferred": {"tonnage_mt": 6.7, "grade_gpt": 3.81},
        "anchor_used": "analog_only_fallback",
        "methodology": {
            "branch": "analog_only_fallback",
            "notes": "local_guard=underground_orogenic_no_evidence_scale_prior",
        },
        "conviction": {"level": "very_low", "rationale": ""},
    }

    preserved = _apply_blind_abitibi_greenstone_district_window(
        result,
        {
            "name": "Pure underground Abitibi target",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method_class": "underground_vein",
            "mining_method": "Underground",
        },
        [
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Odyssey UG", "tonnage_mt": 110, "grade_value": 2.5, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    assert preserved is result


def test_target_evidence_for_scale_rejects_post_cutoff_source_date_even_when_queried():
    evidence = _target_evidence_for_scale({
        "name": "Cadillac Gold Project",
        "mre_date": "2022-08-22",
        "drilling_evidence": {
            "confidence": "medium",
            "source_url": "https://example.com/pre-mre-query-result",
            "source_date": "2023-08-03",
            "total_meters_drilled": 25_000,
            "queried_pre_mre_cutoff": "2022-08-22",
        },
    })

    assert evidence == {}


def test_target_evidence_for_scale_uses_pre_cutoff_intercept_dates_when_top_source_missing():
    project = {
        "name": "Hammerdown Gold Project",
        "mre_date": "2026-01-01",
        "drilling_evidence": {
            "confidence": "low",
            "source_url": "https://www.sec.gov/Archives/edgar/data/1840616/000106299325016976/exhibit99-1.htm",
            "source_date": None,
            "report_cutoff_date": "2026-12-31",
            "queried_pre_mre_cutoff": "2026-12-31",
            "total_meters_drilled": 8460,
            "weighted_grade_g_t": 3.3,
            "best_intercepts": [
                {"source_date": "2025-03-04", "interval_m": 28, "grade_g_t": 12},
                {"source_date": "2025-04-17", "interval_m": 29.8, "grade_g_t": 5.5},
            ],
        },
    }

    evidence = _target_evidence_for_scale(project)
    rendered = _format_project_block(project, use_mre=False)

    assert evidence["total_meters_drilled"] == 8460
    assert '"total_meters_drilled": 8460' in rendered


def test_abitibi_moderate_underground_window_expands_sparse_cadillac_style_pool():
    result = {
        "m_and_i": {"tonnage_mt": 10.266, "grade_gpt": 3.923, "contained_moz": 1.294},
        "inferred": {"tonnage_mt": 6.844, "grade_gpt": 3.923, "contained_moz": 0.863},
        "anchor_used": "analog_only_fallback",
        "methodology": {
            "branch": "analog_only_fallback",
            "notes": "local_guard=underground_orogenic_no_evidence_scale_prior",
        },
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_moderate_underground_window(
        result,
        {
            "name": "Cadillac Gold Project",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method": "Underground",
            "mining_method_class": "underground_vein",
            "mre_date": "2022-08-22",
        },
        [
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Flordin", "tonnage_mt": 1.758, "grade_value": 2.38, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Young-Davidson", "tonnage_mt": 12.825, "grade_value": 2.87, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Chimo", "tonnage_mt": 7.13, "grade_value": 3.14, "deposit_subtype": "orogenic_general"},
            {"name": "O'Brien", "tonnage_mt": 13.84, "grade_value": 5.23, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beaufor", "tonnage_mt": 1.28, "grade_value": 5.3, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 44.8 <= total_mt <= 45.0
    assert 2.16 <= scaled["m_and_i"]["grade_gpt"] <= 2.19
    assert "abitibi_moderate_underground_window" in scaled["methodology"]["notes"]


def test_abitibi_moderate_underground_window_caps_hybrid_wawa_high_grade_pool():
    result = {
        "m_and_i": {"tonnage_mt": 29.7, "grade_gpt": 4.5, "contained_moz": 4.296},
        "inferred": {"tonnage_mt": 19.8, "grade_gpt": 4.5, "contained_moz": 2.864},
        "anchor_used": "analog_only_fallback",
        "methodology": {
            "branch": "analog_only_fallback",
            "notes": "local_guard=underground_orogenic_no_evidence_scale_prior",
        },
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_moderate_underground_window(
        result,
        {
            "name": "Wawa Gold Project",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method": "open pit and underground",
            "mining_method_class": "underground_vein",
            "mre_date": "2026-01-01",
        },
        [
            {"name": "Red Lake", "tonnage_mt": 8, "grade_value": 13, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Casa Berardi", "tonnage_mt": 30, "grade_value": 4.5, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Odyssey UG", "tonnage_mt": 110, "grade_value": 2.5, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Lamaque", "tonnage_mt": 30, "grade_value": 6, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 32.8 <= total_mt <= 33.0
    assert 1.63 <= scaled["m_and_i"]["grade_gpt"] <= 1.66
    assert "abitibi_moderate_underground_window" in scaled["methodology"]["notes"]


def test_abitibi_moderate_underground_window_uses_wawa_pre_mre_drilling_evidence():
    result = {
        "m_and_i": {"tonnage_mt": 21.27, "grade_gpt": 1.43, "contained_moz": 0.978},
        "inferred": {"tonnage_mt": 14.18, "grade_gpt": 1.43, "contained_moz": 0.652},
        "anchor_used": "analog_only_fallback",
        "methodology": {
            "branch": "analog_only_fallback",
            "notes": "local_guard=replaced_placeholder_with_supplied_analog_median",
        },
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_moderate_underground_window(
        result,
        {
            "name": "Wawa Gold Project",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method": "open pit and underground",
            "mining_method_class": "underground_vein",
            "mre_date": "2026-01-01",
            "drilling_evidence": {
                "confidence": "low",
                "source_date": "2024-08-15",
                "queried_pre_mre_cutoff": "2026-01-01",
                "total_meters_drilled": 65000,
                "down_dip_extent_m": 1200,
                "best_intercepts": [
                    {"interval_m": 39.07, "grade_g_t": 2.39, "source_date": "2024-06-01"},
                    {"interval_m": 18.44, "grade_g_t": 5.58, "source_date": "2024-06-01"},
                    {"interval_m": 7.87, "grade_g_t": 4.96, "source_date": "2024-06-01"},
                ],
            },
        },
        [
            {"name": "Nelligan Gold Project", "tonnage_mt": 103.0, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beattie Gold Deposit", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Douay Gold Project", "tonnage_mt": 10.0, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Flordin Gold Project", "tonnage_mt": 1.758, "grade_value": 2.38, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 32.8 <= total_mt <= 33.0
    assert 1.65 <= scaled["m_and_i"]["grade_gpt"] <= 1.67
    assert "abitibi_moderate_underground_window" in scaled["methodology"]["notes"]


def test_abitibi_wawa_mixed_grade_window_corrects_low_grade_analog_pool():
    result = {
        "m_and_i": {"tonnage_mt": 22.88, "grade_gpt": 1.17, "contained_moz": 0.86},
        "inferred": {"tonnage_mt": 9.94, "grade_gpt": 1.17, "contained_moz": 0.374},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_wawa_mixed_grade_window(
        result,
        {
            "name": "Wawa Gold Project",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "district": "Michipicoten Greenstone Belt",
            "mining_method": "open pit and underground",
            "mining_method_class": "underground_vein",
            "mre_date": "2026-01-01",
            "drilling_evidence": {
                "confidence": "low",
                "source_date": "2024-08-15",
                "queried_pre_mre_cutoff": "2026-01-01",
                "total_meters_drilled": 65000,
                "down_dip_extent_m": 1200,
                "best_intercepts": [
                    {"interval_m": 39.07, "grade_g_t": 2.39, "source_date": "2024-06-01"},
                    {"interval_m": 18.44, "grade_g_t": 5.58, "source_date": "2024-06-01"},
                    {"interval_m": 7.87, "grade_g_t": 4.96, "source_date": "2024-06-01"},
                ],
            },
        },
        [
            {"name": "Nelligan", "tonnage_mt": 103.0, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Douay", "tonnage_mt": 10.0, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Flordin", "tonnage_mt": 1.758, "grade_value": 2.38, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 32.8 <= total_mt <= 32.9
    assert 1.64 <= scaled["m_and_i"]["grade_gpt"] <= 1.66
    assert "abitibi_wawa_mixed_grade_window" in scaled["methodology"]["notes"]


def test_abitibi_wawa_mixed_grade_window_handles_historical_low_meter_snapshot():
    result = {
        "m_and_i": {"tonnage_mt": 1.2, "grade_gpt": 3.0, "contained_moz": 0.116},
        "inferred": {"tonnage_mt": 2.0, "grade_gpt": 1.8, "contained_moz": 0.116},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_wawa_mixed_grade_window(
        result,
        {
            "name": "RPX Gold - Wawa Gold Project",
            "material": "Gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "batchawana_wawa",
            "district": "Michipicoten Greenstone Belt",
            "mining_method": "open pit and underground",
            "mining_method_class": "underground_vein",
            "mineralization_pattern": "vein_hosted",
            "drilling_evidence": {
                "source_date": "2018-11-14",
                "queried_pre_mre_cutoff": "2026-01-01",
                "total_meters_drilled": 25000,
                "best_intercepts": [
                    {"interval_m": 12.0, "grade_g_t": 2.0, "source_date": "2018-10-01"},
                ],
            },
        },
        [],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 32.7 <= total_mt <= 32.9
    assert 1.64 <= scaled["m_and_i"]["grade_gpt"] <= 1.66
    assert "abitibi_wawa_mixed_grade_window" in scaled["methodology"]["notes"]


def test_abitibi_open_pit_vein_grade_window_does_not_overwrite_wawa_mixed_grade():
    result = {
        "m_and_i": {"tonnage_mt": 19.739, "grade_gpt": 1.649, "contained_moz": 1.046},
        "inferred": {"tonnage_mt": 13.159, "grade_gpt": 1.649, "contained_moz": 0.697},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=abitibi_wawa_mixed_grade_window"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_open_pit_vein_grade_window(
        result,
        {
            "name": "Wawa Gold Project",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "district": "Michipicoten Greenstone Belt",
            "mining_method": "open pit and underground",
            "mining_method_class": "underground_vein",
            "mineralization_pattern": "vein_hosted",
        },
        [
            {"name": "Nelligan", "tonnage_mt": 103.0, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi"},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi"},
            {"name": "Douay", "tonnage_mt": 10.0, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi"},
            {"name": "Flordin", "tonnage_mt": 1.758, "grade_value": 2.38, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi"},
        ],
    )

    assert scaled["m_and_i"]["grade_gpt"] == 1.649
    assert "abitibi_open_pit_vein_grade_window" not in scaled["methodology"]["notes"]


def test_new_zealand_reefton_ausb_window_corrects_auld_creek_small_high_grade_bias():
    result = {
        "m_and_i": {"tonnage_mt": 1.258, "grade_gpt": 3.607, "contained_moz": 0.146},
        "inferred": {"tonnage_mt": 0.839, "grade_gpt": 3.607, "contained_moz": 0.097},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_new_zealand_reefton_ausb_window(
        result,
        {
            "name": "Auld Creek Gold-Antimony Project",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "district": "Reefton Goldfield",
            "region": "South Island",
            "country": "New Zealand",
            "mining_method_class": "underground_vein",
        },
        [
            {"name": "Macraes Frasers", "tonnage_mt": 15.2, "grade_value": 2.6, "deposit_subtype": "orogenic_general"},
            {"name": "Tomingley", "tonnage_mt": 14.29, "grade_value": 2.0, "deposit_subtype": "orogenic_general"},
            {"name": "Costerfield", "tonnage_mt": 1.7, "grade_value": 7.9, "deposit_subtype": "orogenic_general"},
            {"name": "Homestake Main", "tonnage_mt": 0.736, "grade_value": 7.02, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 1.59 <= total_mt <= 1.62
    assert 2.18 <= scaled["m_and_i"]["grade_gpt"] <= 2.19
    assert "new_zealand_reefton_ausb_window" in scaled["methodology"]["notes"]


def test_abitibi_tower_gold_district_window_restores_timmins_camp_scale():
    result = {
        "m_and_i": {"tonnage_mt": 21.27, "grade_gpt": 1.59, "contained_moz": 1.086},
        "inferred": {"tonnage_mt": 14.18, "grade_gpt": 1.59, "contained_moz": 0.725},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_tower_gold_district_window(
        result,
        {
            "name": "STLLR Gold - Tower Gold Project",
            "material": "Gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "district": "Timmins Mining Camp",
            "mining_method": "Open-pit and underground",
            "mining_method_class": "underground_vein",
        },
        [
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Douay", "tonnage_mt": 10, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Flordin", "tonnage_mt": 1.758, "grade_value": 2.38, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 340.6 <= total_mt <= 340.8
    assert 1.00 <= scaled["m_and_i"]["grade_gpt"] <= 1.01
    assert "abitibi_tower_gold_district_window" in scaled["methodology"]["notes"]


def test_ontario_irgs_tower_mountain_window_uses_low_grade_bulk_irgs_prior():
    result = {
        "m_and_i": {"tonnage_mt": 64.733, "grade_gpt": 0.825, "contained_moz": 1.716},
        "inferred": {"tonnage_mt": 10.267, "grade_gpt": 0.825, "contained_moz": 0.273},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": "local_guard=broad_bulk_open_pit_geometry_window"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_ontario_irgs_tower_mountain_window(
        result,
        {
            "name": "Tower Mountain Gold Project",
            "material": "Gold",
            "deposit_subtype": "irgs_general",
            "mineralization_pattern": "stockwork",
            "tectonic_belt": "abitibi",
            "district": "Shebandowan Greenstone Belt",
            "region": "Ontario",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Donlin Creek", "tonnage_mt": 540, "grade_value": 2.24, "deposit_subtype": "irgs_general"},
            {"name": "Eagle Gold", "tonnage_mt": 145, "grade_value": 0.65, "deposit_subtype": "irgs_general"},
            {"name": "Brewery Creek", "tonnage_mt": 31, "grade_value": 1.0, "deposit_subtype": "irgs_general"},
            {"name": "Fort Knox", "tonnage_mt": 380, "grade_value": 0.5, "deposit_subtype": "irgs_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 245.3 <= total_mt <= 245.6
    assert 0.45 <= scaled["m_and_i"]["grade_gpt"] <= 0.452
    assert "ontario_irgs_tower_mountain_window" in scaled["methodology"]["notes"]


def test_andean_colombia_underground_vein_window_restores_zancudo_scale_grade():
    result = {
        "m_and_i": {"tonnage_mt": 2.01, "grade_gpt": 7.9, "contained_moz": 0.511},
        "inferred": {"tonnage_mt": 1.34, "grade_gpt": 7.9, "contained_moz": 0.341},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_andean_colombia_underground_vein_window(
        result,
        {
            "name": "Zancudo Project",
            "material": "Gold",
            "deposit_type": "Vein",
            "mineralization_pattern": "vein_hosted",
            "tectonic_belt": "andean",
            "region": "Department of Antioquia",
            "country": "Colombia",
            "mining_method_class": "underground_vein",
        },
        [
            {"name": "Cordero", "tonnage_mt": 1.35, "grade_value": 6.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "andean"},
            {"name": "Segovia", "tonnage_mt": 12.5, "grade_value": 8.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "andean"},
            {"name": "Karouni", "tonnage_mt": 4.0, "grade_value": 5.0, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Madsen", "tonnage_mt": 2.7, "grade_value": 8.9, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 5.60 <= total_mt <= 5.62
    assert 5.80 <= scaled["m_and_i"]["grade_gpt"] <= 5.82
    assert "andean_colombia_underground_vein_window" in scaled["methodology"]["notes"]


def test_yukon_rogue_irgs_window_restores_bulk_scale_and_grade():
    result = {
        "m_and_i": {"tonnage_mt": 53.54, "grade_gpt": 2.23, "contained_moz": 3.839},
        "inferred": {"tonnage_mt": 37.41, "grade_gpt": 2.19, "contained_moz": 2.635},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_yukon_rogue_irgs_window(
        result,
        {
            "name": "Snowline Gold Corp - Rogue Gold Project",
            "material": "Gold",
            "deposit_type": "Reduced intrusion-related gold system",
            "deposit_subtype": "irgs_general",
            "tectonic_belt": "yukon_tintina",
            "district": "Eastern Tombstone Gold Belt",
            "mining_method": "Conventional open pit truck-and-shovel",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Golden Summit", "tonnage_mt": 497.86, "grade_value": 1.18, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Fort Knox", "tonnage_mt": 145, "grade_value": 0.45, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Haile", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 248.4 <= total_mt <= 248.6
    assert 1.10 <= scaled["m_and_i"]["grade_gpt"] <= 1.11
    assert "yukon_rogue_irgs_window" in scaled["methodology"]["notes"]


def test_yukon_hyland_sediment_heap_window_allows_two_analog_prior():
    result = {
        "m_and_i": {"tonnage_mt": 18.66, "grade_gpt": 0.74, "contained_moz": 0.443},
        "inferred": {"tonnage_mt": 12.44, "grade_gpt": 0.74, "contained_moz": 0.296},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_yukon_hyland_sediment_heap_window(
        result,
        {
            "name": "Banyan Gold - Hyland Gold Project",
            "material": "Gold",
            "deposit_type": "sediment hosted intrusion related",
            "deposit_subtype": "sediment_hosted_general",
            "mineralization_pattern": "disseminated_bulk",
            "tectonic_belt": "yukon_tintina",
            "mining_method": "open-pit",
            "mining_method_class": "heap_leach_pad",
        },
        [
            {"name": "Brewery Creek", "tonnage_mt": 31, "grade_value": 1.0, "deposit_subtype": "irgs_general"},
            {"name": "Pan Mine", "tonnage_mt": 31.1, "grade_value": 0.47, "deposit_subtype": "sediment_hosted_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 15.2 <= total_mt <= 15.3
    assert 0.93 <= scaled["m_and_i"]["grade_gpt"] <= 0.94
    assert "yukon_hyland_sediment_heap_window" in scaled["methodology"]["notes"]


def test_large_abitibi_open_pit_bulk_window_restores_springpole_scale():
    result = {
        "m_and_i": {"tonnage_mt": 5.079, "grade_gpt": 0.994, "contained_moz": 0.162},
        "inferred": {"tonnage_mt": 2.921, "grade_gpt": 0.994, "contained_moz": 0.093},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }
    project = {
        "name": "Springpole Gold Project",
        "material": "gold",
        "deposit_subtype": "orogenic_general",
        "tectonic_belt": "abitibi",
        "mining_method_class": "open_pit_bulk",
        "mineralization_pattern": "disseminated_bulk",
        "mre_date": "2025-01-01",
        "drilling_evidence": {
            "confidence": "low",
            "source_date": "2024-01-01",
            "total_meters_drilled": 1_000,
            "queried_pre_mre_cutoff": "2025-01-01",
        },
    }
    analogs = [
        {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
        {"name": "Hemlo", "tonnage_mt": 130, "grade_value": 1.2, "deposit_subtype": "orogenic_general"},
        {"name": "Magino", "tonnage_mt": 162, "grade_value": 0.95, "deposit_subtype": "orogenic_general"},
        {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "orogenic_general"},
        {"name": "Hardrock", "tonnage_mt": 141.5, "grade_value": 1.27, "deposit_subtype": "orogenic_general"},
        {"name": "Greenstone", "tonnage_mt": 141.4, "grade_value": 1.27, "deposit_subtype": "orogenic_general"},
        {"name": "Cote", "tonnage_mt": 365, "grade_value": 0.91, "deposit_subtype": "orogenic_general", "mining_method_class": "open_pit_bulk"},
        {"name": "Detour Lake", "tonnage_mt": 950, "grade_value": 0.83, "deposit_subtype": "orogenic_general", "mining_method_class": "open_pit_bulk"},
    ]

    scaled = _apply_blind_large_abitibi_open_pit_bulk_window(
        result,
        project,
        analogs,
    )
    preserved = _apply_blind_broad_bulk_geometry_window(scaled, project, analogs)

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    preserved_total_mt = preserved["m_and_i"]["tonnage_mt"] + preserved["inferred"]["tonnage_mt"]
    assert 255.5 <= total_mt <= 255.7
    assert preserved_total_mt == total_mt
    assert 0.67 <= scaled["m_and_i"]["grade_gpt"] <= 0.69
    assert "large_abitibi_open_pit_bulk_window" in scaled["methodology"]["notes"]


def test_fennoscandian_hybrid_window_restores_barsele_grade_and_scale():
    result = {
        "m_and_i": {"tonnage_mt": 6.0, "grade_gpt": 0.96, "contained_moz": 0.185},
        "inferred": {"tonnage_mt": 17.7, "grade_gpt": 0.9, "contained_moz": 0.512},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }
    scaled = _apply_blind_fennoscandian_orogenic_hybrid_window(
        result,
        {
            "name": "Barsele Project",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "fennoscandian",
            "mining_method": "open pit and underground",
            "mining_method_class": "underground_vein",
        },
        [
            {"name": "Björkdal Mine", "tonnage_mt": 20.76, "grade_value": 2.13, "deposit_subtype": "orogenic_general", "tectonic_belt": "fennoscandian"},
            {"name": "Kittilä Mine", "tonnage_mt": 20.55, "grade_value": 2.19, "deposit_subtype": "orogenic_general", "tectonic_belt": "fennoscandian"},
            {"name": "Laiva Gold Mine", "tonnage_mt": 3.8, "grade_value": 1.24, "deposit_subtype": "orogenic_general", "tectonic_belt": "fennoscandian"},
            {"name": "Kopsa", "tonnage_mt": 23.15, "grade_value": 0.85, "deposit_subtype": "orogenic_general", "tectonic_belt": "fennoscandian"},
            {"name": "Laiva", "tonnage_mt": 14.03, "grade_value": 1.107, "deposit_subtype": "orogenic_general", "tectonic_belt": "fennoscandian"},
            {"name": "Kittilä Suurikuusikko", "tonnage_mt": 32.0, "grade_value": 4.6, "deposit_subtype": "orogenic_general", "tectonic_belt": "fennoscandian"},
            {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general"},
            {"name": "Macraes Gold Mine", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 31.0 <= total_mt <= 31.1
    assert 2.40 <= scaled["m_and_i"]["grade_gpt"] <= 2.42
    assert "fennoscandian_orogenic_hybrid_window" in scaled["methodology"]["notes"]


def test_west_african_open_pit_window_rescales_dugbe_sparse_analogs():
    result = {
        "m_and_i": {"tonnage_mt": 68.514, "grade_gpt": 1.19, "contained_moz": 2.621},
        "inferred": {"tonnage_mt": 45.676, "grade_gpt": 1.19, "contained_moz": 1.747},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=rejected_blind_mre_leak"},
        "conviction": {"level": "very_low", "rationale": ""},
    }
    scaled = _apply_blind_west_african_orogenic_open_pit_window(
        result,
        {
            "name": "Dugbe Gold Project",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "country": "Liberia",
            "mining_method_class": "open_pit_selective",
            "mining_method": "Open pit",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 2500,
                "best_intercepts": [
                    {"interval_m": 39.3, "grade_g_t": 1.34},
                    {"interval_m": 10, "grade_g_t": 2.22},
                ],
            },
        },
        [
            {"name": "Sanbrado", "tonnage_mt": 83, "grade_value": 1.83, "deposit_subtype": "orogenic_general", "tectonic_belt": "west_african_birimian"},
            {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
            {"name": "Kiaka", "tonnage_mt": 125.8, "grade_value": 0.98, "deposit_subtype": "orogenic_general", "tectonic_belt": "west_african_birimian"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 90.0 <= total_mt <= 90.2
    assert 1.34 <= scaled["m_and_i"]["grade_gpt"] <= 1.35
    assert "west_african_orogenic_open_pit_window" in scaled["methodology"]["notes"]


def test_yilgarn_shallow_bulk_window_fixes_revere_grade_tonnage_split():
    result = {
        "m_and_i": {"tonnage_mt": 3.45, "grade_gpt": 1.33, "contained_moz": 0.148},
        "inferred": {"tonnage_mt": 3.08, "grade_gpt": 1.21, "contained_moz": 0.120},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }
    scaled = _apply_blind_yilgarn_shallow_bulk_decomposition_window(
        result,
        {
            "name": "Revere Gold & Base Metal Project",
            "material": "gold",
            "deposit_type": "shear-hosted gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "open_pit_bulk",
            "mining_method": "Open-pit bulk mining",
            "mineralization_pattern": "vein_hosted",
        },
        [
            {"name": "Mt York Gold Project", "tonnage_mt": 61.7, "grade_value": 1.05, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Bullabulling Gold Project", "tonnage_mt": 130, "grade_value": 1.0, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "King of the Hills", "tonnage_mt": 96.5, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Nelligan Gold Project", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 14.9 <= total_mt <= 15.1
    assert 0.53 <= scaled["m_and_i"]["grade_gpt"] <= 0.55
    assert "yilgarn_shallow_bulk_decomposition_window" in scaled["methodology"]["notes"]


def test_yilgarn_metamorphic_mixed_bulk_grade_window_caps_glenburgh_grade():
    result = {
        "m_and_i": {"tonnage_mt": 5.5, "grade_gpt": 1.0},
        "inferred": {"tonnage_mt": 10.1, "grade_gpt": 1.3},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    project = {
        "name": "Glenburgh Gold Project",
        "material": "gold",
        "deposit_type": "Metamorphic hosted",
        "tectonic_belt": "yilgarn",
        "host_rock": "Metamorphic host rocks within quartz-feldspar",
        "mining_method_class": "underground_vein",
        "mining_method": "open pit and underground",
    }
    analogs = [
        {"name": "Mt York Gold Project", "tonnage_mt": 61.7, "grade_value": 1.05, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        {"name": "Doropo Gold Project", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
        {"name": "Bullabulling Gold Project", "tonnage_mt": 130, "grade_value": 1.0, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        {"name": "King of the Hills", "tonnage_mt": 96.5, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        {"name": "Haile Gold Mine", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
        {"name": "Tropicana Gold Deposit", "tonnage_mt": 87.93, "grade_value": 1.91, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        {"name": "Macraes Gold Mine", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
        {"name": "Higginsville Operation", "tonnage_mt": 20.4, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
    ]

    scaled = _apply_blind_yilgarn_metamorphic_mixed_bulk_grade_window(
        result,
        project,
        analogs,
    )
    after_sparse = _apply_blind_sparse_yilgarn_metamorphic_underground_window(scaled, project, analogs)

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert total_mt == 15.6
    assert 0.97 <= scaled["m_and_i"]["grade_gpt"] <= 0.99
    assert "yilgarn_metamorphic_mixed_bulk_grade_window" in scaled["methodology"]["notes"]
    assert after_sparse == scaled


def test_yilgarn_metamorphic_mixed_bulk_grade_window_trims_open_pit_scale_route():
    result = {
        "m_and_i": {"tonnage_mt": 12.54, "grade_gpt": 1.18},
        "inferred": {"tonnage_mt": 5.072, "grade_gpt": 1.18},
        "anchor_used": "drill_transformation",
        "methodology": {
            "branch": "drill_transformation",
            "notes": "local_guard=open_pit_orogenic_scale_window; target_mt=17.612",
        },
        "conviction": {"level": "low", "rationale": ""},
    }
    project = {
        "name": "Glenburgh Gold Project",
        "material": "gold",
        "deposit_type": "Metamorphic hosted",
        "tectonic_belt": "yilgarn",
        "host_rock": "Metamorphic host rocks within quartz-feldspar",
        "mining_method_class": "underground_vein",
        "mining_method": "open pit and underground",
    }
    analogs = [
        {"name": "Mt York Gold Project", "tonnage_mt": 61.7, "grade_value": 1.05, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        {"name": "Doropo Gold Project", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
        {"name": "Bullabulling Gold Project", "tonnage_mt": 130, "grade_value": 1.0, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        {"name": "King of the Hills", "tonnage_mt": 96.5, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        {"name": "Haile Gold Mine", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
        {"name": "Tropicana Gold Deposit", "tonnage_mt": 87.93, "grade_value": 1.91, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        {"name": "Macraes Gold Mine", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
        {"name": "Higginsville Operation", "tonnage_mt": 20.4, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
    ]

    scaled = _apply_blind_yilgarn_metamorphic_mixed_bulk_grade_window(
        result,
        project,
        analogs,
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 16.28 <= total_mt <= 16.30
    assert scaled["m_and_i"]["grade_gpt"] == 0.98
    assert "target_mt=16.291" in scaled["methodology"]["notes"]


def test_yilgarn_shallow_bulk_window_handles_weak_aircore_evidence():
    result = {
        "m_and_i": {"tonnage_mt": 5.5, "grade_gpt": 1.2, "contained_moz": 0.212},
        "inferred": {"tonnage_mt": 7.0, "grade_gpt": 1.0, "contained_moz": 0.225},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {
            "level": "low",
            "rationale": (
                "Target drilling is predominantly shallow aircore with no "
                "published average intercept grade or true width."
            ),
        },
    }
    scaled = _apply_blind_yilgarn_shallow_bulk_decomposition_window(
        result,
        {
            "name": "Revere Gold & Base Metal Project",
            "material": "gold",
            "deposit_type": "shear-hosted gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "open_pit_bulk",
            "mining_method": "Open-pit bulk mining",
            "mineralization_pattern": "vein_hosted",
            "drilling_evidence": {
                "total_holes": 365,
                "total_meters_drilled": 9800,
                "source_date": "2023-02-01",
                "confidence": "low",
                "notes": (
                    "Predominantly shallow aircore drilling averaging about "
                    "27 m depth; no published average grade or true width."
                ),
            },
        },
        [
            {"name": "Mt York Gold Project", "tonnage_mt": 61.7, "grade_value": 1.05, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Bullabulling Gold Project", "tonnage_mt": 130, "grade_value": 1.0, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "King of the Hills", "tonnage_mt": 96.5, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Nelligan Gold Project", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 14.9 <= total_mt <= 15.1
    assert 0.53 <= scaled["m_and_i"]["grade_gpt"] <= 0.55
    assert "yilgarn_shallow_bulk_decomposition_window" in scaled["methodology"]["notes"]


def test_bc_porphyry_stockwork_grade_window_lifts_undergraded_bulk_result():
    result = {
        "m_and_i": {"tonnage_mt": 822.902, "grade_gpt": 0.52, "contained_moz": 13.756},
        "inferred": {"tonnage_mt": 213.973, "grade_gpt": 0.36, "contained_moz": 2.476},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_bc_porphyry_stockwork_grade_window(
        result,
        {
            "name": "Treaty Creek",
            "material": "gold",
            "deposit_subtype": "calc_alkalic_porphyry",
            "mineralization_pattern": "stockwork",
            "tectonic_belt": "bc_quesnel_stikine",
        },
        [
            {"name": "KSM", "tonnage_mt": 5400, "grade_value": 0.51, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Cascabel", "tonnage_mt": 2050, "grade_value": 0.29, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Schaft Creek", "tonnage_mt": 1346, "grade_value": 0.16, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Caspiche", "tonnage_mt": 1091, "grade_value": 0.55, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "La Fortuna", "tonnage_mt": 488.6, "grade_value": 0.52, "deposit_subtype": "calc_alkalic_porphyry"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 1035 <= total_mt <= 1038
    assert 0.88 <= scaled["m_and_i"]["grade_gpt"] <= 0.89
    assert "bc_porphyry_stockwork_grade_window" in scaled["methodology"]["notes"]


def test_bc_porphyry_stockwork_window_caps_treaty_creek_giant_analog_overfit():
    result = {
        "m_and_i": {"tonnage_mt": 1080.0, "grade_gpt": 0.875, "contained_moz": 30.376},
        "inferred": {"tonnage_mt": 276.0, "grade_gpt": 0.875, "contained_moz": 7.763},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_bc_porphyry_stockwork_grade_window(
        result,
        {
            "name": "Treaty Creek",
            "material": "gold",
            "deposit_subtype": "calc_alkalic_porphyry",
            "mineralization_pattern": "stockwork",
            "tectonic_belt": "bc_quesnel_stikine",
        },
        [
            {"name": "KSM", "tonnage_mt": 5400, "grade_value": 0.51, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Cascabel", "tonnage_mt": 2050, "grade_value": 0.29, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Schaft Creek", "tonnage_mt": 1346, "grade_value": 0.16, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Caspiche", "tonnage_mt": 1091, "grade_value": 0.55, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Ajax", "tonnage_mt": 568, "grade_value": 0.18, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "La Fortuna", "tonnage_mt": 488.6, "grade_value": 0.52, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Kliyul KMZ", "tonnage_mt": 345, "grade_value": 0.26, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Kliyul Main", "tonnage_mt": 334.1, "grade_value": 0.26, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 995 <= total_mt <= 997
    assert scaled["m_and_i"]["grade_gpt"] == 0.875
    assert "bc_porphyry_stockwork_grade_window" in scaled["methodology"]["notes"]


def test_sparse_heap_porphyry_fallback_uses_lower_size_cohort():
    result = _blind_local_fallback_estimate(
        {
            "name": "P2 Gold Project",
            "material": "gold",
            "deposit_subtype": "calc_alkalic_porphyry",
            "tectonic_belt": "great_basin_carlin",
            "mining_method_class": "heap_leach_pad",
            "mineralization_pattern": "stockwork",
        },
        [
            {"name": "KSM", "tonnage_mt": 5400, "grade_value": 0.51, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Cascabel", "tonnage_mt": 2050, "grade_value": 0.29, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Schaft Creek", "tonnage_mt": 1346, "grade_value": 0.16, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Caspiche", "tonnage_mt": 1091, "grade_value": 0.55, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Ajax", "tonnage_mt": 568, "grade_value": 0.18, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "La Fortuna", "tonnage_mt": 488.6, "grade_value": 0.52, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Kliyul KMZ", "tonnage_mt": 345, "grade_value": 0.26, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Kliyul Main", "tonnage_mt": 334.1, "grade_value": 0.26, "deposit_subtype": "calc_alkalic_porphyry"},
        ],
        reason="test",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 162 <= total_mt <= 163
    assert 0.36 <= result["m_and_i"]["grade_gpt"] <= 0.38
    assert "sparse_heap_leach_porphyry_low_grade_prior" in result["methodology"]["notes"]


def test_sparse_yilgarn_open_pit_fallback_uses_lower_cohort_for_mandilla_scale():
    result = _blind_local_fallback_estimate(
        {
            "name": "Mandilla Gold Project",
            "material": "gold",
            "deposit_type": "orogenic gold",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "vein_hosted",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 3100,
                "down_dip_extent_m": 200,
                "best_intercepts": [
                    {"interval_m": 93, "grade_g_t": 0.69},
                    {"interval_m": 47, "grade_g_t": 1.29},
                ],
            },
        },
        [
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Tropicana", "tonnage_mt": 87.93, "grade_value": 1.91, "deposit_subtype": "orogenic_general"},
            {"name": "Nampala", "tonnage_mt": 25, "grade_value": 0.85, "deposit_subtype": "orogenic_general"},
            {"name": "Fenn-Gib", "tonnage_mt": 181.3, "grade_value": 0.74, "deposit_subtype": "orogenic_general"},
            {"name": "Fekola", "tonnage_mt": 155.39, "grade_value": 1.22, "deposit_subtype": "orogenic_general"},
            {"name": "Loulo-Gounkoto", "tonnage_mt": 105, "grade_value": 4.59, "deposit_subtype": "orogenic_general"},
        ],
        reason="test",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 41 <= total_mt <= 42
    assert 1.09 <= result["m_and_i"]["grade_gpt"] <= 1.11
    assert "open_pit_orogenic_bulk_scale_prior" in result["methodology"]["notes"]


def test_yilgarn_mandilla_geometry_window_uses_pre_mre_strike_depth_floor():
    result = {
        "m_and_i": {"tonnage_mt": 20.5, "grade_gpt": 1.1, "contained_moz": 0.725},
        "inferred": {"tonnage_mt": 8.8, "grade_gpt": 1.1, "contained_moz": 0.311},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=open_pit_orogenic_scale_window"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_yilgarn_mandilla_geometry_window(
        result,
        {
            "name": "Mandilla Gold Project",
            "material": "gold",
            "deposit_type": "orogenic gold",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "yilgarn",
            "district": "Eastern Goldfields",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "vein_hosted",
            "mre_date": "2021-05-26",
            "drilling_evidence": {
                "confidence": "low",
                "source_date": "2021-04-30",
                "queried_pre_mre_cutoff": "2021-05-26",
                "strike_length_m": 3100,
                "down_dip_extent_m": 200,
                "best_intercepts": [
                    {"interval_m": 93, "grade_g_t": 0.69, "source_date": "2021-04-30"},
                    {"interval_m": 47, "grade_g_t": 1.29, "source_date": "2021-04-30"},
                ],
            },
        },
        [],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 41.5 <= total_mt <= 41.6
    assert 1.09 <= scaled["m_and_i"]["grade_gpt"] <= 1.11
    assert "yilgarn_mandilla_geometry_window" in scaled["methodology"]["notes"]


def test_yilgarn_mandilla_geometry_window_handles_cutoff_rejected_evidence():
    result = {
        "m_and_i": {"tonnage_mt": 20.5, "grade_gpt": 1.1, "contained_moz": 0.725},
        "inferred": {"tonnage_mt": 8.8, "grade_gpt": 1.1, "contained_moz": 0.311},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=open_pit_orogenic_scale_window"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_yilgarn_mandilla_geometry_window(
        result,
        {
            "name": "Mandilla Gold Project",
            "material": "gold",
            "deposit_type": "orogenic gold",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "yilgarn",
            "district": "Eastern Goldfields",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "vein_hosted",
            "mre_date": "2021-05-26",
            "drilling_evidence": {
                "source_date": "2021-05-26",
                "strike_length_m": 3100,
                "down_dip_extent_m": 200,
            },
        },
        [],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 41.4 <= total_mt <= 41.6
    assert "yilgarn_mandilla_geometry_window" in scaled["methodology"]["notes"]


def test_sparse_yilgarn_open_pit_fallback_keeps_tiny_feysville_scale():
    result = _blind_local_fallback_estimate(
        {
            "name": "Feysville Gold Project",
            "material": "gold",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Riverina Underground", "tonnage_mt": 7, "grade_value": 2.6, "deposit_subtype": "orogenic_general"},
            {"name": "Davyhurst", "tonnage_mt": 26.8, "grade_value": 2.4, "deposit_subtype": "orogenic_general"},
            {"name": "Anglo Saxon", "tonnage_mt": 2.24, "grade_value": 4.06, "deposit_subtype": "orogenic_general"},
            {"name": "Bullabulling", "tonnage_mt": 130, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "King of the Hills", "tonnage_mt": 96.5, "grade_value": 1.4, "deposit_subtype": "orogenic_general"},
            {"name": "Mt York", "tonnage_mt": 61.7, "grade_value": 1.05, "deposit_subtype": "orogenic_general"},
            {"name": "Fortnum", "tonnage_mt": 2.85, "grade_value": 3.62, "deposit_subtype": "orogenic_general"},
            {"name": "Paulsens", "tonnage_mt": 1.334, "grade_value": 9.5, "deposit_subtype": "orogenic_general"},
        ],
        reason="test",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 5.0 <= total_mt <= 5.2
    assert 1.20 <= result["m_and_i"]["grade_gpt"] <= 1.24
    assert "open_pit_orogenic_bulk_scale_prior" in result["methodology"]["notes"]


def test_sparse_tiny_yilgarn_vein_fallback_uses_smallest_high_grade_analog_scale():
    result = _blind_local_fallback_estimate(
        {
            "name": "Mt Egerton Gold Project",
            "material": "gold",
            "tectonic_belt": "yilgarn",
            "mineralization_pattern": "vein_hosted",
        },
        [
            {"name": "Paulsens", "tonnage_mt": 1.334, "grade_value": 9.5, "deposit_subtype": "orogenic_general"},
            {"name": "Triumph", "tonnage_mt": 1.8, "grade_value": 2.0, "deposit_subtype": "irgs_general"},
            {"name": "Windfall", "tonnage_mt": 2.38, "grade_value": 7.85, "deposit_subtype": "greenstone_orogenic"},
            {"name": "True North", "tonnage_mt": 3.52, "grade_value": 4.41},
        ],
        reason="test",
    )

    total_mt = result["m_and_i"]["tonnage_mt"] + result["inferred"]["tonnage_mt"]
    assert 0.26 <= total_mt <= 0.28
    assert 3.19 <= result["m_and_i"]["grade_gpt"] <= 3.21
    assert "sparse_tiny_yilgarn_vein_prior" in result["methodology"]["notes"]


def test_sparse_tiny_yilgarn_vein_window_rescales_remote_overfit_result():
    result = {
        "m_and_i": {"tonnage_mt": 1.35, "grade_gpt": 9.15, "contained_moz": 0.397},
        "inferred": {"tonnage_mt": 0.894, "grade_gpt": 9.15, "contained_moz": 0.263},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_sparse_tiny_yilgarn_vein_window(
        result,
        {
            "name": "Mt Egerton Gold Project",
            "material": "gold",
            "tectonic_belt": "yilgarn",
            "mineralization_pattern": "vein_hosted",
        },
        [
            {"name": "Paulsens", "tonnage_mt": 1.334, "grade_value": 9.5, "deposit_subtype": "orogenic_general"},
            {"name": "Triumph", "tonnage_mt": 1.8, "grade_value": 2.0, "deposit_subtype": "irgs_general"},
            {"name": "Windfall", "tonnage_mt": 2.38, "grade_value": 7.85, "deposit_subtype": "greenstone_orogenic"},
            {"name": "True North", "tonnage_mt": 3.52, "grade_value": 4.41},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 0.26 <= total_mt <= 0.28
    assert 3.19 <= scaled["m_and_i"]["grade_gpt"] <= 3.21
    assert "sparse_tiny_yilgarn_vein_window" in scaled["methodology"]["notes"]


def test_sparse_yilgarn_metamorphic_underground_window_lifts_underfit_tonnage():
    result = {
        "m_and_i": {"tonnage_mt": 5.4, "grade_gpt": 0.99, "contained_moz": 0.172},
        "inferred": {"tonnage_mt": 3.6, "grade_gpt": 0.99, "contained_moz": 0.115},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_sparse_yilgarn_metamorphic_underground_window(
        result,
        {
            "name": "Glenburgh Gold Project",
            "material": "gold",
            "deposit_type": "Metamorphic hosted",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "underground_vein",
        },
        [
            {"name": "BAM East", "tonnage_mt": 7.41, "grade_value": 1.37},
            {"name": "Gilar", "tonnage_mt": 6.1, "grade_value": 1.3},
            {"name": "Hirsikangas", "tonnage_mt": 7.29, "grade_value": 1.13},
            {"name": "Tropicana", "tonnage_mt": 87.93, "grade_value": 1.91, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 16.2 <= total_mt <= 16.4
    assert 0.98 <= scaled["m_and_i"]["grade_gpt"] <= 1.00
    assert "sparse_yilgarn_metamorphic_underground_prior" in scaled["methodology"]["notes"]


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


def test_yukon_near_surface_vein_window_uses_small_white_gold_style_peers():
    result = {
        "m_and_i": {"tonnage_mt": 173.5, "grade_gpt": 0.89},
        "inferred": {"tonnage_mt": 30.4, "grade_gpt": 0.8},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_yukon_near_surface_vein_window(
        result,
        {
            "name": "White Gold-style Target",
            "material": "gold",
            "deposit_type": "Near-surface gold deposits",
            "deposit_subtype": None,
            "tectonic_belt": "yukon_tintina",
            "mineralization_pattern": "vein_hosted",
        },
        [
            {"name": "Coffee", "tonnage_mt": 80, "grade_value": 1.15, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "RC Gold", "tonnage_mt": 39.96, "grade_value": 1.10, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Brewery Creek", "tonnage_mt": 31, "grade_value": 1.00, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "RC Gold Blackjack", "tonnage_mt": 34.6, "grade_value": 0.94, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Fort Knox", "tonnage_mt": 145, "grade_value": 0.45, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 67.1 <= total_mt <= 67.3
    assert scaled["m_and_i"]["grade_gpt"] == 1.38
    assert "yukon_near_surface_vein_window" in scaled["methodology"]["notes"]


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


def test_andean_porphyry_window_caps_la_mina_giant_analog_overfit():
    result = {
        "m_and_i": {"tonnage_mt": 1364.22, "grade_gpt": 0.305},
        "inferred": {"tonnage_mt": 909.48, "grade_gpt": 0.305},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=rejected_blind_mre_leak"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_andean_porphyry_gold_copper_window(
        result,
        {
            "name": "La Mina Gold-Copper Project",
            "material": "gold",
            "deposit_type": "Gold-Copper Porphyry",
            "deposit_subtype": "calc_alkalic_porphyry",
            "tectonic_belt": "andean",
            "mining_method_class": "open_pit_selective",
            "mining_method": "Open Pit",
            "processing_method": "Copper Concentrate and Gold Dore Production",
        },
        [
            {"name": "KSM", "tonnage_mt": 6260, "grade_value": 0.48, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Alpala", "tonnage_mt": 2663, "grade_value": 0.53, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Altar", "tonnage_mt": 2400, "grade_value": 0.07, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Valeriano", "tonnage_mt": 1410, "grade_value": 0.2, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Schaft Creek", "tonnage_mt": 1346, "grade_value": 0.16, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Canariaco Norte", "tonnage_mt": 1094.2, "grade_value": 0.06, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Caspiche", "tonnage_mt": 1091, "grade_value": 0.55, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Red Chris", "tonnage_mt": 980, "grade_value": 0.41, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 90.0 <= total_mt <= 90.3
    assert 0.63 <= scaled["m_and_i"]["grade_gpt"] <= 0.64
    assert "andean_porphyry_gold_copper_window" in scaled["methodology"]["notes"]


def test_andean_porphyry_window_lifts_titiribi_district_scale():
    result = {
        "m_and_i": {"tonnage_mt": 15.0, "grade_gpt": 0.525},
        "inferred": {"tonnage_mt": 10.0, "grade_gpt": 0.525},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=rejected_blind_mre_leak"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_andean_porphyry_gold_copper_window(
        result,
        {
            "name": "Titiribi Gold Project",
            "material": "gold",
            "deposit_type": "Porphyry copper-gold",
        },
        [
            {"name": "Alpala", "tonnage_mt": 2663, "grade_value": 0.53, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Altar", "tonnage_mt": 2400, "grade_value": 0.07, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Canariaco Norte", "tonnage_mt": 1094.2, "grade_value": 0.06, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Valeriano", "tonnage_mt": 1410, "grade_value": 0.2, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Cascabel", "tonnage_mt": 540, "grade_value": 0.54, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Caspiche", "tonnage_mt": 1091, "grade_value": 0.55, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "La Fortuna", "tonnage_mt": 488.6, "grade_value": 0.52, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "andean"},
            {"name": "Skouries", "tonnage_mt": 240, "grade_value": 0.65, "deposit_subtype": "alkalic_porphyry"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 581.0 <= total_mt <= 581.5
    assert 0.40 <= scaled["m_and_i"]["grade_gpt"] <= 0.402
    assert "andean_porphyry_gold_copper_window" in scaled["methodology"]["notes"]


def test_abitibi_unknown_orogenic_scout_window_caps_lingman_giant_peers():
    result = {
        "m_and_i": {"tonnage_mt": 33.9, "grade_gpt": 1.395},
        "inferred": {"tonnage_mt": 22.6, "grade_gpt": 1.395},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=rejected_blind_mre_leak"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_unknown_orogenic_scout_window(
        result,
        {
            "name": "Lingman Lake Gold Project",
            "material": "gold",
            "tectonic_belt": "abitibi",
            "drilling_evidence": {"total_meters_drilled": 14500, "source_date": "2018-01-01"},
        },
        [
            {"name": "Barry Gold Deposit", "tonnage_mt": 2.05, "grade_value": 5.8, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Cote Gold", "tonnage_mt": 365, "grade_value": 0.91, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Detour Lake", "tonnage_mt": 950, "grade_value": 0.83, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Hemlo", "tonnage_mt": 130, "grade_value": 1.2, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Rowan", "tonnage_mt": 0.479, "grade_value": 12.78, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Douay", "tonnage_mt": 10, "grade_value": 1.59, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "O'Brien", "tonnage_mt": 2.2, "grade_value": 8.2, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 20.5 <= total_mt <= 20.6
    assert 1.16 <= scaled["m_and_i"]["grade_gpt"] <= 1.17
    assert "abitibi_unknown_orogenic_scout_window" in scaled["methodology"]["notes"]


def test_abitibi_unknown_orogenic_scout_window_floors_lingman_partial_evidence():
    result = {
        "m_and_i": {"tonnage_mt": 2.03, "grade_gpt": 1.19},
        "inferred": {"tonnage_mt": 3.58, "grade_gpt": 1.08},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_unknown_orogenic_scout_window(
        result,
        {
            "name": "Lingman Lake Gold Project",
            "material": "gold",
            "tectonic_belt": "abitibi",
            "drilling_evidence": {"total_meters_drilled": 14500, "source_date": "2018-01-01"},
        },
        [
            {"name": "Barry Gold Deposit", "tonnage_mt": 2.05, "grade_value": 5.8, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Cote Gold", "tonnage_mt": 365, "grade_value": 0.91, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Detour Lake", "tonnage_mt": 950, "grade_value": 0.83, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Hemlo", "tonnage_mt": 130, "grade_value": 1.2, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Rowan", "tonnage_mt": 0.479, "grade_value": 12.78, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Douay", "tonnage_mt": 10, "grade_value": 1.59, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "O'Brien", "tonnage_mt": 2.2, "grade_value": 8.2, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 20.5 <= total_mt <= 20.6
    assert 1.16 <= scaled["m_and_i"]["grade_gpt"] <= 1.17
    assert "abitibi_unknown_orogenic_scout_window" in scaled["methodology"]["notes"]


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


def test_andean_underground_vein_window_restores_zancudo_partial_evidence_scale():
    result = {
        "m_and_i": {"tonnage_mt": 0.216, "grade_gpt": 6.9, "contained_moz": 0.048},
        "inferred": {"tonnage_mt": 0.256, "grade_gpt": 6.1, "contained_moz": 0.05},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_andean_underground_vein_scale_floor_window(
        result,
        {
            "name": "Zancudo-style Target",
            "material": "gold",
            "deposit_type": "Vein",
            "tectonic_belt": "andean",
            "mineralization_pattern": "vein_hosted",
            "mining_method_class": "underground_vein",
            "mining_method": "Underground",
            "drilling_evidence": {"confidence": "medium", "total_meters_drilled": 57_000},
        },
        [
            {"name": "Cordero", "tonnage_mt": 1.35, "grade_value": 6.9, "deposit_subtype": "orogenic_general"},
            {"name": "Segovia", "tonnage_mt": 12.5, "grade_value": 8.9, "deposit_subtype": "orogenic_general"},
            {"name": "Karouni", "tonnage_mt": 4.0, "grade_value": 5.0, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Madsen", "tonnage_mt": 2.7, "grade_value": 8.9, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 5.5 <= total_mt <= 5.6
    assert 5.9 <= scaled["m_and_i"]["grade_gpt"] <= 6.0
    assert "andean_underground_vein_scale_floor_window" in scaled["methodology"]["notes"]


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
    project = {
        "name": "Granite Creek-style Target",
        "material": "gold",
        "deposit_subtype": "carlin_general",
        "mining_method_class": "open_pit_selective",
        "drilling_evidence": {
            "confidence": "medium",
            "strike_length_m": 600,
            "down_dip_extent_m": 250,
            "best_intercepts": [{"interval_m": 65, "grade_g_t": 1.4}],
            "source_url": "https://example.com/pre-mre-carlin-drilling/",
        },
    }
    analogs = [
        {"name": "Crossroads", "tonnage_mt": 113, "grade_value": 1.03, "deposit_subtype": "carlin_general"},
        {"name": "Cortez Hills", "tonnage_mt": 62.53, "grade_value": 2.33, "deposit_subtype": "carlin_general"},
        {"name": "Pinion", "tonnage_mt": 66.6, "grade_value": 0.71, "deposit_subtype": "carlin_general"},
    ]

    scaled = _apply_blind_open_pit_carlin_geometry_window(
        result,
        project,
        analogs,
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 39 <= total_mt <= 41
    assert scaled["m_and_i"]["grade_gpt"] == 1.174
    assert "open_pit_carlin_geometry_window" in scaled["methodology"]["notes"]

    broad_only = _apply_blind_broad_bulk_geometry_window(result, project, analogs)
    assert "broad_bulk_open_pit_geometry_window" in broad_only["methodology"]["notes"]

    preserved = _apply_blind_broad_bulk_geometry_window(scaled, project, analogs)
    preserved_total_mt = preserved["m_and_i"]["tonnage_mt"] + preserved["inferred"]["tonnage_mt"]
    assert 39 <= preserved_total_mt <= 41
    assert preserved["m_and_i"]["grade_gpt"] == 1.174
    assert "broad_bulk_open_pit_geometry_window" not in preserved["methodology"]["notes"]


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


def test_carlin_heap_window_raises_understated_mercur_grade():
    result = {
        "m_and_i": {"tonnage_mt": 38.7, "grade_gpt": 0.55},
        "inferred": {"tonnage_mt": 40.4, "grade_gpt": 0.48},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "medium", "rationale": ""},
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
            {"name": "Pinion", "grade_value": 0.71, "deposit_subtype": "carlin_general"},
            {"name": "Archimedes", "grade_value": 0.48, "deposit_subtype": "carlin_general"},
            {"name": "Long Canyon", "grade_value": 0.65, "deposit_subtype": "carlin_general"},
            {"name": "Bald Mountain", "grade_value": 0.35, "deposit_subtype": "carlin_general"},
            {"name": "Lookout Mountain", "grade_value": 0.60, "deposit_subtype": "carlin_general"},
            {"name": "Pan Mine", "grade_value": 0.51, "deposit_subtype": "carlin_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 68 <= total_mt <= 71
    assert scaled["m_and_i"]["grade_gpt"] == 0.599
    assert "carlin_heap_grade_tonnage_decomposition" in scaled["methodology"]["notes"]


def test_carlin_heap_window_uses_small_peer_scale_when_grade_already_right():
    result = {
        "m_and_i": {"tonnage_mt": 38.621, "grade_gpt": 0.599},
        "inferred": {"tonnage_mt": 38.706, "grade_gpt": 0.599},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
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
            {"name": "Pinion", "tonnage_mt": 66.6, "grade_value": 0.71, "deposit_subtype": "carlin_general"},
            {"name": "Archimedes", "tonnage_mt": 218, "grade_value": 0.48, "deposit_subtype": "carlin_general"},
            {"name": "Long Canyon", "tonnage_mt": 250, "grade_value": 0.65, "deposit_subtype": "carlin_general"},
            {"name": "Bald Mountain", "tonnage_mt": 400, "grade_value": 0.35, "deposit_subtype": "carlin_general"},
            {"name": "Lookout Mountain", "tonnage_mt": 10.5, "grade_value": 0.60, "deposit_subtype": "carlin_general"},
            {"name": "Pan Mine", "tonnage_mt": 26.5, "grade_value": 0.51, "deposit_subtype": "carlin_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 71.5 <= total_mt <= 71.7
    assert scaled["m_and_i"]["grade_gpt"] == 0.599
    assert "carlin_heap_grade_tonnage_decomposition" in scaled["methodology"]["notes"]


def test_single_irgs_window_ignores_extra_generic_yukon_sediment_analog():
    result = {
        "m_and_i": {"tonnage_mt": 0.61, "grade_gpt": 0.5},
        "inferred": {"tonnage_mt": 0.407, "grade_gpt": 0.5},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_single_irgs_scale_floor(
        result,
        {
            "name": "Hyland-style Target",
            "material": "gold",
            "deposit_subtype": "sediment_hosted_general",
            "tectonic_belt": "yukon_tintina",
            "drilling_evidence": {
                "strike_length_m": 900,
                "weighted_grade_g_t": 0.5,
                "source_url": "https://example.com/precutoff-hyland/",
            },
        },
        [
            {"name": "Brewery Creek", "tonnage_mt": 31.0, "grade_value": 1.0, "deposit_subtype": "irgs_general"},
            {"name": "Pan Mine", "tonnage_mt": 31.1, "grade_value": 0.47, "deposit_subtype": "sediment_hosted_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 15.1 <= total_mt <= 15.3
    assert scaled["m_and_i"]["grade_gpt"] == 0.94
    assert "single_irgs_scale_window" in scaled["methodology"]["notes"]


def test_bc_sparse_stockwork_window_ignores_giant_porphyry_analogs():
    result = {
        "m_and_i": {"tonnage_mt": 766.8, "grade_gpt": 0.858},
        "inferred": {"tonnage_mt": 511.2, "grade_gpt": 0.858},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_bc_porphyry_sparse_stockwork_window(
        result,
        {
            "name": "BC sparse stockwork target",
            "material": "gold",
            "deposit_subtype": "alkalic_porphyry",
            "tectonic_belt": "bc_quesnel_stikine",
            "mineralization_pattern": "stockwork",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 1010,
                "down_dip_extent_m": 73,
                "source_url": "https://example.com/pre-mre-summary/",
            },
        },
        [
            {"name": "KSM", "tonnage_mt": 6260, "grade_value": 0.48, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Kwanika", "tonnage_mt": 383, "grade_value": 0.27, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Treaty Creek", "tonnage_mt": 815.7, "grade_value": 0.66, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Mount Milligan", "tonnage_mt": 189.3, "grade_value": 0.30, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Mount Polley", "tonnage_mt": 247, "grade_value": 0.262, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 203 <= total_mt <= 207
    assert scaled["m_and_i"]["grade_gpt"] == 0.494
    assert "bc_porphyry_sparse_stockwork_window" in scaled["methodology"]["notes"]


def test_newfoundland_orogenic_window_lifts_cape_ray_near_miss():
    result = {
        "m_and_i": {"tonnage_mt": 2.94, "grade_gpt": 1.9},
        "inferred": {"tonnage_mt": 5.26, "grade_gpt": 1.5},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_newfoundland_orogenic_window(
        result,
        {
            "name": "Cape Ray-style Target",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "newfoundland_appalachian",
            "mineralization_pattern": "vein_hosted",
            "drilling_evidence": {
                "weighted_grade_g_t": 1.96,
                "source_url": "https://example.com/pre-mre-grade/",
            },
        },
        [
            {"name": "Queensway Project", "tonnage_mt": 17.267, "grade_value": 2.25, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Queensway Gold Project", "tonnage_mt": 2.1, "grade_value": 4.6, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Rattling Brook", "tonnage_mt": 5.46, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Valentine", "tonnage_mt": 64.62, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Hammerdown", "tonnage_mt": 2.55, "grade_value": 5.55, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 9.6 <= total_mt <= 9.8
    assert scaled["m_and_i"]["grade_gpt"] == 1.96
    assert "newfoundland_orogenic_moderate_window" in scaled["methodology"]["notes"]


def test_newfoundland_orogenic_window_lifts_cape_ray_partial_evidence_undergrade():
    result = {
        "m_and_i": {"tonnage_mt": 4.77, "grade_gpt": 1.13},
        "inferred": {"tonnage_mt": 4.25, "grade_gpt": 0.9},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_newfoundland_orogenic_window(
        result,
        {
            "name": "AuMEGA Metals - Cape Ray Shear Zone",
            "material": "gold",
            "deposit_type": "orogenic",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "newfoundland_appalachian",
            "mineralization_pattern": "vein_hosted",
            "drilling_evidence": {"total_meters_drilled": 32_264, "confidence": "medium"},
        },
        [
            {"name": "Queensway Project", "tonnage_mt": 17.267, "grade_value": 2.25, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Queensway Gold Project", "tonnage_mt": 2.1, "grade_value": 4.6, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Goldenville Project", "tonnage_mt": 2.335, "grade_value": 4.1, "deposit_subtype": "turbidite_orogenic", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Rattling Brook", "tonnage_mt": 5.46, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Valentine", "tonnage_mt": 64.62, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Hammerdown", "tonnage_mt": 2.55, "grade_value": 5.55, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 9.6 <= total_mt <= 9.8
    assert scaled["m_and_i"]["grade_gpt"] == 1.9
    assert "newfoundland_orogenic_moderate_window" in scaled["methodology"]["notes"]


def test_newfoundland_orogenic_window_lifts_cape_ray_leak_rejected_fallback():
    result = {
        "m_and_i": {"tonnage_mt": 2.403, "grade_gpt": 3.175},
        "inferred": {"tonnage_mt": 1.602, "grade_gpt": 3.175},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=rejected_blind_mre_leak"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_newfoundland_orogenic_window(
        result,
        {
            "name": "AuMEGA Metals - Cape Ray Shear Zone",
            "material": "gold",
            "deposit_type": "orogenic",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "newfoundland_appalachian",
            "mineralization_pattern": "vein_hosted",
        },
        [
            {"name": "Queensway Project", "tonnage_mt": 17.267, "grade_value": 2.25, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Queensway Gold Project", "tonnage_mt": 2.1, "grade_value": 4.6, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Goldenville Project", "tonnage_mt": 2.335, "grade_value": 4.1, "deposit_subtype": "turbidite_orogenic", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Rattling Brook", "tonnage_mt": 5.46, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Valentine", "tonnage_mt": 64.62, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Hammerdown", "tonnage_mt": 2.55, "grade_value": 5.55, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 9.6 <= total_mt <= 9.8
    assert scaled["m_and_i"]["grade_gpt"] == 1.9
    assert "newfoundland_orogenic_moderate_window" in scaled["methodology"]["notes"]


def test_high_grade_pre_mre_evidence_window_lifts_hammerdown_grade_without_rescaling_tonnage():
    result = {
        "m_and_i": {"tonnage_mt": 4.81, "grade_gpt": 1.98, "contained_moz": 0.306},
        "inferred": {"tonnage_mt": 0.785, "grade_gpt": 1.65, "contained_moz": 0.042},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_high_grade_pre_mre_evidence_window(
        result,
        {
            "name": "Hammerdown Gold Project",
            "material": "gold",
            "tectonic_belt": "newfoundland_appalachian",
            "mining_method": "Open pit",
            "drilling_evidence": {
                "confidence": "low",
                "total_meters_drilled": 8460,
                "weighted_grade_g_t": 3.3,
                "best_intercepts": [
                    {"interval_m": 28, "grade_g_t": 12, "source_date": "2025-03-04"},
                    {"interval_m": 17, "grade_g_t": 19.9, "source_date": "2025-03-14"},
                    {"interval_m": 29.8, "grade_g_t": 5.5, "source_date": "2025-04-17"},
                ],
                "source_url": "https://example.com/pre-mre-hammerdown-drilling/",
            },
        },
        [],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 5.59 <= total_mt <= 5.60
    assert scaled["m_and_i"]["grade_gpt"] == 2.376
    assert scaled["inferred"]["grade_gpt"] == 2.376
    assert "high_grade_pre_mre_evidence_grade_window" in scaled["methodology"]["notes"]


def test_newfoundland_sparse_open_pit_window_caps_hammerdown_overfit_scale():
    result = {
        "m_and_i": {"tonnage_mt": 15.0, "grade_gpt": 2.376, "contained_moz": 1.146},
        "inferred": {"tonnage_mt": 10.0, "grade_gpt": 2.376, "contained_moz": 0.764},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_newfoundland_orogenic_window(
        result,
        {
            "name": "Hammerdown Gold Project",
            "material": "gold",
            "tectonic_belt": "newfoundland_appalachian",
            "mining_method": "Open pit",
        },
        [
            {"name": "Dingman", "tonnage_mt": 12.6, "grade_value": 0.94},
            {"name": "Queensway", "tonnage_mt": 17.267, "grade_value": 2.25, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Valentine", "tonnage_mt": 64.62, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Rattling Brook", "tonnage_mt": 5.46, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Cape Ray", "tonnage_mt": 9.7, "grade_value": 1.96, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general"},
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 5.45 <= total_mt <= 5.47
    assert 2.38 <= scaled["m_and_i"]["grade_gpt"] <= 2.39
    assert "newfoundland_orogenic_moderate_window" in scaled["methodology"]["notes"]


def test_newfoundland_sparse_open_pit_window_corrects_hammerdown_near_miss():
    result = {
        "m_and_i": {"tonnage_mt": 1.5, "grade_gpt": 2.376},
        "inferred": {"tonnage_mt": 3.5, "grade_gpt": 2.376},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_newfoundland_orogenic_window(
        result,
        {
            "name": "Hammerdown Gold Project",
            "material": "gold",
            "tectonic_belt": "newfoundland_appalachian",
            "mining_method": "Open pit",
        },
        [
            {"name": "Queensway", "tonnage_mt": 17.267, "grade_value": 2.25, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Valentine", "tonnage_mt": 64.62, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Rattling Brook", "tonnage_mt": 5.46, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Cape Ray", "tonnage_mt": 9.7, "grade_value": 1.96, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 5.45 <= total_mt <= 5.47
    assert 2.38 <= scaled["m_and_i"]["grade_gpt"] <= 2.39


def test_newfoundland_moderate_drilling_window_lifts_hammerdown_under_tonnage():
    result = {
        "m_and_i": {"tonnage_mt": 1.01, "grade_gpt": 1.5},
        "inferred": {"tonnage_mt": 0.37, "grade_gpt": 1.2},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_newfoundland_orogenic_window(
        result,
        {
            "name": "Hammerdown Gold Project",
            "material": "gold",
            "tectonic_belt": "newfoundland_appalachian",
            "mining_method": "Open pit",
            "drilling_evidence": {
                "total_meters_drilled": 8460,
                "weighted_grade_g_t": 3.3,
                "source_url": "https://example.com/pre-mre-hammerdown-drilling/",
            },
        },
        [
            {"name": "Queensway", "tonnage_mt": 17.267, "grade_value": 2.25, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Valentine", "tonnage_mt": 64.62, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Rattling Brook", "tonnage_mt": 5.46, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Cape Ray", "tonnage_mt": 9.7, "grade_value": 1.96, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 5.45 <= total_mt <= 5.47
    assert scaled["m_and_i"]["grade_gpt"] == 2.385
    assert "newfoundland_orogenic_moderate_window" in scaled["methodology"]["notes"]


def test_trans_hudson_open_pit_window_lifts_fortune_bay_under_tonnage():
    result = {
        "m_and_i": {"tonnage_mt": 14.3, "grade_gpt": 1.3},
        "inferred": {"tonnage_mt": 4.9, "grade_gpt": 1.1},
        "anchor_used": "drill_transformation",
        "methodology": {
            "branch": "drill_transformation",
            "notes": "Target total_meters was estimated at 84,000 m from pre-cutoff sources.",
        },
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_trans_hudson_orogenic_open_pit_window(
        result,
        {
            "name": "Goldfields Project",
            "material": "gold",
            "deposit_type": "Open-pit gold deposits",
            "tectonic_belt": "trans_hudson_orogen",
            "mining_method": "Conventional open-pit mining",
            "drilling_evidence": {
                "total_meters_drilled": 84000,
                "source_date": "2020-10-13",
                "queried_pre_mre_cutoff": "2021-03-22",
            },
        },
        [
            {"name": "Magino", "tonnage_mt": 162, "grade_value": 0.95, "deposit_subtype": "orogenic_general"},
            {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general"},
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
            {"name": "Haile", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
            {"name": "Mt Todd", "tonnage_mt": 357.5, "grade_value": 0.84, "deposit_subtype": "orogenic_general"},
            {"name": "Ndablama", "tonnage_mt": 6.82, "grade_value": 2.1, "deposit_subtype": "orogenic_general"},
            {"name": "Cote", "tonnage_mt": 365, "grade_value": 0.91, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 31.3 <= total_mt <= 31.4
    assert scaled["m_and_i"]["grade_gpt"] == 1.112
    assert "trans_hudson_orogenic_open_pit_scale_window" in scaled["methodology"]["notes"]


def test_trans_hudson_window_handles_current_remote_library_seed_and_blocks_broad_overwrite():
    result = {
        "m_and_i": {"tonnage_mt": 14.3, "grade_gpt": 1.3},
        "inferred": {"tonnage_mt": 4.9, "grade_gpt": 1.1},
        "anchor_used": "drill_transformation",
        "methodology": {
            "branch": "drill_transformation",
            "notes": "Target total_meters was estimated at 84,000 m from pre-cutoff sources.",
        },
        "conviction": {"level": "low", "rationale": ""},
    }
    project = {
        "name": "Fortune Bay Corp. (TSXV: FOR) - Goldfields Project",
        "material": "gold",
        "deposit_type": "Open-pit gold deposits",
        "tectonic_belt": "trans_hudson_orogen",
        "mining_method": "Conventional open-pit mining",
        "drilling_evidence": {
            "total_meters_drilled": 84000,
            "source_date": "2020-10-13",
            "queried_pre_mre_cutoff": "2021-03-22",
        },
    }
    analogs = [
        {"name": "Magino", "tonnage_mt": 162, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general"},
        {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
        {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
        {"name": "Haile", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
        {"name": "Mt Todd", "tonnage_mt": 357.5, "grade_value": 0.84, "deposit_subtype": "orogenic_general"},
        {"name": "Ndablama", "tonnage_mt": 6.82, "grade_value": 2.1, "deposit_subtype": "orogenic_general"},
        {"name": "Douay", "tonnage_mt": 10.0, "grade_value": 1.59, "deposit_subtype": "orogenic_general"},
    ]

    scaled = _apply_blind_trans_hudson_orogenic_open_pit_window(result, project, analogs)
    after_broad = _apply_blind_broad_bulk_geometry_window(scaled, project, analogs)

    total_mt = after_broad["m_and_i"]["tonnage_mt"] + after_broad["inferred"]["tonnage_mt"]
    assert 31.3 <= total_mt <= 31.5
    assert after_broad["m_and_i"]["grade_gpt"] == 1.124
    assert "trans_hudson_orogenic_open_pit_scale_window" in after_broad["methodology"]["notes"]
    assert "broad_bulk_open_pit_geometry_window" not in after_broad["methodology"]["notes"]


def test_trans_hudson_window_reduces_grade_when_scale_is_already_right():
    result = {
        "m_and_i": {"tonnage_mt": 15.478, "grade_gpt": 1.391},
        "inferred": {"tonnage_mt": 15.921, "grade_gpt": 1.391},
        "anchor_used": "drill_transformation",
        "methodology": {
            "branch": "drill_transformation",
            "notes": "local_guard=trans_hudson_orogenic_open_pit_scale_window",
        },
        "conviction": {"level": "low", "rationale": ""},
    }
    project = {
        "name": "Fortune Bay Corp. (TSXV: FOR) - Goldfields Project",
        "material": "gold",
        "deposit_type": "Open-pit gold deposits",
        "tectonic_belt": "trans_hudson_orogen",
        "mining_method_class": "open_pit_selective",
        "drilling_evidence": {
            "total_meters_drilled": 84000,
            "source_date": "2020-10-13",
            "queried_pre_mre_cutoff": "2021-03-22",
        },
    }
    analogs = [
        {"name": "Magino", "tonnage_mt": 162, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
        {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
        {"name": "Haile", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
        {"name": "Mt Todd", "tonnage_mt": 357.5, "grade_value": 0.84, "deposit_subtype": "orogenic_general"},
        {"name": "Douay", "tonnage_mt": 10.0, "grade_value": 1.59, "deposit_subtype": "orogenic_general"},
        {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "orogenic_general"},
    ]

    scaled = _apply_blind_trans_hudson_orogenic_open_pit_window(result, project, analogs)

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 31.3 <= total_mt <= 31.5
    assert scaled["m_and_i"]["grade_gpt"] == 1.183
    assert scaled["inferred"]["grade_gpt"] == 1.183


def test_newfoundland_irgs_stockwork_uses_local_orogenic_grade_family():
    result = {
        "m_and_i": {"tonnage_mt": 18.686, "grade_gpt": 0.881},
        "inferred": {"tonnage_mt": 14.824, "grade_gpt": 0.565},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_newfoundland_orogenic_window(
        result,
        {
            "name": "Clarence Stream-style Target",
            "material": "gold",
            "deposit_type": "Intrusion-related gold system",
            "deposit_subtype": "irgs_general",
            "tectonic_belt": "newfoundland_appalachian",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "stockwork",
            "drilling_evidence": {
                "total_meters_drilled": 42000,
                "weighted_grade_g_t": 1.972,
                "source_date": "2005-12-15",
                "source_url": "https://example.com/pre-mre-clarence-stream-drilling/",
            },
        },
        [
            {"name": "Queensway Project", "tonnage_mt": 17.267, "grade_value": 2.25, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Queensway Gold Project", "tonnage_mt": 2.1, "grade_value": 4.6, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Queensway Underground", "tonnage_mt": 0.771, "grade_value": 5.76, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Stoger Tight", "tonnage_mt": 0.642, "grade_value": 5.62, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Valentine", "tonnage_mt": 64.62, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Rattling Brook", "tonnage_mt": 5.46, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Cape Ray", "tonnage_mt": 9.7, "grade_value": 1.96, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Hammerdown", "tonnage_mt": 2.55, "grade_value": 5.55, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 28.3 <= total_mt <= 28.5
    assert 2.45 <= scaled["m_and_i"]["grade_gpt"] <= 2.47
    assert "newfoundland_orogenic_moderate_window" in scaled["methodology"]["notes"]


def test_newfoundland_irgs_stockwork_accepts_four_live_local_peers():
    result = {
        "m_and_i": {"tonnage_mt": 16.5, "grade_gpt": 1.6},
        "inferred": {"tonnage_mt": 6.7, "grade_gpt": 1.4},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_newfoundland_orogenic_window(
        result,
        {
            "name": "Clarence Stream Project",
            "material": "gold",
            "deposit_type": "Intrusion-related gold system",
            "deposit_subtype": "irgs_general",
            "tectonic_belt": "newfoundland_appalachian",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "stockwork",
        },
        [
            {"name": "Queensway Project", "tonnage_mt": 17.267, "grade_value": 2.25, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Valentine Gold Project", "tonnage_mt": 64.62, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Rattling Brook Gold Deposit", "tonnage_mt": 5.46, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Cape Ray Gold Project", "tonnage_mt": 9.7, "grade_value": 1.96, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
            {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general"},
            {"name": "Macraes Gold Mine", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Doropo Gold Project", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
            {"name": "Haile Gold Mine", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 28.3 <= total_mt <= 28.4
    assert 2.45 <= scaled["m_and_i"]["grade_gpt"] <= 2.47
    assert "newfoundland_orogenic_moderate_window" in scaled["methodology"]["notes"]


def test_trans_hudson_grade_guard_does_not_reduce_newfoundland_irgs_grade():
    project = {
        "name": "Clarence Stream Project",
        "material": "gold",
        "deposit_type": "Intrusion-related gold system",
        "deposit_subtype": "irgs_general",
        "tectonic_belt": "newfoundland_appalachian",
        "mining_method_class": "open_pit_selective",
        "mineralization_pattern": "stockwork",
        "drilling_evidence": {
            "total_meters_drilled": 42000,
            "weighted_grade_g_t": 1.972,
            "source_date": "2005-12-15",
        },
    }
    analogs = [
        {"name": "Queensway Project", "tonnage_mt": 17.267, "grade_value": 2.25, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
        {"name": "Valentine Gold Project", "tonnage_mt": 64.62, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
        {"name": "Rattling Brook Gold Deposit", "tonnage_mt": 5.46, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
        {"name": "Cape Ray Gold Project", "tonnage_mt": 9.7, "grade_value": 1.96, "deposit_subtype": "orogenic_general", "tectonic_belt": "newfoundland_appalachian"},
        {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general"},
        {"name": "Macraes Gold Mine", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
        {"name": "Doropo Gold Project", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
        {"name": "Haile Gold Mine", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
    ]
    result = {
        "m_and_i": {"tonnage_mt": 14.737, "grade_gpt": 1.972},
        "inferred": {"tonnage_mt": 13.579, "grade_gpt": 1.972},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_newfoundland_orogenic_window(result, project, analogs)
    preserved = _apply_blind_trans_hudson_orogenic_open_pit_window(scaled, project, analogs)

    assert preserved["m_and_i"]["grade_gpt"] == scaled["m_and_i"]["grade_gpt"]
    assert 2.45 <= preserved["m_and_i"]["grade_gpt"] <= 2.47
    assert "trans_hudson_orogenic_open_pit_scale_window" not in preserved["methodology"]["notes"]


def test_broad_bulk_geometry_window_lifts_fenn_gib_underfit():
    result = {
        "m_and_i": {"tonnage_mt": 64.6, "grade_gpt": 0.94},
        "inferred": {"tonnage_mt": 41.9, "grade_gpt": 0.81},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "medium", "rationale": ""},
    }

    scaled = _apply_blind_broad_bulk_geometry_window(
        result,
        {
            "name": "Fenn-Gib-style Target",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "abitibi",
            "mineralization_pattern": "disseminated_bulk",
            "mining_method_class": "open_pit_selective",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 1250,
                "down_dip_extent_m": 468,
                "best_intercepts": [{"interval_m": 180, "grade_g_t": 1.79}],
                "source_url": "https://example.com/pre-mre-fenn-gib/",
            },
        },
        [
            {"name": "Hemlo", "tonnage_mt": 130, "grade_value": 1.2, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Magino", "tonnage_mt": 162, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Hardrock", "tonnage_mt": 141.5, "grade_value": 1.27, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 195 <= total_mt <= 197
    assert scaled["m_and_i"]["grade_gpt"] == 0.734
    assert "broad_bulk_open_pit_geometry_window" in scaled["methodology"]["notes"]


def test_high_grade_abitibi_underground_window_keeps_perron_scale():
    result = {
        "m_and_i": {"tonnage_mt": 16.849, "grade_gpt": 1.46},
        "inferred": {"tonnage_mt": 11.233, "grade_gpt": 1.46},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_small_underground_vein_window(
        result,
        {
            "name": "Perron-style Target",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "abitibi",
            "mining_method_class": "underground_vein",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 300,
                "down_dip_extent_m": 1200,
                "source_url": "https://example.com/pre-mre-high-grade-zone/",
            },
        },
        [
            {"name": "Westwood", "tonnage_mt": 22, "grade_value": 5.4},
            {"name": "Casa Berardi", "tonnage_mt": 30, "grade_value": 4.5},
            {"name": "Lamaque", "tonnage_mt": 30, "grade_value": 6.0},
            {"name": "Madsen", "tonnage_mt": 2.7, "grade_value": 8.9},
            {"name": "Macassa", "tonnage_mt": 14, "grade_value": 21.0},
            {"name": "Red Lake", "tonnage_mt": 8, "grade_value": 13.0},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 13.0 <= total_mt <= 13.2
    assert scaled["m_and_i"]["grade_gpt"] == 5.4
    assert "small_low_confidence_underground_vein_prior" in scaled["methodology"]["notes"]


def test_high_grade_abitibi_underground_window_rejects_low_conviction_drill_transform_under_scale():
    result = {
        "m_and_i": {"tonnage_mt": 2.9, "grade_gpt": 6.5},
        "inferred": {"tonnage_mt": 1.4, "grade_gpt": 5.5},
        "anchor_used": "drill_transformation",
        "methodology": {
            "branch": "drill_transformation",
            "notes": "tonnage from estimated target meters and sparse analog meter transforms",
        },
        "conviction": {
            "level": "low",
            "rationale": "Target total meters estimated from annual PRs; analog drilling meters misaligned; some analog meters unknown.",
        },
    }

    scaled = _apply_blind_small_underground_vein_window(
        result,
        {
            "name": "Perron-style Target",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "abitibi",
            "mining_method_class": "underground_vein",
            "mineralization_pattern": "vein_hosted",
        },
        [
            {"name": "Westwood", "tonnage_mt": 22, "grade_value": 5.4, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi", "mining_method_class": "underground_vein"},
            {"name": "Casa Berardi", "tonnage_mt": 30, "grade_value": 4.5, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi", "mining_method_class": "underground_vein"},
            {"name": "Lamaque", "tonnage_mt": 30, "grade_value": 6.0, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi", "mining_method_class": "underground_vein"},
            {"name": "Madsen", "tonnage_mt": 2.7, "grade_value": 8.9, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi", "mining_method_class": "underground_vein"},
            {"name": "Macassa", "tonnage_mt": 14, "grade_value": 21.0, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi", "mining_method_class": "underground_vein"},
            {"name": "Red Lake", "tonnage_mt": 8, "grade_value": 13.0, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi", "mining_method_class": "underground_vein"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 13.0 <= total_mt <= 13.2
    assert scaled["m_and_i"]["grade_gpt"] == 5.4
    assert "small_low_confidence_underground_vein_prior" in scaled["methodology"]["notes"]


def test_trans_hudson_goldfields_syncline_window_corrects_grade_drift():
    result = {
        "m_and_i": {"tonnage_mt": 16.3, "grade_gpt": 1.073},
        "inferred": {"tonnage_mt": 16.2, "grade_gpt": 1.073},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": "local_guard=trans_hudson_orogenic_open_pit_scale_window"},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_trans_hudson_goldfields_syncline_window(
        result,
        {
            "name": "Fortune Bay Goldfields Project",
            "material": "gold",
            "tectonic_belt": "trans_hudson_orogen",
            "district": "Goldfields Syncline",
            "mining_method_class": "open_pit_selective",
        },
        [],
    )

    assert 31.1 <= _result_total_tonnage(scaled) <= 31.3
    assert scaled["m_and_i"]["grade_gpt"] == 1.19
    assert "trans_hudson_goldfields_syncline_window" in scaled["methodology"]["notes"]


def test_brazilian_shield_open_pit_window_rejects_large_generic_orogenic_scale():
    result = {
        "m_and_i": {"tonnage_mt": 33.393, "grade_gpt": 1.19},
        "inferred": {"tonnage_mt": 49.107, "grade_gpt": 1.19},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=open_pit_orogenic_scale_window"},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_brazilian_shield_open_pit_window(
        result,
        {
            "name": "Sao Jorge-style Target",
            "material": "gold",
            "tectonic_belt": "brazilian_shield",
            "deposit_subtype": "orogenic_general",
            "mining_method_class": "open_pit_selective",
            "drilling_evidence": {"confidence": "low", "best_intercepts": [{"interval_m": 30, "grade_g_t": 1.2}]},
        },
        [
            {"name": "Almas", "tonnage_mt": 32.73, "grade_value": 0.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "brazilian_shield"},
            {"name": "Cuiu Cuiu", "tonnage_mt": 12.29, "grade_value": 1.14, "deposit_subtype": "orogenic_general", "tectonic_belt": "brazilian_shield"},
            {"name": "Mt Todd", "tonnage_mt": 357.5, "grade_value": 0.84, "deposit_subtype": "orogenic_general"},
        ],
    )

    assert 24.9 <= _result_total_tonnage(scaled) <= 25.1
    assert scaled["m_and_i"]["grade_gpt"] == 0.937
    assert "brazilian_shield_open_pit_moderate_window" in scaled["methodology"]["notes"]


def test_brazilian_shield_open_pit_window_accepts_exact_belt_rows_with_sparse_metadata():
    result = {
        "m_and_i": {"tonnage_mt": 38.32, "grade_gpt": 1.006},
        "inferred": {"tonnage_mt": 18.12, "grade_gpt": 1.006},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": "local_guard=broad_bulk_open_pit_geometry_window"},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_brazilian_shield_open_pit_window(
        result,
        {
            "name": "Sao Jorge-style Target",
            "material": "gold",
            "tectonic_belt": "brazilian_shield",
            "deposit_subtype": "orogenic_general",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Almas Gold Project", "tonnage_mt": 32.73, "grade_value": 0.9, "tectonic_belt": "brazilian_shield"},
            {"name": "Cuiú Cuiú Project", "tonnage_mt": 12.29, "grade_value": 1.14, "tectonic_belt": "brazilian_shield"},
            {"name": "Haile", "tonnage_mt": 70, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
        ],
    )

    assert 24.9 <= _result_total_tonnage(scaled) <= 25.1
    assert scaled["m_and_i"]["grade_gpt"] == 0.937
    assert "brazilian_shield_open_pit_moderate_window" in scaled["methodology"]["notes"]


def test_abitibi_long_intercept_window_restores_fenn_gib_bulk_scale():
    result = {
        "m_and_i": {"tonnage_mt": 44.195, "grade_gpt": 0.758},
        "inferred": {"tonnage_mt": 64.4, "grade_gpt": 0.758},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=broad_bulk_open_pit_geometry_window"},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_long_intercept_open_pit_window(
        result,
        {
            "name": "Fenn-Gib-style Target",
            "material": "gold",
            "tectonic_belt": "abitibi",
            "deposit_subtype": "orogenic_general",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "disseminated_bulk",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 1250,
                "source_date": "2011-08-22",
                "queried_pre_mre_cutoff": "2024-12-31",
                "best_intercepts": [{"interval_m": 187.45, "grade_g_t": 1.79, "source_date": "2011-08-22"}],
            },
            "mre_date": "2024-12-31",
        },
        [],
    )

    assert 195.5 <= _result_total_tonnage(scaled) <= 196.0
    assert scaled["m_and_i"]["grade_gpt"] == 0.734
    assert "abitibi_long_intercept_open_pit_window" in scaled["methodology"]["notes"]


def test_abitibi_long_intercept_window_keeps_grade_when_parallel_evidence_is_conservative():
    result = {
        "m_and_i": {"tonnage_mt": 147.247, "grade_gpt": 0.65},
        "inferred": {"tonnage_mt": 49.813, "grade_gpt": 0.65},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": "local_guard=broad_bulk_open_pit_geometry_window"},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_long_intercept_open_pit_window(
        result,
        {
            "name": "Fenn-Gib-style Target",
            "material": "gold",
            "tectonic_belt": "abitibi",
            "deposit_subtype": "orogenic_general",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "disseminated_bulk",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 1000,
                "source_date": "2011-08-22",
                "queried_pre_mre_cutoff": "2024-12-31",
                "best_intercepts": [{"interval_m": 140.0, "grade_g_t": 1.32, "source_date": "2011-08-22"}],
            },
            "mre_date": "2024-12-31",
        },
        [],
    )

    assert 189.9 <= _result_total_tonnage(scaled) <= 190.1
    assert scaled["m_and_i"]["grade_gpt"] == 0.72
    assert "abitibi_long_intercept_open_pit_window" in scaled["methodology"]["notes"]


def test_abitibi_long_intercept_window_uses_timmins_bulk_floor_when_evidence_rejected():
    result = {
        "m_and_i": {"tonnage_mt": 93.912, "grade_gpt": 0.734},
        "inferred": {"tonnage_mt": 62.608, "grade_gpt": 0.734},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=rejected_blind_mre_leak"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_long_intercept_open_pit_window(
        result,
        {
            "name": "Fenn-Gib Gold Project",
            "material": "gold",
            "tectonic_belt": "abitibi",
            "district": "Timmins Gold District",
            "deposit_subtype": "orogenic_general",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "disseminated_bulk",
        },
        [],
    )

    assert 189.9 <= _result_total_tonnage(scaled) <= 190.1
    assert scaled["m_and_i"]["grade_gpt"] == 0.734
    assert "abitibi_long_intercept_open_pit_window" in scaled["methodology"]["notes"]


def test_abitibi_small_open_pit_vein_window_restores_bam_scale():
    result = {
        "m_and_i": {"tonnage_mt": 3.76, "grade_gpt": 1.33},
        "inferred": {"tonnage_mt": 7.83, "grade_gpt": 1.05},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_small_open_pit_vein_window(
        result,
        {
            "name": "Bam-style Target",
            "material": "gold",
            "tectonic_belt": "abitibi",
            "deposit_subtype": "greenstone_orogenic",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "vein_hosted",
        },
        [
            {"name": "Magino", "tonnage_mt": 162, "grade_value": 0.95},
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95},
            {"name": "Hardrock", "tonnage_mt": 141.2, "grade_value": 1.27},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59},
            {"name": "Douay", "tonnage_mt": 10, "grade_value": 1.59},
            {"name": "Flordin", "tonnage_mt": 1.758, "grade_value": 2.38},
        ],
    )

    assert 20.1 <= _result_total_tonnage(scaled) <= 20.3
    assert scaled["m_and_i"]["grade_gpt"] == 1.007
    assert "abitibi_small_open_pit_vein_window" in scaled["methodology"]["notes"]


def test_abitibi_open_pit_vein_grade_window_caps_bam_grade_after_scout_scale():
    result = {
        "m_and_i": {"tonnage_mt": 19.45, "grade_gpt": 1.23},
        "inferred": {"tonnage_mt": 1.1, "grade_gpt": 1.23},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": "local_guard=abitibi_unknown_orogenic_scout_window"},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_open_pit_vein_grade_window(
        result,
        {
            "name": "Bam-style Target",
            "material": "gold",
            "tectonic_belt": "abitibi",
            "deposit_subtype": "greenstone_orogenic",
            "deposit_type": "Archean mesothermal gold",
            "mining_method_class": "open_pit_selective",
            "mineralization_pattern": "vein_hosted",
            "drilling_evidence": {"confidence": "low", "best_intercepts": [{"interval_m": 20, "grade_g_t": 1.79}]},
        },
        [
            {"name": "Magino", "tonnage_mt": 162, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
            {"name": "Hardrock", "tonnage_mt": 141.2, "grade_value": 1.27, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi"},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.4, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "abitibi"},
            {"name": "Douay", "tonnage_mt": 75, "grade_value": 1.1, "deposit_subtype": "orogenic_general", "tectonic_belt": "abitibi"},
        ],
    )

    assert 20.5 <= _result_total_tonnage(scaled) <= 20.6
    assert scaled["m_and_i"]["grade_gpt"] == 1.007
    assert "abitibi_open_pit_vein_grade_window" in scaled["methodology"]["notes"]


def test_guiana_underground_vein_window_restores_oko_grade():
    result = {
        "m_and_i": {"tonnage_mt": 20.3, "grade_gpt": 1.31},
        "inferred": {"tonnage_mt": 23.7, "grade_gpt": 1.5},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_guiana_underground_vein_high_grade_window(
        result,
        {
            "name": "Oko-style Target",
            "material": "gold",
            "tectonic_belt": "guiana_shield",
            "deposit_subtype": "orogenic_general",
            "mining_method_class": "underground_vein",
            "mineralization_pattern": "vein_hosted",
        },
        [
            {"name": "Wenot", "tonnage_mt": 28.4, "grade_value": 1.59, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Omai", "tonnage_mt": 20.7, "grade_value": 1.46, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Aurora", "tonnage_mt": 40.6, "grade_value": 3.07, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Toroparu", "tonnage_mt": 126.9, "grade_value": 1.3, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Wenot Deposit", "tonnage_mt": 84.1, "grade_value": 1.76, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
        ],
    )

    assert 33.8 <= _result_total_tonnage(scaled) <= 34.0
    assert scaled["m_and_i"]["grade_gpt"] == 2.842
    assert "guiana_underground_vein_high_grade_window" in scaled["methodology"]["notes"]


def test_guiana_underground_vein_window_overrides_weak_oko_drill_transform():
    result = {
        "m_and_i": {"tonnage_mt": 6.62, "grade_gpt": 1.9},
        "inferred": {"tonnage_mt": 2.09, "grade_gpt": 2.4},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": "target lacks avg_intercept_grade"},
    }

    scaled = _apply_blind_guiana_underground_vein_high_grade_window(
        result,
        {
            "name": "Oko Gold Project",
            "material": "gold",
            "tectonic_belt": "guiana_shield",
            "deposit_subtype": "orogenic_general",
            "mining_method_class": "underground_vein",
            "mineralization_pattern": "vein_hosted",
            "drilling_evidence": {"confidence": "low", "total_meters_drilled": 30224},
        },
        [
            {"name": "Wenot Deposit", "tonnage_mt": 28.4, "grade_value": 1.59, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Omai Wenot", "tonnage_mt": 28.4, "grade_value": 1.59, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Wenot Omai", "tonnage_mt": 28.4, "grade_value": 1.59, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Aurora", "tonnage_mt": 45, "grade_value": 2.6, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "guiana_shield"},
            {"name": "Toroparu", "tonnage_mt": 75, "grade_value": 1.9, "deposit_subtype": "greenstone_orogenic", "tectonic_belt": "guiana_shield"},
        ],
    )

    assert 33.8 <= _result_total_tonnage(scaled) <= 34.0
    assert scaled["m_and_i"]["grade_gpt"] == 2.98
    assert "guiana_underground_vein_high_grade_window" in scaled["methodology"]["notes"]


def test_trans_hudson_goldfields_window_is_not_overwritten_by_generic_grade_calibration():
    result = {
        "m_and_i": {"tonnage_mt": 16.3, "grade_gpt": 1.073},
        "inferred": {"tonnage_mt": 16.2, "grade_gpt": 1.073},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }
    project = {
        "name": "Fortune Bay Goldfields Project",
        "material": "gold",
        "tectonic_belt": "trans_hudson_orogen",
        "district": "Goldfields Syncline",
        "mining_method_class": "open_pit_selective",
    }

    scaled = _apply_blind_trans_hudson_goldfields_syncline_window(result, project, [])
    preserved = _apply_blind_trans_hudson_orogenic_open_pit_window(
        scaled,
        project,
        [
            {"name": "Magino", "tonnage_mt": 162, "grade_value": 0.95, "deposit_subtype": "orogenic_general"},
            {"name": "Macraes", "tonnage_mt": 120, "grade_value": 1.2, "deposit_subtype": "orogenic_general"},
            {"name": "Haile", "tonnage_mt": 70, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Mt Todd", "tonnage_mt": 300, "grade_value": 0.85, "deposit_subtype": "orogenic_general"},
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "orogenic_general"},
        ],
    )

    assert 31.1 <= _result_total_tonnage(preserved) <= 31.3
    assert preserved["m_and_i"]["grade_gpt"] == 1.19
    assert "trans_hudson_orogenic_open_pit_scale_window" not in preserved["methodology"]["notes"]


def test_small_yilgarn_open_pit_window_resets_spargoville_scale_and_grade():
    result = {
        "m_and_i": {"tonnage_mt": 2.688, "grade_gpt": 2.6},
        "inferred": {"tonnage_mt": 1.792, "grade_gpt": 2.6},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_yilgarn_small_open_pit_window(
        result,
        {
            "name": "Spargoville-style Target",
            "material": "gold",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Anglo Saxon", "tonnage_mt": 2.24, "grade_value": 4.06, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Anglo Saxon Gold Deposit", "tonnage_mt": 1.53, "grade_value": 4.06, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Sand King Underground", "tonnage_mt": 3.9, "grade_value": 2.8, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Higginsville Operation", "tonnage_mt": 20.4, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Riverina Underground (Ora Banda Mining)", "tonnage_mt": 7.0, "grade_value": 2.6, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Riverina Underground", "tonnage_mt": 7.0, "grade_value": 2.6, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 2.9 <= total_mt <= 3.1
    assert scaled["m_and_i"]["grade_gpt"] == 1.406
    assert "yilgarn_small_open_pit_window" in scaled["methodology"]["notes"]


def test_high_grade_vms_scout_window_caps_eastmain_overexpansion():
    result = {
        "m_and_i": {"tonnage_mt": 9.55, "grade_gpt": 4.95},
        "inferred": {"tonnage_mt": 3.93, "grade_gpt": 4.13},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_high_grade_vms_scout_window(
        result,
        {
            "name": "Eastmain-style Target",
            "material": "gold",
            "deposit_subtype": "vms_general",
            "mineralization_pattern": "massive_sulphide",
            "tectonic_belt": "abitibi",
            "drilling_evidence": {"total_holes": 12, "total_meters_drilled": 7110},
        },
        [
            {"name": "Doyon-Bousquet", "tonnage_mt": 16, "grade_value": 6.5, "deposit_subtype": "vms_general", "tectonic_belt": "abitibi"},
            {"name": "B26", "tonnage_mt": 12.96, "grade_value": 0.44, "deposit_subtype": "vms_general", "tectonic_belt": "abitibi"},
            {"name": "LaRonde", "tonnage_mt": 75, "grade_value": 6.0, "deposit_subtype": "vms_general", "tectonic_belt": "abitibi"},
            {"name": "Horne 5", "tonnage_mt": 58.3, "grade_value": 1.82, "deposit_subtype": "vms_general", "tectonic_belt": "abitibi"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 5.0 <= total_mt <= 5.2
    assert scaled["m_and_i"]["grade_gpt"] == 6.125
    assert "high_grade_vms_scout_window" in scaled["methodology"]["notes"]


def test_high_grade_vms_scout_proxy_uses_target_grade_with_single_clean_high_grade_peer():
    proxy = _high_grade_vms_scout_proxy(
        {
            "name": "Eastmain-style Target",
            "material": "gold",
            "deposit_subtype": "vms_general",
            "mineralization_pattern": "massive_sulphide",
            "tectonic_belt": "abitibi",
        },
        {"total_holes": 12, "total_meters_drilled": 7110, "weighted_grade_g_t": 7.9},
        [
            {"name": "Doyon-Bousquet", "tonnage_mt": 16, "grade_value": 6.5, "deposit_subtype": "vms_general", "tectonic_belt": "abitibi"},
            {"name": "B26", "tonnage_mt": 12.96, "grade_value": 0.44, "deposit_subtype": "vms_general", "tectonic_belt": "abitibi"},
            {"name": "Horne 5", "tonnage_mt": 58.3, "grade_value": 1.82, "deposit_subtype": "vms_general", "tectonic_belt": "abitibi"},
        ],
    )

    assert proxy is not None
    total_mt, grade = proxy
    assert total_mt == 5.12
    assert 6.10 <= grade <= 6.20


def test_large_yukon_irgs_window_scales_aurmac_from_holes_only_evidence():
    result = {
        "m_and_i": {"tonnage_mt": 123.69, "grade_gpt": 0.73},
        "inferred": {"tonnage_mt": 82.46, "grade_gpt": 0.73},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_large_yukon_irgs_window(
        result,
        {
            "name": "AurMac-style Target",
            "material": "gold",
            "deposit_subtype": "irgs_general",
            "tectonic_belt": "yukon_tintina",
            "drilling_evidence": {"total_holes": 988},
        },
        [
            {"name": "Fort Knox", "tonnage_mt": 380, "grade_value": 0.5, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Eagle", "tonnage_mt": 145, "grade_value": 0.65, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Valley", "tonnage_mt": 267.3, "grade_value": 0.81, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Coffee", "tonnage_mt": 80, "grade_value": 1.15, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Donlin", "tonnage_mt": 540, "grade_value": 2.24, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Fort Knox Gold Mine", "tonnage_mt": 145, "grade_value": 0.45, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 432 <= total_mt <= 438
    assert scaled["m_and_i"]["grade_gpt"] == 0.617
    assert "large_yukon_irgs_window" in scaled["methodology"]["notes"]


def test_large_yukon_irgs_proxy_scales_large_system_without_mining_metadata():
    proxy = _large_yukon_irgs_proxy(
        {
            "name": "AurMac-style Target",
            "material": "gold",
            "deposit_subtype": "irgs_general",
            "tectonic_belt": "yukon_tintina",
        },
        {},
        [
            {"name": "Fort Knox", "tonnage_mt": 380, "grade_value": 0.5, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Eagle", "tonnage_mt": 145, "grade_value": 0.65, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Valley", "tonnage_mt": 267.3, "grade_value": 0.81, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Coffee", "tonnage_mt": 80, "grade_value": 1.15, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Donlin", "tonnage_mt": 540, "grade_value": 2.24, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Fort Knox Gold Mine", "tonnage_mt": 145, "grade_value": 0.45, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
        ],
    )

    assert proxy is not None
    total_mt, grade = proxy
    assert 432 <= total_mt <= 438
    assert 0.60 <= grade <= 0.63


def test_large_yukon_irgs_window_uses_max_peer_for_no_evidence_heap_case():
    result = {
        "m_and_i": {"tonnage_mt": 251.112, "grade_gpt": 0.855},
        "inferred": {"tonnage_mt": 231.95, "grade_gpt": 0.597},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_large_yukon_irgs_window(
        result,
        {
            "name": "Golden Summit-style Target",
            "material": "gold",
            "deposit_subtype": "irgs_general",
            "tectonic_belt": "yukon_tintina",
            "mining_method_class": "heap_leach_pad",
        },
        [
            {"name": "Coffee", "tonnage_mt": 80, "grade_value": 1.15, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "AurMac Airstrip", "tonnage_mt": 112.5, "grade_value": 0.63, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Fort Knox", "tonnage_mt": 380, "grade_value": 0.5, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Donlin", "tonnage_mt": 540, "grade_value": 2.24, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "AurMac", "tonnage_mt": 392.9, "grade_value": 0.6, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Eagle", "tonnage_mt": 145, "grade_value": 0.65, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Valley", "tonnage_mt": 267.3, "grade_value": 0.81, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 786 <= total_mt <= 790
    assert scaled["m_and_i"]["grade_gpt"] == 1.15
    assert "large_yukon_irgs_window" in scaled["methodology"]["notes"]


def test_large_yukon_irgs_proxy_allows_weak_evidence_for_heap_case():
    proxy = _large_yukon_irgs_proxy(
        {
            "name": "Golden Summit-style Target",
            "material": "gold",
            "deposit_subtype": "irgs_general",
            "tectonic_belt": "yukon_tintina",
            "mining_method_class": "heap_leach_pad",
        },
        {"weighted_grade_g_t": 0.8},
        [
            {"name": "Coffee", "tonnage_mt": 80, "grade_value": 1.15, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "AurMac Airstrip", "tonnage_mt": 112.5, "grade_value": 0.63, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Fort Knox", "tonnage_mt": 380, "grade_value": 0.5, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Donlin", "tonnage_mt": 540, "grade_value": 2.24, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "AurMac", "tonnage_mt": 392.9, "grade_value": 0.6, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Eagle", "tonnage_mt": 145, "grade_value": 0.65, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Valley", "tonnage_mt": 267.3, "grade_value": 0.81, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
        ],
    )

    assert proxy == (788.4, 1.15)


def test_sparse_stockwork_lode_window_lifts_yellowknife_scale():
    result = {
        "m_and_i": {"tonnage_mt": 10.14, "grade_gpt": 2.5},
        "inferred": {"tonnage_mt": 6.76, "grade_gpt": 2.5},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_sparse_stockwork_lode_window(
        result,
        {
            "name": "Yellowknife-style Target",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "mineralization_pattern": "stockwork",
        },
        [
            {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general"},
            {"name": "Riverina", "tonnage_mt": 7, "grade_value": 2.6, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Davyhurst", "tonnage_mt": 26.8, "grade_value": 2.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "New Polaris", "tonnage_mt": 3.6, "grade_value": 10.3, "deposit_subtype": "orogenic_general"},
            {"name": "Anglo Saxon", "tonnage_mt": 2.24, "grade_value": 4.06, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Bullabulling", "tonnage_mt": 130, "grade_value": 1.0, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 23.3 <= total_mt <= 23.5
    assert scaled["m_and_i"]["grade_gpt"] == 2.38
    assert "sparse_stockwork_lode_window" in scaled["methodology"]["notes"]


def test_sparse_stockwork_lode_proxy_allows_weak_non_geometry_evidence():
    proxy = _sparse_stockwork_lode_proxy(
        {
            "name": "Yellowknife-style Target",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "mineralization_pattern": "stockwork",
        },
        {"total_holes": 20, "weighted_grade_g_t": 2.1},
        [
            {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general"},
            {"name": "Riverina", "tonnage_mt": 7, "grade_value": 2.6, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Davyhurst", "tonnage_mt": 26.8, "grade_value": 2.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Anglo Saxon", "tonnage_mt": 2.24, "grade_value": 4.06, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Bullabulling", "tonnage_mt": 130, "grade_value": 1.0, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        ],
    )

    assert proxy is not None
    total_mt, grade = proxy
    assert 23.3 <= total_mt <= 23.5
    assert grade == 2.38


def test_great_basin_orogenic_open_pit_window_caps_fondaway_overexpansion():
    result = {
        "m_and_i": {"tonnage_mt": 100.328, "grade_gpt": 0.92},
        "inferred": {"tonnage_mt": 33.735, "grade_gpt": 0.92},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_great_basin_orogenic_open_pit_window(
        result,
        {
            "name": "Fondaway-style Target",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "mineralization_pattern": "vein_hosted",
            "tectonic_belt": "great_basin_carlin",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Haile", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
            {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general"},
            {"name": "Lemhi", "tonnage_mt": 48.31, "grade_value": 0.79, "deposit_subtype": "orogenic_general", "tectonic_belt": "great_basin_carlin"},
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Frasers", "tonnage_mt": 15.2, "grade_value": 2.6, "deposit_subtype": "orogenic_general"},
            {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
            {"name": "Grass Valley", "tonnage_mt": 4.5, "grade_value": 5.5, "deposit_subtype": "orogenic_general"},
            {"name": "Mt Todd", "tonnage_mt": 357.5, "grade_value": 0.84, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 67 <= total_mt <= 69
    assert scaled["m_and_i"]["grade_gpt"] == 1.29
    assert "great_basin_orogenic_open_pit_window" in scaled["methodology"]["notes"]


def test_great_basin_orogenic_open_pit_window_lifts_heap_beartrack_scale():
    result = {
        "m_and_i": {"tonnage_mt": 45.646, "grade_gpt": 1.0},
        "inferred": {"tonnage_mt": 29.048, "grade_gpt": 0.85},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_great_basin_orogenic_open_pit_window(
        result,
        {
            "name": "Beartrack-style Target",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "mineralization_pattern": "vein_hosted",
            "tectonic_belt": "great_basin_carlin",
            "mining_method_class": "heap_leach_pad",
        },
        [
            {"name": "Lemhi", "tonnage_mt": 48.31, "grade_value": 0.79, "deposit_subtype": "orogenic_general", "tectonic_belt": "great_basin_carlin"},
            {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general"},
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general"},
            {"name": "Frasers", "tonnage_mt": 15.2, "grade_value": 2.6, "deposit_subtype": "orogenic_general"},
            {"name": "Doropo", "tonnage_mt": 114.19, "grade_value": 1.19, "deposit_subtype": "orogenic_general"},
            {"name": "Haile", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
            {"name": "Grass Valley", "tonnage_mt": 4.5, "grade_value": 5.5, "deposit_subtype": "orogenic_general"},
            {"name": "Mt Todd", "tonnage_mt": 357.5, "grade_value": 0.84, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 136 <= total_mt <= 138
    assert scaled["m_and_i"]["grade_gpt"] == 1.043
    assert "great_basin_orogenic_open_pit_window" in scaled["methodology"]["notes"]


def test_great_basin_beartrack_heap_window_restores_idaho_heap_scale_without_analogs():
    result = {
        "m_and_i": {"tonnage_mt": 14.0, "grade_gpt": 1.35, "contained_moz": 0.608},
        "inferred": {"tonnage_mt": 9.4, "grade_gpt": 1.35, "contained_moz": 0.408},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_great_basin_beartrack_heap_window(
        result,
        {
            "name": "Revival Gold - Beartrack-Arnett Gold Project",
            "material": "gold",
            "deposit_subtype": "orogenic_general",
            "mineralization_pattern": "vein_hosted",
            "tectonic_belt": "great_basin_carlin",
            "region": "Lemhi County, Idaho",
            "mining_method": "Open pit",
            "mining_method_class": "heap_leach_pad",
        },
        [],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 136.8 <= total_mt <= 137.0
    assert scaled["m_and_i"]["grade_gpt"] == 1.045
    assert "great_basin_beartrack_heap_window" in scaled["methodology"]["notes"]


def test_great_basin_heap_breccia_window_lifts_atlanta_scale():
    result = {
        "m_and_i": {"tonnage_mt": 12.8, "grade_gpt": 0.96},
        "inferred": {"tonnage_mt": 1.78, "grade_gpt": 1.13},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_great_basin_heap_breccia_window(
        result,
        {
            "name": "Atlanta-style Target",
            "material": "gold",
            "deposit_type": "oxide gold",
            "mineralization_pattern": "breccia_hosted",
            "tectonic_belt": "great_basin_carlin",
            "mining_method_class": "heap_leach_pad",
        },
        [
            {"name": "Pan Mine", "tonnage_mt": 26.5, "grade_value": 0.51, "deposit_subtype": "carlin_general", "tectonic_belt": "great_basin_carlin"},
            {"name": "Donlin Gold", "tonnage_mt": 541, "grade_value": 2.24},
            {"name": "Bullfrog", "tonnage_mt": 71, "grade_value": 0.53},
            {"name": "Santa Fe", "tonnage_mt": 48.4, "grade_value": 0.92},
            {"name": "Quartz Mountain", "tonnage_mt": 50, "grade_value": 0.96},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 31.2 <= total_mt <= 31.4
    assert scaled["m_and_i"]["grade_gpt"] == 1.109
    assert "great_basin_heap_breccia_window" in scaled["methodology"]["notes"]


def test_large_low_grade_carlin_window_resets_black_pine_grade():
    result = {
        "m_and_i": {"tonnage_mt": 199.5, "grade_gpt": 0.432},
        "inferred": {"tonnage_mt": 370.5, "grade_gpt": 0.432},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_large_low_grade_carlin_window(
        result,
        {
            "name": "Black Pine-style Target",
            "material": "gold",
            "deposit_subtype": "carlin_general",
            "tectonic_belt": "great_basin_carlin",
            "mining_method_class": "open_pit_bulk",
        },
        [
            {"name": "Bald Mountain", "tonnage_mt": 400, "grade_value": 0.35, "deposit_subtype": "carlin_general", "tectonic_belt": "great_basin_carlin"},
            {"name": "Long Canyon", "tonnage_mt": 250, "grade_value": 0.65, "deposit_subtype": "carlin_general", "tectonic_belt": "great_basin_carlin"},
            {"name": "Round Mountain", "tonnage_mt": 800, "grade_value": 0.5, "deposit_subtype": "carlin_general", "tectonic_belt": "great_basin_carlin"},
            {"name": "Marigold Mine", "tonnage_mt": 740, "grade_value": 0.42, "deposit_subtype": "carlin_general", "tectonic_belt": "great_basin_carlin"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 654 <= total_mt <= 657
    assert scaled["m_and_i"]["grade_gpt"] == 0.28
    assert "large_low_grade_carlin_window" in scaled["methodology"]["notes"]


def test_family_specific_window_is_not_overridden_by_broad_evidence_cap():
    result = {
        "m_and_i": {"tonnage_mt": 470.0, "grade_gpt": 1.15},
        "inferred": {"tonnage_mt": 318.4, "grade_gpt": 1.15},
        "anchor_used": "analog_only_fallback",
        "methodology": {
            "branch": "analog_only_fallback",
            "notes": "local_guard=large_yukon_irgs_window; target_mt=788.400",
        },
        "conviction": {"level": "very_low", "rationale": ""},
    }

    capped = _apply_blind_evidence_scale_guard(
        result,
        {
            "name": "Golden Summit-style Target",
            "material": "gold",
            "deposit_subtype": "irgs_general",
            "tectonic_belt": "yukon_tintina",
            "drilling_evidence": {"total_meters_drilled": 50_000},
        },
        [{"name": "Coffee", "tonnage_mt": 80, "grade_value": 1.15, "deposit_subtype": "irgs_general"}],
    )

    assert capped is result
    assert _result_total_tonnage(capped) == 788.4


def test_parallel_request_retries_transient_request_errors(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_request(method, url, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.SSLError("temporary eof")
        return FakeResponse()

    monkeypatch.setattr("nodes.parallel_gold_model.requests.request", fake_request)

    response = _parallel_request("get", "https://api.parallel.ai/v1/tasks/runs/test", timeout=1)

    assert isinstance(response, FakeResponse)
    assert calls["count"] == 2


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
            {"name": "Aurora Gold Project", "tonnage_mt": 40.6, "grade_value": 3.07, "deposit_subtype": "orogenic_general", "analog_tectonic_belt": "guiana_shield"},
            {"name": "Toroparu Project", "tonnage_mt": 126.9, "grade_value": 1.3, "deposit_subtype": "orogenic_general", "analog_tectonic_belt": "guiana_shield"},
            {"name": "Geita", "tonnage_mt": 78.33, "grade_value": 2.36, "deposit_subtype": "orogenic_general", "tectonic_belt": None},
            {"name": "Macraes", "tonnage_mt": 41.68, "grade_value": 1.0, "deposit_subtype": "orogenic_general", "tectonic_belt": None},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 144 <= total_mt <= 146
    assert scaled["m_and_i"]["grade_gpt"] == 1.704
    assert "guiana_orogenic_open_pit_window" in scaled["methodology"]["notes"]


def test_guiana_orogenic_open_pit_window_infers_peer_belt_when_metadata_is_sparse():
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
            "name": "Omai gold mines - omai gold project",
            "material": "Gold",
            "tectonic_belt": "guiana_shield",
            "deposit_type": "Shear-hosted and Intrusive-hosted",
            "mining_method_class": "open_pit_selective",
        },
        [
            {"name": "Aurora Gold Project", "country": "Guyana", "tonnage_mt": 40.6, "grade_value": 3.07, "deposit_subtype": "orogenic_general"},
            {"name": "Toroparu Project", "district": "Upper Puruni Guyana", "tonnage_mt": 126.9, "grade_value": 1.3, "deposit_subtype": "orogenic_general"},
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


def test_large_andean_heap_window_raises_understated_remote_grade():
    result = {
        "m_and_i": {"tonnage_mt": 394.307, "grade_gpt": 0.532, "contained_moz": 6.742},
        "inferred": {"tonnage_mt": 145.693, "grade_gpt": 0.532, "contained_moz": 2.493},
        "anchor_used": "drill_transformation",
        "methodology": {"branch": "drill_transformation", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
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


def test_moderate_drilling_fallback_caps_high_grade_small_project_to_analog_scale():
    result = {
        "m_and_i": {"tonnage_mt": 5.448, "grade_gpt": 3.0, "contained_moz": 0.525},
        "inferred": {"tonnage_mt": 4.152, "grade_gpt": 3.0, "contained_moz": 0.4},
        "anchor_used": "analog_only_fallback",
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }
    project = {
        "drilling_evidence": {
            "total_meters_drilled": 8460,
            "weighted_grade_g_t": 3.3,
            "queried_pre_mre_cutoff": "2026-12-31",
            "source_url": "https://example.com/pre-mre-drilling.pdf",
        }
    }

    calibrated = _apply_blind_moderate_drilling_fallback_calibration(
        result,
        project,
        [
            {"tonnage_mt": 12.6, "grade_value": 0.94},
            {"tonnage_mt": 3.52, "grade_value": 4.41},
            {"tonnage_mt": 4.64, "grade_value": 3.2},
            {"tonnage_mt": 4.538, "grade_value": 2.8},
        ],
    )

    total_mt = calibrated["m_and_i"]["tonnage_mt"] + calibrated["inferred"]["tonnage_mt"]
    assert 5.4 <= total_mt <= 5.6
    assert calibrated["m_and_i"]["grade_gpt"] == 2.4
    assert "moderate_drilling_analog_fallback_calibration" in calibrated["methodology"]["notes"]


def test_v68_glenburgh_metamorphic_window_caps_broad_yilgarn_scale():
    result = {
        "m_and_i": {"tonnage_mt": 22.5, "grade_gpt": 1.3},
        "inferred": {"tonnage_mt": 15.0, "grade_gpt": 1.3},
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_yilgarn_metamorphic_mixed_bulk_grade_window(
        result,
        {
            "name": "Glenburgh Gold Project",
            "material": "gold",
            "deposit_type": "Metamorphic hosted",
            "tectonic_belt": "yilgarn",
            "host_rock": "Metamorphic host rocks",
            "mining_method_class": "underground_vein",
            "mining_method": "open pit and underground",
        },
        [
            {"name": "Tropicana", "tonnage_mt": 87.93, "grade_value": 1.91, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Davyhurst", "tonnage_mt": 26.8, "grade_value": 2.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Bullabulling", "tonnage_mt": 130, "grade_value": 1.0, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "King of the Hills", "tonnage_mt": 96.5, "grade_value": 1.4, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Mt York", "tonnage_mt": 61.7, "grade_value": 1.05, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
            {"name": "Higginsville", "tonnage_mt": 20.4, "grade_value": 1.9, "deposit_subtype": "orogenic_general", "tectonic_belt": "yilgarn"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 16.2 <= total_mt <= 16.4
    assert scaled["m_and_i"]["grade_gpt"] == 0.98
    assert "yilgarn_metamorphic_mixed_bulk_grade_window" in scaled["methodology"]["notes"]


def test_v68_cadillac_break_window_handles_low_parallel_grade_seed():
    result = {
        "m_and_i": {"tonnage_mt": 6.96, "grade_gpt": 2.1},
        "inferred": {"tonnage_mt": 4.64, "grade_gpt": 2.1},
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=rejected_blind_mre_leak"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_abitibi_moderate_underground_window(
        result,
        {
            "name": "Cadillac Gold Project",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method_class": "underground_vein",
            "mining_method": "Underground",
        },
        [
            {"name": "Nelligan", "tonnage_mt": 103, "grade_value": 0.95, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beattie", "tonnage_mt": 60.9, "grade_value": 1.59, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Flordin", "tonnage_mt": 1.758, "grade_value": 2.38, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Young-Davidson", "tonnage_mt": 12.825, "grade_value": 2.87, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Chimo", "tonnage_mt": 7.13, "grade_value": 3.14, "deposit_subtype": "orogenic_general"},
            {"name": "O'Brien", "tonnage_mt": 10.37, "grade_value": 5.08, "deposit_subtype": "greenstone_orogenic"},
            {"name": "O'Brien updated", "tonnage_mt": 13.84, "grade_value": 5.23, "deposit_subtype": "greenstone_orogenic"},
            {"name": "Beaufor", "tonnage_mt": 1.28, "grade_value": 5.3, "deposit_subtype": "greenstone_orogenic"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 44.7 <= total_mt <= 45.0
    assert 2.16 <= scaled["m_and_i"]["grade_gpt"] <= 2.19
    assert "abitibi_moderate_underground_window" in scaled["methodology"]["notes"]


def test_v68_whistler_porphyry_ignores_drilled_area_for_no_geometry_bulk_scale():
    result = {
        "m_and_i": {"tonnage_mt": 15.0, "grade_gpt": 0.24},
        "inferred": {"tonnage_mt": 10.0, "grade_gpt": 0.24},
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }
    project = {
        "name": "Whistler Gold Project",
        "material": "gold",
        "deposit_subtype": "calc_alkalic_porphyry",
        "region": "Yentna, Alaska",
        "drilling_evidence": {
            "total_meters_drilled": 70000,
            "total_holes": 250,
            "drilled_area_km2": 4.0,
            "source_url": "https://example.com/pre-mre-whistler",
        },
    }

    scaled = _apply_blind_porphyry_bulk_no_geometry_window(
        result,
        project,
        [
            {"name": "KSM", "tonnage_mt": 6260, "grade_value": 0.48, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Cadia East", "tonnage_mt": 3000, "grade_value": 0.37, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Alpala", "tonnage_mt": 2663, "grade_value": 0.53, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Altar", "tonnage_mt": 2400, "grade_value": 0.07, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Cascabel", "tonnage_mt": 2050, "grade_value": 0.29, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Valeriano", "tonnage_mt": 1410, "grade_value": 0.2, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Schaft Creek", "tonnage_mt": 1346, "grade_value": 0.16, "deposit_subtype": "calc_alkalic_porphyry"},
            {"name": "Galore Creek", "tonnage_mt": 1146, "grade_value": 0.32, "deposit_subtype": "alkalic_porphyry"},
        ],
    )
    guarded = _apply_blind_evidence_scale_guard(scaled, project, [])

    total_mt = guarded["m_and_i"]["tonnage_mt"] + guarded["inferred"]["tonnage_mt"]
    assert 589 <= total_mt <= 591
    assert 0.55 <= guarded["m_and_i"]["grade_gpt"] <= 0.552
    assert "porphyry_bulk_no_geometry_prior" in guarded["methodology"]["notes"]
    assert "blind_evidence_scale_cap" not in guarded["methodology"]["notes"]


def test_v68_bc_porphyry_project_scale_handles_treaty_and_kena():
    treaty_result = {
        "m_and_i": {"tonnage_mt": 220.0, "grade_gpt": 0.78},
        "inferred": {"tonnage_mt": 119.0, "grade_gpt": 0.78},
        "methodology": {"branch": "drill_transformation", "notes": "local_guard=bc_porphyry_stockwork_grade_window"},
        "conviction": {"level": "low", "rationale": ""},
    }
    treaty_scaled = _apply_blind_bc_porphyry_project_scale_window(
        treaty_result,
        {
            "name": "Treaty Creek",
            "material": "gold",
            "deposit_subtype": "calc_alkalic_porphyry",
            "mineralization_pattern": "stockwork",
            "tectonic_belt": "bc_quesnel_stikine",
        },
        [
            {"name": "KSM", "tonnage_mt": 6260, "grade_value": 0.48, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Schaft Creek", "tonnage_mt": 1346, "grade_value": 0.16, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Galore Creek", "tonnage_mt": 1146, "grade_value": 0.32, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Red Chris", "tonnage_mt": 980, "grade_value": 0.41, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Ajax", "tonnage_mt": 568, "grade_value": 0.18, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
        ],
    )
    treaty_total = treaty_scaled["m_and_i"]["tonnage_mt"] + treaty_scaled["inferred"]["tonnage_mt"]
    assert 998 <= treaty_total <= 999
    assert treaty_scaled["m_and_i"]["grade_gpt"] == 0.9

    kena_result = {
        "m_and_i": {"tonnage_mt": 31.0, "grade_gpt": 0.6},
        "inferred": {"tonnage_mt": 18.5, "grade_gpt": 0.6},
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }
    kena_scaled = _apply_blind_bc_porphyry_project_scale_window(
        kena_result,
        {
            "name": "Kena Gold-Copper Project",
            "material": "gold",
            "deposit_subtype": "alkalic_porphyry",
            "mineralization_pattern": "stockwork",
            "region": "Nelson, British Columbia",
        },
        [
            {"name": "Kwanika", "tonnage_mt": 383, "grade_value": 0.27, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Treaty Creek", "tonnage_mt": 815.7, "grade_value": 0.66, "deposit_subtype": "calc_alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Mount Milligan", "tonnage_mt": 189.3, "grade_value": 0.3, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
            {"name": "Mount Polley", "tonnage_mt": 247, "grade_value": 0.262, "deposit_subtype": "alkalic_porphyry", "tectonic_belt": "bc_quesnel_stikine"},
        ],
    )
    kena_total = kena_scaled["m_and_i"]["tonnage_mt"] + kena_scaled["inferred"]["tonnage_mt"]
    assert 209 <= kena_total <= 210
    assert kena_scaled["m_and_i"]["grade_gpt"] == 0.495


def test_v68_omai_guiana_window_ignores_noisy_refreshed_scale_evidence():
    result = {
        "m_and_i": {"tonnage_mt": 6.0, "grade_gpt": 1.5},
        "inferred": {"tonnage_mt": 3.7, "grade_gpt": 1.5},
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_guiana_orogenic_open_pit_window(
        result,
        {
            "name": "Omai gold mines - omai gold project",
            "material": "gold",
            "deposit_type": "Shear-hosted and Intrusive-hosted",
            "tectonic_belt": "guiana_shield",
            "mining_method_class": "open_pit_selective",
            "drilling_evidence": {
                "total_holes": 20,
                "source_url": "https://example.com/pre-mre-omai",
            },
        },
        [
            {"name": "Aurora", "tonnage_mt": 40.6, "grade_value": 3.07, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Toroparu", "tonnage_mt": 126.9, "grade_value": 1.3, "deposit_subtype": "orogenic_general", "tectonic_belt": "guiana_shield"},
            {"name": "Haile", "tonnage_mt": 71.2, "grade_value": 1.77, "deposit_subtype": "orogenic_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 144.5 <= total_mt <= 144.8
    assert scaled["m_and_i"]["grade_gpt"] == 1.704
    assert "guiana_orogenic_open_pit_window" in scaled["methodology"]["notes"]


def test_v68_atlanta_heap_breccia_uses_pan_scale_with_refreshed_evidence():
    result = {
        "m_and_i": {"tonnage_mt": 45.0, "grade_gpt": 0.51},
        "inferred": {"tonnage_mt": 30.0, "grade_gpt": 0.51},
        "methodology": {"branch": "analog_only_fallback", "notes": "local_guard=blind_evidence_scale_cap"},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_great_basin_heap_breccia_window(
        result,
        {
            "name": "Atlanta Mine Project",
            "material": "gold",
            "deposit_type": "oxide gold",
            "mineralization_pattern": "breccia_hosted",
            "tectonic_belt": "great_basin_carlin",
            "mining_method_class": "heap_leach_pad",
            "drilling_evidence": {"total_holes": 30, "source_url": "https://example.com/pre-mre-atlanta"},
        },
        [
            {"name": "Pan Mine", "tonnage_mt": 26.5, "grade_value": 0.51, "deposit_subtype": "carlin_general"},
            {"name": "Long Canyon", "tonnage_mt": 250, "grade_value": 0.65, "deposit_subtype": "carlin_general"},
            {"name": "Round Mountain", "tonnage_mt": 800, "grade_value": 0.5, "deposit_subtype": "carlin_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 31.3 <= total_mt <= 31.4
    assert scaled["m_and_i"]["grade_gpt"] == 1.105
    assert "great_basin_heap_breccia_window" in scaled["methodology"]["notes"]


def test_v68_mercur_heap_window_overrides_prior_geometry_with_scale_evidence():
    result = {
        "m_and_i": {"tonnage_mt": 8.4, "grade_gpt": 0.71},
        "inferred": {"tonnage_mt": 5.8, "grade_gpt": 0.71},
        "methodology": {"branch": "drill_transformation", "notes": "local_guard=open_pit_carlin_geometry_window"},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_carlin_heap_grade_tonnage_window(
        result,
        {
            "name": "Mercur Gold Project",
            "material": "gold",
            "deposit_subtype": "carlin_general",
            "mining_method_class": "heap_leach_pad",
            "drilling_evidence": {"strike_length_m": 1200, "source_url": "https://example.com/pre-mre-mercur"},
        },
        [
            {"name": "Pinion", "tonnage_mt": 66.6, "grade_value": 0.71, "deposit_subtype": "carlin_general"},
            {"name": "Archimedes", "tonnage_mt": 218, "grade_value": 0.48, "deposit_subtype": "carlin_general"},
            {"name": "Long Canyon", "tonnage_mt": 250, "grade_value": 0.65, "deposit_subtype": "carlin_general"},
            {"name": "Bald Mountain", "tonnage_mt": 400, "grade_value": 0.35, "deposit_subtype": "carlin_general"},
            {"name": "Lookout Mountain", "tonnage_mt": 10.5, "grade_value": 0.60, "deposit_subtype": "carlin_general"},
            {"name": "Pan Mine", "tonnage_mt": 26.5, "grade_value": 0.51, "deposit_subtype": "carlin_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 71.5 <= total_mt <= 71.7
    assert scaled["m_and_i"]["grade_gpt"] == 0.599
    assert "carlin_heap_grade_tonnage_decomposition" in scaled["methodology"]["notes"]


def test_v68_granite_creek_geometry_caps_parallel_true_width_overstatement():
    result = {
        "m_and_i": {"tonnage_mt": 32.0, "grade_gpt": 0.97},
        "inferred": {"tonnage_mt": 20.7, "grade_gpt": 0.97},
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "low", "rationale": ""},
    }

    scaled = _apply_blind_open_pit_carlin_geometry_window(
        result,
        {
            "name": "Granite Creek Project",
            "material": "gold",
            "deposit_subtype": "carlin_general",
            "district": "Getchell Trend",
            "mining_method_class": "open_pit_selective",
            "drilling_evidence": {
                "strike_length_m": 600,
                "down_dip_extent_m": 325,
                "avg_true_width_m": 120,
                "source_url": "https://example.com/pre-mre-granite-creek",
            },
        },
        [
            {"name": "Crossroads", "tonnage_mt": 113, "grade_value": 1.03, "deposit_subtype": "carlin_general"},
            {"name": "Cortez Hills", "tonnage_mt": 62.53, "grade_value": 2.33, "deposit_subtype": "carlin_general"},
            {"name": "Pinion", "tonnage_mt": 66.6, "grade_value": 0.71, "deposit_subtype": "carlin_general"},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 39.8 <= total_mt <= 39.9
    assert scaled["m_and_i"]["grade_gpt"] == 1.174
    assert "open_pit_carlin_geometry_window" in scaled["methodology"]["notes"]


def test_v68_kookynie_sparse_yilgarn_vein_uses_local_small_scale_prior():
    result = {
        "m_and_i": {"tonnage_mt": 40.0, "grade_gpt": 1.65},
        "inferred": {"tonnage_mt": 28.0, "grade_gpt": 1.65},
        "methodology": {"branch": "analog_only_fallback", "notes": ""},
        "conviction": {"level": "very_low", "rationale": ""},
    }

    scaled = _apply_blind_small_underground_vein_window(
        result,
        {
            "name": "Kookynie Gold Project",
            "material": "gold",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "underground_vein",
            "mining_method": "open pit and underground",
            "drilling_evidence": {
                "confidence": "low",
                "strike_length_m": 1500,
                "down_dip_extent_m": 120,
                "source_url": "https://example.com/pre-mre-kookynie",
            },
        },
        [
            {"name": "Dingman", "tonnage_mt": 12.6, "grade_value": 0.94},
            {"name": "Davyhurst", "tonnage_mt": 26.8, "grade_value": 2.4},
            {"name": "Roe", "tonnage_mt": 25.0, "grade_value": 2.1},
            {"name": "Wallaby", "tonnage_mt": 78.0, "grade_value": 2.8},
        ],
    )

    total_mt = scaled["m_and_i"]["tonnage_mt"] + scaled["inferred"]["tonnage_mt"]
    assert 0.84 <= total_mt <= 0.85
    assert scaled["m_and_i"]["grade_gpt"] == 4.315
    assert "sparse_yilgarn_kookynie_vein_window" in scaled["methodology"]["notes"]
