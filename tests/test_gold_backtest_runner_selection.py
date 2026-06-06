from __future__ import annotations

import json

from scripts.run_parallel_gold_backtest import (
    _analog_diagnostics,
    _blind_library_analog_is_compatible,
    _blind_gold_needs_library_expansion,
    _blind_cutoff_from_mre_run,
    _evidence_is_pre_cutoff,
    _evidence_score_value,
    _gold_library_filters,
    _error_class,
    _is_parallel_quota_error,
    _merge_library_analogs,
    _parse_loose_date,
    _resume_project_ids_from_leaderboard,
    _select_audit_target_rows,
    _select_truth_target_rows,
    _supplement_with_library_analogs,
)


def _truth_row(project_id: str, name: str) -> dict:
    return {
        "id": project_id,
        "name": name,
        "mre_mi_tonnage_mt": 1.0,
        "mre_mi_grade": 1.0,
        "mre_inferred_tonnage_mt": 1.0,
        "mre_inferred_grade": 1.0,
    }


def test_random_truth_target_selection_excludes_prior_projects_and_is_seeded():
    rows = [
        _truth_row("p1", "Alpha"),
        _truth_row("p2", "Beta"),
        _truth_row("p3", "Gamma"),
        _truth_row("p4", "Delta"),
        _truth_row("p5", "Epsilon"),
        {"id": "p6", "name": "No Truth"},
    ]

    first = _select_truth_target_rows(
        rows,
        limit=3,
        exclude_project_ids={"p2"},
        random_seed="holdout-1",
        randomize=True,
    )
    second = _select_truth_target_rows(
        rows,
        limit=3,
        exclude_project_ids={"p2"},
        random_seed="holdout-1",
        randomize=True,
    )

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert "p2" not in {row["id"] for row in first}
    assert "p6" not in {row["id"] for row in first}
    assert len(first) == 3


def test_default_truth_target_selection_preserves_available_order_after_exclusions():
    rows = [
        _truth_row("p1", "Alpha"),
        _truth_row("p2", "Beta"),
        _truth_row("p3", "Gamma"),
    ]

    selected = _select_truth_target_rows(
        rows,
        limit=2,
        exclude_project_ids={"p2"},
    )

    assert [row["id"] for row in selected] == ["p1", "p3"]


def test_audit_target_selection_uses_queue_statuses_and_truth_gate():
    rows = [
        {
            "project_id": "p1",
            "name": "Validated",
            "has_official_split": True,
            "backtest_status": "validated_pass",
        },
        {
            "project_id": "p2",
            "name": "Quota",
            "has_official_split": True,
            "backtest_status": "retry_after_quota",
        },
        {
            "project_id": "p3",
            "name": "Miss",
            "has_official_split": True,
            "backtest_status": "needs_accuracy_review",
        },
        {
            "project_id": "p4",
            "name": "No Truth",
            "has_official_split": False,
            "backtest_status": "retry_after_quota",
        },
    ]

    selected = _select_audit_target_rows(
        rows,
        statuses={"retry_after_quota", "needs_accuracy_review"},
        limit=10,
        exclude_project_ids={"p2"},
    )

    assert [row["project_id"] for row in selected] == ["p3"]


def test_audit_target_selection_is_seeded():
    rows = [
        {
            "project_id": f"p{i}",
            "name": f"Project {i}",
            "has_official_split": True,
            "backtest_status": "retry_after_quota",
        }
        for i in range(8)
    ]

    first = _select_audit_target_rows(
        rows,
        statuses={"retry_after_quota"},
        limit=4,
        random_seed="queue-1",
        randomize=True,
    )
    second = _select_audit_target_rows(
        rows,
        statuses={"retry_after_quota"},
        limit=4,
        random_seed="queue-1",
        randomize=True,
    )

    assert [row["project_id"] for row in first] == [row["project_id"] for row in second]
    assert len(first) == 4


def test_parallel_quota_errors_are_classified_without_retrying_as_transient():
    error = "Parallel API call failed: 402 Client Error: Payment Required for url"

    assert _is_parallel_quota_error(error)
    assert _error_class(error) == "parallel_quota"
    assert _error_class("Parallel API call failed: 500 Server Error") == "parallel_api"
    assert _error_class("project not found") == "error"


def test_analog_diagnostics_flags_sparse_clean_pool():
    diagnostics = _analog_diagnostics(
        {"name": "Sparse Gold", "material": "gold", "deposit_subtype": "orogenic_general"},
        [{"name": "Only Analog", "tonnage_mt": 10, "grade_value": 1.0}],
    )

    assert diagnostics["supplied_count"] == 1
    assert diagnostics["clean_count"] == 1
    assert diagnostics["needs_analog_refresh"] is True
    assert diagnostics["clean_names"] == ["Only Analog"]


def test_resume_leaderboard_modes_select_retry_targets(tmp_path):
    path = tmp_path / "leaderboard.json"
    path.write_text(
        json.dumps({
            "target_selection": {"project_ids": ["p1", "p2", "p3", "p4"]},
            "leaderboard": [
                {"project": "Alpha", "project_id": "p1", "pass": True},
                {"project": "Beta", "project_id": "p2", "pass": False},
            ],
            "errors": [
                {"project": "Gamma", "project_id": "p3", "error_class": "parallel_quota"},
                {"project": "Delta", "project_id": "p4", "error_class": "parallel_quota_skipped"},
            ],
        }),
        encoding="utf-8",
    )

    assert _resume_project_ids_from_leaderboard(path, mode="errors") == ["p3", "p4"]
    assert _resume_project_ids_from_leaderboard(path, mode="misses") == ["p2"]
    assert _resume_project_ids_from_leaderboard(path, mode="non_passed") == ["p3", "p4", "p2"]
    assert _resume_project_ids_from_leaderboard(path, mode="incomplete") == ["p2", "p3", "p4"]


def test_resume_leaderboard_dedupes_targets(tmp_path):
    path = tmp_path / "leaderboard.json"
    path.write_text(
        json.dumps({
            "target_selection": {"project_ids": ["p1", "p2"]},
            "leaderboard": [
                {"project": "Beta", "project_id": "p2", "pass": False},
            ],
            "errors": [
                {"project": "Beta", "project_id": "p2", "error_class": "parallel_api"},
            ],
        }),
        encoding="utf-8",
    )

    assert _resume_project_ids_from_leaderboard(path, mode="non_passed") == ["p2"]


def test_library_analog_merge_dedupes_and_drops_self_named_rows():
    project = {"name": "Example Gold Project", "material": "gold"}
    supplied = [{"name": "Supplied Analog", "tonnage_mt": 10, "grade_value": 1.0}]
    library = [
        {"name": "Example Gold Project", "tonnage_mt": 99, "grade_value": 1.0},
        {"name": "Supplied Analog", "tonnage_mt": 11, "grade_value": 1.0},
        {"name": "Library Analog A", "tonnage_mt": 20, "grade_value": 1.1},
        {"name": "Library Analog B", "tonnage_mt": 30, "grade_value": 1.2},
    ]

    merged = _merge_library_analogs(project, supplied, library)

    assert [row["name"] for row in merged] == [
        "Supplied Analog",
        "Library Analog A",
        "Library Analog B",
    ]


def test_library_analog_merge_dedupes_near_identical_resource_variants():
    project = {"name": "Cadillac Gold Project", "material": "gold"}
    library = [
        {"name": "Chimo Mine", "tonnage_mt": 7.13, "grade_value": 3.14, "deposit_subtype": "orogenic_general"},
        {
            "name": "Chimo Mine and West Nordeau",
            "tonnage_mt": 7.128,
            "grade_value": 3.14,
            "deposit_subtype": "greenstone_orogenic",
        },
        {"name": "Lamaque", "tonnage_mt": 30, "grade_value": 6, "deposit_subtype": "greenstone_orogenic"},
    ]

    merged = _merge_library_analogs(project, [], library)

    assert [row["name"] for row in merged] == ["Chimo Mine", "Lamaque"]


def test_blind_library_filter_rejects_low_grade_bulk_for_underground_gold():
    project = {
        "name": "High Grade Underground",
        "material": "gold",
        "mining_method_class": "underground_vein",
    }

    assert not _blind_library_analog_is_compatible(
        project,
        {"name": "Pinion", "tonnage_mt": 66.6, "grade_value": 0.71},
    )
    assert not _blind_library_analog_is_compatible(
        project,
        {"name": "Carlin Complex", "tonnage_mt": 230, "grade_value": 3.43},
    )
    assert _blind_library_analog_is_compatible(
        project,
        {"name": "Jerritt Canyon", "tonnage_mt": 10.3, "grade_value": 4.65},
    )


def test_blind_library_filter_prefers_bulk_porphyry_over_bad_underground_tag():
    project = {
        "name": "Treaty Creek",
        "material": "gold",
        "deposit_subtype": "calc_alkalic_porphyry",
        "mineralization_pattern": "stockwork",
        "mining_method_class": "underground_vein",
    }

    assert _blind_library_analog_is_compatible(
        project,
        {
            "name": "KSM",
            "tonnage_mt": 5400,
            "grade_value": 0.51,
            "deposit_subtype": "calc_alkalic_porphyry",
        },
    )
    assert not _blind_library_analog_is_compatible(
        project,
        {
            "name": "Ana Paula",
            "tonnage_mt": 21.4,
            "grade_value": 2.16,
            "deposit_subtype": "calc_alkalic_porphyry",
        },
    )


def test_blind_library_expansion_keeps_high_grade_underground_cohort():
    project = {
        "name": "Perron-style Target",
        "material": "gold",
        "deposit_subtype": "orogenic_general",
        "tectonic_belt": "abitibi",
        "mining_method_class": "underground_vein",
    }
    analogs = [
        {"name": "Westwood", "tonnage_mt": 22, "grade_value": 5.4},
        {"name": "Casa Berardi", "tonnage_mt": 30, "grade_value": 4.5},
        {"name": "Lamaque", "tonnage_mt": 30, "grade_value": 6.0},
        {"name": "Madsen", "tonnage_mt": 2.7, "grade_value": 8.9},
    ]

    assert not _blind_gold_needs_library_expansion(project, analogs)


def test_blind_library_expansion_keeps_single_yukon_sediment_hosted_anchor():
    project = {
        "name": "Hyland-style Target",
        "material": "gold",
        "deposit_subtype": "sediment_hosted_general",
        "tectonic_belt": "yukon_tintina",
    }
    analogs = [
        {"name": "Brewery Creek", "tonnage_mt": 31.0, "grade_value": 1.0, "deposit_subtype": "irgs_general"},
    ]

    assert not _blind_gold_needs_library_expansion(project, analogs)


def test_blind_library_expansion_keeps_sparse_bc_stockwork_cohort():
    project = {
        "name": "Kena-style Target",
        "material": "gold",
        "deposit_subtype": "alkalic_porphyry",
        "mineralization_pattern": "stockwork",
        "tectonic_belt": "bc_quesnel_stikine",
    }
    analogs = [
        {"name": "Kwanika", "tonnage_mt": 383, "grade_value": 0.27},
        {"name": "Treaty Creek", "tonnage_mt": 815.7, "grade_value": 0.66},
        {"name": "Mount Milligan", "tonnage_mt": 189.3, "grade_value": 0.30},
        {"name": "Mount Polley", "tonnage_mt": 247, "grade_value": 0.262},
    ]

    assert not _blind_gold_needs_library_expansion(project, analogs)


def test_supplement_expands_narrow_abitibi_greenstone_cohort(monkeypatch):
    calls = []

    def fake_get_approved_analogs(**kwargs):
        calls.append(kwargs)
        return [
            {
                "name": "Nelligan Gold Project",
                "tonnage_mt": 103,
                "grade_value": 0.95,
                "deposit_subtype": "greenstone_orogenic",
            },
            {
                "name": "Beattie Gold Deposit",
                "tonnage_mt": 60.9,
                "grade_value": 1.59,
                "deposit_subtype": "greenstone_orogenic",
            },
        ]

    monkeypatch.setattr(
        "scripts.run_parallel_gold_backtest.supabase_ops.get_approved_analogs",
        fake_get_approved_analogs,
    )

    analogs = _supplement_with_library_analogs(
        {
            "name": "Tower Gold Project",
            "material": "gold",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method_class": "underground_vein",
        },
        [
            {
                "name": "Canadian Malartic - Odyssey UG",
                "tonnage_mt": 110,
                "grade_value": 2.5,
                "deposit_subtype": "greenstone_orogenic",
                "mining_method_class": "underground_vein",
            },
            {
                "name": "Casa Berardi",
                "tonnage_mt": 30,
                "grade_value": 4.5,
                "deposit_subtype": "greenstone_orogenic",
                "mining_method_class": "underground_vein",
            },
            {
                "name": "Lamaque Complex",
                "tonnage_mt": 30,
                "grade_value": 6,
                "deposit_subtype": "greenstone_orogenic",
                "mining_method_class": "underground_vein",
            },
        ],
    )

    assert calls
    assert [analog["name"] for analog in analogs[:2]] == [
        "Nelligan Gold Project",
        "Beattie Gold Deposit",
    ]


def test_supplement_decides_from_cutoff_cleaned_abitibi_pool(monkeypatch):
    calls = []

    def fake_get_approved_analogs(**kwargs):
        calls.append(kwargs)
        return [
            {
                "name": "Nelligan Gold Project",
                "tonnage_mt": 103,
                "grade_value": 0.95,
                "deposit_subtype": "greenstone_orogenic",
                "tectonic_belt": "abitibi",
            },
            {
                "name": "Beattie Gold Deposit",
                "tonnage_mt": 60.9,
                "grade_value": 1.59,
                "deposit_subtype": "greenstone_orogenic",
                "tectonic_belt": "abitibi",
            },
        ]

    monkeypatch.setattr(
        "scripts.run_parallel_gold_backtest.supabase_ops.get_approved_analogs",
        fake_get_approved_analogs,
    )

    analogs = _supplement_with_library_analogs(
        {
            "name": "Cadillac Gold Project",
            "material": "gold",
            "deposit_type": "Gold-bearing structures",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method_class": "underground_vein",
            "mre_date": "2022-08-22",
        },
        [
            {
                "name": "Canadian Malartic - Odyssey UG",
                "tonnage_mt": 110,
                "grade_value": 2.5,
                "deposit_subtype": "greenstone_orogenic",
                "tectonic_belt": "abitibi",
                "data_source": "resource table 2023",
            },
            {
                "name": "Lamaque Complex",
                "tonnage_mt": 30,
                "grade_value": 6,
                "deposit_subtype": "greenstone_orogenic",
                "tectonic_belt": "abitibi",
                "data_source": "resource table 2023",
            },
            {"name": "Chimo Mine", "tonnage_mt": 7.13, "grade_value": 3.14, "deposit_subtype": "orogenic_general"},
            {
                "name": "Chimo Mine and West Nordeau",
                "tonnage_mt": 7.128,
                "grade_value": 3.14,
                "deposit_subtype": "greenstone_orogenic",
            },
        ],
    )

    names = [analog["name"] for analog in analogs]
    assert calls
    assert "Canadian Malartic - Odyssey UG" not in names
    assert "Lamaque Complex" not in names
    assert names[:2] == ["Nelligan Gold Project", "Beattie Gold Deposit"]


def test_gold_library_filters_infer_orogenic_from_archean_gold_belt():
    filters = _gold_library_filters({
        "name": "Moss",
        "material": "gold",
        "tectonic_belt": "abitibi",
    })

    assert filters["deposit_type"] == "orogenic gold"
    assert filters["deposit_subtype"] == "orogenic_general"


def test_gold_library_filters_infer_high_sulfidation_from_andean_heap_leach():
    filters = _gold_library_filters({
        "name": "Volcan Gold Project",
        "material": "gold",
        "tectonic_belt": "andean",
        "district": "Maricunga Gold Belt",
        "mining_method": "open pit",
        "processing_method": "heap leach",
        "mining_method_class": "heap_leach_pad",
    })

    assert filters["deposit_type"] == "epithermal-HS"
    assert filters["deposit_subtype"] == "high_sulfidation_epithermal"


def test_gold_library_filters_infer_irgs_from_yukon_near_surface_gold():
    filters = _gold_library_filters({
        "name": "White Gold Project",
        "material": "gold",
        "deposit_type": "Near-surface gold deposits",
        "tectonic_belt": "yukon_tintina",
        "region": "Yukon",
    })

    assert filters["deposit_type"] == "intrusion-related gold"
    assert filters["deposit_subtype"] == "irgs_general"


def test_gold_library_filters_route_carlin_targets_to_great_basin_pool():
    filters = _gold_library_filters({
        "name": "Mercur Gold Project",
        "material": "gold",
        "deposit_type": "Carlin-type",
        "deposit_subtype": "carlin_general",
        "tectonic_belt": "laramide_southwest",
        "region": "Utah",
        "district": "Camp Floyd and Ophir Mining District",
    })

    assert filters["target_tectonic_belt"] == "great_basin_carlin"


def test_gold_library_filters_route_guiana_shear_hosted_to_orogenic():
    filters = _gold_library_filters({
        "name": "Omai gold project",
        "material": "gold",
        "deposit_type": "Shear-hosted and Intrusive-hosted",
        "tectonic_belt": "guiana_shield",
    })

    assert filters["deposit_type"] == "orogenic gold"
    assert filters["deposit_subtype"] == "orogenic_general"


def test_mineral_resource_evidence_url_is_not_pre_mre_clean():
    assert not _evidence_is_pre_cutoff(
        {
            "source_url": "https://example.com/company-announces-mineral-resource-for-wawa-gold-project",
            "source_date": "2024-08-28",
            "queried_pre_mre_cutoff": "2026-01-01",
        },
        _parse_loose_date("2026-01-01"),
    )


def test_pre_mre_evidence_uses_latest_intercept_date_when_top_source_missing():
    assert _evidence_is_pre_cutoff(
        {
            "source_url": "https://www.sec.gov/Archives/edgar/data/1840616/000106299325016976/exhibit99-1.htm",
            "source_date": None,
            "report_cutoff_date": "2026-12-31",
            "queried_pre_mre_cutoff": "2026-12-31",
            "best_intercepts": [
                {"source_date": "2025-03-04", "interval_m": 28, "grade_g_t": 12},
                {"source_date": "2025-04-17", "interval_m": 29.8, "grade_g_t": 5.5},
            ],
        },
        _parse_loose_date("2026-01-01"),
    )


def test_supplement_uses_local_orogenic_fallback_for_newfoundland_irgs(monkeypatch):
    calls = []

    def fake_get_approved_analogs(**kwargs):
        calls.append(kwargs)
        if kwargs.get("target_tectonic_belt") == "newfoundland_appalachian" and kwargs.get("deposit_subtype") == "irgs_general":
            return []
        if kwargs.get("target_tectonic_belt") == "newfoundland_appalachian" and kwargs.get("deposit_subtype") == "orogenic_general":
            return [
                {
                    "name": "Queensway Project",
                    "tonnage_mt": 17.267,
                    "grade_value": 2.25,
                    "deposit_subtype": "orogenic_general",
                    "tectonic_belt": "newfoundland_appalachian",
                },
                {
                    "name": "Valentine Gold Project",
                    "tonnage_mt": 64.62,
                    "grade_value": 1.9,
                    "deposit_subtype": "orogenic_general",
                    "tectonic_belt": "newfoundland_appalachian",
                },
                {
                    "name": "Cape Ray Gold Project",
                    "tonnage_mt": 9.7,
                    "grade_value": 1.96,
                    "deposit_subtype": "orogenic_general",
                    "tectonic_belt": "newfoundland_appalachian",
                },
            ]
        return [
            {
                "name": "Coffee Gold Project",
                "tonnage_mt": 80,
                "grade_value": 1.15,
                "deposit_subtype": "irgs_general",
                "tectonic_belt": "yukon_tintina",
            }
        ]

    monkeypatch.setattr(
        "scripts.run_parallel_gold_backtest.supabase_ops.get_approved_analogs",
        fake_get_approved_analogs,
    )

    analogs = _supplement_with_library_analogs(
        {
            "name": "Clarence Stream Project",
            "material": "gold",
            "deposit_subtype": "irgs_general",
            "tectonic_belt": "newfoundland_appalachian",
        },
        [],
    )

    assert [call.get("target_tectonic_belt") for call in calls] == [
        "newfoundland_appalachian",
        "newfoundland_appalachian",
    ]
    assert [analog["name"] for analog in analogs] == [
        "Queensway Project",
        "Valentine Gold Project",
        "Cape Ray Gold Project",
    ]


def test_supplement_prioritizes_small_yukon_near_surface_vein_peers(monkeypatch):
    def fake_get_approved_analogs(**_kwargs):
        return [
            {"name": "Fort Knox", "tonnage_mt": 380, "grade_value": 0.5, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Donlin Creek", "tonnage_mt": 540, "grade_value": 2.24, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Coffee Gold Project", "tonnage_mt": 80, "grade_value": 1.15, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "RC Gold", "tonnage_mt": 39.96, "grade_value": 1.1, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Brewery Creek", "tonnage_mt": 31, "grade_value": 1.0, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "RC Gold Blackjack", "tonnage_mt": 34.6, "grade_value": 0.94, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
            {"name": "Pogo Mine", "tonnage_mt": 30, "grade_value": 11, "deposit_subtype": "irgs_general", "tectonic_belt": "yukon_tintina"},
        ]

    monkeypatch.setattr(
        "scripts.run_parallel_gold_backtest.supabase_ops.get_approved_analogs",
        fake_get_approved_analogs,
    )

    analogs = _supplement_with_library_analogs(
        {
            "name": "White Gold Project",
            "material": "gold",
            "deposit_type": "Near-surface gold deposits",
            "deposit_subtype": None,
            "tectonic_belt": "yukon_tintina",
            "mineralization_pattern": "vein_hosted",
        },
        [],
    )

    names = [analog["name"] for analog in analogs]
    assert names[:4] == [
        "Coffee Gold Project",
        "RC Gold",
        "RC Gold Blackjack",
        "Brewery Creek",
    ]
    assert "Donlin Creek" not in names
    assert "Pogo Mine" not in names


def test_backfilled_year_only_mre_uses_conservative_blind_cutoff():
    cutoff = _blind_cutoff_from_mre_run({
        "source": "exa_2pass_mre_truth_backfill",
        "effective_date": "2026-12-31",
    })

    assert cutoff == "2026-01-01"


def test_evidence_score_value_extracts_numeric_score_for_comparison():
    richer = {
        "queried_pre_mre_cutoff": "2025-01-01",
        "source_url": "https://example.com/pre-mre",
        "confidence": "medium",
        "total_meters_drilled": 23000,
        "weighted_grade_g_t": 1.0,
        "strike_length_m": 750,
    }
    thinner = {
        "queried_pre_mre_cutoff": "2025-01-01",
        "source_url": "https://example.com/pre-mre",
        "confidence": "low",
    }

    assert _evidence_score_value(richer) > _evidence_score_value(thinner)
