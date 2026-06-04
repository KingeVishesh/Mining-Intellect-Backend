from __future__ import annotations

from scripts.run_parallel_gold_backtest import (
    _blind_library_analog_is_compatible,
    _blind_cutoff_from_mre_run,
    _gold_library_filters,
    _merge_library_analogs,
    _select_truth_target_rows,
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


def test_gold_library_filters_infer_orogenic_from_archean_gold_belt():
    filters = _gold_library_filters({
        "name": "Moss",
        "material": "gold",
        "tectonic_belt": "abitibi",
    })

    assert filters["deposit_type"] == "orogenic gold"
    assert filters["deposit_subtype"] == "orogenic_general"


def test_backfilled_year_only_mre_uses_conservative_blind_cutoff():
    cutoff = _blind_cutoff_from_mre_run({
        "source": "exa_2pass_mre_truth_backfill",
        "effective_date": "2026-12-31",
    })

    assert cutoff == "2026-01-01"
