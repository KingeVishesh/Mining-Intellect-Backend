from __future__ import annotations

from scripts.run_parallel_gold_backtest import _select_truth_target_rows


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
