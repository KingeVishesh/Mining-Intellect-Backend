from __future__ import annotations

import json

from scripts.audit_gold_split_coverage import (
    classify_gold_project,
    load_backtest_history,
    project_name_to_id_map,
    summarize,
)


def test_classify_official_split_ready_project():
    row = {
        "id": "p1",
        "name": "Official",
        "tonnage_mt": 10,
        "grade_value": 1,
        "mre_mi_tonnage_mt": 6,
        "mre_mi_grade": 1.1,
        "mre_inferred_tonnage_mt": 4,
        "mre_inferred_grade": 0.9,
    }

    classified = classify_gold_project(row)

    assert classified["primary_status"] == "official_split_ready"
    assert classified["next_action"] == "use_for_backtest"
    assert classified["has_any_split"]


def test_classify_total_resource_without_split_as_truth_extraction_queue():
    row = {"id": "p2", "name": "Total Only", "tonnage_mt": 10, "grade_value": 1}

    classified = classify_gold_project(row)

    assert classified["primary_status"] == "needs_official_split_extraction"
    assert classified["next_action"] == "run_backfill_gold_mre_truth"
    assert not classified["has_any_split"]


def test_classify_no_resource_as_model_run_queue():
    row = {"id": "p3", "name": "No Resource"}

    classified = classify_gold_project(row)

    assert classified["primary_status"] == "needs_model_run"
    assert classified["next_action"] == "run_blind_model_after_accuracy_gate"


def test_summarize_counts_overlap_between_official_and_model_splits():
    payload = summarize([
        {
            "id": "p1",
            "name": "Official",
            "tonnage_mt": 10,
            "grade_value": 1,
            "mre_mi_tonnage_mt": 6,
            "mre_mi_grade": 1.1,
            "mre_inferred_tonnage_mt": 4,
            "mre_inferred_grade": 0.9,
        },
        {
            "id": "p2",
            "name": "Model",
            "mi_tonnage_mt": 5,
            "mi_grade": 1.2,
            "inferred_resource_mt": 5,
            "inferred_grade": 1.0,
        },
        {"id": "p3", "name": "Total Only", "tonnage_mt": 10, "grade_value": 1},
        {"id": "p4", "name": "No Resource"},
    ])

    assert payload["summary"]["gold_projects"] == 4
    assert payload["summary"]["official_split_ready"] == 1
    assert payload["summary"]["model_split_ready"] == 1
    assert payload["summary"]["any_split_ready"] == 2
    assert payload["summary"]["needs_official_split_extraction"] == 1
    assert payload["summary"]["needs_model_run"] == 1


def test_backtest_history_marks_quota_retry_queue(tmp_path):
    artifact = tmp_path / "quota.json"
    artifact.write_text(
        json.dumps({
            "batch_id": "gold_blind_quota",
            "target_selection": {
                "project_ids": ["p1", "p2"],
                "project_names": ["Alpha", "Beta"],
            },
            "leaderboard": [],
            "errors": [
                {"project": "Alpha", "project_id": "p1", "error_class": "parallel_quota"},
                {"project": "Beta", "project_id": "p2", "error_class": "parallel_quota_skipped"},
            ],
        }),
        encoding="utf-8",
    )
    history = load_backtest_history([artifact])

    payload = summarize([
        {
            "id": "p1",
            "name": "Alpha",
            "mre_mi_tonnage_mt": 6,
            "mre_mi_grade": 1.1,
            "mre_inferred_tonnage_mt": 4,
            "mre_inferred_grade": 0.9,
        },
        {
            "id": "p2",
            "name": "Beta",
            "mre_mi_tonnage_mt": 6,
            "mre_mi_grade": 1.1,
            "mre_inferred_tonnage_mt": 4,
            "mre_inferred_grade": 0.9,
        },
    ], backtest_history=history)

    assert payload["summary"]["backtest_retry_after_quota"] == 2
    assert [row["backtest_status"] for row in payload["projects"]] == [
        "retry_after_quota",
        "retry_after_quota",
    ]


def test_backtest_history_marks_pass_and_miss_queues(tmp_path):
    artifact = tmp_path / "leaderboard.json"
    artifact.write_text(
        json.dumps({
            "batch_id": "gold_blind_eval",
            "target_selection": {"project_ids": ["p1", "p2"]},
            "leaderboard": [
                {"project": "Alpha", "project_id": "p1", "pass": True},
                {"project": "Beta", "project_id": "p2", "pass": False},
            ],
            "errors": [],
        }),
        encoding="utf-8",
    )
    history = load_backtest_history([artifact])

    payload = summarize([
        {
            "id": "p1",
            "name": "Alpha",
            "mre_mi_tonnage_mt": 6,
            "mre_mi_grade": 1.1,
            "mre_inferred_tonnage_mt": 4,
            "mre_inferred_grade": 0.9,
        },
        {
            "id": "p2",
            "name": "Beta",
            "mre_mi_tonnage_mt": 6,
            "mre_mi_grade": 1.1,
            "mre_inferred_tonnage_mt": 4,
            "mre_inferred_grade": 0.9,
        },
        {
            "id": "p3",
            "name": "Gamma",
            "mre_mi_tonnage_mt": 6,
            "mre_mi_grade": 1.1,
            "mre_inferred_tonnage_mt": 4,
            "mre_inferred_grade": 0.9,
        },
    ], backtest_history=history)

    assert payload["summary"]["backtest_validated_pass"] == 1
    assert payload["summary"]["backtest_needs_accuracy_review"] == 1
    assert payload["summary"]["backtest_ready_untested"] == 1
    assert [row["backtest_status"] for row in payload["projects"]] == [
        "validated_pass",
        "needs_accuracy_review",
        "ready_untested",
    ]


def test_backtest_history_resolves_legacy_rows_by_project_name(tmp_path):
    artifact = tmp_path / "legacy.json"
    artifact.write_text(
        json.dumps({
            "batch_id": "gold_blind_legacy",
            "target_selection": {"project_ids": ["p1"]},
            "leaderboard": [
                {"project": "Alpha Gold Project", "pass": True},
            ],
            "errors": [],
        }),
        encoding="utf-8",
    )
    projects = [{
        "id": "p1",
        "name": "Alpha Gold Project",
        "mre_mi_tonnage_mt": 6,
        "mre_mi_grade": 1.1,
        "mre_inferred_tonnage_mt": 4,
        "mre_inferred_grade": 0.9,
    }]

    history = load_backtest_history(
        [artifact],
        project_name_to_id=project_name_to_id_map(projects),
    )
    payload = summarize(projects, backtest_history=history)

    assert history["p1"]["pass_count"] == 1
    assert payload["projects"][0]["backtest_status"] == "validated_pass"
