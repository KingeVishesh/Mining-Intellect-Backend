from __future__ import annotations

import json

from scripts.audit_gold_split_coverage import (
    classify_gold_project,
    load_backtest_history,
    load_truth_backfill_history,
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


def test_truth_backfill_history_marks_manual_review_queue(tmp_path):
    artifact = tmp_path / "truth.json"
    artifact.write_text(
        json.dumps([
            {
                "project_id": "p1",
                "name": "Review Project",
                "status": "review",
                "reason": "split weighted grade differs from known total",
                "validation": {"errors_pct": {"grade": -17.0}},
            },
        ]),
        encoding="utf-8",
    )
    history = load_truth_backfill_history([artifact])

    classified = classify_gold_project(
        {"id": "p1", "name": "Review Project", "tonnage_mt": 10, "grade_value": 1},
        truth_backfill_history=history,
    )

    assert classified["truth_backfill_attempt_count"] == 1
    assert classified["truth_backfill_last_status"] == "review"
    assert classified["truth_backfill_last_validation"] == {"errors_pct": {"grade": -17.0}}
    assert classified["next_action"] == "manual_review_official_split"


def test_truth_backfill_history_resolves_rows_by_name(tmp_path):
    artifact = tmp_path / "truth.json"
    artifact.write_text(
        json.dumps([
            {"name": "Mapped Project", "status": "failed", "reason": "no extraction"},
        ]),
        encoding="utf-8",
    )

    history = load_truth_backfill_history(
        [artifact],
        project_name_to_id={"mapped project": "p1"},
    )

    assert history["p1"]["truth_backfill_failed_count"] == 1
    assert history["p1"]["truth_backfill_last_reason"] == "no extraction"


def test_truth_backfill_history_treats_relaxed_usable_as_review(tmp_path):
    artifact = tmp_path / "truth.json"
    artifact.write_text(
        json.dumps([
            {
                "project_id": "p1",
                "status": "usable",
                "reason": "accepted despite local total mismatch: T -32.4%, G -37.9%",
            },
        ]),
        encoding="utf-8",
    )

    history = load_truth_backfill_history([artifact])

    assert history["p1"]["truth_backfill_last_raw_status"] == "usable"
    assert history["p1"]["truth_backfill_last_status"] == "review"
    assert history["p1"]["truth_backfill_review_count"] == 1
    assert history["p1"]["truth_backfill_usable_count"] == 0


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


def test_summarize_counts_truth_backfill_subqueues(tmp_path):
    history = {
        "p1": {
            "truth_backfill_attempt_count": 1,
            "truth_backfill_failed_count": 0,
            "truth_backfill_rejected_count": 0,
            "truth_backfill_review_count": 1,
            "truth_backfill_usable_count": 0,
            "truth_backfill_applied_count": 0,
            "truth_backfill_last_status": "review",
            "truth_backfill_last_reason": "manual check",
            "truth_backfill_last_artifact": "truth.json",
            "truth_backfill_last_validation": None,
        },
        "p2": {
            "truth_backfill_attempt_count": 1,
            "truth_backfill_failed_count": 1,
            "truth_backfill_rejected_count": 0,
            "truth_backfill_review_count": 0,
            "truth_backfill_usable_count": 0,
            "truth_backfill_applied_count": 0,
            "truth_backfill_last_status": "failed",
            "truth_backfill_last_reason": "no extraction",
            "truth_backfill_last_artifact": "truth.json",
            "truth_backfill_last_validation": None,
        },
        "p3": {
            "truth_backfill_attempt_count": 1,
            "truth_backfill_failed_count": 0,
            "truth_backfill_rejected_count": 0,
            "truth_backfill_review_count": 0,
            "truth_backfill_usable_count": 1,
            "truth_backfill_applied_count": 0,
            "truth_backfill_last_status": "usable",
            "truth_backfill_last_reason": "ok",
            "truth_backfill_last_artifact": "truth.json",
            "truth_backfill_last_validation": None,
        },
    }
    payload = summarize([
        {"id": "p1", "name": "Review", "tonnage_mt": 10, "grade_value": 1},
        {"id": "p2", "name": "Failed", "tonnage_mt": 10, "grade_value": 1},
        {"id": "p3", "name": "Usable", "tonnage_mt": 10, "grade_value": 1},
        {"id": "p4", "name": "Fresh", "tonnage_mt": 10, "grade_value": 1},
    ], truth_backfill_history=history)

    assert payload["summary"]["truth_backfill_attempted"] == 3
    assert payload["summary"]["truth_backfill_failed"] == 1
    assert payload["summary"]["truth_backfill_review"] == 1
    assert payload["summary"]["truth_backfill_usable_pending_apply"] == 1
    assert payload["summary"]["truth_backfill_unattempted_extraction"] == 1
    assert [row["next_action"] for row in payload["projects"]] == [
        "manual_review_official_split",
        "triage_backfill_failure_or_model_after_accuracy_gate",
        "apply_verified_backfill_truth",
        "run_backfill_gold_mre_truth",
    ]


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
