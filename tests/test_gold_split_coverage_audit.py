from __future__ import annotations

from scripts.audit_gold_split_coverage import classify_gold_project, summarize


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
