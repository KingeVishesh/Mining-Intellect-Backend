from __future__ import annotations

from scripts.gold_backtest_diagnostics import (
    classify_failure,
    evidence_quality_score,
    extract_local_guards,
    leaderboard_row,
)


def test_evidence_quality_scores_pre_mre_drilling_geometry_and_grade():
    score = evidence_quality_score({
        "queried_pre_mre_cutoff": "2024-12-31",
        "source_url": "https://example.com/pre-mre-release",
        "confidence": "medium",
        "total_meters_drilled": 12_000,
        "weighted_grade_g_t": 1.5,
        "strike_length_m": 800,
        "best_intercepts": [{"hole_id": "A"}],
    })

    assert score["grade"] == "high"
    assert "meters" in score["signals"]
    assert "grade_proxy" in score["signals"]
    assert "geometry" in score["signals"]


def test_classifies_large_low_grade_irgs_under_tonnage():
    failure = classify_failure(
        errors={"tonnage": -0.52, "grade": 0.02, "contained": -0.51},
        project={"deposit_subtype": "irgs_general"},
        model={"methodology": {"notes": "local_guard=parallel_no_result"}},
        evidence_score={"grade": "none"},
    )

    assert failure["class"] == "under_tonnage_large_low_grade_irgs"
    assert "large-system analog prior" in failure["lesson"]
    assert failure["guards"] == ["parallel_no_result"]


def test_extracts_multiple_local_guards():
    guards = extract_local_guards({
        "methodology": {
            "notes": (
                "local_guard=parallel_no_result; "
                "local_guard=low_grade_geometry_tonnage_proxy"
            )
        }
    })

    assert guards == ["parallel_no_result", "low_grade_geometry_tonnage_proxy"]


def test_leaderboard_row_is_machine_readable():
    row = leaderboard_row(
        project_name="Test Gold",
        errors={"tonnage": 0.0123, "grade": -0.02, "contained": -0.01},
        passed=True,
        failure={"class": "core_pass", "lesson": "ok"},
        guards=["parallel_no_result"],
    )

    assert row["project"] == "Test Gold"
    assert row["pass"] is True
    assert row["tonnage_error_pct"] == 1.23
    assert row["failure_class"] == "core_pass"
