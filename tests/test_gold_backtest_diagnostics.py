from __future__ import annotations

from scripts.gold_backtest_diagnostics import (
    analog_quality_score,
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


def test_analog_quality_scores_coherent_source_backed_cohort_high():
    project = {
        "deposit_subtype": "greenstone_orogenic",
        "tectonic_belt": "abitibi",
        "mining_method_class": "underground_vein",
        "tonnage_mt": 20,
        "grade_value": 4.0,
    }
    analogs = [
        {
            "name": f"Analog {idx}",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "mining_method_class": "underground_vein",
            "tonnage_mt": 15 + idx,
            "grade_value": 3.5,
            "source_url": "https://example.com/report.pdf",
        }
        for idx in range(5)
    ]

    quality = analog_quality_score(project=project, analogs=analogs)

    assert quality["grade"] == "high"
    assert quality["score"] >= 75
    assert quality["resource_backed_count"] == 5


def test_analog_quality_flags_thin_unsourced_cohort_low():
    quality = analog_quality_score(
        project={
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "abitibi",
            "tonnage_mt": 20,
            "grade_value": 4.0,
        },
        analogs=[
            {
                "name": "Weak Analog",
                "deposit_subtype": "carlin_general",
                "tectonic_belt": "great_basin_carlin",
                "tonnage_mt": 250,
                "grade_value": 0.7,
            }
        ],
    )

    assert quality["grade"] in {"low", "reject"}
    assert "thin_analog_count" in quality["flags"]
    assert "weak_source_backing" in quality["flags"]


def test_analog_quality_caps_unknown_core_metadata_below_high():
    quality = analog_quality_score(
        project={
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "guiana_shield",
            "mining_method_class": "open_pit_selective",
            "tonnage_mt": 80,
            "grade_value": 1.1,
        },
        analogs=[
            {
                "name": f"Unknown Metadata Analog {idx}",
                "tonnage_mt": 70 + idx,
                "grade_value": 1.0,
                "source_url": "https://example.com/report.pdf",
            }
            for idx in range(5)
        ],
    )

    assert quality["grade"] == "medium"
    assert "subtype_match_rate_unknown" in quality["flags"]
    assert "belt_match_rate_unknown" in quality["flags"]
    assert "mining_method_match_rate_unknown" in quality["flags"]


def test_analog_quality_uses_pre_mre_grade_proxy_when_mre_grade_absent():
    quality = analog_quality_score(
        project={
            "deposit_subtype": "orogenic_general",
            "tectonic_belt": "abitibi",
            "drilling_evidence": {"weighted_grade_g_t": 1.0},
        },
        analogs=[
            {
                "name": "Low Grade Analog",
                "deposit_subtype": "orogenic_general",
                "tectonic_belt": "abitibi",
                "tonnage_mt": 40,
                "grade_value": 1.2,
                "source_url": "https://example.com/report.pdf",
            }
        ],
    )

    assert quality["metrics"]["grade_band_match_rate"] == 1.0
    assert quality["metrics"]["tonnage_band_match_rate"] is None
