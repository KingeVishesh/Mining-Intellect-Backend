from __future__ import annotations

import math
from datetime import date

from nodes.gold_resource_predictor import (
    contained_gold_oz,
    gate_gold_analog,
    predict_gold_resource,
    score_gold_prediction,
    validate_pre_mre_evidence,
)


CUTOFF = date(2025, 1, 1)


def _project(**overrides):
    base = {
        "project_name": "Strict Gold",
        "deposit_subtype": "greenstone_orogenic",
        "tectonic_belt": "abitibi",
        "mining_method_class": "open_pit_bulk",
        "project_stage_class": "exploration",
    }
    base.update(overrides)
    return base


def _evidence(fact_type: str, value_num: float, **overrides):
    base = {
        "cutoff_date": CUTOFF,
        "source_url": "https://example.com/drilling-results-2024",
        "source_date": date(2024, 6, 1),
        "fact_type": fact_type,
        "value_num": value_num,
        "confidence": "high",
    }
    base.update(overrides)
    return base


def _analog(name: str, tonnage: float = 50.0, grade: float = 1.5, **overrides):
    base = {
        "candidate_project_name": name,
        "candidate_deposit_subtype": "greenstone_orogenic",
        "candidate_tectonic_belt": "abitibi",
        "candidate_mining_method_class": "open_pit_bulk",
        "candidate_project_stage_class": "resource_m_and_i",
        "source_url": f"https://example.com/{name.lower().replace(' ', '-')}-resource",
        "source_date": date(2024, 1, 1),
        "resource_standard": "ni_43_101",
        "total_tonnage_mt": tonnage,
        "total_grade_gpt": grade,
        "mi_tonnage_mt": tonnage / 2,
        "mi_grade_gpt": grade,
        "inferred_tonnage_mt": tonnage / 2,
        "inferred_grade_gpt": grade,
    }
    base.update(overrides)
    return base


def test_rejects_post_cutoff_and_mre_tainted_evidence():
    ok, reasons = validate_pre_mre_evidence(_evidence(
        "weighted_grade_gpt",
        1.5,
        source_url="https://example.com/mineral-resource-estimate-technical-report.pdf",
        source_date=date(2025, 1, 1),
    ))

    assert ok is False
    assert "source_not_before_mre_cutoff" in reasons
    assert "mre_tainted_source" in reasons


def test_missing_source_date_blocks_evidence():
    ok, reasons = validate_pre_mre_evidence(_evidence(
        "geometry_tonnage_mt",
        50,
        source_date=None,
    ))

    assert ok is False
    assert reasons == ["missing_source_date"]


def test_bad_analog_is_rejected_for_core_geology_scale_and_mining():
    decision = gate_gold_analog(
        _project(),
        _analog(
            "Bad Analog",
            tonnage=300,
            grade=0.4,
            candidate_deposit_subtype="carlin_general",
            candidate_tectonic_belt="great_basin_carlin",
            candidate_mining_method_class="underground_vein",
        ),
        cutoff_date=CUTOFF,
        target_tonnage_mt=50,
        target_grade_gpt=1.5,
    )

    assert decision["decision"] == "rejected"
    assert "deposit_subtype_mismatch" in decision["rejection_reasons"]
    assert "tectonic_belt_incompatible" in decision["rejection_reasons"]
    assert "mining_method_incompatible" in decision["rejection_reasons"]
    assert "tonnage_band_mismatch" in decision["rejection_reasons"]
    assert "grade_band_mismatch" in decision["rejection_reasons"]


def test_predicts_from_strict_evidence_and_clean_split_ready_analogs():
    prediction = predict_gold_resource(
        _project(),
        [
            _evidence("geometry_tonnage_mt", 50),
            _evidence("weighted_grade_gpt", 1.5),
        ],
        [
            _analog("Analog A", tonnage=50, grade=1.5),
            _analog("Analog B", tonnage=55, grade=1.6),
            _analog("Analog C", tonnage=45, grade=1.4),
        ],
        cutoff_date=CUTOFF,
    )

    assert prediction["run_status"] == "predicted"
    assert prediction["calculator_trace"]["accepted_analog_count"] == 3
    assert math.isclose(prediction["predicted_total_tonnage_mt"], 50.0)
    assert math.isclose(prediction["predicted_total_grade_gpt"], 1.5)

    truth = {
        "project_id": "strict-gold",
        "publication_date": CUTOFF,
        "source_url": "https://example.com/first-resource",
        "mi_tonnage_mt": 25,
        "mi_grade_gpt": 1.5,
        "inferred_tonnage_mt": 25,
        "inferred_grade_gpt": 1.5,
        "total_tonnage_mt": 50,
        "total_grade_gpt": 1.5,
        "total_contained_oz": contained_gold_oz(50, 1.5),
    }
    score = score_gold_prediction(prediction, truth, threshold=0.05)

    assert score["core_pass"] is True
    assert score["split_pass"] is True
    assert score["production_like_pass"] is True


def test_no_prediction_when_clean_analog_cohort_is_too_thin():
    prediction = predict_gold_resource(
        _project(),
        [
            _evidence("geometry_tonnage_mt", 50),
            _evidence("weighted_grade_gpt", 1.5),
        ],
        [
            _analog("Analog A", tonnage=50, grade=1.5),
            _analog("Analog B", tonnage=55, grade=1.6),
        ],
        cutoff_date=CUTOFF,
    )

    assert prediction["run_status"] == "no_prediction"
    assert "insufficient_clean_analog_cohort" in prediction["no_prediction_reasons"]
    assert "insufficient_split_ready_analog_cohort" in prediction["no_prediction_reasons"]


def test_no_prediction_when_geometry_is_incomplete():
    prediction = predict_gold_resource(
        _project(),
        [
            _evidence("strike_length_m", 1000),
            _evidence("down_dip_extent_m", 250),
            _evidence("weighted_grade_gpt", 1.5),
        ],
        [_analog("Analog A"), _analog("Analog B"), _analog("Analog C")],
        cutoff_date=CUTOFF,
    )

    assert prediction["run_status"] == "no_prediction"
    assert prediction["no_prediction_reasons"] == ["insufficient_pre_mre_tonnage_evidence"]


def test_replay_input_hash_is_stable():
    kwargs = {
        "project": _project(),
        "evidence_facts": [
            _evidence("geometry_tonnage_mt", 50),
            _evidence("weighted_grade_gpt", 1.5),
        ],
        "analog_candidates": [
            _analog("Analog A", tonnage=50, grade=1.5),
            _analog("Analog B", tonnage=55, grade=1.6),
            _analog("Analog C", tonnage=45, grade=1.4),
        ],
        "cutoff_date": CUTOFF,
    }

    first = predict_gold_resource(**kwargs)
    second = predict_gold_resource(**kwargs)

    assert first["input_hash"] == second["input_hash"]
