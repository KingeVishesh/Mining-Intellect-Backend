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


def test_accepts_pre_cutoff_drilling_release_ahead_of_maiden_resource():
    ok, reasons = validate_pre_mre_evidence(_evidence(
        "average_intercept_grade_gpt",
        1.64,
        source_title="Further Strong Drilling Results at Mandilla Ahead of Maiden Mineral Resource",
        source_url="https://example.com/mandilla-drilling-ahead-of-maiden-resource.pdf",
        source_date=date(2024, 12, 1),
        confidence="medium",
    ))

    assert ok is True
    assert reasons == []


def test_accepts_pre_cutoff_jorc_exploration_target_context():
    ok, reasons = validate_pre_mre_evidence(_evidence(
        "geometry_tonnage_mt",
        3.3,
        source_title="Bulk Sampling of High Grade Reef for JORC Resource Definition",
        source_url="https://example.com/revere-resource-definition.pdf",
        source_date=date(2024, 10, 5),
        confidence="medium",
        fact_payload={
            "notes": "JORC Exploration Target of 2.5-4.1 Mt at 1-2.5 g/t Au; not a mineral resource estimate.",
            "rejected_sources": [{"source_title": "Maiden Mineral Resource Estimate"}],
        },
    ))

    assert ok is True
    assert reasons == []


def test_accepts_pre_cutoff_drilling_fact_with_rejected_mre_context_in_payload():
    ok, reasons = validate_pre_mre_evidence(_evidence(
        "total_drill_meters",
        8539,
        source_title="Landore Resources: Progress Report BAM East Gold Prospect Junior Lake Property",
        source_url="https://example.com/bam-progress-report",
        source_date=date(2017, 1, 12),
        cutoff_date=date(2017, 1, 27),
        confidence="medium",
        fact_payload={
            "notes": (
                "PRIMARY SOURCE: pre-MRE drilling progress report with 44 diamond "
                "drill holes for 8,539m. MAIDEN MRE CONTEXT: post-cutoff resource "
                "estimate excluded."
            ),
            "rejected_sources": [
                {"source_title": "Updated Mineral Resource Estimate - BAM East"}
            ],
        },
    ))

    assert ok is True
    assert reasons == []


def test_accepts_pre_cutoff_fact_when_notes_only_mention_excluded_first_mre_context():
    ok, reasons = validate_pre_mre_evidence(_evidence(
        "total_drill_meters",
        50000,
        source_title="Sunward intercepts gold and strong copper mineralization at Titiribi",
        source_url="https://example.com/titiribi-drilling",
        source_date=date(2011, 7, 25),
        cutoff_date=date(2011, 9, 8),
        confidence="medium",
        fact_payload={
            "notes": (
                "Pre-MRE exploration evidence from verified pre-cutoff press releases. "
                "TOTAL METERS DRILLED: Over 50,000 m as of July 25, 2011. "
                "No bulk density, weighted grade, or non-MRE tonnage estimates were found. "
                "The first NI 43-101 MRE was dated May 19, 2010, but contains "
                "MRE data that cannot be quoted."
            ),
        },
    ))

    assert ok is True
    assert reasons == []


def test_accepts_pre_cutoff_fact_when_notes_only_mention_excluded_post_mre_sources():
    ok, reasons = validate_pre_mre_evidence(_evidence(
        "drill_holes",
        16,
        source_title="Omai Gold Announces Final Wenot 2021 Drill Results",
        source_url="https://example.com/omai-2021-drilling",
        source_date=date(2021, 12, 8),
        cutoff_date=date(2022, 1, 4),
        confidence="medium",
        fact_payload={
            "notes": (
                "PRIMARY SOURCE: Dec 8, 2021 press release before the initial MRE cutoff. "
                "The program comprised 16 diamond drill holes totalling 8,181 m. "
                "Post-MRE sources state bulk density values, but those are excluded."
            ),
        },
    ))

    assert ok is True
    assert reasons == []


def test_accepts_pre_cutoff_fact_when_notes_audit_rejected_post_cutoff_sources():
    ok, reasons = validate_pre_mre_evidence(_evidence(
        "total_drill_meters",
        45600,
        source_title="Teuton expands Goldstorm mineralization along the Northeast Axis",
        source_url="https://example.com/treaty-creek-2020-drilling",
        source_date=date(2020, 9, 10),
        cutoff_date=date(2021, 3, 1),
        confidence="low",
        fact_payload={
            "notes": (
                "Pre-MRE evidence for Treaty Creek. STRIKE LENGTH: 1,100 m along "
                "the northeast axis, stated in the September 10, 2020 release. "
                "BULK DENSITY: 2.80 t/m3 appears in the April 26, 2021 NI 43-101 "
                "Technical Report (POST-CUTOFF) and was rejected. "
                "Current website references 2026 MRE data and is excluded."
            ),
            "rejected_sources": [
                {"source_title": "Initial Mineral Resource Estimate", "source_date": "2021-04-26"}
            ],
        },
    ))

    assert ok is True
    assert reasons == []


def test_rejects_pre_cutoff_fact_when_payload_itself_quotes_resource_estimate():
    ok, reasons = validate_pre_mre_evidence(_evidence(
        "total_drill_meters",
        8539,
        source_title="BAM East Gold Project",
        source_url="https://example.com/bam-resource-summary",
        source_date=date(2017, 1, 12),
        cutoff_date=date(2017, 1, 27),
        confidence="medium",
        fact_payload={
            "notes": "The same source quotes a NI 43-101 mineral resource estimate.",
        },
    ))

    assert ok is False
    assert "mre_tainted_source" in reasons


def test_no_prediction_when_only_anchor_is_exploration_target_midpoint():
    target_context = {
        "source_title": "Bulk Sampling of High Grade Reef for JORC Resource Definition",
        "source_url": "https://example.com/revere-resource-definition.pdf",
        "source_date": date(2024, 10, 5),
        "confidence": "medium",
        "fact_payload": {
            "notes": "JORC Exploration Target of 2.5-4.1 Mt at 1-2.5 g/t Au; not a mineral resource estimate.",
        },
    }

    prediction = predict_gold_resource(
        _project(tectonic_belt="yilgarn"),
        [
            _evidence("geometry_tonnage_mt", 3.3, **target_context),
            _evidence("grade_proxy_gpt", 1.75, **target_context),
        ],
        [
            _analog("Analog A", tonnage=3.1, grade=1.6, candidate_tectonic_belt="yilgarn"),
            _analog("Analog B", tonnage=3.5, grade=1.7, candidate_tectonic_belt="yilgarn"),
            _analog("Analog C", tonnage=4.0, grade=1.5, candidate_tectonic_belt="yilgarn"),
        ],
        cutoff_date=CUTOFF,
    )

    assert prediction["run_status"] == "no_prediction"
    assert "exploration_target_tonnage_anchor_insufficient" in prediction["no_prediction_reasons"]
    assert "exploration_target_grade_anchor_insufficient" in prediction["no_prediction_reasons"]
    assert prediction["calculator_trace"]["accepted_evidence_count"] == 2
    assert prediction["calculator_trace"]["evidence_tonnage"]["rejected_anchor_facts"][0]["fact_type"] == "geometry_tonnage_mt"
    assert prediction["calculator_trace"]["evidence_grade"]["rejected_anchor_facts"][0]["fact_type"] == "grade_proxy_gpt"


def test_rejects_pre_cutoff_jorc_resource_estimate_context():
    ok, reasons = validate_pre_mre_evidence(_evidence(
        "geometry_tonnage_mt",
        3.3,
        source_title="JORC Mineral Resource Estimate",
        source_url="https://example.com/jorc-mineral-resource-estimate.pdf",
        source_date=date(2024, 10, 5),
    ))

    assert ok is False
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
