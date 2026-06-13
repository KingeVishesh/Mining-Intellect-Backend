from __future__ import annotations

from datetime import date

from scripts.run_gold_resource_backtest_v2 import (
    analog_candidate_row,
    build_truth_row,
    decision_rows_for_candidates,
    evidence_rows_from_payload,
)
from scripts.run_gold_resource_predictor_v2 import _audit_summary, _prediction_run_row


def test_truth_builder_selects_earliest_validated_full_split_mre():
    project = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "First MRE Gold",
        "mre_mi_tonnage_mt": 10,
        "mre_mi_grade": 2,
        "mre_inferred_tonnage_mt": 20,
        "mre_inferred_grade": 1,
    }
    truth, reason = build_truth_row(project, [
        {
            "id": "later",
            "effective_date": "2024-01-01",
            "mi_tonnage_mt": 12,
            "mi_grade": 2,
            "inferred_tonnage_mt": 24,
            "inferred_grade": 1,
            "source_url": "https://example.com/later",
        },
        {
            "id": "first",
            "effective_date": "2023-01-01",
            "mi_tonnage_mt": 10,
            "mi_grade": 2,
            "inferred_tonnage_mt": 20,
            "inferred_grade": 1,
            "source_url": "https://example.com/first",
        },
    ])

    assert reason is None
    assert truth is not None
    assert truth["publication_date"] == date(2023, 1, 1)
    assert truth["mi_tonnage_mt"] == 10
    assert truth["inferred_tonnage_mt"] == 20
    assert "cutoff_date" not in truth


def test_evidence_builder_stores_rejected_payloads_without_post_cutoff_source_date():
    rows = evidence_rows_from_payload(
        project_id="project-1",
        truth_id="truth-1",
        cutoff_date=date(2024, 1, 1),
        evidence={
            "source_url": "https://example.com/2024-resource-update",
            "source_date": "2024-02-01",
            "weighted_grade_g_t": 1.2,
            "confidence": "high",
        },
    )

    assert len(rows) == 1
    assert rows[0]["evidence_status"] == "rejected"
    assert rows[0]["source_date"] is None
    assert rows[0]["fact_payload"]["rejected_source_date"] == "2024-02-01"


def test_evidence_builder_normalizes_legacy_confidence_objects():
    rows = evidence_rows_from_payload(
        project_id="project-1",
        truth_id="truth-1",
        cutoff_date=date(2024, 1, 1),
        evidence={
            "source_url": "https://example.com/2023-drilling",
            "source_date": "2023-02-01",
            "weighted_grade_g_t": 1.2,
            "confidence": {"level": "High"},
        },
    )

    assert len(rows) == 1
    assert rows[0]["confidence"] == "high"


def test_analog_candidate_derives_mi_split_from_total_and_inferred():
    row = analog_candidate_row("target-1", {
        "analog_name": "Split Analog",
        "analog_tonnage_mt": 30,
        "analog_grade_value": 2,
        "analog_inferred_tonnage_mt": 10,
        "analog_inferred_grade": 1,
        "analog_resource_vintage_year": 2022,
        "analog_resource_compliance_standard": "ni_43_101",
        "source_url": "https://example.com/split-analog",
    })

    assert row is not None
    assert row["mi_tonnage_mt"] == 20
    assert row["mi_grade_gpt"] == 2.5
    assert row["source_date"] == date(2022, 12, 31)


def test_decision_builder_rejects_analogs_when_target_evidence_missing():
    project = {
        "id": "target-1",
        "project_name": "Target",
        "deposit_subtype": "orogenic_general",
        "tectonic_belt": "abitibi",
        "mining_method_class": "open_pit_bulk",
        "project_stage_class": "exploration",
    }
    candidate = {
        "id": "analog-1",
        "target_project_id": "target-1",
        "candidate_project_name": "Analog",
        "source_url": "https://example.com/analog",
        "source_date": date(2020, 1, 1),
        "resource_standard": "ni_43_101",
        "total_tonnage_mt": 10,
        "total_grade_gpt": 1,
    }

    decisions = decision_rows_for_candidates(project, [], [candidate], cutoff_date=date(2024, 1, 1))

    assert decisions[0]["decision"] == "rejected"
    assert "target_missing_pre_mre_tonnage_proxy" in decisions[0]["rejection_reasons"]
    assert "target_missing_pre_mre_grade_proxy" in decisions[0]["rejection_reasons"]


def test_replay_audit_summary_reports_rejected_evidence_and_analog_reasons():
    bundle = {
        "all_evidence": [
            {"evidence_status": "accepted", "fact_type": "weighted_grade_gpt"},
            {
                "evidence_status": "rejected",
                "fact_type": "strike_length_m",
                "rejection_reason": "mre_tainted_source;low_confidence_weak_fact",
            },
        ],
        "rejected_evidence": [
            {
                "evidence_status": "rejected",
                "fact_type": "strike_length_m",
                "rejection_reason": "mre_tainted_source;low_confidence_weak_fact",
            },
        ],
        "analog_candidates": [{"id": "analog-1"}],
        "analog_decisions": [
            {
                "id": "decision-1",
                "decision": "rejected",
                "rejection_reasons": ["target_missing_pre_mre_tonnage_proxy"],
            }
        ],
    }

    audit = _audit_summary(bundle)

    assert audit["evidence"]["accepted_count"] == 1
    assert audit["evidence"]["rejected_count"] == 1
    assert audit["evidence"]["rejection_reasons"]["mre_tainted_source"] == 1
    assert audit["analogs"]["candidate_count"] == 1
    assert audit["analogs"]["decision_counts"]["rejected"] == 1
    assert audit["analogs"]["rejection_reasons"]["target_missing_pre_mre_tonnage_proxy"] == 1


def test_replay_prediction_row_persists_existing_analog_decision_ids():
    row = _prediction_run_row(
        "project-1",
        {"id": "truth-1"},
        {
            "run_status": "no_prediction",
            "input_hash": "hash-1",
            "cutoff_date": "2024-01-01",
            "predictor_version": "test",
            "no_prediction_reasons": ["insufficient_pre_mre_tonnage_evidence"],
            "calculator_trace": {},
        },
        [
            {
                "id": "decision-1",
                "decision": "rejected",
                "analog_candidate_id": "analog-1",
            },
            {
                "id": "decision-2",
                "decision": "accepted",
                "analog_candidate_id": "analog-2",
            },
        ],
    )

    assert row["analog_decision_ids"] == ["decision-1", "decision-2"]
    assert row["analog_candidate_ids"] == ["analog-2"]
