from __future__ import annotations

from scripts.backfill_gold_mre_truth import (
    _candidate_status,
    _candidate_priority,
    _normalise_extracted_tonnage_units,
    _reviewable_total_mismatch,
    _resource_category_kind,
    _validate_against_known_total,
)


def _row(**overrides):
    base = {
        "id": "p1",
        "name": "Partial Gold",
        "material": "Gold",
        "tonnage_mt": 30.0,
        "grade_value": 1.5,
        "mre_mi_tonnage_mt": None,
        "mre_mi_grade": None,
        "mre_inferred_tonnage_mt": None,
        "mre_inferred_grade": None,
    }
    base.update(overrides)
    return base


def test_db_gold_candidate_requires_missing_truth_and_known_total():
    ok, reason = _candidate_status(_row())

    assert ok is True
    assert "mre_mi_tonnage_mt" in reason


def test_db_gold_candidate_skips_full_truth_unless_forced():
    full = _row(
        mre_mi_tonnage_mt=10.0,
        mre_mi_grade=1.8,
        mre_inferred_tonnage_mt=20.0,
        mre_inferred_grade=1.35,
    )

    ok, reason = _candidate_status(full)
    forced_ok, forced_reason = _candidate_status(full, include_full_truth=True)

    assert ok is False
    assert "already" in reason
    assert forced_ok is True
    assert forced_reason == "full truth included by --force"


def test_db_gold_candidate_skips_rows_without_known_total_by_default():
    ok, reason = _candidate_status(_row(tonnage_mt=None))
    relaxed_ok, _ = _candidate_status(_row(tonnage_mt=None), require_known_total=False)

    assert ok is False
    assert "known total" in reason
    assert relaxed_ok is True


def test_resource_category_kind_detects_full_split_disclosures():
    assert _resource_category_kind(_row(resource_category="Measured + Indicated + Inferred")) == "mi_and_inferred"
    assert _resource_category_kind(_row(resource_category="Indicated and Inferred")) == "mi_and_inferred"
    assert _resource_category_kind(_row(resource_category="Inferred")) == "inferred_only"
    assert _resource_category_kind(_row(resource_category="Measured + Indicated")) == "mi_only"
    assert _resource_category_kind(_row(resource_category=None)) == "unknown"


def test_candidate_priority_puts_likely_full_split_rows_first():
    rows = [
        _row(id="unknown", name="Unknown", resource_category=None),
        _row(id="inferred", name="Inferred", resource_category="Inferred"),
        _row(id="mi", name="MI", resource_category="Measured + Indicated"),
        _row(id="split", name="Split", resource_category="Indicated + Inferred"),
    ]

    ordered = sorted(rows, key=_candidate_priority)

    assert [row["id"] for row in ordered] == ["split", "unknown", "mi", "inferred"]


def test_extracted_split_must_reconcile_to_known_total():
    row = _row(tonnage_mt=30.0, grade_value=1.5)
    extracted = {
        "mi_tonnage_mt": 10.0,
        "mi_grade": 1.8,
        "inferred_tonnage_mt": 20.0,
        "inferred_grade": 1.35,
    }

    ok, reason = _validate_against_known_total(row, extracted)

    assert ok is True
    assert "cross-check ok" in reason


def test_extracted_split_rejects_known_total_mismatch():
    row = _row(tonnage_mt=30.0, grade_value=1.5)
    extracted = {
        "mi_tonnage_mt": 40.0,
        "mi_grade": 1.8,
        "inferred_tonnage_mt": 20.0,
        "inferred_grade": 1.35,
    }

    ok, reason = _validate_against_known_total(row, extracted)

    assert ok is False
    assert "tonnage differs" in reason


def test_complete_sane_split_with_total_mismatch_is_reviewable():
    row = _row(tonnage_mt=30.0, grade_value=1.5)
    extracted = {
        "mi_tonnage_mt": 40.0,
        "mi_grade": 1.8,
        "inferred_tonnage_mt": 20.0,
        "inferred_grade": 1.35,
    }
    ok, reason = _validate_against_known_total(row, extracted)

    assert ok is False
    assert _reviewable_total_mismatch(row, extracted, reason)


def test_extracted_split_normalises_kt_tonnage_only_when_total_matches():
    row = _row(tonnage_mt=5.224, grade_value=8.74)
    extracted = {
        "mi_tonnage_mt": 1177.0,
        "mi_grade": 8.2,
        "inferred_tonnage_mt": 4047.0,
        "inferred_grade": 8.9,
    }

    normalised, note = _normalise_extracted_tonnage_units(row, extracted)
    ok, reason = _validate_against_known_total(row, normalised)

    assert note is not None
    assert normalised["mi_tonnage_mt"] == 1.177
    assert normalised["inferred_tonnage_mt"] == 4.047
    assert ok is True
    assert "cross-check ok" in reason


def test_extracted_split_normalises_obvious_tonnes_even_when_local_total_is_stale():
    row = _row(tonnage_mt=428.6952, grade_value=1.222)
    extracted = {
        "mi_tonnage_mt": 431949000.0,
        "mi_grade": 1.24,
        "inferred_tonnage_mt": 357614000.0,
        "inferred_grade": 1.04,
    }

    normalised, note = _normalise_extracted_tonnage_units(row, extracted)
    strict_ok, strict_reason = _validate_against_known_total(row, normalised)
    relaxed_ok, relaxed_reason = _validate_against_known_total(
        row,
        normalised,
        allow_total_mismatch=True,
    )

    assert note == "normalised extracted tonnage from tonnes to Mt"
    assert normalised["mi_tonnage_mt"] == 431.949
    assert normalised["inferred_tonnage_mt"] == 357.614
    assert strict_ok is False
    assert "tonnage differs" in strict_reason
    assert relaxed_ok is True
    assert "accepted despite local total mismatch" in relaxed_reason
