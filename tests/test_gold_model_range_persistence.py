from __future__ import annotations

import sys
from types import SimpleNamespace

sys.modules.setdefault("nodes.pdf_generator", SimpleNamespace())

from graphs.gold_model_builder import (
    _fields_from_parallel,
    _resource_range_rows_from_parallel,
    _source_rows_from_parallel,
)


def test_gold_parallel_ranges_feed_model_run_percentiles_and_range_rows():
    parallel_out = {
        "m_and_i": {
            "tonnage_mt": 10,
            "grade_gpt": 1.5,
            "contained_moz": 0.482,
            "tonnage_range_mt": {"p10": 5, "p50": 10, "p90": 20},
            "grade_range_gpt": {"p10": 1.0, "p50": 1.5, "p90": 2.0},
            "contained_range_moz": {"p10": 0.2, "p50": 0.482, "p90": 1.0},
        },
        "inferred": {
            "tonnage_mt": 4,
            "grade_gpt": 1.0,
            "contained_moz": 0.129,
            "tonnage_range_mt": {"p10": 2, "p50": 4, "p90": 8},
            "grade_range_gpt": {"p10": 0.7, "p50": 1.0, "p90": 1.4},
            "contained_range_moz": {"p10": 0.05, "p50": 0.129, "p90": 0.35},
        },
        "conviction": {"level": "medium", "rationale": "range test"},
        "sources_used": [],
        "sources_rejected": [],
    }

    fields = _fields_from_parallel({"material": "gold"}, parallel_out)
    rows = _resource_range_rows_from_parallel(parallel_out)

    assert fields["mi_tonnage_mt"] == 10
    assert fields["p10_tonnage_mt"] == 7
    assert fields["p50_tonnage_mt"] == 14
    assert fields["p90_tonnage_mt"] == 28
    assert fields["p50_contained"] == 611000
    assert any(
        row["resource_category"] == "m_and_i"
        and row["metric"] == "contained_moz"
        and row["p90"] == 1
        for row in rows
    )
    assert any(
        row["resource_category"] == "total"
        and row["metric"] == "tonnage_mt"
        and row["p50"] == 14
        for row in rows
    )


def test_gold_source_rows_include_parallel_and_target_evidence_sources():
    project = {
        "drilling_evidence": {
            "source_url": "https://example.com/pre-mre-drilling",
            "source_date": "2024-01-15",
            "confidence": "high",
        }
    }
    parallel_out = {
        "sources_used": [{
            "role": "analog_resource_source",
            "used_for": ["range_calibration"],
            "title": "Analog technical report",
            "url": "https://example.com/analog-report",
            "summary": "M&I and inferred resource statement.",
            "confidence": "medium",
        }],
        "analogs_used": ["Analog A | 0.8 | https://example.com/analog-report | ratios"],
    }

    rows = _source_rows_from_parallel(project, [], parallel_out)

    assert any(row["url"] == "https://example.com/analog-report" for row in rows)
    assert any(row["role"] == "target_pre_mre_evidence" for row in rows)
