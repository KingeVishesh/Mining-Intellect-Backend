"""Tests for the drilling-evidence signal added to Model 1 v2 (P3).

Covers:
  - _drilling_signal returns (None, None, audit) when project drilling is missing
  - Returns a T-signal when ≥2 analogs have drilling ratios
  - tonnage_per_meter geomean math
  - Grade signal scaling with intercept count
  - build_model_1 fuses the drilling signal correctly into the joint posterior
  - drilling_extractor.should_refetch staleness behaviour
"""
from __future__ import annotations
import math
from datetime import datetime, timezone, timedelta

import pytest

from nodes import drilling_extractor
from nodes.model_builder import _drilling_signal, build_model_1


# ── staleness ────────────────────────────────────────────────────────────────

class TestShouldRefetch:
    def test_refetch_when_missing(self):
        assert drilling_extractor.should_refetch(None, None) is True

    def test_refetch_when_force(self):
        ev = {"total_holes": 50}
        ts = datetime.now(timezone.utc).isoformat()
        assert drilling_extractor.should_refetch(ev, ts, force=True) is True

    def test_use_cache_when_fresh(self):
        ev = {"total_holes": 50}
        ts = datetime.now(timezone.utc).isoformat()
        assert drilling_extractor.should_refetch(ev, ts, max_age_days=7) is False

    def test_refetch_when_stale(self):
        ev = {"total_holes": 50}
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        assert drilling_extractor.should_refetch(ev, old, max_age_days=7) is True

    def test_refetch_when_timestamp_malformed(self):
        # If the timestamp can't be parsed, we should refetch.
        assert drilling_extractor.should_refetch({"total_holes": 50}, "garbage") is True


# ── _drilling_signal math ────────────────────────────────────────────────────

class TestDrillingSignal:
    def test_returns_none_without_project_drilling(self):
        T, G, audit = _drilling_signal(
            project_drilling=None,
            analog_drillings=[{"total_meters_drilled": 10000}],
            analog_tonnages_mt=[50],
            analog_grades=[1.5],
            weights=[10000],
        )
        assert T is None and G is None
        assert audit["applied"] is False

    def test_returns_none_when_project_meters_zero(self):
        T, G, _ = _drilling_signal(
            project_drilling={"total_meters_drilled": 0},
            analog_drillings=[{"total_meters_drilled": 10000}],
            analog_tonnages_mt=[50],
            analog_grades=[1.5],
            weights=[10000],
        )
        assert T is None

    def test_tonnage_per_meter_geomean_two_analogs(self):
        # Two analogs: 50 Mt @ 10,000 m → 0.005 Mt/m
        #              80 Mt @ 20,000 m → 0.004 Mt/m
        # Geomean of {0.005, 0.004} = sqrt(0.005 × 0.004) ≈ 0.004472 Mt/m
        # Project: 15,000 m → 15000 × 0.004472 ≈ 67.08 Mt
        T, G, audit = _drilling_signal(
            project_drilling={"total_meters_drilled": 15000},
            analog_drillings=[
                {"total_meters_drilled": 10000},
                {"total_meters_drilled": 20000},
            ],
            analog_tonnages_mt=[50.0, 80.0],
            analog_grades=[1.5, 1.5],
            weights=[10000, 10000],
        )
        assert T is not None
        mu_logT, sigma_logT = T
        predicted_mt = math.exp(mu_logT)
        assert predicted_mt == pytest.approx(67.08, rel=0.02)
        # σ should be at floor (0.20) since the two ratios are close
        assert sigma_logT >= 0.20
        assert audit["applied"] is True
        assert audit["n_analogs_with_drilling"] == 2

    def test_skips_analogs_with_missing_drilling(self):
        T, _, audit = _drilling_signal(
            project_drilling={"total_meters_drilled": 10000},
            analog_drillings=[
                {"total_meters_drilled": 5000},
                None,
                {"total_meters_drilled": 0},  # zero is invalid
                {"total_meters_drilled": 8000},
            ],
            analog_tonnages_mt=[10.0, 20.0, 30.0, 16.0],
            analog_grades=[2.0, 2.5, 3.0, 2.2],
            weights=[10000, 10000, 10000, 10000],
        )
        assert audit["n_analogs_with_drilling"] == 2
        assert T is not None

    def test_single_analog_uses_conservative_sigma(self):
        T, _, audit = _drilling_signal(
            project_drilling={"total_meters_drilled": 10000},
            analog_drillings=[{"total_meters_drilled": 5000}],
            analog_tonnages_mt=[10.0],
            analog_grades=[2.0],
            weights=[10000],
        )
        assert T is not None
        _, sigma_T = T
        assert sigma_T >= 0.50  # single-analog conservative σ
        assert audit["n_analogs_with_drilling"] == 1

    def test_grade_signal_from_project_weighted_intercept(self):
        _, G, _ = _drilling_signal(
            project_drilling={
                "total_meters_drilled": 10000,
                "weighted_grade_g_t": 4.8,
                "best_intercepts": [
                    {"interval_m": 10, "grade_g_t": 4.0},
                    {"interval_m": 15, "grade_g_t": 5.2},
                    {"interval_m": 8,  "grade_g_t": 4.5},
                ],
            },
            analog_drillings=[],
            analog_tonnages_mt=[],
            analog_grades=[],
            weights=[],
        )
        assert G is not None
        mu_G, sigma_G = G
        assert math.exp(mu_G) == pytest.approx(4.8, rel=0.001)
        # σ should tighten as we add intercepts; with 3 intercepts, σ ≈ 0.30/√3 ≈ 0.173
        assert sigma_G < 0.30

    def test_no_grade_signal_without_weighted_grade(self):
        _, G, _ = _drilling_signal(
            project_drilling={
                "total_meters_drilled": 10000,
                "weighted_grade_g_t": None,
                "best_intercepts": [],
            },
            analog_drillings=[],
            analog_tonnages_mt=[],
            analog_grades=[],
            weights=[],
        )
        assert G is None


# ── Integration with build_model_1 ───────────────────────────────────────────

def _analog(name, t, g, similarity=80.0, drilling=None):
    return {
        "name": name,
        "tonnage_mt": t,
        "grade_value": g,
        "similarity_score": similarity,
        "project_stage": "production",
        "source": "library",
        "deposit_type": "orogenic gold",
        "mineralization_pattern": "vein_hosted",
        "drilling_evidence": drilling,
    }


class TestBuildModel1WithDrilling:
    def test_drilling_signal_applied_when_agrees_with_analog(self):
        # When the drilling-derived tonnage agrees with the analog pool's
        # central tendency (within 2σ), the consistency check passes and
        # the signal tightens the posterior. Analog pool: 4 vein-orogenic
        # at 10-15 Mt, each with ~5000 m drilled (3000 t/m ratio). Project
        # at 4500 m → drilling predicts ~13.5 Mt — matches analog mean.
        analogs = [
            _analog("A1", 12, 5.0, similarity=85,
                    drilling={"total_meters_drilled": 4000,
                              "weighted_grade_g_t": 5.0,
                              "best_intercepts": [
                                  {"interval_m": 5, "grade_g_t": 5.0},
                                  {"interval_m": 4, "grade_g_t": 5.1},
                                  {"interval_m": 6, "grade_g_t": 4.9},
                              ]}),
            _analog("A2", 14, 5.2, similarity=85,
                    drilling={"total_meters_drilled": 4800,
                              "weighted_grade_g_t": 5.2,
                              "best_intercepts": [
                                  {"interval_m": 6, "grade_g_t": 5.2},
                                  {"interval_m": 5, "grade_g_t": 5.3},
                                  {"interval_m": 4, "grade_g_t": 5.1},
                              ]}),
            _analog("A3", 11, 4.8, similarity=85,
                    drilling={"total_meters_drilled": 3800,
                              "weighted_grade_g_t": 4.8,
                              "best_intercepts": [
                                  {"interval_m": 5, "grade_g_t": 4.8},
                                  {"interval_m": 4, "grade_g_t": 4.7},
                                  {"interval_m": 6, "grade_g_t": 4.9},
                              ]}),
            _analog("A4", 13, 5.1, similarity=85,
                    drilling={"total_meters_drilled": 4500,
                              "weighted_grade_g_t": 5.1,
                              "best_intercepts": [
                                  {"interval_m": 5, "grade_g_t": 5.1},
                                  {"interval_m": 4, "grade_g_t": 5.0},
                                  {"interval_m": 6, "grade_g_t": 5.2},
                              ]}),
        ]
        project = {"id": "p", "name": "T", "material": "gold",
                   "deposit_type": "orogenic gold",
                   "mineralization_pattern": "vein_hosted",
                   "project_stage": "production",
                   "drilling_evidence": {
                       "total_meters_drilled": 4500,
                       "weighted_grade_g_t": 5.1,
                       "best_intercepts": [
                           {"interval_m": 6, "grade_g_t": 5.0},
                           {"interval_m": 5, "grade_g_t": 5.2},
                           {"interval_m": 4, "grade_g_t": 5.1},
                       ],
                   }}
        out = build_model_1(analogs, project, {})
        drill = out["signal_contributions"]["drilling"]
        assert drill["audit"]["applied"] is True
        assert drill["audit"]["n_analogs_with_drilling"] == 4
        # T-signal should pass the consistency check and be applied
        assert drill["T_signal"]["applied"] is True

    def test_drilling_signal_dropped_when_disagrees_with_analog(self):
        # Project meters way above analog meters → drilling-predicted
        # tonnage is ~5× the analog mean → consistency check drops it.
        # This protects against the Cadillac-style failure where mature-
        # mine cumulative drilling produces a wildly different ratio.
        analogs = [
            _analog("A1", 12, 5.0, similarity=85,
                    drilling={"total_meters_drilled": 200_000,
                              "weighted_grade_g_t": 5.0,
                              "best_intercepts": [{"interval_m": 5, "grade_g_t": 5.0}]}),
        ] * 4
        project = {"id": "p", "name": "T", "material": "gold",
                   "deposit_type": "orogenic gold",
                   "mineralization_pattern": "vein_hosted",
                   "project_stage": "production",
                   "drilling_evidence": {"total_meters_drilled": 30_000}}
        out = build_model_1(analogs, project, {})
        drill = out["signal_contributions"]["drilling"]
        # Signal was computed but consistency check rejected it
        assert drill["T_signal"]["applied"] is False
        assert "drilling-signal disagrees" in drill["T_signal"]["reason"]

    def test_drilling_signal_off_when_project_drilling_missing(self):
        # Same pool but project has NO drilling data — signal should not apply
        analogs = [
            _analog("A1", 12, 5.0,
                    drilling={"total_meters_drilled": 4000, "weighted_grade_g_t": 5.0,
                              "best_intercepts": [{"interval_m": 5, "grade_g_t": 5.0}]}),
        ] * 4  # all the same — minimal pool
        project = {"id": "p", "name": "T", "material": "gold",
                   "deposit_type": "orogenic gold",
                   "mineralization_pattern": "vein_hosted",
                   "project_stage": "production"}
        out = build_model_1(analogs, project, {})
        drill_contrib = out["signal_contributions"]["drilling"]
        assert drill_contrib["audit"]["applied"] is False
        assert "T_signal" not in drill_contrib
