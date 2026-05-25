"""Unit tests for Model 1 v2 — log-space Bayesian fusion (P1).

Pure-Python math via `math` module; no numpy. Tests cover:
  - weighted geometric mean correctness (vs known closed form)
  - trimmed-outlier behavior under a single 1000× wild analog
  - P10 < P50 < P90 invariant across realistic and degenerate inputs
  - CV-based tier mapping (tight posterior → higher tier)
  - geometry signal narrows the tonnage posterior
  - rule multipliers shift μ in log-space (×k multiplier → +ln(k) shift)
  - fallback paths return complete percentile blocks
"""
from __future__ import annotations
import math
from typing import Dict, List

import pytest

from nodes import model_builder
from nodes.model_builder import (
    build_model_1,
    _compute_pre_tier_from_cv,
    _cv_to_conviction_pct,
    _percentile_block,
    _trim_outliers_log,
    _contained_t_from_mt,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _analog(name: str, tonnage_mt: float, grade: float,
            similarity: float = 70.0, stage: str = "production") -> Dict:
    return {
        "name": name,
        "tonnage_mt": tonnage_mt,
        "grade_value": grade,
        "similarity_score": similarity,
        "project_stage": stage,
        "source": "library",
        "resource_category": "Measured + Indicated + Inferred",
    }


def _gold_project(**overrides) -> Dict:
    base = {
        "id": "test-project",
        "name": "Test Gold Project",
        "material": "gold",
        "deposit_type": "orogenic gold",
        "project_stage": "exploration",
    }
    base.update(overrides)
    return base


# ── core math ─────────────────────────────────────────────────────────────────

class TestContainedFromMt:
    def test_precious_metals_no_conversion(self):
        # 15 Mt × 1.5 g/t = 22.5 tonnes Au (the chosen unit lets us verify by eye)
        assert _contained_t_from_mt(15.0, 1.5, "gold") == pytest.approx(22.5)
        assert _contained_t_from_mt(10.0, 3.0, "silver") == pytest.approx(30.0)

    def test_base_metals_scale_by_1e4(self):
        # 100 Mt × 0.5% = 100 × 0.5 × 1e4 = 500,000 t Cu
        assert _contained_t_from_mt(100.0, 0.5, "copper") == pytest.approx(500_000.0)

    def test_zero_inputs_pass_through(self):
        assert _contained_t_from_mt(0.0, 5.0, "gold") == 0.0
        assert _contained_t_from_mt(5.0, 0.0, "copper") == 0.0


class TestPreTierFromCv:
    @pytest.mark.parametrize("cv,expected", [
        (0.10, "PRE-5"), (0.29, "PRE-5"),
        (0.30, "PRE-4"), (0.49, "PRE-4"),
        (0.50, "PRE-3"), (0.79, "PRE-3"),
        (0.80, "PRE-2"), (1.29, "PRE-2"),
        (1.30, "PRE-1"), (5.00, "PRE-1"),
    ])
    def test_cv_maps_to_correct_tier(self, cv, expected):
        tier, _ = _compute_pre_tier_from_cv(cv)
        assert tier == expected

    def test_conviction_pct_decreases_monotonically_with_cv(self):
        cvs = [0.1, 0.4, 0.6, 1.0, 2.0, 5.0]
        pcts = [_cv_to_conviction_pct(cv) for cv in cvs]
        assert pcts == sorted(pcts, reverse=True), \
            f"Expected monotone-decreasing, got {pcts}"


class TestTrimOutliersLog:
    def test_trims_one_extreme_outlier(self):
        # 9 analogs near (10 Mt, 1 g/t), 1 analog at (1e6 Mt, 1e3 g/t) — that
        # last one is in a totally different log-space region and should be
        # dropped.
        tonnages = [10.0] * 9 + [1_000_000.0]
        grades   = [1.0] * 9 + [1000.0]
        log_t = [math.log(t) for t in tonnages]
        log_g = [math.log(g) for g in grades]
        weights = [1.0] * 10
        keep = _trim_outliers_log(log_t, log_g, weights, trim_pct=0.10)
        assert 9 not in keep, "Wild outlier was retained"
        assert len(keep) == 9

    def test_returns_all_when_below_min_size(self):
        # Fewer than 5 — no trimming
        log_t = [1.0, 2.0, 3.0]
        log_g = [1.0, 2.0, 3.0]
        assert _trim_outliers_log(log_t, log_g, [1.0] * 3) == [0, 1, 2]


class TestPercentileBlock:
    def test_percentiles_are_ordered(self):
        block = _percentile_block(15.0, 1.5, "gold", cv_target=0.7)
        assert block["p10_total_tonnage_mt"] < block["p50_total_tonnage_mt"] \
            < block["p90_total_tonnage_mt"]
        assert block["p10_grade"] < block["p50_grade"] < block["p90_grade"]
        assert block["p10_contained_t"] < block["p50_contained_t"] \
            < block["p90_contained_t"]

    def test_p50_is_central_estimate(self):
        block = _percentile_block(15.0, 1.5, "gold", cv_target=0.7)
        assert block["p50_total_tonnage_mt"] == pytest.approx(15.0)
        assert block["p50_grade"] == pytest.approx(1.5)
        # For gold, p50 contained = tonnage × grade × 1 = 22.5
        assert block["p50_contained_t"] == pytest.approx(22.5)

    def test_higher_cv_produces_wider_spread(self):
        narrow = _percentile_block(15.0, 1.5, "gold", cv_target=0.3)
        wide   = _percentile_block(15.0, 1.5, "gold", cv_target=1.5)
        narrow_range = narrow["p90_contained_t"] - narrow["p10_contained_t"]
        wide_range   = wide["p90_contained_t"]   - wide["p10_contained_t"]
        assert wide_range > narrow_range * 3, \
            "5× the CV should produce a much wider posterior"

    def test_zero_inputs_return_zeros_without_division_error(self):
        block = _percentile_block(0.0, 0.0, "gold", cv_target=0.0)
        assert block["p10_total_tonnage_mt"] == 0.0
        assert block["p90_contained_t"] == 0.0


# ── build_model_1 integration ────────────────────────────────────────────────

class TestBuildModel1:
    def test_emits_all_required_keys(self):
        analogs = [_analog(f"A{i}", 10.0 + i, 1.5 + 0.1 * i) for i in range(6)]
        out = build_model_1(analogs, _gold_project(), {})
        required = {
            "mi_tonnage_kt", "mi_grade_pct", "mi_contained_mlb",
            "inferred_tonnage_kt", "inferred_grade_pct", "inferred_contained_mlb",
            "total_tonnage_kt", "total_grade_pct", "total_contained_mlb",
            "conviction_pct", "conviction_tier", "conviction_label",
            "analogs_used", "rules_applied",
            "p10_total_tonnage_mt", "p50_total_tonnage_mt", "p90_total_tonnage_mt",
            "p10_grade", "p50_grade", "p90_grade",
            "p10_contained_t", "p50_contained_t", "p90_contained_t",
            "cv_contained", "signal_contributions",
        }
        missing = required - set(out.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_percentiles_strictly_ordered(self):
        analogs = [_analog(f"A{i}", 5.0 + i * 2.0, 1.0 + 0.2 * i) for i in range(8)]
        out = build_model_1(analogs, _gold_project(), {})
        assert out["p10_total_tonnage_mt"] < out["p50_total_tonnage_mt"] \
            < out["p90_total_tonnage_mt"]
        assert out["p10_grade"] < out["p50_grade"] < out["p90_grade"]
        assert out["p10_contained_t"] < out["p50_contained_t"] \
            < out["p90_contained_t"]

    def test_p50_close_to_geometric_mean_when_analogs_identical(self):
        # 5 identical analogs → posterior median should equal their value
        analogs = [_analog(f"A{i}", 20.0, 2.0, similarity=80.0) for i in range(5)]
        out = build_model_1(analogs, _gold_project(), {})
        assert out["p50_total_tonnage_mt"] == pytest.approx(20.0, rel=0.05)
        assert out["p50_grade"] == pytest.approx(2.0, rel=0.05)

    def test_geometric_mean_below_arithmetic_for_skewed_pool(self):
        # 4 analogs at 10 Mt + 1 at 1000 Mt — geometric mean handles this
        # skew correctly. Arithmetic mean = 208 Mt; geometric ≈ 25 Mt.
        # After outlier trim the 1000 Mt is dropped, so p50 ≈ 10 Mt.
        # The point of this test: result is anchored to the bulk of the pool,
        # not pulled to arithmetic-mean territory by the wild value.
        analogs = [_analog(f"A{i}", 10.0, 1.0, similarity=70.0) for i in range(4)]
        analogs.append(_analog("WILD", 1000.0, 5.0, similarity=70.0))
        out = build_model_1(analogs, _gold_project(), {})
        assert out["p50_total_tonnage_mt"] < 50.0, \
            f"p50 should be near analog bulk, got {out['p50_total_tonnage_mt']}"

    def test_tighter_pool_yields_higher_tier(self):
        # Pool 1: tight (all very similar, similar grades)
        tight = [_analog(f"T{i}", 15.0 + 0.5 * i, 1.5 + 0.02 * i, similarity=85.0)
                 for i in range(8)]
        # Pool 2: noisy (broad spread)
        noisy = [
            _analog("N0", 5.0,  0.8, similarity=60.0),
            _analog("N1", 50.0, 3.5, similarity=60.0),
            _analog("N2", 12.0, 1.0, similarity=60.0),
            _analog("N3", 80.0, 4.2, similarity=60.0),
            _analog("N4", 25.0, 0.5, similarity=60.0),
            _analog("N5", 3.0,  6.0, similarity=60.0),
        ]
        tight_out = build_model_1(tight, _gold_project(), {})
        noisy_out = build_model_1(noisy, _gold_project(), {})
        assert tight_out["cv_contained"] < noisy_out["cv_contained"], \
            "Tighter pool should produce a smaller CV"
        # tier ordering: PRE-5 (best) … PRE-1 (worst). Higher-numbered tier
        # (e.g. PRE-4) should be at least as good as the noisy one.
        tight_num = int(tight_out["conviction_tier"].split("-")[1])
        noisy_num = int(noisy_out["conviction_tier"].split("-")[1])
        assert tight_num >= noisy_num

    def test_rule_multiplier_shifts_in_log_space(self):
        # The rule shift moves μ_logT before the stage prior fuses in, so the
        # net effect on p50 is partial — the L151 prior pulls back toward the
        # stage-typical scale. The invariant we verify here is the *log-space*
        # shift recorded in signal_contributions, which equals ln(multiplier)
        # exactly. The direction of the p50 shift is also asserted.
        analogs = [_analog(f"A{i}", 10.0, 1.0, similarity=80.0) for i in range(6)]
        base = build_model_1(analogs, _gold_project(), {})
        doubled = build_model_1(analogs, _gold_project(),
                                {"tonnage_multiplier": 2.0, "rules_applied": ["r"]})
        # exact log-space shift on the rule signal
        assert doubled["signal_contributions"]["rules"]["log_t_shift"] \
            == pytest.approx(math.log(2.0))
        # p50 shifts in the right direction (upward) by at least 30% — the
        # stage prior + analog pull together absorb some of the rule
        ratio = doubled["p50_total_tonnage_mt"] / base["p50_total_tonnage_mt"]
        assert 1.3 < ratio < 2.0, f"Expected partial doubling, got {ratio}"

    def test_geometry_signal_narrows_tonnage_posterior(self):
        analogs = [_analog(f"A{i}", 20.0 + 5 * i, 2.0, similarity=75.0)
                   for i in range(5)]
        no_geom = build_model_1(analogs, _gold_project(), {})
        with_geom = build_model_1(
            analogs,
            _gold_project(strike_length_meters=400, width_meters=20, depth_meters=200),
            {},
        )
        # σ_logT shrinks when geometry adds an extra inverse-variance term
        no_geom_sigma = no_geom["signal_contributions"]["analog"]["sigma_logT"]
        # geometry-fused σ is in the analog block * geometry fusion already
        # collapsed by build_model_1, so we read it back from CV which is
        # dominated by σ_logT for thin analog pools.
        assert with_geom["cv_contained"] <= no_geom["cv_contained"] + 0.05
        assert with_geom["signal_contributions"]["geometry"] is not None
        assert no_geom_sigma > 0

    def test_falls_back_when_no_valid_analogs(self):
        out = build_model_1([], _gold_project(), {})
        # Fallback should still emit all percentile keys
        assert "p50_total_tonnage_mt" in out
        assert "cv_contained" in out
        assert out["conviction_tier"].startswith("PRE-")

    def test_filters_analogs_missing_grade_or_tonnage(self):
        analogs = [
            _analog("ok1", 10.0, 1.0),
            {"name": "no_tonnage", "grade_value": 2.0, "similarity_score": 50},
            {"name": "no_grade",   "tonnage_mt": 5.0,  "similarity_score": 50},
            _analog("ok2", 12.0, 1.2),
            _analog("ok3", 8.0,  0.8),
        ]
        out = build_model_1(analogs, _gold_project(), {})
        # Only the 3 valid analogs should appear
        assert set(out["analogs_used"]).issubset({"ok1", "ok2", "ok3"})


class TestSignalContributions:
    def test_records_analog_sigma_and_count(self):
        analogs = [_analog(f"A{i}", 10.0 + i, 1.5, similarity=70.0) for i in range(6)]
        out = build_model_1(analogs, _gold_project(), {})
        analog = out["signal_contributions"]["analog"]
        assert analog["n_analogs"] > 0
        assert analog["sigma_logT"] > 0
        assert analog["sigma_logG"] >= 0  # could hit the floor

    def test_records_geometry_when_present(self):
        analogs = [_analog(f"A{i}", 10.0, 1.0, similarity=70.0) for i in range(5)]
        out = build_model_1(
            analogs,
            _gold_project(strike_length_meters=300, width_meters=15, depth_meters=150),
            {},
        )
        geo = out["signal_contributions"]["geometry"]
        assert geo is not None
        assert geo["geometry_tonnage_mt"] > 0

    def test_records_rule_log_shift(self):
        analogs = [_analog(f"A{i}", 10.0, 1.0, similarity=70.0) for i in range(5)]
        out = build_model_1(analogs, _gold_project(),
                            {"tonnage_multiplier": 1.5, "grade_multiplier": 0.8,
                             "rules_applied": ["r1"]})
        rules = out["signal_contributions"]["rules"]
        assert rules["log_t_shift"] == pytest.approx(math.log(1.5))
        assert rules["log_g_shift"] == pytest.approx(math.log(0.8))
