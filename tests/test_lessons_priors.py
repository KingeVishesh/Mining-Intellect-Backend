"""Unit tests for `nodes/lessons_priors.py` (P1.5).

Validates that the Gold/Silver Lessons Learned transcriptions produce the
right log-space (μ, σ) for each deposit-family × stage combination, and that
the M&I/Inferred split correctly picks deposit-aware ratios.
"""
from __future__ import annotations
import math

import pytest

from nodes.lessons_priors import (
    stage_tonnage_prior,
    mi_inferred_split,
    _classify_deposit_family,
    _classify_stage,
)


class TestClassifyDepositFamily:
    @pytest.mark.parametrize("deposit_type,mp,expected", [
        ("porphyry copper-gold", "", "porphyry"),
        ("IOCG", "", "porphyry"),
        ("copper skarn", "", "porphyry"),
        ("orogenic gold", "", "vein"),
        ("low-sulfidation epithermal vein", "", "vein"),
        ("Carlin-style sediment-hosted disseminated gold", "", "bulk"),
        ("heap-leach gold", "", "bulk"),
        ("epithermal gold-silver", "vein_hosted", "vein"),
        ("", "", "bulk"),  # default for unknown
    ])
    def test_classifies_correctly(self, deposit_type, mp, expected):
        assert _classify_deposit_family(deposit_type, mp) == expected


class TestClassifyStage:
    @pytest.mark.parametrize("stage,expected", [
        ("Feasibility Study", "mature"),
        ("In Production", "mature"),
        ("BFS",         "mature"),
        ("PEA",         "mid"),
        ("Pre-Feasibility", "mid"),
        ("Scoping",     "mid"),
        ("Advanced Exploration", "mid"),
        ("Exploration", "early"),
        ("Early Exploration", "early"),
        ("",            "mid"),  # default when unknown
    ])
    def test_classifies_correctly(self, stage, expected):
        assert _classify_stage(stage) == expected


class TestStageTonnagePrior:
    def test_early_vein_centered_around_10mt(self):
        # Range (5, 20) → geometric mean exp((ln5+ln20)/2) = sqrt(100) = 10
        mu, sigma = stage_tonnage_prior(
            "gold", "orogenic gold", "Early Exploration",
        )
        assert math.exp(mu) == pytest.approx(10.0, rel=0.05)
        # σ = (ln(20) - ln(5)) / 2 / z80 ≈ 0.541
        assert sigma == pytest.approx(0.541, abs=0.01)

    def test_mature_porphyry_centered_around_547mt(self):
        # Range (200, 1500) → exp((ln200+ln1500)/2) = sqrt(300000) ≈ 547.7
        mu, sigma = stage_tonnage_prior(
            "copper", "porphyry copper-gold", "Production",
        )
        assert math.exp(mu) == pytest.approx(547.7, rel=0.05)
        assert sigma > 0

    def test_bulk_au_ag_mid_stage(self):
        # Range (50, 200) → exp((ln50+ln200)/2) = sqrt(10000) = 100
        mu, sigma = stage_tonnage_prior(
            "gold", "Carlin-style disseminated gold", "PFS",
        )
        assert math.exp(mu) == pytest.approx(100.0, rel=0.05)

    def test_unknown_combination_falls_back_to_material_range(self):
        # Material fallback for "gold" → (3, 300) → geomean = ~30
        mu, sigma = stage_tonnage_prior(
            "gold", "unknown deposit", "", mineralization_pattern="",
        )
        # Default stage for empty is "mid", deposit family for unknown is "bulk"
        # Range (50, 200) → 100 Mt
        assert math.exp(mu) == pytest.approx(100.0, rel=0.10)

    def test_sigma_is_positive_for_all_known_combinations(self):
        for material in ("gold", "silver", "copper"):
            for stage in ("Exploration", "PEA", "Production"):
                for deposit in ("orogenic gold", "porphyry copper-gold",
                                "Carlin-style sediment-hosted gold"):
                    mu, sigma = stage_tonnage_prior(material, deposit, stage)
                    assert sigma > 0, f"σ=0 for {material}/{deposit}/{stage}"


class TestMiInferredSplit:
    def test_default_70_30(self):
        mi, inf = mi_inferred_split(
            "porphyry copper-gold",
            "",
            "PFS",
        )
        assert mi == pytest.approx(0.70)
        assert inf == pytest.approx(0.30)
        assert mi + inf == pytest.approx(1.0)

    def test_mature_near_depleted_epithermal_vein_high_mi(self):
        # L143: Inferred = 10–15% of M&I → ~87/13 split
        mi, inf = mi_inferred_split(
            "low-sulfidation epithermal",
            "vein_hosted",
            "Production",
            mine_life_years=1.5,
        )
        assert mi == pytest.approx(0.87)
        assert inf == pytest.approx(0.13)

    def test_bulk_carlin_with_halos(self):
        # L143: 60–90% M&I → 80/20
        mi, inf = mi_inferred_split(
            "Carlin-style bulk disseminated gold with low-grade halos",
            "",
            "PFS",
        )
        assert mi == pytest.approx(0.80)
        assert inf == pytest.approx(0.20)

    def test_ls_epithermal_stockwork_mid_stage(self):
        # L143/L145: 15–25% Inferred → 80/20
        mi, inf = mi_inferred_split(
            "low-sulfidation epithermal stockwork",
            "",
            "Pre-Feasibility",
        )
        assert mi == pytest.approx(0.80)
        assert inf == pytest.approx(0.20)

    def test_hs_epithermal_mature_relaxed(self):
        # L143: 15–30% Inferred for mature HS epithermal → 77/23
        mi, inf = mi_inferred_split(
            "high-sulfidation epithermal",
            "",
            "Operating",
        )
        assert mi == pytest.approx(0.77)
        assert inf == pytest.approx(0.23)

    def test_early_stage_skews_to_inferred(self):
        # L143: early-stage defaults to Inferred-heavy
        mi, inf = mi_inferred_split(
            "orogenic gold",
            "vein_hosted",
            "Early Exploration",
        )
        assert mi == pytest.approx(0.40)
        assert inf == pytest.approx(0.60)

    def test_split_always_sums_to_one(self):
        # Property test across many combos
        cases = [
            ("porphyry", "", "PEA", None),
            ("orogenic gold", "vein_hosted", "Exploration", None),
            ("low-sulfidation epithermal", "vein_hosted", "Production", 1.0),
            ("Carlin-style bulk halos", "", "Feasibility", None),
            ("VMS copper-zinc", "", "PFS", None),
        ]
        for dt, mp, stage, life in cases:
            mi, inf = mi_inferred_split(dt, mp, stage, life)
            assert mi + inf == pytest.approx(1.0), \
                f"Split doesn't sum to 1 for {dt}/{mp}/{stage}: ({mi}, {inf})"
            assert 0 <= mi <= 1 and 0 <= inf <= 1


class TestIntegrationWithBuildModel1:
    """Verify that build_model_1 actually consumes the priors as expected."""

    def test_stage_prior_is_in_signal_contributions(self):
        from nodes.model_builder import build_model_1
        analogs = [{"name": f"A{i}", "tonnage_mt": 10.0, "grade_value": 1.5,
                    "similarity_score": 70.0, "project_stage": "production",
                    "source": "library"} for i in range(5)]
        project = {"id": "p", "name": "T", "material": "gold",
                   "deposit_type": "orogenic gold", "project_stage": "exploration"}
        out = build_model_1(analogs, project, {})
        sp = out["signal_contributions"]["stage_prior"]
        assert sp["source"] == "L151_stage_tonnage_prior"
        assert sp["sigma_logT"] > 0
        # vein/early prior → geomean 10 Mt → mu ≈ ln(10) ≈ 2.30
        assert sp["mu_logT"] == pytest.approx(math.log(10.0), abs=0.1)

    def test_split_metadata_records_lesson_source(self):
        from nodes.model_builder import build_model_1
        analogs = [{"name": f"A{i}", "tonnage_mt": 10.0, "grade_value": 1.5,
                    "similarity_score": 70.0, "project_stage": "production",
                    "source": "library"} for i in range(5)]
        project = {"id": "p", "name": "T", "material": "gold",
                   "deposit_type": "Carlin-style bulk disseminated gold with halos",
                   "project_stage": "PFS"}
        out = build_model_1(analogs, project, {})
        split = out["signal_contributions"]["split"]
        # Carlin bulk with halos at mid stage → 80/20
        assert split["mi_frac"] == pytest.approx(0.80)
        assert split["inf_frac"] == pytest.approx(0.20)
        # Verify the actual fields also reflect this split
        total_kt = out["total_tonnage_kt"]
        mi_kt = out["mi_tonnage_kt"]
        assert mi_kt == pytest.approx(total_kt * 0.80, rel=0.01)
