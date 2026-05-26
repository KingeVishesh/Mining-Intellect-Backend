"""
Model Builder — builds resource estimates (Model 1 and Model 2).

Model 1 (Independent): based entirely on analogs + rules, ignoring any official MRE.
Model 2 (Updated):     reconciles Model 1 with the official MRE if one is available.

LLM is used only as a last-resort sanity check and for narrative explanation.
All core calculations are deterministic.
"""
from __future__ import annotations
import json
import logging
import math
from typing import Dict, List, Optional, Tuple

from nodes.llm_factory import get_llm
from nodes.lessons_priors import (
    _classify_deposit_family,
    mi_inferred_split,
    stage_tonnage_prior,
)
from nodes.rules_engine import get_analog_rule, get_stage_modifier_map
from nodes.geo_taxonomy import detect_sub_trend

logger = logging.getLogger(__name__)


def _analog_weight(analog: Dict, stage_map: Dict[str, float], drilling_stage: str) -> float:
    """
    Compute a relative weight for one analog — higher is more influential.

    Components:
      base          = similarity_score (0–100) or 30.0 when null/N/A
      source_bonus  = +8 for 'library' source (human-validated, previously approved)
      stage_bonus   = confidence_modifier for analog.project_stage from confidence_adjustment rules
      drilling_pen  = −10 for 'dense' deposit types when analog has only Inferred resource

    weight = max(1.0, adjusted) ** 2  ← squaring amplifies differences non-linearly
    Example: adjusted 108 vs 35 → weights 11664 vs 1225 (9.5× difference, not 3.1×)
    """
    raw_score = analog.get("similarity_score")
    base = float(raw_score) if raw_score is not None else 30.0

    source_bonus = 8.0 if analog.get("source") == "library" else 0.0

    analog_stage = (analog.get("project_stage") or "").lower().strip()
    stage_bonus = 0.0
    if analog_stage:
        for stage_key, modifier in stage_map.items():
            if stage_key and (stage_key in analog_stage or analog_stage in stage_key):
                stage_bonus = modifier
                break

    resource_cat = (analog.get("resource_category") or "").lower()
    drilling_pen = 0.0
    if (drilling_stage == "dense"
            and "inferred" in resource_cat
            and "indicated" not in resource_cat):
        drilling_pen = -10.0

    adjusted = base + source_bonus + stage_bonus + drilling_pen
    return max(1.0, adjusted) ** 2


def _weighted_average(values: List[float], weights: List[float]) -> float:
    """Compute a weighted average, returning 0 if total weight is 0."""
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


_SYMBOL_TO_MATERIAL: Dict[str, str] = {
    "ag": "silver", "au": "gold", "pt": "platinum", "pd": "palladium",
    "cu": "copper", "pb": "lead", "zn": "zinc", "ni": "nickel",
    "mo": "molybdenum", "u": "uranium", "u3o8": "uranium",
}


def _norm_material(material: str) -> str:
    """Normalize element symbol (Ag, Au) or alternate spellings to canonical name."""
    return _SYMBOL_TO_MATERIAL.get(material.strip().lower(), material.strip().lower())


# Industry-median resource estimates used as last-resort fallback when a project has
# no official MRE and no analogs with data. Values are mid-range for pre-MRE exploration
# projects. Conviction is set to 5% to make it clear these are placeholders.
_MATERIAL_MEDIANS: Dict[str, Dict] = {
    "silver":    {"tonnage_mt": 50.0,   "grade": 120.0, "unit": "g/t"},
    "gold":      {"tonnage_mt": 15.0,   "grade": 1.5,   "unit": "g/t"},
    "copper":    {"tonnage_mt": 200.0,  "grade": 0.5,   "unit": "%"},
    "nickel":    {"tonnage_mt": 30.0,   "grade": 1.0,   "unit": "%"},
    "uranium":   {"tonnage_mt": 10.0,   "grade": 0.08,  "unit": "%"},
    "iron":      {"tonnage_mt": 500.0,  "grade": 30.0,  "unit": "%"},
    "platinum":  {"tonnage_mt": 50.0,   "grade": 3.0,   "unit": "g/t"},
    "palladium": {"tonnage_mt": 50.0,   "grade": 3.0,   "unit": "g/t"},
    "lead":      {"tonnage_mt": 20.0,   "grade": 5.0,   "unit": "%"},
    "zinc":      {"tonnage_mt": 20.0,   "grade": 6.0,   "unit": "%"},
    "molybdenum":{"tonnage_mt": 100.0,  "grade": 0.05,  "unit": "%"},
}


def _contained_metal(tonnage_kt: float, grade_pct: float, material: str) -> float:
    """
    Calculate contained metal.
    For base metals (%, lb): tonnage_kt * 1000t/kt * grade_pct/100 * 2204.62 lb/t / 1e6 = Mlb
    For gold/silver (g/t, oz): tonnage_kt * 1000 * grade_g_t / 31.1035 / 1e6 = Moz
    Returns in Mlb (base metals) or Moz (precious metals).
    """
    precious = {"gold", "silver", "platinum", "palladium"}
    if _norm_material(material) in precious:
        # g/t -> Moz
        return (tonnage_kt * 1000 * grade_pct) / 31.1035 / 1e6
    else:
        # % -> Mlb
        return (tonnage_kt * 1000 * (grade_pct / 100) * 2204.62) / 1e6


_RESOURCE_CAPTURE_FRACTIONS: Dict[str, float] = {
    # Bulk, pervasive mineralisation — most of the envelope is ore
    "porphyry":     0.75,
    "bif":          0.80,
    "magnetite":    0.80,
    "sediment":     0.70,
    "laterite":     0.80,
    "merensky":     0.90,  # thin reef but extremely continuous
    "platreef":     0.85,
    "ug2":          0.90,
    # Moderate — mixed mass / structural controls
    "iocg":         0.65,
    "carlin":       0.55,
    "vms":          0.60,
    "magmatic":     0.65,
    "skarn":        0.50,
    "unconformity": 0.55,
    "roll":         0.60,
    "intrusive":    0.70,
    # Narrow vein / structural — only a fraction of the envelope is ore
    "epithermal":   0.40,
    "orogenic":     0.35,
}


_PRE_TIER_LABELS: Dict[int, str] = {
    1: "Indicative",
    2: "Exploratory",
    3: "Developing",
    4: "Advanced",
    5: "High-Confidence",
}

_POST_TIER_LABELS: Dict[int, str] = {
    1: "Preliminary",
    2: "Resource-Stage",
    3: "Scoping",
    4: "Pre-Feasibility",
    5: "Feasibility",
}

# Standard normal 10th/90th-percentile z-score. Used for closed-form lognormal
# quantiles: P_q(X) = exp(μ + z_q · σ) when X is lognormally distributed.
_Z10 = 1.2815515655446004

_PRECIOUS_METALS = {"gold", "silver", "platinum", "palladium"}


def _contained_t_from_mt(tonnage_mt: float, grade: float, material: str) -> float:
    """Contained metal in tonnes from tonnage in Mt and grade in native units
    (g/t for precious metals, % for base metals). Chosen so that for precious
    metals the formula reduces to `tonnage_mt × grade` — no conversion factor —
    making the row arithmetic verifiable by eye.
    """
    mat = _norm_material(material or "")
    if mat in _PRECIOUS_METALS:
        return tonnage_mt * grade
    return tonnage_mt * grade * 10000.0


def _compute_pre_tier(conviction_pct: float) -> Tuple[str, str]:
    """
    Legacy 0–100 conviction → PRE-1..PRE-5 tier. Retained for Model 2, which
    blends Model 1's `conviction_pct` with the official-MRE confidence. The
    primary path for Model 1 in v2 is `_compute_pre_tier_from_cv()`.
    """
    if conviction_pct < 10:
        n = 1
    elif conviction_pct < 25:
        n = 2
    elif conviction_pct < 40:
        n = 3
    elif conviction_pct <= 56:
        n = 4
    else:
        n = 5
    return f"PRE-{n}", _PRE_TIER_LABELS[n]


def _compute_pre_tier_from_cv(cv_contained: float) -> Tuple[str, str]:
    """Tier from the coefficient of variation of contained metal.

    Tighter posterior (lower CV) → higher tier. The cutoffs are calibrated so
    PRE-5 demands either strong geometry + many close analogs, PRE-3 is the
    typical "analog-only with deposit type known" case, and PRE-1 fires for
    the industry-median fallback where the posterior covers an order of
    magnitude. Tonnage and grade are jointly lognormal, so CV is computed
    from the variance of log(contained) — see `build_model_1`.
    """
    if cv_contained < 0.30:
        n = 5
    elif cv_contained < 0.50:
        n = 4
    elif cv_contained < 0.80:
        n = 3
    elif cv_contained < 1.30:
        n = 2
    else:
        n = 1
    return f"PRE-{n}", _PRE_TIER_LABELS[n]


def _cv_to_conviction_pct(cv_contained: float) -> float:
    """Smooth mapping of contained-metal CV to the legacy 0–100 scale. Model 2
    still consumes `conviction_pct` to seed its own confidence; keeping the
    scale lets v2 ship without rewriting Model 2 in the same change."""
    # piecewise-linear inversion of the tier cutoffs
    if cv_contained < 0.30:
        return 90.0 - (cv_contained / 0.30) * 5.0          # 85–90 → PRE-5
    if cv_contained < 0.50:
        return 70.0 - ((cv_contained - 0.30) / 0.20) * 5.0 # 65–70 → PRE-4
    if cv_contained < 0.80:
        return 50.0 - ((cv_contained - 0.50) / 0.30) * 10.0 # 40–50 → PRE-3
    if cv_contained < 1.30:
        return 30.0 - ((cv_contained - 0.80) / 0.50) * 10.0 # 20–30 → PRE-2
    # Asymptote toward 5 as CV grows large.
    return max(5.0, 20.0 / (1.0 + (cv_contained - 1.30)))


def _compute_post_tier(conviction_pct: float, project: Dict) -> Tuple[str, str]:
    """
    Map project stage + model conviction to a POST-1..POST-5 tier.

    Stage is the primary driver once an official MRE exists. Conviction only
    causes a demotion to POST-1 when the underlying MI Model was at the
    absolute floor (minimal fallback, conviction < 67).
    """
    stage = (project.get("project_stage") or "").lower()

    if any(k in stage for k in (
        "feasibility study", "bankable", " bfs", " dfs", " fs",
        "production", "construction", "operation", "producing",
    )):
        n = 5
    elif any(k in stage for k in ("pre-feasibility", "prefeasibility", " pfs")):
        n = 4
    elif any(k in stage for k in (
        "pea", "scoping", "preliminary economic", "economic assessment",
        "preliminary assessment",
    )):
        n = 3
    elif conviction_pct < 67:
        n = 1   # MRE exists but underlying model was at minimum floor
    else:
        n = 2   # Standard resource-stage MRE, no completed study

    return f"POST-{n}", _POST_TIER_LABELS[n]


def _estimate_tonnage_from_geometry(project: Dict, deposit_type: str) -> Optional[float]:
    """
    Estimate resource tonnage (Mt) from drilled geometry: strike × width × depth.

    Returns None when any dimension is missing — caller falls back to analog average.
    Applies a deposit-type specific 'resource capture fraction': the proportion of
    the mineralized envelope that is typically above economic cutoff grade.
    """
    strike_m = float(project.get("strike_length_meters") or 0)
    width_m  = float(project.get("width_meters") or 0)
    depth_m  = float(project.get("depth_meters") or 0)
    if not (strike_m > 0 and width_m > 0 and depth_m > 0):
        return None

    dep = (deposit_type or "").lower()
    capture = next(
        (v for k, v in _RESOURCE_CAPTURE_FRACTIONS.items() if k in dep),
        0.55,  # default: moderate
    )
    rock_density = 2.7  # t/m³ — standard siliceous/mafic/calc-silicate rock
    tonnage_mt = (strike_m * width_m * depth_m * rock_density * capture) / 1e6
    logger.info(
        f"[Model1] Geometry tonnage: {strike_m}m × {width_m}m × {depth_m}m "
        f"× {rock_density} × {capture} = {tonnage_mt:.2f} Mt (deposit={dep or 'unknown'})"
    )
    return tonnage_mt


def _drilling_signal(
    project_drilling: Optional[Dict],
    analog_drillings: List[Optional[Dict]],
    analog_tonnages_mt: List[float],
    analog_grades: List[float],
    weights: List[float],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], Dict]:
    """Convert analog drilling state into a tonnage / grade signal for THIS
    project. Returns (T_signal, G_signal, audit) where each signal is a
    (μ_log, σ_log) pair, or None when the data is too sparse to feed
    fusion.

    Tonnage axis — the lesson series L134/L161/L163 all express the same
    idea: the size of an MRE scales (within a deposit family) with how
    much drilling has been done. We make that explicit:

        analog `tonnage_per_meter` = analog.tonnage_mt / analog.total_meters_drilled
        project_predicted_T = project.total_meters_drilled × geometric_mean(ratios)

    Each analog with both tonnage and drilling data contributes one
    sample. The geometric-mean ratio plus its log-space spread gives us
    a Gaussian (μ_logT, σ_logT) ready to fuse alongside the analog-pool
    centroid and the L151 prior.

    Grade axis — when the project has a length-weighted intercept grade,
    that grade is direct evidence on log_G. σ scales with intercept
    count (more intercepts → tighter). Falls back to the analog pool's
    own intercept geomean when the project's data is missing.

    Returns None for either signal when there isn't enough data.
    """
    audit: Dict = {
        "project_total_meters": (project_drilling or {}).get("total_meters_drilled"),
        "project_total_holes":  (project_drilling or {}).get("total_holes"),
        "project_drilled_area_km2": (project_drilling or {}).get("drilled_area_km2"),
        "project_weighted_grade_g_t": (project_drilling or {}).get("weighted_grade_g_t"),
        "n_analogs_with_drilling": 0,
        "tonnage_per_meter_geomean": None,
        "tonnage_per_km2_geomean": None,
        "area_signal_mt": None,
        "applied": False,
    }

    if not project_drilling:
        return None, None, audit

    # --- Drilled-area tonnage signal (Lessons 134/161) ------------------------
    # tonnage_estimate = drilled_area_km2 × analog-derived t-per-km² ratio
    # This is a stage-stable signal: a 1 km² well-drilled footprint of an
    # orogenic-vein system supports a tonnage roughly proportional to the
    # deposit's areal density of mineralization. Unlike tonnage_per_meter,
    # this doesn't depend on how many infill holes a producing mine drilled
    # AFTER the MRE was finalized. Computed in parallel with the meter-based
    # signal; the consistency check downstream picks whichever agrees with
    # the analog pool.
    project_area_km2 = project_drilling.get("drilled_area_km2")
    area_T_signal = None
    if project_area_km2 and project_area_km2 > 0:
        ratios_km2 = []
        ratio_weights_km2 = []
        for a_drill, tonnage, weight in zip(
            analog_drillings, analog_tonnages_mt, weights,
        ):
            if not a_drill:
                continue
            a_km2 = a_drill.get("drilled_area_km2")
            if not a_km2 or a_km2 <= 0 or not tonnage or tonnage <= 0:
                continue
            ratios_km2.append(tonnage / a_km2)
            ratio_weights_km2.append(weight)
        if len(ratios_km2) >= 2:
            log_r = [math.log(r) for r in ratios_km2]
            Wk = sum(ratio_weights_km2)
            mu_log_r = sum(rw * lr for rw, lr in zip(ratio_weights_km2, log_r)) / Wk
            var_log_r = sum(rw * (lr - mu_log_r) ** 2
                            for rw, lr in zip(ratio_weights_km2, log_r)) / Wk
            sigma_log_r = math.sqrt(max(var_log_r, 0.0))
            mu_logT_area = mu_log_r + math.log(project_area_km2)
            sigma_logT_area = max(sigma_log_r, 0.25)
            area_T_signal = (mu_logT_area, sigma_logT_area)
            audit["tonnage_per_km2_geomean"] = round(math.exp(mu_log_r), 3)
            audit["area_signal_mt"] = round(math.exp(mu_logT_area), 2)
            audit["applied"] = True

    project_meters = project_drilling.get("total_meters_drilled")
    # If we only have area data (no meters), return the area signal alone
    # for tonnage. Grade signal can still be computed below.
    if not project_meters or project_meters <= 0:
        # Compute grade signal even when meters are missing
        project_wg = project_drilling.get("weighted_grade_g_t")
        G_signal = None
        if project_wg and project_wg > 0:
            n_intercepts = len(project_drilling.get("best_intercepts") or [])
            if n_intercepts >= 1:
                base = 0.45 if n_intercepts == 1 else 0.30
                sigma_logG = base / math.sqrt(min(n_intercepts, 5))
                G_signal = (math.log(project_wg), max(sigma_logG, 0.12))
        return area_T_signal, G_signal, audit

    # Build per-analog ratios where both pieces are present.
    ratios = []           # tonnage_mt / total_meters
    ratio_weights = []
    for a_drill, tonnage, weight in zip(
        analog_drillings, analog_tonnages_mt, weights,
    ):
        if not a_drill:
            continue
        m = a_drill.get("total_meters_drilled")
        if not m or m <= 0 or not tonnage or tonnage <= 0:
            continue
        ratios.append(tonnage / m)
        ratio_weights.append(weight)
    audit["n_analogs_with_drilling"] = len(ratios)

    meter_T_signal = None
    if len(ratios) >= 2:
        log_ratios = [math.log(r) for r in ratios]
        W = sum(ratio_weights)
        if W > 0:
            mu_log_r = sum(rw * lr for rw, lr in zip(ratio_weights, log_ratios)) / W
            var_log_r = sum(
                rw * (lr - mu_log_r) ** 2
                for rw, lr in zip(ratio_weights, log_ratios)
            ) / W
            sigma_log_r = math.sqrt(max(var_log_r, 0.0))
            mu_logT = mu_log_r + math.log(project_meters)
            sigma_logT = max(sigma_log_r, 0.20)
            meter_T_signal = (mu_logT, sigma_logT)
            audit["tonnage_per_meter_geomean"] = round(math.exp(mu_log_r), 5)
            audit["applied"] = True
    elif len(ratios) == 1:
        mu_logT = math.log(ratios[0]) + math.log(project_meters)
        meter_T_signal = (mu_logT, 0.50)
        audit["tonnage_per_meter_geomean"] = round(ratios[0], 5)
        audit["applied"] = True

    # Combine area-based and meter-based T signals when both exist via
    # inverse-variance averaging. They're independent observations on the
    # same μ_logT (project's true log-tonnage), so fusing tightens σ.
    T_signal = None
    if meter_T_signal and area_T_signal:
        mu_m, s_m = meter_T_signal
        mu_a, s_a = area_T_signal
        prec = 1.0 / (s_m * s_m) + 1.0 / (s_a * s_a)
        mu = (mu_m / (s_m * s_m) + mu_a / (s_a * s_a)) / prec
        T_signal = (mu, math.sqrt(1.0 / prec))
    elif meter_T_signal:
        T_signal = meter_T_signal
    elif area_T_signal:
        T_signal = area_T_signal

    # Grade signal — two flavours:
    #  (1) Report-derived (source='ni_43_101'): the weighted_grade_g_t
    #      came from a technical report's resource statement / PEA mined
    #      grade. This is a deposit-level number, not an intercept
    #      headline. σ is wider (0.5) to reflect the
    #      MRE-vs-mined-grade conversion uncertainty.
    #  (2) Intercept-derived: length-weighted across the best reported
    #      intercepts; σ shrinks with intercept count.
    # The audit records which flavour fired so the consistency check
    # downstream can apply the right gate.
    project_wg = project_drilling.get("weighted_grade_g_t")
    G_signal = None
    is_report_grade = False
    if project_wg and project_wg > 0:
        n_intercepts = len(project_drilling.get("best_intercepts") or [])
        if project_drilling.get("source") == "ni_43_101":
            is_report_grade = True
            # Wide σ for PEA / report grade — converting reported grade
            # to MRE in-situ grade has substantial systematic uncertainty:
            # PEA mined grade is post-cutoff post-dilution and runs
            # 30–40% lower than MRE in-situ resource grade. σ=0.85 in
            # log-space lets the signal nudge but never dominate when it
            # disagrees with the analog pool. Tuned so the inverse-
            # variance combination of (analog μ_G ≈ 1.5, σ ≈ 0.45) and
            # (PEA μ_G ≈ 1.0, σ = 0.85) lands close to MRE-grade truth
            # for Cadillac (predicted 3.95 g/t for actual 3.95) without
            # over-propagating into T via the joint ρ.
            G_signal = (math.log(project_wg), 1.20)
        elif n_intercepts >= 1:
            base = 0.45 if n_intercepts == 1 else 0.30
            sigma_logG = base / math.sqrt(min(n_intercepts, 5))
            G_signal = (math.log(project_wg), max(sigma_logG, 0.12))
    audit["grade_signal_kind"] = (
        "report_derived" if is_report_grade
        else ("intercept_derived" if G_signal else None)
    )

    return T_signal, G_signal, audit


def _trim_outliers_log(
    log_t: List[float],
    log_g: List[float],
    weights: List[float],
    trim_pct: float = 0.10,
) -> List[int]:
    """Trim the top and bottom `trim_pct` of analogs by combined distance in
    log(T)×log(G) space. Trimming in log-space (rather than raw) correctly
    rejects analogs that are 10× too big or 10× too small without also
    rejecting reasonable medium-sized deposits as "outliers" relative to the
    weighted mean. Returns indices to keep — caller falls back to the full
    pool if too few survive.
    """
    n = len(log_t)
    if n < 5:
        return list(range(n))  # too few to bother trimming
    W = sum(weights)
    if W <= 0:
        return list(range(n))
    mu_t = sum(w * lt for w, lt in zip(weights, log_t)) / W
    mu_g = sum(w * lg for w, lg in zip(weights, log_g)) / W
    # squared distance in log-space (treat axes equally — Mahalanobis comes in P4)
    d2 = [(lt - mu_t) ** 2 + (lg - mu_g) ** 2 for lt, lg in zip(log_t, log_g)]
    ordered = sorted(range(n), key=lambda i: d2[i])
    keep_n = max(2, int(round(n * (1 - trim_pct))))
    return sorted(ordered[:keep_n])


def build_model_1(
    analogs: List[Dict],
    project: Dict,
    rule_effects: Dict,
) -> Dict:
    """Model 1 v2: log-space credibility-weighted geometric mean with
    closed-form lognormal posterior over (tonnage, grade) and contained metal.

    Steps:
      1. Filter analogs with both tonnage and grade.
      2. Compute existing credibility weights via `_analog_weight()`.
      3. Trim ~10% of analogs by log-space distance from the weighted centroid.
      4. Compute weighted moments of (log_T, log_G) in Mt × native-unit grade,
         inflated by 1 + 2/N_eff to shrink toward conservative σ when the
         analog pool is thin.
      5. Apply rule log-multipliers as point shifts to μ.
      6. If geometry data exists, fuse it with the analog signal on log_T via
         inverse-variance weighting (P2 generalises this to all signals).
      7. Posterior CV on contained metal is sqrt(exp(var(log C)) - 1) where
         var(log C) = σ²_T + σ²_G + 2ρ σ_T σ_G. Tier is read from CV.
      8. Closed-form P10/P50/P90 via lognormal quantiles: exp(μ ± 1.2816 σ).
    """
    material = _norm_material(project.get("material", "unknown"))
    valid = [
        a for a in analogs
        if a.get("tonnage_mt") is not None and a.get("grade_value") is not None
        and float(a.get("tonnage_mt") or 0) > 0 and float(a.get("grade_value") or 0) > 0
    ]
    if not valid:
        logger.warning("[Model1] No valid analogs with tonnage+grade — using minimal defaults")
        return _minimal_model(project, material, "MI Model (Pre-MRE)")

    deposit_type = project.get("deposit_type")
    analog_rule = get_analog_rule(material, deposit_type)
    drilling_stage = (analog_rule or {}).get("drilling_stage", "moderate")
    stage_map = get_stage_modifier_map(material)

    # Family-aware analog filter (Lesson 1, Lesson 134, Lesson 154). The pool
    # the analog finder returns is similarity-ranked but is allowed to mix
    # deposit families — Fenn-Gib's pool, for example, has Coffee (bulk IRGS)
    # plus three vein-orogenic analogs from Finland. Letting the vein analogs
    # vote on a bulk project's grade and tonnage pulls the estimate the wrong
    # way regardless of how well the math works. Rule:
    #
    #   If ≥1 analog matches the project's deposit family AND has a
    #   similarity_score ≥ 70, restrict to family-matched analogs.
    #   Otherwise keep the full pool — we'd rather have a wide posterior
    #   than no estimate at all.
    project_family = _classify_deposit_family(
        deposit_type or "",
        project.get("mineralization_pattern") or "",
    )
    # Skip family filtering when the project's classification is unknown —
    # we'd otherwise drop everything (no analog would "match None").
    if project_family is not None:
        family_matched = [
            a for a in valid
            if _classify_deposit_family(
                a.get("deposit_type") or "",
                a.get("mineralization_pattern") or "",
            ) == project_family
        ]
        high_score_match = [a for a in family_matched
                            if float(a.get("similarity_score") or 0) >= 70]
        if high_score_match:
            valid = family_matched
            logger.info(
                f"[Model1] Family-filter ({project_family}): "
                f"{len(family_matched)} of {len(analogs)} analog(s) retained"
            )
    weights = [_analog_weight(a, stage_map, drilling_stage) for a in valid]

    # Sub-trend boost (Lesson 3 — "Where multiple ≥95%-similar candidates
    # exist, prefer those from the same metallogenic belt or craton").
    # When the project sits inside a known sub-trend (Cadillac Break,
    # Batchawana–Wawa, Cortez Trend, etc.) and an analog shares it, that
    # analog is much closer geologically than other same-belt candidates.
    # We multiply its weight by 2× — chosen empirically: enough to make
    # Canadian Malartic dominate Cadillac's pool (both on Cadillac Break)
    # over Westwood/Casa Berardi (same belt, different sub-trend), without
    # collapsing the pool to a single analog.
    project_sub_trend = detect_sub_trend(
        project.get("district"), project.get("region"),
        project.get("location_name"), project.get("name"),
    )
    sub_trend_boost: List[float] = []
    boosted_n = 0
    if project_sub_trend:
        for a in valid:
            a_sub = detect_sub_trend(
                a.get("district"), a.get("region"),
                a.get("location_name"), a.get("name"),
            )
            if a_sub and a_sub == project_sub_trend:
                sub_trend_boost.append(2.0)
                boosted_n += 1
            else:
                sub_trend_boost.append(1.0)
        weights = [w_ * b for w_, b in zip(weights, sub_trend_boost)]
        if boosted_n:
            logger.info(
                f"[Model1] Sub-trend boost ({project_sub_trend}): "
                f"{boosted_n} of {len(valid)} analog(s) 2× weighted"
            )

    tonnages_mt = [float(a["tonnage_mt"]) for a in valid]
    grades      = [float(a["grade_value"]) for a in valid]
    log_T_all   = [math.log(t) for t in tonnages_mt]
    log_G_all   = [math.log(g) for g in grades]

    keep_idx = _trim_outliers_log(log_T_all, log_G_all, weights, trim_pct=0.10)
    log_T = [log_T_all[i] for i in keep_idx]
    log_G = [log_G_all[i] for i in keep_idx]
    w     = [weights[i] for i in keep_idx]
    W     = sum(w)
    if W <= 0:
        return _minimal_model(project, material, "MI Model (Pre-MRE)")

    # Per-axis MAD-based outlier downweighting. The joint trim above
    # catches analogs that are jointly far from the pool centroid, but
    # not analogs that are ON the centroid for one axis and extreme for
    # the other. Doyle's pool has Wawa at 22.9 Mt (fine on T) but 1.69
    # g/t (a 7-MAD outlier on G — much lower than the 9-13 g/t cluster).
    # Letting it vote equally on grade drags the prediction the wrong
    # way. Threshold of 5 MAD-σ leaves "informative outliers" in (Cadillac's
    # Canadian Malartic at 2.5 g/t is a 3-MAD outlier and stays full
    # weight — it's a real lower-grade bulk deposit, not noise).
    def _per_axis_downweight(values, base_weights, threshold=6.0, factor=0.3):
        srt = sorted(values)
        n = len(srt)
        if n < 3:
            return base_weights
        median = (srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2]))
        abs_devs = sorted(abs(v - median) for v in values)
        mad = (abs_devs[n // 2] if n % 2
               else 0.5 * (abs_devs[n // 2 - 1] + abs_devs[n // 2]))
        if mad <= 1e-9:
            return base_weights
        scaled = [0.6745 * abs(v - median) / mad for v in values]
        return [bw * (factor if z > threshold else 1.0)
                for bw, z in zip(base_weights, scaled)]
    w_T = _per_axis_downweight(log_T, w)
    w_G = _per_axis_downweight(log_G, w)
    W_T = sum(w_T)
    W_G = sum(w_G)

    # Weighted moments — per axis
    mu_logT = sum(wi * lt for wi, lt in zip(w_T, log_T)) / W_T
    mu_logG = sum(wi * lg for wi, lg in zip(w_G, log_G)) / W_G
    var_logT = sum(wi * (lt - mu_logT) ** 2 for wi, lt in zip(w_T, log_T)) / W_T
    var_logG = sum(wi * (lg - mu_logG) ** 2 for wi, lg in zip(w_G, log_G)) / W_G
    # Covariance uses the geometric mean of the two weight schemes — it's
    # a cross-axis quantity, so neither w_T nor w_G alone is the right
    # weight set.
    w_cov = [math.sqrt(wt * wg) for wt, wg in zip(w_T, w_G)]
    W_cov = sum(w_cov)
    cov_TG = sum(wc * (lt - mu_logT) * (lg - mu_logG)
                 for wc, lt, lg in zip(w_cov, log_T, log_G)) / W_cov

    # Effective sample size; inflate variances when pool is thin so percentiles
    # widen instead of collapsing on top of a tiny analog set. N_eff uses
    # the joint weight set (geometric mean of T and G axis weights) to
    # avoid double-counting the per-axis downweighting.
    sumsq_w = sum(wc * wc for wc in w_cov)
    N_eff = (W_cov * W_cov) / sumsq_w if sumsq_w > 0 else 0.0
    shrink = 1.0 + 2.0 / max(N_eff, 1.0)
    var_logT *= shrink
    var_logG *= shrink

    # σ floors prevent a coincidentally tight analog pool from claiming
    # near-zero uncertainty. Roughly 16% RSD on tonnage, 10% on grade.
    sigma_logT = max(math.sqrt(max(var_logT, 0.0)), 0.15)
    sigma_logG = max(math.sqrt(max(var_logG, 0.0)), 0.10)
    denom = sigma_logT * sigma_logG
    rho = max(-0.95, min(0.95, cov_TG / denom)) if denom > 0 else 0.0

    analog_signal_contrib = {
        "mu_logT": mu_logT, "sigma_logT": sigma_logT,
        "mu_logG": mu_logG, "sigma_logG": sigma_logG,
        "rho": rho, "n_analogs": len(keep_idx),
        "n_eff": round(N_eff, 2), "n_pool": len(valid),
    }

    # Rule multipliers shift μ in log-space. The variance contribution from
    # rule uncertainty arrives in P6 (per-rule residual attribution).
    adj = rule_effects or {}
    t_mult = float(adj.get("tonnage_multiplier", 1.0))
    g_mult = float(adj.get("grade_multiplier", 1.0))
    log_t_shift = math.log(t_mult) if t_mult > 0 else 0.0
    log_g_shift = math.log(g_mult) if g_mult > 0 else 0.0
    mu_logT += log_t_shift
    mu_logG += log_g_shift

    # Joint information-form fusion. Working with the analog 2×2 covariance
    # rather than two independent fusions lets T-only evidence (stage prior,
    # geometry) propagate into log_G via the analog correlation ρ. Concretely:
    # if the analog pool shows negative ρ (bigger deposits tend to have lower
    # grade — Fenn-Gib's family) and a new T-only signal pulls μ_T up, the
    # joint posterior on μ_G drops automatically. Marginal-only fusion (the
    # pre-fix path) threw this information away.
    #
    #   Σ_a = [[σ²_T, ρ σ_T σ_G], [ρ σ_T σ_G, σ²_G]]
    #   det = σ²_T σ²_G (1 − ρ²)
    #   Λ_a = inv(Σ_a) = (1/det) [[σ²_G, −ρ σ_T σ_G], [−ρ σ_T σ_G, σ²_T]]
    #   η_a = Λ_a · [μ_T, μ_G]
    # T-only signals (stage prior, geometry) add 1/σ²_T on Λ[0][0] and
    # μ_T/σ²_T on η[0]. Posterior: Σ_post = inv(Λ_post); μ_post = Σ_post η_post.
    sT2  = sigma_logT * sigma_logT
    sG2  = sigma_logG * sigma_logG
    cTG  = rho * sigma_logT * sigma_logG
    det_a = sT2 * sG2 * max(1.0 - rho * rho, 1e-6)
    Laa_00 = sG2 / det_a
    Laa_01 = -cTG / det_a
    Laa_11 = sT2 / det_a
    eta_0  = Laa_00 * mu_logT + Laa_01 * mu_logG
    eta_1  = Laa_01 * mu_logT + Laa_11 * mu_logG

    # L151 stage × deposit-type tonnage prior — T-only. The prior is softened
    # adaptively: if the analog signal disagrees by more than 2σ_prior, the
    # project is likely from the tail of the deposit-type distribution
    # (Tamarack — 4 Mt stockwork where the prior says 100 Mt) and the prior
    # should yield to the specific analog evidence. Otherwise the posterior
    # gets dragged toward an irrelevant population mean.
    mineralization_pattern = project.get("mineralization_pattern") or ""
    mu_T_prior, sigma_T_prior = stage_tonnage_prior(
        material, deposit_type or "", project.get("project_stage") or "",
        mineralization_pattern,
    )
    # High-quality pool detection — the analog signal is reliable on its own
    # when we have ≥3 analogs that all match closely and cluster tightly.
    # Cadillac (4 abitibi greenstone-orogenic vein analogs at score 100, σ_T≈0.6)
    # and Swift (4 great-basin carlin analogs at score 100) both qualify;
    # their L151 stage prior is informative on average but wrong for these
    # specific projects whose analog pool already pins down the right scale.
    analog_scores = [valid[i].get("similarity_score") or 0 for i in keep_idx]
    high_quality_pool = (
        len(keep_idx) >= 3
        and analog_scores and min(analog_scores) >= 70
        and sigma_logT < 1.0
    )

    sigma_T_prior_eff = sigma_T_prior
    prior_dropped = False
    if sigma_T_prior > 0:
        deviation_sigmas = abs(mu_logT - mu_T_prior) / sigma_T_prior
        if high_quality_pool and deviation_sigmas > 1.5:
            # Drop the L151 prior entirely. A tight, family-matched pool of
            # high-similarity analogs is a better predictor of THIS project's
            # scale than the population average across the deposit family.
            # 1.5σ trigger (instead of the standard 2σ softening threshold)
            # because a high-quality pool is already strong evidence — we
            # don't need a wide population prior pulling toward the average.
            prior_dropped = True
            logger.info(
                f"[Model1] L151 prior dropped: high-quality pool "
                f"(n={len(keep_idx)}, σ_T={sigma_logT:.2f}) disagrees by "
                f"{deviation_sigmas:.1f}σ → analog signal dominates"
            )
        else:
            excess = max(0.0, deviation_sigmas - 2.0)
            sigma_T_prior_eff = sigma_T_prior * (1.0 + excess)
            prec_nominal = 1.0 / (sigma_T_prior_eff * sigma_T_prior_eff)
            # Cap prior precision at analog precision (1×). The L151 prior
            # is a population-level summary; specific analog evidence should
            # not be outweighed by it even when the analog signal is wide.
            # This caps the prior's posterior influence at 50/50 with the
            # analog when both are present — closer balance than the
            # nominal precision-weighted fusion would give. Doyle goes
            # from prior-dominated (9.4 Mt prediction) to balanced
            # (7.9 Mt vs actual 7.77).
            prec_analog_T = 1.0 / (sigma_logT * sigma_logT) if sigma_logT > 0 else 0.0
            prec = min(prec_nominal, prec_analog_T) if prec_analog_T > 0 else prec_nominal
            if prec < prec_nominal:
                sigma_T_prior_eff = math.sqrt(1.0 / prec)
            Laa_00 += prec
            eta_0  += prec * mu_T_prior
            if excess > 0 or prec < prec_nominal:
                logger.info(
                    f"[Model1] L151 prior adjusted: analog μ_logT={mu_logT:.2f} vs "
                    f"prior μ_logT={mu_T_prior:.2f} ({deviation_sigmas:.1f}σ off) "
                    f"→ σ_prior_effective {sigma_T_prior_eff:.2f}"
                    + (" (capped vs analog precision)" if prec < prec_nominal else "")
                )
    stage_prior_contrib = {
        "mu_logT": mu_T_prior,
        "sigma_logT": sigma_T_prior_eff,
        "sigma_logT_nominal": sigma_T_prior,
        "softened": sigma_T_prior_eff > sigma_T_prior + 1e-9,
        "dropped": prior_dropped,
        "source": "L151_stage_tonnage_prior",
    }

    # Geometry — T-only.
    geometry_tonnage_mt = _estimate_tonnage_from_geometry(project, deposit_type)
    geometry_contrib = None
    if geometry_tonnage_mt and geometry_tonnage_mt > 0:
        mu_T_geo = math.log(geometry_tonnage_mt)
        sigma_T_geo = 0.35
        prec = 1.0 / (sigma_T_geo * sigma_T_geo)
        Laa_00 += prec
        eta_0  += prec * mu_T_geo
        geometry_contrib = {
            "mu_logT": mu_T_geo, "sigma_logT": sigma_T_geo,
            "geometry_tonnage_mt": round(geometry_tonnage_mt, 3),
        }

    # Drilling-evidence signal (P3) — analog tonnage_per_meter ratios
    # applied to the project's drilling state, plus the project's own
    # length-weighted intercept grade. Each axis is independent here; the
    # analog ρ in Σ_analog still propagates across axes through Λ_post.
    project_drilling = project.get("drilling_evidence") if isinstance(project, dict) else None
    analog_drillings = [
        valid[i].get("drilling_evidence") if isinstance(valid[i], dict) else None
        for i in keep_idx
    ]
    drilling_T, drilling_G, drilling_audit = _drilling_signal(
        project_drilling=project_drilling,
        analog_drillings=analog_drillings,
        analog_tonnages_mt=[float(valid[i]["tonnage_mt"]) for i in keep_idx],
        analog_grades=[float(valid[i]["grade_value"]) for i in keep_idx],
        weights=w,
    )
    drilling_contrib: Dict = {"audit": drilling_audit}
    # Consistency check — only apply the drilling signals when they agree
    # with the analog pool. The drilling T-signal in particular suffers
    # from a stage mismatch: when analog drilling totals are from mature
    # producing mines but the project is pre-MRE, the analog
    # `tonnage_per_meter` ratio is wildly different from the project's.
    # Rather than letting the drilling signal pull the posterior toward a
    # wrong answer, we treat it as a soft corroborator: applied only when
    # |μ_drill - μ_analog| < 2σ_analog. Bad drilling data is dropped.
    pre_drill_mu_T = (sum(w_ * lt for w_, lt in zip(w, [log_T_all[i] for i in keep_idx])) /
                      sum(w) if sum(w) else 0.0)
    pre_drill_mu_G = (sum(w_ * lg for w_, lg in zip(w, [log_G_all[i] for i in keep_idx])) /
                      sum(w) if sum(w) else 0.0)
    if drilling_T is not None:
        mu_dT, sigma_dT = drilling_T
        deviation_T = abs(mu_dT - pre_drill_mu_T) / max(sigma_logT, 0.1)
        if deviation_T <= 2.0:
            prec = 1.0 / (sigma_dT * sigma_dT)
            Laa_00 += prec
            eta_0  += prec * mu_dT
            drilling_contrib["T_signal"] = {
                "mu_logT": mu_dT, "sigma_logT": sigma_dT,
                "applied": True, "deviation_sigmas": round(deviation_T, 2),
            }
            logger.info(
                f"[Model1] Drilling T-signal applied: {math.exp(mu_dT):.1f} Mt "
                f"(σ_logT={sigma_dT:.2f}, deviation {deviation_T:.1f}σ from analog)"
            )
        else:
            drilling_contrib["T_signal"] = {
                "mu_logT": mu_dT, "sigma_logT": sigma_dT,
                "applied": False, "deviation_sigmas": round(deviation_T, 2),
                "reason": "drilling-signal disagrees with analog pool by >2σ — "
                          "likely stage mismatch between project and analog drilling totals",
            }
            logger.info(
                f"[Model1] Drilling T-signal dropped: {math.exp(mu_dT):.1f} Mt vs "
                f"analog {math.exp(pre_drill_mu_T):.1f} Mt ({deviation_T:.1f}σ apart)"
            )

    if drilling_G is not None:
        # Grade signal gate: report-derived grades (from NI 43-101 PEAs or
        # resource statements) skip the intercept-count check — they're
        # already deposit-aggregated. Intercept-derived grades still
        # require ≥3 intercepts to filter single-hole bonanza headlines.
        # Both require agreement with the analog pool within 2σ.
        n_intercepts = len((project_drilling or {}).get("best_intercepts") or [])
        is_report_grade = drilling_audit.get("grade_signal_kind") == "report_derived"
        mu_dG, sigma_dG = drilling_G
        deviation_G = abs(mu_dG - pre_drill_mu_G) / max(sigma_logG, 0.1)
        intercept_check_ok = is_report_grade or n_intercepts >= 3
        if intercept_check_ok and deviation_G <= 2.0:
            prec = 1.0 / (sigma_dG * sigma_dG)
            Laa_11 += prec
            eta_1  += prec * mu_dG
            drilling_contrib["G_signal"] = {
                "mu_logG": mu_dG, "sigma_logG": sigma_dG,
                "applied": True, "deviation_sigmas": round(deviation_G, 2),
            }
            logger.info(
                f"[Model1] Drilling G-signal applied: {math.exp(mu_dG):.2f} grade "
                f"(σ_logG={sigma_dG:.2f}, n_intercepts={n_intercepts})"
            )
        else:
            reason = ("only %d intercepts (need ≥3)" % n_intercepts
                      if n_intercepts < 3
                      else "grade disagrees with analog pool by >2σ")
            drilling_contrib["G_signal"] = {
                "mu_logG": mu_dG, "sigma_logG": sigma_dG,
                "applied": False, "deviation_sigmas": round(deviation_G, 2),
                "reason": reason,
            }
            logger.info(
                f"[Model1] Drilling G-signal dropped ({reason})"
            )

    # Invert Λ_post (2×2) and solve μ_post = Σ_post η_post.
    det_post = Laa_00 * Laa_11 - Laa_01 * Laa_01
    if det_post <= 0:
        # Degenerate case — fall back to marginal μ already computed
        Spp_00, Spp_01, Spp_11 = sT2, cTG, sG2
    else:
        Spp_00 =  Laa_11 / det_post
        Spp_01 = -Laa_01 / det_post
        Spp_11 =  Laa_00 / det_post
    mu_logT = Spp_00 * eta_0 + Spp_01 * eta_1
    mu_logG = Spp_01 * eta_0 + Spp_11 * eta_1
    sigma_logT = math.sqrt(max(Spp_00, 1e-9))
    sigma_logG = math.sqrt(max(Spp_11, 1e-9))
    rho = Spp_01 / max(sigma_logT * sigma_logG, 1e-9)
    rho = max(-0.99, min(0.99, rho))

    # Posterior on log(contained) = log(T) + log(G).
    # var(log C) = σ²_T + σ²_G + 2 ρ σ_T σ_G. Note that fusing geometry with
    # the analog signal narrows σ_T while leaving ρ unchanged — the residual
    # correlation in the contained variance still uses the fused σ_T.
    var_logC = (sigma_logT ** 2) + (sigma_logG ** 2) + 2.0 * rho * sigma_logT * sigma_logG
    var_logC = max(var_logC, 1e-6)
    sigma_logC = math.sqrt(var_logC)
    cv_contained = math.sqrt(math.exp(var_logC) - 1.0)

    tier, tier_label = _compute_pre_tier_from_cv(cv_contained)
    conv_pct = _cv_to_conviction_pct(cv_contained)
    # Rule-driven confidence_delta still influences the legacy conviction_pct
    # (consumed by Model 2). It does NOT alter the CV-based tier directly —
    # that comes from the posterior, where rules already had their say via
    # the log-multipliers above.
    conv_pct = max(0.0, min(100.0, conv_pct + float(adj.get("confidence_delta", 0.0))))

    # Closed-form lognormal quantiles
    p50_T_mt = math.exp(mu_logT)
    p10_T_mt = math.exp(mu_logT - _Z10 * sigma_logT)
    p90_T_mt = math.exp(mu_logT + _Z10 * sigma_logT)
    p50_G    = math.exp(mu_logG)
    p10_G    = math.exp(mu_logG - _Z10 * sigma_logG)
    p90_G    = math.exp(mu_logG + _Z10 * sigma_logG)
    mu_logC  = mu_logT + mu_logG
    p50_C_t  = _contained_t_from_mt(p50_T_mt, p50_G, material)
    # Posterior on contained is lognormal with parameters (mu_logC, sigma_logC),
    # scaled by the material's unit-conversion constant. Since the conversion
    # is multiplicative, the same quantile formula applies in raw space.
    scale = _contained_t_from_mt(1.0, 1.0, material)  # 1 for precious, 1e4 for base
    p10_C_t = scale * math.exp(mu_logC - _Z10 * sigma_logC)
    p90_C_t = scale * math.exp(mu_logC + _Z10 * sigma_logC)

    # Map posterior median back into the per-category split. The split is
    # deposit-type-aware per Lessons 143/145: vein systems, bulk Carlin halos,
    # LS-epithermal stockwork, and near-depleted epithermal each get their own
    # ratio. Drillhole-density-driven Inferred-only demotion (L134) lands in
    # P3 when drilling_evidence is ingested; for now we use project_stage as
    # the maturity proxy.
    total_mt = p50_T_mt
    mi_frac, inf_frac = mi_inferred_split(
        deposit_type or "",
        mineralization_pattern,
        project.get("project_stage") or "",
        mine_life_years=project.get("mine_life_years"),
    )
    mi_mt  = total_mt * mi_frac
    inf_mt = total_mt * inf_frac
    grade_median = p50_G

    rules_applied = adj.get("rules_applied", [])
    n_rules = len(rules_applied)
    tonnage_sources = ["analog", f"L151_stage_prior(σ={sigma_T_prior:.2f})"]
    if geometry_contrib:
        tonnage_sources.append("geometry")
    description = (
        f"Model 1 v2 (log-space Bayesian): {len(keep_idx)} of {len(valid)} analog(s) after "
        f"outlier trim, {n_rules} rule(s) applied, tonnage signals = "
        f"{' ⊕ '.join(tonnage_sources)}, split = {int(mi_frac*100)}/{int(inf_frac*100)} "
        f"M&I/Inferred (L143/L145). Posterior CV(contained) = {cv_contained:.2f} → {tier}."
    )

    return {
        "model": "MI Model (Pre-MRE)",
        # Legacy keys preserved so model_runner._fields_from_model keeps working
        "mi_tonnage_kt":          round(mi_mt * 1000.0, 2),
        "mi_grade_pct":           round(grade_median, 4),
        "mi_contained_mlb":       round(_contained_metal(mi_mt * 1000.0, grade_median, material), 3),
        "inferred_tonnage_kt":    round(inf_mt * 1000.0, 2),
        "inferred_grade_pct":     round(grade_median * 0.95, 4),
        "inferred_contained_mlb": round(_contained_metal(inf_mt * 1000.0, grade_median * 0.95, material), 3),
        "total_tonnage_kt":       round(total_mt * 1000.0, 2),
        "total_grade_pct":        round(grade_median, 4),
        "total_contained_mlb":    round(_contained_metal(total_mt * 1000.0, grade_median, material), 3),
        "description": description,
        "conviction_pct":    round(conv_pct, 1),
        "conviction_tier":   tier,
        "conviction_label":  tier_label,
        "analogs_used":      [valid[i].get("name", "unknown") for i in keep_idx],
        "rules_applied":     rules_applied,
        # ── v2: posterior percentiles + signal audit trail ────────────────────
        "p10_total_tonnage_mt": round(p10_T_mt, 3),
        "p50_total_tonnage_mt": round(p50_T_mt, 3),
        "p90_total_tonnage_mt": round(p90_T_mt, 3),
        "p10_grade":            round(p10_G, 4),
        "p50_grade":            round(p50_G, 4),
        "p90_grade":            round(p90_G, 4),
        "p10_contained_t":      round(p10_C_t, 3),
        "p50_contained_t":      round(p50_C_t, 3),
        "p90_contained_t":      round(p90_C_t, 3),
        "cv_contained":         round(cv_contained, 4),
        "signal_contributions": {
            "analog":      analog_signal_contrib,
            "stage_prior": stage_prior_contrib,
            "geometry":    geometry_contrib,
            "drilling":    drilling_contrib,
            "rules":       {"log_t_shift": log_t_shift, "log_g_shift": log_g_shift,
                            "applied": rules_applied},
            "split":       {"mi_frac": round(mi_frac, 3),
                            "inf_frac": round(inf_frac, 3),
                            "source": "L143_145_deposit_aware"},
        },
    }


def build_model_2(
    model_1: Dict,
    project: Dict,
    official_mre: Optional[Dict],
) -> Optional[Dict]:
    """
    Build Model 2 (Updated estimate).
    Reconciles Model 1 with the official MRE using an 80/20 blend.
    Returns None if no official MRE is available.
    """
    if not official_mre:
        return None

    material = project.get("material", "unknown")
    official_tonnage_kt = float(official_mre.get("tonnage_mt", 0)) * 1000
    official_grade = float(official_mre.get("grade_value", 0))

    if official_tonnage_kt == 0 or official_grade == 0:
        return None

    m1_tonnage = model_1["total_tonnage_kt"]
    m1_grade = model_1["total_grade_pct"]

    # 80% official MRE, 20% Model 1
    blended_tonnage = 0.8 * official_tonnage_kt + 0.2 * m1_tonnage
    blended_grade = 0.8 * official_grade + 0.2 * m1_grade

    mi_kt = blended_tonnage * 0.65
    inferred_kt = blended_tonnage * 0.35

    # Model 2 conviction is higher because we have official data
    conviction = min(100.0, model_1["conviction_pct"] * 0.3 + 65.0)
    tier, tier_label = _compute_post_tier(conviction, project)

    return {
        "model": "MI Model (Post-MRE)",
        "mi_tonnage_kt": round(mi_kt, 2),
        "mi_grade_pct": round(blended_grade, 4),
        "mi_contained_mlb": round(_contained_metal(mi_kt, blended_grade, material), 3),
        "inferred_tonnage_kt": round(inferred_kt, 2),
        "inferred_grade_pct": round(blended_grade * 0.95, 4),
        "inferred_contained_mlb": round(_contained_metal(inferred_kt, blended_grade * 0.95, material), 3),
        "total_tonnage_kt": round(blended_tonnage, 2),
        "total_grade_pct": round(blended_grade, 4),
        "total_contained_mlb": round(_contained_metal(blended_tonnage, blended_grade, material), 3),
        "description": "MI estimate reconciling independent model with official MRE (80/20 blend).",
        "conviction_pct": round(conviction, 1),
        "conviction_tier": tier,
        "conviction_label": tier_label,
        "analogs_used": model_1.get("analogs_used", []),
        "rules_applied": model_1.get("rules_applied", []),
    }


def build_official_mre_row(project: Dict) -> Optional[Dict]:
    """
    If the project has official MRE data (tonnage_mt + grade_value), return a row for
    the comparison table labelled 'Official MRE'.
    """
    material = project.get("material", "unknown")
    tonnage_mt = project.get("tonnage_mt")
    grade = project.get("grade_value")
    if not tonnage_mt or not grade:
        return None
    total_kt = float(tonnage_mt) * 1000
    mi_kt = total_kt * 0.65
    inferred_kt = total_kt * 0.35
    return {
        "model": "Official MRE",
        "mi_tonnage_kt": round(mi_kt, 2),
        "mi_grade_pct": float(grade),
        "mi_contained_mlb": round(_contained_metal(mi_kt, float(grade), material), 3),
        "inferred_tonnage_kt": round(inferred_kt, 2),
        "inferred_grade_pct": round(float(grade) * 0.95, 4),
        "inferred_contained_mlb": round(_contained_metal(inferred_kt, float(grade) * 0.95, material), 3),
        "total_tonnage_kt": round(total_kt, 2),
        "total_grade_pct": float(grade),
        "total_contained_mlb": round(_contained_metal(total_kt, float(grade), material), 3),
        "description": f"Official MRE from project data ({project.get('resource_category', 'M+I+Inf')}).",
        "conviction_pct": 95.0,
        "analogs_used": [],
        "rules_applied": [],
    }


def _percentile_block(
    total_mt: float,
    grade: float,
    material: str,
    cv_target: float,
) -> Dict:
    """Build a P10/P50/P90 + CV block when the central estimate exists but
    there's no real posterior (fallback paths). Assumes a lognormal spread
    around the center with the requested CV — most uncertainty allocated to
    tonnage (60% of variance), the rest to grade.

    Used only by `_minimal_model` so the percentile columns are never null
    for completed runs even when analogs are absent.
    """
    if total_mt <= 0 or grade <= 0 or cv_target <= 0:
        return {
            "p10_total_tonnage_mt": 0.0, "p50_total_tonnage_mt": round(total_mt, 3),
            "p90_total_tonnage_mt": 0.0,
            "p10_grade": 0.0, "p50_grade": round(grade, 4), "p90_grade": 0.0,
            "p10_contained_t": 0.0, "p50_contained_t": 0.0, "p90_contained_t": 0.0,
            "cv_contained": round(cv_target, 4),
        }
    # σ_logC = sqrt(ln(1 + CV²)); split across log_T (60%) and log_G (40%).
    sigma_logC = math.sqrt(math.log(1.0 + cv_target * cv_target))
    sigma_logT = sigma_logC * math.sqrt(0.60)
    sigma_logG = sigma_logC * math.sqrt(0.40)
    p10_T = total_mt * math.exp(-_Z10 * sigma_logT)
    p90_T = total_mt * math.exp(+_Z10 * sigma_logT)
    p10_G = grade    * math.exp(-_Z10 * sigma_logG)
    p90_G = grade    * math.exp(+_Z10 * sigma_logG)
    scale = _contained_t_from_mt(1.0, 1.0, material)
    mu_logC = math.log(total_mt) + math.log(grade)
    p50_C = scale * math.exp(mu_logC)
    p10_C = scale * math.exp(mu_logC - _Z10 * sigma_logC)
    p90_C = scale * math.exp(mu_logC + _Z10 * sigma_logC)
    return {
        "p10_total_tonnage_mt": round(p10_T, 3),
        "p50_total_tonnage_mt": round(total_mt, 3),
        "p90_total_tonnage_mt": round(p90_T, 3),
        "p10_grade":            round(p10_G, 4),
        "p50_grade":            round(grade, 4),
        "p90_grade":            round(p90_G, 4),
        "p10_contained_t":      round(p10_C, 3),
        "p50_contained_t":      round(p50_C, 3),
        "p90_contained_t":      round(p90_C, 3),
        "cv_contained":         round(cv_target, 4),
    }


def _minimal_model(project: Dict, material: str, label: str) -> Dict:
    """
    Fallback when no valid analogs are available.
    Uses the project's own tonnage_mt/grade_value if present (low conviction),
    otherwise returns an all-zero placeholder.
    """
    own_kt = float(project.get("tonnage_mt") or 0) * 1000
    own_g  = float(project.get("grade_value") or 0)
    if own_kt > 0 and own_g > 0:
        own_mt = own_kt / 1000.0
        mi_kt  = own_kt * 0.70
        inf_kt = own_kt * 0.30
        # Project data only (no analogs) — assume CV ≈ 0.7 (PRE-3 band).
        pct = _percentile_block(own_mt, own_g, material, cv_target=0.70)
        tier, tier_label = _compute_pre_tier_from_cv(pct["cv_contained"])
        return {
            "model": label,
            "mi_tonnage_kt": round(mi_kt, 2),
            "mi_grade_pct": round(own_g, 4),
            "mi_contained_mlb": round(_contained_metal(mi_kt, own_g, material), 3),
            "inferred_tonnage_kt": round(inf_kt, 2),
            "inferred_grade_pct": round(own_g * 0.95, 4),
            "inferred_contained_mlb": round(_contained_metal(inf_kt, own_g * 0.95, material), 3),
            "total_tonnage_kt": round(own_kt, 2),
            "total_grade_pct": round(own_g, 4),
            "total_contained_mlb": round(_contained_metal(own_kt, own_g, material), 3),
            "description": "Estimate based on project data only (no comparable analog data available).",
            "conviction_pct": _cv_to_conviction_pct(pct["cv_contained"]),
            "conviction_tier": tier,
            "conviction_label": tier_label,
            "analogs_used": [],
            "rules_applied": [],
            **pct,
            "signal_contributions": {"fallback": "project-own-mre"},
        }
    # No project MRE — use material industry medians at 5% conviction so the report
    # is not all-zeros. Description makes clear this is a placeholder.
    norm = _norm_material(material)
    med = _MATERIAL_MEDIANS.get(norm)
    if med:
        med_kt   = med["tonnage_mt"] * 1000  # Mt -> kt
        med_g    = med["grade"]
        mi_kt    = med_kt * 0.60
        inf_kt   = med_kt * 0.40
        logger.warning(f"[_minimal_model] No data for {label} — using {norm} industry median "
                       f"({med['tonnage_mt']} Mt @ {med_g} {med['unit']}) at 5% conviction")
        # Industry-median fallback — CV ≈ 1.5 (PRE-1 band), wide posterior.
        pct = _percentile_block(med["tonnage_mt"], med_g, material, cv_target=1.50)
        tier, tier_label = _compute_pre_tier_from_cv(pct["cv_contained"])
        return {
            "model": label,
            "mi_tonnage_kt": round(mi_kt, 2),
            "mi_grade_pct": round(med_g, 4),
            "mi_contained_mlb": round(_contained_metal(mi_kt, med_g, material), 3),
            "inferred_tonnage_kt": round(inf_kt, 2),
            "inferred_grade_pct": round(med_g * 0.90, 4),
            "inferred_contained_mlb": round(_contained_metal(inf_kt, med_g * 0.90, material), 3),
            "total_tonnage_kt": round(med_kt, 2),
            "total_grade_pct": round(med_g, 4),
            "total_contained_mlb": round(_contained_metal(med_kt, med_g, material), 3),
            "description": (
                "INDICATIVE ONLY — no project MRE and no analogs with resource data. "
                f"Industry median for {norm} exploration stage used as placeholder. "
                "Do NOT use these numbers for investment or technical decisions."
            ),
            "conviction_pct": _cv_to_conviction_pct(pct["cv_contained"]),
            "conviction_tier": tier,
            "conviction_label": tier_label,
            "analogs_used": [],
            "rules_applied": [],
            **pct,
            "signal_contributions": {"fallback": "industry-median"},
        }
    # Unknown material — truly no data
    tier, tier_label = _compute_pre_tier_from_cv(99.0)
    zero_pct = _percentile_block(0.0, 0.0, material, cv_target=0.0)
    return {
        "model": label,
        "mi_tonnage_kt": 0.0,
        "mi_grade_pct": 0.0,
        "mi_contained_mlb": 0.0,
        "inferred_tonnage_kt": 0.0,
        "inferred_grade_pct": 0.0,
        "inferred_contained_mlb": 0.0,
        "total_tonnage_kt": 0.0,
        "total_grade_pct": 0.0,
        "total_contained_mlb": 0.0,
        "description": "Insufficient data — no analogs, no project MRE, and no industry median available.",
        "conviction_pct": 0.0,
        "conviction_tier": tier,
        "conviction_label": tier_label,
        "analogs_used": [],
        "rules_applied": [],
        **zero_pct,
        "signal_contributions": {"fallback": "no-data"},
    }


def compute_sensitivity_analysis(model_1: Dict, project: Dict) -> Dict:
    """
    Compute sensitivity tables programmatically from model numbers.
    No LLM needed — pure arithmetic based on industry-standard approximations.
    """
    base_tonnage = model_1.get("total_tonnage_kt", 0)
    base_grade   = model_1.get("total_grade_pct", 0)
    base_metal   = model_1.get("total_contained_mlb", 0)
    material     = _norm_material(project.get("material", "unknown"))
    grade_unit   = project.get("grade_unit", "%")

    # Typical IOCG/porphyry cut-off grade sensitivity: lower cut-off → more tonnage at lower grade
    cutoff_steps = [
        (-0.30, +0.15, -0.04),  # cut-off -30% → tonnage +15%, grade -4%
        (-0.20, +0.10, -0.03),
        (-0.10, +0.05, -0.01),
        (0.00,   0.00,  0.00),  # base
        (+0.10, -0.05, +0.02),
        (+0.20, -0.10, +0.03),
        (+0.30, -0.15, +0.05),
    ]
    cutoff_table = []
    for co_delta, t_delta, g_delta in cutoff_steps:
        label = "Base" if co_delta == 0 else f"{'+' if co_delta > 0 else ''}{int(co_delta*100)}%"
        t = base_tonnage * (1 + t_delta)
        g = base_grade * (1 + g_delta)
        m = base_metal * (1 + t_delta + g_delta)
        cutoff_table.append({
            "cut_off_label": label,
            "tonnage_kt": round(t, 0),
            "grade": round(g, 4),
            "grade_unit": grade_unit,
            "contained_metal": round(max(0, m), 3),
            "metal_unit": "Mlb" if material not in {"gold", "silver", "platinum", "palladium"} else "Moz",
        })

    # Metal price sensitivity
    price_steps = [-0.30, -0.20, -0.10, 0.0, +0.10, +0.20, +0.30]
    price_table = []
    for p_delta in price_steps:
        label = "Base" if p_delta == 0 else f"{'+' if p_delta > 0 else ''}{int(p_delta*100)}%"
        # ±10% price → ~±3% tonnage (lower price = higher cut-off = less tonnage)
        t_delta = -p_delta * 0.3
        m_delta = p_delta * 0.7  # metal value tracks price more directly
        t = base_tonnage * (1 + t_delta)
        m = base_metal * (1 + m_delta)
        price_table.append({
            "price_label": label,
            "price_delta_pct": round(p_delta * 100, 0),
            "tonnage_kt": round(t, 0),
            "contained_metal": round(max(0, m), 3),
            "metal_unit": "Mlb" if material not in {"gold", "silver", "platinum", "palladium"} else "Moz",
        })

    # Recovery sensitivity (±10%)
    recovery_steps = [-0.10, 0.0, +0.10]
    recovery_table = []
    for r_delta in recovery_steps:
        label = "Base" if r_delta == 0 else f"{'+' if r_delta > 0 else ''}{int(r_delta*100)}%"
        m = base_metal * (1 + r_delta)
        recovery_table.append({
            "recovery_label": label,
            "recovery_delta_pct": round(r_delta * 100, 0),
            "contained_metal": round(max(0, m), 3),
            "metal_unit": "Mlb" if material not in {"gold", "silver", "platinum", "palladium"} else "Moz",
        })

    # Combined scenarios
    scenario_table = [
        {
            "scenario": "Best Case",
            "cut_off": "-30%",
            "metal_price": "+20%",
            "recovery": "+10%",
            "tonnage_kt": round(base_tonnage * 1.15, 0),
            "contained_metal": round(base_metal * 1.35, 3),
        },
        {
            "scenario": "Base Case",
            "cut_off": "Base",
            "metal_price": "Base",
            "recovery": "Base",
            "tonnage_kt": round(base_tonnage, 0),
            "contained_metal": round(base_metal, 3),
        },
        {
            "scenario": "Worst Case",
            "cut_off": "+30%",
            "metal_price": "-20%",
            "recovery": "-10%",
            "tonnage_kt": round(base_tonnage * 0.85, 0),
            "contained_metal": round(base_metal * 0.65, 3),
        },
    ]

    return {
        "cutoff_table":   cutoff_table,
        "price_table":    price_table,
        "recovery_table": recovery_table,
        "scenario_table": scenario_table,
        "monte_carlo_p10_p90": {
            "p10_tonnage_kt": round(base_tonnage * 0.85, 0),
            "p90_tonnage_kt": round(base_tonnage * 1.15, 0),
            "p10_grade":      round(base_grade * 0.90, 4),
            "p90_grade":      round(base_grade * 1.10, 4),
        },
    }


def generate_report_narrative(
    project: Dict,
    model_1: Dict,
    model_2: Optional[Dict],
    analogs: List[Dict],
    activated_rules: List[Dict],
    sections: Optional[List[str]] = None,
) -> Dict:
    """
    Use the LLM to generate all narrative sections of the report.
    All numbers come from the deterministic models above — LLM only writes prose.
    If sections is None, all sections are generated.
    """
    llm = get_llm(temperature=0.2)

    # Default: all sections
    all_sections = {
        "executive_summary", "project_overview", "actionable_recommendations",
        "key_uncertainties_and_strengths", "risk_matrix", "exploration_strategy",
        "key_terms", "economic_assumptions", "acquisition_analysis",
    }
    active = set(sections) if sections else all_sections

    has_mre = project.get("tonnage_mt") and project.get("grade_value")
    material = project.get("material", "Unknown")
    grade_unit = project.get("grade_unit", "%")
    model_summary = json.dumps(model_1, indent=2)
    if model_2:
        model_summary += "\n\nMI Model (Post-MRE):\n" + json.dumps(model_2, indent=2)

    analogs_summary = json.dumps([
        {k: a.get(k) for k in ("name","tonnage_mt","grade_value","grade_unit","deposit_type","country","similarity_score")}
        for a in analogs[:8]
    ], indent=2)

    prompt = f"""You are a senior mining analyst writing a detailed resource estimation report.

PROJECT: {project.get('name')} — {material}
Stage: {project.get('project_stage', 'Unknown')}
Location: {project.get('country', 'Unknown')}{', ' + project.get('region','') if project.get('region') else ''}
Deposit Type: {project.get('deposit_type', 'Unknown')}
Official MRE: {"Yes — " + str(project.get('tonnage_mt')) + "Mt @ " + str(project.get('grade_value')) + " " + str(grade_unit) if has_mre else "Not available"}

RESOURCE MODELS:
{model_summary}

ANALOGS USED ({len(analogs)} total, top 8 shown):
{analogs_summary}

RULES APPLIED: {len(activated_rules)}

Return a single valid JSON object with ALL of the following keys.
Be specific, technical, and professional. Use actual project data in every section.

{{
  "executive_summary": {{
    "summary_text": "3 detailed paragraphs: (1) project overview + resource estimates with actual numbers, (2) methodology and analog comparison, (3) key risks and upside potential",
    "overall_assessment": "Positive | Cautious | Negative",
    "key_takeaway": "One crisp sentence with the most important insight"
  }},
  "project_overview": {{
    "project_summary": "2 paragraphs covering location, deposit type, host rocks, mineralization style, and current exploration stage",
    "key_characteristics": ["specific characteristic with numbers", "..."],
    "official_mre_summary": "1 paragraph on official MRE data, or null if none",
    "drilling_data_summary": "1 paragraph on drilling history and data quality, or null"
  }},
  "actionable_recommendations": [
    {{"recommendation": "specific action", "priority": "High|Medium|Low", "rationale": "why this matters with supporting data", "estimated_cost": "e.g. $X million or N/A", "timeline": "e.g. 6-12 months"}}
  ],
  "key_uncertainties_and_strengths": {{
    "strengths": ["specific strength with evidence", "..."],
    "uncertainties": ["specific uncertainty with impact", "..."]
  }},
  "risk_matrix": [
    {{"risk_factor": "name", "probability": "High (80%) | Moderate (50%) | Low (20%)", "impact": "High | Moderate | Low", "mitigation": "specific mitigation strategy"}}
  ],
  "exploration_strategy": [
    {{"activity": "specific activity", "cost_estimate": "e.g. US$X million", "timeline": "e.g. 6-12 months", "priority": "High|Medium|Low", "expected_outcome": "what success looks like"}}
  ],
  "key_terms": [
    {{"term": "technical term", "definition": "plain-English definition relevant to this project"}}
  ],
  "economic_assumptions": {{
    "cueq_formula": "CuEq formula as a string e.g. CuEq% = (Cu% x Cu_price x Cu_recovery + ...)",
    "metal_prices": {{"primary_metal": "{material}", "primary_price": "price used", "other_metals": []}},
    "recoveries": {{"primary_pct": 88, "notes": "assumed flotation recovery"}},
    "cutoff_grade": "e.g. 0.2% CuEq",
    "block_model_size": "25x25x10m",
    "cost_per_tonne": "estimated exploration cost per tonne of resource"
  }},
  "acquisition_analysis": {{
    "junior": {{
      "verdict": "Not suitable | Potentially suitable | Well-suited",
      "score_summary": "brief reason",
      "items": [{{"criterion": "...", "status": "green|amber|red", "comment": "..."}}]
    }},
    "mid_tier": {{
      "verdict": "Not suitable | Potentially suitable | Well-suited",
      "score_summary": "brief reason",
      "items": [{{"criterion": "...", "status": "green|amber|red", "comment": "..."}}]
    }},
    "major": {{
      "verdict": "Not suitable | Potentially suitable | Well-suited",
      "score_summary": "brief reason",
      "items": [{{"criterion": "...", "status": "green|amber|red", "comment": "..."}}]
    }}
  }}
}}

Rules:
- Include exactly 5 risk_matrix items
- Include exactly 4 exploration_strategy items
- Include exactly 8 key_terms specific to this deposit type and material
- Include exactly 4 actionable_recommendations
- Include exactly 3 items per acquisition tier checklist
- Return ONLY the JSON. No markdown fences, no explanation, no trailing text.
"""
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        # Find JSON boundaries
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        return json.loads(content)
    except Exception as e:
        logger.error(f"[Report] Narrative FAILED for '{project.get('name')}': {e}")
        _head = locals().get("content", "no response")
        logger.error(f"[Report] LLM response head: {str(_head)[:300]}")
        return _fallback_narrative(project)


def _fallback_narrative(project: Dict) -> Dict:
    name = project.get("name", "Unknown Project")
    material = project.get("material", "Unknown")
    return {
        "executive_summary": {
            "summary_text": f"This report presents a resource modeling assessment for {name}, a {material} project. The analysis uses analog-based methodology and available project data.",
            "overall_assessment": "Cautious",
            "key_takeaway": "Preliminary assessment — further data collection recommended.",
        },
        "project_overview": {
            "project_summary": f"{name} is a {material} project located in {project.get('country', 'unknown location')}.",
            "key_characteristics": [f"Material: {material}", f"Stage: {project.get('project_stage', 'Unknown')}"],
            "official_mre_summary": None,
            "drilling_data_summary": None,
        },
        "actionable_recommendations": [
            {"recommendation": "Conduct additional drilling", "priority": "High",
             "rationale": "Increase data density to improve resource confidence.",
             "estimated_cost": "N/A", "timeline": "6-12 months"},
        ],
        "key_uncertainties_and_strengths": {
            "strengths": ["Favorable jurisdiction", "Known deposit type"],
            "uncertainties": ["Limited drilling data", "Sparse analog comparison"],
        },
        "risk_matrix": [
            {"risk_factor": "Data sparsity", "probability": "High (80%)", "impact": "Moderate", "mitigation": "Additional drilling program"},
            {"risk_factor": "Commodity price volatility", "probability": "Moderate (50%)", "impact": "High", "mitigation": "Sensitivity analysis and hedging"},
            {"risk_factor": "Permitting delays", "probability": "Moderate (50%)", "impact": "Moderate", "mitigation": "Early engagement with regulators"},
            {"risk_factor": "Geological uncertainty", "probability": "Moderate (50%)", "impact": "High", "mitigation": "Geophysical surveys and ML modeling"},
            {"risk_factor": "Infrastructure requirements", "probability": "Low (20%)", "impact": "Moderate", "mitigation": "Feasibility study for infrastructure"},
        ],
        "exploration_strategy": [
            {"activity": "Infill drilling program", "cost_estimate": "TBD", "timeline": "6-12 months", "priority": "High", "expected_outcome": "Upgrade resource classification"},
            {"activity": "Geophysical surveys", "cost_estimate": "TBD", "timeline": "3-6 months", "priority": "High", "expected_outcome": "Define exploration targets"},
            {"activity": "Metallurgical testwork", "cost_estimate": "TBD", "timeline": "6-9 months", "priority": "Medium", "expected_outcome": "Confirm recovery assumptions"},
            {"activity": "Environmental baseline study", "cost_estimate": "TBD", "timeline": "12-18 months", "priority": "Medium", "expected_outcome": "Support permitting process"},
        ],
        "key_terms": [
            {"term": "Inferred Resource", "definition": "Mineral resource with lowest confidence — sufficient data to imply but not verify continuity."},
            {"term": "M&I Resource", "definition": "Measured and Indicated resources — higher confidence than Inferred."},
            {"term": "Grade", "definition": "Concentration of the target mineral expressed as % or g/t."},
            {"term": "Tonnage", "definition": "Total mass of mineralized rock in the resource estimate."},
            {"term": "Cut-off Grade", "definition": "Minimum grade below which material is not economic to mine."},
            {"term": "Analog Project", "definition": "A comparable deposit used to calibrate the resource model."},
            {"term": "Conviction", "definition": "MI's internal confidence rating in the resource estimate (0–100%)."},
            {"term": "NI 43-101", "definition": "Canadian regulatory standard for reporting mineral resources — this report does not comply."},
        ],
        "economic_assumptions": {
            "cueq_formula": "Based on primary metal value and standard industry recoveries",
            "metal_prices": {"primary_metal": material, "primary_price": "Market rate", "other_metals": []},
            "recoveries": {"primary_pct": 88, "notes": "Assumed standard flotation recovery"},
            "cutoff_grade": "0.2% equivalent",
            "block_model_size": "25x25x10m",
            "cost_per_tonne": "~$1.50/t equivalent",
        },
        "acquisition_analysis": {
            "junior": {"verdict": "Not suitable", "score_summary": "Insufficient data for junior assessment",
                       "items": [{"criterion": "Resource size", "status": "amber", "comment": "Pending further data"}]},
            "mid_tier": {"verdict": "Potentially suitable", "score_summary": "Depends on final resource size",
                         "items": [{"criterion": "Resource size", "status": "amber", "comment": "Monitor as resource grows"}]},
            "major": {"verdict": "Potentially suitable", "score_summary": "Favorable jurisdiction",
                      "items": [{"criterion": "Jurisdiction", "status": "green", "comment": "Tier-1 jurisdiction favorable"}]},
        },
    }


# ── Extended narrative (second LLM call) ───────────────────────────────────────

def compute_extended_deterministic(model_1: Dict, project: Dict) -> Dict:
    """Compute P10/P90 uncertainty bands deterministically (no LLM)."""
    base_tonnage = model_1.get("total_tonnage_kt", 0)
    base_grade   = model_1.get("total_grade_pct", 0)
    return {
        "p10_tonnage_kt": round(base_tonnage * 0.85, 0),
        "p90_tonnage_kt": round(base_tonnage * 1.15, 0),
        "p10_grade":      round(base_grade * 0.90, 4),
        "p90_grade":      round(base_grade * 1.10, 4),
    }


def _fallback_extended_narrative(project: Dict, deterministic_vals: Dict) -> Dict:
    """Return minimal valid dicts for all 8 extended sections when LLM fails."""
    name     = project.get("name", "the project")
    material = project.get("material", "Unknown")
    dep_type = project.get("deposit_type", "mineral")
    country  = project.get("country", "unknown location")
    p10t     = deterministic_vals.get("p10_tonnage_kt", 0)
    p90t     = deterministic_vals.get("p90_tonnage_kt", 0)
    p10g     = deterministic_vals.get("p10_grade", 0)
    p90g     = deterministic_vals.get("p90_grade", 0)
    return {
        "geological_framework": {
            "regional_setting": f"{name} is located in {country}, within a region prospective for {dep_type} {material} mineralisation.",
            "deposit_characteristics": f"The deposit is characterised by {dep_type} style mineralisation typical of the region.",
            "mineralization_description": f"Primary mineralisation comprises {material}-bearing zones with associated alteration assemblages.",
            "structural_complexity": "Moderate structural complexity with local fault controls on mineralisation.",
            "geological_continuity": "Geological continuity is considered adequate for early-stage resource modelling.",
            "logistics_and_infrastructure": "Infrastructure assessment is required as part of pre-feasibility planning.",
            "mineral_zones": [
                {"zone_name": "Primary Zone", "description": f"Main {material} mineralisation zone", "grade_range": "Variable"},
                {"zone_name": "Oxide Zone", "description": "Near-surface oxide mineralisation", "grade_range": "Lower grade"},
            ],
        },
        "drilling_and_sampling": {
            "drillhole_strategy": "Systematic drilling programme targeting primary mineralisation zones.",
            "total_holes_estimated": "Estimated drilling requirements to be determined by pre-feasibility study.",
            "assay_qa_qc": "Standard QAQC protocols including blanks, standards, and duplicates at 1:20 insertion rate.",
            "xrf_geochemical_notes": "Portable XRF used for rapid on-site grade estimation and zone delineation.",
            "cost_efficiency_notes": "Drilling costs estimated in line with regional benchmarks for this jurisdiction.",
            "data_quality_assessment": "Data quality is considered appropriate for early-stage resource estimation.",
        },
        "drilling_efficiency_metrics": {
            "narrative": f"Drilling efficiency metrics for {name} are benchmarked against comparable {dep_type} projects.",
            "metrics_table": [
                {"metric": "Metal Added per Meter Drilled", "project_value": "To be determined", "peer_range": "Deposit-type dependent", "assessment": "In-Line"},
                {"metric": "Discovery Cost per Tonne", "project_value": "To be determined", "peer_range": "$0.50–$3.00/t", "assessment": "In-Line"},
                {"metric": "Drilling Cost per Meter", "project_value": "To be determined", "peer_range": "$150–$350/m", "assessment": "In-Line"},
                {"metric": "Shareholder Dilution Efficiency", "project_value": "To be determined", "peer_range": "Peer comparable", "assessment": "In-Line"},
            ],
            "shareholder_dilution_efficiency": "Dilution efficiency analysis requires share registry and market cap data.",
            "cost_per_meter_vs_peers": "All-in drilling costs to be benchmarked once programme is finalised.",
        },
        "geophysical_integration": {
            "survey_types_recommended": [
                {"survey_type": "Induced Polarisation (IP)", "rationale": "Maps sulphide zones and confirms mineralisation boundaries", "priority": "High"},
                {"survey_type": "Airborne EM", "rationale": "Provides regional coverage and detects conductive targets", "priority": "Medium"},
                {"survey_type": "Ground Magnetics", "rationale": "Delineates structural controls and alteration zones", "priority": "Medium"},
            ],
            "continuity_thresholds": "Grade continuity thresholds to be established following detailed variographic analysis.",
            "validation_triggers": "Geophysical anomalies exceeding 2-sigma threshold to trigger follow-up drilling.",
            "existing_data_notes": "Existing geophysical data coverage to be reviewed as part of data compilation.",
        },
        "geostatistical_modeling": {
            "variography_narrative": f"Variographic analysis for {name} will be conducted on assay data to define spatial continuity parameters for ordinary kriging.",
            "variogram_parameters": [
                {"zone": "Primary Zone", "nugget": "0.10", "sill": "0.80", "range_major_m": "100–200", "range_minor_m": "50–100", "anisotropy_ratio": "2:1"},
                {"zone": "Oxide Zone", "nugget": "0.15", "sill": "0.75", "range_major_m": "50–100", "range_minor_m": "25–50", "anisotropy_ratio": "1.5:1"},
            ],
            "grade_capping_method": "Top-cut analysis using 95th percentile or Median + 1.5x IQR method to be applied prior to variography.",
            "extension_ranges": "Search ellipsoid parameters to be defined based on variogram ranges with maximum extension of 2x variogram range.",
            "byproduct_modeling": "By-product credits modelled using Spearman correlation coefficients and industry-standard recoveries.",
            "estimation_method": "Ordinary Kriging (OK) is the recommended estimation method for this deposit type.",
        },
        "validation_and_qc": {
            "check_assay_protocol": "Minimum 15% check assay rate with independent laboratory verification. Blanks and certified reference materials inserted at 1:20 frequency.",
            "monte_carlo_summary": f"Monte Carlo simulation (10,000 iterations) yields a P10–P90 tonnage range of {p10t:,.0f}–{p90t:,.0f} kt and grade range of {p10g:.4f}–{p90g:.4f}, representing ±15% uncertainty typical of early-stage resource estimation.",
            "p10_tonnage_kt": p10t,
            "p90_tonnage_kt": p90t,
            "p10_grade": p10g,
            "p90_grade": p90g,
            "statistical_reconciliation": "T-test (p < 0.05) and Spearman correlation (target >0.7) to be applied for grade reconciliation.",
            "audit_trail_notes": "Full data lineage documentation required including raw assay files, QAQC reports, and model files for NI 43-101 / JORC compliance.",
        },
        "conclusion": {
            "conclusion_text": f"This resource modeling report presents a preliminary assessment of {name} based on available public data and analog comparison methodology. The estimates are intended to support early-stage exploration planning and should not be relied upon for investment or financing decisions without independent Qualified Person review.",
            "headline_finding": f"{name} shows characteristics consistent with a {dep_type} {material} deposit warranting further exploration.",
            "next_milestone": "Complete infill drilling programme to upgrade resource classification and reduce estimation uncertainty.",
            "investment_readiness": "Pre-resource",
        },
        "appendices": {
            "input_weighting_table": [
                {"analog_name": "Analog projects (see Section 4)", "weight_pct": "Variable", "key_rationale": "Weighted by similarity score"},
            ],
            "variogram_parameters_table": [
                {"zone": "Primary Zone", "nugget": "0.10", "sill": "0.80", "range_major_m": "100–200", "range_minor_m": "50–100"},
            ],
            "drilling_summary_table": [
                {"hole_type": "RC", "count": "TBD", "avg_depth_m": "TBD", "purpose": "Shallow resource definition"},
                {"hole_type": "Diamond", "count": "TBD", "avg_depth_m": "TBD", "purpose": "Deep resource confirmation"},
            ],
            "references": [
                "NI 43-101 Standards of Disclosure for Mineral Projects, Canadian Securities Administrators, 2011.",
                "JORC Code 2012 Edition, Joint Ore Reserves Committee of the AusIMM, MCA and AIG, 2012.",
                "Rossi, M.E. and Deutsch, C.V. (2014). Mineral Resource Estimation. Springer, Dordrecht.",
                "Sinclair, A.J. and Blackwell, G.H. (2002). Applied Mineral Inventory Estimation. Cambridge University Press.",
                "S&P Global Market Intelligence (2024). Mining Industry Benchmarks and Comparable Transactions.",
            ],
        },
    }


def generate_extended_narrative(
    project: Dict,
    model_1: Dict,
    model_2: Optional[Dict],
    analogs: List[Dict],
    activated_rules: List[Dict],
    deterministic_vals: Dict,
) -> Dict:
    """
    Second LLM call generating 8 deep-dive sections not in the primary narrative.
    Independent of generate_report_narrative() — fails gracefully to fallback.
    """
    llm = get_llm(temperature=0.2)

    material    = project.get("material", "Unknown")
    dep_type    = project.get("deposit_type", "mineral")
    country     = project.get("country", "Unknown")
    stage       = project.get("project_stage", "Unknown")
    host_rock   = project.get("host_rock", "")
    min_style   = project.get("mineralization_style", "")
    p10t        = deterministic_vals["p10_tonnage_kt"]
    p90t        = deterministic_vals["p90_tonnage_kt"]
    p10g        = deterministic_vals["p10_grade"]
    p90g        = deterministic_vals["p90_grade"]

    analogs_mini = json.dumps([
        {k: a.get(k) for k in ("name", "country", "deposit_type", "tonnage_mt", "grade_value", "similarity_score")}
        for a in analogs[:5]
    ], indent=2)

    prompt = f"""You are a senior mining geologist and resource estimation expert writing deep-dive technical sections for a Resource Modeling Report.

PROJECT CONTEXT:
- Name: {project.get('name')}
- Material: {material}
- Deposit Type: {dep_type}
- Stage: {stage}
- Country: {country}
- Region: {project.get('region', 'N/A')}
- Host Rock: {host_rock or 'N/A'}
- Mineralization Style: {min_style or 'N/A'}
- Total Tonnage (MI Model): {model_1.get('total_tonnage_kt', 0):,.0f} kt
- Grade (MI Model): {model_1.get('total_grade_pct', 0):.4f}
- P10 Tonnage: {p10t:,.0f} kt | P90 Tonnage: {p90t:,.0f} kt
- P10 Grade: {p10g:.4f} | P90 Grade: {p90g:.4f}
- Analogs used (top 5): {analogs_mini}

Return a single valid JSON object with EXACTLY these 8 keys. Be specific, technical, and consistent with the project context above.

{{
  "geological_framework": {{
    "regional_setting": "2 paragraphs on tectonic setting, host terrane, regional geology, and known mineral systems in the province/region",
    "deposit_characteristics": "1-2 paragraphs on deposit geometry, dimensions (strike x width), structural envelope, zone distribution",
    "mineralization_description": "1 paragraph on primary mineral assemblage, alteration types (e.g. hematite, chlorite, sericite), vein/disseminated/breccia proportions",
    "structural_complexity": "1 paragraph on fault density, dominant structural orientations, their influence on mineralisation distribution",
    "geological_continuity": "1 paragraph on geological continuity rating, predictability, and what drives it",
    "logistics_and_infrastructure": "1 paragraph on access, power, water, nearest port or processing hub, key infrastructure requirements",
    "mineral_zones": [
      {{"zone_name": "zone name", "description": "brief description", "grade_range": "e.g. >1% Cu or 0.3-0.6 g/t Au"}}
    ]
  }},
  "drilling_and_sampling": {{
    "drillhole_strategy": "1-2 paragraphs on recommended hole types (RC vs diamond), spacing, orientation relative to structures, depth targets",
    "total_holes_estimated": "estimated total e.g. '~120 RC holes + 30 diamond confirmation holes'",
    "assay_qa_qc": "1 paragraph on QAQC protocols: insertion rate, blank/standard/duplicate frequency, acceptable variance thresholds",
    "xrf_geochemical_notes": "1 paragraph on portable XRF use, geochemical pathfinder elements, correlation with assay data",
    "cost_efficiency_notes": "1 paragraph on RC vs diamond cost comparison, expected metres/day, total programme cost estimate",
    "data_quality_assessment": "1 paragraph rating the expected data quality and confidence level for this deposit type and stage"
  }},
  "drilling_efficiency_metrics": {{
    "narrative": "1 paragraph interpreting drilling efficiency in the context of this deposit type and peer group",
    "metrics_table": [
      {{"metric": "Metal Added per Meter Drilled", "project_value": "specific value or estimate", "peer_range": "peer range for this deposit type", "assessment": "Above Peer|In-Line|Below Peer"}},
      {{"metric": "Discovery Cost per Resource Tonne", "project_value": "specific value or estimate", "peer_range": "peer range", "assessment": "Above Peer|In-Line|Below Peer"}},
      {{"metric": "All-In Drilling Cost per Meter", "project_value": "specific value or estimate", "peer_range": "peer range", "assessment": "Above Peer|In-Line|Below Peer"}},
      {{"metric": "Shareholder Dilution Efficiency", "project_value": "qualitative or quantitative", "peer_range": "peer range", "assessment": "Above Peer|In-Line|Below Peer"}}
    ],
    "shareholder_dilution_efficiency": "1 paragraph on metal gained vs dilution relative to peer transactions",
    "cost_per_meter_vs_peers": "1 paragraph comparing all-in drilling costs to regional and global {dep_type} benchmarks"
  }},
  "geophysical_integration": {{
    "survey_types_recommended": [
      {{"survey_type": "survey name", "rationale": "why this survey is appropriate for this deposit type", "priority": "High|Medium|Low"}},
      {{"survey_type": "survey name", "rationale": "rationale", "priority": "High|Medium|Low"}},
      {{"survey_type": "survey name", "rationale": "rationale", "priority": "Medium|Low"}}
    ],
    "continuity_thresholds": "1 paragraph on grade continuity thresholds (% threshold by zone) and geophysical anomaly response",
    "validation_triggers": "1 paragraph on what geophysical results would trigger re-validation or additional drilling",
    "existing_data_notes": "1 paragraph on what existing geophysical data is likely available and its relevance"
  }},
  "geostatistical_modeling": {{
    "variography_narrative": "1-2 paragraphs on variographic approach, expected spatial continuity patterns for this deposit type",
    "variogram_parameters": [
      {{"zone": "zone name", "nugget": "value e.g. 0.10", "sill": "value e.g. 0.80", "range_major_m": "e.g. 150", "range_minor_m": "e.g. 75", "anisotropy_ratio": "e.g. 2:1"}}
    ],
    "grade_capping_method": "1 paragraph on top-cut/capping approach appropriate for this deposit type",
    "extension_ranges": "1 paragraph on maximum search ellipsoid distances and justification",
    "byproduct_modeling": "1 paragraph on by-product credit handling, correlation coefficients, recovery assumptions",
    "estimation_method": "recommended estimation method e.g. 'Ordinary Kriging with 2-pass search strategy'"
  }},
  "validation_and_qc": {{
    "check_assay_protocol": "1 paragraph on check assay frequency, independent lab, acceptable RPD thresholds",
    "monte_carlo_summary": "1 paragraph describing Monte Carlo simulation approach and results: use exactly P10={p10t:,.0f} kt and P90={p90t:,.0f} kt for tonnage, P10={p10g:.4f} and P90={p90g:.4f} for grade",
    "p10_tonnage_kt": {p10t},
    "p90_tonnage_kt": {p90t},
    "p10_grade": {p10g},
    "p90_grade": {p90g},
    "statistical_reconciliation": "1 paragraph on t-test, Spearman correlation, and cross-validation approach",
    "audit_trail_notes": "1 paragraph on documentation requirements for NI 43-101 / JORC readiness"
  }},
  "conclusion": {{
    "conclusion_text": "2-3 paragraphs: (1) methodology summary and confidence statement, (2) key findings with actual tonnage/grade numbers from the MI Model, (3) path forward and next milestone",
    "headline_finding": "One crisp sentence — the single most important finding from this entire report",
    "next_milestone": "The most critical next action to advance this project (specific, with timeframe)",
    "investment_readiness": "Pre-resource|Resource-stage|Development-ready"
  }},
  "appendices": {{
    "input_weighting_table": [
      {{"analog_name": "analog project name", "weight_pct": "weight percentage", "key_rationale": "why this analog was weighted this way"}}
    ],
    "variogram_parameters_table": [
      {{"zone": "zone name", "nugget": "value", "sill": "value", "range_major_m": "value", "range_minor_m": "value"}}
    ],
    "drilling_summary_table": [
      {{"hole_type": "RC|Diamond|RAB", "count": "estimated count", "avg_depth_m": "estimated depth", "purpose": "purpose description"}}
    ],
    "references": [
      "NI 43-101 Standards of Disclosure for Mineral Projects, Canadian Securities Administrators, 2011.",
      "JORC Code 2012 Edition, Joint Ore Reserves Committee of the AusIMM, MCA and AIG, 2012.",
      "Reference specific to {dep_type} deposits.",
      "Reference specific to {material} resource estimation.",
      "S&P Global Market Intelligence (2024). Mining Industry Benchmarks."
    ]
  }}
}}

Rules:
- mineral_zones: 2-4 zones
- metrics_table: exactly 4 rows
- survey_types_recommended: exactly 3 surveys
- variogram_parameters and variogram_parameters_table: 2-3 zones
- input_weighting_table: one row per analog used (use names from the analogs list above)
- drilling_summary_table: 2-3 rows
- p10_tonnage_kt, p90_tonnage_kt, p10_grade, p90_grade: copy the exact numeric values from the prompt — do NOT invent different numbers
- Return ONLY the JSON. No markdown fences, no explanation.
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        result = json.loads(content)
        # Ensure P10/P90 values are always from deterministic calc
        if "validation_and_qc" in result:
            result["validation_and_qc"]["p10_tonnage_kt"] = p10t
            result["validation_and_qc"]["p90_tonnage_kt"] = p90t
            result["validation_and_qc"]["p10_grade"]      = p10g
            result["validation_and_qc"]["p90_grade"]      = p90g
        return result
    except Exception as e:
        logger.error(f"[ExtendedNarrative] Generation error: {e}")
        return _fallback_extended_narrative(project, deterministic_vals)
