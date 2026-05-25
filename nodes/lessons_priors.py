"""Lessons-derived priors and split rules for Model 1 v2.

Source: `/Users/visheshjain/Documents/Mining intellect/Lessons Learned/Gold
and Silver Lessons Learned revised.docx` — the revised gold/silver lessons
document the user manages by hand. The rules here are direct transcriptions
of the numeric guidance in those lessons, structured so the model can apply
them at inference and so the recalibration job (P5) can refit them as
prediction outcomes accumulate.

Two helpers are exposed:

1. `stage_tonnage_prior(material, deposit_type, project_stage)` →
   `(mu_log_mt, sigma_log_mt)` — the L151 stage × deposit-type tonnage
   prior in log-space. Used as a fused signal alongside the analog pool
   and geometry to prevent Model 1 from predicting a 500 Mt resource for
   an early-stage vein project (or 5 Mt for a mature porphyry).

2. `mi_inferred_split(deposit_type, mineralization_pattern, project_stage)`
   → `(mi_frac, inf_frac)` — the L143/L145 deposit-type-aware M&I/Inferred
   split. Replaces the previous hard-coded 70/30 with rules that reflect
   how analogous deposits classify resource categories at the same stage.

Both functions return defaults rather than raising on unknown inputs, so a
new or unclassifiable project still gets a finite estimate — just one that
falls back to material-level industry averages with a wider σ.

When the P2 `deposit_type_priors` table exists, this module's hard-coded
tables become the seed data for that table and the live snapshot takes
over.
"""
from __future__ import annotations
import math
from typing import Dict, Tuple


# ── L151: Stage × deposit-type tonnage ranges (Mt) ───────────────────────────
# Each entry is a (low_p10, high_p90) tuple representing the 80% credible
# interval for typical total resource tonnage at that maturity. Ranges are
# transcribed verbatim from the Gold/Silver Lessons Learned revised doc,
# Lesson 151 — "Stage-Specific Tonnage Guidance". The default σ comes from
# the log-space half-width of each range (z=1.2816 for the 80% band).

_STAGE_TONNAGE_RANGES_MT: Dict[Tuple[str, str], Tuple[float, float]] = {
    # Vein systems (orogenic, epithermal LS/HS veins, CRD/manto veins)
    ("vein", "early"):    (5.0,   20.0),
    ("vein", "mid"):      (5.0,   25.0),
    ("vein", "mature"):   (5.0,   15.0),
    # Bulk Au-Ag (Carlin disseminated, bulk epithermal, RIRGS, heap-leach)
    ("bulk", "early"):    (20.0,  100.0),
    ("bulk", "mid"):      (50.0,  200.0),
    ("bulk", "mature"):   (100.0, 500.0),
    # Porphyry / IOCG / skarn copper-gold
    ("porphyry", "early"):  (50.0,  300.0),
    ("porphyry", "mid"):    (100.0, 800.0),
    ("porphyry", "mature"): (200.0, 1500.0),
}

# Material-level conservative fallback when (material, deposit_type, stage)
# can't be resolved — used so the prior signal still contributes rather than
# collapsing to ∞ variance. σ here is wide on purpose; the analog pool
# dominates whenever it's well-populated.
_MATERIAL_FALLBACK_RANGE_MT: Dict[str, Tuple[float, float]] = {
    "gold":      (3.0,  300.0),
    "silver":    (3.0,  300.0),
    "copper":    (30.0, 1500.0),
    "lead":      (3.0,  150.0),
    "zinc":      (3.0,  150.0),
    "nickel":    (10.0, 500.0),
    "uranium":   (1.0,  100.0),
    "platinum":  (10.0, 500.0),
    "palladium": (10.0, 500.0),
    "iron":      (100.0, 5000.0),
    "molybdenum":(20.0, 1000.0),
}

_Z80 = 1.2815515655446004  # standard normal z for P10/P90


def _classify_deposit_family(deposit_type: str, mineralization_pattern: str):
    """Map freeform deposit_type + mineralization_pattern strings to one of
    the three families the lesson tables index on: vein / bulk / porphyry.
    Returns None when there's no information at all to classify on —
    callers fall back to the wide material-level prior in that case
    rather than mis-applying a "bulk" default (which drove the Hammerdown
    backtest +455% error).

    Resolution order: the controlled-vocab `mineralization_pattern` slug
    wins over the freeform `deposit_type` text. A project whose pattern
    is `disseminated_bulk` is a bulk system regardless of whether its
    deposit_type says "orogenic" — orogenic gold systems include both
    bulk (Detour Lake, Canadian Malartic, Fenn-Gib) and vein-hosted
    (Brucejack, True North) types.
    """
    dt = (deposit_type or "").lower().strip()
    mp = (mineralization_pattern or "").lower().strip()

    # No signal at all — let the caller fall back to the material-wide range.
    if not dt and not mp:
        return None

    # Porphyry / IOCG / skarn — deposit_type is decisive
    if any(k in dt for k in ("porphyry", "iocg", "skarn")):
        return "porphyry"

    # Explicit bulk-pattern slugs win — the project_research/analog_finder
    # writes `disseminated_bulk` for systems that are bulk-tonnage regardless
    # of geological family. Same for "stockwork" and "halo" indicators.
    if any(b in mp for b in ("disseminated", "bulk", "stockwork", "halo")):
        return "bulk"
    if any(b in dt for b in ("disseminated", "bulk", "carlin", "halo",
                              "heap-leach", "heap leach")):
        return "bulk"

    # Vein indicators in pattern, then in deposit_type
    if "vein" in mp or mp in {"vein_hosted", "vein-hosted"}:
        return "vein"
    if any(k in dt for k in ("orogenic", "epithermal vein", "ls epithermal vein",
                              "hs epithermal vein", "crd", "manto",
                              "low-sulfidation epithermal", "high-sulfidation epithermal")):
        return "vein"

    # Some signal but ambiguous — bulk is still the broadest catch-all
    return "bulk"


def _classify_stage(project_stage: str) -> str:
    """Map freeform project_stage strings to one of the three lesson tiers:
    early / mid / mature.

    Mirrors the `_compute_post_tier` stage matching in `model_builder.py` so
    a project landing in POST-5 (Feasibility) here lands in `mature`, and
    early-exploration / target-generation lands in `early`. Tokens that look
    like abbreviations (BFS, PFS, PEA, DFS) are matched on space-bounded
    boundaries so they don't accidentally match inside another word.
    """
    s = (project_stage or "").lower().strip()
    if not s:
        return "mid"  # default to the broadest band when stage is unknown
    # Pad so " pfs " matches both bare "PFS" and "Phase 1 PFS".
    padded = f" {s} "

    if any(p in padded for p in (
        "feasibility study", "bankable", " bfs ", " dfs ", " fs ",
        "definitive feasibility", "production", "construction",
        "operating", "operation", "operational", "producing", "in production",
    )):
        return "mature"
    if any(p in padded for p in (
        "pre-feasibility", "prefeasibility", " pfs ", " pea ",
        "scoping", "preliminary economic", "economic assessment",
        "preliminary assessment", "advanced exploration",
        "resource definition", "infill",
    )):
        return "mid"
    return "early"


def stage_tonnage_prior(
    material: str,
    deposit_type: str,
    project_stage: str,
    mineralization_pattern: str = "",
) -> Tuple[float, float]:
    """Return `(mu_log_tonnage_mt, sigma_log_tonnage_mt)` per Lesson 151.

    Combines deposit-type family classification with stage tier classification
    to look up the 80% credible interval, then converts to log-space (μ, σ).
    Unknown combinations fall back to a wide material-level range so the
    signal still contributes something rather than being absent.
    """
    family = _classify_deposit_family(deposit_type, mineralization_pattern)
    stage = _classify_stage(project_stage)
    # When the classifier returned None (no signal at all), or the
    # (family, stage) lookup misses, drop to the material-wide range. This
    # is intentionally wide so the analog signal dominates — better than
    # mis-applying a tight prior centred on the wrong scale.
    rng = _STAGE_TONNAGE_RANGES_MT.get((family, stage)) if family else None
    if rng is None:
        rng = _MATERIAL_FALLBACK_RANGE_MT.get(
            (material or "").strip().lower(),
            (10.0, 1000.0),
        )
    lo, hi = rng
    if lo <= 0 or hi <= 0 or hi <= lo:
        return 0.0, 1.5  # degenerate guard
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    mu = 0.5 * (log_lo + log_hi)
    sigma = (log_hi - log_lo) / (2.0 * _Z80)
    return mu, sigma


# ── L143/L145: Deposit-type-aware M&I / Inferred split ───────────────────────
# Default is 70/30 (the prior). Specific overrides come from Lessons 143 and
# 145 — each captures a real adjustment the lessons doc cites against
# historical projects (Goldboro, La Coipa, San Jose, etc.).

def mi_inferred_split(
    deposit_type: str,
    mineralization_pattern: str,
    project_stage: str,
    mine_life_years: float = None,
) -> Tuple[float, float]:
    """Return `(mi_frac, inf_frac)` summing to 1.0.

    Order of precedence (most specific first):
      1. Mature near-depleted epithermal vein with <2yr mine life
         → M&I 0.87, Inferred 0.13  (L143: "Inferred = 10–15% of M&I")
      2. Bulk Carlin/orogenic with low-grade halos at mid+ stage
         → M&I 0.80, Inferred 0.20  (L143: 60–90% M&I — pick midpoint 80)
      3. LS epithermal stockwork / breccia at mid stage
         → M&I 0.80, Inferred 0.20  (L143/L145: 15–25% Inferred for mid-stage)
      4. HS epithermal mature/depleted with >10 Moz district history
         → M&I 0.77, Inferred 0.23  (L143: 15–30% Inferred)
      5. Early-stage anything → M&I 0.40, Inferred 0.60
         (L143: early-stage with sparse drilling defaults to Inferred-only or
          near-Inferred; not enforced as zero so the model still produces an
          M&I figure for the back-compat columns)
      6. Default → M&I 0.70, Inferred 0.30
    """
    dt = (deposit_type or "").lower()
    mp = (mineralization_pattern or "").lower()
    stage = _classify_stage(project_stage)
    family = _classify_deposit_family(deposit_type, mineralization_pattern)

    is_vein = family == "vein"
    is_epithermal = "epithermal" in dt
    is_carlin = "carlin" in dt
    is_orogenic = "orogenic" in dt
    is_hs = "high-sulfidation" in dt or "high sulfidation" in dt or " hs " in f" {dt} "
    is_ls = "low-sulfidation" in dt or "low sulfidation" in dt or " ls " in f" {dt} "
    has_low_grade_halo = "bulk" in dt or "halo" in dt or "disseminated" in dt
    is_stockwork = "stockwork" in dt or "stockwork" in mp or "breccia" in dt

    # 1. Mature near-depleted epithermal vein
    if (is_epithermal and is_vein and stage == "mature"
            and mine_life_years is not None and mine_life_years < 2.0):
        return 0.87, 0.13

    # 2. Bulk Carlin/orogenic with halos
    if ((is_carlin or is_orogenic) and has_low_grade_halo
            and stage in ("mid", "mature")):
        return 0.80, 0.20

    # 3. LS epithermal stockwork/breccia at mid stage
    if is_ls and is_stockwork and stage == "mid":
        return 0.80, 0.20

    # 4. HS epithermal mature/depleted (district-wide upside, deferred from
    # explicit district-production check — applied to any mature HS epithermal)
    if is_hs and is_epithermal and stage == "mature":
        return 0.77, 0.23

    # 5. Early-stage default — pre-MRE confidence skews toward Inferred
    if stage == "early":
        return 0.40, 0.60

    # 6. Industry-prior fallback
    return 0.70, 0.30
