"""
Graph 2: analog_finder v2

Flow (parallel fan-out):
  load_project_and_rule
      ↓  (parallel)
  library_search   exa_search
      ↓  (merge)
  combine_filter_score
      ↓
  human_review → save_analogs → END

Changes from v1:
- library_search replaces db_analog_search: uses report_analogs (curated approved analogs)
- library_search + exa_search run in parallel
- exa_search uses rule-driven targeted query (deposit type, grade range, geo criteria)
- scoring is fully deterministic — no LLM, no score=50 fallback
- similarity_score is None when < 2 factors can be scored (shown as N/A in frontend)
- self-analog exclusion by name (not just by id)
- deposit-type exclusion rules parsed from analog_criteria ("Exclude X analogs")
"""
from __future__ import annotations
import logging
import re
from typing import Dict, List, Optional, TypedDict, Tuple

from langgraph.graph import StateGraph, END

from nodes import exa_search, field_extractor, rules_engine, supabase_ops, geo_taxonomy, lessons

logger = logging.getLogger(__name__)


# ── Bootstrap: keep compiled_rules in sync with code on every backend boot ─
# Runs once at module import. If the rule hash in DB differs from code, every
# rule is upserted. UI edits to compiled_rules get overwritten — code is the
# single source of truth. See nodes/bootstrap.py for the full mechanism.
try:
    from nodes.bootstrap import bootstrap_rules as _bootstrap_rules
    _bootstrap_rules()
except Exception as _boot_err:
    logger.warning(f"[startup] bootstrap_rules failed (non-fatal): {_boot_err}")


# ── State ──────────────────────────────────────────────────────────────────────

class AnalogState(TypedDict, total=False):
    project_id: str
    project: Optional[Dict]
    analog_rule: Optional[Dict]         # matched analog_selection rule from compiled_rules
    target_profile: Optional[Dict]      # geological identity of the target project
    library_analogs: List[Dict]         # from report_analogs (previously approved)
    exa_analogs: List[Dict]             # from Exa web search
    scored_analogs: List[Dict]
    low_confidence: bool                # True when <2 candidates passed L1-L5
    profile_warning: Optional[str]      # Human-readable message when profile too weak
    audit_events: List[Dict]            # Per-candidate decision audit trail
    human_approved: bool
    approved_analogs: List[Dict]
    saved: bool
    error: Optional[str]


# ── Scoring helpers ────────────────────────────────────────────────────────────

# Commodity aliases: maps a canonical material name to all acceptable commodity strings
# an analog can carry and still be considered the same commodity.
_COMMODITY_ALIASES: dict[str, set[str]] = {
    "gold":    {"gold", "au", "gold-silver", "gold_silver", "au-ag"},
    "silver":  {"silver", "ag", "gold-silver", "gold_silver", "au-ag"},
    "copper":  {"copper", "cu"},
    "nickel":  {"nickel", "ni"},
    "uranium": {"uranium", "u", "u3o8"},
    "iron":    {"iron", "iron ore", "fe"},
    "pgm":     {"pgm", "platinum", "palladium", "pt", "pd", "pge"},
}


def _materials_compatible(target: str, candidate: str) -> bool:
    """True if candidate commodity string is compatible with target material."""
    t = target.strip().lower()
    c = candidate.strip().lower()
    if not c:
        return True  # unknown commodity — let through, other filters may catch it
    if t == c:
        return True
    return c in _COMMODITY_ALIASES.get(t, {t})


# Grade-unit families: grades can only be compared within the same family.
_GT_FAMILY = {"g/t", "g/t au", "g/t ag", "g/t pt", "g/t pd", "g/tpt", "ppm", "gpt", "oz/t"}
_PCT_FAMILY = {"%", "percent", "pct", "% cu", "% ni", "% zn", "% pb", "% fe", "% u3o8", "% co"}


def _grade_units_compatible(u1: str, u2: str) -> bool:
    """False when units are from different families (e.g. g/t vs %) — comparing would give garbage ratios."""
    if not u1 or not u2:
        return True
    n1 = u1.strip().lower()
    n2 = u2.strip().lower()
    in_gt1 = any(n1.startswith(k) or k in n1 for k in ("g/t", "ppm", "oz/t", "gpt"))
    in_gt2 = any(n2.startswith(k) or k in n2 for k in ("g/t", "ppm", "oz/t", "gpt"))
    in_pct1 = n1.startswith("%") or "percent" in n1
    in_pct2 = n2.startswith("%") or "percent" in n2
    if (in_gt1 and in_pct2) or (in_pct1 and in_gt2):
        return False
    return True


_CONTINENT = {
    "australia": "oceania", "canada": "north_america", "usa": "north_america",
    "united states": "north_america", "mexico": "north_america",
    "brazil": "south_america", "chile": "south_america", "peru": "south_america",
    "argentina": "south_america", "colombia": "south_america", "ecuador": "south_america",
    "south africa": "africa", "zambia": "africa", "zimbabwe": "africa",
    "ghana": "africa", "mali": "africa", "burkina faso": "africa", "senegal": "africa",
    "congo": "africa", "drc": "africa", "tanzania": "africa", "kenya": "africa",
    "namibia": "africa", "botswana": "africa", "niger": "africa", "guinea": "africa",
    "russia": "europe_asia", "kazakhstan": "europe_asia", "uzbekistan": "europe_asia",
    "indonesia": "asia_pacific", "philippines": "asia_pacific",
    "new caledonia": "asia_pacific", "papua new guinea": "asia_pacific",
    "china": "asia_pacific", "mongolia": "asia_pacific",
    "sweden": "europe", "finland": "europe", "norway": "europe",
    "ireland": "europe", "portugal": "europe", "spain": "europe",
}


def _continent(country: str) -> Optional[str]:
    return _CONTINENT.get(country.lower().strip())


def _ratio_score(a: float, b: float, bands: list[tuple[float, float]]) -> float:
    """Score a ratio against bands: [(max_ratio, points), ...] sorted ascending."""
    ratio = max(a, b) / min(a, b) if min(a, b) > 0 else float("inf")
    for max_ratio, pts in bands:
        if ratio <= max_ratio:
            return pts
    return 0.0


def _deposit_type_family(dep: str) -> Optional[str]:
    """
    Return the geological deposit-type family for a deposit type string.
    Returns None when unrecognized — the gate only fires when BOTH sides return a known family.

    Families are intentionally coarse: porphyry-Cu and porphyry-Au share a family;
    porphyry and epithermal do not, even though they share a genetic link.
    """
    if not dep:
        return None
    d = dep.strip().lower().replace("-", " ").replace("_", " ")
    for keyword, family in (
        ("porphyry",               "porphyry"),
        ("epithermal",             "epithermal"),
        ("low sulphidation",       "epithermal"),
        ("high sulphidation",      "epithermal"),
        ("intermediate sulphidation", "epithermal"),
        ("orogenic",               "orogenic"),
        ("mesothermal",            "orogenic"),
        ("lode gold",              "orogenic"),
        ("vms",                    "vms"),
        ("vhms",                   "vms"),
        ("volcanic hosted",        "vms"),
        ("volcanogenic",           "vms"),
        ("carlin",                 "carlin"),
        ("iocg",                   "iocg"),
        ("iron oxide copper",      "iocg"),
        ("skarn",                  "skarn"),
        ("sediment hosted",        "sediment_hosted"),
        ("sediment",               "sediment_hosted"),
        ("sedex",                  "sediment_hosted"),
        ("manto",                  "sediment_hosted"),
        ("crd",                    "sediment_hosted"),
        ("mvt",                    "sediment_hosted"),
        ("carbonate replacement",  "sediment_hosted"),
        ("bif",                    "bif"),
        ("magnetite",              "bif"),
        ("banded iron",            "bif"),
        ("laterite",               "laterite"),
        ("saprolite",              "laterite"),
        ("magmatic sulphide",      "magmatic_sulphide"),
        ("magmatic",               "magmatic_sulphide"),
        ("komatiite",              "magmatic_sulphide"),
        ("unconformity",           "unconformity"),
        ("roll front",             "rollfront"),
        ("rollfront",              "rollfront"),
        ("merensky",               "pgm_reef"),
        ("platreef",               "pgm_reef"),
        ("ug2",                    "pgm_reef"),
    ):
        if keyword in d:
            return family
    return None


# Material + country heuristics to infer deposit family when deposit_type is unknown.
# Nickel laterites only form in tropical weathering belts; sulphides are found globally.
# If we can rule out a family from geography, block analogs of that family.
_NICKEL_LATERITE_COUNTRIES = frozenset({
    "indonesia", "philippines", "new caledonia", "cuba", "brazil", "colombia",
    "guatemala", "dominican republic", "madagascar", "russia",
})
_NICKEL_SULPHIDE_COUNTRIES = frozenset({
    "canada", "australia", "finland", "norway", "sweden", "botswana",
    "zimbabwe", "south africa", "greenland", "scotland",
})


def _infer_excluded_families(material: str, country: str) -> frozenset:
    """
    Return deposit-type families that are geologically impossible for this
    material + country combination. Used when deposit_type is unknown on the target.

    Conservative: only excludes when we're highly confident (e.g. nickel laterite
    in Canada is essentially impossible). Returns empty set when uncertain.
    """
    m = material.strip().lower()
    c = country.strip().lower() if country else ""
    excluded: set[str] = set()
    if m == "nickel":
        if c in _NICKEL_SULPHIDE_COUNTRIES:
            excluded.add("laterite")       # laterites don't form in temperate regions
        elif c in _NICKEL_LATERITE_COUNTRIES:
            excluded.add("magmatic_sulphide")  # primary sulphide deposits rare in these belts
    return frozenset(excluded)


_GEO_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "with", "type", "style", "hosted",
    "deposit", "mineralisation", "mineralization", "bearing", "rich", "related",
    "associated", "in", "at", "from", "by", "to",
})


def _geo_text_score(a: str, b: str, max_pts: float) -> float:
    """
    Score geological text similarity by meaningful word overlap (Jaccard + coverage).

    Three tiers:
      ≥70% overlap → full score      (same geological description)
      ≥30% overlap → 65% of max     (same family, different qualifier)
      any 1+ word overlap → 25% of max  (related but distinct)
    Returns 0 when no overlap.
    """
    a_words = {w for w in re.split(r"[\s\-_/,]+", a.lower())
               if len(w) > 2 and w not in _GEO_STOP_WORDS}
    b_words = {w for w in re.split(r"[\s\-_/,]+", b.lower())
               if len(w) > 2 and w not in _GEO_STOP_WORDS}
    if not a_words or not b_words:
        return 0.0
    overlap = len(a_words & b_words)
    if overlap == 0:
        return 0.0
    union = len(a_words | b_words)
    jaccard = overlap / union
    coverage = overlap / min(len(a_words), len(b_words))
    ratio = max(jaccard, coverage)
    if ratio >= 0.70:
        return max_pts
    elif ratio >= 0.30:
        return round(max_pts * 0.65, 1)
    else:
        return round(max_pts * 0.25, 1)


def _build_profile(row: dict) -> dict:
    """
    Build a geological identity profile for a project or analog. Reads the new
    schema columns when present; falls back to geo_taxonomy heuristics over the
    existing freeform text. Used identically for the target project and each
    candidate so cascading match compares apples to apples.
    """
    # Sanitize deposit_type once so every detect_*() heuristic sees clean text
    # even if a legacy row has "{Epithermal}" or similar set-literal noise.
    clean_deposit_type = rules_engine.sanitize_deposit_type(row.get("deposit_type"))
    row = {**row, "deposit_type": clean_deposit_type}
    return {
        "material":             (row.get("material") or "").strip().lower(),
        "deposit_type_family":  _deposit_type_family(row.get("deposit_type") or ""),
        "deposit_subtype":      row.get("deposit_subtype") or geo_taxonomy.detect_subtype(
            row.get("deposit_type"), row.get("mineralization_style"),
            row.get("alteration_signature"), row.get("district") or row.get("location_name"),
        ),
        "mineralization_mode":  row.get("mineralization_mode") or geo_taxonomy.detect_mode(
            row.get("processing_method"), row.get("mineralization_style"),
            row.get("district") or row.get("location_name"), row.get("deposit_type"),
        ),
        "tectonic_belt":        row.get("tectonic_belt") or geo_taxonomy.detect_belt(
            row.get("country"), row.get("region"), row.get("district"),
        ),
        "metal_suite":          row.get("metal_suite") or geo_taxonomy.detect_metal_suite(
            row.get("material"), None, row.get("district"), row.get("deposit_type"),
        ),
        "alteration_signature": row.get("alteration_signature") or geo_taxonomy.detect_alteration_signature(
            None, row.get("district") or row.get("location_name"), row.get("deposit_type"),
        ),
        "recovery_method":      row.get("recovery_method") or geo_taxonomy.detect_recovery_method(
            row.get("processing_method"), row.get("district") or row.get("location_name"),
            row.get("deposit_type"),
        ),
        # Mineralization pattern (orebody geometry) — L4.5 filter
        "mineralization_pattern": row.get("mineralization_pattern") or geo_taxonomy.detect_pattern(
            row.get("mineralization_style"), row.get("mining_method"),
            row.get("processing_method"), row.get("deposit_type"),
        ),
        # Host rock class — L4.7 filter
        "host_rock_class":      row.get("host_rock_class") or geo_taxonomy.detect_host_class(
            row.get("host_rock"), row.get("deposit_type"),
            row.get("mineralization_style"),
        ),
        # Project stage class — L4.6 filter
        "project_stage_class":  row.get("project_stage_class") or geo_taxonomy.detect_stage_class(
            row.get("project_stage"), None, row.get("location_name"),
        ),
        # Mining method class — L4.8 filter
        "mining_method_class":  row.get("mining_method_class") or geo_taxonomy.detect_mining_method_class(
            row.get("mining_method"), row.get("processing_method"),
            row.get("location_name"),
        ),
        # Resource category class — L4.9 filter
        "resource_category_class": (
            row.get("resource_category_class")
            or geo_taxonomy.detect_resource_category_class(row.get("resource_category"))
        ),
        # Compliance + vintage — L4.95 filter
        "resource_compliance_standard": (
            row.get("resource_compliance_standard")
            or geo_taxonomy.detect_resource_compliance(
                row.get("resource_category"), row.get("location_name"),
                row.get("source_url"),
            )
        ),
        "resource_vintage_year": row.get("resource_vintage_year"),
        "grade_value":          row.get("grade_value"),
        "grade_unit":           row.get("grade_unit"),
        "tonnage_mt":           row.get("tonnage_mt"),
        "country":              (row.get("country") or "").strip().lower(),
        "district":             (row.get("district") or "").strip(),
        "host_rock":            (row.get("host_rock") or "").strip(),
        "mineralization_style": (row.get("mineralization_style") or "").strip(),
        "source_url":           row.get("source_url"),
        "company_name":         (row.get("company_name") or "").strip().lower(),
        "project_id":           row.get("project_id") or row.get("id"),
    }


def _cascading_match(
    target: dict,
    candidate: dict,
    analog_rule: Optional[dict] = None,
) -> Tuple[bool, int, int, int, List[str], Optional[str]]:
    """
    Apply cascading geological-similarity match. Returns:
      (passes_hard_filter, ranking_pts, dimensions_matched, dimensions_evaluated,
       reasons, dropped_at)

    Levels L1-L5 are hard filters; L6-L11 contribute ranking points and matched-
    dimension counts. `dropped_at` is the level slug ("L3","L4","L5") on rejection,
    None on pass. Lesson IDs from the rule's `applies_lessons` are echoed in
    reason strings for LangSmith observability.
    """
    reasons: List[str] = []
    matched = 0
    evaluated = 0
    rank_pts = 0

    rule_lessons = ",".join((analog_rule or {}).get("applies_lessons") or []) or "L36"

    # ── L1: Same material ──────────────────────────────────────────────────
    if target["material"] and candidate["material"]:
        if not _materials_compatible(target["material"], candidate["material"]):
            return False, 0, 0, 0, [f"L1 commodity mismatch ({candidate['material']} ≠ {target['material']}): {rule_lessons}"], "L1"
        matched += 1
        evaluated += 1
        reasons.append(f"L1 material match ({candidate['material']}): {rule_lessons}")

    # ── L2: Same deposit-type family ───────────────────────────────────────
    t_fam = target["deposit_type_family"]
    c_fam = candidate["deposit_type_family"]
    if t_fam and c_fam:
        if t_fam != c_fam:
            return False, 0, 0, 0, [f"L2 deposit family mismatch ({c_fam} ≠ {t_fam}): {rule_lessons}"], "L2"
        matched += 1
        evaluated += 1
        reasons.append(f"L2 deposit family match ({c_fam}): {rule_lessons}")

    # ── L3: Same deposit sub-type (HARD — Lessons L86/L101/L124) ───────────
    # Sub-type is the single most important geological similarity dimension.
    # alkalic_porphyry ≠ laramide_porphyry ≠ iocg_oxide.
    # Sibling sub-types listed together in the rule's required_subtypes are
    # treated as compatible (e.g. greenstone_orogenic ≈ turbidite_orogenic
    # both appear in analog_sel_gold_orogenic.required_subtypes).
    t_sub = target["deposit_subtype"]
    c_sub = candidate["deposit_subtype"]
    rule_required = set((analog_rule or {}).get("required_subtypes") or [])
    if t_sub and c_sub:
        siblings = t_sub in rule_required and c_sub in rule_required
        if t_sub != c_sub and not siblings:
            return False, 0, 0, 0, [f"L3 sub-type mismatch ({c_sub} ≠ {t_sub}): {rule_lessons}"], "L3"
        matched += 1
        evaluated += 1
        # Exact match gets +25; sibling match gets +18
        rank_pts += 25 if t_sub == c_sub else 18
        suffix = "" if t_sub == c_sub else f" sibling-of {t_sub}"
        reasons.append(f"L3 sub-type match ({c_sub}{suffix}): {rule_lessons}")
    elif t_sub and not c_sub:
        # Target has subtype, candidate's is unknown. Don't drop — but record uncertainty.
        evaluated += 1
        reasons.append(f"L3 sub-type unknown on candidate (target={t_sub}): pass-through")

    # ── L4: Mineralization mode (HARD — Lessons L86/L101) ──────────────────
    # primary_sulfide ≠ supergene_oxide (different mineralogy + different metallurgy)
    if not geo_taxonomy.mode_compatible(target["mineralization_mode"], candidate["mineralization_mode"]):
        return False, 0, 0, 0, [
            f"L4 mode mismatch ({candidate['mineralization_mode']} ≠ {target['mineralization_mode']}): {rule_lessons}"
        ], "L4"
    if target["mineralization_mode"] and candidate["mineralization_mode"]:
        matched += 1
        evaluated += 1
        rank_pts += 15
        reasons.append(f"L4 mode match ({candidate['mineralization_mode']}): {rule_lessons}")

    # ── L4.4: Project stage compatibility (HARD when both sides have data) ─
    # Comparing exploration-stage to production-stage analogs inflates
    # over-confidence. STAGE_COMPATIBILITY allows adjacent stages.
    t_stage = target["project_stage_class"]
    c_stage = candidate["project_stage_class"]
    if t_stage and c_stage:
        if not geo_taxonomy.stage_compatible(t_stage, c_stage):
            return False, 0, 0, 0, [
                f"L4.4 stage mismatch ({c_stage} not compatible with {t_stage}): {rule_lessons}"
            ], "L4.4"
        if t_stage == c_stage:
            matched += 1
            evaluated += 1
            rank_pts += 10
            reasons.append(f"L4.4 stage match ({c_stage}): {rule_lessons}")

    # ── L4.5: Mineralization pattern (HARD — Gold Lesson LG19) ────────────
    # Vein-hosted ≠ disseminated_bulk ≠ stockwork ≠ replacement even within
    # the same family + subtype. This is the wedge between True North (vein)
    # and Springpole (bulk disseminated), and between Black Pine (bulk
    # disseminated Carlin) and Trixie (replacement Carlin).
    t_pattern = target["mineralization_pattern"]
    c_pattern = candidate["mineralization_pattern"]
    if t_pattern and c_pattern and t_pattern != c_pattern:
        return False, 0, 0, 0, [
            f"L4.5 pattern mismatch ({c_pattern} ≠ {t_pattern}): {rule_lessons}"
        ], "L4.5"
    if t_pattern and c_pattern:
        matched += 1
        evaluated += 1
        rank_pts += 20
        reasons.append(f"L4.5 pattern match ({c_pattern}): {rule_lessons}")

    # ── L4.7: Host rock class (RANK +15, or HARD if rule pins host classes) ─
    # Gabbro shear veins ≠ gneiss breccia ≠ syenite-hosted disseminated even
    # when all are orogenic gold. Rule's required_host_classes drives the
    # hard filter (checked in combine_filter_score_node); here we use it
    # only as a ranking signal.
    t_host = target["host_rock_class"]
    c_host = candidate["host_rock_class"]
    if t_host and c_host:
        evaluated += 1
        if t_host == c_host:
            matched += 1
            rank_pts += 15
            reasons.append(f"L4.7 host class match ({c_host}): {rule_lessons}")

    # ── L4.8: Mining method compatibility (HARD — Gold Lesson LG154) ───────
    # Open-pit bulk ≠ underground vein ≠ ISCR. Different cut-off, dilution,
    # recovery economics.
    t_mining = target["mining_method_class"]
    c_mining = candidate["mining_method_class"]
    if t_mining and c_mining:
        if not geo_taxonomy.mining_method_compatible(t_mining, c_mining):
            return False, 0, 0, 0, [
                f"L4.8 mining-method mismatch ({c_mining} ≠ {t_mining}): {rule_lessons}"
            ], "L4.8"
        if t_mining == c_mining:
            matched += 1
            evaluated += 1
            rank_pts += 10
            reasons.append(f"L4.8 mining-method match ({c_mining}): {rule_lessons}")

    # ── L4.9: Resource category meets rule minimum (HARD when rule pins it) ─
    min_cat = (analog_rule or {}).get("min_resource_category")
    if min_cat:
        c_cat = candidate["resource_category_class"]
        if c_cat and not geo_taxonomy.resource_category_at_least(c_cat, min_cat):
            return False, 0, 0, 0, [
                f"L4.9 resource category {c_cat} below rule minimum {min_cat}: {rule_lessons}"
            ], "L4.9"

    # ── L4.95: Compliance + vintage (HARD when standard is non-compliant) ─
    # Historical / press-release / internal estimates are never modellable.
    # If the rule sets min_resource_year, vintage_year must meet it.
    c_compliance = candidate["resource_compliance_standard"]
    min_year = (analog_rule or {}).get("min_resource_year")
    c_year = candidate.get("resource_vintage_year")
    if c_compliance in ("historical", "press_release", "internal"):
        return False, 0, 0, 0, [
            f"L4.95 non-compliant resource ({c_compliance}): {rule_lessons}"
        ], "L4.95"
    if min_year and c_year and c_year < int(min_year):
        return False, 0, 0, 0, [
            f"L4.95 resource vintage {c_year} < required {min_year}: {rule_lessons}"
        ], "L4.95"

    # ── L5: Recovery method compatibility (HARD — Lessons L19/L73/L82) ─────
    # flotation ≠ heap-leach ≠ ISCR — different metallurgical regime
    if not geo_taxonomy.recovery_compatible(target["recovery_method"], candidate["recovery_method"]):
        return False, 0, 0, 0, [
            f"L5 recovery incompatible ({candidate['recovery_method']} vs {target['recovery_method']}): {rule_lessons}"
        ], "L5"
    if target["recovery_method"] and candidate["recovery_method"]:
        if target["recovery_method"] == candidate["recovery_method"]:
            matched += 1
            evaluated += 1
            rank_pts += 10
            reasons.append(f"L5 recovery match ({candidate['recovery_method']}): {rule_lessons}")

    # ── L5.5: Tonnage tolerance (HARD when rule specifies) ─────────────────
    # Gold Lesson LG136: >20–25% tonnage mismatch must be penalised heavily.
    # Each rule can declare a tonnage_match_max_ratio (e.g. 5.0 for super-
    # large Carlin, 4.0 for orogenic-vein) — beyond that ratio, drop. When
    # the rule doesn't specify a tolerance, scale is just a ranking signal
    # at L9 (untouched here).
    tol = (analog_rule or {}).get("tonnage_match_max_ratio")
    if tol is not None:
        t_t = float(target.get("tonnage_mt") or 0)
        c_t = float(candidate.get("tonnage_mt") or 0)
        if t_t > 0 and c_t > 0:
            ratio = max(t_t, c_t) / min(t_t, c_t)
            if ratio > float(tol):
                return False, 0, 0, 0, [
                    f"L5.5 scale mismatch ({ratio:.1f}× > {tol}×, "
                    f"{c_t:.0f} Mt vs target {t_t:.0f} Mt): {rule_lessons}"
                ], "L5.5"

    # ── L5.6: Grade tolerance (HARD when rule specifies) — Gold L136 ──────
    g_tol = (analog_rule or {}).get("grade_match_max_ratio")
    if g_tol is not None:
        t_g = float(target.get("grade_value") or 0)
        c_g = float(candidate.get("grade_value") or 0)
        if t_g > 0 and c_g > 0 and _grade_units_compatible(
            target.get("grade_unit") or "", candidate.get("grade_unit") or ""
        ):
            ratio = max(t_g, c_g) / min(t_g, c_g)
            if ratio > float(g_tol):
                return False, 0, 0, 0, [
                    f"L5.6 grade mismatch ({ratio:.1f}× > {g_tol}×, "
                    f"{c_g} vs target {t_g} {target.get('grade_unit')}): {rule_lessons}"
                ], "L5.6"

    # ── L6: Tectonic belt (RANK +30) ───────────────────────────────────────
    if target["tectonic_belt"] and candidate["tectonic_belt"]:
        evaluated += 1
        if target["tectonic_belt"] == candidate["tectonic_belt"]:
            matched += 1
            rank_pts += 30
            reasons.append(f"L6 belt match ({candidate['tectonic_belt']}): +30")
        else:
            reasons.append(f"L6 belt different ({candidate['tectonic_belt']} ≠ {target['tectonic_belt']}): +0")

    # ── L7: Metal suite (RANK +20) ─────────────────────────────────────────
    if target["metal_suite"] and candidate["metal_suite"]:
        evaluated += 1
        if target["metal_suite"] == candidate["metal_suite"]:
            matched += 1
            rank_pts += 20
            reasons.append(f"L7 metal suite match ({candidate['metal_suite']}): +20")

    # ── L8: Grade band overlap (RANK +15) ──────────────────────────────────
    p_g = float(target.get("grade_value") or 0)
    c_g = float(candidate.get("grade_value") or 0)
    if p_g > 0 and c_g > 0 and _grade_units_compatible(target.get("grade_unit") or "", candidate.get("grade_unit") or ""):
        evaluated += 1
        pts = _ratio_score(p_g, c_g, [(1.5, 15), (2.0, 10), (3.0, 5), (5.0, 1)])
        if pts > 0:
            matched += 1
            rank_pts += int(pts)
            ratio = round(max(p_g, c_g) / min(p_g, c_g), 1)
            reasons.append(f"L8 grade {ratio}× match: +{int(pts)}")

    # ── L9: Tonnage same order of magnitude (RANK +10) ─────────────────────
    p_t = float(target.get("tonnage_mt") or 0)
    c_t = float(candidate.get("tonnage_mt") or 0)
    if p_t > 0 and c_t > 0:
        evaluated += 1
        ratio = max(p_t, c_t) / min(p_t, c_t)
        if ratio <= 10:
            matched += 1
            pts = 10 if ratio <= 2 else (7 if ratio <= 5 else 4)
            rank_pts += pts
            reasons.append(f"L9 tonnage {ratio:.1f}× scale: +{pts}")

    # ── L10: Same country (RANK +10) ───────────────────────────────────────
    t_c = target.get("country", "")
    c_c = candidate.get("country", "")
    if t_c and c_c:
        evaluated += 1
        if t_c == c_c:
            matched += 1
            rank_pts += 10
            reasons.append(f"L10 country match ({c_c}): +10")
        else:
            t_cont = _continent(t_c)
            c_cont = _continent(c_c)
            if t_cont and t_cont == c_cont:
                rank_pts += 4
                reasons.append(f"L10 same continent ({t_cont}): +4")

    # ── L11: Free-text overlap on host rock / mineralization style (RANK +0-15) ─
    if target.get("host_rock") and candidate.get("host_rock"):
        evaluated += 1
        pts = _geo_text_score(target["host_rock"], candidate["host_rock"], 8.0)
        if pts > 0:
            matched += 1
            rank_pts += int(pts)
            reasons.append(f"L11 host rock overlap: +{int(pts)}")
    if target.get("mineralization_style") and candidate.get("mineralization_style"):
        evaluated += 1
        pts = _geo_text_score(target["mineralization_style"], candidate["mineralization_style"], 7.0)
        if pts > 0:
            matched += 1
            rank_pts += int(pts)
            reasons.append(f"L11 style overlap: +{int(pts)}")

    return True, rank_pts, matched, evaluated, reasons, None


def _norm_name(name: str) -> str:
    """Normalize a project name: lowercase, remove punctuation and stop words."""
    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    stops = {"project", "mine", "mining", "deposit", "property", "corp", "inc",
             "ltd", "limited", "metals", "resources", "mineral", "minerals", "the", "a"}
    words = [w for w in cleaned.split() if w not in stops and len(w) > 1]
    return " ".join(sorted(words))


def _is_self_analog(
    project_name: str, candidate_name: str,
    project_id: Optional[str] = None, candidate_project_id: Optional[str] = None,
    project_company: Optional[str] = None, candidate_company: Optional[str] = None,
) -> bool:
    """True if candidate is likely the same project as the target.

    Three-level check:
      1. project_id equality (exact, when the candidate originated from the
         `projects` table — the most authoritative signal)
      2. company match + name fuzzy match (e.g., 'Doubleview Hat' vs 'Hat
         Copper' both owned by 'Doubleview Gold')
      3. Pure name fuzzy match (>=70% word overlap on normalised names)
    """
    # Level 1: project_id
    if project_id and candidate_project_id and str(project_id) == str(candidate_project_id):
        return True
    p = _norm_name(project_name or "")
    c = _norm_name(candidate_name or "")
    if not p or not c:
        return False
    if p == c:
        return True
    p_words = set(p.split())
    c_words = set(c.split())
    if not p_words or not c_words:
        return False
    overlap = len(p_words & c_words)
    name_overlap = overlap / min(len(p_words), len(c_words))
    # Level 2: company match + lower name overlap threshold (≥40%)
    if project_company and candidate_company:
        pc = project_company.strip().lower()
        cc = candidate_company.strip().lower()
        if pc and cc and (pc == cc or pc in cc or cc in pc):
            if name_overlap >= 0.40:
                return True
    # Level 3: pure name fuzzy
    return name_overlap >= 0.70


def _parse_exclusions(analog_criteria: list) -> list[str]:
    """Extract deposit type terms to exclude from analog_criteria 'Exclude X analogs' lines."""
    exclusions = []
    for c in (analog_criteria or []):
        if not c.lower().startswith("exclude"):
            continue
        match = re.search(r"exclude\s+(.*?)\s+analog", c, re.IGNORECASE)
        if match:
            terms = match.group(1)
            for term in re.split(r"\s+or\s+|\s+and\s+|,\s*", terms):
                t = term.strip().lower()
                if t and t not in ("or", "and", "the"):
                    exclusions.append(t)
    return exclusions


# ── Nodes ──────────────────────────────────────────────────────────────────────

def load_project_and_rule_node(state: AnalogState) -> AnalogState:
    """Load project + fetch matching analog_selection rule."""
    project_id = state["project_id"]
    project = supabase_ops.get_project(project_id)
    if not project:
        return {"error": f"Project {project_id} not found"}

    material = project.get("material") or ""
    # Some legacy projects have deposit_type stored as a Python set-literal
    # string like "{Epithermal}" — sanitize before any downstream use.
    deposit_type = rules_engine.sanitize_deposit_type(project.get("deposit_type"))
    # Subtype takes precedence — alkalic_porphyry routes to the dedicated alkalic
    # rule even when deposit_type is just "porphyry copper-gold".
    deposit_subtype = project.get("deposit_subtype") or geo_taxonomy.detect_subtype(
        deposit_type, project.get("mineralization_style"),
        project.get("alteration_signature"),
        project.get("district") or project.get("location_name"),
    )
    # Mineralization pattern (vein vs disseminated_bulk etc.) — disambiguates
    # sub-rules that share a subtype (orogenic-vein vs orogenic-bulk both have
    # required_subtypes=["orogenic_general"] but different required_patterns).
    pattern = project.get("mineralization_pattern") or geo_taxonomy.detect_pattern(
        project.get("mineralization_style"), project.get("mining_method"),
        project.get("processing_method"), deposit_type,
    )
    analog_rule = rules_engine.get_analog_rule(material, deposit_type, deposit_subtype, pattern)

    logger.info(
        f"[load] {project.get('name')} | material={material} deposit={deposit_type} "
        f"subtype={deposit_subtype} pattern={pattern} "
        f"rule={'✓ ' + analog_rule.get('rule_id', '') if analog_rule else '✗ none'}"
    )
    return {"project": project, "analog_rule": analog_rule, "error": None}


_PROFILE_DIMENSIONS = (
    "deposit_subtype", "mineralization_mode", "tectonic_belt",
    "metal_suite", "alteration_signature", "recovery_method",
    "mineralization_pattern", "host_rock_class",
    "project_stage_class", "mining_method_class",
)
PROFILE_STRENGTH_DEFAULT = 4  # default; per-rule override via rule.min_profile_strength


def _profile_strength(profile: Dict) -> int:
    """Count of non-null geological dimensions in a profile."""
    return sum(1 for k in _PROFILE_DIMENSIONS if profile.get(k))


def build_target_profile_node(state: AnalogState) -> AnalogState:
    """Derive the target project's geological identity profile (used for cascading match)."""
    if state.get("error"):
        return {"target_profile": None}
    project = state["project"] or {}
    profile = _build_profile(project)
    strength = _profile_strength(profile)
    logger.info(
        f"[profile] target subtype={profile['deposit_subtype']!r} "
        f"mode={profile['mineralization_mode']!r} belt={profile['tectonic_belt']!r} "
        f"metal={profile['metal_suite']!r} recovery={profile['recovery_method']!r} "
        f"strength={strength}/6"
    )
    return {"target_profile": profile}


def library_search_node(state: AnalogState) -> AnalogState:
    """Search report_analogs for previously approved analogs of this commodity."""
    if state.get("error"):
        return {"library_analogs": []}

    project = state["project"]
    analog_rule = state.get("analog_rule")
    material = project.get("material") or ""
    deposit_type = rules_engine.sanitize_deposit_type(project.get("deposit_type"))
    # Prefer the structured subtype slug for matching — it's exact and
    # survives freeform-text variations (Carlin-style vs Carlin-type,
    # word reordering, optional suffixes). The freeform deposit_type is
    # the fallback when subtype is null.
    deposit_subtype = project.get("deposit_subtype") or (
        state.get("target_profile") or {}
    ).get("deposit_subtype")
    # When the rule lists multiple acceptable subtypes (e.g. orogenic-vein
    # accepts greenstone + turbidite + bif-hosted), pass them all so the
    # library returns valid siblings. Without this, Fosterville
    # (turbidite_orogenic) gets dropped for a True North-style target
    # whose subtype is greenstone_orogenic — even though both are explicit
    # required_subtypes on the orogenic-vein rule.
    accepted_subtypes = list((analog_rule or {}).get("required_subtypes") or [])
    if deposit_subtype and deposit_subtype not in accepted_subtypes:
        accepted_subtypes = accepted_subtypes + [deposit_subtype]

    analogs = supabase_ops.get_approved_analogs(
        material, deposit_type, limit=20,
        deposit_subtype=deposit_subtype,
        deposit_subtypes=accepted_subtypes or None,
    )
    logger.info(
        f"[library] Found {len(analogs)} previously approved analogs "
        f"(filter: subtypes={accepted_subtypes!r} dep={deposit_type!r})"
    )
    return {"library_analogs": analogs}


def exa_search_node(state: AnalogState) -> AnalogState:
    """Find comparable projects via Exa using rule-driven targeted query."""
    if state.get("error"):
        return {"exa_analogs": []}

    project = state["project"]
    analog_rule = state.get("analog_rule")
    material = project.get("material", "")
    deposit_type = rules_engine.sanitize_deposit_type(project.get("deposit_type")) or ""
    project_name = project.get("name", "")

    text, sources = exa_search.search_analog_projects(
        material=material,
        deposit_type=deposit_type,
        project_name=project_name,
        analog_rule=analog_rule,
        grade_value=project.get("grade_value"),
        grade_unit=project.get("grade_unit"),
        tonnage_mt=project.get("tonnage_mt"),
        country=project.get("country"),
        host_rock=project.get("host_rock"),
        mineralization_style=project.get("mineralization_style"),
        target_profile=state.get("target_profile"),
    )

    exa_analogs = []
    if text:
        raw = field_extractor.extract_analog_projects(text, material, sources)
        for i, a in enumerate(raw):
            exa_analogs.append({
                "name": a.get("name", f"Unknown project {i}"),
                # Use extracted commodity — enables real material validation downstream.
                # Fall back to project material only when extraction returned nothing.
                "material": (a.get("commodity") or "").strip().lower() or material,
                "deposit_type": a.get("deposit_type"),
                "host_rock": a.get("host_rock"),
                "mineralization_style": a.get("mineralization_style"),
                "district": a.get("district"),
                "region": a.get("region"),
                "tonnage_mt": a.get("tonnage_mt"),
                "grade_value": a.get("grade_value"),
                "grade_unit": a.get("grade_unit"),
                "country": a.get("country"),
                "project_stage": a.get("project_stage"),
                "mining_method": a.get("mining_method"),
                "processing_method": a.get("processing_method"),
                # Geological profile (cascading match)
                "deposit_subtype":      a.get("deposit_subtype"),
                "mineralization_mode":  a.get("mineralization_mode"),
                "tectonic_belt":        a.get("tectonic_belt"),
                "metal_suite":          a.get("metal_suite"),
                "alteration_signature": a.get("alteration_signature"),
                "recovery_method":      a.get("recovery_method"),
                "source": "exa",
                "source_url": a.get("source_url") or (sources[i] if i < len(sources) else None),
            })

    logger.info(f"[exa] Extracted {len(exa_analogs)} candidates from Exa")
    return {"exa_analogs": exa_analogs}


def combine_filter_score_node(state: AnalogState) -> AnalogState:
    """
    Cascading-match analog selection. For each candidate:
      L1-L5 are hard filters (commodity, deposit family, sub-type, mineralization
        mode, recovery method). Any mismatch when BOTH sides have data → drop.
      L6-L11 contribute ranking points (belt, metal suite, grade, tonnage,
        country, free-text overlap).
    Returns top 4-6 survivors sorted by ranking points. When fewer than 2
    candidates survive L1-L5, sets low_confidence=True and returns the best
    available without padding with bad analogs.
    """
    if state.get("error"):
        return {"scored_analogs": [], "low_confidence": False}

    project = state["project"]
    analog_rule = state.get("analog_rule")
    target_profile = state.get("target_profile") or _build_profile(project)
    library = state.get("library_analogs") or []
    exa = state.get("exa_analogs") or []
    all_candidates = library + exa

    project_name = project.get("name") or ""

    # No-rule fallback — if get_analog_rule returned None we don't have a
    # commodity/subtype-specific rule loaded. Refuse to run the strict
    # cascade; surface a clear "no rule for this deposit class" message to
    # the user so they enrich the project or add a rule.
    if analog_rule is None:
        logger.warning(
            f"[cascade] NO RULE for project material={project.get('material','?')} "
            f"deposit_type={project.get('deposit_type','?')} — refusing to score, "
            f"flagging low_confidence."
        )
        return {
            "scored_analogs": [],
            "low_confidence": True,
            "profile_warning": (
                f"No analog_selection rule for "
                f"{project.get('material','?')} / "
                f"{project.get('deposit_type','?')}. The cascading filter would "
                f"degrade to commodity-only matching, which is unreliable. Add "
                f"the rule to scripts/seed_analog_rules.py, or set the project's "
                f"deposit_type / deposit_subtype to one we already cover."
            ),
            "audit_events": [],
        }

    # Profile-strength gate — per-rule minimum (default 4 of 10 dimensions).
    # When the target is below the rule's strictness threshold, we DON'T refuse
    # to score (that produces silent 0-analog results). Instead we drop into
    # "relaxed mode": skip the rule's required_* lists (so we don't filter on
    # dimensions the target itself doesn't have) but keep all excluded_* lists
    # and the cascade hard gates. The user still gets analogs, but flagged
    # low_confidence with a clear warning explaining the trade-off.
    required_subtypes_pregate = set(analog_rule.get("required_subtypes") or [])
    min_strength = int(analog_rule.get("min_profile_strength") or PROFILE_STRENGTH_DEFAULT)
    strength = _profile_strength(target_profile)
    relaxed_mode = False
    relaxed_warning = None
    if required_subtypes_pregate and strength < min_strength:
        missing = [d for d in _PROFILE_DIMENSIONS if not target_profile.get(d)]
        logger.warning(
            f"[cascade] PROFILE TOO WEAK for strict rule "
            f"{analog_rule.get('rule_id','?')}: only {strength}/{len(_PROFILE_DIMENSIONS)} "
            f"dimensions (rule requires {min_strength}); "
            f"missing {missing}. Falling back to RELAXED MODE."
        )
        relaxed_mode = True
        relaxed_warning = (
            f"Project has only {strength}/{len(_PROFILE_DIMENSIONS)} geological "
            f"dimensions populated (rule {analog_rule.get('rule_id','?')} prefers "
            f"≥{min_strength}). Running in relaxed mode: rule's required_* lists "
            f"are skipped; exclusions and cascade hard filters still apply. "
            f"Missing dimensions: {', '.join(missing)}. Enrich the project to "
            f"raise confidence."
        )
        # Drop the rule's required_* lists so we don't filter on absent fields.
        required_subtypes = set()
        required_patterns = set()
        required_host_classes = set()
        required_stages = set()
        required_mining_methods = set()
        required_metal_suites = set()
    rule_exclusions = _parse_exclusions((analog_rule or {}).get("analog_criteria") or [])
    # Structured exclusions from the rule (subtypes/modes/recovery the rule forbids)
    excluded_subtypes = set((analog_rule or {}).get("excluded_subtypes") or [])
    excluded_modes = set((analog_rule or {}).get("excluded_modes") or [])
    excluded_recovery = set((analog_rule or {}).get("excluded_recovery") or [])
    excluded_patterns = set((analog_rule or {}).get("excluded_patterns") or [])
    excluded_host_classes = set((analog_rule or {}).get("excluded_host_classes") or [])
    excluded_stages = set((analog_rule or {}).get("excluded_stages") or [])
    excluded_mining_methods = set((analog_rule or {}).get("excluded_mining_methods") or [])
    excluded_resource_categories = set((analog_rule or {}).get("excluded_resource_categories") or [])
    excluded_metal_suites = set((analog_rule or {}).get("excluded_metal_suites") or [])
    # Positive required lists — when the rule specifies required_subtypes
    # (or _patterns / _host_classes), any candidate with a CONFIDENTLY-DETECTED
    # value outside the list is dropped. Candidates with null detection are
    # NOT dropped here (would discard too many poorly-enriched library/exa
    # candidates); they fall through to L3 / L4.5 / L4.7 instead.
    required_subtypes = set((analog_rule or {}).get("required_subtypes") or [])
    required_patterns = set((analog_rule or {}).get("required_patterns") or [])
    required_host_classes = set((analog_rule or {}).get("required_host_classes") or [])
    required_stages = set((analog_rule or {}).get("required_stages") or [])
    required_mining_methods = set((analog_rule or {}).get("required_mining_methods") or [])
    required_metal_suites = set((analog_rule or {}).get("required_metal_suites") or [])

    logger.info(
        f"[cascade] {len(library)} library + {len(exa)} exa = {len(all_candidates)} "
        f"| rule={(analog_rule or {}).get('rule_id','none')} "
        f"target_subtype={target_profile['deposit_subtype']} "
        f"target_belt={target_profile['tectonic_belt']}"
    )

    # ── Step A: Dedup by normalized name ───────────────────────────────────
    seen: dict[str, bool] = {}
    deduped: list[dict] = []
    for c in all_candidates:
        norm = _norm_name(c.get("name") or "")
        if norm and norm not in seen:
            seen[norm] = True
            deduped.append(c)
    logger.info(f"[cascade] {len(deduped)} after dedup")

    # ── Audit event accumulator (Phase 7) ─────────────────────────────────
    # Every candidate (passed or dropped) gets a structured audit event so
    # LangSmith traces, the admin dashboard, and the frontend Audit Trail
    # tab can render exactly why each decision was made.
    audit_events: list[dict] = []
    rule_id_for_audit = (analog_rule or {}).get("rule_id")
    rule_lesson_ids_for_audit = list((analog_rule or {}).get("applies_lessons") or [])

    def _emit(decision: str, level: str, candidate: dict, profile: dict,
              reason: str, rank_pts: int | None = None,
              score: float | None = None) -> None:
        # Library candidates that get dropped emit REVOKED_LIBRARY instead of
        # plain DROP — flags rules that have tightened since the analog was
        # approved (Batch D: library rot prevention).
        effective_decision = decision
        if decision == "DROP" and (candidate.get("source") == "library"):
            effective_decision = "REVOKED_LIBRARY"
        audit_events.append({
            "candidate_name": candidate.get("name") or "Unknown",
            "candidate_source": candidate.get("source"),
            "decision": effective_decision,
            "level": level,
            "rule_id": rule_id_for_audit,
            "lessons": lessons.resolve_lesson_ids(rule_lesson_ids_for_audit),
            "detected_profile": {k: profile.get(k) for k in _PROFILE_DIMENSIONS},
            "reason": reason,
            "rank_pts": rank_pts,
            "similarity_score": score,
        })

    # ── Step B: Self-analog pre-filter + hallucination guard ───────────────
    project_id_str = state.get("project_id")
    project_company = (project.get("company_name") or "").strip()
    pre_filtered: list[dict] = []
    for c in deduped:
        # Self-analog (3-level check: id, company+name, name fuzzy)
        if project_name and _is_self_analog(
            project_name, c.get("name") or "",
            project_id_str, c.get("project_id") or c.get("id"),
            project_company, c.get("company_name") or c.get("company"),
        ):
            logger.info(f"[cascade] DROP self-analog: {c.get('name')}")
            _emit("DROP", "self_analog", c, {}, "matches target project (name/id/company)")
            continue
        # Hallucination guard — an Exa-sourced candidate with NO source_url AND
        # NO numeric tonnage AND NO grade is almost certainly Grok inventing
        # a plausible-sounding project name. Drop with SUSPECTED_HALLUCINATION.
        if c.get("source") == "exa":
            has_url = bool((c.get("source_url") or "").strip())
            has_numbers = bool(c.get("tonnage_mt")) or bool(c.get("grade_value"))
            if not has_url and not has_numbers:
                logger.info(f"[cascade] DROP suspected hallucination (no URL, no numbers): {c.get('name')}")
                _emit("DROP", "suspected_hallucination", c, {},
                      "no source_url and no tonnage/grade — likely Grok-invented")
                continue
        pre_filtered.append(c)

    # ── Step C: Cascading match per candidate ──────────────────────────────
    survivors: list[dict] = []
    dropped_counts: dict[str, int] = {}
    for c in pre_filtered:
        cand_profile = _build_profile(c)

        # Rule-driven structured exclusions (Lessons L86/L101 from the rule itself)
        if cand_profile["deposit_subtype"] and cand_profile["deposit_subtype"] in excluded_subtypes:
            reason = f"rule excluded sub-type ({cand_profile['deposit_subtype']})"
            logger.info(f"[cascade] DROP {reason}: {c.get('name')}")
            dropped_counts["rule_subtype"] = dropped_counts.get("rule_subtype", 0) + 1
            _emit("DROP", "rule_subtype", c, cand_profile, reason)
            continue
        if cand_profile["mineralization_mode"] and cand_profile["mineralization_mode"] in excluded_modes:
            reason = f"rule excluded mode ({cand_profile['mineralization_mode']})"
            logger.info(f"[cascade] DROP {reason}: {c.get('name')}")
            dropped_counts["rule_mode"] = dropped_counts.get("rule_mode", 0) + 1
            _emit("DROP", "rule_mode", c, cand_profile, reason)
            continue
        if cand_profile["recovery_method"] and cand_profile["recovery_method"] in excluded_recovery:
            reason = f"rule excluded recovery ({cand_profile['recovery_method']})"
            logger.info(f"[cascade] DROP {reason}: {c.get('name')}")
            dropped_counts["rule_recovery"] = dropped_counts.get("rule_recovery", 0) + 1
            _emit("DROP", "rule_recovery", c, cand_profile, reason)
            continue

        # Rule-driven stage filters
        if cand_profile["project_stage_class"] and cand_profile["project_stage_class"] in excluded_stages:
            reason = f"rule excluded stage ({cand_profile['project_stage_class']})"
            logger.info(f"[cascade] DROP {reason}: {c.get('name')}")
            dropped_counts["rule_stage"] = dropped_counts.get("rule_stage", 0) + 1
            _emit("DROP", "rule_stage", c, cand_profile, reason)
            continue
        if required_stages and cand_profile["project_stage_class"]:
            if cand_profile["project_stage_class"] not in required_stages:
                reason = (f"stage {cand_profile['project_stage_class']} "
                          f"not in required {sorted(required_stages)}")
                logger.info(f"[cascade] DROP required-stage mismatch: {c.get('name')}")
                dropped_counts["rule_required_stage"] = dropped_counts.get("rule_required_stage", 0) + 1
                _emit("DROP", "rule_required_stage", c, cand_profile, reason)
                continue

        # Rule-driven mining method filters
        if cand_profile["mining_method_class"] and cand_profile["mining_method_class"] in excluded_mining_methods:
            reason = f"rule excluded mining method ({cand_profile['mining_method_class']})"
            logger.info(f"[cascade] DROP {reason}: {c.get('name')}")
            dropped_counts["rule_mining_method"] = dropped_counts.get("rule_mining_method", 0) + 1
            _emit("DROP", "rule_mining_method", c, cand_profile, reason)
            continue
        if required_mining_methods and cand_profile["mining_method_class"]:
            if cand_profile["mining_method_class"] not in required_mining_methods:
                reason = (f"mining method {cand_profile['mining_method_class']} "
                          f"not in required {sorted(required_mining_methods)}")
                logger.info(f"[cascade] DROP required-mining-method mismatch: {c.get('name')}")
                dropped_counts["rule_required_mining_method"] = dropped_counts.get("rule_required_mining_method", 0) + 1
                _emit("DROP", "rule_required_mining_method", c, cand_profile, reason)
                continue

        # Rule-driven resource category exclusion (e.g., exclude historical)
        if cand_profile["resource_category_class"] and cand_profile["resource_category_class"] in excluded_resource_categories:
            reason = f"rule excluded resource category ({cand_profile['resource_category_class']})"
            logger.info(f"[cascade] DROP {reason}: {c.get('name')}")
            dropped_counts["rule_resource_category"] = dropped_counts.get("rule_resource_category", 0) + 1
            _emit("DROP", "rule_resource_category", c, cand_profile, reason)
            continue

        # Rule-driven metal suite filters
        if cand_profile["metal_suite"] and cand_profile["metal_suite"] in excluded_metal_suites:
            reason = f"rule excluded metal suite ({cand_profile['metal_suite']})"
            logger.info(f"[cascade] DROP {reason}: {c.get('name')}")
            dropped_counts["rule_metal_suite"] = dropped_counts.get("rule_metal_suite", 0) + 1
            _emit("DROP", "rule_metal_suite", c, cand_profile, reason)
            continue
        if required_metal_suites and cand_profile["metal_suite"]:
            if cand_profile["metal_suite"] not in required_metal_suites:
                reason = (f"metal suite {cand_profile['metal_suite']} "
                          f"not in required {sorted(required_metal_suites)}")
                logger.info(f"[cascade] DROP required-metal-suite mismatch: {c.get('name')}")
                dropped_counts["rule_required_metal_suite"] = dropped_counts.get("rule_required_metal_suite", 0) + 1
                _emit("DROP", "rule_required_metal_suite", c, cand_profile, reason)
                continue

        # Rule-driven mineralization_pattern filters
        if cand_profile["mineralization_pattern"] and cand_profile["mineralization_pattern"] in excluded_patterns:
            reason = f"rule excluded pattern ({cand_profile['mineralization_pattern']})"
            logger.info(f"[cascade] DROP {reason}: {c.get('name')}")
            dropped_counts["rule_pattern"] = dropped_counts.get("rule_pattern", 0) + 1
            _emit("DROP", "rule_pattern", c, cand_profile, reason)
            continue
        if required_patterns and cand_profile["mineralization_pattern"]:
            if cand_profile["mineralization_pattern"] not in required_patterns:
                reason = (f"pattern {cand_profile['mineralization_pattern']} "
                          f"not in required {sorted(required_patterns)}")
                logger.info(f"[cascade] DROP required-pattern mismatch: {c.get('name')}")
                dropped_counts["rule_required_pattern"] = dropped_counts.get("rule_required_pattern", 0) + 1
                _emit("DROP", "rule_required_pattern", c, cand_profile, reason)
                continue

        # Rule-driven host_rock_class filters
        if cand_profile["host_rock_class"] and cand_profile["host_rock_class"] in excluded_host_classes:
            reason = f"rule excluded host class ({cand_profile['host_rock_class']})"
            logger.info(f"[cascade] DROP {reason}: {c.get('name')}")
            dropped_counts["rule_host_class"] = dropped_counts.get("rule_host_class", 0) + 1
            _emit("DROP", "rule_host_class", c, cand_profile, reason)
            continue
        if required_host_classes and cand_profile["host_rock_class"]:
            if cand_profile["host_rock_class"] not in required_host_classes:
                reason = (f"host class {cand_profile['host_rock_class']} "
                          f"not in required {sorted(required_host_classes)}")
                logger.info(f"[cascade] DROP required-host-class mismatch: {c.get('name')}")
                dropped_counts["rule_required_host_class"] = dropped_counts.get("rule_required_host_class", 0) + 1
                _emit("DROP", "rule_required_host_class", c, cand_profile, reason)
                continue

        # Positive required_subtypes filter — when the rule pins a subtype list,
        # any candidate with a detected subtype OUTSIDE that list is dropped.
        if required_subtypes and cand_profile["deposit_subtype"]:
            if cand_profile["deposit_subtype"] not in required_subtypes:
                reason = (f"detected sub-type {cand_profile['deposit_subtype']} "
                          f"not in required {sorted(required_subtypes)}")
                logger.info(f"[cascade] DROP required-subtype mismatch: {c.get('name')}")
                dropped_counts["rule_required_subtype"] = dropped_counts.get("rule_required_subtype", 0) + 1
                _emit("DROP", "rule_required_subtype", c, cand_profile, reason)
                continue

        # Strict-mode drop for unenriched candidates
        if required_subtypes and not cand_profile["deposit_subtype"]:
            has_text = bool(
                (c.get("deposit_type") or "").strip()
                or (c.get("mineralization_style") or "").strip()
            )
            if not has_text:
                reason = "candidate has no sub-type, no deposit_type, no mineralization_style"
                logger.info(f"[cascade] DROP unenriched: {c.get('name')}")
                dropped_counts["unenriched"] = dropped_counts.get("unenriched", 0) + 1
                _emit("DROP", "unenriched", c, cand_profile, reason)
                continue

        # Legacy "Exclude X analogs" text patterns
        c_dep_lower = (c.get("deposit_type") or "").lower()
        if c_dep_lower and rule_exclusions:
            if any(excl in c_dep_lower for excl in rule_exclusions):
                reason = f"deposit_type matches rule text-exclusion ({c_dep_lower})"
                logger.info(f"[cascade] DROP rule text-exclusion: {c.get('name')}")
                dropped_counts["rule_text"] = dropped_counts.get("rule_text", 0) + 1
                _emit("DROP", "rule_text", c, cand_profile, reason)
                continue

        passes, rank_pts, matched, evaluated, reasons, dropped_at = _cascading_match(
            target_profile, cand_profile, analog_rule,
        )
        if not passes:
            reason = reasons[0] if reasons else "cascade rejection"
            logger.info(f"[cascade] DROP @{dropped_at}: {c.get('name')} — {reason}")
            dropped_counts[dropped_at or "unknown"] = dropped_counts.get(dropped_at or "unknown", 0) + 1
            _emit("DROP", dropped_at or "unknown", c, cand_profile, reason)
            continue

        # similarity_score = matched / evaluated (as percentage), or None when too few signals
        score = (
            round(matched / evaluated * 100, 1)
            if evaluated >= 2 else None
        )
        rule_lesson_ids = list((analog_rule or {}).get("applies_lessons") or [])
        survivors.append({
            **c,
            "similarity_score": score,
            "similarity_reasons": reasons,
            # Resolve lesson IDs to full dicts (id + title + text + source_doc)
            # so the frontend can render the "why was this picked?" tooltip.
            "lessons": lessons.resolve_lesson_ids(rule_lesson_ids),
            "_rank_pts": rank_pts,
            "_dimensions_matched": matched,
            "_dimensions_evaluated": evaluated,
            "approved": False,
        })
        _emit("PASS", "cascade", c, cand_profile,
              f"matched {matched}/{evaluated} dimensions",
              rank_pts=rank_pts, score=score)

    logger.info(
        f"[cascade] {len(survivors)} survivors | dropped: {dict(dropped_counts) or 'none'}"
    )

    # ── Step D: Rank by total points; HARD CAP at 4 ────────────────────────
    # Per product requirement: max 4 analogs. Better to have 4 strong matches
    # than dilute with weaker candidates.
    ranked = sorted(survivors, key=lambda x: -x["_rank_pts"])
    low_confidence = len(ranked) < 2
    if low_confidence:
        logger.warning(
            f"[cascade] Only {len(ranked)} candidate(s) passed L1-L5 — "
            f"flagging low_confidence; returning best available without padding"
        )
        top = ranked[:2]
    else:
        top = ranked[:4]

    # Near-miss observability — survivors that passed every hard filter but
    # were squeezed out by the top-4 cap get a NEAR_MISS audit event. Makes
    # the cap auditable: if a strong 5th candidate exists, it's visible in
    # the audit trail rather than silently lost.
    near_misses = ranked[4:7]  # top-3 just below the cap
    for nm in near_misses:
        audit_events.append({
            "candidate_name": nm.get("name") or "Unknown",
            "candidate_source": nm.get("source"),
            "decision": "NEAR_MISS",
            "level": "below_top_cap",
            "rule_id": rule_id_for_audit,
            "lessons": lessons.resolve_lesson_ids(rule_lesson_ids_for_audit),
            "detected_profile": {k: nm.get(k) for k in _PROFILE_DIMENSIONS},
            "reason": f"passed cascade but ranked outside top-4 cap (rank_pts={nm.get('_rank_pts')})",
            "rank_pts": nm.get("_rank_pts"),
            "similarity_score": nm.get("similarity_score"),
        })

    # Strip internal-only keys before returning
    for s in top:
        s.pop("_rank_pts", None)
        s.pop("_dimensions_matched", None)
        s.pop("_dimensions_evaluated", None)

    if top:
        best = top[0]
        logger.info(
            f"[cascade] Best analog: {best.get('name')} "
            f"score={best.get('similarity_score')} source={best.get('source')}"
        )
    else:
        logger.warning("[cascade] No analog candidates found")

    # If we entered relaxed mode, surface that to the caller as well.
    result = {
        "scored_analogs": top,
        "low_confidence": low_confidence or relaxed_mode,
        "audit_events": audit_events,
    }
    if relaxed_warning:
        result["profile_warning"] = relaxed_warning
        # Prepend a RELAXED_MODE marker so the audit trail explains why this
        # run is low-confidence before the candidate-level events.
        audit_events.insert(0, {
            "candidate_name": "(run-level)",
            "candidate_source": None,
            "decision": "RELAXED_MODE",
            "level": "profile_strength_below_min",
            "rule_id": rule_id_for_audit,
            "lessons": lessons.resolve_lesson_ids(rule_lesson_ids_for_audit),
            "detected_profile": {k: target_profile.get(k) for k in _PROFILE_DIMENSIONS},
            "reason": relaxed_warning,
            "rank_pts": None,
            "similarity_score": None,
        })
    return result


def save_analogs_node(state: AnalogState, config: Optional[Dict] = None) -> AnalogState:
    """Save scored analogs to Supabase. No human gate."""
    if state.get("error"):
        logger.info(f"[save] Upstream error — not saving: {state['error']}")
        return {"saved": False}

    analogs = state.get("scored_analogs", []) or []
    for a in analogs:
        a["approved"] = True

    cfg = (config or {}).get("configurable") or {}
    thread_id = cfg.get("thread_id")
    run_id = cfg.get("run_id")

    try:
        supabase_ops.save_analogs(state["project_id"], analogs, thread_id=thread_id, run_id=run_id)
        logger.info(f"[save] Saved {len(analogs)} analogs for project {state['project_id']}")
        # Persist audit trail (non-fatal if it fails — analogs are already saved)
        audit_events = state.get("audit_events") or []
        if audit_events:
            try:
                supabase_ops.save_analog_audit_events(state["project_id"], audit_events)
                logger.info(f"[save] Persisted {len(audit_events)} audit events")
            except Exception as ae:
                logger.warning(f"[save] Audit-event persistence failed: {ae}")
        return {"saved": True, "error": None}
    except Exception as e:
        logger.error(f"[save] Error: {e}")
        return {"saved": False, "error": str(e)}


# ── Graph ──────────────────────────────────────────────────────────────────────

def _should_continue(state: AnalogState) -> str:
    return END if state.get("error") else "parallel_search"


builder = StateGraph(AnalogState)

builder.add_node("load_project_and_rule", load_project_and_rule_node)
builder.add_node("build_target_profile", build_target_profile_node)
builder.add_node("library_search", library_search_node)
builder.add_node("exa_search", exa_search_node)
builder.add_node("combine_filter_score", combine_filter_score_node)
builder.add_node("save_analogs", save_analogs_node)

builder.set_entry_point("load_project_and_rule")

# load → build_target_profile → parallel fan-out (library + exa)
builder.add_edge("load_project_and_rule", "build_target_profile")
builder.add_edge("build_target_profile", "library_search")
builder.add_edge("build_target_profile", "exa_search")

# Fan-in: both feed into combine (LangGraph waits for both before running combine)
builder.add_edge("library_search", "combine_filter_score")
builder.add_edge("exa_search", "combine_filter_score")

builder.add_edge("combine_filter_score", "save_analogs")
builder.add_edge("save_analogs", END)

graph = builder.compile()
