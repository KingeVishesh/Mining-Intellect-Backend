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
- library_search replaces db_analog_search: uses analogs (curated approved analogs)
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
from nodes import gap_detector

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
    library_analogs: List[Dict]         # from `analogs` table (previously approved)
    exa_analogs: List[Dict]             # from Exa web search
    scored_analogs: List[Dict]
    low_confidence: bool                # True when <2 candidates passed L1-L5
    profile_warning: Optional[str]      # Human-readable message when profile too weak
    audit_events: List[Dict]            # Per-candidate decision audit trail
    human_approved: bool
    approved_analogs: List[Dict]
    saved: bool
    error: Optional[str]
    research_attempted: bool            # auto-research loop-guard (set by load_project_and_rule_node)
    skip_exa: bool                      # test/backfill path: use the vetted library only


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
        ("intrusion related",      "intrusion_related"),
        ("irgs",                   "intrusion_related"),
        ("intrusive related",      "intrusion_related"),
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


def _deposit_type_family_from_subtype(subtype: str) -> Optional[str]:
    """Map canonical subtypes to the coarser family used by L2."""
    if not subtype:
        return None
    s = subtype.strip().lower()
    for token, family in (
        ("orogenic", "orogenic"),
        ("irgs", "intrusion_related"),
        ("intrusion", "intrusion_related"),
        ("epithermal", "epithermal"),
        ("carlin", "carlin"),
        ("porphyry", "porphyry"),
        ("vms", "vms"),
        ("vhms", "vms"),
        ("sediment_hosted", "sediment_hosted"),
        ("skarn", "skarn"),
        ("iocg", "iocg"),
    ):
        if token in s:
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
    material = (row.get("material") or "").strip().lower()
    belt = geo_taxonomy.detect_belt_from_row(row)
    explicit_subtype = row.get("deposit_subtype") or geo_taxonomy.detect_subtype(
        row.get("deposit_type"), row.get("mineralization_style"),
        row.get("alteration_signature"), row.get("district") or row.get("location_name"),
    )
    context_blob = " ".join(
        str(row.get(k) or "")
        for k in (
            "tectonic_belt", "district", "region", "location_name",
            "mining_method", "mining_method_class", "processing_method",
            "recovery_method",
        )
    ).lower()
    if (
        material in {"gold", "au"}
        and explicit_subtype == "orogenic_general"
        and str(clean_deposit_type or "").strip().lower() in {"orogenic", "orogenic gold"}
    ):
        clean_deposit_type = "orogenic gold"
        row = {**row, "deposit_type": clean_deposit_type}
    if (
        not clean_deposit_type
        and not explicit_subtype
        and material in {"gold", "au"}
        and belt in geo_taxonomy.BELT_COMPATIBILITY_GROUPS.get("archean_greenstone", frozenset())
    ):
        clean_deposit_type = "orogenic gold"
        row = {**row, "deposit_type": clean_deposit_type}
        explicit_subtype = "orogenic_general"
    if (
        not explicit_subtype
        and material in {"gold", "au"}
        and (belt == "guiana_shield" or "shear" in str(clean_deposit_type or "").lower())
    ):
        clean_deposit_type = "orogenic gold"
        row = {**row, "deposit_type": clean_deposit_type}
        explicit_subtype = "orogenic_general"
    if (
        not explicit_subtype
        and material in {"gold", "au"}
        and (
            belt == "newfoundland_appalachian"
            or "newfoundland" in context_blob
            or "appalachian" in context_blob
        )
    ):
        clean_deposit_type = "orogenic gold"
        row = {**row, "deposit_type": clean_deposit_type}
        explicit_subtype = "orogenic_general"
    if (
        not explicit_subtype
        and material in {"gold", "au"}
        and "open-pit gold" in str(clean_deposit_type or "").lower()
    ):
        clean_deposit_type = "orogenic gold"
        row = {**row, "deposit_type": clean_deposit_type}
        explicit_subtype = "orogenic_general"
    if (
        not explicit_subtype
        and material in {"gold", "au"}
        and "near-surface" in str(clean_deposit_type or "").lower()
        and (belt == "yukon_tintina" or "yukon" in context_blob or "tintina" in context_blob)
    ):
        clean_deposit_type = "intrusion-related gold"
        row = {**row, "deposit_type": clean_deposit_type}
        explicit_subtype = "irgs_general"
    if (
        not clean_deposit_type
        and not explicit_subtype
        and material in {"gold", "au"}
        and (belt == "andean" or "andean" in context_blob or "maricunga" in context_blob)
        and (
            "heap" in context_blob
            or "open pit" in context_blob
            or "open-pit" in context_blob
            or "heap_leach_pad" in context_blob
        )
    ):
        clean_deposit_type = "epithermal-HS"
        row = {**row, "deposit_type": clean_deposit_type}
        explicit_subtype = "high_sulfidation_epithermal"
    if material in {"gold", "au"} and explicit_subtype == "carlin_general":
        belt = "great_basin_carlin"
    subtype_family = _deposit_type_family_from_subtype(explicit_subtype)
    deposit_family = _deposit_type_family(row.get("deposit_type") or "")
    if subtype_family == "sediment_hosted" and deposit_family == "intrusion_related":
        family = deposit_family
    else:
        family = subtype_family or deposit_family
    pattern = row.get("mineralization_pattern") or geo_taxonomy.detect_pattern(
        row.get("mineralization_style"), row.get("mining_method"),
        row.get("processing_method"), row.get("deposit_type"),
    )
    mining_method = (row.get("mining_method") or "").strip().lower()
    mining_class = (row.get("mining_method_class") or "").strip().lower()
    tonnage = _as_positive_float(row.get("tonnage_mt"))
    grade = _as_positive_float(row.get("grade_value"))
    if explicit_subtype == "high_sulfidation_epithermal" and (
        "heap" in mining_method
        or "heap" in mining_class
        or "open" in mining_method
        or "open" in mining_class
    ):
        pattern = "disseminated_bulk"
    elif (
        material in {"gold", "au"}
        and ("open" in mining_method or "open" in mining_class)
        and (
            "open_pit_selective" in mining_class
            or (
                tonnage is not None and tonnage >= 20
                and grade is not None and grade <= 1.5
            )
        )
    ):
        pattern = "disseminated_bulk"
    elif not pattern and explicit_subtype == "high_sulfidation_epithermal":
        pattern = "disseminated_bulk"
    elif not pattern and explicit_subtype == "orogenic_general":
        pattern = "vein_hosted"
    return {
        "name":                 row.get("name"),
        "deposit_type":         clean_deposit_type,
        "mining_method":        row.get("mining_method"),
        "processing_method":    row.get("processing_method"),
        "material":             material,
        "deposit_type_family":  family,
        "deposit_subtype":      explicit_subtype,
        "mineralization_mode":  row.get("mineralization_mode") or geo_taxonomy.detect_mode(
            row.get("processing_method"), row.get("mineralization_style"),
            row.get("district") or row.get("location_name"), row.get("deposit_type"),
        ),
        "tectonic_belt":        belt,
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
        "mineralization_pattern": pattern,
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
        # Sub-trend (geological neighborhood within the tectonic belt).
        # Cortez vs Carlin vs Battle Mountain-Eureka vs Getchell are all
        # great_basin_carlin but very different host stratigraphy. Used
        # for L6.5 rank bonus and to bias the Exa search toward in-trend
        # canonical analogs. None when the location text doesn't match
        # any sub-trend in SUB_TRENDS.
        # Project name added as a detector input — Abore-style cases
        # where district/region are too coarse but the project name
        # itself is the sub-trend signal ("Abore" → asankrangwa_belt).
        "sub_trend":            geo_taxonomy.detect_sub_trend(
            row.get("district"), row.get("region"), row.get("location_name"),
            row.get("name"),
        ),
    }


def _is_gold_like(material: str) -> bool:
    m = (material or "").strip().lower().replace("-", "_").replace(" ", "_")
    return m in {"gold", "au", "gold_silver", "goldandsilver", "gold_and_silver"}


def _context_blob(row: dict, profile: Optional[dict] = None) -> str:
    profile = profile or {}
    fields = (
        "name",
        "deposit_type",
        "deposit_subtype",
        "mineralization_pattern",
        "mining_method",
        "mining_method_class",
        "processing_method",
        "recovery_method",
        "stage",
        "district",
        "region",
        "country",
    )
    return " ".join(
        str(source.get(key) or "")
        for source in (row or {}, profile)
        for key in fields
    ).lower()


def _is_tailings_context(row: dict, profile: Optional[dict] = None) -> bool:
    blob = _context_blob(row, profile)
    return any(token in blob for token in ("tailings", "tailing", "reprocessing", "re-process"))


def _is_open_pit_context(row: dict, profile: Optional[dict] = None) -> bool:
    blob = _context_blob(row, profile)
    return any(
        token in blob
        for token in (
            "open pit",
            "open-pit",
            "open_pit",
            "openpit",
            "heap leach",
            "heap_leach",
        )
    )


def _is_underground_context(row: dict, profile: Optional[dict] = None) -> bool:
    blob = _context_blob(row, profile)
    return any(token in blob for token in ("underground", "ug mine", "narrow vein", "high grade vein"))


_MODELLING_CORE_FIELDS = (
    "deposit_subtype",
    "mineralization_pattern",
    "mineralization_mode",
    "tectonic_belt",
    "host_rock_class",
    "mining_method_class",
    "resource_compliance_standard",
    "resource_vintage_year",
)


def _profile_completeness(profile: dict) -> tuple[int, list[str]]:
    """Return populated count and missing fields for audit / rank damping."""
    missing = [k for k in _MODELLING_CORE_FIELDS if not profile.get(k)]
    return len(_MODELLING_CORE_FIELDS) - len(missing), missing


def _as_positive_float(value) -> float:
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return v if v > 0 else 0.0


def _modellable_resource_issue(candidate: dict, profile: dict, target: dict) -> Optional[str]:
    """Reject candidates that cannot actually support resource modelling.

    The gold lessons require source-backed analogs with published compliant
    resources, not just similar geology. This gate is applied before the
    cascade so every rejection is captured in the analog audit trail and
    stale library rows are marked REVOKED_LIBRARY.
    """
    name = candidate.get("name") or "Unknown"
    t = _as_positive_float(candidate.get("tonnage_mt"))
    g = _as_positive_float(candidate.get("grade_value"))
    if t <= 0 or g <= 0:
        return (
            f"{name} lacks modellable tonnage/grade; analogs must have a "
            "published resource with both tonnage and grade"
        )

    c_unit = candidate.get("grade_unit") or ""
    t_unit = target.get("grade_unit") or ""
    if t_unit and c_unit and not _grade_units_compatible(t_unit, c_unit):
        return (
            f"{name} grade unit {c_unit!r} is incompatible with target unit "
            f"{t_unit!r}"
        )

    material = target.get("material") or candidate.get("material") or ""
    source_url = (candidate.get("source_url") or "").strip()
    if _is_gold_like(material) and candidate.get("source") == "exa" and not source_url:
        return (
            f"{name} is Exa-sourced but has no primary source URL; gold "
            "analog modeling requires source-backed resource numbers"
        )

    compliance = profile.get("resource_compliance_standard")
    if compliance in ("historical", "press_release", "internal"):
        return f"{name} resource standard is non-compliant ({compliance})"

    if _is_gold_like(material):
        if _is_tailings_context(target) and not _is_tailings_context(candidate, profile):
            return f"{name} is not a tailings/reprocessing analog for a tailings gold target"
        if _is_tailings_context(candidate, profile) and not _is_tailings_context(target):
            return f"{name} is a tailings/reprocessing analog and is unsafe for a non-tailings gold target"
        t_mining = (target.get("mining_method_class") or "").strip().lower()
        c_mining = (profile.get("mining_method_class") or "").strip().lower()
        target_open_pit_like = _is_open_pit_context(target) or t_mining == "open_pit_selective"
        target_underground_like = _is_underground_context(target) or t_mining == "underground_vein"
        candidate_open_pit_like = _is_open_pit_context(candidate, profile) or "open" in c_mining
        candidate_underground_like = _is_underground_context(candidate, profile) or "underground" in c_mining
        target_grade = _as_positive_float(target.get("grade_value"))
        target_belt = (target.get("tectonic_belt") or "").strip().lower()
        target_blob = _context_blob(target)
        target_subtype = (target.get("deposit_subtype") or target.get("deposit_type") or "").lower()
        if (
            target_open_pit_like
            and (
                target_belt == "central_african_orogenic"
                or any(token in target_blob for token in ("cameroon", "central african republic", "chad"))
            )
            and (
                any(token in target_subtype for token in ("orogenic", "greenstone", "gold"))
                or target.get("mineralization_pattern") == "vein_hosted"
            )
        ):
            c_belt = (profile.get("tectonic_belt") or "").strip().lower()
            if c_belt == "central_african_copperbelt":
                return (
                    f"{name} is Central African Copperbelt style; unsafe analog "
                    "for a Central African orogenic open-pit gold target"
                )
            if t > 120 or g < 1.5 or g > 2.7:
                return (
                    f"{name} scale/grade ({t:g} Mt at {g:g} g/t) is outside "
                    "the Central African orogenic open-pit gold calibration band"
                )
        if target_underground_like and not target_open_pit_like:
            if c_mining and not geo_taxonomy.mining_method_compatible(t_mining, c_mining):
                return (
                    f"{name} mining method {c_mining!r} is incompatible with "
                    "an underground-vein gold target"
                )
            if not c_mining and g < 2.0:
                return (
                    f"{name} has no mining-method class and low-grade gold "
                    f"({g:g} g/t); unsafe analog for an underground-vein target"
                )
            if not c_mining and t >= 50:
                return (
                    f"{name} has no mining-method class and large resource "
                    f"scale ({t:g} Mt); unsafe analog for an underground-vein target"
                )
        if target_open_pit_like:
            compatible_target_mining = "open_pit_selective"
            if candidate_underground_like and not candidate_open_pit_like:
                return (
                    f"{name} is underground/high-grade-vein style and is unsafe "
                    "for an open-pit-selective gold target"
                )
            if c_mining and not geo_taxonomy.mining_method_compatible(compatible_target_mining, c_mining):
                return (
                    f"{name} mining method {c_mining!r} is incompatible with "
                    "an open-pit-selective gold target"
                )
            if not c_mining and target_grade and target_grade <= 1.5 and g >= 2.0:
                return (
                    f"{name} has no mining-method class and high-grade gold "
                    f"({g:g} g/t); unsafe analog for a low-grade open-pit-selective target"
                )
            if not c_mining and target_grade and target_grade <= 1.5 and t < 10:
                return (
                    f"{name} has no mining-method class and narrow/small resource "
                    f"scale ({t:g} Mt); unsafe analog for an open-pit-selective target"
                )
            if not c_mining and not candidate_open_pit_like and g >= 2.5 and t < 25:
                return (
                    f"{name} has no mining-method class and high-grade/small-resource "
                    f"gold ({t:g} Mt at {g:g} g/t); unsafe analog for an open-pit-selective target"
                )
    return None


def _resource_variant_key(candidate: dict, profile: dict) -> Optional[tuple]:
    """Key near-duplicate resource rows so they cannot overweight a cohort."""
    tonnage = _as_positive_float(candidate.get("tonnage_mt"))
    grade = _as_positive_float(candidate.get("grade_value"))
    if tonnage <= 0 or grade <= 0:
        return None
    family = profile.get("deposit_type_family") or profile.get("deposit_subtype") or ""
    return (family, round(tonnage, 1), round(grade, 1))


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

    # ── L2.5: Tectonic belt compatibility (HARD — fixes Cartier-Cadillac) ──
    # An Abitibi target should not get analogs from the Phanerozoic arcs or
    # Guiana Shield even when subtype + pattern + mining-method all match.
    # belt_compatible() uses BELT_COMPATIBILITY_GROUPS: Archean greenstone
    # belts (Abitibi/Yilgarn/Fennoscandian/Birimian/Guiana/Tanzania/Trans-
    # Hudson) match each other; Phanerozoic arcs (BC/Yukon/Laramide/Andean/
    # Indo-Phil/Lachlan/CAOB) match each other; everything else is strict.
    # A null belt on either side passes through.
    # Per-rule override: belt_strict=False on the magmatic-sulphide Ni rule
    # so Sudbury / Voisey's Bay / Kambalda can analog each other across
    # cratons (industry standard for that deposit class).
    t_belt = target["tectonic_belt"]
    c_belt = candidate["tectonic_belt"]
    rule_belt_strict = (analog_rule or {}).get("belt_strict", True)
    if rule_belt_strict and t_belt and c_belt:
        if not geo_taxonomy.belt_compatible(t_belt, c_belt):
            return False, 0, 0, 0, [
                f"L2.5 belt incompatible ({c_belt} not in same group as {t_belt}): {rule_lessons}"
            ], "L2.5"

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
        if _is_gold_like(target.get("material") or ""):
            return False, 0, 0, 0, [
                f"L3 sub-type unknown on gold candidate (target={t_sub}); enrich or reject before modeling: {rule_lessons}"
            ], "L3"
        # Target has subtype, candidate's is unknown. Don't drop — but record uncertainty.
        evaluated += 1
        reasons.append(f"L3 sub-type unknown on candidate (target={t_sub}): pass-through")

    # ── L4: Mineralization mode (HARD — Lessons L86/L101) ──────────────────
    # primary_sulfide ≠ supergene_oxide (different mineralogy + different metallurgy).
    # BUT: when subtype + belt both match, the mode difference reflects an
    # oxidation/preservation state of the same genetic system rather than a
    # different deposit family. IRGS in the Tintina belt are the canonical
    # example: Fort Knox (primary_sulfide stockwork), Eagle Gold (free-milling
    # oxide cap), Brewery Creek (supergene oxide blanket), and Donlin (deep
    # refractory sulfide) are all the same deposit type at different oxidation
    # states. Soft pass (no rank credit, no hard reject) preserves them in the
    # pool with reduced influence rather than dropping useful scale/grade signal.
    same_subtype_or_rule_sibling = bool(
        target.get("deposit_subtype") and candidate.get("deposit_subtype")
        and (
            target["deposit_subtype"] == candidate["deposit_subtype"]
            or (
                target["deposit_subtype"] in rule_required
                and candidate["deposit_subtype"] in rule_required
            )
        )
    )
    same_subtype_and_belt = bool(
        same_subtype_or_rule_sibling
        and target.get("tectonic_belt") and candidate.get("tectonic_belt")
        and geo_taxonomy.belt_compatible(target["tectonic_belt"], candidate["tectonic_belt"])
    )
    if not geo_taxonomy.mode_compatible(target["mineralization_mode"], candidate["mineralization_mode"]):
        if same_subtype_and_belt:
            # Soft pass — note the mismatch but don't reject. No rank points
            # awarded for L4 since the mode actually disagrees.
            evaluated += 1
            reasons.append(
                f"L4 mode soft-pass ({candidate['mineralization_mode']} ≠ "
                f"{target['mineralization_mode']}; same {target['deposit_subtype']} "
                f"+ {target['tectonic_belt']} → oxidation/preservation variant)"
            )
        else:
            return False, 0, 0, 0, [
                f"L4 mode mismatch ({candidate['mineralization_mode']} ≠ {target['mineralization_mode']}): {rule_lessons}"
            ], "L4"
    elif target["mineralization_mode"] and candidate["mineralization_mode"]:
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
    #
    # Exception: same subtype + same tectonic belt. IRGS in the Tintina belt
    # spans stockwork (Fort Knox), disseminated_bulk (Coffee), sheeted_vein
    # (AurMac), and vein_hosted (Valley/Rogue) — these are continuum
    # variants of the same intrusion-related system, not distinct deposit
    # types. Soft pass with reduced rank credit preserves them in the pool.
    t_pattern = target["mineralization_pattern"]
    c_pattern = candidate["mineralization_pattern"]
    if t_pattern and c_pattern and t_pattern != c_pattern:
        if same_subtype_and_belt:
            evaluated += 1
            reasons.append(
                f"L4.5 pattern soft-pass ({c_pattern} ≠ {t_pattern}; same "
                f"{target['deposit_subtype']} + {target['tectonic_belt']} → "
                f"system geometry variant)"
            )
        else:
            return False, 0, 0, 0, [
                f"L4.5 pattern mismatch ({c_pattern} ≠ {t_pattern}): {rule_lessons}"
            ], "L4.5"
    elif t_pattern and c_pattern:
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

    # ── L6: Tectonic belt rank (HARD at L2.5; here only rank shading) ──────
    # By the time a candidate reaches L6 it has already passed the L2.5
    # belt-group hard filter. Same exact belt earns full points;
    # same-group-different-belt earns half (so Abitibi vs Yilgarn ranks
    # below Abitibi vs Abitibi but still ahead of unknown-belt analogs).
    if target["tectonic_belt"] and candidate["tectonic_belt"]:
        evaluated += 1
        if target["tectonic_belt"] == candidate["tectonic_belt"]:
            matched += 1
            rank_pts += 30
            reasons.append(f"L6 belt match ({candidate['tectonic_belt']}): +30")
        else:
            matched += 1
            rank_pts += 15
            reasons.append(
                f"L6 belt same-group ({candidate['tectonic_belt']} ~ "
                f"{target['tectonic_belt']}): +15"
            )

    # ── L6.5: Sub-trend rank (RANK +40) ────────────────────────────────────
    # Same sub-trend within the same belt is geologically much closer than
    # same-belt-different-sub-trend (Cortez Trend vs Battle Mountain-Eureka,
    # both great_basin_carlin). Same host stratigraphy, same age of
    # mineralization, same structural plumbing.
    #
    # Bonus was +15 → +25 → +40 across successive judge audits.
    # At +25 the LLM judge still saw in-camp matches losing the top-4
    # race to off-camp library entries with strong L8/L9/L10 scores
    # (e.g. Westwood/Casa Berardi for Cartier-Cadillac). +40 reliably
    # dominates: same-sub-trend is the single strongest signal short of
    # the L1-L5 hard filters, because in-camp geology = same host
    # stratigraphy + same orogenic event + same structural plumbing.
    t_subtrend = target.get("sub_trend")
    c_subtrend = candidate.get("sub_trend")
    if t_subtrend and c_subtrend and t_subtrend == c_subtrend:
        matched += 1
        evaluated += 1
        rank_pts += 40
        reasons.append(f"L6.5 sub-trend match ({c_subtrend}): +40")

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
    stops = {"project", "mine", "mines", "mining", "deposit", "property", "corp", "inc",
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
    broad_words = {
        "gold", "silver", "copper", "nickel", "uranium", "iron", "ore",
        "zone", "trend", "camp", "belt", "shear", "north", "south", "east",
        "west", "central", "main",
    }
    p_core = p_words - broad_words
    c_core = c_words - broad_words
    if p_core and c_core and (p_core <= c_core or c_core <= p_core):
        return True
    distinctive_overlap = {
        w for w in (p_words & c_words)
        if len(w) >= 5 and w not in broad_words
    }
    if distinctive_overlap and min(len(p_words), len(c_words)) <= 4:
        return True
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

def _derive_rule_inputs(project: Dict) -> tuple:
    """Pull (material, deposit_type, deposit_subtype, pattern) from a project
    row, applying the same sanitisation + heuristic detection used by the
    rule lookup. Factored out so the auto-research recovery path can re-
    derive after the project is re-enriched, without duplicating logic."""
    material = project.get("material") or ""
    deposit_type = rules_engine.sanitize_deposit_type(project.get("deposit_type"))
    deposit_subtype = project.get("deposit_subtype") or geo_taxonomy.detect_subtype(
        deposit_type, project.get("mineralization_style"),
        project.get("alteration_signature"),
        project.get("district") or project.get("location_name"),
    )
    belt = geo_taxonomy.detect_belt_from_row(project)
    context_blob = " ".join(
        str(project.get(k) or "")
        for k in (
            "tectonic_belt", "district", "region", "location_name",
            "mining_method", "mining_method_class", "processing_method",
            "recovery_method",
        )
    ).lower()
    if (
        (material or "").strip().lower() in {"gold", "au"}
        and deposit_subtype == "orogenic_general"
        and str(deposit_type or "").strip().lower() in {"orogenic", "orogenic gold"}
    ):
        deposit_type = "orogenic gold"
    if (
        not deposit_type
        and not deposit_subtype
        and (material or "").strip().lower() in {"gold", "au"}
        and belt in geo_taxonomy.BELT_COMPATIBILITY_GROUPS.get("archean_greenstone", frozenset())
    ):
        deposit_type = "orogenic gold"
        deposit_subtype = "orogenic_general"
    if (
        not deposit_subtype
        and (material or "").strip().lower() in {"gold", "au"}
        and (belt == "guiana_shield" or "shear" in str(deposit_type or "").lower())
    ):
        deposit_type = "orogenic gold"
        deposit_subtype = "orogenic_general"
    if (
        not deposit_subtype
        and (material or "").strip().lower() in {"gold", "au"}
        and (
            belt == "newfoundland_appalachian"
            or "newfoundland" in context_blob
            or "appalachian" in context_blob
        )
    ):
        deposit_type = "orogenic gold"
        deposit_subtype = "orogenic_general"
    if (
        not deposit_subtype
        and (material or "").strip().lower() in {"gold", "au"}
        and "open-pit gold" in str(deposit_type or "").lower()
    ):
        deposit_type = "orogenic gold"
        deposit_subtype = "orogenic_general"
    if (
        not deposit_subtype
        and (material or "").strip().lower() in {"gold", "au"}
        and "near-surface" in str(deposit_type or "").lower()
        and (belt == "yukon_tintina" or "yukon" in context_blob or "tintina" in context_blob)
    ):
        deposit_type = "intrusion-related gold"
        deposit_subtype = "irgs_general"
    if (
        not deposit_type
        and not deposit_subtype
        and (material or "").strip().lower() in {"gold", "au"}
        and (belt == "andean" or "andean" in context_blob or "maricunga" in context_blob)
        and (
            "heap" in context_blob
            or "open pit" in context_blob
            or "open-pit" in context_blob
            or "heap_leach_pad" in context_blob
        )
    ):
        deposit_type = "epithermal-HS"
        deposit_subtype = "high_sulfidation_epithermal"
    if (material or "").strip().lower() in {"gold", "au"} and deposit_subtype == "carlin_general":
        belt = "great_basin_carlin"
    pattern = project.get("mineralization_pattern") or geo_taxonomy.detect_pattern(
        project.get("mineralization_style"), project.get("mining_method"),
        project.get("processing_method"), deposit_type,
    )
    if deposit_subtype == "high_sulfidation_epithermal" and (
        not pattern
        or "heap" in context_blob
        or "open pit" in context_blob
        or "open-pit" in context_blob
        or "heap_leach_pad" in context_blob
    ):
        pattern = "disseminated_bulk"
    if deposit_subtype == "orogenic_general":
        mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
        tonnage = _as_positive_float(project.get("tonnage_mt"))
        grade = _as_positive_float(project.get("grade_value"))
        if (
            "open_pit_selective" in mining
            or (
                "open" in mining
                and tonnage is not None and tonnage >= 20
                and grade is not None and grade <= 1.5
            )
        ):
            pattern = "disseminated_bulk"
        elif not pattern:
            pattern = "vein_hosted"
    return material, deposit_type, deposit_subtype, pattern


def _trigger_project_research(project: Dict) -> Optional[Dict]:
    """Run the project_research graph to fill missing fields, then return
    the freshly-loaded project record. Returns None on any failure so the
    caller can surface a clear "research is incomplete" warning instead.

    Imported lazily to avoid a circular import at module load — both
    graphs.analog_finder and graphs.project_research are top-level graphs."""
    from graphs import project_research  # lazy to avoid module-init order issues

    project_id = project.get("id")
    if not project_id:
        return None
    research_input = {
        "project_id": project_id,
        "project_name": project.get("name") or "",
        "material": project.get("material") or "",
        "company": project.get("company_name") or "",
    }
    logger.info(
        f"[auto-research] Triggering project_research for {project_id} "
        f"({project.get('name')!r}) — deposit_type and deposit_subtype both empty"
    )
    research_result = project_research.graph.invoke(research_input)
    if research_result.get("error"):
        logger.warning(f"[auto-research] project_research failed: {research_result['error']}")
        return None
    if not research_result.get("saved"):
        logger.warning("[auto-research] project_research returned without saving")
        return None
    reloaded = supabase_ops.get_project(project_id)
    if not reloaded:
        logger.warning(f"[auto-research] project {project_id} disappeared after research")
        return None
    logger.info(
        f"[auto-research] reload OK | deposit_type={reloaded.get('deposit_type')!r} "
        f"subtype={reloaded.get('deposit_subtype')!r} "
        f"belt={reloaded.get('tectonic_belt')!r}"
    )
    return reloaded


def load_project_and_rule_node(state: AnalogState) -> AnalogState:
    """Load project + fetch matching analog_selection rule.

    When the rule lookup fails because the project lacks both deposit_type
    and deposit_subtype, the project_research graph is triggered inline to
    fill the gap, the project is reloaded, and the rule lookup is retried.
    A `research_attempted` sentinel prevents an infinite loop if the
    research pass still doesn't yield the fields. Net effect: callers no
    longer have to manually run project_research before re-trying analogs.
    """
    project_id = state["project_id"]
    project = supabase_ops.get_project(project_id)
    if not project:
        return {"error": f"Project {project_id} not found"}

    material, deposit_type, deposit_subtype, pattern = _derive_rule_inputs(project)
    analog_rule = rules_engine.get_analog_rule(material, deposit_type, deposit_subtype, pattern)

    # Auto-research fallback: when no rule was found AND both rule-routing
    # fields (deposit_type, deposit_subtype) are empty, the only thing
    # standing between the user and analogs is missing enrichment. Run
    # project_research once, reload, and retry the rule lookup.
    if analog_rule is None and not deposit_type and not deposit_subtype and not state.get("research_attempted"):
        try:
            refreshed = _trigger_project_research(project)
        except Exception as e:
            logger.warning(f"[auto-research] non-fatal exception: {e}")
            refreshed = None
        if refreshed:
            project = refreshed
            material, deposit_type, deposit_subtype, pattern = _derive_rule_inputs(project)
            analog_rule = rules_engine.get_analog_rule(material, deposit_type, deposit_subtype, pattern)

    logger.info(
        f"[load] {project.get('name')} | material={material} deposit={deposit_type} "
        f"subtype={deposit_subtype} pattern={pattern} "
        f"rule={'✓ ' + analog_rule.get('rule_id', '') if analog_rule else '✗ none'}"
    )
    return {
        "project": project,
        "analog_rule": analog_rule,
        "research_attempted": True,
        "error": None,
    }


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
    """Search the `analogs` table for previously approved analogs of this commodity."""
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

    # Pass the target belt so the library query can spend its 200-row
    # budget on belt-compatible candidates only. Without this, the cascade's
    # L2.5 belt-group filter would later REVOKE most of what we fetched,
    # while in-belt candidates (e.g. Bulyanhulu for a Tanzanian target) get
    # silently truncated at the SQL layer.
    target_belt = (state.get("target_profile") or {}).get("tectonic_belt")
    analogs = supabase_ops.get_approved_analogs(
        material, deposit_type, limit=200,
        deposit_subtype=deposit_subtype,
        deposit_subtypes=accepted_subtypes or None,
        target_tectonic_belt=target_belt,
    )
    logger.info(
        f"[library] Found {len(analogs)} previously approved analogs "
        f"(filter: subtypes={accepted_subtypes!r} dep={deposit_type!r})"
    )
    return {"library_analogs": analogs}


def exa_search_node(state: AnalogState) -> AnalogState:
    """Find comparable projects via Exa using rule-driven targeted query."""
    if state.get("error") or state.get("skip_exa"):
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

    # No-rule case — get_analog_rule returned None because the project
    # research step left deposit_type AND deposit_subtype empty (or the
    # values don't map to any of our deposit-type-specific rules). We
    # refuse to score: a wrong analog is worse than no analog. The user is
    # told exactly what's missing so they can re-research the project.
    if analog_rule is None:
        material = project.get("material") or "?"
        dep_t = project.get("deposit_type") or "(empty)"
        dep_s = project.get("deposit_subtype") or "(empty)"
        logger.warning(
            f"[cascade] NO RULE for project material={material} "
            f"deposit_type={dep_t} subtype={dep_s} — refusing to score."
        )
        # Flag the data gap so the weekly digest captures it.
        try:
            gap_detector.save_gaps(gap_detector.detect_gaps(
                project=project,
                target_profile=target_profile,
                analog_rule=None,
                library_analogs=[],
                scored_analogs=[],
                low_confidence=True,
                relaxed_mode=False,
            ))
        except Exception as gd_err:
            logger.warning(f"[gap_detector] no-rule path non-fatal: {gd_err}")
        # If the load step already tried an inline project_research pass
        # and we still don't have a rule, the data gap is genuine — Exa
        # couldn't recover the geological fields. Tell the user that
        # plainly so they don't expect another auto-retry to help.
        auto_tried = state.get("research_attempted")
        action_hint = (
            "Auto-research was already attempted on this run and could not "
            "recover the missing fields — set deposit_type/deposit_subtype "
            "manually, then re-run analog_finder."
            if auto_tried
            else "Re-run project_research or set deposit_type / deposit_subtype "
                 "manually, then re-run analog_finder."
        )
        return {
            "scored_analogs": [],
            "low_confidence": True,
            "profile_warning": (
                f"Cannot select analogs: project research is incomplete. "
                f"Material={material}; deposit_type={dep_t}; "
                f"deposit_subtype={dep_s}. The analog finder requires at "
                f"least one of (deposit_type, deposit_subtype) to map the "
                f"project to a geological rule. {action_hint}"
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

    # If target enrichment detects a more specific pattern/mining method than
    # the rule lookup used, do not let the rule exclude the target's own class.
    target_pattern = target_profile.get("mineralization_pattern")
    if target_pattern in excluded_patterns:
        excluded_patterns.remove(target_pattern)
        logger.info(
            f"[cascade] target pattern {target_pattern!r} overrides "
            f"rule excluded_patterns for {(analog_rule or {}).get('rule_id','none')}"
        )
    if target_pattern and required_patterns and target_pattern not in required_patterns:
        logger.info(
            f"[cascade] target pattern {target_pattern!r} overrides "
            f"rule required_patterns={sorted(required_patterns)} for "
            f"{(analog_rule or {}).get('rule_id','none')}"
        )
        required_patterns = {target_pattern}
    target_mining_method = target_profile.get("mining_method_class")
    if target_mining_method in excluded_mining_methods:
        excluded_mining_methods.remove(target_mining_method)
        logger.info(
            f"[cascade] target mining method {target_mining_method!r} overrides "
            f"rule excluded_mining_methods for {(analog_rule or {}).get('rule_id','none')}"
        )
    if (
        target_mining_method
        and required_mining_methods
        and target_mining_method not in required_mining_methods
    ):
        logger.info(
            f"[cascade] target mining method {target_mining_method!r} overrides "
            f"rule required_mining_methods={sorted(required_mining_methods)} for "
            f"{(analog_rule or {}).get('rule_id','none')}"
        )
        required_mining_methods = {target_mining_method}

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
    seen_resource_variants: set[tuple] = set()
    for c in pre_filtered:
        cand_profile = _build_profile(c)

        resource_issue = _modellable_resource_issue(c, cand_profile, target_profile)
        if resource_issue:
            logger.info(f"[cascade] DROP non-modellable resource: {c.get('name')} — {resource_issue}")
            dropped_counts["non_modellable_resource"] = dropped_counts.get("non_modellable_resource", 0) + 1
            _emit("DROP", "non_modellable_resource", c, cand_profile, resource_issue)
            continue

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

        variant_key = _resource_variant_key(c, cand_profile)
        if variant_key and variant_key in seen_resource_variants:
            reason = (
                "near-duplicate resource variant already passed "
                f"(family/tonnage/grade key={variant_key!r})"
            )
            logger.info(f"[cascade] DROP duplicate resource variant: {c.get('name')} — {reason}")
            dropped_counts["duplicate_resource_variant"] = (
                dropped_counts.get("duplicate_resource_variant", 0) + 1
            )
            _emit("DROP", "duplicate_resource_variant", c, cand_profile, reason)
            continue
        if variant_key:
            seen_resource_variants.add(variant_key)

        completeness, missing_core = _profile_completeness(cand_profile)
        if missing_core:
            # Do not hard-drop otherwise valid analogs simply because a source
            # omitted host/recovery/stage metadata, but make the uncertainty
            # visible and let better-enriched analogs rank ahead. Gold lessons
            # treat these fields as crucial to the 95% analog standard.
            penalty = min(20, 3 * len(missing_core))
            rank_pts = max(0, rank_pts - penalty)
            reasons.append(
                f"Profile completeness {completeness}/{len(_MODELLING_CORE_FIELDS)}; "
                f"missing {', '.join(missing_core)}: -{penalty}"
            )

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
            "_profile_completeness": completeness,
            "_missing_core_fields": missing_core,
            # Internal-only; used by the sub-trend semi-hard filter below.
            # The raw candidate dict `c` doesn't carry sub_trend — that's
            # derived in _build_profile.
            "_sub_trend": cand_profile.get("sub_trend"),
            "approved": False,
        })
        _emit("PASS", "cascade", c, cand_profile,
              f"matched {matched}/{evaluated} dimensions",
              rank_pts=rank_pts, score=score)

    logger.info(
        f"[cascade] {len(survivors)} survivors | dropped: {dict(dropped_counts) or 'none'}"
    )

    # ── Step D: Rank by total points; HARD CAP at 6 ────────────────────────
    # Gold backtests showed that 1-2 analog cohorts create unstable drill-to-MRE
    # transformations. Keep the cap tight for UX, but allow enough candidates
    # for a median transform and mark thinner cohorts low-confidence.
    ranked = sorted(survivors, key=lambda x: -x["_rank_pts"])

    # ── Step D.1: Semi-hard sub-trend filter (Buckreef audit, 2026-05-22) ──
    # When the target has a sub-trend and ≥3 in-sub-trend candidates pass
    # the cascade, restrict the top-4 to in-sub-trend candidates ONLY.
    # Cross-sub-trend candidates are diverted to NEAR_MISS audit events
    # (SUB_TREND_FILTERED level) rather than backfilling top-4. This stops
    # the case where a Tanzanian Buckreef target gets Canadian Malartic UG
    # as the 4th pick just because Exa returned only 3 in-trend candidates.
    #
    # Falls back to the previous behavior when in-trend coverage is thin
    # (<3 in-trend), so obscure-geology projects still get a top-4 cohort.
    target_subtrend = target_profile.get("sub_trend") if target_profile else None
    sub_trend_filtered: List[Dict] = []
    if target_subtrend:
        in_trend = [r for r in ranked if r.get("_sub_trend") == target_subtrend]
        cross_trend = [r for r in ranked if r.get("_sub_trend") != target_subtrend]
        if len(in_trend) >= 3:
            logger.info(
                f"[cascade] sub-trend semi-hard filter: "
                f"{len(in_trend)} in-trend ({target_subtrend}) candidates "
                f"— restricting top-4 to in-trend; "
                f"{len(cross_trend)} cross-trend dropped to NEAR_MISS"
            )
            sub_trend_filtered = cross_trend
            ranked = in_trend

    min_cohort_size = 3 if (project.get("material") or "").strip().lower() in {"gold", "au"} else 2
    low_confidence = len(ranked) < min_cohort_size
    if low_confidence:
        logger.warning(
            f"[cascade] Only {len(ranked)} candidate(s) passed L1-L5 — "
            f"flagging low_confidence; returning best available without padding"
        )
        top = ranked[:min_cohort_size]
    else:
        top = ranked[:6]

    # Audit events for cross-sub-trend candidates that the semi-hard
    # filter pushed out of contention. Useful so the user can see WHY a
    # geologically-valid candidate (e.g. Westwood for a Tanzanian target)
    # didn't make top-4.
    for stf in sub_trend_filtered[:5]:  # cap at 5 to avoid log bloat
        audit_events.append({
            "candidate_name": stf.get("name") or "Unknown",
            "candidate_source": stf.get("source"),
            "decision": "NEAR_MISS",
            "level": "SUB_TREND_FILTERED",
            "rule_id": rule_id_for_audit,
            "lessons": lessons.resolve_lesson_ids(rule_lesson_ids_for_audit),
            "detected_profile": {k: stf.get(k) for k in _PROFILE_DIMENSIONS},
            "reason": (
                f"passed cascade (rank_pts={stf.get('_rank_pts')}) but dropped "
                f"from top-4 by sub-trend semi-hard filter: target sub-trend "
                f"is {target_subtrend!r}, candidate sub-trend is "
                f"{stf.get('sub_trend')!r}"
            ),
            "rank_pts": stf.get("_rank_pts"),
            "similarity_score": stf.get("similarity_score"),
        })

    # Near-miss observability — survivors that passed every hard filter but
    # were squeezed out by the top-4 cap get a NEAR_MISS audit event. Makes
    # the cap auditable: if a strong 5th candidate exists, it's visible in
    # the audit trail rather than silently lost.
    near_misses = ranked[6:9]  # top-3 just below the cap
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
        s.pop("_profile_completeness", None)
        s.pop("_missing_core_fields", None)
        s.pop("_sub_trend", None)

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

    # ── Gap detection — write structured diagnostics to analog_quality_gaps ─
    # Runs after the cascade finishes so we can inspect the actual library/
    # scored output. Non-fatal: any failure here is swallowed and logged.
    try:
        gaps = gap_detector.detect_gaps(
            project=project,
            target_profile=target_profile,
            analog_rule=analog_rule,
            library_analogs=library,
            scored_analogs=result.get("scored_analogs", []),
            low_confidence=result.get("low_confidence", False),
            relaxed_mode=relaxed_mode,
        )
        if gaps:
            gap_detector.save_gaps(gaps)
            logger.info(
                f"[gap_detector] flagged {len(gaps)} quality gap(s): "
                f"{[g['gap_type'] for g in gaps]}"
            )
    except Exception as gd_err:
        logger.warning(f"[gap_detector] non-fatal failure: {gd_err}")

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
        # Upsert the cascade-passing analogs into the shared `analogs` library
        # so future runs of any project can reuse them. Cascade-pass is the
        # quality gate — no human approval needed.
        try:
            supabase_ops.upsert_analog_library(
                project_id=state["project_id"],
                approved=analogs,
            )
        except Exception as le:
            logger.warning(f"[save] Library upsert failed: {le}")
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
