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

from nodes import exa_search, field_extractor, rules_engine, supabase_ops, geo_taxonomy

logger = logging.getLogger(__name__)


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
        "grade_value":          row.get("grade_value"),
        "grade_unit":           row.get("grade_unit"),
        "tonnage_mt":           row.get("tonnage_mt"),
        "country":              (row.get("country") or "").strip().lower(),
        "district":             (row.get("district") or "").strip(),
        "host_rock":            (row.get("host_rock") or "").strip(),
        "mineralization_style": (row.get("mineralization_style") or "").strip(),
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


def _is_self_analog(project_name: str, candidate_name: str) -> bool:
    """True if candidate is likely the same project as the target."""
    p = _norm_name(project_name)
    c = _norm_name(candidate_name)
    if not p or not c:
        return False
    if p == c:
        return True
    p_words = set(p.split())
    c_words = set(c.split())
    if not p_words or not c_words:
        return False
    overlap = len(p_words & c_words)
    # >70% word overlap in the shorter name → same project
    return overlap / min(len(p_words), len(c_words)) >= 0.70


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
    deposit_type = project.get("deposit_type")
    # Subtype takes precedence — alkalic_porphyry routes to the dedicated alkalic
    # rule even when deposit_type is just "porphyry copper-gold".
    deposit_subtype = project.get("deposit_subtype") or geo_taxonomy.detect_subtype(
        deposit_type, project.get("mineralization_style"),
        project.get("alteration_signature"),
        project.get("district") or project.get("location_name"),
    )
    analog_rule = rules_engine.get_analog_rule(material, deposit_type, deposit_subtype)

    logger.info(
        f"[load] {project.get('name')} | material={material} deposit={deposit_type} "
        f"subtype={deposit_subtype} rule={'✓ ' + analog_rule.get('rule_id', '') if analog_rule else '✗ none'}"
    )
    return {"project": project, "analog_rule": analog_rule, "error": None}


def build_target_profile_node(state: AnalogState) -> AnalogState:
    """Derive the target project's geological identity profile (used for cascading match)."""
    if state.get("error"):
        return {"target_profile": None}
    project = state["project"] or {}
    profile = _build_profile(project)
    logger.info(
        f"[profile] target subtype={profile['deposit_subtype']!r} "
        f"mode={profile['mineralization_mode']!r} belt={profile['tectonic_belt']!r} "
        f"metal={profile['metal_suite']!r} recovery={profile['recovery_method']!r}"
    )
    return {"target_profile": profile}


def library_search_node(state: AnalogState) -> AnalogState:
    """Search report_analogs for previously approved analogs of this commodity."""
    if state.get("error"):
        return {"library_analogs": []}

    project = state["project"]
    material = project.get("material") or ""
    deposit_type = project.get("deposit_type")

    analogs = supabase_ops.get_approved_analogs(material, deposit_type, limit=20)
    logger.info(f"[library] Found {len(analogs)} previously approved analogs")
    return {"library_analogs": analogs}


def exa_search_node(state: AnalogState) -> AnalogState:
    """Find comparable projects via Exa using rule-driven targeted query."""
    if state.get("error"):
        return {"exa_analogs": []}

    project = state["project"]
    analog_rule = state.get("analog_rule")
    material = project.get("material", "")
    deposit_type = project.get("deposit_type", "")
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
    rule_exclusions = _parse_exclusions((analog_rule or {}).get("analog_criteria") or [])
    # Structured exclusions from the rule (subtypes/modes/recovery the rule forbids)
    excluded_subtypes = set((analog_rule or {}).get("excluded_subtypes") or [])
    excluded_modes = set((analog_rule or {}).get("excluded_modes") or [])
    excluded_recovery = set((analog_rule or {}).get("excluded_recovery") or [])
    # Positive required list — when the rule specifies required_subtypes,
    # any candidate with a CONFIDENTLY-DETECTED different subtype is dropped.
    # Candidates with null subtype are NOT dropped here (would discard too many
    # poorly-enriched library/exa candidates); they fall through to L3 instead.
    required_subtypes = set((analog_rule or {}).get("required_subtypes") or [])

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

    # ── Step B: Self-analog pre-filter (cheap; before profile build) ───────
    pre_filtered: list[dict] = []
    for c in deduped:
        if project_name and _is_self_analog(project_name, c.get("name") or ""):
            logger.info(f"[cascade] DROP self-analog: {c.get('name')}")
            continue
        pre_filtered.append(c)

    # ── Step C: Cascading match per candidate ──────────────────────────────
    survivors: list[dict] = []
    dropped_counts: dict[str, int] = {}
    for c in pre_filtered:
        cand_profile = _build_profile(c)

        # Rule-driven structured exclusions (Lessons L86/L101 from the rule itself)
        if cand_profile["deposit_subtype"] and cand_profile["deposit_subtype"] in excluded_subtypes:
            logger.info(f"[cascade] DROP rule subtype-exclusion ({cand_profile['deposit_subtype']}): {c.get('name')}")
            dropped_counts["rule_subtype"] = dropped_counts.get("rule_subtype", 0) + 1
            continue
        if cand_profile["mineralization_mode"] and cand_profile["mineralization_mode"] in excluded_modes:
            logger.info(f"[cascade] DROP rule mode-exclusion ({cand_profile['mineralization_mode']}): {c.get('name')}")
            dropped_counts["rule_mode"] = dropped_counts.get("rule_mode", 0) + 1
            continue
        if cand_profile["recovery_method"] and cand_profile["recovery_method"] in excluded_recovery:
            logger.info(f"[cascade] DROP rule recovery-exclusion ({cand_profile['recovery_method']}): {c.get('name')}")
            dropped_counts["rule_recovery"] = dropped_counts.get("rule_recovery", 0) + 1
            continue

        # Positive required_subtypes filter — when the rule pins a subtype list,
        # any candidate with a detected subtype OUTSIDE that list is dropped.
        # This catches candidates that fell through the negative excluded_subtypes
        # check because the exclusion list wasn't exhaustive (e.g. Jasperoide skarn
        # for the alkalic-porphyry rule whose excluded list didn't enumerate skarn).
        if required_subtypes and cand_profile["deposit_subtype"]:
            if cand_profile["deposit_subtype"] not in required_subtypes:
                logger.info(
                    f"[cascade] DROP rule required-subtype mismatch "
                    f"({cand_profile['deposit_subtype']} not in {sorted(required_subtypes)}): {c.get('name')}"
                )
                dropped_counts["rule_required_subtype"] = dropped_counts.get("rule_required_subtype", 0) + 1
                continue

        # Strict-mode drop for unenriched candidates — when the rule pins a
        # subtype list, candidates with NO subtype AND NO deposit_type / no
        # mineralization_style cannot be classified. Better to drop than
        # include unsubstantiated analogs (the Hat Copper run picked
        # La Granja and Sherridon precisely because they had no enrichment
        # and the cascade had nothing to filter on).
        if required_subtypes and not cand_profile["deposit_subtype"]:
            has_text = bool(
                (c.get("deposit_type") or "").strip()
                or (c.get("mineralization_style") or "").strip()
            )
            if not has_text:
                logger.info(
                    f"[cascade] DROP unenriched (no subtype, no deposit_type, no min_style): {c.get('name')}"
                )
                dropped_counts["unenriched"] = dropped_counts.get("unenriched", 0) + 1
                continue

        # Legacy "Exclude X analogs" text patterns (for rules without structured fields)
        c_dep_lower = (c.get("deposit_type") or "").lower()
        if c_dep_lower and rule_exclusions:
            if any(excl in c_dep_lower for excl in rule_exclusions):
                logger.info(f"[cascade] DROP rule text-exclusion: {c.get('name')}")
                dropped_counts["rule_text"] = dropped_counts.get("rule_text", 0) + 1
                continue

        passes, rank_pts, matched, evaluated, reasons, dropped_at = _cascading_match(
            target_profile, cand_profile, analog_rule,
        )
        if not passes:
            logger.info(f"[cascade] DROP @{dropped_at}: {c.get('name')} — {reasons[0] if reasons else ''}")
            dropped_counts[dropped_at or "unknown"] = dropped_counts.get(dropped_at or "unknown", 0) + 1
            continue

        # similarity_score = matched / evaluated (as percentage), or None when too few signals
        score = (
            round(matched / evaluated * 100, 1)
            if evaluated >= 2 else None
        )
        survivors.append({
            **c,
            "similarity_score": score,
            "similarity_reasons": reasons,
            "_rank_pts": rank_pts,
            "_dimensions_matched": matched,
            "_dimensions_evaluated": evaluated,
            "approved": False,
        })

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

    return {"scored_analogs": top, "low_confidence": low_confidence}


def human_review_analog_node(state: AnalogState) -> AnalogState:
    """Auto-approve all scored analogs — no human interrupt in pipeline mode."""
    approved = state.get("scored_analogs", [])
    return {"human_approved": True, "approved_analogs": approved}


def save_analogs_node(state: AnalogState) -> AnalogState:
    """Save approved analogs to Supabase."""
    if not state.get("human_approved"):
        logger.info("[save] Human rejected — not saving")
        return {"saved": False}

    analogs = state.get("approved_analogs", [])
    for a in analogs:
        a["approved"] = True

    try:
        supabase_ops.save_analogs(state["project_id"], analogs)
        logger.info(f"[save] Saved {len(analogs)} analogs for project {state['project_id']}")
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
builder.add_node("human_review", human_review_analog_node)
builder.add_node("save_analogs", save_analogs_node)

builder.set_entry_point("load_project_and_rule")

# load → build_target_profile → parallel fan-out (library + exa)
builder.add_edge("load_project_and_rule", "build_target_profile")
builder.add_edge("build_target_profile", "library_search")
builder.add_edge("build_target_profile", "exa_search")

# Fan-in: both feed into combine (LangGraph waits for both before running combine)
builder.add_edge("library_search", "combine_filter_score")
builder.add_edge("exa_search", "combine_filter_score")

builder.add_edge("combine_filter_score", "human_review")
builder.add_edge("human_review", "save_analogs")
builder.add_edge("save_analogs", END)

graph = builder.compile()
