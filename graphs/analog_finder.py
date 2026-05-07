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
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from nodes import exa_search, field_extractor, rules_engine, supabase_ops

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────

class AnalogState(TypedDict, total=False):
    project_id: str
    project: Optional[Dict]
    analog_rule: Optional[Dict]         # matched analog_selection rule from compiled_rules
    library_analogs: List[Dict]         # from report_analogs (previously approved)
    exa_analogs: List[Dict]             # from Exa web search
    scored_analogs: List[Dict]
    human_approved: bool
    approved_analogs: List[Dict]
    saved: bool
    error: Optional[str]


# ── Scoring helpers ────────────────────────────────────────────────────────────

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


def _deposit_family(dep: str) -> str:
    """Reduce a deposit_type to a family key for partial matching."""
    dep = dep.lower().replace("-", " ").replace("_", " ")
    for family in ("epithermal", "porphyry", "vms", "skarn", "sediment",
                   "orogenic", "carlin", "laterite", "magmatic", "unconformity",
                   "roll front", "bif", "magnetite", "merensky", "platreef", "ug2"):
        if family in dep:
            return family
    return dep


def _score_candidate(project: dict, analog: dict) -> tuple[Optional[float], list[str]]:
    """
    Deterministic 5-factor scoring. Returns (score | None, reasons).
    score is None when < 2 factors can be evaluated (insufficient data).
    Max possible score: 100.
    """
    earned = 0.0
    possible = 0.0
    reasons: list[str] = []

    # ── Factor 1: Grade similarity (max 35) ─────────────────────────────────
    p_g = float(project.get("grade_value") or 0)
    a_g = float(analog.get("grade_value") or 0)
    if p_g > 0 and a_g > 0:
        pts = _ratio_score(p_g, a_g, [(1.5, 35), (2.0, 25), (3.0, 15), (5.0, 5)])
        ratio = round(max(p_g, a_g) / min(p_g, a_g), 1)
        earned += pts
        possible += 35
        reasons.append(f"Grade {ratio}× match: {int(pts)}/35 pts")

    # ── Factor 2: Tonnage similarity (max 25) ───────────────────────────────
    p_t = float(project.get("tonnage_mt") or 0)
    a_t = float(analog.get("tonnage_mt") or 0)
    if p_t > 0 and a_t > 0:
        pts = _ratio_score(p_t, a_t, [(2.0, 25), (3.0, 18), (5.0, 10), (10.0, 3)])
        ratio = round(max(p_t, a_t) / min(p_t, a_t), 1)
        earned += pts
        possible += 25
        reasons.append(f"Tonnage {ratio}× match: {int(pts)}/25 pts")

    # ── Factor 3: Deposit type (max 20, always evaluated — 0 if missing) ────
    p_dep = (project.get("deposit_type") or "").strip().lower()
    a_dep = (analog.get("deposit_type") or "").strip().lower()
    possible += 20
    if p_dep and a_dep:
        if p_dep == a_dep or p_dep in a_dep or a_dep in p_dep:
            earned += 20
            reasons.append(f"Deposit type exact match ({a_dep}): 20/20 pts")
        elif _deposit_family(p_dep) == _deposit_family(a_dep):
            earned += 10
            reasons.append(f"Deposit type family match ({_deposit_family(p_dep)}): 10/20 pts")
        else:
            reasons.append(f"Deposit type mismatch ({p_dep} vs {a_dep}): 0/20 pts")
    else:
        reasons.append("Deposit type: unknown, 0/20 pts")

    # ── Factor 4: Mining method (max 10) ────────────────────────────────────
    p_m = (project.get("mining_method") or "").strip().lower()
    a_m = (analog.get("mining_method") or "").strip().lower()
    if p_m and a_m:
        pts = 10.0 if p_m == a_m else 0.0
        earned += pts
        possible += 10
        reasons.append(f"Mining method: {int(pts)}/10 pts")

    # ── Factor 5: Country / region (max 10) ─────────────────────────────────
    p_c = (project.get("country") or "").strip().lower()
    a_c = (analog.get("country") or "").strip().lower()
    if p_c and a_c:
        if p_c == a_c:
            earned += 10; possible += 10
            reasons.append(f"Same country ({a_c}): 10/10 pts")
        else:
            p_cont = _continent(p_c)
            a_cont = _continent(a_c)
            if p_cont and p_cont == a_cont:
                earned += 5; possible += 10
                reasons.append(f"Same region ({p_cont}): 5/10 pts")
            else:
                possible += 10
                reasons.append("Different country/region: 0/10 pts")

    # ── Need at least 2 factors to produce a meaningful score ────────────────
    # Deposit type always counts; check if any other factor was evaluated
    other_factors_scored = p_g > 0 and a_g > 0 or p_t > 0 and a_t > 0 or (p_m and a_m) or (p_c and a_c)
    if not other_factors_scored:
        return None, ["Insufficient data for scoring (< 2 factors available)"]

    score = round(earned, 1)
    return score, reasons


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
    analog_rule = rules_engine.get_analog_rule(material, deposit_type)

    logger.info(
        f"[load] {project.get('name')} | material={material} deposit={deposit_type} "
        f"rule={'✓ ' + analog_rule.get('rule_id', '') if analog_rule else '✗ none'}"
    )
    return {"project": project, "analog_rule": analog_rule, "error": None}


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
    )

    exa_analogs = []
    if text:
        raw = field_extractor.extract_analog_projects(text, material, sources)
        for i, a in enumerate(raw):
            exa_analogs.append({
                "name": a.get("name", f"Unknown project {i}"),
                "material": material,
                "deposit_type": a.get("deposit_type"),
                "tonnage_mt": a.get("tonnage_mt"),
                "grade_value": a.get("grade_value"),
                "grade_unit": a.get("grade_unit"),
                "country": a.get("country"),
                "project_stage": a.get("project_stage"),
                "mining_method": a.get("mining_method"),
                "source": "exa",
                "source_url": a.get("source_url") or (sources[i] if i < len(sources) else None),
            })

    logger.info(f"[exa] Extracted {len(exa_analogs)} candidates from Exa")
    return {"exa_analogs": exa_analogs}


def combine_filter_score_node(state: AnalogState) -> AnalogState:
    """Combine library + Exa candidates, apply hard filters, score deterministically."""
    if state.get("error"):
        return {"scored_analogs": []}

    project = state["project"]
    analog_rule = state.get("analog_rule")
    library = state.get("library_analogs") or []
    exa = state.get("exa_analogs") or []
    all_candidates = library + exa

    project_name = project.get("name") or ""
    target_material = (project.get("material") or "").lower()
    p_tonnage = float(project.get("tonnage_mt") or 0)

    # Grade range from rule for deposit-level validation
    rule_grade_min = float((analog_rule or {}).get("grade_min") or 0)
    rule_grade_max = float((analog_rule or {}).get("grade_max") or 0)
    exclusions = _parse_exclusions((analog_rule or {}).get("analog_criteria") or [])
    logger.info(
        f"[score] {len(library)} library + {len(exa)} exa = {len(all_candidates)} candidates | "
        f"exclusions={exclusions}"
    )

    # ── Step A: Dedup by normalized name ───────────────────────────────────
    seen: dict[str, bool] = {}
    deduped = []
    for c in all_candidates:
        norm = _norm_name(c.get("name") or "")
        if norm and norm not in seen:
            seen[norm] = True
            deduped.append(c)
    logger.info(f"[score] {len(deduped)} after dedup")

    # ── Step B: Hard disqualify ────────────────────────────────────────────
    filtered = []
    for c in deduped:
        name = c.get("name") or ""
        c_material = (c.get("material") or "").lower()
        c_dep = (c.get("deposit_type") or "").lower()
        c_tonnage = float(c.get("tonnage_mt") or 0)
        c_grade = float(c.get("grade_value") or 0)

        # 1. Self-analog
        if project_name and _is_self_analog(project_name, name):
            logger.info(f"[filter] DISQUALIFY (self-analog): {name}")
            continue

        # 2. Material mismatch
        if c_material and target_material and c_material != target_material:
            logger.info(f"[filter] DISQUALIFY (material): {name} — {c_material} ≠ {target_material}")
            continue

        # 3. Tonnage >20x
        if p_tonnage > 0 and c_tonnage > 0:
            ratio = max(p_tonnage, c_tonnage) / min(p_tonnage, c_tonnage)
            if ratio > 20:
                logger.info(f"[filter] DISQUALIFY (tonnage {ratio:.0f}×): {name}")
                continue

        # 4. Deposit type exclusion from rule criteria
        if c_dep and exclusions:
            excluded = any(excl in c_dep for excl in exclusions)
            if excluded:
                logger.info(f"[filter] DISQUALIFY (excluded deposit type '{c_dep}'): {name}")
                continue

        # 5. Grade outside deposit-type range by >3x
        if c_grade > 0 and rule_grade_min > 0 and rule_grade_max > 0:
            if c_grade < rule_grade_min / 3 or c_grade > rule_grade_max * 3:
                logger.info(f"[filter] DISQUALIFY (grade {c_grade} outside rule range {rule_grade_min}-{rule_grade_max} ×3): {name}")
                continue

        filtered.append(c)

    logger.info(f"[score] {len(filtered)} candidates after hard filters")

    # ── Step C: Score ──────────────────────────────────────────────────────
    scored = []
    for c in filtered:
        score, reasons = _score_candidate(project, c)
        scored.append({
            **c,
            "similarity_score": score,
            "similarity_reasons": reasons,
            "approved": False,
        })

    # ── Step D: Select ─────────────────────────────────────────────────────
    # Sort: scored candidates first (by score desc), then None-score candidates
    ranked = sorted(
        scored,
        key=lambda x: (x.get("similarity_score") is None, -(x.get("similarity_score") or 0)),
    )

    MIN_SCORE = 40
    above = [a for a in ranked if a.get("similarity_score") is not None
             and a["similarity_score"] >= MIN_SCORE]
    # Take top 6 above threshold; fallback to top 4 if fewer than 2 pass
    if len(above) >= 2:
        top = above[:6]
    else:
        top = ranked[:4]
        logger.warning(f"[score] Fewer than 2 analogs above MIN_SCORE={MIN_SCORE} — using best available")

    if top:
        best = top[0]
        logger.info(
            f"[score] Best analog: {best.get('name')} "
            f"score={best.get('similarity_score')} source={best.get('source')}"
        )
    else:
        logger.warning("[score] No analog candidates found")

    return {"scored_analogs": top}


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
builder.add_node("library_search", library_search_node)
builder.add_node("exa_search", exa_search_node)
builder.add_node("combine_filter_score", combine_filter_score_node)
builder.add_node("human_review", human_review_analog_node)
builder.add_node("save_analogs", save_analogs_node)

builder.set_entry_point("load_project_and_rule")

# Parallel fan-out: load → library + exa simultaneously
builder.add_edge("load_project_and_rule", "library_search")
builder.add_edge("load_project_and_rule", "exa_search")

# Fan-in: both feed into combine (LangGraph waits for both before running combine)
builder.add_edge("library_search", "combine_filter_score")
builder.add_edge("exa_search", "combine_filter_score")

builder.add_edge("combine_filter_score", "human_review")
builder.add_edge("human_review", "save_analogs")
builder.add_edge("save_analogs", END)

graph = builder.compile()
