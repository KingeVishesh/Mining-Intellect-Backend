"""
Graph: model_runner

Standalone pipeline that builds Model 1 (+ Model 2 when an official MRE
exists) WITHOUT generating a PDF or saving a reports row. Used by the
/projects-back "Build Models" button so the user can iterate on model
output without paying the cost of report generation.

Flow:
  load_project_and_analogs
    → check_analogs_present
        ↳ END if no analogs
    → load_rules → activate_rules
    → build_model_1 → build_model_2 → save_model_run → END

Persistence:
  - INSERT row into model_runs for each Model produced (history).
  - UPDATE projects with the latest Model values so the table view reflects them.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from nodes import supabase_ops, model_builder, drilling_extractor
from nodes import ni_43_101_extractor, rules_engine
from graphs.report_generator import (
    load_project_and_analogs_node,
    load_rules_node,
    activate_rules_node,
)


def build_model_node(state: "ModelRunnerState") -> "ModelRunnerState":
    """Run the single Model with `use_mre` driven by the LangGraph input.

    Defaults to True (production behaviour: incorporate the published MRE
    when available, producing a POST-tier estimate). Set False for
    backtesting and pre-MRE prediction (PRE-tier estimate from analogs +
    drilling + L151 prior, with the MRE deliberately hidden from the
    model so it can be compared against ground truth).
    """
    if state.get("error"):
        return {}
    project = state["project"]
    analogs = state.get("analogs", [])
    activated_rules = state.get("activated_rules", [])
    use_mre = state.get("use_mre", True)

    base_tonnage = float(project.get("tonnage_mt") or 0) * 1000  # Mt -> kt
    base_grade = float(project.get("grade_value") or 0)
    rule_effects = rules_engine.apply_rule_multipliers(
        base_tonnage=base_tonnage or 1000,
        base_grade=base_grade or 1.0,
        activated_rules=activated_rules,
    )

    model = model_builder.build_model_1(analogs, project, rule_effects, use_mre=use_mre)
    logger.info(
        f"[model] built ({'post-MRE' if model.get('mre_signal_applied') else 'pre-MRE'}); "
        f"use_mre={use_mre}, tier={model.get('conviction_tier')}"
    )
    return {"model_1": model, "rule_effects": rule_effects}

logger = logging.getLogger(__name__)


class ModelRunnerState(TypedDict, total=False):
    # Input
    project_id: str
    # Force-refresh drilling data even if the cached copy is < 7 days old.
    # When omitted or false, fetch_drilling_evidence_node uses the cache
    # unless it's stale or missing.
    fetch_recent_drill_holes: bool

    # Loaded
    project: Optional[Dict]
    analogs: List[Dict]
    all_rules: List[Dict]
    activated_rules: List[Dict]
    rule_effects: Dict

    # Models
    model_1: Optional[Dict]
    model_2: Optional[Dict]
    official_mre_row: Optional[Dict]

    # Output
    saved: bool
    error: Optional[str]


def check_analogs_present_node(state: ModelRunnerState) -> ModelRunnerState:
    """Block model building when there are no analogs to score against."""
    if state.get("error"):
        return {}
    analogs = state.get("analogs") or []
    if not analogs:
        msg = "Cannot run models: project has no analogs. Run analog finder first."
        logger.warning(f"[model_runner] {msg}")
        return {"error": msg}
    return {}


def fetch_drilling_evidence_node(state: ModelRunnerState) -> ModelRunnerState:
    """Pull or refresh drilling data for the project and its top analogs.

    Strategy:
      * Project: refetch if `fetch_recent_drill_holes=True` was passed as a
        LangGraph input, OR if the cached drilling_evidence is missing /
        older than 7 days. Save back to projects.drilling_evidence.
      * Analogs: lazily extract once per analog and cache on the analogs
        table. We only fetch the top-N analogs by similarity_score so a
        model run doesn't kick off a dozen Exa requests.
      * Errors are tolerated — Model 1 still runs with whatever drilling
        evidence is available (or none).
    """
    if state.get("error"):
        return {}

    project = state.get("project") or {}
    analogs = state.get("analogs") or []
    project_id = state["project_id"]
    force = bool(state.get("fetch_recent_drill_holes"))

    # ── Project drilling evidence ────────────────────────────────────────
    cached_evidence = project.get("drilling_evidence")
    cached_at = project.get("drilling_evidence_fetched_at")
    if drilling_extractor.should_refetch(cached_evidence, cached_at, force=force):
        logger.info(
            f"[fetch_drilling] Project {project_id} drilling-data refetch "
            f"(force={force}, cached_at={cached_at})"
        )
        # Try the NI 43-101 technical-report extractor first — it returns
        # cumulative project-history drilling totals rather than the press-
        # release headlines that the Answer-API extractor surfaces. Fall
        # back to the cheaper press-release path if no technical report
        # text can be retrieved.
        evidence = ni_43_101_extractor.extract_from_ni_43_101(
            project_name=project.get("name") or "",
            material=project.get("material") or "",
            country=project.get("country"),
            region=project.get("region"),
            deposit_type=project.get("deposit_type"),
            pre_mre=True,
        )
        if not evidence:
            evidence = drilling_extractor.extract_drilling_evidence(
                project_name=project.get("name") or "",
                material=project.get("material") or "",
                country=project.get("country"),
                region=project.get("region"),
                deposit_type=project.get("deposit_type"),
            )
        if evidence:
            try:
                supabase_ops.save_project_drilling_evidence(project_id, evidence)
            except Exception as e:
                logger.warning(f"[fetch_drilling] DB save failed: {e}")
            project = {**project, "drilling_evidence": evidence}

    # ── Analog drilling evidence (lazy, top-N by similarity) ─────────────
    # Only extract for analogs with valid tonnage+grade; without those the
    # drilling-density ratio (tonnage / total_meters) can't be computed.
    enriched_analogs: List[Dict] = []
    candidates = [
        a for a in analogs
        if a.get("tonnage_mt") and a.get("grade_value")
    ]
    candidates.sort(
        key=lambda a: float(a.get("similarity_score") or 0),
        reverse=True,
    )
    TOP_N = 5  # cap Exa hits per model run
    for i, a in enumerate(analogs):
        if a not in candidates[:TOP_N]:
            enriched_analogs.append(a)
            continue
        # Check whether this analog already has drilling_evidence inline
        if a.get("drilling_evidence"):
            enriched_analogs.append(a)
            continue
        try:
            cached, cached_when = supabase_ops.get_analog_drilling_evidence(
                a.get("name", ""), a.get("material", project.get("material", "")),
            )
        except Exception:
            cached, cached_when = None, None
        if cached and not drilling_extractor.should_refetch(cached, cached_when, force=force):
            enriched_analogs.append({**a, "drilling_evidence": cached})
            continue
        # Fetch fresh for this analog — NI 43-101 first, press-release fallback
        ev = ni_43_101_extractor.extract_from_ni_43_101(
            project_name=a.get("name") or "",
            material=a.get("material") or project.get("material") or "",
            country=a.get("country"),
            region=a.get("region"),
            deposit_type=a.get("deposit_type"),
            pre_mre=False,  # analogs use their as-published totals, not pre-MRE
        )
        if not ev:
            ev = drilling_extractor.extract_drilling_evidence(
                project_name=a.get("name") or "",
                material=a.get("material") or project.get("material") or "",
                country=a.get("country"),
                region=a.get("region"),
                deposit_type=a.get("deposit_type"),
            )
        if ev:
            try:
                supabase_ops.save_analog_drilling_evidence(
                    a.get("name", ""), a.get("material", project.get("material", "")), ev,
                )
            except Exception as e:
                logger.warning(f"[fetch_drilling] analog DB save failed: {e}")
            enriched_analogs.append({**a, "drilling_evidence": ev})
        else:
            enriched_analogs.append(a)

    return {"project": project, "analogs": enriched_analogs}


def _route_after_check(state: ModelRunnerState) -> str:
    return END if state.get("error") else "fetch_drilling_evidence"


def _round(x: Optional[float], digits: int = 4) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), digits)
    except (TypeError, ValueError):
        return None


_PRECIOUS_METALS = {"gold", "silver", "platinum", "palladium"}

# Troy ounces per tonne — same constant as model_builder._TROY_OZ_PER_TONNE.
# Duplicated locally to avoid a private-symbol import; the value is fixed by
# definition (1 troy oz = 31.1034768 g exactly).
_TROY_OZ_PER_TONNE = 32150.7466


def _contained_native_unit(
    tonnage_kt: float,
    grade: Optional[float],
    material: str,
) -> Optional[float]:
    """Contained metal in the material's industry-reporting unit:

    Precious metals (Au/Ag/Pt/Pd): troy ounces. Grade is g/t, tonnage is
    kt. 1 kt × 1 g/t = 1 kg metal = 1 / 31.1034768 / 1000 koz ⇒
    contained_oz = tonnage_kt × grade_g_t × _TROY_OZ_PER_TONNE / 1000.
    For 9.9 Mt × 2.40 g/t Au this returns ~763,861 oz, matching how
    Cartier publishes the Cadillac MRE.

    Base metals: tonnes of metal. Grade is %, tonnage is kt. 1 kt × 1 %
    = 10 t metal ⇒ contained_t = tonnage_kt × grade_pct × 10.
    """
    if not tonnage_kt or grade is None:
        return None
    mat = model_builder._norm_material(material or "")
    if mat in _PRECIOUS_METALS:
        return tonnage_kt * float(grade) * _TROY_OZ_PER_TONNE / 1000.0
    return tonnage_kt * float(grade) * 10.0


# Back-compat alias — older code paths in this file call _contained_tonnes.
# Both names route to the same function; the body itself decides the unit
# from material (oz for precious, tonnes for base).
_contained_tonnes = _contained_native_unit


def _fields_from_model(project: Dict, model: Dict, is_post_mre: bool) -> Dict:
    """Translate a Model 1 / Model 2 output dict into the 9 columns we persist.

    Industry resource statements bundle Measured + Indicated together as
    "M&I" — the model itself doesn't split them, and re-splitting them
    via a stage heuristic added noise without information. We now store
    M&I as a single triple (tonnage, grade, contained) alongside the
    Inferred triple and the totals.

    Conviction is the tier code ("PRE-1".."PRE-5" / "POST-1".."POST-5");
    conviction_tier carries the full human label.
    """
    material = project.get("material") or ""

    # When the project carries a published MRE breakdown on the projects.
    # mre_* columns, prefer those values directly over the heuristic split
    # the model computed. The model's per-bucket split is a best-effort
    # estimate from analog/lesson patterns; if the operator has published
    # the actual Measured+Indicated and Inferred breakdown, use the real
    # numbers. Cadillac's published MRE is M&I 9.9 Mt @ 2.40 + Inferred
    # 35 Mt @ 2.14 — the heuristic 75/25 split of model.total_tonnage
    # would only give M&I 7.5 Mt and Inferred 2.5 Mt, missing the huge
    # Inferred halo that Cartier reports.
    mre_mi_t  = project.get("mre_mi_tonnage_mt")
    mre_mi_g  = project.get("mre_mi_grade")
    mre_inf_t = project.get("mre_inferred_tonnage_mt")
    mre_inf_g = project.get("mre_inferred_grade")
    breakdown_overridden = (
        is_post_mre
        and mre_mi_t is not None and mre_mi_g is not None
        and mre_inf_t is not None and mre_inf_g is not None
    )
    if breakdown_overridden:
        mi_kt   = float(mre_mi_t)  * 1000.0
        mi_g    = float(mre_mi_g)
        inf_kt  = float(mre_inf_t) * 1000.0
        inf_g   = float(mre_inf_g)
    else:
        mi_kt   = float(model.get("mi_tonnage_kt") or 0)
        mi_g    = model.get("mi_grade_pct")
        inf_kt  = float(model.get("inferred_tonnage_kt") or 0)
        inf_g   = model.get("inferred_grade_pct")

    mi_mt       = _round(mi_kt / 1000.0) if mi_kt else None
    inferred_mt = _round(inf_kt / 1000.0) if inf_kt else None

    # Contained metal in the material's industry-reporting unit. For
    # precious metals (Au/Ag/Pt/Pd) the result is in troy ounces, matching
    # how NI 43-101 / JORC / SK-1300 disclosures publish the figure (e.g.
    # Cadillac M&I = 9.9 Mt × 2.40 g/t ≈ 763,861 oz). For base metals the
    # result is in tonnes of metal.
    mi_contained_native  = _contained_native_unit(mi_kt, mi_g, material)
    inf_contained_native = _contained_native_unit(inf_kt, inf_g, material)

    # Totals are recomputed from the per-category values so the table is
    # internally consistent: total tonnage = M&I + Inferred, total contained
    # = M&I + Inferred, and avg grade is tonnage-weighted so it equals
    # total_contained / total_tonnage in the right units.
    total_kt = mi_kt + inf_kt
    total_mt = _round(total_kt / 1000.0) if total_kt else None

    total_contained_native = None
    if mi_contained_native is not None or inf_contained_native is not None:
        total_contained_native = _round(
            (mi_contained_native or 0.0) + (inf_contained_native or 0.0), 3
        )

    mi_g_f  = float(mi_g) if mi_g is not None else 0.0
    inf_g_f = float(inf_g) if inf_g is not None else 0.0
    avg_grade = (
        _round((mi_g_f * mi_kt + inf_g_f * inf_kt) / total_kt)
        if total_kt > 0 and (mi_g is not None or inf_g is not None) else None
    )

    conviction_num = float(model.get("conviction_pct") or 0)
    if is_post_mre:
        tier_code, tier_label = model_builder._compute_post_tier(conviction_num, project)
    else:
        tier_code, tier_label = model_builder._compute_pre_tier(conviction_num)

    # ── P1: posterior percentiles + CV come through from build_model_1 ─────────
    # Model 1 v2 attaches a joint log-normal posterior. Model 2 doesn't have one
    # yet (P5 follow-up) — when the keys are absent we pass None, which is what
    # the nullable model_runs columns expect.
    return {
        # Measured + Indicated combined (M&I)
        "mi_tonnage_mt":         mi_mt,
        "mi_grade":              _round(mi_g),
        "mi_contained":          _round(mi_contained_native, 3),
        # Inferred
        "inferred_resource_mt":  inferred_mt,
        "inferred_grade":        _round(inf_g),
        "inferred_contained":    _round(inf_contained_native, 3),
        # Totals — derived as sums / weighted average for arithmetic consistency
        "tonnage_mt":            total_mt,
        "grade_value":           avg_grade,
        "total_contained":       total_contained_native,
        # Conviction
        "conviction_score":      tier_code,
        "conviction_tier":       f"{tier_code}: {tier_label}",
        # Posterior percentiles (Model 1 only in P1 — Model 2 leaves these null)
        "p10_tonnage_mt":        _round(model.get("p10_total_tonnage_mt"), 3),
        "p50_tonnage_mt":        _round(model.get("p50_total_tonnage_mt"), 3),
        "p90_tonnage_mt":        _round(model.get("p90_total_tonnage_mt"), 3),
        "p10_grade":             _round(model.get("p10_grade"), 4),
        "p50_grade":             _round(model.get("p50_grade"), 4),
        "p90_grade":             _round(model.get("p90_grade"), 4),
        "p10_contained":         _round(model.get("p10_contained_t"), 3),
        "p50_contained":         _round(model.get("p50_contained_t"), 3),
        "p90_contained":         _round(model.get("p90_contained_t"), 3),
        "cv_contained":          _round(model.get("cv_contained"), 4),
        "signal_contributions_json": model.get("signal_contributions"),
    }


def save_model_run_node(
    state: ModelRunnerState,
    config: Optional[Dict] = None,
) -> ModelRunnerState:
    """Persist the single Model run to model_runs and overwrite the latest
    projects.* fields. There's only one model now — pre-MRE vs post-MRE
    is a flag on the run (use_mre), not a separate model row.
    """
    if state.get("error"):
        return {}

    project = state.get("project") or {}
    project_id = state["project_id"]
    cfg = (config or {}).get("configurable") or {}
    thread_id = cfg.get("thread_id")
    run_id = cfg.get("run_id")

    model = state.get("model_1") or state.get("model_2")
    if not model:
        return {"saved": False, "error": "No model output produced"}

    mre_used = bool(model.get("mre_signal_applied"))
    fields = _fields_from_model(project, model, is_post_mre=mre_used)
    supabase_ops.save_model_run(
        project_id=project_id,
        model_type=("post_mre" if mre_used else "pre_mre"),
        fields=fields,
        model_output_json=model,
        thread_id=thread_id,
        run_id=run_id,
    )
    supabase_ops.update_project_latest_model(project_id, fields)
    return {"saved": True, "error": None}


# ── Graph ──────────────────────────────────────────────────────────────────────

builder = StateGraph(ModelRunnerState)
builder.add_node("load_project_and_analogs", load_project_and_analogs_node)
builder.add_node("check_analogs_present", check_analogs_present_node)
builder.add_node("fetch_drilling_evidence", fetch_drilling_evidence_node)
builder.add_node("load_rules", load_rules_node)
builder.add_node("activate_rules", activate_rules_node)
builder.add_node("build_model", build_model_node)
builder.add_node("save_model_run", save_model_run_node)

builder.set_entry_point("load_project_and_analogs")
builder.add_edge("load_project_and_analogs", "check_analogs_present")
builder.add_conditional_edges("check_analogs_present", _route_after_check, {
    "fetch_drilling_evidence": "fetch_drilling_evidence",
    END: END,
})
builder.add_edge("fetch_drilling_evidence", "load_rules")
builder.add_edge("load_rules", "activate_rules")
builder.add_edge("activate_rules", "build_model")
builder.add_edge("build_model", "save_model_run")
builder.add_edge("save_model_run", END)

graph = builder.compile()
