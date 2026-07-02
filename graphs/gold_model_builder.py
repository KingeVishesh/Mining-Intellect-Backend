"""
Graph: gold_model_builder

A single-model pipeline that outsources gold-project M&I / Inferred
estimation to Parallel.ai's deep-research agent. The agent is given the
FULL project + analog context (drilling evidence, MRE breakdowns, etc.) so
it doesn't have to re-discover facts the rest of the pipeline already
extracted. It learns drilling -> MRE conversion ratios from the analog
cohort and applies them to the target's drilling profile.

Flow:
  load_project_and_analogs
    → check_analogs_present
        ↳ END if no analogs
    → fetch_drilling_evidence      (reuse model_runner helper)
    → fetch_inferred_evidence      (reuse model_runner helper)
    → call_parallel_gold_model     (the big Parallel.ai call)
    → save_model_run               (persist + overwrite latest projects.* fields)
    → END

Inputs:
  project_id            : str
  use_mre               : bool (default True) — when False, Parallel ignores
                          the project's published MRE and produces a blind
                          pre-MRE estimate (for backtesting).
  find_analogs          : bool (default False) — when True, Parallel discovers
                          and weights its own analog cohort in the same call.
                          Skips the no-analogs error gate. ~2-3× runtime/cost.
  fetch_recent_drill_holes : bool — force-refresh cached drilling evidence.

Persistence:
  - INSERT row into model_runs with model_type = "parallel_pre_mre" or
    "parallel_post_mre". model_output_json carries the raw Parallel response
    so methodology + analog weights + sources are fully auditable.
  - UPDATE projects with the latest M&I / Inferred / total values so the
    /projects-back table reflects them.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from nodes import supabase_ops
from nodes.parallel_gold_model import parallel_gold_model_node
from nodes.project_intelligence import project_intelligence_node, validate_rule_guided_prediction
from graphs.report_generator import load_project_and_analogs_node
from graphs.model_runner import (
    fetch_drilling_evidence_node,
    fetch_inferred_evidence_node,
    _round,
    _contained_native_unit,
)

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class GoldModelBuilderState(TypedDict, total=False):
    # Input
    project_id: str
    # When False, Parallel pretends the official MRE doesn't exist. Default
    # True — incorporate the published MRE when present (post-MRE estimate).
    use_mre: bool
    # When True, Parallel discovers + weights its own analog cohort in the
    # same call. Default False — use the Supabase-stored cohort.
    find_analogs: bool
    # Forward to fetch_drilling_evidence_node so cached evidence can be
    # bypassed when the caller wants a fresh extraction.
    fetch_recent_drill_holes: bool
    # Cache-first rule-guided project intelligence. Defaults true; false keeps
    # the legacy direct Parallel gold prediction path available for rollback.
    use_intelligence_layer: bool
    refresh_project_intelligence: bool

    # Loaded
    project: Optional[Dict]
    analogs: List[Dict]
    project_intelligence: Optional[Dict]
    intelligence_run_id: Optional[str]
    rule_pack_hash: Optional[str]

    # Parallel output
    parallel_model: Optional[Dict]

    # Persistence
    saved: bool
    error: Optional[str]


# ── Helpers ──────────────────────────────────────────────────────────────────

def check_analogs_or_skip_node(state: GoldModelBuilderState) -> GoldModelBuilderState:
    """Block model building when there are no analogs AND we aren't asking
    Parallel to discover its own. When find_analogs=True, an empty cohort is
    fine — the Parallel call will populate it.
    """
    if state.get("error"):
        return {}
    if state.get("find_analogs"):
        return {}
    analogs = state.get("analogs") or []
    if not analogs:
        return {"error": (
            "Project has no analogs and find_analogs=False. Either run "
            "analog_finder first or pass find_analogs=True so Parallel "
            "discovers its own cohort."
        )}
    return {}


def _route_after_check(state: GoldModelBuilderState) -> str:
    return END if state.get("error") else "fetch_drilling_evidence"


def _route_after_parallel(state: GoldModelBuilderState) -> str:
    return END if state.get("error") else "validate_rule_guided_model"


def _route_after_validation(state: GoldModelBuilderState) -> str:
    return END if state.get("error") else "save_model_run"


_GOLD_MATERIAL = "gold"
_URL_RE = re.compile(r"https?://[^\s),\]]+")


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _range_value(block: Dict[str, Any], range_key: str, point_key: str) -> Dict[str, Optional[float]]:
    raw = block.get(range_key) if isinstance(block.get(range_key), dict) else {}
    p50 = _as_float(raw.get("p50")) or _as_float(block.get(point_key))
    p10 = _as_float(raw.get("p10"))
    p90 = _as_float(raw.get("p90"))
    if p50 is None:
        return {"p10": None, "p50": None, "p90": None}
    if p10 is None or p10 > p50:
        p10 = p50
    if p90 is None or p90 < p50:
        p90 = p50
    return {"p10": p10, "p50": p50, "p90": p90}


def _contained_range_value(block: Dict[str, Any]) -> Dict[str, Optional[float]]:
    values = _range_value(block, "contained_range_moz", "contained_moz")
    if values["p50"] is not None:
        return values
    tonnage = _range_value(block, "tonnage_range_mt", "tonnage_mt")
    grade = _range_value(block, "grade_range_gpt", "grade_gpt")
    derived: Dict[str, Optional[float]] = {}
    for key in ("p10", "p50", "p90"):
        t = tonnage.get(key)
        g = grade.get(key)
        derived[key] = (t * g * 0.032151) if t is not None and g is not None else None
    return derived


def _weighted_grade_range(
    tonnage_a: Dict[str, Optional[float]],
    grade_a: Dict[str, Optional[float]],
    tonnage_b: Dict[str, Optional[float]],
    grade_b: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in ("p10", "p50", "p90"):
        ta = tonnage_a.get(key) or 0.0
        tb = tonnage_b.get(key) or 0.0
        ga = grade_a.get(key)
        gb = grade_b.get(key)
        total_t = ta + tb
        if total_t <= 0 or (ga is None and gb is None):
            out[key] = None
        else:
            out[key] = (((ga or 0.0) * ta) + ((gb or 0.0) * tb)) / total_t
    return out


def _metric_range_rows(category: str, block: Dict[str, Any]) -> List[Dict[str, Any]]:
    metric_specs = (
        ("tonnage_mt", "tonnage_range_mt", "tonnage_mt", "Mt"),
        ("grade_gpt", "grade_range_gpt", "grade_gpt", "g/t"),
        ("contained_moz", "contained_range_moz", "contained_moz", "Moz"),
    )
    rows: List[Dict[str, Any]] = []
    for metric, range_key, point_key, unit in metric_specs:
        values = (
            _contained_range_value(block)
            if metric == "contained_moz"
            else _range_value(block, range_key, point_key)
        )
        if values["p50"] is None:
            continue
        rows.append({
            "resource_category": category,
            "metric": metric,
            "p10": _round(values["p10"], 6),
            "p50": _round(values["p50"], 6),
            "p90": _round(values["p90"], 6),
            "unit": unit,
            "payload": {"source": "parallel_gold_model", "range_key": range_key},
        })
    return rows


def validate_rule_guided_model_node(state: GoldModelBuilderState) -> GoldModelBuilderState:
    """Fail closed when the rule-guided prediction violates the contract."""
    if state.get("error"):
        return {}
    intelligence = state.get("project_intelligence") or {}
    if not intelligence:
        return {}
    parallel_out = state.get("parallel_model") or {}
    errors = validate_rule_guided_prediction(
        parallel_out,
        intelligence=intelligence,
        use_mre=bool(state.get("use_mre", True)),
    )
    if errors:
        return {"error": "Rule-guided prediction validation failed: " + "; ".join(errors)}
    return {}


def _resource_range_rows_from_parallel(parallel_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    total_ranges = _derived_total_ranges_from_parallel(parallel_out)
    rows = _metric_range_rows("m_and_i", parallel_out.get("m_and_i") or {})
    rows.extend(_metric_range_rows("inferred", parallel_out.get("inferred") or {}))

    for metric, key, unit in (
        ("tonnage_mt", "tonnage_range_mt", "Mt"),
        ("grade_gpt", "grade_range_gpt", "g/t"),
        ("contained_moz", "contained_range_moz", "Moz"),
    ):
        values = total_ranges.get(key) or {}
        if values.get("p50") is None:
            continue
        rows.append({
            "resource_category": "total",
            "metric": metric,
            "p10": _round(values.get("p10"), 6),
            "p50": _round(values.get("p50"), 6),
            "p90": _round(values.get("p90"), 6),
            "unit": unit,
            "payload": {"source": "parallel_gold_model", "derived": True},
        })
    return rows


def _derived_total_ranges_from_parallel(parallel_out: Dict[str, Any]) -> Dict[str, Any]:
    mi_block = parallel_out.get("m_and_i") or {}
    inf_block = parallel_out.get("inferred") or {}
    mi_t = _range_value(mi_block, "tonnage_range_mt", "tonnage_mt")
    inf_t = _range_value(inf_block, "tonnage_range_mt", "tonnage_mt")
    mi_g = _range_value(mi_block, "grade_range_gpt", "grade_gpt")
    inf_g = _range_value(inf_block, "grade_range_gpt", "grade_gpt")
    mi_c = _contained_range_value(mi_block)
    inf_c = _contained_range_value(inf_block)
    total_t = {key: (mi_t.get(key) or 0.0) + (inf_t.get(key) or 0.0) for key in ("p10", "p50", "p90")}
    total_c = {key: (mi_c.get(key) or 0.0) + (inf_c.get(key) or 0.0) for key in ("p10", "p50", "p90")}
    total_g = _weighted_grade_range(mi_t, mi_g, inf_t, inf_g)
    return {
        "tonnage_range_mt": {key: _round(total_t.get(key), 6) for key in ("p10", "p50", "p90")},
        "grade_range_gpt": {key: _round(total_g.get(key), 6) for key in ("p10", "p50", "p90")},
        "contained_range_moz": {key: _round(total_c.get(key), 6) for key in ("p10", "p50", "p90")},
    }


def _source_url_from_text(text: str) -> Optional[str]:
    match = _URL_RE.search(text or "")
    return match.group(0).rstrip(".") if match else None


def _source_rows_from_parallel(project: Dict[str, Any], analogs: List[Dict], parallel_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(row: Dict[str, Any]) -> None:
        title = row.get("title") or row.get("source_title") or row.get("summary") or row.get("url")
        summary = row.get("summary") or row.get("notes") or row.get("reason") or title
        if not title and not summary:
            return
        used_for = row.get("used_for") if isinstance(row.get("used_for"), list) else []
        rows.append({
            "role": row.get("role") or "evidence",
            "used_for": [str(item) for item in used_for],
            "title": str(title or "")[:500],
            "url": row.get("url") or row.get("source_url") or _source_url_from_text(str(summary or "")),
            "publisher": row.get("publisher") or row.get("source_name"),
            "source_date": row.get("source_date"),
            "summary": str(summary or "")[:2000],
            "excerpt": row.get("excerpt"),
            "confidence": row.get("confidence"),
            "payload": row,
        })

    for item in parallel_out.get("sources_used") or []:
        if isinstance(item, dict):
            add(item)
        elif isinstance(item, str):
            add({"role": "parallel_source", "title": item, "summary": item})

    drilling = project.get("drilling_evidence")
    if isinstance(drilling, dict) and drilling.get("source_url") and not drilling.get("redacted"):
        add({
            "role": "target_pre_mre_evidence",
            "used_for": ["tonnage_range", "grade_range", "contained_range"],
            "title": drilling.get("source_title") or "Target pre-MRE drilling evidence",
            "url": drilling.get("source_url"),
            "source_date": drilling.get("source_date"),
            "summary": drilling.get("notes") or "Target drilling/geometry evidence supplied to the blind gold model.",
            "confidence": drilling.get("confidence"),
        })

    for text in parallel_out.get("analogs_used") or []:
        if isinstance(text, str):
            add({
                "role": "analog_weighting",
                "used_for": ["analog_selection", "range_calibration"],
                "title": text.split("|", 1)[0].strip() or "Analog used",
                "url": _source_url_from_text(text),
                "summary": text,
            })

    for analog in analogs or []:
        source_url = analog.get("source_url") or analog.get("mre_source_url")
        if source_url:
            add({
                "role": "analog_resource_source",
                "used_for": ["analog_selection", "range_calibration"],
                "title": analog.get("name") or analog.get("analog_name") or source_url,
                "url": source_url,
                "source_date": analog.get("mre_date"),
                "summary": "Approved analog source available to the gold model.",
                "payload": {
                    "name": analog.get("name") or analog.get("analog_name"),
                    "tonnage_mt": analog.get("tonnage_mt"),
                    "grade_value": analog.get("grade_value"),
                },
            })

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = (row.get("url") or "", row.get("title") or "", row.get("summary") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _fields_from_parallel(project: Dict, parallel_out: Dict) -> Dict[str, Any]:
    """Translate Parallel's JSON into the column shape `save_model_run`
    expects. Mirrors model_runner._fields_from_model.
    """
    material = project.get("material") or _GOLD_MATERIAL

    mi_block = parallel_out.get("m_and_i") or {}
    inf_block = parallel_out.get("inferred") or {}

    mi_mt = _as_float(mi_block.get("tonnage_mt"))
    mi_g = _as_float(mi_block.get("grade_gpt"))
    inf_mt = _as_float(inf_block.get("tonnage_mt"))
    inf_g = _as_float(inf_block.get("grade_gpt"))

    mi_kt = float(mi_mt) * 1000.0 if mi_mt is not None else 0.0
    inf_kt = float(inf_mt) * 1000.0 if inf_mt is not None else 0.0

    # contained_moz reported by Parallel is in Moz; we persist contained in
    # the material's native unit (troy oz for gold). Convert Moz -> oz when
    # present; otherwise derive from tonnage × grade for consistency.
    mi_contained_moz = _as_float(mi_block.get("contained_moz"))
    inf_contained_moz = _as_float(inf_block.get("contained_moz"))
    mi_contained_oz = (
        mi_contained_moz * 1_000_000.0
        if mi_contained_moz is not None
        else _contained_native_unit(mi_kt, mi_g, material)
    )
    inf_contained_oz = (
        inf_contained_moz * 1_000_000.0
        if inf_contained_moz is not None
        else _contained_native_unit(inf_kt, inf_g, material)
    )

    total_kt = mi_kt + inf_kt
    total_mt = _round(total_kt / 1000.0) if total_kt else None
    total_contained = None
    if mi_contained_oz is not None or inf_contained_oz is not None:
        total_contained = _round((mi_contained_oz or 0.0) + (inf_contained_oz or 0.0), 3)

    mi_g_f = float(mi_g) if mi_g is not None else 0.0
    inf_g_f = float(inf_g) if inf_g is not None else 0.0
    avg_grade = (
        _round((mi_g_f * mi_kt + inf_g_f * inf_kt) / total_kt)
        if total_kt > 0 and (mi_g is not None or inf_g is not None)
        else None
    )

    mi_t_range = _range_value(mi_block, "tonnage_range_mt", "tonnage_mt")
    inf_t_range = _range_value(inf_block, "tonnage_range_mt", "tonnage_mt")
    mi_g_range = _range_value(mi_block, "grade_range_gpt", "grade_gpt")
    inf_g_range = _range_value(inf_block, "grade_range_gpt", "grade_gpt")
    mi_c_range = _range_value(mi_block, "contained_range_moz", "contained_moz")
    inf_c_range = _range_value(inf_block, "contained_range_moz", "contained_moz")
    total_t_range = {
        key: (mi_t_range.get(key) or 0.0) + (inf_t_range.get(key) or 0.0)
        for key in ("p10", "p50", "p90")
    }
    total_g_range = _weighted_grade_range(mi_t_range, mi_g_range, inf_t_range, inf_g_range)
    total_c_range_oz = {
        key: ((mi_c_range.get(key) or 0.0) + (inf_c_range.get(key) or 0.0)) * 1_000_000.0
        for key in ("p10", "p50", "p90")
    }

    conviction = (parallel_out.get("conviction") or {})
    conviction_level = conviction.get("level") or ""
    tier_code = f"PARALLEL-{conviction_level.upper()}" if conviction_level else "PARALLEL-UNKNOWN"
    tier_label = conviction.get("rationale") or "Parallel.ai deep-research estimate"

    return {
        # M&I
        "mi_tonnage_mt":       _round(mi_mt),
        "mi_grade":            _round(mi_g),
        "mi_contained":        _round(mi_contained_oz, 3),
        # Inferred
        "inferred_resource_mt": _round(inf_mt),
        "inferred_grade":       _round(inf_g),
        "inferred_contained":   _round(inf_contained_oz, 3),
        # Totals (derived for arithmetic consistency)
        "tonnage_mt":          total_mt,
        "grade_value":         avg_grade,
        "total_contained":     total_contained,
        # Conviction
        "conviction_score":    tier_code,
        "conviction_tier":     f"{tier_code}: {tier_label}",
        # Percentile / CV columns: for gold ranges, P50 mirrors the scalar
        # latest-project fields while P10/P90 carry the blind uncertainty band.
        "p10_tonnage_mt": _round(total_t_range.get("p10"), 3),
        "p50_tonnage_mt": _round(total_t_range.get("p50"), 3),
        "p90_tonnage_mt": _round(total_t_range.get("p90"), 3),
        "p10_grade": _round(total_g_range.get("p10"), 4),
        "p50_grade": _round(total_g_range.get("p50"), 4),
        "p90_grade": _round(total_g_range.get("p90"), 4),
        "p10_contained": _round(total_c_range_oz.get("p10"), 3),
        "p50_contained": _round(total_c_range_oz.get("p50"), 3),
        "p90_contained": _round(total_c_range_oz.get("p90"), 3),
        "cv_contained": None,
        # Audit trail — the full Parallel response is also saved into
        # model_output_json below; this column gets the analogs-used trace
        # so downstream calibration can attribute residual error per analog.
        "signal_contributions_json": {
            "source": "parallel.ai",
            "anchor_used": parallel_out.get("anchor_used"),
            "methodology": parallel_out.get("methodology"),
            "analogs_used": parallel_out.get("analogs_used"),
            "analogs_rejected": parallel_out.get("analogs_rejected"),
            "sources_used": parallel_out.get("sources_used"),
            "sources_rejected": parallel_out.get("sources_rejected"),
        },
    }


# ── Nodes ────────────────────────────────────────────────────────────────────

def save_model_run_node(
    state: GoldModelBuilderState,
    config: Optional[Dict] = None,
) -> GoldModelBuilderState:
    """Persist the Parallel-produced model to model_runs + projects."""
    if state.get("error"):
        return {}

    project = state.get("project") or {}
    project_id = state["project_id"]
    parallel_out = state.get("parallel_model")
    if not parallel_out:
        return {"saved": False, "error": "No Parallel output to persist"}

    use_mre = bool(state.get("use_mre", True))
    model_type = "parallel_post_mre" if use_mre else "parallel_pre_mre"

    cfg = (config or {}).get("configurable") or {}
    thread_id = cfg.get("thread_id")
    run_id = cfg.get("run_id")

    parallel_out = {
        **parallel_out,
        "total": {
            **_derived_total_ranges_from_parallel(parallel_out),
            "derived": True,
        },
    }
    fields = _fields_from_parallel(project, parallel_out)
    intelligence = state.get("project_intelligence") or {}
    if intelligence:
        signals = fields.get("signal_contributions_json")
        if not isinstance(signals, dict):
            signals = {"legacy_signal_contributions": signals} if signals else {}
        signals["project_intelligence"] = {
            "id": state.get("intelligence_run_id") or intelligence.get("id"),
            "rule_pack_hash": state.get("rule_pack_hash") or intelligence.get("rule_pack_hash"),
            "mode": intelligence.get("mode"),
            "commodity": intelligence.get("commodity"),
            "archetype": (intelligence.get("rule_pack") or {}).get("archetype"),
        }
        fields["signal_contributions_json"] = signals
        parallel_out = {
            **parallel_out,
            "intelligence_run_id": state.get("intelligence_run_id") or intelligence.get("id"),
            "rule_pack_hash": state.get("rule_pack_hash") or intelligence.get("rule_pack_hash"),
            "project_intelligence_summary": {
                "mode": intelligence.get("mode"),
                "commodity": intelligence.get("commodity"),
                "archetype": (intelligence.get("rule_pack") or {}).get("archetype"),
                "evidence_gaps": intelligence.get("evidence_gaps") or [],
            },
        }
    model_run_id = supabase_ops.save_model_run(
        project_id=project_id,
        model_type=model_type,
        fields=fields,
        model_output_json=parallel_out,
        thread_id=thread_id,
        run_id=run_id,
    )
    if model_run_id:
        supabase_ops.save_model_run_resource_ranges(
            model_run_id=model_run_id,
            project_id=project_id,
            ranges=_resource_range_rows_from_parallel(parallel_out),
        )
        supabase_ops.save_model_run_sources(
            model_run_id=model_run_id,
            project_id=project_id,
            sources=_source_rows_from_parallel(
                project=project,
                analogs=state.get("analogs") or [],
                parallel_out=parallel_out,
            ),
        )
    supabase_ops.update_project_latest_model(project_id, fields)
    return {"saved": True, "error": None}


# ── Graph wiring ─────────────────────────────────────────────────────────────

builder = StateGraph(GoldModelBuilderState)
builder.add_node("load_project_and_analogs", load_project_and_analogs_node)
builder.add_node("check_analogs_present", check_analogs_or_skip_node)
builder.add_node("fetch_drilling_evidence", fetch_drilling_evidence_node)
builder.add_node("fetch_inferred_evidence", fetch_inferred_evidence_node)
builder.add_node("build_project_intelligence", project_intelligence_node)
builder.add_node("call_parallel_gold_model", parallel_gold_model_node)
builder.add_node("validate_rule_guided_model", validate_rule_guided_model_node)
builder.add_node("save_model_run", save_model_run_node)

builder.set_entry_point("load_project_and_analogs")
builder.add_edge("load_project_and_analogs", "check_analogs_present")
builder.add_conditional_edges("check_analogs_present", _route_after_check, {
    "fetch_drilling_evidence": "fetch_drilling_evidence",
    END: END,
})
builder.add_edge("fetch_drilling_evidence", "fetch_inferred_evidence")
builder.add_edge("fetch_inferred_evidence", "build_project_intelligence")
builder.add_edge("build_project_intelligence", "call_parallel_gold_model")
builder.add_conditional_edges("call_parallel_gold_model", _route_after_parallel, {
    "validate_rule_guided_model": "validate_rule_guided_model",
    END: END,
})
builder.add_conditional_edges("validate_rule_guided_model", _route_after_validation, {
    "save_model_run": "save_model_run",
    END: END,
})
builder.add_edge("save_model_run", END)

graph = builder.compile()
