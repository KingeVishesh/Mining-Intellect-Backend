"""
Project intelligence layer for rule-guided model builds.

This is the first step of the cache-first two-step flow:
  project + analogs + evidence -> project dossier + rule pack

Gold consumes the resulting rule pack in a second Parallel prediction call.
Other commodities persist the same intelligence scaffold first, then keep the
existing deterministic model path until a commodity-specific predictor exists.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from nodes import supabase_ops
from nodes.parallel_gold_model import (
    _blind_result_mentions_mre_anchor,
    _chronology_directive,
    _format_analogs_block,
    _format_project_block,
    _run_parallel_task,
    _target_mre_cutoff,
)

logger = logging.getLogger(__name__)

_CACHE_TTL_DAYS = 14
_MODES = {"blind_pre_mre", "post_mre"}

_PROJECT_FINGERPRINT_FIELDS = (
    "id", "name", "company_name", "material", "commodity", "primary_commodity",
    "secondary_commodity", "country", "region", "district", "project_stage",
    "deposit_type", "deposit_subtype", "tectonic_belt", "host_rock",
    "host_rock_class", "mineralization_style", "mineralization_mode",
    "mineralization_pattern", "alteration_signature", "mining_method",
    "mining_method_class", "recovery_method", "resource_compliance_standard",
    "resource_vintage_year", "strike_length_meters", "width_meters",
    "depth_meters", "drilling_evidence",
)

_MRE_FINGERPRINT_FIELDS = (
    "tonnage_mt", "grade_value", "total_contained",
    "mre_mi_tonnage_mt", "mre_mi_grade", "mre_mi_contained",
    "mre_inferred_tonnage_mt", "mre_inferred_grade", "mre_inferred_contained",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _commodity(project: Dict[str, Any]) -> str:
    return str(project.get("material") or project.get("commodity") or "unknown").strip().lower() or "unknown"


def _mode(use_mre: bool) -> str:
    return "post_mre" if use_mre else "blind_pre_mre"


def _cutoff_date(project: Dict[str, Any], use_mre: bool) -> Optional[str]:
    if use_mre:
        return None
    cutoff = _target_mre_cutoff(project)
    return cutoff.isoformat() if cutoff else None


def _project_fingerprint(project: Dict[str, Any], *, use_mre: bool) -> Dict[str, Any]:
    keys = list(_PROJECT_FINGERPRINT_FIELDS)
    if use_mre:
        keys.extend(_MRE_FINGERPRINT_FIELDS)
    return {key: project.get(key) for key in keys if key in project}


def _analog_fingerprint(analogs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for analog in analogs or []:
        rows.append({
            "name": analog.get("name") or analog.get("analog_name"),
            "material": analog.get("material") or analog.get("analog_material"),
            "deposit_type": analog.get("deposit_type") or analog.get("analog_deposit_type"),
            "deposit_subtype": analog.get("deposit_subtype") or analog.get("analog_deposit_subtype"),
            "tectonic_belt": analog.get("tectonic_belt") or analog.get("analog_tectonic_belt"),
            "tonnage_mt": analog.get("tonnage_mt") or analog.get("analog_tonnage_mt"),
            "grade_value": analog.get("grade_value") or analog.get("analog_grade_value"),
            "mre_mi_tonnage_mt": analog.get("mre_mi_tonnage_mt"),
            "mre_inferred_tonnage_mt": analog.get("mre_inferred_tonnage_mt") or analog.get("inferred_tonnage_mt"),
            "source_url": analog.get("source_url") or analog.get("mre_source_url"),
            "drilling_evidence": analog.get("drilling_evidence"),
        })
    return sorted(rows, key=lambda row: str(row.get("name") or "").lower())


def build_intelligence_cache_key(
    project: Dict[str, Any],
    analogs: List[Dict[str, Any]],
    *,
    use_mre: bool,
) -> Tuple[str, Dict[str, Any]]:
    """Return the stable cache key and the fingerprint payload used for it."""
    payload = {
        "version": 1,
        "project_id": project.get("id"),
        "commodity": _commodity(project),
        "mode": _mode(use_mre),
        "cutoff_date": _cutoff_date(project, use_mre),
        "project": _project_fingerprint(project, use_mre=use_mre),
        "analogs": _analog_fingerprint(analogs),
    }
    return hash_json(payload), payload


def _intelligence_schema() -> Dict[str, Any]:
    string = {"type": "string"}
    obj = {"type": "object", "additionalProperties": True}
    source = {
        "type": "object",
        "additionalProperties": False,
        "required": ["role", "used_for", "title", "summary", "confidence"],
        "properties": {
            "role": string,
            "used_for": {"type": "array", "items": string},
            "title": string,
            "url": string,
            "publisher": string,
            "source_date": string,
            "summary": string,
            "excerpt": string,
            "confidence": {"type": "string", "enum": ["low", "medium", "high", ""]},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "project_dossier", "deposit_classification", "evidence_inventory",
            "evidence_gaps", "analog_logic", "rule_pack", "sources_used",
        ],
        "properties": {
            "project_dossier": obj,
            "deposit_classification": obj,
            "evidence_inventory": {"type": "array", "items": obj},
            "evidence_gaps": {"type": "array", "items": string},
            "analog_logic": obj,
            "rule_pack": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "commodity", "archetype", "rules", "scale_logic",
                    "grade_logic", "contained_logic", "mi_inferred_split_logic",
                    "uncertainty_logic",
                ],
                "properties": {
                    "commodity": string,
                    "archetype": string,
                    "rules": {"type": "array", "items": obj},
                    "scale_logic": obj,
                    "grade_logic": obj,
                    "contained_logic": obj,
                    "mi_inferred_split_logic": obj,
                    "uncertainty_logic": obj,
                },
            },
            "sources_used": {"type": "array", "items": source},
        },
    }


def _build_intelligence_prompt(
    *,
    project: Dict[str, Any],
    analogs: List[Dict[str, Any]],
    use_mre: bool,
    cache_key: str,
) -> str:
    commodity = _commodity(project)
    mode = _mode(use_mre)
    cutoff = _target_mre_cutoff(project) if not use_mre else None
    project_block = _format_project_block(project, use_mre=use_mre)
    analogs_block = _format_analogs_block(analogs, cutoff_date=cutoff)
    chronology = _chronology_directive(cutoff) if not use_mre else "POST-MRE MODE\nOfficial MRE fields may be used when supplied."
    return f"""
You are building the PROJECT INTELLIGENCE artifact for a mining resource model.
This is NOT the final prediction step. Your output is a structured dossier and
a project-specific rule pack that a later prediction step must apply exactly.

Commodity: {commodity}
Mode: {mode}
Cache key: {cache_key}

Rules for this intelligence artifact:
  • Research the supplied project context and analog context deeply enough to
    classify the deposit/archetype and scale logic.
  • Create project-specific rules for scale, grade, contained metal,
    M&I/Inferred split, and uncertainty.
  • Select and reject analog logic explicitly.
  • List every material source used. Prefer primary operator releases,
    technical reports, exchange filings, annual reports, and database pages
    only when they are used as secondary context.
  • Do not output final MRE predictions here.
  • In blind/pre-MRE mode, do not quote, paraphrase, cite, or use target MRE
    numbers or post-cutoff target MRE sources.

{chronology}

================================================================
TARGET PROJECT
================================================================
{project_block}

================================================================
ANALOG COHORT
================================================================
{analogs_block}

Return ONLY JSON matching the schema. Keep summaries compact.
""".strip()


def normalize_intelligence_output(
    result: Dict[str, Any],
    *,
    project: Dict[str, Any],
    use_mre: bool,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Project intelligence output is not a JSON object")
    missing = [
        key for key in (
            "project_dossier", "deposit_classification", "evidence_inventory",
            "evidence_gaps", "analog_logic", "rule_pack", "sources_used",
        )
        if key not in result
    ]
    if missing:
        raise ValueError(f"Project intelligence output missing keys: {', '.join(missing)}")
    sources = result.get("sources_used")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Project intelligence output has no material sources")
    rule_pack = result.get("rule_pack")
    if not isinstance(rule_pack, dict) or not rule_pack.get("archetype"):
        raise ValueError("Project intelligence output has no rule_pack.archetype")
    if not rule_pack.get("commodity"):
        rule_pack["commodity"] = _commodity(project)
    normalized = dict(result)
    normalized["rule_pack"] = rule_pack
    normalized["rule_pack_hash"] = hash_json(rule_pack)
    normalized["mode"] = _mode(use_mre)
    normalized["commodity"] = _commodity(project)
    normalized["cutoff_date"] = _cutoff_date(project, use_mre)
    return normalized


def _row_to_intelligence(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "project_id": row.get("project_id"),
        "commodity": row.get("commodity"),
        "mode": row.get("mode"),
        "cutoff_date": row.get("cutoff_date"),
        "cache_key": row.get("cache_key"),
        "rule_pack_hash": row.get("rule_pack_hash"),
        "project_dossier": row.get("dossier_json") or {},
        "deposit_classification": row.get("classification_json") or {},
        "evidence_inventory": (row.get("quality_json") or {}).get("evidence_inventory") or [],
        "evidence_gaps": (row.get("quality_json") or {}).get("evidence_gaps") or [],
        "analog_logic": (row.get("quality_json") or {}).get("analog_logic") or {},
        "rule_pack": row.get("rule_pack_json") or {},
        "quality": row.get("quality_json") or {},
        "status": row.get("status"),
    }


def project_intelligence_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("error"):
        return {}
    if state.get("use_intelligence_layer") is False:
        return {}

    project = state.get("project") or {}
    analogs = state.get("analogs") or []
    project_id = state.get("project_id") or project.get("id")
    use_mre = bool(state.get("use_mre", True))
    refresh = bool(state.get("refresh_project_intelligence"))
    cache_key, fingerprint = build_intelligence_cache_key(project, analogs, use_mre=use_mre)

    if not refresh:
        cached = supabase_ops.get_cached_project_intelligence_run(cache_key)
        if cached:
            intelligence = _row_to_intelligence(cached)
            logger.info("[project_intelligence] cache hit project=%s run=%s", project_id, cached.get("id"))
            return {
                "project_intelligence": intelligence,
                "intelligence_run_id": cached.get("id"),
                "rule_pack_hash": cached.get("rule_pack_hash"),
            }

    if not settings.parallel_api_key:
        return {"error": "PARALLEL_API_KEY not configured — cannot build project intelligence"}

    prompt = _build_intelligence_prompt(
        project=project,
        analogs=analogs,
        use_mre=use_mre,
        cache_key=cache_key,
    )
    try:
        meta = _run_parallel_task(
            prompt=prompt,
            output_schema=_intelligence_schema(),
            return_meta=True,
        ) or {}
        raw = meta.get("result")
        intelligence = normalize_intelligence_output(raw, project=project, use_mre=use_mre)
    except Exception as exc:
        logger.exception("[project_intelligence] failed for project=%s", project_id)
        return {"error": f"Project intelligence failed: {exc}"}

    quality = {
        "cache_key": cache_key,
        "fingerprint_hash": hash_json(fingerprint),
        "analog_count": len(analogs),
        "source_count": len(intelligence.get("sources_used") or []),
        "evidence_inventory": intelligence.get("evidence_inventory") or [],
        "evidence_gaps": intelligence.get("evidence_gaps") or [],
        "analog_logic": intelligence.get("analog_logic") or {},
    }
    row = {
        "project_id": project_id,
        "commodity": intelligence["commodity"],
        "mode": intelligence["mode"],
        "cutoff_date": intelligence.get("cutoff_date"),
        "cache_key": cache_key,
        "rule_pack_hash": intelligence["rule_pack_hash"],
        "status": "complete",
        "provider_task_id": meta.get("run_id"),
        "processor": settings.parallel_processor,
        "dossier_json": intelligence.get("project_dossier") or {},
        "classification_json": intelligence.get("deposit_classification") or {},
        "rule_pack_json": intelligence.get("rule_pack") or {},
        "quality_json": quality,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=_CACHE_TTL_DAYS)).isoformat(),
    }
    saved = supabase_ops.save_project_intelligence_run(row)
    if saved:
        intelligence["id"] = saved.get("id")
        intelligence["cache_key"] = cache_key
        intelligence["quality"] = quality
        supabase_ops.save_project_intelligence_sources(
            intelligence_run_id=saved["id"],
            project_id=project_id,
            sources=intelligence.get("sources_used") or [],
        )
    return {
        "project_intelligence": intelligence,
        "intelligence_run_id": intelligence.get("id"),
        "rule_pack_hash": intelligence.get("rule_pack_hash"),
    }


def validate_rule_guided_prediction(
    prediction: Dict[str, Any],
    *,
    intelligence: Dict[str, Any],
    use_mre: bool,
) -> List[str]:
    errors: List[str] = []
    if not intelligence:
        return errors
    expected_hash = intelligence.get("rule_pack_hash")
    actual_hash = prediction.get("rule_pack_hash")
    if expected_hash and actual_hash != expected_hash:
        errors.append("prediction rule_pack_hash does not match project intelligence")
    units = prediction.get("units") or {}
    if units.get("tonnage") != "Mt" or units.get("grade") != "g/t" or units.get("contained") != "Moz":
        errors.append("prediction units must be tonnage=Mt, grade=g/t, contained=Moz")
    if not prediction.get("sources_used"):
        errors.append("prediction has no material sources")
    if not use_mre and _blind_result_mentions_mre_anchor(prediction):
        errors.append("blind/pre-MRE prediction appears to reference target MRE leakage")

    for category in ("m_and_i", "inferred"):
        block = prediction.get(category)
        if not isinstance(block, dict):
            errors.append(f"prediction missing {category} block")
            continue
        for range_key in ("tonnage_range_mt", "grade_range_gpt", "contained_range_moz"):
            rng = block.get(range_key)
            if not isinstance(rng, dict):
                errors.append(f"{category}.{range_key} is missing")
                continue
            p10, p50, p90 = rng.get("p10"), rng.get("p50"), rng.get("p90")
            if p10 is None or p50 is None or p90 is None:
                errors.append(f"{category}.{range_key} must include p10/p50/p90")
                continue
            try:
                if not (float(p10) <= float(p50) <= float(p90)):
                    errors.append(f"{category}.{range_key} must satisfy p10 <= p50 <= p90")
            except (TypeError, ValueError):
                errors.append(f"{category}.{range_key} contains non-numeric values")
    return errors
