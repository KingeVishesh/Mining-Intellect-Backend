"""
Rules Engine — loads compiled rules and activates the relevant ones for a project.
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional

from nodes.supabase_ops import get_compiled_rules

logger = logging.getLogger(__name__)


def load_rules(material: str, rule_type: Optional[str] = None) -> List[Dict]:
    """Load compiled rules for the given material, optionally filtered by rule_type.

    rule_type options:
      'analog_selection'     — criteria for selecting analog projects (used in analog_finder)
      'model_adjustment'     — tonnage/grade multipliers per deposit type (used in report_generator)
      'confidence_adjustment'— confidence deltas based on project stage (used in report_generator)
      'data_quality'         — drill-program quality checks (dormant until drill data available)
      None                   — load all rule types
    """
    rules = get_compiled_rules(material, rule_type=rule_type)
    logger.info(f"[Rules] Loaded {len(rules)} rules for material={material} type={rule_type}")
    return rules


def activate_rules(project: Dict, rules: List[Dict]) -> List[Dict]:
    """Deterministic rule activation based on first-class condition columns.

    Evaluates grade range, tonnage range, deposit type, and project stage filters
    against project data. No LLM call needed — fast and reproducible.
    Falls back to activating all rules if no first-class conditions are set
    (e.g. for legacy data_quality rules before schema migration).
    """
    if not rules:
        return []

    p_grade = float(project.get("grade_value") or 0)
    p_tonnage = float(project.get("tonnage_mt") or 0)
    p_stage = (project.get("project_stage") or "").lower().strip()
    p_deposit = (project.get("deposit_type") or "").lower().strip()

    activated = []
    for r in rules:
        # ── Grade range filter ──────────────────────────────────────────────
        g_min = r.get("grade_min")
        g_max = r.get("grade_max")
        if g_min is not None and p_grade > 0 and p_grade < float(g_min):
            continue
        if g_max is not None and p_grade > 0 and p_grade > float(g_max):
            continue

        # ── Tonnage range filter ────────────────────────────────────────────
        t_min = r.get("tonnage_min_mt")
        t_max = r.get("tonnage_max_mt")
        if t_min is not None and p_tonnage > 0 and p_tonnage < float(t_min):
            continue
        if t_max is not None and p_tonnage > 0 and p_tonnage > float(t_max):
            continue

        # ── Deposit type filter (partial match both ways) ───────────────────
        r_deposit = (r.get("deposit_type") or "").lower().strip()
        if r_deposit and p_deposit:
            if r_deposit not in p_deposit and p_deposit not in r_deposit:
                continue

        # ── Project stage filter ────────────────────────────────────────────
        r_stage = (r.get("project_stage_filter") or "").lower().strip()
        if r_stage and p_stage:
            if r_stage not in p_stage and p_stage not in r_stage:
                continue

        activated.append(r)

    logger.info(f"[Rules] {len(activated)}/{len(rules)} rules activated for project "
                f"deposit_type={p_deposit!r} stage={p_stage!r}")
    return activated


def get_analog_rule(material: str, deposit_type: Optional[str] = None) -> Optional[Dict]:
    """Return the best matching analog_selection rule for this project.

    Priority:
      1. Primary-material rule with matching deposit_type
      2. Any rule with matching deposit_type (cross-material e.g. gold_silver)
      3. None — when deposit_type is unknown, no rule is safer than the wrong rule.
         A laterite rule applied to a sulphide project (or vice versa) poisons the
         Exa query with wrong geological criteria and wrong grade ranges.

    Deliberately no material-only fallback: all compiled analog_selection rules are
    deposit-type-specific (e.g. nickel_laterite vs nickel_magmatic_sulphide). Returning
    the first rule alphabetically when deposit_type is unknown would silently apply the
    wrong rule. Callers should handle None by running a material-only Exa query.
    """
    rules = get_compiled_rules(material, rule_type="analog_selection")
    if not rules:
        return None

    if not deposit_type:
        return None

    mat_lower = material.strip().lower()
    dep_lower = deposit_type.strip().lower()

    # Pass 1: primary material + deposit_type match
    for r in rules:
        r_dep = (r.get("deposit_type") or "").strip().lower()
        r_mat = (r.get("source_material") or "").strip().lower()
        if r_dep and r_mat == mat_lower and (r_dep in dep_lower or dep_lower in r_dep):
            return r

    # Pass 2: any deposit_type match regardless of material (gold_silver, etc.)
    for r in rules:
        r_dep = (r.get("deposit_type") or "").strip().lower()
        if r_dep and (r_dep in dep_lower or dep_lower in r_dep):
            return r

    return None


def get_stage_modifier_map(material: str) -> Dict[str, float]:
    """Return {stage_str: confidence_modifier} from confidence_adjustment rules.

    Used by model_builder to weight analogs by their own project stage — a
    'production' analog (modifier=+15) gets more weight than an 'early exploration'
    analog (modifier=−25) with the same raw similarity score.
    """
    rules = get_compiled_rules(material, rule_type="confidence_adjustment")
    result: Dict[str, float] = {}
    for r in rules:
        stage = (r.get("project_stage_filter") or "").strip().lower()
        modifier = r.get("confidence_modifier")
        if stage and modifier is not None:
            result[stage] = float(modifier)
    return result


def apply_rule_multipliers(
    base_tonnage: float,
    base_grade: float,
    activated_rules: List[Dict],
) -> Dict[str, Any]:
    """Apply deterministic rule multipliers to the base estimate.

    Reads from model_effects_json (preferred) or model_contributions_json (legacy fallback).
    Returns adjusted estimates and a confidence modifier.
    """
    tonnage_multiplier = 1.0
    grade_multiplier = 1.0
    confidence_delta = 0.0
    applied = []

    for rule in activated_rules:
        # Prefer model_effects_json; fall back to model_contributions_json for legacy rules
        effects = rule.get("model_effects_json") or rule.get("model_contributions_json") or {}
        if isinstance(effects, str):
            try:
                effects = json.loads(effects)
            except Exception:
                effects = {}

        t_mult = effects.get("tonnage_multiplier", 1.0) or 1.0
        g_mult = effects.get("grade_multiplier", 1.0) or 1.0

        # For data_quality rules the contribution is a log factor, not a multiplier.
        # Convert: multiplier = exp(tonnage_log_factor) capped to reasonable range.
        if "tonnage_log_factor" in effects and "tonnage_multiplier" not in effects:
            import math
            log_f = float(effects.get("tonnage_log_factor", 0))
            t_mult = math.exp(log_f)

        conf = rule.get("confidence_modifier") or float(effects.get("confidence_delta", 0))

        # Cap individual rule multipliers to avoid runaway compounding
        t_mult = max(0.5, min(2.0, float(t_mult)))
        g_mult = max(0.5, min(2.0, float(g_mult)))

        tonnage_multiplier *= t_mult
        grade_multiplier *= g_mult
        confidence_delta += float(conf)
        applied.append(rule.get("rule_id", "unknown"))

    adjusted_tonnage = base_tonnage * tonnage_multiplier
    adjusted_grade = base_grade * grade_multiplier

    return {
        "adjusted_tonnage": adjusted_tonnage,
        "adjusted_grade": adjusted_grade,
        "tonnage_multiplier": tonnage_multiplier,
        "grade_multiplier": grade_multiplier,
        "confidence_delta": confidence_delta,
        "rules_applied": applied,
    }
