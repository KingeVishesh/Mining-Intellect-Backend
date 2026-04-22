"""
Rules Engine — loads compiled rules and activates the relevant ones for a project.
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, List

from nodes.supabase_ops import get_compiled_rules
from nodes.llm_factory import get_llm

logger = logging.getLogger(__name__)


def load_rules(material: str) -> List[Dict]:
    """Load all compiled rules for the given material from Supabase."""
    rules = get_compiled_rules(material)
    logger.info(f"[Rules] Loaded {len(rules)} rules for material={material}")
    return rules


def activate_rules(project: Dict, rules: List[Dict]) -> List[Dict]:
    """
    Use the LLM to select which rules are relevant for this specific project.
    Returns a filtered list of activated rules.
    """
    if not rules:
        return []

    # Build compact rule summaries for the LLM prompt
    rule_summaries = []
    for r in rules:
        rule_summaries.append({
            "rule_id": r.get("rule_id"),
            "impact": r.get("impact"),
            "risk": r.get("risk"),
            "conditions": r.get("conditions_json"),
            "confidence_modifier": r.get("confidence_modifier"),
            "weight": r.get("weight"),
        })

    project_summary = {
        "material": project.get("material"),
        "deposit_type": project.get("deposit_type"),
        "project_stage": project.get("project_stage"),
        "mining_method": project.get("mining_method"),
        "country": project.get("country"),
        "tonnage_mt": project.get("tonnage_mt"),
        "grade_value": project.get("grade_value"),
        "grade_unit": project.get("grade_unit"),
    }

    prompt = f"""You are a mining resource estimation expert.
Select which rules from the list below are RELEVANT to this specific project.

PROJECT:
{json.dumps(project_summary, indent=2)}

AVAILABLE RULES (first {min(50, len(rule_summaries))} rules):
{json.dumps(rule_summaries[:50], indent=2)}

Return ONLY a JSON object:
{{
  "activated_rule_ids": ["rule_id_1", "rule_id_2", ...],
  "reasoning": "brief explanation"
}}
"""
    llm = get_llm()
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        activated_ids = set(parsed.get("activated_rule_ids", []))
        activated = [r for r in rules if r.get("rule_id") in activated_ids]
        logger.info(f"[Rules] {len(activated)}/{len(rules)} rules activated")
        return activated
    except Exception as e:
        logger.error(f"[Rules] Activation error: {e} — returning all rules")
        return rules


def apply_rule_multipliers(
    base_tonnage: float,
    base_grade: float,
    activated_rules: List[Dict],
) -> Dict[str, Any]:
    """
    Apply deterministic rule multipliers to the base estimate.
    Returns adjusted estimates and a confidence modifier.
    """
    tonnage_multiplier = 1.0
    grade_multiplier = 1.0
    confidence_delta = 0.0
    applied = []

    for rule in activated_rules:
        effects = rule.get("model_effects_json") or {}
        if isinstance(effects, str):
            try:
                effects = json.loads(effects)
            except Exception:
                effects = {}

        t_mult = effects.get("tonnage_multiplier", 1.0) or 1.0
        g_mult = effects.get("grade_multiplier", 1.0) or 1.0
        conf = rule.get("confidence_modifier") or 0.0

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
