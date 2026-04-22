"""
compile_rules.py — Convert lesson text files into compiled rules in Supabase.

Usage:
    python scripts/compile_rules.py --material uranium
    python scripts/compile_rules.py --material all

This script reads lesson text from the old backend's storage/lesson_library directory
(or from a lessons_text variable below) and uses Grok to compile them into structured rules
that are saved to Supabase's `compiled_rules` table.
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

# Add parent to path so we can import from nodes/
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from nodes.supabase_ops import get_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-3"


def compile_lesson_to_rule(lesson_text: str, material: str, lesson_id: str) -> dict:
    """Use Grok to convert a lesson text into a structured compiled rule."""
    prompt = f"""You are a mining resource estimation expert.
Convert the following lesson text into a structured rule for resource estimation.

LESSON TEXT:
{lesson_text}

Return ONLY this JSON:
{{
  "rule_id": "{lesson_id}",
  "source_material": "{material}",
  "source_lesson": "brief lesson title (5-10 words)",
  "impact": "High|Medium|Low",
  "risk": "High|Medium|Low",
  "weight": 0.0-1.0 (how often this applies),
  "confidence_modifier": -20.0 to +20.0 (effect on model confidence percentage),
  "conditions_json": {{
    "deposit_type": string | null,
    "project_stage": string | null,
    "grade_range": {{"min": number, "max": number}} | null,
    "tonnage_range": {{"min": number, "max": number}} | null
  }},
  "model_effects_json": {{
    "tonnage_multiplier": 0.5-2.0,
    "grade_multiplier": 0.5-2.0
  }},
  "evidence_patterns_json": ["pattern to look for in project data"],
  "activation_json": {{
    "description": "when to apply this rule"
  }}
}}
"""
    headers = {
        "Authorization": f"Bearer {settings.grok_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(f"[compile] Error for lesson {lesson_id}: {e}")
        return {}


def save_rule_to_supabase(rule: dict) -> bool:
    """Save a compiled rule to Supabase."""
    import hashlib
    checksum = hashlib.md5(json.dumps(rule, sort_keys=True).encode()).hexdigest()

    row = {
        "id": str(uuid4()),
        "rule_id": rule.get("rule_id", str(uuid4())),
        "source_material": rule.get("source_material", "unknown"),
        "source_lesson": rule.get("source_lesson", ""),
        "checksum": checksum,
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "compiler_version": "v2",
        "impact": rule.get("impact"),
        "risk": rule.get("risk"),
        "weight": rule.get("weight"),
        "confidence_modifier": rule.get("confidence_modifier"),
        "conditions_json": rule.get("conditions_json"),
        "model_effects_json": rule.get("model_effects_json"),
        "evidence_patterns_json": rule.get("evidence_patterns_json"),
        "activation_json": rule.get("activation_json"),
    }
    try:
        get_client().table("compiled_rules").upsert(row, on_conflict="rule_id").execute()
        return True
    except Exception as e:
        logger.error(f"[save] Supabase error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compile lesson text into Supabase rules")
    parser.add_argument("--material", default="all", help="Material to compile rules for")
    parser.add_argument("--lesson-dir", default=None, help="Directory containing lesson text files")
    args = parser.parse_args()

    if not settings.grok_api_key:
        logger.error("GROK_API_KEY not set")
        sys.exit(1)
    if not settings.supabase_url:
        logger.error("SUPABASE_URL not set")
        sys.exit(1)

    # Default lesson directory from old backend
    lesson_dir = Path(args.lesson_dir) if args.lesson_dir else (
        Path(__file__).parent.parent.parent / "backend" / "storage" / "lesson_library"
    )

    if not lesson_dir.exists():
        logger.error(f"Lesson directory not found: {lesson_dir}")
        sys.exit(1)

    lesson_files = list(lesson_dir.glob("*.txt")) + list(lesson_dir.glob("*.md"))
    if args.material != "all":
        lesson_files = [f for f in lesson_files if args.material.lower() in f.name.lower()]

    logger.info(f"Found {len(lesson_files)} lesson files")

    compiled = 0
    for lesson_file in lesson_files:
        lesson_text = lesson_file.read_text(encoding="utf-8")
        # Infer material from filename
        material = args.material if args.material != "all" else lesson_file.stem.split("_")[0]
        lesson_id = lesson_file.stem

        logger.info(f"Compiling: {lesson_file.name} (material={material})")
        rule = compile_lesson_to_rule(lesson_text, material, lesson_id)

        if rule:
            if save_rule_to_supabase(rule):
                compiled += 1
                logger.info(f"  ✓ Saved rule: {rule.get('rule_id')}")
            else:
                logger.error(f"  ✗ Failed to save rule for {lesson_file.name}")

    logger.info(f"\n=== Done: {compiled}/{len(lesson_files)} rules compiled ===")


if __name__ == "__main__":
    main()
