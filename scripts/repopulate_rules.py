"""
repopulate_rules.py — Populate conditions_json / model_effects_json for all 955 compiled rules.

All rules currently have null content fields (conditions_json, model_effects_json, impact, risk).
This script uses Grok to generate structured content for each rule from its rule_id and
source_lesson (which encode the deposit subtype), then upserts back to Supabase.

Usage:
    python scripts/repopulate_rules.py
    python scripts/repopulate_rules.py --material gold_silver   # single commodity
    python scripts/repopulate_rules.py --dry-run                # print first batch, don't write

Cost: ~$0.50 in Grok API calls for all 955 rules.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from nodes.supabase_ops import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-3"
BATCH_SIZE = 10


def generate_rule_content(rules_batch: list[dict]) -> list[dict]:
    """Ask Grok to fill in conditions_json and model_effects_json for a batch of rules."""
    stubs = [
        {
            "rule_id": r["rule_id"],
            "source_material": r["source_material"],
            "source_lesson": r["source_lesson"],
        }
        for r in rules_batch
    ]

    prompt = f"""You are a senior mining resource estimation expert.
The following rule stubs were compiled from industry lessons but are missing their content.
For each rule, generate the structured content based on the rule_id and source_lesson —
the deposit subtype is often encoded in the name (e.g. "porphyry", "VMS", "Merensky", "orogenic").

RULE STUBS:
{json.dumps(stubs, indent=2)}

For each rule return a JSON object with these fields:
{{
  "rule_id": "<same rule_id>",
  "impact": "High" | "Medium" | "Low",
  "risk": "High" | "Medium" | "Low",
  "conditions_json": {{
    "deposit_type": string | null,
    "project_stage": string | null,
    "grade_range": {{"min": number, "max": number}} | null,
    "tonnage_range": {{"min": number, "max": number}} | null,
    "analog_selection_criteria": [
      "criterion describing what makes a good analog for this deposit/commodity type",
      "another key selection criterion"
    ]
  }},
  "model_effects_json": {{
    "tonnage_multiplier": number (0.5-2.0),
    "grade_multiplier": number (0.5-2.0),
    "reasoning": "one-line explanation of why these multipliers apply"
  }}
}}

Return ONLY a JSON array of objects, one per rule. No other text.
"""
    headers = {
        "Authorization": f"Bearer {settings.grok_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        # Handle both array and {"rules": [...]} response shapes
        if isinstance(parsed, list):
            return parsed
        for v in parsed.values():
            if isinstance(v, list):
                return v
        logger.warning("Unexpected Grok response shape, skipping batch")
        return []
    except Exception as e:
        logger.error(f"Grok error: {e}")
        return []


def upsert_rules(generated: list[dict], dry_run: bool) -> int:
    """Upsert the generated content back to Supabase. Returns count saved."""
    if not generated:
        return 0

    rows = []
    for g in generated:
        rule_id = g.get("rule_id")
        if not rule_id:
            continue
        rows.append({
            "rule_id": rule_id,
            "impact": g.get("impact"),
            "risk": g.get("risk"),
            "conditions_json": g.get("conditions_json"),
            "model_effects_json": g.get("model_effects_json"),
        })

    if dry_run:
        logger.info(f"[dry-run] Would upsert {len(rows)} rows")
        for r in rows[:2]:
            logger.info(f"  Sample: {json.dumps(r, indent=2)}")
        return len(rows)

    try:
        get_client().table("compiled_rules").upsert(rows, on_conflict="rule_id").execute()
        return len(rows)
    except Exception as e:
        logger.error(f"Supabase upsert error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Populate compiled_rules content via Grok")
    parser.add_argument("--material", default=None, help="Limit to one material (e.g. gold_silver)")
    parser.add_argument("--dry-run", action="store_true", help="Print first batch, don't write")
    args = parser.parse_args()

    if not settings.grok_api_key:
        logger.error("GROK_API_KEY not set in environment")
        sys.exit(1)
    if not settings.supabase_url:
        logger.error("SUPABASE_URL not set in environment")
        sys.exit(1)

    # Fetch all rules that still have null conditions_json
    query = get_client().table("compiled_rules").select("rule_id,source_material,source_lesson")
    if args.material:
        query = query.eq("source_material", args.material)
    res = query.is_("conditions_json", "null").limit(1000).execute()
    rules = res.data or []
    logger.info(f"Found {len(rules)} rules with null conditions_json")

    if not rules:
        logger.info("Nothing to do — all rules already populated.")
        return

    total_saved = 0
    batches = [rules[i:i + BATCH_SIZE] for i in range(0, len(rules), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        logger.info(f"Batch {i+1}/{len(batches)} ({len(batch)} rules) — material={batch[0]['source_material']}")
        generated = generate_rule_content(batch)
        saved = upsert_rules(generated, dry_run=args.dry_run)
        total_saved += saved
        logger.info(f"  Saved {saved}/{len(batch)}")

        if args.dry_run:
            logger.info("[dry-run] Stopping after first batch.")
            break

        # Gentle rate limiting — Grok handles ~10 req/s but no need to hammer it
        time.sleep(0.5)

    logger.info(f"\n=== Done: {total_saved} rules populated ===")


if __name__ == "__main__":
    main()
