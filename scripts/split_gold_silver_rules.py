"""
split_gold_silver_rules.py — Split gold_silver rules into gold / silver / gold_silver buckets.

Current state: 215 rules tagged source_material='gold_silver'
After split:
  - Rules with grade conditions in gold range (g/t Au: 0.1-20)  → source_material='gold'
  - Rules with grade conditions in silver range (g/t Ag: >20)   → source_material='silver'
  - Rules with no grade condition, or shared deposit types       → stay as 'gold_silver'

The split is done by UPDATING source_material in-place (no duplication needed).
A rule belongs to 'gold' if it's gold-specific, 'silver' if silver-specific,
or 'gold_silver' if it applies to both (e.g. shared epithermal / porphyry rules).

Usage:
    python scripts/split_gold_silver_rules.py
    python scripts/split_gold_silver_rules.py --dry-run
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from nodes.supabase_ops import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Deposit types that are exclusively gold
GOLD_ONLY_DEPOSITS = {"orogenic", "carlin", "carlin-type", "heap-leach", "epithermal au"}

# Deposit types that are exclusively silver
SILVER_ONLY_DEPOSITS = {"crd", "carbonate replacement", "manto", "primary silver", "ag epithermal"}

# Activation fields: if a rule fires on grade_percent in the gold range → gold rule
GOLD_GRADE_MAX = 20.0   # g/t Au — above this is clearly silver
SILVER_GRADE_MIN = 25.0  # g/t Ag — below this is more likely gold


def classify_rule(rule: dict) -> str:
    """Return 'gold', 'silver', or 'gold_silver' for a gold_silver rule."""
    rule_id = (rule.get("rule_id") or "").lower()
    source_lesson = (rule.get("source_lesson") or "").lower()

    # Check deposit type hints in rule_id / lesson name
    combined = f"{rule_id} {source_lesson}"
    for dep in GOLD_ONLY_DEPOSITS:
        if dep in combined:
            return "gold"
    for dep in SILVER_ONLY_DEPOSITS:
        if dep in combined:
            return "silver"

    # Check activation_json grade conditions
    aj = rule.get("activation_json") or {}
    conditions = aj.get("conditions", [])
    for cond in conditions:
        if cond.get("field") == "grade_percent":
            val = float(cond.get("value", 0))
            op = cond.get("op", "")
            # If condition is grade_percent > X where X < GOLD_GRADE_MAX → gold rule
            if op in (">", ">=") and val < GOLD_GRADE_MAX:
                return "gold"
            # If condition is grade_percent > X where X >= SILVER_GRADE_MIN → silver rule
            if op in (">", ">=") and val >= SILVER_GRADE_MIN:
                return "silver"
            # If condition is grade_percent < X where X <= GOLD_GRADE_MAX → gold rule (low-grade)
            if op in ("<", "<=") and val <= GOLD_GRADE_MAX:
                return "gold"

    # Default: shared rule (applies to both)
    return "gold_silver"


def main():
    parser = argparse.ArgumentParser(description="Split gold_silver compiled rules")
    parser.add_argument("--dry-run", action="store_true", help="Print classifications, don't write")
    args = parser.parse_args()

    if not settings.supabase_url:
        logger.error("SUPABASE_URL not set")
        sys.exit(1)

    # Fetch all gold_silver rules
    res = get_client().table("compiled_rules") \
        .select("id,rule_id,source_lesson,activation_json") \
        .eq("source_material", "gold_silver") \
        .limit(500).execute()
    rules = res.data or []
    logger.info(f"Found {len(rules)} gold_silver rules to classify")

    counts = {"gold": 0, "silver": 0, "gold_silver": 0}
    to_update: dict[str, list[str]] = {"gold": [], "silver": []}

    for r in rules:
        bucket = classify_rule(r)
        counts[bucket] += 1
        if bucket in ("gold", "silver"):
            to_update[bucket].append(r["id"])
            if args.dry_run:
                logger.info(f"  [{bucket}] {r['rule_id']}")

    logger.info(f"Classification: gold={counts['gold']} silver={counts['silver']} "
                f"gold_silver={counts['gold_silver']}")

    if args.dry_run:
        logger.info("[dry-run] No changes written.")
        return

    client = get_client()
    for bucket, ids in to_update.items():
        if not ids:
            continue
        # Update in batches of 100
        for i in range(0, len(ids), 100):
            batch = ids[i:i+100]
            client.table("compiled_rules") \
                .update({"source_material": bucket}) \
                .in_("id", batch) \
                .execute()
        logger.info(f"  Updated {len(ids)} rules → source_material='{bucket}'")

    logger.info("Split complete.")


if __name__ == "__main__":
    main()
