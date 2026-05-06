"""
migrate_rules_schema.py — Add new columns to compiled_rules and migrate existing data.

Run once after deploying. Safe to re-run (idempotent).

What it does:
1. Adds new columns: rule_type, deposit_type, analog_criteria, grade_min/max, tonnage_min/max_mt,
   project_stage_filter, drilling_stage, active, title, description
2. Tags all existing 955 rules as rule_type='data_quality'
3. Copies model_contributions_json → model_effects_json for existing rules
4. Creates performance indexes

Usage:
    python scripts/migrate_rules_schema.py
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from nodes.supabase_ops import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Supabase doesn't expose raw DDL via the REST client, so we use the pg RPC endpoint
# (exec_sql must be enabled, or run this SQL manually via Supabase Dashboard → SQL Editor)
MIGRATION_SQL = """
-- Step 1: Add new columns
ALTER TABLE compiled_rules
  ADD COLUMN IF NOT EXISTS rule_type TEXT DEFAULT 'data_quality',
  ADD COLUMN IF NOT EXISTS deposit_type TEXT,
  ADD COLUMN IF NOT EXISTS analog_criteria TEXT[],
  ADD COLUMN IF NOT EXISTS grade_min FLOAT,
  ADD COLUMN IF NOT EXISTS grade_max FLOAT,
  ADD COLUMN IF NOT EXISTS grade_unit TEXT,
  ADD COLUMN IF NOT EXISTS tonnage_min_mt FLOAT,
  ADD COLUMN IF NOT EXISTS tonnage_max_mt FLOAT,
  ADD COLUMN IF NOT EXISTS project_stage_filter TEXT,
  ADD COLUMN IF NOT EXISTS drilling_stage TEXT,
  ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS title TEXT,
  ADD COLUMN IF NOT EXISTS description TEXT;

-- Step 2: Tag existing rules as data_quality
UPDATE compiled_rules
SET rule_type = 'data_quality'
WHERE rule_type IS NULL;

-- Step 3: Bridge model_contributions_json → model_effects_json for data_quality rules
UPDATE compiled_rules
SET model_effects_json = model_contributions_json
WHERE model_effects_json IS NULL
  AND model_contributions_json IS NOT NULL;

-- Step 4: Performance indexes
CREATE INDEX IF NOT EXISTS idx_compiled_rules_type_material
  ON compiled_rules (rule_type, source_material);

CREATE INDEX IF NOT EXISTS idx_compiled_rules_deposit
  ON compiled_rules (source_material, deposit_type);
"""


def run_via_rpc():
    """Try to execute migration via Supabase pg_net / exec_sql RPC."""
    import requests
    url = f"{settings.supabase_url}/rest/v1/rpc/exec_sql"
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json={"sql": MIGRATION_SQL}, timeout=60)
    if resp.status_code == 404:
        return False, "exec_sql RPC not available"
    resp.raise_for_status()
    return True, resp.json()


def run_via_psycopg2():
    """Fall back to direct postgres connection if psycopg2 is available."""
    try:
        import psycopg2
    except ImportError:
        return False, "psycopg2 not installed"

    # Supabase postgres connection string format
    project_ref = settings.supabase_url.split("//")[1].split(".")[0]
    conn_str = f"postgresql://postgres:{settings.supabase_service_role_key}@db.{project_ref}.supabase.co:5432/postgres"
    try:
        conn = psycopg2.connect(conn_str)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(MIGRATION_SQL)
        conn.close()
        return True, "migration complete via psycopg2"
    except Exception as e:
        return False, str(e)


def main():
    if not settings.supabase_url or not settings.supabase_service_role_key:
        logger.error("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        sys.exit(1)

    logger.info("Attempting migration via RPC...")
    ok, msg = run_via_rpc()
    if ok:
        logger.info(f"Migration via RPC succeeded: {msg}")
        return

    logger.warning(f"RPC failed ({msg}), trying psycopg2...")
    ok, msg = run_via_psycopg2()
    if ok:
        logger.info(f"Migration via psycopg2 succeeded: {msg}")
        return

    logger.error(f"Both methods failed: {msg}")
    logger.info("\n" + "="*60)
    logger.info("Run this SQL manually in Supabase Dashboard → SQL Editor:")
    logger.info("="*60)
    print(MIGRATION_SQL)
    sys.exit(1)


if __name__ == "__main__":
    main()
