"""
Run the companies table SQL migration via Supabase REST API.
Usage:
    SUPABASE_SERVICE_ROLE_KEY=eyJ... python3 scripts/run_sql_migration.py

Or set SUPABASE_SERVICE_ROLE_KEY in your .env file first.
"""
from __future__ import annotations
import os
import sys
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SUPABASE_URL = "https://imxmfbjeezjantpcrnfc.supabase.co"
SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

if not SERVICE_ROLE_KEY:
    print("ERROR: SUPABASE_SERVICE_ROLE_KEY environment variable not set.")
    print("Run:  export SUPABASE_SERVICE_ROLE_KEY=eyJ...")
    sys.exit(1)

SQL = """
-- Step 1: Make user_id nullable (preserves existing data)
ALTER TABLE public.companies ALTER COLUMN user_id DROP NOT NULL;

-- Step 2: Drop user-scoped RLS policies
DROP POLICY IF EXISTS "Users can view their own companies" ON public.companies;
DROP POLICY IF EXISTS "Users can create their own companies" ON public.companies;
DROP POLICY IF EXISTS "Users can update their own companies" ON public.companies;
DROP POLICY IF EXISTS "Users can delete their own companies" ON public.companies;

-- Step 3: New policy - any authenticated user can read all companies
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'companies'
    AND policyname = 'Authenticated users can view companies'
  ) THEN
    CREATE POLICY "Authenticated users can view companies"
      ON public.companies FOR SELECT TO authenticated USING (true);
  END IF;
END $$;
"""

headers = {
    "apikey": SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}

resp = requests.post(
    f"{SUPABASE_URL}/rest/v1/rpc/exec_sql",
    headers=headers,
    json={"query": SQL},
    timeout=30,
)

if resp.status_code == 404:
    # exec_sql RPC not available — use pg_meta API instead
    resp = requests.post(
        f"{SUPABASE_URL}/pg",
        headers={**headers, "Content-Type": "text/plain"},
        data=SQL,
        timeout=30,
    )

if resp.ok:
    print("SQL migration completed successfully.")
else:
    print(f"HTTP {resp.status_code}: {resp.text[:500]}")
    print("\nThe Supabase REST API doesn't expose direct SQL execution.")
    print("Please run this SQL manually in the Supabase dashboard SQL editor:")
    print("-" * 60)
    print(SQL)
    print("-" * 60)
