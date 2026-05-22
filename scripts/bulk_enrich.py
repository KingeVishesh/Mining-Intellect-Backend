"""
bulk_enrich.py — Run the project_research pipeline on all unenriched / partial projects.

Calls LangGraph Cloud directly (no frontend auth needed).
Auto-approves the human_review interrupt for every project.
Runs up to CONCURRENCY projects in parallel.

Requirements:
  LANGGRAPH_API_KEY         — LangGraph Cloud API key
  SUPABASE_URL              — e.g. https://imxmfbjeezjantpcrnfc.supabase.co
  SUPABASE_SERVICE_ROLE_KEY — bypasses RLS

Usage:
  python3 scripts/bulk_enrich.py --test           # 1 project only
  python3 scripts/bulk_enrich.py                  # all unenriched/partial
  python3 scripts/bulk_enrich.py --all            # all including complete
  python3 scripts/bulk_enrich.py --limit 20       # cap at 20
  python3 scripts/bulk_enrich.py --material gold  # filter by material
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from datetime import datetime
from pathlib import Path
import httpx

LANGGRAPH_BASE = "https://vishesh-mi-backend-960260026e5555ce9409bc144c51efc8.us.langgraph.app"
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "https://imxmfbjeezjantpcrnfc.supabase.co")
SUPABASE_KEY   = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
LANGGRAPH_KEY  = os.environ.get("LANGGRAPH_API_KEY", "")

CONCURRENCY    = 5
POLL_INTERVAL  = 8
TIMEOUT        = 600

def lg_headers():  return {"X-Api-Key": LANGGRAPH_KEY, "Content-Type": "application/json"}
def sb_headers():  return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}

NEW_FIELDS = ["host_rock","mineralization_style","resource_size_value","by_product_commodities",
              "final_product","ownership_type","district","start_year","end_year",
              "energy_source","climate_terrain","permitting_status","elevation_meters"]

async def fetch_projects(client, material, include_all):
    params = {"select": "id,name,material,company_name,enrichment_status", "order": "created_at.asc"}
    if not include_all:
        params["enrichment_status"] = "in.(unenriched,partial,null)"
    if material:
        params["material"] = f"ilike.*{material}*"
    r = await client.get(f"{SUPABASE_URL}/rest/v1/projects", headers=sb_headers(), params=params)
    r.raise_for_status()
    return r.json()

async def create_thread(client):
    r = await client.post(f"{LANGGRAPH_BASE}/threads", headers=lg_headers(), json={})
    r.raise_for_status()
    return r.json()["thread_id"]

async def start_run(client, thread_id, project):
    r = await client.post(f"{LANGGRAPH_BASE}/threads/{thread_id}/runs", headers=lg_headers(), json={
        "assistant_id": "project_research",
        "input": {"project_id": project["id"], "project_name": project["name"],
                  "material": project["material"], "company": project.get("company_name") or project["name"]},
    })
    r.raise_for_status()
    return r.json()["run_id"]

async def poll_run(client, thread_id, run_id):
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        r = await client.get(f"{LANGGRAPH_BASE}/threads/{thread_id}/runs/{run_id}", headers=lg_headers())
        if r.is_success:
            data = r.json()
            if data.get("status") in ("interrupted","success","error"):
                return data
        await asyncio.sleep(POLL_INTERVAL)
    return {"status": "timeout"}

async def get_state(client, thread_id):
    r = await client.get(f"{LANGGRAPH_BASE}/threads/{thread_id}/state", headers=lg_headers())
    return r.json().get("values", {}) if r.is_success else {}

async def auto_approve(client, thread_id):
    await client.post(f"{LANGGRAPH_BASE}/threads/{thread_id}/state", headers=lg_headers(),
                      json={"values": {"human_approved": True, "human_edits": {}}})
    r = await client.post(f"{LANGGRAPH_BASE}/threads/{thread_id}/runs", headers=lg_headers(),
                          json={"assistant_id": None})
    r.raise_for_status()
    return r.json()["run_id"]

async def enrich_project(client, project, idx, total):
    name, mat = project["name"], project["material"]
    label = f"[{idx}/{total}] {name} ({mat})"
    result = {"id": project["id"], "name": name, "status": "error", "fields_found": 0, "new_fields": {}, "error": None}

    try:
        thread_id = await create_thread(client)
        run_id = await start_run(client, thread_id, project)
        print(f"  {label} — started {run_id[:8]}…")

        run_data = await poll_run(client, thread_id, run_id)
        status = run_data.get("status")

        if status in ("error", "timeout"):
            result["error"] = status
            print(f"  {label} — {status.upper()}")
            return result

        if status == "interrupted":
            values = await get_state(client, thread_id)
            ef = values.get("extracted_fields", {})
            fields_found = sum(1 for v in ef.values() if v not in (None, [], ""))
            extracted_new = {f: ef.get(f) for f in NEW_FIELDS if ef.get(f) not in (None, [], "")}
            result["fields_found"] = fields_found
            result["new_fields"] = extracted_new
            print(f"  {label} — {fields_found} fields extracted | new: {list(extracted_new.keys()) or 'none'}")

            resume_id = await auto_approve(client, thread_id)
            final = await poll_run(client, thread_id, resume_id)
            if final.get("status") == "success":
                result["status"] = "success"
                print(f"  {label} — SAVED ✓")
            else:
                result["error"] = f"post-approval: {final.get('status')}"
                print(f"  {label} — ERROR post-approval")

        elif status == "success":
            result["status"] = "success"
            print(f"  {label} — completed without interrupt")

    except Exception as e:
        result["error"] = str(e)
        print(f"  {label} — EXCEPTION: {e}")

    return result

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--material", type=str)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    if not LANGGRAPH_KEY:
        print("ERROR: set LANGGRAPH_API_KEY"); sys.exit(1)
    if not SUPABASE_KEY:
        print("ERROR: set SUPABASE_SERVICE_ROLE_KEY"); sys.exit(1)

    async with httpx.AsyncClient(timeout=30) as client:
        print("Fetching projects…")
        projects = await fetch_projects(client, args.material, args.all)
        if not projects:
            print("No projects found."); return

        if args.test:   projects = projects[:1]
        elif args.limit: projects = projects[:args.limit]

        total = len(projects)
        concurrency = args.concurrency
        print(f"Processing {total} project(s) | concurrency={concurrency}\n")

        sem = asyncio.Semaphore(concurrency)
        async def run(p, i):
            async with sem:
                return await enrich_project(client, p, i, total)

        results = await asyncio.gather(*[run(p, i+1) for i, p in enumerate(projects)], return_exceptions=True)

    ok  = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    err = [r for r in results if not (isinstance(r, dict) and r.get("status") == "success")]
    print(f"\n{'='*60}")
    print(f"Done: {ok}/{total} saved successfully")
    if err:
        print(f"Failures ({len(err)}):")
        for r in err:
            print(f"  - {r['name'] if isinstance(r, dict) else r}: {r.get('error','') if isinstance(r, dict) else r}")

    log = Path(__file__).parent / f"bulk_enrich_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log.write_text(json.dumps([r if isinstance(r, dict) else {"error": str(r)} for r in results], indent=2))
    print(f"Log: {log}")

if __name__ == "__main__":
    asyncio.run(main())
