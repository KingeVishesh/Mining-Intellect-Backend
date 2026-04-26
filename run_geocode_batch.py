"""
Batch geocoder — finds all projects missing lat/lng and geocodes them
via the geocode_project LangGraph graph on LangGraph Cloud.

Usage:
    python run_geocode_batch.py [--dry-run] [--limit N]

Requires environment variables:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    LANGGRAPH_API_KEY   (lsv2_pt_... personal access token)
"""
import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LANGGRAPH_URL = "https://vishesh-mi-backend-960260026e5555ce9409bc144c51efc8.us.langgraph.app"
ASSISTANT_ID = "geocode_project"
BATCH_SIZE = 10
POLL_INTERVAL = 5        # seconds
MAX_POLL_ATTEMPTS = 36   # 3 minutes max per run


def _lg_headers() -> dict:
    key = os.environ.get("LANGGRAPH_API_KEY", "")
    if not key:
        raise RuntimeError("LANGGRAPH_API_KEY not set")
    return {"x-api-key": key, "Content-Type": "application/json"}


def _sb_headers() -> dict:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY not set")
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def _sb_base() -> str:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("SUPABASE_URL not set")
    return f"{url}/rest/v1"


# ---------------------------------------------------------------------------
# Fetch ungeocoded projects from Supabase
# ---------------------------------------------------------------------------

def fetch_ungeocoded(limit: int = 0) -> list[dict]:
    params = {
        "latitude": "is.null",
        "select": "id,name,company_name,region,country",
        "order": "created_at.asc",
    }
    if limit:
        params["limit"] = str(limit)
    resp = requests.get(f"{_sb_base()}/projects", headers=_sb_headers(), params=params, timeout=30)
    resp.raise_for_status()
    projects = resp.json()
    logger.info(f"Found {len(projects)} projects with missing coordinates")
    return projects


# ---------------------------------------------------------------------------
# Run one project through LangGraph Cloud
# ---------------------------------------------------------------------------

def run_geocode(project: dict, dry_run: bool = False) -> dict:
    pid = project["id"]
    name = project.get("name", pid)
    result = {"project_id": pid, "project_name": name, "method": None,
              "latitude": None, "longitude": None, "status": "unknown", "error": None}

    if dry_run:
        logger.info(f"[DRY RUN] {name} ({pid})")
        result["status"] = "dry_run"
        return result

    hdrs = _lg_headers()

    # Create thread
    try:
        r = requests.post(f"{LANGGRAPH_URL}/threads", headers=hdrs, json={}, timeout=30)
        r.raise_for_status()
        thread_id = r.json()["thread_id"]
    except Exception as e:
        result.update(status="error", error=f"create thread: {e}")
        logger.error(f"[{name}] {result['error']}")
        return result

    # Start run
    try:
        r = requests.post(
            f"{LANGGRAPH_URL}/threads/{thread_id}/runs",
            headers=hdrs,
            json={"assistant_id": ASSISTANT_ID, "input": {"project_id": pid}},
            timeout=30,
        )
        r.raise_for_status()
        run_id = r.json()["run_id"]
    except Exception as e:
        result.update(status="error", error=f"start run: {e}")
        logger.error(f"[{name}] {result['error']}")
        return result

    # Poll
    for _ in range(MAX_POLL_ATTEMPTS):
        time.sleep(POLL_INTERVAL)
        try:
            r = requests.get(f"{LANGGRAPH_URL}/threads/{thread_id}/runs/{run_id}", headers=hdrs, timeout=15)
            r.raise_for_status()
            run_status = r.json().get("status", "")
            if run_status in ("success", "error", "timeout"):
                break
        except Exception:
            continue
    else:
        result.update(status="timeout", error="exceeded max poll attempts")
        logger.error(f"[{name}] Timed out")
        return result

    if run_status != "success":
        result.update(status="error", error=f"run status: {run_status}")
        logger.error(f"[{name}] {result['error']}")
        return result

    # Read output state
    try:
        r = requests.get(f"{LANGGRAPH_URL}/threads/{thread_id}/state", headers=hdrs, timeout=15)
        r.raise_for_status()
        state = r.json().get("values", {})
    except Exception as e:
        result.update(status="error", error=f"state fetch: {e}")
        return result

    result["method"] = state.get("method")
    result["latitude"] = state.get("latitude")
    result["longitude"] = state.get("longitude")

    if state.get("error"):
        result.update(status="error", error=state["error"])
    elif state.get("saved"):
        result["status"] = "success"
    else:
        result["status"] = "no_coords"

    logger.info(f"[{name}] method={result['method']} lat={result['latitude']} lng={result['longitude']} status={result['status']}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch geocode mining projects")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Max projects (0=all)")
    args = parser.parse_args()

    projects = fetch_ungeocoded(limit=args.limit)
    if not projects:
        logger.info("No projects to geocode.")
        return

    logger.info(f"Processing {len(projects)} projects {'(DRY RUN)' if args.dry_run else ''}")

    results = []
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
        futures = {pool.submit(run_geocode, p, args.dry_run): p for p in projects}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                p = futures[future]
                results.append({"project_id": p["id"], "project_name": p.get("name"),
                                 "status": "error", "error": str(e), "method": None,
                                 "latitude": None, "longitude": None})

    # Summary
    by_method: dict[str, int] = {}
    errors = []
    for r in results:
        if r["status"] == "success":
            m = r.get("method") or "unknown"
            by_method[m] = by_method.get(m, 0) + 1
        elif r["status"] not in ("dry_run",):
            errors.append(r)

    print("\n" + "=" * 60)
    print("GEOCODE BATCH SUMMARY")
    print("=" * 60)
    print(f"Total:           {len(results)}")
    for method, count in sorted(by_method.items()):
        print(f"  via {method:12s}: {count}")
    print(f"No coords found: {sum(1 for r in results if r['status'] == 'no_coords')}")
    print(f"Errors:          {len(errors)}")
    if errors:
        print("\nFailed:")
        for r in errors[:20]:
            print(f"  {r['project_name']}: {r['error']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
