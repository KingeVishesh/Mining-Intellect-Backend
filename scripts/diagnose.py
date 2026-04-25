"""
diagnose.py — Layered diagnostic test for all 4 LangGraph workflows.

Tests in order:
  1. Package imports
  2. Graph compilation (no env vars needed)
  3. Config / env var presence
  4. Supabase connectivity
  5. Exa API connectivity
  6. Grok API connectivity
  7. Full graph run (project_research) — auto-approves the human review

Run from repo root:
  python scripts/diagnose.py
  python scripts/diagnose.py --project-id <uuid> --name "Name" --material copper
"""
import sys, os, json, traceback, argparse
from pathlib import Path

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []


def check(name, fn):
    try:
        detail = fn()
        print(f"{PASS} {name}" + (f" — {detail}" if detail else ""))
        results.append((name, True, None))
        return True
    except Exception as e:
        print(f"{FAIL} {name} — {e}")
        tb = traceback.format_exc()
        # Print short traceback (last 3 lines)
        lines = [l for l in tb.strip().splitlines() if l.strip()]
        for l in lines[-3:]:
            print(f"       {l}")
        results.append((name, False, str(e)))
        return False


def skip(name, reason):
    print(f"{SKIP} {name} — {reason}")
    results.append((name, None, reason))


# ── 1. Package Imports ────────────────────────────────────────────────────────

print("\n=== 1. Package Imports ===")

check("import langgraph", lambda: __import__("langgraph"))
check("import langchain", lambda: __import__("langchain"))
check("import langchain_openai", lambda: __import__("langchain_openai"))
check("import supabase", lambda: __import__("supabase"))
check("import pydantic", lambda: __import__("pydantic"))
check("import pydantic_settings", lambda: __import__("pydantic_settings"))
check("import requests", lambda: __import__("requests"))
check("import geopy", lambda: __import__("geopy"))

# ── 2. Graph Compilation ──────────────────────────────────────────────────────

print("\n=== 2. Graph Compilation (no API calls) ===")

def compile_project_research():
    from graphs.project_research import graph
    assert graph is not None, "graph is None"
    nodes = list(graph.nodes)
    return f"{len(nodes)} nodes: {nodes}"

def compile_analog_finder():
    from graphs.analog_finder import graph
    assert graph is not None
    return f"{len(list(graph.nodes))} nodes"

def compile_report_generator():
    from graphs.report_generator import graph
    assert graph is not None
    return f"{len(list(graph.nodes))} nodes"

def compile_project_discovery():
    from graphs.project_discovery import graph
    assert graph is not None
    return f"{len(list(graph.nodes))} nodes"

check("compile project_research graph", compile_project_research)
check("compile analog_finder graph", compile_analog_finder)
check("compile report_generator graph", compile_report_generator)
check("compile project_discovery graph", compile_project_discovery)

# ── 3. Config / Env Vars ─────────────────────────────────────────────────────

print("\n=== 3. Config & Environment Variables ===")

def load_config():
    from config import settings
    return "loaded"

config_ok = check("load config (pydantic-settings)", load_config)

if config_ok:
    from config import settings

    def check_supabase_url():
        assert settings.supabase_url, "SUPABASE_URL is not set"
        return settings.supabase_url[:40] + "..."

    def check_supabase_key():
        assert settings.supabase_service_role_key, "SUPABASE_SERVICE_ROLE_KEY is not set"
        return f"***{settings.supabase_service_role_key[-4:]}"

    def check_exa_key():
        assert settings.exa_api_key, "EXA_API_KEY is not set"
        return f"***{settings.exa_api_key[-4:]}"

    def check_grok_key():
        assert settings.grok_api_key, "GROK_API_KEY is not set"
        return f"***{settings.grok_api_key[-4:]}"

    sb_url_ok = check("SUPABASE_URL present", check_supabase_url)
    sb_key_ok = check("SUPABASE_SERVICE_ROLE_KEY present", check_supabase_key)
    exa_ok = check("EXA_API_KEY present", check_exa_key)
    grok_ok = check("GROK_API_KEY present", check_grok_key)
else:
    sb_url_ok = sb_key_ok = exa_ok = grok_ok = False

# ── 4. Supabase Connectivity ──────────────────────────────────────────────────

print("\n=== 4. Supabase Connectivity ===")

if sb_url_ok and sb_key_ok:
    def test_supabase_connect():
        from nodes.supabase_ops import get_client
        client = get_client()
        # Try a lightweight query
        res = client.table("projects").select("id").limit(1).execute()
        count = len(res.data) if res.data else 0
        return f"connected — {count} projects in DB (limit 1)"

    def test_project_lookup(project_id):
        from nodes.supabase_ops import get_project
        proj = get_project(project_id)
        if proj:
            return f"found: {proj.get('name')} ({proj.get('material')})"
        return "not found (will be created by research step)"

    check("supabase: connect + list projects", test_supabase_connect)
else:
    skip("supabase: connect + list projects", "missing SUPABASE_URL or key")

# ── 5. Exa API ────────────────────────────────────────────────────────────────

print("\n=== 5. Exa API ===")

if exa_ok:
    def test_exa_call():
        import requests as req
        from config import settings
        # Minimal search call to verify key works
        resp = req.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": settings.exa_api_key, "Content-Type": "application/json"},
            json={"query": "mining project uranium Canada resource estimate", "type": "deep", "numResults": 1},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            results_count = len(data.get("results", []))
            return f"HTTP 200 — {results_count} results"
        else:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")

    check("exa: deep search API call", test_exa_call)
else:
    skip("exa: deep search API call", "EXA_API_KEY not set")

# ── 6. Grok (xAI) API ────────────────────────────────────────────────────────

print("\n=== 6. Grok API ===")

if grok_ok:
    def test_grok_call():
        import requests as req
        from config import settings
        resp = req.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.grok_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-3",
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "temperature": 0,
                "max_tokens": 5,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            reply = resp.json()["choices"][0]["message"]["content"].strip()
            return f"HTTP 200 — reply: {reply!r}"
        else:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")

    check("grok: chat completion call", test_grok_call)
else:
    skip("grok: chat completion call", "GROK_API_KEY not set")

# ── 7. Full Graph Run ─────────────────────────────────────────────────────────

print("\n=== 7. Full Graph Run (project_research, auto-approve) ===")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--project-id", default="diag-test-001")
parser.add_argument("--name", default="Athabasca Basin Uranium Project")
parser.add_argument("--material", default="uranium")
parser.add_argument("--company", default="NexGen Energy")
parser.add_argument("--skip-graph-run", action="store_true")
args, _ = parser.parse_known_args()

if args.skip_graph_run:
    skip("full project_research graph run", "--skip-graph-run flag set")
elif not (sb_url_ok and sb_key_ok and exa_ok and grok_ok):
    skip("full project_research graph run", "one or more API keys missing")
else:
    def test_full_run():
        from graphs.project_research import graph
        initial_state = {
            "project_id": args.project_id,
            "project_name": args.name,
            "material": args.material,
            "company": args.company or args.name,
        }
        config = {"configurable": {"thread_id": f"diag-{args.project_id}"}}

        print(f"\n{INFO} Starting project_research run...")
        print(f"{INFO} project_id={args.project_id}  name={args.name}  material={args.material}")

        # Stream until interrupt
        for event in graph.stream(initial_state, config, stream_mode="values"):
            keys = list(event.keys())
            exa_done = "exa_text" in event
            fields_done = "extracted_fields" in event
            status = []
            if exa_done:
                text_len = len(event.get("exa_text", ""))
                status.append(f"exa_text={text_len}chars")
            if fields_done:
                fields = event.get("extracted_fields", {})
                found = sum(1 for v in fields.values() if v is not None)
                status.append(f"fields={found}/25")
            if event.get("error"):
                status.append(f"ERROR: {event['error']}")
            print(f"  → node output: {', '.join(status) if status else str(keys[:5])}")

        state = graph.get_state(config)
        next_nodes = list(state.next) if state.next else []
        print(f"{INFO} Graph paused at: {next_nodes}")

        if not next_nodes:
            error = state.values.get("error")
            if error:
                raise Exception(f"Graph ended early with error: {error}")
            return "Graph completed without interrupt (unexpected)"

        if "human_review" not in next_nodes:
            raise Exception(f"Expected human_review interrupt, got: {next_nodes}")

        # Show extracted data
        fields = state.values.get("extracted_fields", {})
        found = {k: v for k, v in fields.items() if v is not None}
        print(f"{INFO} Fields found ({len(found)}/25):")
        for k, v in list(found.items())[:8]:
            print(f"       {k}: {v}")
        if len(found) > 8:
            print(f"       ... and {len(found) - 8} more")

        errors = state.values.get("validation_errors", [])
        if errors:
            print(f"{INFO} Validation warnings: {errors}")

        # Auto-approve
        print(f"\n{INFO} Auto-approving (test mode)...")
        graph.update_state(config, {"human_approved": True, "human_edits": {}}, as_node="human_review")

        for event in graph.stream(None, config, stream_mode="values"):
            if event.get("saved") is not None:
                print(f"  → saved={event.get('saved')}  error={event.get('error')}")

        final = graph.get_state(config)
        saved = final.values.get("saved")
        err = final.values.get("error")
        if err:
            raise Exception(f"Save step error: {err}")
        return f"saved={saved}"

    check("full project_research graph run (Exa + Grok + Supabase)", test_full_run)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
passed = sum(1 for _, ok, _ in results if ok is True)
failed = sum(1 for _, ok, _ in results if ok is False)
skipped = sum(1 for _, ok, _ in results if ok is None)
print(f"  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Skipped: {skipped}")

if failed:
    print("\nFailed checks:")
    for name, ok, detail in results:
        if ok is False:
            print(f"  ✗ {name}: {detail}")
    sys.exit(1)
else:
    print("\nAll checks passed!" if not skipped else "\nAll checks passed (some skipped — set env vars to run full suite).")
    sys.exit(0)
