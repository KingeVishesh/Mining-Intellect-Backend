"""
One-time migration: fill missing company_names, link company_id on all projects,
and report duplicate project groups.

Run from the repo root:
    python scripts/migrate_companies.py

Passes:
  1. Fill missing company_name via Exa Answer API (skips low-confidence answers)
  2. Populate company_id for all projects that have a company_name but no company_id
  3. Print duplicate project groups (same canonical name + material + company)
"""
from __future__ import annotations
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from nodes.supabase_ops import get_client, upsert_company, _canonical
from nodes.exa_answer import ask_company_name


def run() -> None:
    client = get_client()

    print("Fetching all projects...")
    projects = (
        client.table("projects")
        .select("id, name, material, country, region, company_name, company_id")
        .execute()
        .data or []
    )
    print(f"  {len(projects)} projects found\n")

    # ── Pass 1: fill missing company_name via Exa Answer API ──────────────────
    filled = 0
    skipped_low_confidence = 0
    skipped_exa_failure = 0

    missing_company = [p for p in projects if not p.get("company_name")]
    print(f"Pass 1: filling company_name for {len(missing_company)} projects with no company name...")

    for p in missing_company:
        name, confidence = ask_company_name(
            p["name"],
            p.get("material") or "",
            p.get("country"),
            p.get("region"),
        )
        if name and confidence in ("high", "medium"):
            client.table("projects").update({"company_name": name}).eq("id", p["id"]).execute()
            p["company_name"] = name
            filled += 1
            print(f"  ✓ '{p['name']}' → '{name}' ({confidence})")
        elif name and confidence == "low":
            skipped_low_confidence += 1
            print(f"  ~ '{p['name']}' → '{name}' (skipped: low confidence)")
        else:
            skipped_exa_failure += 1
            print(f"  ✗ '{p['name']}' → no result from Exa")

    print(f"\n  Filled: {filled} | Low-confidence skipped: {skipped_low_confidence} | Exa failures: {skipped_exa_failure}\n")

    # ── Pass 2: populate company_id ───────────────────────────────────────────
    needs_company_id = [p for p in projects if p.get("company_name") and not p.get("company_id")]
    print(f"Pass 2: linking company_id for {len(needs_company_id)} projects...")

    linked = 0
    for p in needs_company_id:
        try:
            cid = upsert_company(p["company_name"])
            client.table("projects").update({"company_id": cid}).eq("id", p["id"]).execute()
            p["company_id"] = cid
            linked += 1
        except Exception as e:
            print(f"  ✗ Error linking '{p['name']}': {e}")

    print(f"  Linked: {linked}\n")

    # ── Pass 3: report duplicate project groups ───────────────────────────────
    print("Pass 3: scanning for duplicate project entries...")
    groups: dict = defaultdict(list)
    for p in projects:
        key = (
            _canonical(p["name"]),
            (p.get("material") or "").lower(),
            _canonical(p.get("company_name") or ""),
        )
        groups[key].append({"id": p["id"], "name": p["name"]})

    dupes = {k: v for k, v in groups.items() if len(v) > 1}

    if dupes:
        print(f"  {len(dupes)} duplicate group(s) found — review manually:")
        for (name_key, material_key, company_key), entries in dupes.items():
            print(f"  [{material_key}] '{name_key}' / '{company_key}':")
            for e in entries:
                print(f"    id={e['id']}  name='{e['name']}'")
    else:
        print("  No duplicate projects found.")

    print("\nDone.")
    print(f"  company_names filled: {filled}")
    print(f"  company_id links created: {linked}")
    print(f"  duplicate project groups: {len(dupes)}")


if __name__ == "__main__":
    run()
