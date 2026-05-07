"""
Local preview tool for the WeasyPrint PDF pipeline.

Usage
-----
  python scripts/preview_pdf.py                         # latest report in DB
  python scripts/preview_pdf.py --report-id <uuid>      # specific report
  python scripts/preview_pdf.py --project-id <uuid>     # latest report for a project
  python scripts/preview_pdf.py --json path/to/file.json # local JSON file

Writes ./out.pdf and opens it (macOS) so you can iterate on templates without
a LangGraph round-trip.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Make the repo importable when invoked from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nodes import pdf_generator  # noqa: E402


def _load_from_supabase(report_id: str | None, project_id: str | None) -> tuple[dict, str]:
    """Pull a report row from Supabase and return (report_json, project_name)."""
    from nodes.supabase_ops import get_client  # local import — avoids env requirement for --json mode

    client = get_client()

    if report_id:
        res = client.table("reports").select("content_json, project_id").eq("id", report_id).single().execute()
        row = res.data
    elif project_id:
        res = (client.table("reports")
               .select("content_json, project_id")
               .eq("project_id", project_id)
               .order("created_at", desc=True)
               .limit(1)
               .execute())
        row = (res.data or [None])[0]
    else:
        res = (client.table("reports")
               .select("content_json, project_id")
               .order("created_at", desc=True)
               .limit(1)
               .execute())
        row = (res.data or [None])[0]

    if not row:
        raise SystemExit("No matching report row found in Supabase.")

    report_json = row["content_json"]
    pid = row["project_id"]
    proj = client.table("projects").select("name").eq("id", pid).single().execute()
    project_name = (proj.data or {}).get("name") or "Unknown Project"
    return report_json, project_name


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a MiningReport JSON to local PDF for preview.")
    ap.add_argument("--report-id", help="Reports table UUID")
    ap.add_argument("--project-id", help="Projects table UUID — uses latest report")
    ap.add_argument("--json", help="Path to a local report_json file")
    ap.add_argument("--out", default="out.pdf", help="Output path (default: ./out.pdf)")
    ap.add_argument("--no-open", action="store_true", help="Don't auto-open the PDF after writing")
    args = ap.parse_args()

    if args.json:
        path = Path(args.json)
        report_json = json.loads(path.read_text())
        project_name = (report_json.get("metadata") or {}).get("project_name") or path.stem
    else:
        report_json, project_name = _load_from_supabase(args.report_id, args.project_id)

    pdf_bytes = pdf_generator.generate_pdf(report_json, project_name)
    out_path = Path(args.out).resolve()
    out_path.write_bytes(pdf_bytes)
    print(f"Wrote {out_path}  ({len(pdf_bytes):,} bytes, project: {project_name})")

    if not args.no_open and sys.platform == "darwin":
        subprocess.run(["open", str(out_path)], check=False)


if __name__ == "__main__":
    main()
