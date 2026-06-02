"""
Smoke test: run the gold_model_builder Parallel node against the
Cartier Cadillac project (post-MRE / use_mre=True) and print the result.

This bypasses the LangGraph wrapper and the PDF-generator import chain
(weasyprint native libs aren't installed locally) and just exercises the
nodes that actually do the work:
    load project + analogs from Supabase
    fetch drilling evidence (cached / Exa fallback)
    fetch inferred breakdown (cached / Exa fallback)
    call parallel_gold_model_node
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from nodes import supabase_ops
from nodes.parallel_gold_model import parallel_gold_model_node

DEFAULT_PROJECT_ID = "29e753c0-85be-4be2-8cfe-45db3e74f823"  # Cartier - Cadillac Gold Project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mre", action="store_true",
                    help="Pre-MRE / blind backtest: strip published MRE from the prompt")
    ap.add_argument("--project-id", default=DEFAULT_PROJECT_ID,
                    help="Override target project_id")
    ap.add_argument("--find-analogs", action="store_true",
                    help="Let Parallel discover its own analog cohort instead "
                         "of using the Supabase-stored one")
    args = ap.parse_args()
    use_mre = not args.no_mre
    project_id = args.project_id
    find_analogs = args.find_analogs

    project = supabase_ops.get_project(project_id)
    if not project:
        print(f"Project {project_id} not found"); return 1
    analogs = supabase_ops.get_analogs(project_id)
    print(f"Loaded project '{project.get('name')}' with {len(analogs)} analogs.")
    print(f"Mode: {'post-MRE (use_mre=True)' if use_mre else 'PRE-MRE BLIND (use_mre=False)'}"
          f"  |  find_analogs={find_analogs}")
    print(f"  Published MRE: M&I {project.get('mre_mi_tonnage_mt')} Mt @ "
          f"{project.get('mre_mi_grade')} g/t; "
          f"Inferred {project.get('mre_inferred_tonnage_mt')} Mt @ "
          f"{project.get('mre_inferred_grade')} g/t")

    # Build minimal state matching what the graph would produce post-load
    state = {
        "project_id": project_id,
        "project": project,
        "analogs": analogs,
        "use_mre": use_mre,
        "find_analogs": find_analogs,
    }

    print("\nCalling Parallel.ai deep-research agent (this can take 5-12 min)…")
    out = parallel_gold_model_node(state)

    if out.get("error"):
        print(f"\nERROR: {out['error']}")
        return 2

    result = out.get("parallel_model") or {}
    print("\n══════════════════ PARALLEL RESULT ══════════════════")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
