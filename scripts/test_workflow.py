"""
test_workflow.py — End-to-end test runner for all 4 graphs.

Usage:
    python scripts/test_workflow.py --graph project_research --project-id <uuid> --name "Athabasca Basin" --material uranium
    python scripts/test_workflow.py --graph analog_finder --project-id <uuid>
    python scripts/test_workflow.py --graph report_generator --project-id <uuid>
    python scripts/test_workflow.py --graph project_discovery
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def test_project_research(project_id: str, project_name: str, material: str, company: str):
    from graphs.project_research import graph

    initial_state = {
        "project_id": project_id,
        "project_name": project_name,
        "material": material,
        "company": company or project_name,
    }

    logger.info(f"=== Testing project_research for: {project_name} ===")
    config = {"configurable": {"thread_id": f"test-research-{project_id}"}}

    # Run until interrupt
    for event in graph.stream(initial_state, config, stream_mode="values"):
        logger.info(f"State keys: {list(event.keys())}")

    state = graph.get_state(config)
    logger.info(f"Graph status: {state.next}")

    if state.next and "human_review" in state.next:
        logger.info("\n=== INTERRUPTED for human review ===")
        pending = state.values.get("extracted_fields", {})
        logger.info(f"Extracted fields preview: {json.dumps({k: v for k, v in list(pending.items())[:5]}, indent=2)}")

        # Auto-approve for testing
        logger.info("\nAuto-approving for test...")
        graph.update_state(config, {"human_approved": True, "human_edits": {}}, as_node="human_review")

        for event in graph.stream(None, config, stream_mode="values"):
            logger.info(f"State keys after resume: {list(event.keys())}")

    final = graph.get_state(config)
    logger.info(f"\n=== Final state ===")
    logger.info(f"  saved: {final.values.get('saved')}")
    logger.info(f"  error: {final.values.get('error')}")


def test_analog_finder(project_id: str):
    from graphs.analog_finder import graph

    logger.info(f"=== Testing analog_finder for project: {project_id} ===")
    config = {"configurable": {"thread_id": f"test-analogs-{project_id}"}}

    for event in graph.stream({"project_id": project_id}, config, stream_mode="values"):
        logger.info(f"State keys: {list(event.keys())}")

    state = graph.get_state(config)
    if state.next and "human_review" in state.next:
        scored = state.values.get("scored_analogs", [])
        logger.info(f"\n=== INTERRUPTED — {len(scored)} analogs for review ===")
        for a in scored[:3]:
            logger.info(f"  {a.get('name')} — score: {a.get('similarity_score')}")

        # Auto-approve top 5 for testing
        top5 = scored[:5]
        graph.update_state(
            config,
            {"human_approved": True, "approved_analogs": top5},
            as_node="human_review",
        )
        for event in graph.stream(None, config, stream_mode="values"):
            logger.info(f"State keys after resume: {list(event.keys())}")

    final = graph.get_state(config)
    logger.info(f"\n=== Final: saved={final.values.get('saved')} error={final.values.get('error')} ===")


def test_report_generator(project_id: str):
    from graphs.report_generator import graph

    logger.info(f"=== Testing report_generator for project: {project_id} ===")
    config = {"configurable": {"thread_id": f"test-report-{project_id}"}}

    for event in graph.stream({"project_id": project_id}, config, stream_mode="values"):
        logger.info(f"State keys: {list(event.keys())}")

    state = graph.get_state(config)
    if state.next and "human_review_model" in state.next:
        m1 = state.values.get("model_1", {})
        logger.info(f"\n=== INTERRUPTED — Model 1: {m1.get('total_tonnage_kt')}kt @ {m1.get('total_grade_pct')} ===")

        graph.update_state(
            config,
            {"human_approved": True, "human_model_edits": {}},
            as_node="human_review_model",
        )
        for event in graph.stream(None, config, stream_mode="values"):
            logger.info(f"State keys after resume: {list(event.keys())}")

    final = graph.get_state(config)
    logger.info(f"\n=== Final: saved={final.values.get('saved')} report_id={final.values.get('report_id')} ===")


def test_project_discovery(materials: list):
    from graphs.project_discovery import graph

    logger.info(f"=== Testing project_discovery for: {materials} ===")
    config = {"configurable": {"thread_id": "test-discovery-1"}}

    for event in graph.stream({"materials": materials}, config, stream_mode="values"):
        logger.info(f"State keys: {list(event.keys())}")

    final = graph.get_state(config)
    logger.info(f"\n=== Final: saved_count={final.values.get('saved_count')} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, choices=["project_research", "analog_finder", "report_generator", "project_discovery"])
    parser.add_argument("--project-id", default="test-project-001")
    parser.add_argument("--name", default="Test Project")
    parser.add_argument("--material", default="uranium")
    parser.add_argument("--company", default="")
    parser.add_argument("--materials", nargs="+", default=["uranium"])
    args = parser.parse_args()

    if args.graph == "project_research":
        test_project_research(args.project_id, args.name, args.material, args.company)
    elif args.graph == "analog_finder":
        test_analog_finder(args.project_id)
    elif args.graph == "report_generator":
        test_report_generator(args.project_id)
    elif args.graph == "project_discovery":
        test_project_discovery(args.materials)


if __name__ == "__main__":
    main()
