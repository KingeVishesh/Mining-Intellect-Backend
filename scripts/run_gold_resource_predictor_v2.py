#!/usr/bin/env python3
"""Run the gold-only predictor from cached database rows.

This script does not call Parallel. It loads a validated gold project bundle
from the new `gold_*` tables, runs the deterministic calculator, scores against
stored MRE truth, and optionally saves the prediction + score rows.

Examples:
    python3 scripts/run_gold_resource_predictor_v2.py --project-id <uuid>
    python3 scripts/run_gold_resource_predictor_v2.py --project-id <uuid> --save
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from nodes.gold_resource_predictor import predict_gold_resource, score_gold_prediction  # noqa: E402
from nodes.gold_resource_storage import (  # noqa: E402
    insert_gold_prediction_run,
    insert_gold_prediction_score,
    load_gold_case_bundle,
)


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _prediction_run_row(project_id: str, truth: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "project_id": project_id,
        "mre_truth_id": truth.get("id"),
        "run_mode": "cached_replay",
        "run_status": prediction["run_status"],
        "input_hash": prediction["input_hash"],
        "cutoff_date": prediction["cutoff_date"],
        "evidence_fact_ids": [
            item.get("id")
            for item in prediction.get("calculator_trace", {}).get("accepted_evidence", [])
            if item.get("id")
        ],
        "analog_candidate_ids": [
            decision.get("analog_candidate_id")
            for decision in prediction.get("analog_decisions", [])
            if decision.get("decision") == "accepted" and decision.get("analog_candidate_id")
        ],
        "analog_decision_ids": [],
        "no_prediction_reasons": prediction.get("no_prediction_reasons") or [],
        "predicted_total_tonnage_mt": prediction.get("predicted_total_tonnage_mt"),
        "predicted_total_grade_gpt": prediction.get("predicted_total_grade_gpt"),
        "predicted_total_contained_oz": prediction.get("predicted_total_contained_oz"),
        "predicted_mi_tonnage_mt": prediction.get("predicted_mi_tonnage_mt"),
        "predicted_mi_grade_gpt": prediction.get("predicted_mi_grade_gpt"),
        "predicted_inferred_tonnage_mt": prediction.get("predicted_inferred_tonnage_mt"),
        "predicted_inferred_grade_gpt": prediction.get("predicted_inferred_grade_gpt"),
        "predictor_version": prediction["predictor_version"],
        "calculator_trace": prediction.get("calculator_trace") or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True, help="gold_projects.id to replay")
    parser.add_argument("--threshold", type=float, default=0.05, help="Pass threshold as a fraction. Default: 0.05")
    parser.add_argument("--save", action="store_true", help="Persist prediction and score rows")
    args = parser.parse_args()

    bundle = load_gold_case_bundle(args.project_id)
    if not bundle.get("project"):
        print(f"ERROR: no gold_projects row found for {args.project_id}", file=sys.stderr)
        return 2
    if not bundle.get("truth"):
        print(f"ERROR: no validated gold_mre_truths row found for {args.project_id}", file=sys.stderr)
        return 2

    truth = bundle["truth"]
    cutoff = truth.get("cutoff_date") or truth.get("effective_date") or truth.get("publication_date")
    if not cutoff:
        print("ERROR: validated truth has no cutoff/effective/publication date", file=sys.stderr)
        return 2

    prediction = predict_gold_resource(
        bundle["project"],
        bundle.get("evidence") or [],
        bundle.get("analog_candidates") or [],
        cutoff_date=_parse_date(cutoff),
    )
    score = score_gold_prediction(prediction, truth, threshold=args.threshold)

    saved = {}
    if args.save:
        run = insert_gold_prediction_run(_prediction_run_row(args.project_id, truth, prediction))
        saved["prediction_run"] = run
        if run.get("id") and truth.get("id"):
            score_row = {
                "prediction_run_id": run["id"],
                "mre_truth_id": truth["id"],
                **score,
            }
            saved["prediction_score"] = insert_gold_prediction_score(score_row)

    print(json.dumps({
        "project_id": args.project_id,
        "prediction": prediction,
        "score": score,
        "saved": saved,
    }, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
