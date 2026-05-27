"""Verify analog M&I + Inferred breakdowns against primary sources.

Runs the cross-validated 2-pass extractor for one or more analogs and
prints a per-analog report showing:

  * Pass-A and Pass-B raw values from each Exa query
  * Consensus value + relative disagreement
  * Per-field confidence (high / medium / low)
  * Source URL + publisher + publication year

Use this BEFORE adding analog Inferred values to a fixture file or
trusting a production-extracted value. The output is also machine-
readable (JSON when --json is passed) so it can be piped into a fixture
update.

Examples:
    # Single analog
    python -m scripts.verify_analog_inferred --name "Fort Knox" --material gold

    # Multiple analogs at once
    python -m scripts.verify_analog_inferred \\
        --name "Fort Knox" --name "Eagle Gold Project" \\
        --name "Brewery Creek" --name "Coffee Gold Project" \\
        --material gold --json > verified.json

    # Optional location context — improves Exa's ranking when the analog
    # name is ambiguous (e.g. multiple "Eagle" projects worldwide)
    python -m scripts.verify_analog_inferred --name "Eagle Gold Project" \\
        --material gold --country Canada --region Yukon
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from typing import List, Optional

# Wire root logger to stderr so the JSON output on stdout stays clean.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stderr,
)

from nodes import inferred_extractor


def _color(conf: str) -> str:
    return {"high": "\033[32m", "medium": "\033[33m", "low": "\033[31m"}.get(conf, "")


def _print_report(name: str, result: Optional[dict]) -> None:
    print()
    print("=" * 78)
    print(f"Analog: {name}")
    print("-" * 78)
    if not result:
        print("  Extraction failed — no MRE breakdown found by either Exa pass.")
        print("  Possible reasons: ambiguous name, no public NI 43-101 / JORC")
        print("  filing, or analog is too obscure for Exa's index.")
        return

    cv = result.get("cross_validation", {}) or {}

    print(f"  Source URL:  {result.get('source_url') or '—'}")
    print(f"  Publisher:   {result.get('publisher') or '—'}")
    print(f"  As of:       {result.get('as_of_year') or '—'}")
    print(f"  Overall conf: {_color(result.get('confidence', 'low'))}"
          f"{result.get('confidence', 'low')}\033[0m")
    print()
    print(f"  {'Field':22s}  {'Pass A':>14s}  {'Pass B':>14s}  "
          f"{'Consensus':>14s}  {'Δ%':>8s}  Conf")
    for field, label in (
        ("mi_tonnage_mt",       "M&I tonnage (Mt)"),
        ("mi_grade",            "M&I grade"),
        ("inferred_tonnage_mt", "Inferred tonnage (Mt)"),
        ("inferred_grade",      "Inferred grade"),
    ):
        c = cv.get(field, {}) or {}
        a = c.get("pass_a")
        b = c.get("pass_b")
        cons = c.get("consensus")
        rd = c.get("rel_diff")
        conf = c.get("confidence", "none")
        a_s = f"{a:14.4g}" if a is not None else f"{'—':>14s}"
        b_s = f"{b:14.4g}" if b is not None else f"{'—':>14s}"
        cons_s = f"{cons:14.4g}" if cons is not None else f"{'—':>14s}"
        rd_s = f"{rd*100:7.1f}%" if rd is not None else f"{'—':>8s}"
        col = _color(conf)
        print(f"  {label:22s}  {a_s}  {b_s}  {cons_s}  {rd_s}  "
              f"{col}{conf}\033[0m")
    print()
    flagged = [
        field for field, c in cv.items()
        if (c or {}).get("confidence") == "low"
    ]
    if flagged:
        print(f"  \033[31m⚠ FLAGGED for human review:\033[0m {', '.join(flagged)}")
        print("    Two passes disagreed by >20%. Don't trust these values until")
        print("    cross-checked manually against the source URL above.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", action="append", required=True,
                        help="Analog name (repeatable for batch verification).")
    parser.add_argument("--material", default="gold",
                        help="Material code (gold, copper, silver, etc.). Default: gold.")
    parser.add_argument("--country", default=None,
                        help="Optional country hint (improves Exa ranking).")
    parser.add_argument("--region", default=None,
                        help="Optional region/state hint.")
    parser.add_argument("--deposit-type", default=None,
                        help="Optional deposit-type hint (e.g. 'IRGS', 'orogenic gold').")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON to stdout (one object per analog) instead of a human report.")
    args = parser.parse_args(argv)

    out: List[dict] = []
    for name in args.name:
        result = inferred_extractor.extract_inferred_breakdown(
            analog_name=name,
            material=args.material,
            country=args.country,
            region=args.region,
            deposit_type=args.deposit_type,
        )
        if args.json:
            out.append({
                "name":      name,
                "material":  args.material,
                "result":    result,
            })
        else:
            _print_report(name, result)

    if args.json:
        json.dump(out, sys.stdout, indent=2)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
