# Gold Resource Predictor v2

Gold v2 is a clean, gold-only prediction path. It does not rely on legacy
`projects`, `analogs`, or `model_runs` rows for prediction.

## Contract

- Accuracy over speed or cost.
- `no_prediction` is better than a weak estimate.
- Target MRE data is never used in blind mode.
- Evidence must have a source date before the target MRE cutoff.
- MRE-tainted evidence is rejected.
- Analog candidates must pass deterministic gates before they can influence a
  prediction.
- Cached DB rows can be replayed without Parallel.

## Files

- `sql/gold_resource_predictor_schema.sql` creates the fresh `gold_*` tables.
- `schemas/gold_resource_predictor.py` defines typed rows for the new surface.
- `nodes/gold_resource_predictor.py` contains the deterministic calculator and
  score logic.
- `nodes/gold_resource_storage.py` reads and writes only the new `gold_*` tables.
- `scripts/run_gold_resource_predictor_v2.py` replays one cached DB project and
  optionally saves prediction and score rows.
- `scripts/run_gold_resource_backtest_v2.py` populates the DB-backed loop from
  validated legacy gold truth, pre-MRE evidence, analog candidates/decisions,
  prediction runs, scores, batches, and optional cached Parallel evidence
  research.

## Apply DB Schema

Use Supabase migration tooling or the Supabase MCP migration tool. The SQL is
additive and creates new tables, RLS, indexes, and service-role grants.

```bash
supabase migration new gold_resource_predictor_v2
# paste sql/gold_resource_predictor_schema.sql into the generated file
supabase db push
```

Do not drop old tables as part of this migration. The new path can ignore them
while we prove the DB-backed gold predictor.

## Replay A Cached Project

This does not call Parallel.

```bash
python3 scripts/run_gold_resource_predictor_v2.py --project-id <gold_projects.id>
python3 scripts/run_gold_resource_predictor_v2.py --project-id <gold_projects.id> --save
```

Replay output includes an `audit` block with accepted/rejected evidence counts,
evidence rejection reasons, analog candidate counts, analog decision counts, and
analog rejection reasons. Use this before deciding whether a `no_prediction`
needs more research or is the correct strict outcome.

## Run The DB-Backed Batch Loop

Replay from DB first, without paid Parallel research:

```bash
python3 scripts/run_gold_resource_backtest_v2.py --limit 3
```

Permit cached/Parallel pre-MRE evidence research only when the stored evidence
cannot support a prediction:

```bash
python3 scripts/run_gold_resource_backtest_v2.py \
  --project-id <legacy-project-uuid> \
  --research-missing-evidence \
  --max-parallel-projects 1
```

Permit cached/Parallel first-MRE truth repair only when a project has no
validated first-MRE truth after deterministic screening:

```bash
python3 scripts/run_gold_resource_backtest_v2.py \
  --project-id <legacy-project-uuid> \
  --research-missing-truth \
  --max-parallel-truth-projects 1
```

Truth repair is cached as `task_kind = 'mre_truth'` in `gold_parallel_cache`.
Cached hits replay without counting as new Parallel spend, even when
`--max-parallel-truth-projects 0` blocks new paid calls. Low-confidence,
updated/revised/latest MREs, post-MRE study sources, year-end placeholder dates,
missing source URLs, and incomplete M&I/Inferred split outputs remain excluded.

The runner writes all artifacts to `gold_*` tables. Rejected evidence and analog
decisions are persisted with reasons. If pre-MRE tonnage, grade, or analog
support is insufficient, the result is `no_prediction`.

## Live Test Checklist

1. Pick one gold project with known MRE truth in `gold_mre_truths`.
2. Run `scripts/run_gold_resource_backtest_v2.py --project-id <id>` or replay
   it with `scripts/run_gold_resource_predictor_v2.py --project-id <id> --save`.
3. Confirm accepted evidence has `source_date < cutoff_date`.
4. Confirm rejected evidence/analogs have explicit rejection reasons.
5. Compare predicted tonnage, grade, contained ounces, M&I split, and inferred
   split to the official MRE score row.
6. If the result is `no_prediction`, inspect `no_prediction_reasons` and confirm
   the missing evidence/analog support is genuinely insufficient.

## Required Data Before Prediction

A project will return `no_prediction` unless it has:

- one validated first-MRE `gold_mre_truths` row for scoring; updated MREs,
  post-MRE study sources, year-end placeholder dates, missing source URLs, and
  legacy project MRE mirrors are rejected
- pre-MRE accepted evidence for tonnage and grade
- at least three clean analogs
- at least three split-ready analogs with M&I and inferred tonnage/grade
- compatible subtype, belt, mining method, stage, tonnage band, and grade band

## Current State

This is the DB-backed rebuild path, not a production accuracy claim. The next
production proof step is to use the batch runner on the full validated gold
truth pool, spend Parallel research only where cached DB evidence is insufficient,
and improve the research/analog path project by project from saved failures.
