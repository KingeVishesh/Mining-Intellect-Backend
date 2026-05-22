# Plan: Analog Weighting via Rules

## What the Rules Tell Us

### `confidence_adjustment` rules (48 rules, 6 stages × 8 materials)
Each maps `project_stage_filter` → `confidence_modifier`:
| Stage | Modifier |
|---|---|
| early exploration | −25 |
| advanced exploration | −15 |
| pea | −5 |
| pre-feasibility | 0 |
| feasibility | +10 |
| production | +15 |

**Insight**: These are currently only applied to the *target project's* conviction delta. But they are equally valid as *per-analog* quality signals. A "production" analog has a proven, operating resource — it should pull harder. An "early exploration" analog has a speculative estimate — it should barely influence the weighted average.

### `analog_selection` rules — `drilling_stage` field
| Value | Deposit types |
|---|---|
| `"dense"` | VMS, skarn, nickel laterite, uranium unconformity/roll-front, heap leach gold |
| `"moderate"` | porphyry (Cu & Au), orogenic gold, IOCG, epithermal (LS & HS), Carlin, BIF/magnetite iron, sediment-hosted Cu, PGM reefs |

**Insight**: For "dense" deposit types, resource estimates require close drillhole spacing to be valid. An analog that only has an Inferred resource (no Indicated/Measured) is less reliable for a dense deposit — it should be penalised.

---

## Current Bug (model_builder.py line 97)

```python
# BUG: null → 50 (wrong), linear weights (too flat)
weights = [float(a.get("similarity_score", 50)) for a in valid]
```

Problems:
1. **Null → 50**: N/A analogs (insufficient data) treated as median. They should carry near-zero influence.
2. **Linear**: score 90 vs 50 is only 1.8× more influential. A mining expert would weight the high-match analog 5–10× more.
3. **No stage signal**: a "production" analog and an "early exploration" analog with the same score are treated identically.
4. **No source signal**: a human-validated library analog and a fresh Exa guess are treated identically.
5. **`avg_similarity` reused in conviction formula** — this will break once we change weights to non-linear.

---

## New Weight Formula

For each analog `a`:

```
adjusted = base + source_bonus + stage_bonus + drilling_penalty

base          = similarity_score if not None else 30.0
source_bonus  = +8 if source == "library" (human-validated, previously approved) else 0
stage_bonus   = confidence_modifier from confidence_adjustment rules for analog.project_stage
                (0 if project_stage unknown)
drilling_pen  = −10 if deposit's drilling_stage == "dense"
                     AND analog.resource_category is Inferred-only (no Indicated/Measured)
                else 0

weight = max(1.0, adjusted) ** 2
```

### Why squared?
Squaring preserves the 0–100 scale for readability but amplifies differences:
- 90 → 8100, 65 → 4225, 50 → 2500, 35 → 1225 → ratios: 3.2×, 6.6× (much better than linear's 1.8×, 2.6×)
- N/A analog with early exploration: 30+0−25=5 → weight=25 (floor) → ~0.1% share

### Worked example — orogenic gold project, 5 analogs

| Analog | Score | Source | Stage bonus | Adjusted | Weight | Share |
|---|---|---|---|---|---|---|
| Otjikoto Mine (library, production) | 85 | lib | +8+15 | 108 | 11 664 | 47.3% |
| Obuasi UG (exa, feasibility) | 72 | exa | +0+10 | 82 | 6 724 | 27.3% |
| Gruyere (exa, pre-feasibility) | 60 | exa | +0+0 | 60 | 3 600 | 14.6% |
| Generic Exa (advanced exp) | 50 | exa | +0−15 | 35 | 1 225 | 5.0% |
| N/A library, no stage | 30 | lib | +8+0 | 38 | 1 444 | 5.9% |
| **Total** | | | | | **24 657** | **100%** |

Compare to current (linear, null→50): all weights within 1.7×. New: top analog is 9.5× more influential than worst.

---

## Updated Conviction Formula

Current `build_model_1()` lines 118–120:
```python
avg_similarity = _weighted_average(weights, [1.0] * len(weights))
analog_confidence = min(100.0, (len(valid) / 5) * 40 + avg_similarity * 0.4)
conviction = max(0.0, min(100.0, analog_confidence + conf_delta))
```

Problem: `avg_similarity` uses the squared weights which are no longer in the 0–100 range.

New (keep conviction on raw scores, not squared weights):
```python
raw_scores = [float(a["similarity_score"]) for a in valid if a.get("similarity_score") is not None]
avg_raw    = sum(raw_scores) / len(raw_scores) if raw_scores else 50.0
n_high     = sum(1 for s in raw_scores if s >= 70)
pool_q     = min(1.0, min(len(valid), 5) / 5 * 0.6 + n_high / max(1, len(valid)) * 0.4)
analog_confidence = min(100.0, pool_q * 60 + avg_raw * 0.4)
conviction = max(0.0, min(100.0, analog_confidence + conf_delta))
```

This makes conviction depend on:
- How many good analogs we have (up to 5) — 60% weight
- Average raw similarity score — 40% weight
- Stage delta from confidence_adjustment rules (unchanged)

---

## Database Change

`report_analogs` is missing `analog_project_stage`. Without it, library analogs all get stage_bonus=0.

```sql
ALTER TABLE report_analogs ADD COLUMN analog_project_stage TEXT;
```

We backfill what we can from existing `similarity_score` + source context (most library analogs were approved with context), but for now NULL is fine — it means stage_bonus=0 (pre-feasibility equivalent).

---

## Files to Modify

| File | Change |
|---|---|
| `nodes/model_builder.py` | Add `_analog_weight(analog, stage_map, drilling_stage)`, update `build_model_1()` to use it; fix conviction formula |
| `nodes/rules_engine.py` | Add `get_stage_modifier_map(material) → dict[str, float]` |
| `nodes/supabase_ops.py` | `get_approved_analogs()` selects `analog_project_stage` |
| `graphs/analog_finder.py` | Exa parsing: include `project_stage` per analog if present in Exa response |
| Database | `ALTER TABLE report_analogs ADD COLUMN analog_project_stage TEXT` |

---

## How `model_builder.py` Gets the Rule Data

`build_model_1(analogs, project, rule_effects)` currently has no access to `analog_rule` or `stage_map`.

**Approach**: fetch them inside `build_model_1()` using existing `rules_engine` helpers:
```python
from nodes.rules_engine import get_analog_rule, get_stage_modifier_map

def build_model_1(analogs, project, rule_effects):
    material = _norm_material(project.get("material", ""))
    deposit_type = project.get("deposit_type")
    
    analog_rule   = get_analog_rule(material, deposit_type)
    drilling_stage = (analog_rule or {}).get("drilling_stage", "moderate")
    stage_map      = get_stage_modifier_map(material)
    
    ...
    weights = [_analog_weight(a, stage_map, drilling_stage) for a in valid]
```

This keeps `build_model_1`'s signature unchanged — no other callers need updating.

---

## Verification Checklist

1. Run report builder on a gold project — confirm no weight equals 50² = 2500 (old default)
2. Confirm a "production" library analog has higher weight than identical-score "early exploration" exa analog
3. Confirm N/A analogs have weight < 1000 (i.e., adjusted < ~31)
4. Confirm `conviction_pct` values are still in 0–100 range
5. Run on a VMS copper project — confirm Inferred-only analog gets drilling_penalty=−10
