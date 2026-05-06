# Rules Plan — Compiled Rules Schema Redesign

## What This Task Is About

Fix the `compiled_rules` table in Supabase so that:
1. Analog similarity scores go from ~35% to >55% for real projects
2. Gold and Silver rules are properly separated (with shared rules kept)
3. New `analog_selection` rules (derived from Lessons Learned docs) are seeded
4. Drilling data is incorporated into analog selection via `project_stage` as a proxy
5. Deterministic rule activation replaces the unreliable LLM-based activation

---

## Root Causes Found (Why Analog Scores Were 35%)

1. **Material name mismatch** — `projects.material = "Gold"` (capitalized) but `compiled_rules.source_material = "gold_silver"` (lowercase compound). Zero rules ever loaded.
2. **All 955 rules had null content** — `conditions_json`, `activation_json`, `model_contributions_json` were all null. Rules existed but were empty shells.
3. **Rules not used in analog scoring** — The analog_finder graph never loaded compiled rules at all.
4. **LLM scoring prompt was vague** — Base fallback score was 35 with no commodity-specific criteria injected.

---

## Code Changes Already Committed (commit b1892b6)

All code is committed to `main` branch but NOT yet pushed to GitHub (push was blocked by a hook — user needs to push manually or connect via MCP).

### `nodes/supabase_ops.py`
Added `_MATERIAL_TO_RULES_KEYS` dict so "gold" loads both `gold` and `gold_silver` rules:
```python
_MATERIAL_TO_RULES_KEYS = {
    "gold":      ["gold", "gold_silver"],
    "silver":    ["silver", "gold_silver"],
    "copper":    ["copper"],
    "nickel":    ["nickel"],
    "uranium":   ["uranium"],
    "pgm":       ["pgm"],
    "platinum":  ["pgm"],
    "palladium": ["pgm"],
    "iron":      ["iron"],
    "iron ore":  ["iron"],
}
```
`get_compiled_rules(material, rule_type=None)` now accepts optional `rule_type` filter.

### `nodes/rules_engine.py`
- `load_rules(material, rule_type=None)` — delegates to `get_compiled_rules`
- `activate_rules(project, rules)` — **fully deterministic** (no LLM). Filters rules by:
  - `grade_min` / `grade_max` vs `project.grade_value`
  - `tonnage_min_mt` / `tonnage_max_mt` vs `project.tonnage_mt`
  - `deposit_type` partial match vs `project.deposit_type`
  - `project_stage_filter` partial match vs `project.project_stage`

### `graphs/analog_finder.py`
- Loads only `analog_selection` rules in `score_analogs_node`
- Extracts criteria from `analog_criteria` column (first-class) OR legacy `conditions_json.analog_selection_criteria`
- Injects `project_stage` as drilling-density proxy criterion into LLM scoring prompt
- `_proximity_score` raised base 35→45, deposit_type bonus 10→15, added country +5, project_stage +8, host_rock +5
- `MIN_SCORE` lowered 62→55

### `graphs/report_generator.py`
- `load_rules_node` now loads only `model_adjustment` + `confidence_adjustment` rules
- Excludes `data_quality` rules (those fire on raw drill metrics not in the projects table)

---

## New Scripts Created (committed, not yet run)

### `scripts/migrate_rules_schema.py`
Adds new columns to `compiled_rules`. **Cannot run automatically** — direct postgres port is blocked on this network and `exec_sql` RPC is not enabled on this Supabase project.

**SQL to run in Supabase Dashboard → SQL Editor:**
```sql
-- Step 1: Add new columns
ALTER TABLE compiled_rules
  ADD COLUMN IF NOT EXISTS rule_type TEXT DEFAULT 'data_quality',
  ADD COLUMN IF NOT EXISTS deposit_type TEXT,
  ADD COLUMN IF NOT EXISTS analog_criteria TEXT[],
  ADD COLUMN IF NOT EXISTS grade_min FLOAT,
  ADD COLUMN IF NOT EXISTS grade_max FLOAT,
  ADD COLUMN IF NOT EXISTS grade_unit TEXT,
  ADD COLUMN IF NOT EXISTS tonnage_min_mt FLOAT,
  ADD COLUMN IF NOT EXISTS tonnage_max_mt FLOAT,
  ADD COLUMN IF NOT EXISTS project_stage_filter TEXT,
  ADD COLUMN IF NOT EXISTS drilling_stage TEXT,
  ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS title TEXT,
  ADD COLUMN IF NOT EXISTS description TEXT;

-- Step 2: Tag existing 955 rules as data_quality
UPDATE compiled_rules
SET rule_type = 'data_quality'
WHERE rule_type IS NULL;

-- Step 3: Bridge model_contributions_json → model_effects_json
UPDATE compiled_rules
SET model_effects_json = model_contributions_json
WHERE model_effects_json IS NULL
  AND model_contributions_json IS NOT NULL;

-- Step 4: Performance indexes
CREATE INDEX IF NOT EXISTS idx_compiled_rules_type_material
  ON compiled_rules (rule_type, source_material);

CREATE INDEX IF NOT EXISTS idx_compiled_rules_deposit
  ON compiled_rules (source_material, deposit_type);
```

### `scripts/split_gold_silver_rules.py`
Splits the 215 existing `gold_silver` rules into gold-only, silver-only, or shared buckets.
- Run dry-run first: `python3 scripts/split_gold_silver_rules.py --dry-run`
- Then apply: `python3 scripts/split_gold_silver_rules.py`

Logic:
- Rule_id/lesson name contains "orogenic", "carlin", "heap-leach" → `source_material = 'gold'`
- Rule_id/lesson name contains "crd", "manto", "primary silver" → `source_material = 'silver'`
- `activation_json.conditions[].grade_percent` in gold range → `'gold'`
- `activation_json.conditions[].grade_percent` in silver range → `'silver'`
- Everything else stays `'gold_silver'`

### `scripts/seed_analog_rules.py`
Seeds all new `analog_selection` and `confidence_adjustment` rules derived from the Lessons Learned documents in `/Users/visheshjain/Documents/Mining intellect/Lessons Learned/`.

**Rules seeded per commodity:**

| Commodity | Deposit Types | Rule Count |
|---|---|---|
| Gold | orogenic, epithermal-LS, epithermal-HS, porphyry, carlin, heap-leach | 6 |
| Silver | CRD, manto, epithermal-LS, epithermal-HS, porphyry | 5 |
| Copper | porphyry, VMS, IOCG, skarn, sediment-hosted | 5 |
| Uranium | roll-front, unconformity, intrusive | 3 |
| PGM | merensky-reef, UG2, platreef | 3 |
| Nickel | magmatic-sulphide, laterite | 2 |
| Iron | BIF-hematite, magnetite-itabirite | 2 |
| All commodities | confidence_adjustment by project_stage | 7 × 7 = 49 |

**Example analog_selection rule (gold orogenic):**
- `rule_type = 'analog_selection'`
- `source_material = 'gold'`
- `deposit_type = 'orogenic'`
- `grade_min = 1.5`, `grade_max = 15.0`, `grade_unit = 'g/t Au'`
- `analog_criteria = [array of 8 geological matching criteria]`
- `drilling_stage = 'dense'`

**Confidence adjustment rules:**
One rule per project_stage for each commodity:
- Early Exploration: confidence_delta = -25
- Advanced Exploration: confidence_delta = -15
- PEA / Scoping: confidence_delta = -5
- Pre-Feasibility: confidence_delta = 0
- Feasibility: confidence_delta = +10
- Construction / Production: confidence_delta = +15
- Unknown / Other: confidence_delta = -10

---

## Supabase Database Details

- **Project URL:** `https://imxmfbjeezjantpcrnfc.supabase.co`
- **Project Ref:** `imxmfbjeezjantpcrnfc`
- **Service Role Key:** `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlteG1mYmplZXpqYW50cGNybmZjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDQyMDQzNiwiZXhwIjoyMDg1OTk2NDM2fQ.XMAefRu8jE7GtgGnELI5dBtudDtLjXy_iWe56VW_dqI`

**Current state of `compiled_rules` table:**
- 955 total rules (data_quality type — drill program checks)
- copper: 486, gold_silver: 215, uranium: 166, pgm: 72, nickel: 16, others: ~0
- ALL new columns (`rule_type`, `deposit_type`, etc.) DO NOT EXIST YET
- Must run the ALTER TABLE SQL above before seeding new rules

---

## Full Execution Order (once Supabase MCP is connected)

1. **Run ALTER TABLE SQL** — add all new columns to `compiled_rules`
2. **UPDATE existing rules** — tag all 955 as `rule_type = 'data_quality'`
3. **Run split script** — split gold_silver into gold/silver/gold_silver buckets
4. **Run seed script** — insert all new analog_selection + confidence_adjustment rules
5. **Run repopulate script** — fill title/description for existing 955 data_quality rules
6. **Push git commit** — push `b1892b6` to GitHub (blocked by hook currently)
7. **Verify** — run analog_finder for Camernes Gold, confirm score > 55

---

## Lessons Learned Source Files

Located at: `/Users/visheshjain/Documents/Mining intellect/Lessons Learned/`
- Gold Lessons Learned.docx
- Silver Lessons Learned.docx
- Copper Lessons Learned.docx
- Uranium Lessons Learned.docx
- PGM Lessons Learned.docx
- Nickel Lessons Learned.docx
- Iron Ore Lessons Learned.docx

Key data extracted:
- Drilling density standards per commodity/deposit type
- Grade cutoff ranges
- Analog selection match criteria (deposit type, host rock, structural setting, etc.)
- Project stage → drilling density correlation
- Resource classification standards (M&I vs Inferred spacing)
