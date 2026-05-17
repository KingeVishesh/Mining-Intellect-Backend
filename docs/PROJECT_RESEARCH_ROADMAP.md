# Project Research — Improvement Roadmap

Authored after the May-2026 quality audit, which traced ~30% of gold projects
to upstream `project_research` failures (NULL `deposit_type`/`deposit_subtype`
preventing analog routing).

## Current architecture (post-2026-05-17)

```
load_context           Read existing row to avoid clobber
   ↓
exa_search             One deep search; returns narrative text + sources
   ↓
extract_fields         Grok pass 1: extract fields; pass 2: judge fields;
                       per-field "search_miss" retry via targeted Exa call
   ↓
derive_geological_profile   Deterministic taxonomy detectors on freeform
                           text; synthesize deposit_type from subtype;
                           Grok deposit-type probe when location only
   ↓
geocode                Map location_name → lat/lng
   ↓
validate               Required fields + country/location consistency check;
                       on conflict, clear suspect fields
   ↓
save_to_supabase       Drop nulls so re-runs never wipe good data
```

## What's been fixed (this commit)

| Failure mode | Fix |
|---|---|
| Synthesis skipped when Grok set subtype but missed deposit_type | Use existing OR inferred subtype |
| Grok extracted nothing useful for famous projects (Pasofino Dugbe, Roscan Kandiolé) | `probe_deposit_type()` — focused Grok query using project name + location |
| Wrong-project text crosstalk (Latin Metals Crosby = Texas instead of Argentina) | `_detect_country_conflict()` clears suspect fields before save |
| ~10 projects with stale `enrichment_status=partial` from pre-improvement runs | `scripts/refresh_thin_projects.py` re-runs them through the updated graph |

## What's still weak — improvement roadmap

Ranked by impact-per-effort:

### 1. Confidence-weighted field merging (HIGH impact, MEDIUM effort)

**Problem**: Currently each field's value comes from a single Grok pass. If
Grok hallucinates a Mt-tonnage figure, it sticks. The "judge" step accepts
or rejects but doesn't blend.

**Fix**: For high-leverage fields (deposit_type, tonnage_mt, grade_value),
run a second Grok pass with a different prompt, then take the value that
matches between the two. When they disagree, flag low-confidence and fall
through to a third targeted search.

### 2. Multi-source Exa probe (HIGH impact, MEDIUM effort)

**Problem**: Single Exa query returns a single narrative. For projects with
sparse public data (Santa Fe, Tamarack, Goldfields), Exa returns whatever
comes up first — often the wrong project of the same name.

**Fix**: After the primary search, run targeted secondary queries when
critical fields are still null:

```python
if not fields.get("deposit_type"):
    exa_search.probe("What is the deposit type of {project} ({company}, {country})?")
if not fields.get("tonnage_mt"):
    exa_search.probe("NI 43-101 resource estimate for {project}")
```

Each probe writes the candidate value with a source URL; the final merge
picks values whose source URL contains the company domain (more credible)
over generic news aggregator hits.

### 3. Project-name disambiguation (HIGH impact, LOW effort)

**Problem**: "Santa Fe" matches dozens of projects worldwide. "Goldfields"
is a company name + a region + many specific projects. The Crosby fix
catches geographic collisions; this would catch name collisions before
even calling Exa.

**Fix**: A `disambiguate_project_node` that queries
`projects.name LIKE '%{name}%' OR companies.name LIKE '%{company}%'`
and asks Grok to confirm "Which of these candidate projects is the user's
target, given the country/region context?". Only proceeds with extraction
once a single candidate is selected. Low-confidence → flag for human.

### 4. Field-level confidence scoring (MEDIUM impact, MEDIUM effort)

**Problem**: `field_statuses` is stored but never USED to drive behavior.
Once a field is set, downstream code treats it as ground truth.

**Fix**: Persist a `field_confidences` JSON column on `projects` with a
0.0-1.0 score per field. Downstream consumers can:
- Skip low-confidence fields in modelling (use the next-best source)
- Surface low-confidence fields in the UI for human review
- Trigger re-extraction on schedule when confidence < threshold

Confidence comes from: source-URL credibility, judge accept/reject ratio
on adjacent fields, agreement between multi-probe extractions.

### 5. Auto-correction retry loop (MEDIUM impact, MEDIUM effort)

**Problem**: When `validate_node` clears suspect fields (country conflict
case), we save the cleaned record with `enrichment_status=complete` even
though it's now thin. The next run could re-extract — but project_research
is currently triggered manually.

**Fix**: When `validation_errors` is non-empty, set
`enrichment_status="needs_review"` instead of complete. A scheduled job
re-runs `project_research` for any project in `needs_review` state with a
disambiguation prompt that names the specific conflict
("Searching for {project_name} in {country} ONLY — exclude {wrong_country}").

### 6. Tonnage-grade plausibility check (LOW impact, LOW effort)

**Problem**: Grok occasionally extracts "1,000 Mt @ 50 g/t Au" for a
project that should be 1 Mt @ 5 g/t (decimal misplaced or units confused).

**Fix**: A simple plausibility heuristic in `validate_node`:
- Gold > 30 g/t bulk → flag (only narrow vein deposits go that high)
- Tonnage × grade contained metal > 100 Moz → flag (only super-major
  deposits like Witwatersrand reach this)
- Stage = "exploration" but tonnage_mt > 100 → flag (early projects rarely
  have published mega-resources)

Each rule is one comparison; cumulative effect is catching most data-entry
fat-fingers.

### 7. Human-in-the-loop checkpoint (LOW impact, HIGH effort)

**Problem**: Some projects are genuinely ambiguous (small Australian
explorers with single-page websites) and no automated pipeline will get
them right. The current graph saves whatever it has and moves on.

**Fix**: A `human_review` state when `validation_errors` exceeds a
threshold or `field_confidences` average is below 0.5. The frontend shows
a queue of projects awaiting review with the conflicting/missing fields
highlighted. Only after human approval does `enrichment_status` flip to
complete.

## Sequencing recommendation

For the May-June 2026 push:

1. **This commit** — Fixes 1-4 above (synthesis bug, Grok probe, country
   validation, refresh script). Recovers ~80% of the 32 thin projects.

2. **Next commit** — Add disambiguation (#3) and multi-source Exa (#2).
   Recovers ambiguous-name projects (Santa Fe, Tamarack, Goldfields).

3. **Later** — Confidence scoring (#4) and auto-correction loop (#5).
   Reduces the "stale partial" backlog over time. Requires a scheduled
   refresh job.

4. **When user demands it** — Human-in-the-loop checkpoint (#7). A few
   genuinely-impossible projects will always need manual classification.

## How to measure progress

`scripts/analog_quality_judge.py` already does this. The aggregate score
on a random gold-project sample should climb from today's 44.5 (with 4
zero-scoring projects in the 10-sample) toward 70+ once project_research
recovers the thin ones.

`scripts/eval_gold_corpus.py` snapshots the baseline; regressions in CI.

`analog_quality_gaps` table aggregates the deterministic gap types per
run; SELECT gap_type, count(*) FROM analog_quality_gaps GROUP BY gap_type
shows where to focus.
