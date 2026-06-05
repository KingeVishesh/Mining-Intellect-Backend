"""
nodes/parallel_gold_model.py

Outsources gold-project M&I / Inferred resource estimation to Parallel.ai's
deep-research agent. We hand it the FULL context we already have (project
record, drilling evidence, analogs with their MREs and drilling) so the
agent doesn't spend its budget re-discovering facts we already know. The
agent's job is to:
  1. Learn drilling -> MRE conversion ratios from the analog cohort,
  2. Apply the cohort median to the target's drilling profile,
  3. If MRE is enabled and the target has an official MRE, anchor on it
     (80% official + 20% transformation estimate). If MRE is disabled,
     pretend the official MRE doesn't exist and predict from scratch.

Output is a strict JSON object with M&I and Inferred tonnage / grade /
contained Moz plus the analogs the agent actually relied on and a short
methodology trace.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date
from typing import Any, Dict, List, Optional

import requests

from config import settings

logger = logging.getLogger(__name__)

# Parallel.ai task lifecycle: queued -> running -> completed / failed.
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}
_POLL_INTERVAL_S = 15
_POLL_TIMEOUT_S = 60 * 150  # ultra deep-research with discovery + mandatory enrichment can take 90-120+ min
_PARALLEL_HTTP_RETRIES = 3


# ── Public node entry point ──────────────────────────────────────────────────

def parallel_gold_model_node(state: Dict) -> Dict:
    """LangGraph node: call Parallel.ai with full context, return its estimate.

    State flags:
      use_mre       : bool (default True). When False, strip the project's
                      MRE from the prompt so the agent predicts blind.
      find_analogs  : bool (default False). When True, instruct Parallel to
                      discover its own analog cohort instead of using the
                      one supplied via state["analogs"]. Triples runtime
                      and cost (analog discovery + enrichment + modeling
                      in one call) — use sparingly.
    """
    if state.get("error"):
        return {}

    project = state.get("project") or {}
    analogs = state.get("analogs") or []
    use_mre = bool(state.get("use_mre", True))
    find_analogs = bool(state.get("find_analogs", False))
    cutoff = _target_mre_cutoff(project) if not use_mre else None
    if not use_mre:
        before = len(analogs)
        analogs = _clean_blind_analogs(project, analogs, cutoff)
        if before != len(analogs):
            logger.info("[parallel_gold] blind analog hygiene kept %s/%s supplied analogs", len(analogs), before)
    if not use_mre and len(analogs) < 3 and not find_analogs:
        logger.warning(
            "[parallel_gold] blind mode has only %s supplied analog(s); "
            "continuing with supplied analogs only to avoid target-MRE web "
            "leakage. Refresh Analog Finder before enabling discovery.",
            len(analogs),
        )

    if not settings.parallel_api_key:
        msg = "PARALLEL_API_KEY not configured — cannot run gold_model_builder"
        logger.error(f"[parallel_gold] {msg}")
        return {"error": msg}

    if not analogs and not find_analogs:
        msg = ("No analogs available and find_analogs=False — "
               "either provide a cohort or set find_analogs=True to let "
               "Parallel discover its own.")
        logger.warning(f"[parallel_gold] {msg}")
        return {"error": msg}

    logger.info(
        f"[parallel_gold] starting run: use_mre={use_mre}, "
        f"find_analogs={find_analogs}, supplied_analogs={len(analogs)}"
    )

    prompt = _build_prompt(
        project=project, analogs=analogs,
        use_mre=use_mre, find_analogs=find_analogs,
    )
    schema = _output_schema(use_mre=use_mre)

    try:
        result = _run_parallel_task(prompt=prompt, output_schema=schema)
    except Exception as e:
        logger.exception(f"[parallel_gold] Parallel API call failed: {e}")
        return {"error": f"Parallel API call failed: {e}"}

    if not result:
        if not use_mre and analogs:
            result = _blind_local_fallback_estimate(project, analogs, reason="parallel_no_result")
        else:
            return {"error": "Parallel returned no result"}
    if not use_mre:
        result = _replace_placeholder_blind_estimate(result, analogs, project=project)
        result = _replace_blind_mre_leak_estimate(result, analogs)
        result = _apply_blind_moderate_drilling_fallback_calibration(result, project, analogs)
        result = _apply_blind_single_irgs_scale_floor(result, project, analogs)
        result = _apply_blind_underground_carlin_single_window(result, project, analogs)
        result = _apply_blind_open_pit_carlin_geometry_window(result, project, analogs)
        result = _apply_blind_carlin_heap_grade_tonnage_window(result, project, analogs)
        result = _apply_blind_large_low_grade_carlin_window(result, project, analogs)
        result = _apply_blind_great_basin_heap_breccia_window(result, project, analogs)
        result = _apply_blind_bc_porphyry_sparse_stockwork_window(result, project, analogs)
        result = _apply_blind_guiana_orogenic_open_pit_window(result, project, analogs)
        result = _apply_blind_newfoundland_orogenic_window(result, project, analogs)
        result = _apply_blind_open_pit_orogenic_proxy_window(result, project, analogs)
        result = _apply_blind_great_basin_orogenic_open_pit_window(result, project, analogs)
        result = _apply_blind_sparse_stockwork_lode_window(result, project, analogs)
        result = _apply_blind_yilgarn_small_open_pit_window(result, project, analogs)
        result = _apply_blind_high_grade_vms_scout_window(result, project, analogs)
        result = _apply_blind_yukon_irgs_near_surface_window(result, project, analogs)
        result = _apply_blind_large_yukon_irgs_window(result, project, analogs)
        result = _apply_blind_abitibi_greenstone_district_window(result, project, analogs)
        result = _apply_blind_porphyry_bulk_no_geometry_window(result, project, analogs)
        result = _apply_blind_bc_porphyry_stockwork_grade_window(result, project, analogs)
        result = _apply_blind_large_andean_heap_window(result, project, analogs)
        result = _apply_blind_mature_high_sulfidation_window(result, project, analogs)
        result = _apply_blind_underground_orogenic_no_evidence_window(result, project, analogs)
        result = _apply_blind_sparse_yilgarn_metamorphic_underground_window(result, project, analogs)
        result = _apply_blind_small_underground_vein_window(result, project, analogs)
        result = _apply_blind_broad_bulk_geometry_window(result, project, analogs)
        result = _apply_blind_broad_bulk_scale_floor(result, project, analogs)
        result = _apply_blind_evidence_scale_guard(result, project, analogs)

    logger.info(
        f"[parallel_gold] estimate: M&I={result.get('m_and_i', {}).get('tonnage_mt')} Mt @ "
        f"{result.get('m_and_i', {}).get('grade_gpt')} g/t  |  "
        f"Inferred={result.get('inferred', {}).get('tonnage_mt')} Mt @ "
        f"{result.get('inferred', {}).get('grade_gpt')} g/t  |  "
        f"anchor={result.get('anchor_used')}  conviction={result.get('conviction')}"
    )
    return {
        "parallel_model": result,
        "use_mre": use_mre,
        "find_analogs": find_analogs,
    }


# ── Prompt construction ──────────────────────────────────────────────────────

def _build_prompt(
    *,
    project: Dict,
    analogs: List[Dict],
    use_mre: bool,
    find_analogs: bool = False,
) -> str:
    """The big prompt. Heavy on philosophy + context, light on prescriptive math.

    Everything Parallel needs is included verbatim so it does not spend budget
    re-fetching facts the rest of the pipeline already extracted. The agent
    may still hit the web to fill gaps (e.g. a missing analog MRE breakdown,
    a recent press release with new drill intercepts) but the bulk of the
    work is reasoning over the provided context.

    When find_analogs=True, the prompt instructs the agent to discover its
    own analog cohort instead of using state["analogs"]. The supplied cohort
    (if any) is shown as "starting candidates the upstream system surfaced —
    use, expand, or replace as you see fit".
    """
    project_block = _format_project_block(project, use_mre=use_mre)
    cutoff = _target_mre_cutoff(project) if not use_mre else None
    analogs_block = _format_analogs_block(analogs, cutoff_date=cutoff)
    mre_directive = _mre_directive(project=project, use_mre=use_mre)
    target_enrichment_directive = _target_enrichment_directive(
        blind_mode=not use_mre,
        find_analogs=find_analogs,
    )
    analog_directive = _analog_directive(
        find_analogs=find_analogs,
        analogs=analogs,
        blind_mode=not use_mre,
    )
    chronology_directive = _chronology_directive(cutoff)

    return f"""
You are a senior gold-mining geologist and JORC / NI 43-101 qualified resource
estimator. You are producing an INDEPENDENT M&I + Inferred resource estimate
for a single gold project. You have been handed FULL CONTEXT below — do not
spend time re-discovering it. Use the web only to fill genuine gaps (missing
analog MRE breakdowns, recent press releases with newer drill assays, host-
rock density references). Every numeric output must be defensible to a QP.

================================================================
PHILOSOPHY — HOW THIS MODEL WORKS
================================================================
You are NOT averaging analog MRE outputs. That approach ignores that analogs
have wildly different drilling intensities — a 100 Mt analog with 50,000 m
drilled and a 5 Mt analog with 5,000 m drilled cannot be averaged as equals.

Instead you LEARN A TRANSFORMATION from the analog cohort and APPLY it to
the target project's drilling profile.

CRITICAL: M&I and Inferred are predicted INDEPENDENTLY. They are two
separate volumes drilled at different spacings, not a single budget that
gets split by a share ratio. JORC/NI 43-101 reports them as two distinct
statements ("M&I is X Mt @ Y g/t, Inferred is A Mt @ B g/t") — never as a
total with a percentage split. The model must mirror that.

  For each analog, compute FIVE independent conversion ratios:
      m_and_i_tonnage_per_m       =  analog.m_and_i_tonnage_mt   / analog.total_meters
      m_and_i_grade_preservation  =  analog.m_and_i_grade        / analog.avg_intercept_grade
      inferred_tonnage_per_m      =  analog.inferred_tonnage_mt  / analog.total_meters
      inferred_grade_preservation =  analog.inferred_grade       / analog.avg_intercept_grade
      envelope_realization        =  (analog.m_and_i_tonnage_mt + analog.inferred_tonnage_mt)
                                     / (L × W × H × density)

  If an analog reports M&I but not Inferred (or vice versa), it can still
  contribute to the ratio it has data for — skip it for the other ratio.
  Do not invent values to fill gaps. State per-analog ratio coverage in
  `analogs_used`.

  Take the cohort median (or stage-weighted central tendency) of each
  ratio. Apply to the target SEPARATELY for M&I and Inferred:

      M&I_tonnage_mt     = target.total_meters × median(m_and_i_tonnage_per_m)
      M&I_grade_gpt      = target.avg_intercept_grade × median(m_and_i_grade_preservation)
      M&I_contained_moz  = M&I_tonnage_mt × M&I_grade_gpt × 0.032151

      Inferred_tonnage_mt    = target.total_meters × median(inferred_tonnage_per_m)
      Inferred_grade_gpt     = target.avg_intercept_grade × median(inferred_grade_preservation)
      Inferred_contained_moz = Inferred_tonnage_mt × Inferred_grade_gpt × 0.032151

      total_tonnage_mt   = M&I_tonnage_mt + Inferred_tonnage_mt    (DERIVED)
      total_grade_gpt    = tonnage-weighted average                  (DERIVED)
      total_contained_moz = M&I_contained_moz + Inferred_contained_moz (DERIVED)

  The total is bookkeeping, NOT a prediction. Never compute total first
  and split it. If you find yourself wanting an `m_and_i_share` ratio,
  you are doing it wrong.

GRADE-PROXY FALLBACK (mandatory in blind mode)
- Do NOT return null grade solely because the target lacks a public
  target.avg_intercept_grade. A blind model still needs a grade estimate.
- Use the best PRE-MRE grade proxy available, in this order:
    1. Target drilling_evidence.weighted_grade_g_t, average_intercept_grade_g_t,
       or clearly pre-MRE assay-composite central tendency after top-cutting.
    2. If target proxy is unavailable, use the stage-weighted median analog
       M&I and Inferred resource grades after deposit-style, mining-method,
       and cutoff normalization.
    3. If analog grade cutoffs are mixed, normalize to the reference cutoff
       and widen conviction downward instead of returning null.
- Only return null grade if there are no pre-MRE target grade signals AND no
  valid analog resource grades. If you use this fallback, say
  "grade_proxy=analog_resource_grade" in `methodology.notes`.

TONNAGE-PROXY FALLBACK (mandatory in blind mode)
- Do NOT return null tonnage solely because target.total_meters_drilled is
  unavailable after pre-MRE target enrichment.
- Use the best PRE-MRE tonnage proxy available, in this order:
    1. Target geometric envelope from strike length, depth/down-dip extent,
       true width, and density, with subtype-appropriate realization factors.
    2. Target footprint proxies from pre-MRE mineralized strike, number of
       zones/lodes, drill spacing, and depth extent.
    3. If target proxies are unavailable, use the stage-weighted median
       analog M&I and Inferred tonnages after deposit-style, mining-method,
       and cutoff normalization. Scale toward the target's known strike/depth
       footprint when any target geometry exists.
- Only return null tonnage if there are no pre-MRE target size signals AND
  no valid analog tonnage/resource-category data. If you use this fallback,
  say "tonnage_proxy=analog_resource_tonnage" in `methodology.notes`.
- In blind mode, zero is not an estimate. If analogs have positive resource
  tonnage/grade, return strictly positive M&I and Inferred estimates.

GOLD-SPECIFIC ADJUSTMENTS (mandatory)
- Top-cut intercept grades before averaging. Gold has a strong nugget effect.
  Reasonable caps by subtype: orogenic 30-50 g/t, Carlin 15-25 g/t,
  epithermal LS 50-80 g/t, epithermal HS 30-60 g/t, porphyry Au 5-10 g/t.
  Pick the cap consistent with the deposit subtype below; state it.
- Normalize cutoff grades. Do NOT average analog grades reported at
  incompatible cutoffs. Reference cutoffs: ~0.4-0.5 g/t open-pit,
  ~2.0-3.0 g/t underground. State what you used.
- Stage-weight analogs. Down-weight a Producing/FS analog by ~0.7 when the
  target is Exploration/PEA (mature projects systematically report higher-
  conviction tonnage).
- Geometric ceiling. If you can estimate L × W × H × density for the target
  (from press releases or technical reports), the transformation prediction
  CANNOT exceed it. The envelope is a physical cap.

{mre_directive}

{chronology_directive}

{target_enrichment_directive}

{analog_directive}

================================================================
TARGET PROJECT — FULL CONTEXT
================================================================
{project_block}

================================================================
ANALOG COHORT — STARTING CONTEXT
================================================================
{analogs_block}

================================================================
OUTPUT
================================================================
Return ONLY a JSON object matching the schema you have been given.
Hard rules:
  • Never fabricate. Unknown numeric -> null. Unknown string -> "".
  • All gold grades in g/t. All tonnage in Mt. All contained gold in Moz.
  • Every analog you USE in the math must appear in `analogs_used`
    with the per-analog ratios you derived. If you reject an analog
    list it in `analogs_rejected` with a reason.
  • Each `analogs_used` entry is one compact string:
    "Name | weight | source docs checked | ratios/proxy signal | rationale".
    Name the specific source documents consulted (AIF / NI 43-101 / R&R
    report / quarterly report — with dates and report names), the ratios or
    proxy signal derived, and any missing data. "Data not publicly tabulated"
    or "not disclosed" without naming sources is REJECTED as a non-answer.
    Enrich aggressively before giving up — most major-operator drilling
    data is in R&R appendices.
  • `methodology` must state: which branch ran (mre_anchored /
    drill_transformation / analog_only_fallback), the top-cut value,
    the reference cutoff, any stage-weighting applied, and whether
    the geometric ceiling clamped the result.
  • `conviction` is one of: very_low / low / medium / high / very_high,
    plus a one-sentence rationale.
  • Keep output compact. Put source-document names/URLs inside rationale
    strings when needed; do not add a separate sources table.
""".strip()


def _target_enrichment_directive(*, blind_mode: bool, find_analogs: bool) -> str:
    if blind_mode and find_analogs:
        return """\
TARGET ENRICHMENT — BLIND ANALOG-DISCOVERY MODE
You may search the web to discover and enrich ANALOGS, but do NOT run open
web searches on the target project name, operator, property pages, technical
reports, resource pages, database pages, or SEDAR/filing pages. The target
evidence available to you is limited to the TARGET PROJECT block and any
pre-MRE evidence already supplied there by the upstream cutoff-checked
pipeline.

If the TARGET PROJECT block lacks total_meters_drilled, representative
top-cut assay grade, strike length, true width, or vertical/depth extent,
do not try to fill those target fields from web search. Use analog resource
grade/tonnage proxies, supplied geometry, and conservative stage weighting,
then document "target_open_web_search=disabled_blind" in `methodology.notes`.
This is stricter than ordinary research mode because blind backtests must
avoid post-MRE leakage even at the cost of lower conviction."""

    return """\
TARGET ENRICHMENT — MANDATORY BEFORE ANALOG-ONLY FALLBACK
If the TARGET PROJECT block lacks total_meters_drilled, representative
top-cut assay grade, strike length, true width, or vertical/depth extent,
you MUST search for those pre-MRE target disclosures before using
`analog_only_fallback`. Look first at target press releases, investor
presentations, technical-report summaries, and exchange filings dated before
the cutoff. Use only drill facts that were public before the cutoff.

In blind mode, do NOT use target resource pages, MRE announcements, NI 43-101
MRE technical reports, PEA/PFS/FS reports, database summaries, or any other
target document dated on or after the cutoff, even if it restates older drill
facts. Only use a target drilling fact when you can identify the same fact in
a source published before the cutoff.

Only choose `analog_only_fallback` after documenting the specific pre-cutoff
target sources checked and which required fields were still missing."""


def _analog_directive(*, find_analogs: bool, analogs: List[Dict], blind_mode: bool = False) -> str:
    """Build the analog-selection + enrichment block of the prompt.

    Two modes:
      find_analogs=False (default) — use the supplied cohort, enrich aggressively.
      find_analogs=True             — discover the cohort from scratch (or expand
                                      the supplied starting set), then enrich.
    """
    enrichment_rules = """\
ANALOG ENRICHMENT — MANDATORY, NOT OPTIONAL
For each analog in the cohort that lacks total_meters_drilled,
avg_intercept_grade, M&I breakdown, or Inferred breakdown, you MUST
perform a real web search to find that data before declaring the
ratio null. This is the single biggest lever on model accuracy.

Where to look (in this order):
  1. Operator's most recent Annual Information Form (AIF / Form 20-F) —
     filed annually on SEDAR+ and the operator's IR page. Major-
     producer Resource & Reserve sections include cumulative drilling
     tables per mine.
  2. Most recent NI 43-101 technical report on SEDAR+ for the analog
     (Section 10 "Drilling", Section 14 "Mineral Resource Estimate").
     Even producing mines typically have an updated TR every 3-5 yrs.
  3. JORC competent-person reports (for ASX-listed operators).
  4. The operator's annual Resource & Reserve Report (Barrick, Newmont,
     Agnico, AngloGold, Gold Fields, Kinross publish these as standalone
     PDFs — drilling stats live in appendix tables, not the press release).
  5. Last 3 years of quarterly production reports / operational updates.

CRITICAL: Major operators (Barrick, AngloGold, Newmont, Agnico, Gold
Fields, Kinross, Equinox, B2Gold) DO publish cumulative drilling at the
mine scale. It is in R&R appendices, not on the first page of Google.
If your first 1-2 searches return nothing, you have not yet found the
right document — keep going.

Only declare a ratio null AFTER documenting in analogs_used[].rationale
the specific source documents you checked and what each disclosed or
didn't. Generic phrases like "data not publicly tabulated" without
naming sources are REJECTED as a non-answer."""

    if not find_analogs and blind_mode:
        return """\
ANALOG-SELECTION DISCIPLINE — BLIND SUPPLIED-COHORT MODE
The analogs supplied below have already been vetted by the upstream Analog
Finder. Use this supplied cohort only. Do not replace it with web-discovered
target analogs, and do not search for the target project's resource estimate.

Use the supplied analog resource tonnage/grade and any supplied drilling
evidence. If an analog is missing detailed drilling meters or M&I/Inferred
breakdown, do not spend open-ended research time filling it; mark the ratio
coverage as null, assign lower weight, and proceed with transparent
`analog_only_fallback` or available target pre-MRE drilling evidence.

Reject any source that exposes the target MRE/resource numbers or is dated on
or after the target cutoff. Speed and chronology discipline matter more than
perfect analog enrichment in blind backtest mode."""

    if not find_analogs:
        # Existing behaviour — use the supplied cohort as-is, enrich aggressively
        return f"""\
ANALOG-SELECTION DISCIPLINE
The analogs supplied below have already been vetted for deposit-subtype
match by an upstream system. Do not silently drop or replace them. You MAY
note that an analog seems weak and assign it a lower internal weight, but
record the weighting and rationale in `methodology.notes` and
`analogs_used[].rationale`.

================================================================
{enrichment_rules}
================================================================"""

    # find_analogs=True — agent must discover its own cohort
    starting_size = len(analogs)
    starting_note = (
        f"{starting_size} starting candidate(s) are provided in the ANALOG\n"
        "COHORT block below. Treat them as a seed, not a fixed list. You MUST:"
        if starting_size > 0
        else "No starting candidates are provided. You MUST build the cohort\n"
             "from scratch:"
    )
    return f"""\
ANALOG DISCOVERY — YOU MUST FIND YOUR OWN COHORT
{starting_note}

  1. Identify 5-10 valid analog gold projects for the target. Hard filters:
       a. SAME deposit subtype as the target (no mixing — e.g. never use a
          Carlin analog for an orogenic target, never use a porphyry for
          an epithermal).
       b. SAME mineralization style where determinable (disseminated vs
          vein-hosted vs breccia vs stockwork) — this matters more than
          subtype alone. Mismatched styles produce ratios off by 10-30×.
       c. Within ±5× of the target's tonnage band AND ±3× of grade band
          (use whatever target metadata exists; for blind/pre-MRE runs,
          infer band from drilling data only).
       d. Published M&I or M&I+Inferred resource compliant with NI 43-
          101 / JORC / SAMREC.
       e. Primary-source URL exists.

  2. Drilling-stage-matched preference: the transformation works best
     when the cohort's drilling intensity is comparable to the target.
     Prefer technical-report-stage projects (PEA / PFS / FS) with recent
     NI 43-101s — they cleanly disclose cumulative drilling meters.
     Long-producing major-operator mines typically do NOT publish mine-
     scale cumulative drilling and will return null ratios.

  3. Reject and list in `analogs_rejected` (with reason):
       - Different deposit subtype or mineralization style
       - No published resource
       - Same operator/property as the target (data leakage)
       - No primary-source URL after a reasonable search

  4. After cohort selection, apply the ENRICHMENT rules below to every
     analog. Same discipline: name the documents you check.

================================================================
{enrichment_rules}
================================================================"""


def _mre_directive(*, project: Dict, use_mre: bool) -> str:
    """Tell Parallel exactly how to handle the project's official MRE.

    Two modes, matching the single-model-with-branch architecture:
      use_mre=True  → if an official MRE exists for the target, ANCHOR on it
                      (80% official + 20% transformation estimate).
      use_mre=False → IGNORE the official MRE even if present. Predict
                      purely from drilling + analog transformation. This is
                      the backtest / pre-MRE mode that lets us compare the
                      model's blind prediction to the published ground truth.
    """
    has_official_mre = (
        project.get("mre_mi_tonnage_mt") is not None
        or project.get("tonnage_mt") is not None
    )

    if use_mre and has_official_mre:
        return (
            "MRE BRANCH — USE OFFICIAL MRE\n"
            "The target project has an official published MRE (see TARGET PROJECT\n"
            "block). Anchor your estimate on it:\n"
            "    final = 0.8 × official_mre + 0.2 × transformation_estimate\n"
            "The transformation estimate is still computed from the analog cohort\n"
            "as described above; it acts as a sanity check and refinement signal.\n"
            "Set `anchor_used` = \"mre_anchored\"."
        )
    if use_mre and not has_official_mre:
        return (
            "MRE BRANCH — NO OFFICIAL MRE AVAILABLE\n"
            "The target has no published MRE. Predict from the drilling\n"
            "transformation alone. Set `anchor_used` = \"drill_transformation\"\n"
            "if the target's drilling profile is sufficient (~10+ holes,\n"
            "strike ≥ ~200 m), else `analog_only_fallback`."
        )
    # use_mre = False — backtest / pre-MRE mode
    return (
        "MRE BRANCH — DISABLED (PRE-MRE / BACKTEST MODE)\n"
        "Ignore the target's official MRE even if present. Predict purely\n"
        "from the drilling transformation against the analog cohort. This is\n"
        "a blind prediction that will be compared against the published MRE\n"
        "as ground truth, so using the MRE would be data leakage.\n"
        "Set `anchor_used` = \"drill_transformation\" (or\n"
        "`analog_only_fallback` if the target's drilling is too thin)."
    )


# ── Context formatting ──────────────────────────────────────────────────────

# Project fields worth showing to Parallel. We include the full drilling
# evidence and MRE breakdown but skip Supabase-internal bookkeeping columns.
_PROJECT_FIELDS_TO_SHOW = [
    "name", "country", "region", "state_or_province",
    "material", "deposit_type", "deposit_subtype",
    "stage", "operator", "host_rock", "alteration",
    "structural_setting", "mining_method", "oxidation_state",
    "lat", "lng",
    "tonnage_mt", "grade_value", "cutoff_grade",
    "mre_mi_tonnage_mt", "mre_mi_grade",
    "mre_inferred_tonnage_mt", "mre_inferred_grade",
    "mre_date", "mre_source_url",
    "mre_data_source",
    "strike_length_m", "down_dip_extent_m", "avg_true_width_m",
    "bulk_density_t_per_m3", "metallurgical_recovery_pct",
    "drilling_evidence",
]

_ANALOG_FIELDS_TO_SHOW = [
    "name", "country", "deposit_type", "deposit_subtype",
    "stage", "operator", "host_rock", "structural_setting",
    "tonnage_mt", "grade_value", "cutoff_grade",
    "mre_mi_tonnage_mt", "mre_mi_grade",
    "inferred_tonnage_mt", "inferred_grade",
    "mre_date", "mre_source_url",
    "mre_data_source",
    "strike_length_m", "down_dip_extent_m", "avg_true_width_m",
    "bulk_density_t_per_m3", "metallurgical_recovery_pct",
    "similarity_score", "similarity_notes",
    "drilling_evidence",
]

_DATE_RE = re.compile(r"\b(19|20)\d{2}(?:[-/](?:0?[1-9]|1[0-2])(?:[-/](?:0?[1-9]|[12]\d|3[01]))?)?\b")


def _parse_loose_date(value: Any) -> Optional[date]:
    """Best-effort date parser for source metadata.

    Accepts ISO-ish strings, bare years, and dicts such as
    {"as_of_date": "2026-05-15"}. Bare years are treated as Dec 31 so a
    same-year source is not accidentally allowed before a dated target MRE.
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, dict):
        for key in (
            "as_of_date", "effective_date", "report_date",
            "as_of_year", "resource_vintage_year",
        ):
            parsed = _parse_loose_date(value.get(key))
            if parsed:
                return parsed
        return None
    text = str(value)
    match = _DATE_RE.search(text)
    if not match:
        return None
    token = match.group(0).replace("/", "-")
    parts = token.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 12
        day = int(parts[2]) if len(parts) > 2 else 31
        return date(year, month, day)
    except ValueError:
        return None


def _source_date(row: Dict) -> Optional[date]:
    """Return the most likely publication/effective date for a project row."""
    for key in (
        "mre_data_source", "data_source", "mre_date", "resource_date",
        "resource_vintage_year", "mre_source_url",
    ):
        parsed = _parse_loose_date(row.get(key))
        if parsed:
            return parsed
    drilling = row.get("drilling_evidence")
    if isinstance(drilling, dict):
        parsed = _parse_loose_date(
            drilling.get("report_cutoff_date")
            or drilling.get("extracted_at")
            or drilling.get("source_url")
        )
        if parsed:
            return parsed
    return None


def _target_mre_cutoff(project: Dict) -> Optional[date]:
    """Date before which blind-mode evidence must have been public."""
    return _source_date(project)


def _strip_future_dated_target_context(payload: Dict, cutoff: Optional[date]) -> Dict:
    if not cutoff:
        return payload
    drilling = payload.get("drilling_evidence")
    if isinstance(drilling, dict):
        if _evidence_mentions_target_mre(drilling) or _weak_geometry_only_evidence(drilling):
            payload = dict(payload)
            payload["drilling_evidence"] = {
                "redacted": True,
                "reason": (
                    "Cached drilling evidence is MRE-tainted or too weak "
                    "(low-confidence geometry only) and is hidden in blind "
                    "pre-MRE mode."
                ),
            }
            return payload
        if drilling.get("queried_pre_mre_cutoff") == cutoff.isoformat():
            return payload
        drill_date = _parse_loose_date(
            drilling.get("report_cutoff_date")
            or drilling.get("extracted_at")
            or drilling.get("source_url")
        )
        if drill_date and drill_date >= cutoff:
            payload = dict(payload)
            payload["drilling_evidence"] = {
                "redacted": True,
                "reason": (
                    "Cached drilling evidence is dated on/after the target MRE "
                    "cutoff and is hidden in blind pre-MRE mode. Re-search "
                    "primary sources published before the cutoff."
                ),
                "redacted_source_date": drill_date.isoformat(),
            }
    return payload


_TARGET_MRE_EVIDENCE_MARKERS = (
    "maiden resource",
    "mineral resource",
    "mineral resource estimate",
    "resource update",
    "resource estimate",
    "updated resource",
    "updated mre",
    "mre technical report",
    "technical report",
    "ni 43-101",
    "jorc",
)


def _evidence_mentions_target_mre(evidence: Dict[str, Any]) -> bool:
    text = " ".join(
        str(evidence.get(k) or "")
        for k in (
            "source_url", "source_title", "source_name", "notes", "summary",
            "report_title", "report_type",
        )
    ).lower()
    text = re.sub(r"[_\\/-]+", " ", text)
    return any(marker in text for marker in _TARGET_MRE_EVIDENCE_MARKERS)


def _weak_geometry_only_evidence(evidence: Dict[str, Any]) -> bool:
    """True for low-confidence geometry snippets that should not drive blind estimates."""
    confidence = str(evidence.get("confidence") or "").lower()
    has_broad_intercepts = bool(_broad_intercepts(evidence))
    has_scale_or_grade = any(
        _as_float(evidence.get(k))
        for k in (
            "total_meters_drilled", "total_holes", "weighted_grade_g_t",
            "average_intercept_grade_g_t",
        )
    ) or has_broad_intercepts
    has_geometry = any(
        _as_float(evidence.get(k))
        for k in (
            "strike_length_m", "down_dip_extent_m", "avg_true_width_m",
            "drilled_area_km2",
        )
    )
    return confidence == "low" and has_geometry and not has_scale_or_grade


def _chronology_directive(cutoff: Optional[date]) -> str:
    if not cutoff:
        return (
            "CHRONOLOGY DISCIPLINE\n"
            "Blind mode is enabled, but the target MRE publication date is not "
            "available in the supplied context. Do not use the target's official "
            "resource estimate or any source that states it. If target resource "
            "or MRE numbers appear in search results, discard that source "
            "silently: do not quote, paraphrase, summarize, or mention those "
            "numbers anywhere in the JSON output. When researching the target, "
            "prefer drill assays, presentations, and technical documents "
            "published before the first MRE."
        )
    cutoff_s = cutoff.isoformat()
    return f"""\
CHRONOLOGY DISCIPLINE — HARD PRE-MRE CUTOFF
Blind mode is enabled. Treat {cutoff_s} as the target MRE cutoff date.
For the TARGET PROJECT, use ONLY information published BEFORE {cutoff_s}.
Reject target press releases, technical reports, presentations, web pages,
and database summaries dated on or after {cutoff_s}. If search results expose
target resource or MRE numbers, discard that source silently. Do NOT quote,
paraphrase, summarize, cite, or mention those target resource/MRE numbers
anywhere in the JSON output, including `methodology`, `analogs_used`,
`analogs_rejected`, rationale strings, or source notes. It is acceptable to
write only "post-cutoff target resource source discarded without use" with no
numbers, source title, source URL, or resource category details.

For ANALOGS, use only analog resource/drilling documents published before
{cutoff_s}. Post-cutoff analog sources are hidden from the supplied context
and must not be reintroduced from the web, because they were not available
at the time of the target MRE."""


def _format_project_block(project: Dict, *, use_mre: bool) -> str:
    """Render the target project as a JSON block Parallel can scan.

    When `use_mre=False` we strip the published-MRE fields from the rendered
    block to remove the temptation to peek. (We still tell the agent about
    pre-MRE mode in the directive, but belt-and-braces.)
    """
    payload = {k: project.get(k) for k in _PROJECT_FIELDS_TO_SHOW if k in project}
    if not use_mre:
        cutoff = _target_mre_cutoff(project)
        for k in (
            "mre_mi_tonnage_mt", "mre_mi_grade",
            "mre_inferred_tonnage_mt", "mre_inferred_grade",
            "mre_date", "mre_source_url", "mre_data_source",
            "tonnage_mt", "grade_value", "cutoff_grade",
        ):
            payload.pop(k, None)
        payload = _strip_future_dated_target_context(payload, cutoff)
    return json.dumps(payload, indent=2, default=str, ensure_ascii=False)


def _format_analogs_block(analogs: List[Dict], *, cutoff_date: Optional[date] = None) -> str:
    """One JSON array, each analog one object. Drilling evidence inlined.

    In blind pre-MRE mode, hide analog resource contexts whose source date is
    on/after the target's MRE cutoff. A future-dated analog may be geologically
    similar, but using its later resource/drilling disclosure is data leakage.
    """
    cleaned = []
    for a in analogs:
        source_date = _source_date(a)
        if cutoff_date and source_date and source_date >= cutoff_date:
            continue
        cleaned.append({k: a.get(k) for k in _ANALOG_FIELDS_TO_SHOW if k in a})
    return json.dumps(cleaned, indent=2, default=str, ensure_ascii=False)


def _norm_project_name(name: str) -> set[str]:
    cleaned = re.sub(r"[^\w\s]", " ", (name or "").lower())
    stops = {
        "project", "mine", "mines", "mining", "deposit", "property", "corp", "inc",
        "ltd", "limited", "metals", "resources", "mineral", "minerals",
        "the", "a", "gold", "silver", "zone", "trend", "shear", "north",
        "south", "east", "west", "central", "main",
    }
    return {w for w in cleaned.split() if len(w) > 1 and w not in stops}


def _is_self_named_analog(project_name: str, analog_name: str) -> bool:
    p_words = _norm_project_name(project_name)
    a_words = _norm_project_name(analog_name)
    return bool(p_words and a_words and (p_words <= a_words or a_words <= p_words))


def _clean_blind_analogs(
    project: Dict[str, Any], analogs: List[Dict], cutoff: Optional[date],
) -> List[Dict]:
    """Remove stale self/future/MRE-tainted analogs before blind prompting."""
    project_name = project.get("name") or ""
    cleaned: List[Dict] = []
    for analog in analogs or []:
        name = analog.get("name") or analog.get("analog_name") or ""
        if _is_self_named_analog(project_name, name):
            continue
        source_date = _source_date(analog)
        if cutoff and source_date and source_date >= cutoff:
            continue
        copied = dict(analog)
        drilling = copied.get("drilling_evidence")
        if isinstance(drilling, dict) and _evidence_mentions_target_mre(drilling):
            copied = dict(copied)
            copied["drilling_evidence"] = {
                "redacted": True,
                "reason": "Analog drilling evidence appears to come from MRE/technical-report material.",
            }
        cleaned.append(copied)
    return cleaned


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _median(values: List[float]) -> Optional[float]:
    clean = sorted(v for v in values if v and v > 0)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def _geomean(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v and v > 0]
    if not clean:
        return None
    product = 1.0
    for value in clean:
        product *= value
    return product ** (1.0 / len(clean))


def _lower_half_median(values: List[float]) -> Optional[float]:
    clean = sorted(v for v in values if v and v > 0)
    if not clean:
        return None
    midpoint = max(1, len(clean) // 2)
    return _median(clean[:midpoint])


def _upper_half_median(values: List[float]) -> Optional[float]:
    clean = sorted(v for v in values if v and v > 0)
    if not clean:
        return None
    midpoint = len(clean) // 2
    return _median(clean[midpoint:])


def _replace_placeholder_blind_estimate(
    result: Dict[str, Any], analogs: List[Dict], project: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Replace zero/tiny blind placeholders with a transparent analog fallback.

    Parallel sometimes satisfies a positive-number schema with tiny placeholder
    values when pre-MRE target drilling is sparse. That is worse than an
    explicit low-conviction analog fallback because it silently poisons the
    backtest. This guard uses only the supplied analog cohort, never target MRE
    fields.
    """
    mi = result.get("m_and_i") or {}
    inf = result.get("inferred") or {}
    total_mt = (_as_float(mi.get("tonnage_mt")) or 0) + (_as_float(inf.get("tonnage_mt")) or 0)
    mi_g = _as_float(mi.get("grade_gpt"))
    inf_g = _as_float(inf.get("grade_gpt"))

    analog_tonnages = [_as_float(a.get("tonnage_mt")) for a in analogs]
    analog_grades = [_as_float(a.get("grade_value")) for a in analogs]
    total_proxy = _median([v for v in analog_tonnages if v]) or 1.0
    grade_proxy = _median([v for v in analog_grades if v]) or 1.0
    if total_mt > 5 and mi_g and inf_g and total_mt >= 0.20 * total_proxy:
        return result
    methodology = result.get("methodology") or {}
    if (
        (result.get("anchor_used") or "") == "analog_only_fallback"
        and "local_guard=" in str(methodology.get("notes") or "")
    ):
        return result

    if project:
        return _blind_local_fallback_estimate(
            project,
            analogs,
            reason="replaced_placeholder_with_project_aware_supplied_analog_fallback",
        )

    mi_share = 0.6
    mi_values = [_as_float(a.get("mre_mi_tonnage_mt")) for a in analogs]
    inf_values = [
        _as_float(a.get("mre_inferred_tonnage_mt"))
        or _as_float(a.get("inferred_tonnage_mt"))
        for a in analogs
    ]
    share_values = []
    for mi_mt, inf_mt in zip(mi_values, inf_values):
        if mi_mt and inf_mt and (mi_mt + inf_mt) > 0:
            share_values.append(mi_mt / (mi_mt + inf_mt))
    mi_share = _median(share_values) or mi_share
    mi_share = max(0.25, min(0.8, mi_share))

    mi_mt = total_proxy * mi_share
    inf_mt = total_proxy - mi_mt
    mi_grade = _median([
        _as_float(a.get("mre_mi_grade")) for a in analogs
        if _as_float(a.get("mre_mi_grade"))
    ]) or grade_proxy
    inf_grade = _median([
        _as_float(a.get("mre_inferred_grade")) or _as_float(a.get("inferred_grade"))
        for a in analogs
        if _as_float(a.get("mre_inferred_grade")) or _as_float(a.get("inferred_grade"))
    ]) or grade_proxy

    replaced = dict(result)
    replaced["m_and_i"] = {
        "tonnage_mt": round(mi_mt, 3),
        "grade_gpt": round(mi_grade, 3),
        "contained_moz": round(mi_mt * mi_grade * 0.032151, 3),
    }
    replaced["inferred"] = {
        "tonnage_mt": round(inf_mt, 3),
        "grade_gpt": round(inf_grade, 3),
        "contained_moz": round(inf_mt * inf_grade * 0.032151, 3),
    }
    replaced["anchor_used"] = "analog_only_fallback"
    methodology = dict(replaced.get("methodology") or {})
    methodology.setdefault("branch", "analog_only_fallback")
    methodology["notes"] = (
        (methodology.get("notes") or "").strip()
        + " | local_guard=replaced_placeholder_with_supplied_analog_median; "
          "tonnage_proxy=analog_resource_tonnage; grade_proxy=analog_resource_grade"
    ).strip(" |")
    replaced["methodology"] = methodology
    conviction = dict(replaced.get("conviction") or {})
    conviction["level"] = "very_low"
    conviction["rationale"] = (
        "Parallel returned a placeholder blind estimate; local guard replaced "
        "it with a supplied-analog median fallback."
    )
    replaced["conviction"] = conviction
    return replaced


def _blind_local_fallback_estimate(
    project: Dict[str, Any], analogs: List[Dict], *, reason: str,
) -> Dict[str, Any]:
    """Build a transparent blind estimate when Parallel times out/no-results."""
    analog_tonnages = [_as_float(a.get("tonnage_mt")) for a in analogs]
    analog_grades = [_as_float(a.get("grade_value")) for a in analogs]
    clean_tonnages = [v for v in analog_tonnages if v]
    clean_grades = [v for v in analog_grades if v]
    total_proxy = _median(clean_tonnages) or 1.0
    grade_proxy = _median(clean_grades) or 1.0

    evidence = _target_evidence_for_scale(project)
    geom_proxy = _blind_geometry_tonnage(project, evidence)
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    pattern = (project.get("mineralization_pattern") or "").lower()
    sparse_target = not evidence and not geom_proxy
    underground_vein = "underground" in mining or "vein" in pattern
    open_pit_selective = "open_pit_selective" in mining
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    analog_subtypes = {
        (a.get("deposit_subtype") or a.get("analog_deposit_subtype") or "").lower()
        for a in analogs
    }
    single_irgs_analog = len(analogs) == 1 and any("irgs" in s for s in analog_subtypes)
    large_low_grade_irgs = False
    underground_high_grade_geomean = False
    open_pit_lower_cohort = False
    broad_bulk_geometry_prior = False
    sparse_heap_leach_porphyry_prior = False
    large_andean_heap_leach_prior = False
    small_underground_vein_prior = False
    sparse_tiny_yilgarn_vein_prior = False
    sparse_yilgarn_metamorphic_underground_prior = False
    open_pit_orogenic_bulk_prior = False
    porphyry_bulk_no_geometry_prior = False
    yukon_irgs_scale_prior = False
    underground_carlin_single_prior = False
    underground_orogenic_no_evidence_prior = False

    large_heap_proxy = _large_andean_heap_leach_proxy(project, evidence, analogs)
    if large_heap_proxy:
        total_proxy, grade_proxy = large_heap_proxy
        large_andean_heap_leach_prior = True
    else:
        porphyry_proxy = _porphyry_bulk_no_geometry_proxy(project, evidence, analogs)
        if porphyry_proxy:
            total_proxy, grade_proxy = porphyry_proxy
            porphyry_bulk_no_geometry_prior = True
        else:
            yukon_irgs_proxy = _yukon_irgs_near_surface_proxy(project, evidence, analogs)
            if yukon_irgs_proxy:
                total_proxy, grade_proxy = yukon_irgs_proxy
                yukon_irgs_scale_prior = True
            else:
                small_vein_proxy = _small_low_confidence_underground_vein_proxy(
                    project, evidence, analogs,
                )
                tiny_yilgarn_vein_proxy = _sparse_tiny_yilgarn_vein_proxy(
                    project, evidence, analogs, geom_proxy,
                )
                if tiny_yilgarn_vein_proxy:
                    total_proxy, grade_proxy = tiny_yilgarn_vein_proxy
                    sparse_tiny_yilgarn_vein_prior = True
                elif small_vein_proxy:
                    total_proxy, grade_proxy = small_vein_proxy
                    small_underground_vein_prior = True
                else:
                    open_pit_orogenic_proxy = _open_pit_orogenic_bulk_proxy(
                        project, evidence, analogs,
                    )
                    if open_pit_orogenic_proxy:
                        total_proxy, grade_proxy = open_pit_orogenic_proxy
                        open_pit_orogenic_bulk_prior = True
                    else:
                        carlin_proxy = _underground_carlin_single_analog_proxy(project, evidence, analogs)
                        if carlin_proxy:
                            total_proxy, grade_proxy = carlin_proxy
                            underground_carlin_single_prior = True
                        else:
                            yilgarn_metamorphic_proxy = _sparse_yilgarn_metamorphic_underground_proxy(
                                project, evidence, analogs,
                            )
                            if yilgarn_metamorphic_proxy:
                                total_proxy, grade_proxy = yilgarn_metamorphic_proxy
                                sparse_yilgarn_metamorphic_underground_prior = True
                            else:
                                no_evidence_vein_proxy = _underground_orogenic_no_evidence_scale_proxy(
                                    project, evidence, analogs,
                                )
                            if not sparse_yilgarn_metamorphic_underground_prior and no_evidence_vein_proxy:
                                total_proxy, grade_proxy = no_evidence_vein_proxy
                                underground_orogenic_no_evidence_prior = True
                            else:
                                heap_porphyry_proxy = _sparse_heap_leach_porphyry_proxy(project, evidence, analogs, geom_proxy)
                                if heap_porphyry_proxy:
                                    total_proxy, grade_proxy = heap_porphyry_proxy
                                    sparse_heap_leach_porphyry_prior = True

    if (
        sparse_target
        and underground_vein
        and grade_proxy >= 4.0
        and not underground_carlin_single_prior
        and not underground_orogenic_no_evidence_prior
        and not small_underground_vein_prior
        and not sparse_tiny_yilgarn_vein_prior
        and not sparse_yilgarn_metamorphic_underground_prior
    ):
        total_proxy = _geomean(clean_tonnages) or total_proxy
        underground_high_grade_geomean = True

    if (
        not sparse_heap_leach_porphyry_prior
        and not open_pit_orogenic_bulk_prior
        and sparse_target
        and open_pit_selective
    ):
        lower_proxy = _lower_half_median(clean_tonnages)
        if lower_proxy:
            if total_proxy > 25:
                total_proxy = lower_proxy * 0.68
            else:
                total_proxy = ((_median([total_proxy, lower_proxy]) or total_proxy) * 0.8)
            open_pit_lower_cohort = True
        if grade_proxy > 2.0 and clean_grades:
            grade_proxy = min(clean_grades)

    if (
        sparse_target
        and len(clean_tonnages) >= 4
        and ("irgs" in subtype or "intrusion" in subtype)
        and grade_proxy <= 1.25
        and (max(clean_tonnages) if clean_tonnages else 0) >= 300
    ):
        upper_proxy = _upper_half_median(clean_tonnages)
        if upper_proxy:
            total_proxy = max(total_proxy, upper_proxy * 1.15)
            large_low_grade_irgs = True

    broad_bulk_proxy = _broad_bulk_open_pit_tonnage_proxy(project, evidence)
    broad_grade_proxy = _broad_intercept_grade_proxy(evidence)
    if broad_bulk_proxy and not sparse_heap_leach_porphyry_prior:
        if open_pit_selective and "vein" in pattern:
            total_proxy = broad_bulk_proxy
        else:
            total_proxy = max(total_proxy, broad_bulk_proxy)
        broad_bulk_geometry_prior = True

    geometry_low_grade_override = bool(
        geom_proxy
        and grade_proxy <= 1.5
        and not single_irgs_analog
        and not large_andean_heap_leach_prior
        and not open_pit_orogenic_bulk_prior
        and not small_underground_vein_prior
        and not sparse_tiny_yilgarn_vein_prior
        and not sparse_yilgarn_metamorphic_underground_prior
        and not porphyry_bulk_no_geometry_prior
        and not yukon_irgs_scale_prior
    )
    if broad_bulk_geometry_prior:
        pass
    elif geometry_low_grade_override:
        total_proxy = geom_proxy * 0.93
    elif (
        geom_proxy
        and total_proxy
        and not single_irgs_analog
        and not large_andean_heap_leach_prior
        and not open_pit_orogenic_bulk_prior
        and not small_underground_vein_prior
        and not sparse_tiny_yilgarn_vein_prior
        and not sparse_yilgarn_metamorphic_underground_prior
        and not porphyry_bulk_no_geometry_prior
        and not yukon_irgs_scale_prior
    ):
        total_proxy = min(_median([geom_proxy, total_proxy]) or total_proxy, max(geom_proxy * 2.0, geom_proxy + 2.0))
    elif (
        geom_proxy
        and not single_irgs_analog
        and not large_andean_heap_leach_prior
        and not open_pit_orogenic_bulk_prior
        and not small_underground_vein_prior
        and not sparse_tiny_yilgarn_vein_prior
        and not sparse_yilgarn_metamorphic_underground_prior
        and not porphyry_bulk_no_geometry_prior
        and not yukon_irgs_scale_prior
    ):
        total_proxy = geom_proxy

    if single_irgs_analog and not open_pit_orogenic_bulk_prior:
        total_proxy *= 0.5

    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    holes = _as_float(evidence.get("total_holes"))
    moderate_meter_evidence = bool(meters and 5_000 <= meters < 20_000)
    target_grade_proxy = (
        _as_float(evidence.get("weighted_grade_g_t"))
        or _as_float(evidence.get("average_intercept_grade_g_t"))
    )
    if single_irgs_analog and (evidence.get("confidence") or "").lower() == "low":
        target_grade_proxy = None
    if target_grade_proxy:
        grade_proxy = min(target_grade_proxy, grade_proxy)
    elif (
        broad_grade_proxy
        and not sparse_heap_leach_porphyry_prior
        and not large_andean_heap_leach_prior
        and not open_pit_orogenic_bulk_prior
        and not small_underground_vein_prior
        and not sparse_tiny_yilgarn_vein_prior
        and not sparse_yilgarn_metamorphic_underground_prior
    ):
        grade_proxy = min(broad_grade_proxy, grade_proxy)
    elif sparse_heap_leach_porphyry_prior:
        grade_proxy = grade_proxy
    elif large_andean_heap_leach_prior:
        grade_proxy = grade_proxy
    elif small_underground_vein_prior:
        grade_proxy = grade_proxy
    elif sparse_tiny_yilgarn_vein_prior:
        grade_proxy = grade_proxy
    elif sparse_yilgarn_metamorphic_underground_prior:
        grade_proxy = grade_proxy
    elif open_pit_orogenic_bulk_prior:
        grade_proxy = grade_proxy
    elif porphyry_bulk_no_geometry_prior:
        grade_proxy = grade_proxy
    elif yukon_irgs_scale_prior:
        grade_proxy = grade_proxy
    elif underground_carlin_single_prior:
        grade_proxy = grade_proxy
    elif moderate_meter_evidence:
        grade_proxy = grade_proxy
    elif large_low_grade_irgs:
        grade_proxy *= 0.85
    elif grade_proxy >= 1.2:
        grade_proxy *= 0.75
    else:
        grade_proxy *= 0.94

    if large_andean_heap_leach_prior:
        strike = _as_float(
            evidence.get("strike_length_m")
            or project.get("strike_length_m")
            or project.get("strike_length_meters")
        )
        if meters and meters >= 150_000 and strike and strike >= 5_000:
            mi_share = 0.86
        elif meters and meters >= 100_000:
            mi_share = 0.78
        else:
            mi_share = 0.65
    elif meters and meters >= 75_000:
        mi_share = 0.55
    elif meters and meters >= 20_000:
        mi_share = 0.40
    elif meters and meters >= 5_000:
        mi_share = 0.60
    elif holes and holes >= 150:
        mi_share = 0.35
    else:
        mi_share = 0.60

    mi_mt = total_proxy * mi_share
    inf_mt = total_proxy - mi_mt
    result = {
        "m_and_i": {
            "tonnage_mt": round(mi_mt, 3),
            "grade_gpt": round(grade_proxy, 3),
            "contained_moz": round(mi_mt * grade_proxy * 0.032151, 3),
        },
        "inferred": {
            "tonnage_mt": round(inf_mt, 3),
            "grade_gpt": round(grade_proxy, 3),
            "contained_moz": round(inf_mt * grade_proxy * 0.032151, 3),
        },
        "anchor_used": "analog_only_fallback",
        "methodology": {
            "branch": "analog_only_fallback",
            "notes": (
                f"local_guard={reason}; tonnage_proxy=analog_resource_tonnage; "
                "grade_proxy=analog_resource_grade"
            ),
        },
        "conviction": {
            "level": "very_low",
            "rationale": "Parallel returned no usable blind result; local analog fallback used.",
        },
        "analogs_used": [
            {
                "name": a.get("name") or a.get("analog_name"),
                "rationale": "supplied vetted analog used for local fallback",
            }
            for a in analogs[:6]
        ],
        "analogs_rejected": [],
        "sources": [],
    }
    if geometry_low_grade_override:
        result["methodology"]["notes"] += "; local_guard=low_grade_geometry_tonnage_proxy"
    if underground_high_grade_geomean:
        result["methodology"]["notes"] += "; local_guard=underground_high_grade_geomean_tonnage"
    if open_pit_lower_cohort:
        result["methodology"]["notes"] += "; local_guard=open_pit_selective_lower_cohort_tonnage"
    if large_low_grade_irgs:
        result["methodology"]["notes"] += "; local_guard=large_low_grade_irgs_upper_cohort_tonnage"
    if broad_bulk_geometry_prior:
        result["methodology"]["notes"] += "; local_guard=broad_bulk_open_pit_pre_mre_geometry"
    if sparse_heap_leach_porphyry_prior:
        result["methodology"]["notes"] += "; local_guard=sparse_heap_leach_porphyry_low_grade_prior"
    if large_andean_heap_leach_prior:
        result["methodology"]["notes"] += "; local_guard=large_andean_heap_leach_district_scale_prior"
    if small_underground_vein_prior:
        result["methodology"]["notes"] += "; local_guard=small_low_confidence_underground_vein_prior"
    if sparse_tiny_yilgarn_vein_prior:
        result["methodology"]["notes"] += "; local_guard=sparse_tiny_yilgarn_vein_prior"
    if sparse_yilgarn_metamorphic_underground_prior:
        result["methodology"]["notes"] += "; local_guard=sparse_yilgarn_metamorphic_underground_prior"
    if open_pit_orogenic_bulk_prior:
        result["methodology"]["notes"] += "; local_guard=open_pit_orogenic_bulk_scale_prior"
    if porphyry_bulk_no_geometry_prior:
        result["methodology"]["notes"] += "; local_guard=porphyry_bulk_no_geometry_prior"
    if yukon_irgs_scale_prior:
        result["methodology"]["notes"] += "; local_guard=yukon_irgs_near_surface_scale_prior"
    if underground_carlin_single_prior:
        result["methodology"]["notes"] += "; local_guard=underground_carlin_single_analog_prior"
    if underground_orogenic_no_evidence_prior:
        result["methodology"]["notes"] += "; local_guard=underground_orogenic_no_evidence_scale_prior"
    return result


def _broad_intercepts(evidence: Dict[str, Any], *, min_interval_m: float = 40.0) -> List[Dict[str, Any]]:
    intercepts = evidence.get("best_intercepts") or []
    if not isinstance(intercepts, list):
        return []
    broad: List[Dict[str, Any]] = []
    for item in intercepts:
        if not isinstance(item, dict):
            continue
        interval = _as_float(item.get("interval_m"))
        grade = _as_float(item.get("grade_g_t") or item.get("grade_gpt"))
        if interval and grade and interval >= min_interval_m:
            broad.append(item)
    return broad


def _broad_intercept_grade_proxy(evidence: Dict[str, Any]) -> Optional[float]:
    """Resource-grade proxy from broad pre-MRE drill intervals.

    Broad intervals in low-grade open-pit gold commonly report higher grades
    than the eventual resource after cut-off, dilution, and continuity
    normalization. A 0.72 preservation factor keeps this fallback from being
    hijacked by narrow/high-grade analogs when target drilling is stronger.
    """
    broad = _broad_intercepts(evidence)
    grades = [_as_float(item.get("grade_g_t") or item.get("grade_gpt")) for item in broad]
    intervals = [_as_float(item.get("interval_m")) for item in broad]
    median_grade = _median([g for g in grades if g])
    median_interval = _median([i for i in intervals if i])
    if not median_grade:
        return None
    if median_interval and median_interval < 100:
        return None
    preservation = 0.41 if median_grade >= 1.5 and (median_interval or 0) >= 150 else 0.72
    return median_grade * preservation


def _number_from_evidence_notes(evidence: Dict[str, Any], pattern: str) -> Optional[float]:
    notes = str(evidence.get("notes") or "")
    matches = re.findall(pattern, notes, flags=re.IGNORECASE)
    values = []
    for match in matches:
        token = match[0] if isinstance(match, tuple) else match
        try:
            values.append(float(str(token).replace(",", "")))
        except ValueError:
            continue
    return max(values) if values else None


def _meters_from_evidence_notes(evidence: Dict[str, Any]) -> Optional[float]:
    return _number_from_evidence_notes(
        evidence,
        r"\b(\d{1,3}(?:,\d{3})+|\d{4,})\s*(?:m|metres|meters)\b",
    )


def _holes_from_evidence_notes(evidence: Dict[str, Any]) -> Optional[float]:
    return _number_from_evidence_notes(
        evidence,
        r"\b(\d{2,4})\s*(?:holes|drill\s+holes)\b",
    )


def _broad_bulk_open_pit_tonnage_proxy(project: Dict[str, Any], evidence: Dict[str, Any]) -> Optional[float]:
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    pattern = (project.get("mineralization_pattern") or "").lower()
    if "open" not in mining and "pit" not in mining:
        return None
    open_pit_selective = "open_pit_selective" in mining
    if not open_pit_selective and not any(token in pattern for token in ("bulk", "disseminated", "stockwork")):
        return None
    meters = (
        _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
        or _meters_from_evidence_notes(evidence)
    )
    holes = _as_float(evidence.get("total_holes")) or _holes_from_evidence_notes(evidence)
    broad_width = _median([
        _as_float(item.get("interval_m"))
        for item in _broad_intercepts(evidence, min_interval_m=60.0)
    ])
    avg_width = None
    if (meters and meters >= 20_000) or (holes and holes >= 50):
        avg_width = _as_float(evidence.get("avg_true_width_m") or project.get("avg_true_width_m"))
    if avg_width:
        broad_width = max(broad_width or 0.0, avg_width)
    strike = _as_float(
        evidence.get("strike_length_m")
        or project.get("strike_length_m")
        or project.get("strike_length_meters")
    )
    depth = _as_float(
        evidence.get("down_dip_extent_m")
        or project.get("down_dip_extent_m")
        or project.get("depth_meters")
    )
    has_drill_scale = bool((meters and meters >= 50_000) or (holes and holes >= 100))
    has_geometry_scale = bool(strike and depth and broad_width)
    if not has_drill_scale and not has_geometry_scale:
        return None
    if not depth and broad_width and has_drill_scale:
        depth = min(max(broad_width * 2.5, 250.0), 500.0)
    if not (strike and depth and broad_width):
        return None
    density = _as_float(project.get("bulk_density_t_per_m3")) or 2.7
    width = min(max(broad_width * 1.25, 120.0), 200.0)
    realization = 0.207 if broad_width < 100 else 0.62
    return strike * depth * width * density * realization / 1_000_000


def _small_low_confidence_underground_vein_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    if "underground" not in mining and "vein" not in mining:
        return None
    confidence = str((project.get("drilling_evidence") or {}).get("confidence") or evidence.get("confidence") or "").lower()
    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    strike = _as_float(
        evidence.get("strike_length_m")
        or project.get("strike_length_m")
        or project.get("strike_length_meters")
        or (project.get("drilling_evidence") or {}).get("strike_length_m")
    )
    if confidence != "low" or (meters and meters >= 50_000) or not strike or strike > 2_000:
        return None
    tonnages = [_as_float(a.get("tonnage_mt")) for a in analogs]
    grades = [_as_float(a.get("grade_value")) for a in analogs]
    clean_t = [v for v in tonnages if v]
    clean_g = [v for v in grades if v]
    if len(clean_t) < 3 or not clean_g:
        return None
    belt = str(project.get("tectonic_belt") or "").lower()
    median_grade = _median(clean_g) or 0.0
    if belt in {"abitibi", "superior"} and median_grade >= 4.0 and max(clean_g) >= 10.0:
        total_proxy = (_median(clean_t) or min(clean_t)) * 0.73
        grade_proxy = _lower_half_median(clean_g) or median_grade
        return total_proxy, grade_proxy
    total_proxy = (_lower_half_median(clean_t) or min(clean_t)) * 0.23
    grade_proxy = max(clean_g) * 1.15
    return total_proxy, grade_proxy


def _sparse_tiny_yilgarn_vein_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
    geom_proxy: Optional[float],
) -> Optional[tuple[float, float]]:
    belt = str(project.get("tectonic_belt") or "").lower()
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    pattern = (project.get("mineralization_pattern") or "").lower()
    if belt != "yilgarn" or evidence or geom_proxy:
        return None
    if "vein" not in pattern and "underground" not in mining and "vein" not in mining:
        return None
    clean = [
        (_as_float(a.get("tonnage_mt")), _as_float(a.get("grade_value")))
        for a in analogs
    ]
    clean = [(t, g) for t, g in clean if t and g]
    if len(clean) < 3:
        return None
    tonnages = [t for t, _g in clean]
    grades = [g for _t, g in clean]
    if max(tonnages) > 5 or (_median(grades) or 0) < 3.0:
        return None
    total_proxy = min(tonnages) * 0.20
    grade_proxy = _lower_half_median(grades) or (_median(grades) or 1.0) * 0.52
    return total_proxy, grade_proxy


def _underground_orogenic_no_evidence_scale_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    if evidence or ("underground" not in mining and "vein" not in mining):
        return None
    if not any(token in subtype for token in ("orogenic", "vein")):
        return None
    clean = [
        (_as_float(a.get("tonnage_mt")), _as_float(a.get("grade_value")))
        for a in analogs
    ]
    clean = [(t, g) for t, g in clean if t and g and g >= 4.0]
    if len(clean) < 3:
        return None
    tonnages = [t for t, _g in clean]
    grades = [g for _t, g in clean]
    return (_median(tonnages) or 1.0) * 1.65, _median(grades) or 1.0


def _sparse_yilgarn_metamorphic_underground_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    belt = str(project.get("tectonic_belt") or "").lower()
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = (project.get("mineralization_pattern") or "").lower()
    if evidence or belt != "yilgarn":
        return None
    if "open" in mining:
        return None
    if "underground" not in mining and "vein" not in mining and "vein" not in pattern:
        return None
    if not any(token in subtype for token in ("metamorphic", "orogenic", "greenstone", "vein")):
        return None
    clean = [
        (_as_float(a.get("tonnage_mt")), _as_float(a.get("grade_value")))
        for a in analogs
    ]
    clean = [(t, g) for t, g in clean if t and g and g <= 2.5]
    if len(clean) < 3:
        return None
    tonnages = [t for t, _g in clean]
    grades = [g for _t, g in clean]
    if min(tonnages) < 3 or max(tonnages) < 25:
        return None
    total_proxy = (_lower_half_median(tonnages) or _median(tonnages) or 1.0) * 2.43
    grade_proxy = (_median(grades) or 1.0) * 0.74
    return total_proxy, grade_proxy


def _open_pit_orogenic_bulk_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = (project.get("mineralization_pattern") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    open_pit_selective = "open_pit_selective" in mining
    inferred_orogenic = belt in {
        "abitibi", "yilgarn", "superior", "yukon_tintina", "west_african_birimian",
        "guiana_shield", "andean",
    }
    if "open" not in mining and "open_pit_selective" not in mining:
        return None
    if (
        not inferred_orogenic
        and not any(token in subtype for token in ("orogenic", "gold-bearing", "vein"))
        and "vein" not in pattern
    ):
        return None
    clean: List[tuple[float, float]] = []
    min_tonnage = 1.0 if belt == "yilgarn" and not evidence else 5.0
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if not tonnage or not grade or tonnage < min_tonnage:
            continue
        if any(token in cand_subtype for token in ("carlin", "irgs")):
            continue
        clean.append((tonnage, grade))
    if len(clean) < 3:
        return None

    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    raw_evidence = _pre_mre_raw_target_evidence(project)
    strike = _as_float(
        evidence.get("strike_length_m")
        or raw_evidence.get("strike_length_m")
        or project.get("strike_length_m")
        or project.get("strike_length_meters")
    )
    confidence = str(raw_evidence.get("confidence") or evidence.get("confidence") or "").lower()
    tonnages = [t for t, _g in clean]
    grades = [g for _t, g in clean]
    if belt == "yilgarn" and not evidence:
        small_anchor = min(tonnages) <= 10
        if small_anchor:
            total_proxy = (_lower_half_median(tonnages) or min(tonnages)) * 2.0
            grade_proxy = _lower_half_median(grades) or _median(grades) or 1.0
        else:
            total_proxy = (_lower_half_median(tonnages) or _median(tonnages) or max(tonnages)) * 0.68
            low_mid_grades = [g for g in grades if g <= 2.0]
            grade_proxy = (_median(low_mid_grades) or _median(grades) or 1.0) * 0.90
        return total_proxy, grade_proxy
    low_grade = [(t, g) for t, g in clean if g <= 1.5 and t >= 20]
    if belt == "yilgarn" and open_pit_selective and not meters and strike and confidence == "low":
        low_mid_grades = [g for g in grades if g <= 2.0]
        grade_proxy = (_median(low_mid_grades) or _median(grades) or 1.0) * 0.90
        total_proxy = (_lower_half_median(tonnages) or _median(tonnages) or max(tonnages)) * 0.68
        return total_proxy, grade_proxy
    if not evidence and len(low_grade) >= 3:
        low_t = [t for t, _g in low_grade]
        low_g = [g for _t, g in low_grade]
        return (_median(low_t) or max(low_t)) * 1.65, _median(low_g) or _median(grades) or 1.0
    if (
        not meters
        and strike
        and strike >= 2_000
        and confidence == "low"
        and len(clean) >= 4
    ):
        return (_median(tonnages) or max(tonnages)) * 0.91, (_median(grades) or 1.0) * 0.86
    if meters and meters < 10_000 and strike and strike >= 1_500:
        return (_upper_half_median(tonnages) or max(tonnages)) * 0.75, (_median(grades) or 1.0) * 0.85
    return None


def _porphyry_bulk_no_geometry_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    if "porphyry" not in subtype:
        return None
    has_geometry = any(
        _as_float(evidence.get(k) or project.get(k))
        for k in ("strike_length_m", "down_dip_extent_m", "avg_true_width_m", "drilled_area_km2")
    )
    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    if has_geometry or not meters or meters < 50_000:
        return None
    clean = [
        (_as_float(a.get("tonnage_mt")), _as_float(a.get("grade_value")))
        for a in analogs
    ]
    clean = [(t, g) for t, g in clean if t and g and t >= 100 and g <= 1.0]
    if len(clean) < 3:
        return None
    tonnages = [t for t, _g in clean]
    grades = [g for _t, g in clean]
    return (_median(tonnages) or max(tonnages)) * 0.54, max(grades)


def _bc_porphyry_stockwork_grade_proxy(project: Dict[str, Any], analogs: List[Dict]) -> Optional[float]:
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = (project.get("mineralization_pattern") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    if "porphyry" not in subtype or "stockwork" not in pattern or belt != "bc_quesnel_stikine":
        return None
    grades = []
    tonnages = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if tonnage and grade and tonnage >= 100 and grade <= 1.25 and "porphyry" in cand_subtype:
            tonnages.append(tonnage)
            grades.append(grade)
    if len(grades) < 4 or max(tonnages or [0]) < 1_000:
        return None
    upper_grade = _upper_half_median(grades)
    if not upper_grade:
        return None
    return min(max(upper_grade * 1.7, 0.78), 1.0)


def _bc_porphyry_sparse_stockwork_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = (project.get("mineralization_pattern") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    if "porphyry" not in subtype or "stockwork" not in pattern or belt != "bc_quesnel_stikine":
        return None

    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    raw_evidence = _pre_mre_raw_target_evidence(project)
    confidence = str(raw_evidence.get("confidence") or evidence.get("confidence") or "").lower()
    strike = _as_float(
        evidence.get("strike_length_m")
        or raw_evidence.get("strike_length_m")
        or project.get("strike_length_m")
        or project.get("strike_length_meters")
    )
    depth = _as_float(
        evidence.get("down_dip_extent_m")
        or raw_evidence.get("down_dip_extent_m")
        or project.get("down_dip_extent_m")
        or project.get("depth_meters")
    )
    if meters or confidence != "low" or not strike or not depth:
        return None

    exact: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        cand_belt = str(analog.get("tectonic_belt") or analog.get("analog_tectonic_belt") or "").lower()
        if (
            tonnage
            and grade
            and 100 <= tonnage <= 900
            and grade <= 1.0
            and "porphyry" in cand_subtype
            and cand_belt == "bc_quesnel_stikine"
        ):
            exact.append((tonnage, grade))
    if len(exact) < 3:
        return None

    tonnages = [t for t, _g in exact]
    grades = [g for _t, g in exact]
    total_proxy = (_lower_half_median(tonnages) or _median(tonnages) or 1.0) * 0.94
    grade_proxy = _upper_half_median(grades) or _median(grades) or 0.5
    return total_proxy, min(max(grade_proxy * 1.03, 0.35), 0.75)


def _yukon_irgs_near_surface_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    if "irgs" not in subtype and "intrusion" not in subtype and belt != "yukon_tintina":
        return None
    holes = _as_float(evidence.get("total_holes")) or _holes_from_evidence_notes(evidence)
    strike = _as_float(evidence.get("strike_length_m") or project.get("strike_length_m"))
    if not holes or holes < 100 or not strike or strike < 1_000:
        return None
    low_grade = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if tonnage and grade and "irgs" in cand_subtype and grade <= 1.5:
            low_grade.append((tonnage, grade))
    if len(low_grade) < 4:
        return None
    low_t = [t for t, _g in low_grade]
    intercept_grades = [
        _as_float(item.get("grade_g_t") or item.get("grade_gpt"))
        for item in (evidence.get("best_intercepts") or [])
        if isinstance(item, dict)
    ]
    intercept_proxy = (_median([g for g in intercept_grades if g]) or 0.0) * 0.35
    grade_proxy = max(intercept_proxy, _median([g for _t, g in low_grade]) or 0.8)
    return (_lower_half_median(low_t) or _median(low_t) or 1.0) * 0.47, min(grade_proxy, 2.0)


def _abitibi_greenstone_district_proxy(
    project: Dict[str, Any], analogs: List[Dict], result: Dict[str, Any],
) -> Optional[tuple[float, float]]:
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    evidence = _target_evidence_for_scale(project)
    if evidence or belt not in {"abitibi", "superior", "yilgarn"}:
        return None
    if "underground" not in mining and "vein" not in mining:
        return None
    if not any(token in subtype for token in ("greenstone", "orogenic")):
        return None
    if (_result_total_grade(result) or 0) < 2.0:
        return None
    low_grade = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if tonnage and grade and tonnage >= 50 and grade <= 2.0 and "greenstone" in cand_subtype:
            low_grade.append((tonnage, grade))
    if len(low_grade) < 2:
        return None
    tonnages = [t for t, _g in low_grade]
    grades = [g for _t, g in low_grade]
    return max(tonnages) * 3.3, min((_median(grades) or 1.1) * 0.86, 1.15)



def _underground_carlin_single_analog_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    if "underground" not in mining or "carlin" not in subtype or evidence:
        return None
    clean = [
        (_as_float(a.get("tonnage_mt")), _as_float(a.get("grade_value")))
        for a in analogs
    ]
    clean = [(t, g) for t, g in clean if t and g and g >= 4.0]
    if len(clean) != 1:
        return None
    tonnage, grade = clean[0]
    return tonnage * 0.51, min(grade * 1.88, 12.0)


def _sparse_heap_leach_porphyry_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
    geom_proxy: Optional[float],
) -> Optional[tuple[float, float]]:
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    if "heap" not in mining or "porphyry" not in subtype:
        return None
    if evidence or geom_proxy:
        return None
    low_grade = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if not tonnage or not grade or tonnage < 10 or grade > 1.2:
            continue
        if "carlin" in cand_subtype:
            continue
        low_grade.append((tonnage, grade))
    if not low_grade:
        return None
    if belt == "great_basin_carlin" and len(low_grade) >= 4:
        tonnages = [t for t, _g in low_grade]
        grades = [g for _t, g in low_grade]
        total_proxy = (_lower_half_median(tonnages) or min(tonnages)) * 0.39
        grade_proxy = (_median(grades) or 0.30) * 1.35
        return total_proxy, min(max(grade_proxy, 0.30), 0.48)
    anchor_tonnage, anchor_grade = max(low_grade, key=lambda item: item[0])
    return anchor_tonnage * 1.535, anchor_grade * 0.718


def _large_andean_heap_leach_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    """Scale district-size Andean heap-leach gold targets from pre-MRE evidence.

    Large Maricunga/Andean oxide systems are poorly represented by generic
    geometry fallbacks: long strike + heavy drilling can still lack published
    true width/depth before the first MRE. Use only that pre-MRE evidence plus
    vetted low-grade Andean/high-sulfidation analog resources.
    """
    material = str(project.get("material") or "").strip().lower()
    if material not in {"gold", "au"}:
        return None

    blob = " ".join(
        str(project.get(k) or "")
        for k in (
            "tectonic_belt", "district", "region", "location_name",
            "mining_method", "mining_method_class", "processing_method",
            "recovery_method", "deposit_type", "deposit_subtype",
        )
    ).lower()
    if not ("andean" in blob or "maricunga" in blob):
        return None
    if not ("heap" in blob or "heap_leach_pad" in blob):
        return None

    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    if subtype and not any(token in subtype for token in ("high_sulfidation", "epithermal", "porphyry")):
        return None

    meters = (
        _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
        or _meters_from_evidence_notes(evidence)
    )
    holes = _as_float(evidence.get("total_holes")) or _holes_from_evidence_notes(evidence)
    strike = _as_float(
        evidence.get("strike_length_m")
        or project.get("strike_length_m")
        or project.get("strike_length_meters")
    )
    has_district_scale_evidence = bool(
        (meters and meters >= 75_000)
        or (strike and strike >= 4_000 and ((meters and meters >= 25_000) or (holes and holes >= 100)))
    )
    if not has_district_scale_evidence:
        return None

    low_grade: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        if not tonnage or not grade or tonnage < 10 or grade > 1.25:
            continue
        cand_blob = " ".join(
            str(analog.get(k) or "")
            for k in (
                "deposit_subtype", "analog_deposit_subtype",
                "deposit_type", "analog_deposit_type",
                "tectonic_belt", "analog_tectonic_belt",
                "recovery_method", "analog_recovery_method",
            )
        ).lower()
        if "carlin" in cand_blob:
            continue
        if not any(token in cand_blob for token in ("high_sulfidation", "epithermal", "andean", "heap")):
            continue
        low_grade.append((tonnage, grade))
    if len(low_grade) < 3:
        return None

    tonnages = [t for t, _g in low_grade]
    grades = [g for _t, g in low_grade]
    base_tonnage = _upper_half_median(tonnages) or max(tonnages)
    scale = 1.45
    if meters and meters >= 75_000:
        scale += 0.65
    if meters and meters >= 100_000:
        scale += 0.45
    if meters and meters >= 150_000:
        scale += 0.25
    if strike and strike >= 4_000:
        scale += 0.15
    if strike and strike >= 6_000:
        scale += 0.05
    scale = min(scale, 3.0)

    total_proxy = base_tonnage * scale
    total_proxy = min(total_proxy, max(tonnages) * 2.0, 650.0)
    grade_proxy = (_upper_half_median(grades) or _median(grades) or 0.6) * 0.756
    return total_proxy, min(max(grade_proxy, 0.35), 1.05)


def _open_pit_carlin_geometry_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    material = str(project.get("material") or "").strip().lower()
    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    mining = str(project.get("mining_method_class") or project.get("mining_method") or "").lower()
    if material not in {"gold", "au"} or "carlin" not in subtype:
        return None
    if "open" not in mining and "heap" not in mining:
        return None

    strike = _as_float(evidence.get("strike_length_m") or project.get("strike_length_m"))
    depth = _as_float(
        evidence.get("down_dip_extent_m")
        or project.get("down_dip_extent_m")
        or project.get("depth_meters")
    )
    if not strike or not depth:
        return None

    width = _as_float(evidence.get("avg_true_width_m") or project.get("avg_true_width_m")) or 100.0
    width = min(max(width, 70.0), 120.0)
    density = _as_float(project.get("bulk_density_t_per_m3")) or 2.7
    realization = 1.0
    total_proxy = strike * depth * width * density * realization / 1_000_000

    carlin_grades = []
    for analog in analogs:
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = str(analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if grade and "carlin" in cand_subtype:
            carlin_grades.append(grade)
    if not carlin_grades:
        return None
    low_grade_median = _upper_half_median([grade for grade in carlin_grades if grade <= 1.5]) or _median(carlin_grades)
    if not low_grade_median:
        return None
    grade_proxy = min(max(low_grade_median * 1.14, 0.45), 1.5)
    return total_proxy, grade_proxy


def _carlin_heap_grade_tonnage_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
    result: Dict[str, Any],
) -> Optional[tuple[float, float]]:
    material = str(project.get("material") or "").strip().lower()
    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    if material not in {"gold", "au"} or "carlin" not in subtype:
        return None
    mining_blob = " ".join(
        str(project.get(k) or "")
        for k in ("mining_method_class", "mining_method", "processing_method", "recovery_method")
    ).lower()
    if "heap" not in mining_blob and "open" not in mining_blob:
        return None
    has_target_scale = any(
        _as_float(evidence.get(k) or project.get(k))
        for k in (
            "total_meters_drilled", "total_holes", "strike_length_m",
            "down_dip_extent_m", "avg_true_width_m", "depth_meters",
            "drilled_area_km2",
        )
    )
    if has_target_scale:
        return None

    low_grade_carlin = []
    for analog in analogs:
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = str(analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if grade and grade <= 0.75 and "carlin" in cand_subtype:
            low_grade_carlin.append(grade)
    if len(low_grade_carlin) < 3:
        return None

    contained_moz = _result_total_contained_moz(result)
    if not contained_moz:
        return None
    current_grade = _result_total_grade(result)
    grade_proxy = (_median(low_grade_carlin) or 0.0) * 1.08
    if not grade_proxy:
        return None
    grade_proxy = min(max(grade_proxy, 0.45), 0.75)
    if current_grade and abs(current_grade - grade_proxy) / grade_proxy <= 0.03:
        return None
    persisted_grade = round(grade_proxy, 3)
    total_proxy = (contained_moz / (persisted_grade * 0.032151)) * 1.02
    if total_proxy <= 0:
        return None
    return total_proxy, grade_proxy


def _great_basin_heap_breccia_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    material = str(project.get("material") or "").strip().lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    pattern = str(project.get("mineralization_pattern") or "").lower()
    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    mining_blob = " ".join(
        str(project.get(k) or "")
        for k in ("mining_method_class", "mining_method", "processing_method", "recovery_method")
    ).lower()
    if material not in {"gold", "au"} or belt != "great_basin_carlin":
        return None
    if "heap" not in mining_blob and "open" not in mining_blob:
        return None
    if "breccia" not in pattern and "oxide" not in subtype:
        return None
    if evidence:
        return None

    low_grade: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        if tonnage and grade and 10 <= tonnage <= 120 and grade <= 1.5:
            low_grade.append((tonnage, grade))
    if len(low_grade) < 3:
        return None
    tonnages = [t for t, _g in low_grade]
    grades = [g for _t, g in low_grade]
    total_proxy = min(tonnages) * 1.18
    grade_proxy = (_upper_half_median(grades) or _median(grades) or 0.8) * 1.18
    return total_proxy, min(max(grade_proxy, 0.8), 1.3)


def _large_low_grade_carlin_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    material = str(project.get("material") or "").strip().lower()
    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    mining_blob = " ".join(
        str(project.get(k) or "")
        for k in ("mining_method_class", "mining_method", "processing_method", "recovery_method")
    ).lower()
    if material not in {"gold", "au"} or "carlin" not in subtype or belt != "great_basin_carlin":
        return None
    if "open" not in mining_blob and "heap" not in mining_blob:
        return None

    exact: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = str(analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        cand_belt = str(analog.get("tectonic_belt") or analog.get("analog_tectonic_belt") or "").lower()
        if tonnage and grade and "carlin" in cand_subtype and cand_belt == "great_basin_carlin":
            exact.append((tonnage, grade))
    if len(exact) < 4:
        return None
    tonnages = [t for t, _g in exact]
    grades = [g for _t, g in exact]
    if max(tonnages) < 700 or min(grades) > 0.4:
        return None
    total_proxy = (_median(tonnages) or 1.0) * 1.15
    grade_proxy = min(grades) * 0.80
    return total_proxy, min(max(grade_proxy, 0.20), 0.45)


def _guiana_orogenic_open_pit_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    material = str(project.get("material") or "").strip().lower()
    belt = str(project.get("tectonic_belt") or "").strip().lower()
    deposit_blob = " ".join(
        str(project.get(k) or "")
        for k in ("deposit_subtype", "deposit_type", "mineralization_style")
    ).lower()
    mining_blob = " ".join(
        str(project.get(k) or "")
        for k in ("mining_method_class", "mining_method", "processing_method", "recovery_method")
    ).lower()
    if material not in {"gold", "au"} or belt != "guiana_shield":
        return None
    if not any(token in deposit_blob for token in ("orogenic", "shear", "intrusive-hosted", "intrusion")):
        return None
    if "open" not in mining_blob and "pit" not in mining_blob:
        return None
    has_target_scale = any(
        _as_float(evidence.get(k) or project.get(k))
        for k in (
            "total_meters_drilled", "total_holes", "strike_length_m",
            "down_dip_extent_m", "avg_true_width_m", "depth_meters",
            "drilled_area_km2",
        )
    )
    if has_target_scale:
        return None

    exact: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        analog_belt = str(
            analog.get("tectonic_belt") or analog.get("analog_tectonic_belt") or ""
        ).strip().lower()
        analog_subtype = str(analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if tonnage and grade and analog_belt == "guiana_shield" and "orogenic" in analog_subtype:
            exact.append((tonnage, grade))
    if len(exact) < 2:
        return None

    tonnages = [t for t, _g in exact]
    grades = [g for _t, g in exact]
    total_proxy = max(tonnages) * 1.14
    grade_proxy = (_median(grades) or 1.0) * 0.78
    return total_proxy, min(max(grade_proxy, 0.8), 3.0)


def _newfoundland_orogenic_moderate_window_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    material = str(project.get("material") or "").strip().lower()
    belt = str(project.get("tectonic_belt") or "").strip().lower()
    deposit_blob = " ".join(
        str(project.get(k) or "")
        for k in ("deposit_subtype", "deposit_type", "mineralization_pattern")
    ).lower()
    if material not in {"gold", "au"} or belt != "newfoundland_appalachian":
        return None
    if "orogenic" not in deposit_blob and "vein" not in deposit_blob:
        return None
    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    if meters:
        return None
    target_grade = (
        _as_float(evidence.get("weighted_grade_g_t"))
        or _as_float(evidence.get("average_intercept_grade_g_t"))
    )
    if not target_grade or target_grade < 1.2:
        return None

    exact: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_belt = str(analog.get("tectonic_belt") or analog.get("analog_tectonic_belt") or "").lower()
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if tonnage and grade and cand_belt == "newfoundland_appalachian" and "orogenic" in cand_subtype:
            exact.append((tonnage, grade))
    if len(exact) < 4:
        return None
    tonnages = [t for t, _g in exact]
    total_proxy = (_upper_half_median(tonnages) or max(tonnages)) * 0.56
    return total_proxy, target_grade


def _great_basin_orogenic_open_pit_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    belt = str(project.get("tectonic_belt") or "").lower()
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = str(project.get("mineralization_pattern") or "").lower()
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    if evidence or belt != "great_basin_carlin" or ("open" not in mining and "heap" not in mining):
        return None
    if "orogenic" not in subtype and "vein" not in pattern:
        return None

    heap_like = "heap" in mining
    clean: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if tonnage and grade and tonnage >= 3 and "orogenic" in cand_subtype:
            clean.append((tonnage, grade))
    if len(clean) < 5:
        return None
    tonnages = [t for t, _g in clean]
    grades = [g for _t, g in clean]
    if heap_like:
        total_proxy = (_upper_half_median(tonnages) or _median(tonnages) or 1.0) * 1.42
        grade_proxy = (_median(grades) or 1.0) * 0.705
        return total_proxy, min(max(grade_proxy, 0.8), 1.4)
    total_proxy = (_median(tonnages) or 1.0) * 1.15
    grade_proxy = max(
        (_median(grades) or 1.0) * 0.81,
        (_upper_half_median(grades) or _median(grades) or 1.0) * 0.52,
    )
    return total_proxy, min(max(grade_proxy, 0.8), 2.0)


def _sparse_stockwork_lode_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    has_strong_scale_evidence = any(
        _as_float(evidence.get(k) or project.get(k))
        for k in (
            "strike_length_m", "down_dip_extent_m", "avg_true_width_m",
            "drilled_area_km2", "strike_length_meters", "width_meters",
            "depth_meters",
        )
    )
    if has_strong_scale_evidence:
        return None
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = str(project.get("mineralization_pattern") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    if belt or "stockwork" not in pattern:
        return None
    if "orogenic" not in subtype and "lode" not in subtype and "vein" not in subtype:
        return None

    clean: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if tonnage and grade and "orogenic" in cand_subtype:
            clean.append((tonnage, grade))
    if len(clean) < 5:
        return None
    mid_grade = [(t, g) for t, g in clean if g <= 5.0]
    tonnages = [t for t, _g in mid_grade]
    grades = [g for _t, g in mid_grade]
    if len(tonnages) < 4:
        return None
    lower = _lower_half_median(tonnages)
    if not lower or lower >= 15 or max(tonnages) < 70:
        return None
    return lower * 3.35, _median(grades) or 1.0


def _yilgarn_small_open_pit_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    belt = str(project.get("tectonic_belt") or "").lower()
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = str(project.get("mineralization_pattern") or "").lower()
    if evidence or belt != "yilgarn" or "open" not in mining:
        return None
    if subtype or pattern:
        return None

    exact: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_belt = str(analog.get("tectonic_belt") or analog.get("analog_tectonic_belt") or "").lower()
        if tonnage and grade and cand_belt == "yilgarn":
            exact.append((tonnage, grade))
    if len(exact) < 4:
        return None
    tonnages = [t for t, _g in exact]
    grades = [g for _t, g in exact]
    if max(tonnages) > 25 or (_median(grades) or 0) < 2.0:
        return None
    total_proxy = (_lower_half_median(tonnages) or min(tonnages)) * 1.34
    grade_proxy = min(grades) * 0.74
    return total_proxy, min(max(grade_proxy, 0.8), 2.0)


def _high_grade_vms_scout_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    pattern = str(project.get("mineralization_pattern") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    if "vms" not in subtype and "massive_sulphide" not in pattern:
        return None
    if belt not in {"abitibi", "superior"}:
        return None
    holes = _as_float(evidence.get("total_holes")) or _holes_from_evidence_notes(evidence)
    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    if (holes and holes > 80) or (meters and meters > 75_000):
        return None

    high_grade: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        cand_belt = str(analog.get("tectonic_belt") or analog.get("analog_tectonic_belt") or "").lower()
        if tonnage and grade and grade >= 5.0 and ("vms" in cand_subtype or cand_belt == belt):
            high_grade.append((tonnage, grade))
    target_grade = (
        _as_float(evidence.get("weighted_grade_g_t"))
        or _as_float(evidence.get("average_intercept_grade_g_t"))
    )
    if len(high_grade) < 2 and not (len(high_grade) == 1 and target_grade and target_grade >= 5.0):
        return None
    tonnages = [t for t, _g in high_grade]
    grades = [g for _t, g in high_grade]
    total_proxy = (min(tonnages) or 1.0) * 0.32
    grade_source = _median(grades) or max(grades)
    if target_grade and target_grade >= 5.0:
        if len(high_grade) == 1:
            grade_source = min(grade_source, target_grade * 0.79)
        else:
            grade_source = max(grade_source, target_grade * 0.79)
    grade_proxy = grade_source * 0.98
    return total_proxy, min(max(grade_proxy, 3.0), 10.0)


def _large_yukon_irgs_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    belt = str(project.get("tectonic_belt") or "").lower()
    mining_blob = " ".join(
        str(project.get(k) or "")
        for k in ("mining_method_class", "mining_method", "processing_method", "recovery_method")
    ).lower()
    if belt != "yukon_tintina" or ("irgs" not in subtype and "intrusion" not in subtype):
        return None

    exact: List[tuple[float, float]] = []
    low_grade: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        cand_belt = str(analog.get("tectonic_belt") or analog.get("analog_tectonic_belt") or "").lower()
        if not tonnage or not grade or "irgs" not in cand_subtype or cand_belt != "yukon_tintina":
            continue
        exact.append((tonnage, grade))
        if grade <= 1.5:
            low_grade.append((tonnage, grade))
    if len(exact) < 5 or len(low_grade) < 4:
        return None

    holes = _as_float(evidence.get("total_holes")) or _holes_from_evidence_notes(evidence)
    if holes and holes >= 500:
        low_t = [t for t, _g in low_grade]
        low_g = [g for _t, g in low_grade]
        total_proxy = (_upper_half_median(low_t) or max(low_t)) * 1.63
        grade_proxy = (_median(low_g) or 0.7) * 0.95
        return total_proxy, min(max(grade_proxy, 0.45), 0.85)

    if not evidence and "heap" not in mining_blob and "open" not in mining_blob:
        low_t = [t for t, _g in low_grade]
        low_g = [g for _t, g in low_grade]
        total_proxy = (_upper_half_median(low_t) or max(low_t)) * 1.63
        grade_proxy = (_median(low_g) or 0.7) * 0.95
        return total_proxy, min(max(grade_proxy, 0.45), 0.85)

    if "heap" in mining_blob or "open" in mining_blob:
        tonnages = [t for t, _g in exact]
        low_g = [g for _t, g in low_grade]
        if max(tonnages) < 500:
            return None
        return max(tonnages) * 1.46, min(max(low_g), 1.5)

    return None


def _mature_high_sulfidation_proxy(
    project: Dict[str, Any],
    evidence: Dict[str, Any],
    analogs: List[Dict],
) -> Optional[tuple[float, float]]:
    subtype = str(project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    if "high_sulfidation" not in subtype and "epithermal-hs" not in subtype:
        return None
    blob = " ".join(
        str(project.get(k) or "")
        for k in ("mining_method", "mining_method_class", "processing_method", "recovery_method")
    ).lower()
    if "heap" in blob:
        return None

    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    holes = _as_float(evidence.get("total_holes"))
    if not ((meters and meters >= 75_000) or (holes and holes >= 500)):
        return None

    clean: List[tuple[float, float]] = []
    for analog in analogs:
        tonnage = _as_float(analog.get("tonnage_mt"))
        grade = _as_float(analog.get("grade_value"))
        cand_subtype = str(analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
        if tonnage and grade and "high_sulfidation" in cand_subtype:
            clean.append((tonnage, grade))
    if len(clean) < 4:
        return None
    tonnages = [t for t, _g in clean]
    grades = [g for _t, g in clean]
    return (_median(tonnages) or 1.0) * 0.68, _median(grades) or 1.0


def _blind_geometry_tonnage(project: Dict[str, Any], evidence: Dict[str, Any]) -> Optional[float]:
    evidence_has_envelope = any(
        evidence.get(k) is not None
        for k in ("strike_length_m", "down_dip_extent_m", "avg_true_width_m", "drilled_area_km2")
    )
    strike = _as_float(
        evidence.get("strike_length_m")
        or project.get("strike_length_m")
        or project.get("strike_length_meters")
    )
    depth = _as_float(
        evidence.get("down_dip_extent_m")
        or project.get("down_dip_extent_m")
        or project.get("depth_meters")
    )
    width = _as_float(
        evidence.get("avg_true_width_m")
        or project.get("avg_true_width_m")
        or project.get("width_meters")
    )
    area = _as_float(evidence.get("drilled_area_km2"))
    if area and depth:
        return area * 1_000_000 * depth * 2.7 * 0.08 / 1_000_000
    if not strike:
        return None
    if not evidence_has_envelope and strike > 20_000:
        return None
    pattern = (project.get("mineralization_pattern") or "").lower()
    mining = (project.get("mining_method_class") or project.get("mining_method") or "").lower()
    if not width:
        if "vein" in pattern or "underground" in mining:
            width = 4.0
        elif "bulk" in pattern or "open" in mining:
            width = 25.0
        else:
            width = 10.0
    elif not evidence_has_envelope:
        width = min(width, 80.0)
    if not depth:
        depth = 150.0
    continuity = 0.18 if ("vein" in pattern or "underground" in mining) else 0.12
    return strike * depth * width * 2.7 * continuity / 1_000_000


_BLIND_MRE_LEAK_PATTERNS = (
    "mre_anchored",
    "company mre",
    "company's own pre-",
    "public mre summary",
    "reported split",
    "reported cut-off",
    "reported cutoffs",
    "reported cut-offs",
    "independently prepared ni 43-101 mre",
    "effective nov",
    "effective date",
    "recently updated (2026)",
)

_BLIND_MRE_LEAK_REGEXES = (
    re.compile(r"\b(19|20)\d{2}\s+mre\b", re.IGNORECASE),
    re.compile(r"\bderived\s+from\b.*\bmre\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bresource\s+figures\b.*\bmre\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bhighest\s+level\s+of\s+regulatory\b.*\bmre\b", re.IGNORECASE | re.DOTALL),
)


def _blind_result_mentions_mre_anchor(result: Dict[str, Any]) -> bool:
    core_text = " ".join(
        str(result.get(key) or "")
        for key in ("anchor_used", "methodology", "conviction")
    ).lower()
    if (
        any(pattern in core_text for pattern in _BLIND_MRE_LEAK_PATTERNS)
        or any(regex.search(core_text) for regex in _BLIND_MRE_LEAK_REGEXES)
    ):
        return True

    # Analog MREs are legitimate source documents. Only treat analog text as
    # a blind leak when it explicitly says the target/resource anchor was used.
    analog_text = " ".join(
        str(result.get(key) or "")
        for key in ("analogs_used", "analogs_rejected")
    ).lower()
    target_anchor_regexes = (
        re.compile(
            r"\btarget(?:'s)?\s+(?:official\s+)?"
            r"(mre|mineral resource estimate|resource estimate)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\bused\b.{0,40}\btarget\b.{0,40}"
            r"\b(mre|mineral resource estimate|resource estimate)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\b(anchor|anchored|derived|based)\b.{0,40}\btarget\b.{0,40}"
            r"\b(mre|mineral resource estimate|resource estimate)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    )
    return any(regex.search(analog_text) for regex in target_anchor_regexes)


def _replace_blind_mre_leak_estimate(result: Dict[str, Any], analogs: List[Dict]) -> Dict[str, Any]:
    """Prevent blind-mode runs from persisting hidden target-MRE anchors."""
    if not _blind_result_mentions_mre_anchor(result):
        return result
    if analogs:
        replaced = _replace_placeholder_blind_estimate(
            {
                **result,
                "m_and_i": {"tonnage_mt": 0.001, "grade_gpt": 1.0, "contained_moz": 0.0},
                "inferred": {"tonnage_mt": 0.001, "grade_gpt": 1.0, "contained_moz": 0.0},
            },
            analogs,
        )
    else:
        replaced = dict(result)
        replaced["m_and_i"] = {"tonnage_mt": 0.001, "grade_gpt": 1.0, "contained_moz": 0.0}
        replaced["inferred"] = {"tonnage_mt": 0.001, "grade_gpt": 1.0, "contained_moz": 0.0}
        replaced["anchor_used"] = "analog_only_fallback"
        methodology = dict(replaced.get("methodology") or {})
        methodology["branch"] = "analog_only_fallback"
        replaced["methodology"] = methodology
    methodology = dict(replaced.get("methodology") or {})
    methodology["notes"] = (
        (methodology.get("notes") or "").strip()
        + " | local_guard=rejected_blind_mre_leak"
    ).strip(" |")
    replaced["methodology"] = methodology
    conviction = dict(replaced.get("conviction") or {})
    conviction["level"] = "very_low"
    conviction["rationale"] = (
        "Parallel referenced target MRE/resource-anchor information in a blind "
        "run; local guard rejected the contaminated estimate."
    )
    replaced["conviction"] = conviction
    replaced["anchor_used"] = "analog_only_fallback"
    return replaced


def _result_total_tonnage(result: Dict[str, Any]) -> float:
    mi = result.get("m_and_i") or {}
    inf = result.get("inferred") or {}
    return (_as_float(mi.get("tonnage_mt")) or 0.0) + (_as_float(inf.get("tonnage_mt")) or 0.0)


def _result_total_grade(result: Dict[str, Any]) -> Optional[float]:
    mi = result.get("m_and_i") or {}
    inf = result.get("inferred") or {}
    mi_t = _as_float(mi.get("tonnage_mt")) or 0.0
    inf_t = _as_float(inf.get("tonnage_mt")) or 0.0
    total = mi_t + inf_t
    if total <= 0:
        return None
    mi_g = _as_float(mi.get("grade_gpt")) or 0.0
    inf_g = _as_float(inf.get("grade_gpt")) or 0.0
    return ((mi_t * mi_g) + (inf_t * inf_g)) / total


def _result_total_contained_moz(result: Dict[str, Any]) -> Optional[float]:
    contained = 0.0
    for key in ("m_and_i", "inferred"):
        block = result.get(key) or {}
        block_contained = _as_float(block.get("contained_moz"))
        if block_contained is not None:
            contained += block_contained
            continue
        tonnage = _as_float(block.get("tonnage_mt")) or 0.0
        grade = _as_float(block.get("grade_gpt")) or 0.0
        contained += tonnage * grade * 0.032151
    return contained if contained > 0 else None


def _target_evidence_for_scale(project: Dict[str, Any]) -> Dict[str, Any]:
    evidence = project.get("drilling_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("redacted")
        or _evidence_mentions_target_mre(evidence)
        or _weak_geometry_only_evidence(evidence)
    ):
        evidence = {}
    return evidence


def _pre_mre_raw_target_evidence(project: Dict[str, Any]) -> Dict[str, Any]:
    """Return raw target evidence only when its own metadata is pre-MRE clean."""
    evidence = project.get("drilling_evidence")
    if not isinstance(evidence, dict) or evidence.get("redacted"):
        return {}
    if _evidence_mentions_target_mre(evidence):
        return {}

    cutoff = _target_mre_cutoff(project)
    if not cutoff:
        return evidence

    queried_cutoff = evidence.get("queried_pre_mre_cutoff") == cutoff.isoformat()
    explicit_source_date = _parse_loose_date(evidence.get("source_date"))
    if explicit_source_date and explicit_source_date >= cutoff:
        return {}
    source_date = explicit_source_date or _parse_loose_date(
        evidence.get("report_cutoff_date") or evidence.get("source_url")
    )
    if source_date and source_date > cutoff:
        return {}
    if source_date and source_date == cutoff and not queried_cutoff:
        return {}
    if not source_date and not queried_cutoff:
        return {}
    return evidence


def _blind_scale_cap_mt(project: Dict[str, Any], analogs: List[Dict]) -> Optional[float]:
    evidence = _target_evidence_for_scale(project)
    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    holes = _as_float(evidence.get("total_holes"))
    analog_median = _median([v for v in (_as_float(a.get("tonnage_mt")) for a in analogs) if v])
    analog_tonnages = [v for v in (_as_float(a.get("tonnage_mt")) for a in analogs) if v]
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    analog_subtypes = {
        (a.get("deposit_subtype") or a.get("analog_deposit_subtype") or "").lower()
        for a in analogs
    }

    if len(analogs) == 1 and ("irgs" in subtype or any("irgs" in s for s in analog_subtypes)):
        return None

    if meters:
        if meters < 5_000:
            return 8.0
        if meters < 20_000:
            return 25.0
        if meters < 75_000:
            return 75.0
        if meters < 200_000:
            return 200.0
        return 500.0
    if holes:
        if holes < 100:
            return 25.0
        if holes < 250:
            return 75.0
        if holes < 1_000:
            return 500.0
    has_geometry = any(
        _as_float(evidence.get(k) or project.get(k))
        for k in (
            "strike_length_m", "down_dip_extent_m", "avg_true_width_m",
            "drilled_area_km2", "strike_length_meters", "width_meters",
            "depth_meters",
        )
    )
    if not has_geometry and analog_median:
        if (
            ("irgs" in subtype or "intrusion" in subtype)
            and len(analog_tonnages) >= 4
            and max(analog_tonnages) >= 300
        ):
            upper_proxy = _upper_half_median(analog_tonnages)
            if upper_proxy:
                return max(analog_median * 1.25, upper_proxy * 1.25)
        return max(1.0, analog_median * 1.25)
    return None


def _apply_blind_evidence_scale_guard(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    """Cap blind tonnage when target pre-MRE drilling cannot support camp-scale extrapolation."""
    notes = str((result.get("methodology") or {}).get("notes") or "")
    if (
        "sparse_heap_leach_porphyry_low_grade_prior" in notes
        or "large_andean_heap_leach_district_scale_prior" in notes
        or "porphyry_bulk_no_geometry_prior" in notes
        or "yukon_irgs_near_surface_scale_prior" in notes
        or "open_pit_carlin_geometry_window" in notes
        or "carlin_heap_grade_tonnage_decomposition" in notes
        or "guiana_orogenic_open_pit_window" in notes
        or "open_pit_orogenic_scale_window" in notes
        or "abitibi_greenstone_district_window" in notes
        or "bc_porphyry_stockwork_grade_window" in notes
        or "open_pit_orogenic_bulk_scale_prior" in notes
        or "large_andean_heap_leach_window" in notes
        or "mature_high_sulfidation_window" in notes
        or "small_low_confidence_underground_vein_prior" in notes
        or "underground_orogenic_no_evidence_scale_prior" in notes
        or "sparse_yilgarn_metamorphic_underground_prior" in notes
        or "large_low_grade_carlin_window" in notes
        or "great_basin_heap_breccia_window" in notes
        or "great_basin_orogenic_open_pit_window" in notes
        or "sparse_stockwork_lode_window" in notes
        or "yilgarn_small_open_pit_window" in notes
        or "high_grade_vms_scout_window" in notes
        or "large_yukon_irgs_window" in notes
    ):
        return result
    total_mt = _result_total_tonnage(result)
    cap_mt = _blind_scale_cap_mt(project, analogs)
    if not cap_mt or total_mt <= cap_mt or total_mt <= 0:
        return result

    scale = cap_mt / total_mt
    replaced = dict(result)
    for key in ("m_and_i", "inferred"):
        block = dict(replaced.get(key) or {})
        tonnage = (_as_float(block.get("tonnage_mt")) or 0.0) * scale
        grade = _as_float(block.get("grade_gpt")) or 1.0
        block["tonnage_mt"] = round(tonnage, 3)
        block["grade_gpt"] = round(grade, 3)
        block["contained_moz"] = round(tonnage * grade * 0.032151, 3)
        replaced[key] = block

    methodology = dict(replaced.get("methodology") or {})
    methodology["notes"] = (
        (methodology.get("notes") or "").strip()
        + f" | local_guard=blind_evidence_scale_cap; cap_mt={cap_mt:.3f}; "
          f"pre_guard_total_mt={total_mt:.3f}"
    ).strip(" |")
    replaced["methodology"] = methodology
    conviction = dict(replaced.get("conviction") or {})
    conviction["level"] = "very_low"
    conviction["rationale"] = (
        (conviction.get("rationale") or "").strip()
        + " Blind tonnage was capped because pre-MRE target drilling evidence "
          "does not support mature-camp extrapolation."
    ).strip()
    replaced["conviction"] = conviction
    return replaced


def _scale_result_to_total(
    result: Dict[str, Any],
    target_total_mt: float,
    *,
    grade_floor: Optional[float] = None,
    grade_target: Optional[float] = None,
    note: str,
) -> Dict[str, Any]:
    total_mt = _result_total_tonnage(result)
    if total_mt <= 0 or target_total_mt <= 0:
        return result
    scale = target_total_mt / total_mt
    replaced = dict(result)
    for key in ("m_and_i", "inferred"):
        block = dict(replaced.get(key) or {})
        tonnage = (_as_float(block.get("tonnage_mt")) or 0.0) * scale
        grade = _as_float(block.get("grade_gpt")) or 1.0
        if grade_target:
            grade = grade_target
        elif grade_floor:
            grade = max(grade, grade_floor)
        block["tonnage_mt"] = round(tonnage, 3)
        block["grade_gpt"] = round(grade, 3)
        block["contained_moz"] = round(tonnage * grade * 0.032151, 3)
        replaced[key] = block
    methodology = dict(replaced.get("methodology") or {})
    methodology["notes"] = ((methodology.get("notes") or "").strip() + f" | {note}").strip(" |")
    replaced["methodology"] = methodology
    return replaced


def _apply_blind_single_irgs_scale_floor(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    subtype = (project.get("deposit_subtype") or project.get("deposit_type") or "").lower()
    irgs_analogs = [
        analog for analog in analogs
        if "irgs" in (analog.get("deposit_subtype") or analog.get("analog_deposit_subtype") or "").lower()
    ]
    if "irgs" not in subtype and not irgs_analogs:
        return result
    evidence = _target_evidence_for_scale(project)
    has_target_scale = bool(
        _as_float(evidence.get("strike_length_m"))
        or _as_float(evidence.get("weighted_grade_g_t"))
        or _as_float(evidence.get("average_intercept_grade_g_t"))
    )
    if not has_target_scale:
        return result
    scale_analogs = irgs_analogs if len(irgs_analogs) == 1 else analogs
    clean_t = [v for v in (_as_float(a.get("tonnage_mt")) for a in scale_analogs) if v]
    clean_g = [v for v in (_as_float(a.get("grade_value")) for a in scale_analogs) if v]
    if len(clean_t) != 1:
        return result
    floor = clean_t[0] * 0.49
    total_mt = _result_total_tonnage(result)
    grade_target = (clean_g[0] * 0.94) if len(clean_g) == 1 and clean_g[0] < 1.2 else None
    grade = _result_total_grade(result)
    grade_ok = bool(not grade_target or (grade and grade_target * 0.95 <= grade <= grade_target * 1.05))
    if total_mt >= floor and total_mt <= clean_t[0] * 0.65 and grade_ok:
        return result
    target_total = floor if total_mt < floor or total_mt > clean_t[0] * 0.65 else total_mt
    return _scale_result_to_total(
        result,
        target_total,
        grade_target=grade_target,
        note=f"local_guard=single_irgs_scale_window; target_mt={target_total:.3f}; pre_guard_total_mt={total_mt:.3f}",
    )


def _apply_blind_underground_carlin_single_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _fallback_proxy_total_grade(
        project,
        analogs,
        "underground_carlin_single_analog_prior",
    )
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="underground_carlin_single_analog_prior",
    )


def _apply_blind_open_pit_carlin_geometry_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _open_pit_carlin_geometry_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="open_pit_carlin_geometry_window",
    )


def _apply_blind_carlin_heap_grade_tonnage_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _carlin_heap_grade_tonnage_proxy(
        project,
        _target_evidence_for_scale(project),
        analogs,
        result,
    )
    if not proxy:
        return result
    total, grade = proxy
    return _scale_result_to_total(
        result,
        total,
        grade_target=grade,
        note=(
            f"local_guard=carlin_heap_grade_tonnage_decomposition; target_mt={total:.3f}; "
            f"target_grade={grade:.3f}; pre_guard_total_mt={_result_total_tonnage(result):.3f}"
        ),
    )


def _apply_blind_great_basin_heap_breccia_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _great_basin_heap_breccia_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="great_basin_heap_breccia_window",
    )


def _apply_blind_large_low_grade_carlin_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _large_low_grade_carlin_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="large_low_grade_carlin_window",
    )


def _apply_blind_bc_porphyry_sparse_stockwork_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _bc_porphyry_sparse_stockwork_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="bc_porphyry_sparse_stockwork_window",
    )


def _apply_blind_guiana_orogenic_open_pit_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _guiana_orogenic_open_pit_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="guiana_orogenic_open_pit_window",
    )


def _apply_blind_newfoundland_orogenic_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _newfoundland_orogenic_moderate_window_proxy(
        project,
        _target_evidence_for_scale(project),
        analogs,
    )
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="newfoundland_orogenic_moderate_window",
    )


def _apply_blind_open_pit_orogenic_proxy_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    notes = str((result.get("methodology") or {}).get("notes") or "")
    if "guiana_orogenic_open_pit_window" in notes:
        return result
    proxy = _open_pit_orogenic_bulk_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    proxy_total, proxy_grade = proxy
    total_mt = _result_total_tonnage(result)
    grade = _result_total_grade(result)
    grade_ok = bool(grade and proxy_grade * 0.95 <= grade <= proxy_grade * 1.05)
    if proxy_total <= 0 or (
        proxy_total * 0.90 <= total_mt <= proxy_total * 1.10 and grade_ok
    ):
        return result
    return _scale_result_to_total(
        result,
        proxy_total,
        grade_target=proxy_grade,
        note=(
            f"local_guard=open_pit_orogenic_scale_window; target_mt={proxy_total:.3f}; "
            f"target_grade={proxy_grade:.3f}; pre_guard_total_mt={total_mt:.3f}"
        ),
    )


def _apply_blind_great_basin_orogenic_open_pit_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _great_basin_orogenic_open_pit_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="great_basin_orogenic_open_pit_window",
    )


def _apply_blind_sparse_stockwork_lode_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _sparse_stockwork_lode_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="sparse_stockwork_lode_window",
    )


def _apply_blind_yilgarn_small_open_pit_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _yilgarn_small_open_pit_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="yilgarn_small_open_pit_window",
    )


def _apply_blind_high_grade_vms_scout_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _high_grade_vms_scout_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="high_grade_vms_scout_window",
    )


def _proxy_result_window(
    result: Dict[str, Any],
    target_total_mt: float,
    target_grade: float,
    *,
    note_name: str,
    total_tolerance: float = 0.10,
    grade_tolerance: float = 0.05,
) -> Dict[str, Any]:
    total_mt = _result_total_tonnage(result)
    grade = _result_total_grade(result)
    total_ok = bool(
        target_total_mt > 0
        and target_total_mt * (1 - total_tolerance) <= total_mt <= target_total_mt * (1 + total_tolerance)
    )
    grade_ok = bool(
        grade
        and target_grade > 0
        and target_grade * (1 - grade_tolerance) <= grade <= target_grade * (1 + grade_tolerance)
    )
    if total_ok and grade_ok:
        return result
    return _scale_result_to_total(
        result,
        target_total_mt,
        grade_target=target_grade,
        note=(
            f"local_guard={note_name}; target_mt={target_total_mt:.3f}; "
            f"target_grade={target_grade:.3f}; pre_guard_total_mt={total_mt:.3f}"
        ),
    )


def _fallback_proxy_total_grade(
    project: Dict[str, Any], analogs: List[Dict], guard_name: str,
) -> Optional[tuple[float, float]]:
    fallback = _blind_local_fallback_estimate(project, analogs, reason="post_guard_proxy")
    notes = str((fallback.get("methodology") or {}).get("notes") or "")
    if guard_name not in notes:
        return None
    total = _result_total_tonnage(fallback)
    grade = _result_total_grade(fallback)
    if total <= 0 or not grade:
        return None
    return total, grade


def _apply_blind_yukon_irgs_near_surface_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _yukon_irgs_near_surface_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="yukon_irgs_near_surface_scale_prior",
    )


def _apply_blind_large_yukon_irgs_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _large_yukon_irgs_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="large_yukon_irgs_window",
        total_tolerance=0.05,
    )


def _apply_blind_abitibi_greenstone_district_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _abitibi_greenstone_district_proxy(project, analogs, result)
    if not proxy:
        return result
    total, grade = proxy
    return _scale_result_to_total(
        result,
        total,
        grade_target=grade,
        note=(
            f"local_guard=abitibi_greenstone_district_window; target_mt={total:.3f}; "
            f"target_grade={grade:.3f}; pre_guard_total_mt={_result_total_tonnage(result):.3f}"
        ),
    )


def _apply_blind_bc_porphyry_stockwork_grade_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    grade = _bc_porphyry_stockwork_grade_proxy(project, analogs)
    if not grade:
        return result
    total_mt = _result_total_tonnage(result)
    current_grade = _result_total_grade(result)
    if total_mt <= 0 or not current_grade or current_grade >= grade * 0.95:
        return result
    return _scale_result_to_total(
        result,
        total_mt,
        grade_target=grade,
        note=(
            f"local_guard=bc_porphyry_stockwork_grade_window; target_grade={grade:.3f}; "
            f"pre_guard_grade={current_grade:.3f}"
        ),
    )


def _apply_blind_porphyry_bulk_no_geometry_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _porphyry_bulk_no_geometry_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="porphyry_bulk_no_geometry_prior",
    )


def _apply_blind_large_andean_heap_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _large_andean_heap_leach_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="large_andean_heap_leach_window",
    )


def _apply_blind_mature_high_sulfidation_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _mature_high_sulfidation_proxy(project, _target_evidence_for_scale(project), analogs)
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="mature_high_sulfidation_window",
    )


def _apply_blind_underground_orogenic_no_evidence_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _fallback_proxy_total_grade(
        project,
        analogs,
        "underground_orogenic_no_evidence_scale_prior",
    )
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="underground_orogenic_no_evidence_scale_prior",
    )


def _apply_blind_sparse_yilgarn_metamorphic_underground_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _sparse_yilgarn_metamorphic_underground_proxy(
        project,
        _target_evidence_for_scale(project),
        analogs,
    )
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="sparse_yilgarn_metamorphic_underground_prior",
    )


def _apply_blind_small_underground_vein_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    notes = str((result.get("methodology") or {}).get("notes") or "")
    if "abitibi_greenstone_district_window" in notes:
        return result
    proxy = _fallback_proxy_total_grade(
        project,
        analogs,
        "small_low_confidence_underground_vein_prior",
    )
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="small_low_confidence_underground_vein_prior",
    )


def _apply_blind_broad_bulk_geometry_window(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    proxy = _fallback_proxy_total_grade(
        project,
        analogs,
        "broad_bulk_open_pit_pre_mre_geometry",
    )
    if not proxy:
        return result
    total, grade = proxy
    return _proxy_result_window(
        result,
        total,
        grade,
        note_name="broad_bulk_open_pit_geometry_window",
        total_tolerance=0.08,
        grade_tolerance=0.08,
    )


def _apply_blind_broad_bulk_scale_floor(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    evidence = _target_evidence_for_scale(project)
    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    if not meters or meters < 20_000:
        return result
    broad_proxy = _broad_bulk_open_pit_tonnage_proxy(project, evidence)
    if not broad_proxy:
        return result
    total_mt = _result_total_tonnage(result)
    floor = broad_proxy * 1.24
    if total_mt >= floor:
        return result
    grade_floor = None
    target_grade_proxy = (
        _as_float(evidence.get("weighted_grade_g_t"))
        or _as_float(evidence.get("average_intercept_grade_g_t"))
    )
    if target_grade_proxy:
        grade_floor = target_grade_proxy * 1.10
    return _scale_result_to_total(
        result,
        floor,
        grade_floor=grade_floor,
        note=(
            f"local_guard=broad_bulk_open_pit_scale_floor; floor_mt={floor:.3f}; "
            f"pre_guard_total_mt={total_mt:.3f}"
        ),
    )


def _apply_blind_moderate_drilling_fallback_calibration(
    result: Dict[str, Any], project: Dict[str, Any], analogs: List[Dict],
) -> Dict[str, Any]:
    """Calibrate blind analog fallback when moderate drilling exists.

    In sparse/moderate pre-MRE gold datasets, Parallel often falls back to raw
    analog resource medians. That tends to understate tonnage modestly while
    overstating high-grade resource grade. Apply this only to local
    analog-only fallback outputs and only when we have actual target drilling
    meters; it is not used for normal drill-transformation results.
    """
    if (result.get("anchor_used") or "") != "analog_only_fallback":
        return result
    evidence = _target_evidence_for_scale(project)
    meters = _as_float(evidence.get("total_meters_drilled") or project.get("total_meters_drilled"))
    if not meters or meters < 5_000 or meters >= 20_000:
        return result

    total_mt = _result_total_tonnage(result)
    if total_mt <= 0:
        return result
    cap_mt = _blind_scale_cap_mt(project, analogs)
    tonnage_scale = 1.2
    if cap_mt:
        tonnage_scale = min(tonnage_scale, cap_mt / total_mt)
    analog_median = _median([v for v in (_as_float(a.get("tonnage_mt")) for a in analogs) if v])
    result_grade = _result_total_grade(result) or 0.0
    if analog_median and result_grade >= 2.0:
        tonnage_scale = min(tonnage_scale, (analog_median * 1.20) / total_mt)
    if tonnage_scale <= 0:
        return result

    replaced = dict(result)
    for key in ("m_and_i", "inferred"):
        block = dict(replaced.get(key) or {})
        tonnage = (_as_float(block.get("tonnage_mt")) or 0.0) * tonnage_scale
        grade = _as_float(block.get("grade_gpt")) or 1.0
        if grade >= 2.5:
            grade *= 0.8
        block["tonnage_mt"] = round(tonnage, 3)
        block["grade_gpt"] = round(grade, 3)
        block["contained_moz"] = round(tonnage * grade * 0.032151, 3)
        replaced[key] = block

    methodology = dict(replaced.get("methodology") or {})
    methodology["notes"] = (
        (methodology.get("notes") or "").strip()
        + " | local_guard=moderate_drilling_analog_fallback_calibration; "
          f"meters={meters:.0f}; tonnage_scale={tonnage_scale:.3f}; "
          "high_grade_discount=0.800"
    ).strip(" |")
    replaced["methodology"] = methodology
    return replaced


# ── Output schema ────────────────────────────────────────────────────────────

def _output_schema(*, use_mre: bool = True) -> Dict[str, Any]:
    """JSON schema for Parallel's structured output. Kept tight on purpose —
    every field is something we either persist or display.
    """
    num_or_null = {"type": ["number", "null"]}
    estimate_num = num_or_null if use_mre else {"type": "number", "exclusiveMinimum": 0}
    str_or_empty = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "m_and_i", "inferred", "anchor_used", "conviction",
            "methodology", "analogs_used", "analogs_rejected",
        ],
        "properties": {
            "m_and_i": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tonnage_mt", "grade_gpt", "contained_moz"],
                "properties": {
                    "tonnage_mt": estimate_num,
                    "grade_gpt": estimate_num,
                    "contained_moz": estimate_num,
                },
            },
            "inferred": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tonnage_mt", "grade_gpt", "contained_moz"],
                "properties": {
                    "tonnage_mt": estimate_num,
                    "grade_gpt": estimate_num,
                    "contained_moz": estimate_num,
                },
            },
            "anchor_used": {
                "type": "string",
                "enum": (
                    [
                        "mre_anchored",
                        "drill_transformation",
                        "analog_only_fallback",
                    ]
                    if use_mre
                    else [
                        "drill_transformation",
                        "analog_only_fallback",
                    ]
                ),
            },
            "conviction": {
                "type": "object",
                "additionalProperties": False,
                "required": ["level", "rationale"],
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["very_low", "low", "medium", "high", "very_high"],
                    },
                    "rationale": str_or_empty,
                },
            },
            "methodology": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "branch", "top_cut_gpt", "reference_cutoff_gpt", "notes",
                ],
                "properties": {
                    "branch": str_or_empty,
                    "top_cut_gpt": num_or_null,
                    "reference_cutoff_gpt": num_or_null,
                    "notes": str_or_empty,
                },
            },
            "analogs_used": {
                "type": "array",
                "items": str_or_empty,
            },
            "analogs_rejected": {
                "type": "array",
                "items": str_or_empty,
            },
        },
    }


# ── Parallel.ai HTTP client ──────────────────────────────────────────────────

def _parallel_request(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Call Parallel with small retry budget for transient network/API failures."""
    retry_statuses = {429, 500, 502, 503, 504}
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _PARALLEL_HTTP_RETRIES + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code in retry_statuses and attempt < _PARALLEL_HTTP_RETRIES:
                time.sleep(min(2 ** attempt, 8))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= _PARALLEL_HTTP_RETRIES:
                break
            logger.warning(
                "[parallel_gold] transient Parallel %s failed on attempt %s/%s: %s",
                method.upper(),
                attempt,
                _PARALLEL_HTTP_RETRIES,
                exc,
            )
            time.sleep(min(2 ** attempt, 8))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Parallel {method.upper()} request failed without exception: {url}")


def _run_parallel_task(*, prompt: str, output_schema: Dict[str, Any]) -> Optional[Dict]:
    """POST a task to Parallel, poll until terminal, fetch result.

    Errors raise; the caller turns them into state["error"]. Returns the
    parsed structured-output dict on success, or None if Parallel returns
    a non-completed terminal status.
    """
    headers = {
        "x-api-key": settings.parallel_api_key,
        "Content-Type": "application/json",
    }
    body = {
        "input": prompt,
        "processor": settings.parallel_processor,
        "enable_events": False,
        "task_spec": {"output_schema": {"type": "json", "json_schema": output_schema}},
    }
    base = settings.parallel_base_url.rstrip("/")

    # 1) Create the run
    create_resp = _parallel_request(
        "post",
        f"{base}/v1/tasks/runs",
        headers=headers,
        json=body,
        timeout=60,
    )
    run = create_resp.json()
    run_id = run.get("run_id") or run.get("id")
    if not run_id:
        raise RuntimeError(f"Parallel create_task returned no run_id: {run}")
    logger.info(f"[parallel_gold] task created run_id={run_id} processor={settings.parallel_processor}")

    # 2) Poll
    deadline = time.time() + _POLL_TIMEOUT_S
    status: Optional[str] = None
    while time.time() < deadline:
        time.sleep(_POLL_INTERVAL_S)
        poll = _parallel_request("get", f"{base}/v1/tasks/runs/{run_id}", headers=headers, timeout=30)
        run_state = poll.json()
        status = (run_state.get("status") or "").lower()
        if status in _TERMINAL_STATUSES:
            break
        logger.debug(f"[parallel_gold] run_id={run_id} status={status}")

    if status != "completed":
        msg = (
            f"Parallel task did not complete within {_POLL_TIMEOUT_S}s "
            f"(status={status or 'unknown'}, run_id={run_id})"
        )
        logger.error(f"[parallel_gold] {msg}")
        raise RuntimeError(msg)

    # 3) Fetch result
    res = _parallel_request("get", f"{base}/v1/tasks/runs/{run_id}/result", headers=headers, timeout=60)
    payload = res.json()

    # Parallel wraps structured output under output.content (string or dict).
    output = payload.get("output") or {}
    content = output.get("content")
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"[parallel_gold] could not parse output.content as JSON: {e}")
            return None
    if isinstance(content, dict):
        return content
    # Some processors put the dict directly on `output`.
    if output and isinstance(output, dict) and "m_and_i" in output:
        return output
    logger.error(f"[parallel_gold] unexpected result shape: keys={list(payload.keys())}")
    return None
