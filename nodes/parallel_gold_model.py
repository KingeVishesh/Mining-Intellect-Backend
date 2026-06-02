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
import time
from typing import Any, Dict, List, Optional

import requests

from config import settings

logger = logging.getLogger(__name__)

# Parallel.ai task lifecycle: queued -> running -> completed / failed.
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}
_POLL_INTERVAL_S = 15
_POLL_TIMEOUT_S = 60 * 150  # ultra deep-research with discovery + mandatory enrichment can take 90-120+ min


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
    schema = _output_schema()

    try:
        result = _run_parallel_task(prompt=prompt, output_schema=schema)
    except Exception as e:
        logger.exception(f"[parallel_gold] Parallel API call failed: {e}")
        return {"error": f"Parallel API call failed: {e}"}

    if not result:
        return {"error": "Parallel returned no result"}

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
    analogs_block = _format_analogs_block(analogs)
    mre_directive = _mre_directive(project=project, use_mre=use_mre)
    analog_directive = _analog_directive(find_analogs=find_analogs, analogs=analogs)

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
  • If any ratio for an analog is null in `analogs_used[].implied_ratios`,
    the rationale field MUST name the specific source documents you
    consulted (AIF / NI 43-101 / R&R report / quarterly report — with
    dates and report names) and what each one disclosed or didn't.
    "Data not publicly tabulated" or "not disclosed" without naming
    sources is REJECTED as a non-answer. Enrich aggressively before
    giving up — most major-operator drilling data is in R&R appendices.
  • `methodology` must state: which branch ran (mre_anchored /
    drill_transformation / analog_only_fallback), the top-cut value,
    the reference cutoff, any stage-weighting applied, and whether
    the geometric ceiling clamped the result.
  • `conviction` is one of: very_low / low / medium / high / very_high,
    plus a one-sentence rationale.
""".strip()


def _analog_directive(*, find_analogs: bool, analogs: List[Dict]) -> str:
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
    "strike_length_m", "down_dip_extent_m", "avg_true_width_m",
    "bulk_density_t_per_m3", "metallurgical_recovery_pct",
    "similarity_score", "similarity_notes",
    "drilling_evidence",
]


def _format_project_block(project: Dict, *, use_mre: bool) -> str:
    """Render the target project as a JSON block Parallel can scan.

    When `use_mre=False` we strip the published-MRE fields from the rendered
    block to remove the temptation to peek. (We still tell the agent about
    pre-MRE mode in the directive, but belt-and-braces.)
    """
    payload = {k: project.get(k) for k in _PROJECT_FIELDS_TO_SHOW if k in project}
    if not use_mre:
        for k in (
            "mre_mi_tonnage_mt", "mre_mi_grade",
            "mre_inferred_tonnage_mt", "mre_inferred_grade",
            "mre_date", "mre_source_url",
            "tonnage_mt", "grade_value", "cutoff_grade",
        ):
            payload.pop(k, None)
    return json.dumps(payload, indent=2, default=str, ensure_ascii=False)


def _format_analogs_block(analogs: List[Dict]) -> str:
    """One JSON array, each analog one object. Drilling evidence inlined."""
    cleaned = [
        {k: a.get(k) for k in _ANALOG_FIELDS_TO_SHOW if k in a}
        for a in analogs
    ]
    return json.dumps(cleaned, indent=2, default=str, ensure_ascii=False)


# ── Output schema ────────────────────────────────────────────────────────────

def _output_schema() -> Dict[str, Any]:
    """JSON schema for Parallel's structured output. Kept tight on purpose —
    every field is something we either persist or display.
    """
    num_or_null = {"type": ["number", "null"]}
    str_or_empty = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "m_and_i", "inferred", "total",
            "anchor_used", "conviction", "methodology",
            "analogs_used", "analogs_rejected", "sources",
        ],
        "properties": {
            "m_and_i": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tonnage_mt", "grade_gpt", "contained_moz"],
                "properties": {
                    "tonnage_mt": num_or_null,
                    "grade_gpt": num_or_null,
                    "contained_moz": num_or_null,
                },
            },
            "inferred": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tonnage_mt", "grade_gpt", "contained_moz"],
                "properties": {
                    "tonnage_mt": num_or_null,
                    "grade_gpt": num_or_null,
                    "contained_moz": num_or_null,
                },
            },
            "total": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tonnage_mt", "grade_gpt", "contained_moz"],
                "properties": {
                    "tonnage_mt": num_or_null,
                    "grade_gpt": num_or_null,
                    "contained_moz": num_or_null,
                },
            },
            "anchor_used": {
                "type": "string",
                "enum": [
                    "mre_anchored",
                    "drill_transformation",
                    "analog_only_fallback",
                ],
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
                    "branch", "top_cut_gpt", "reference_cutoff_gpt",
                    "stage_weighting_applied", "geometric_ceiling_applied",
                    "cohort_median_ratios", "notes",
                ],
                "properties": {
                    "branch": str_or_empty,
                    "top_cut_gpt": num_or_null,
                    "reference_cutoff_gpt": num_or_null,
                    "stage_weighting_applied": {"type": "boolean"},
                    "geometric_ceiling_applied": {"type": "boolean"},
                    "cohort_median_ratios": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "m_and_i_tonnage_per_meter",
                            "m_and_i_grade_preservation",
                            "inferred_tonnage_per_meter",
                            "inferred_grade_preservation",
                            "envelope_realization",
                        ],
                        "properties": {
                            "m_and_i_tonnage_per_meter": num_or_null,
                            "m_and_i_grade_preservation": num_or_null,
                            "inferred_tonnage_per_meter": num_or_null,
                            "inferred_grade_preservation": num_or_null,
                            "envelope_realization": num_or_null,
                        },
                    },
                    "notes": str_or_empty,
                },
            },
            "analogs_used": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name", "weight", "implied_ratios", "rationale",
                    ],
                    "properties": {
                        "name": str_or_empty,
                        "weight": num_or_null,
                        "implied_ratios": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "m_and_i_tonnage_per_meter",
                                "m_and_i_grade_preservation",
                                "inferred_tonnage_per_meter",
                                "inferred_grade_preservation",
                                "envelope_realization",
                            ],
                            "properties": {
                                "m_and_i_tonnage_per_meter": num_or_null,
                                "m_and_i_grade_preservation": num_or_null,
                                "inferred_tonnage_per_meter": num_or_null,
                                "inferred_grade_preservation": num_or_null,
                                "envelope_realization": num_or_null,
                            },
                        },
                        "rationale": str_or_empty,
                    },
                },
            },
            "analogs_rejected": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "reason"],
                    "properties": {
                        "name": str_or_empty,
                        "reason": str_or_empty,
                    },
                },
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "url", "used_for"],
                    "properties": {
                        "title": str_or_empty,
                        "url": str_or_empty,
                        "used_for": str_or_empty,
                    },
                },
            },
        },
    }


# ── Parallel.ai HTTP client ──────────────────────────────────────────────────

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
        "task_spec": {"output_schema": {"type": "json", "json_schema": output_schema}},
    }
    base = settings.parallel_base_url.rstrip("/")

    # 1) Create the run
    create_resp = requests.post(
        f"{base}/v1/tasks/runs",
        headers=headers,
        json=body,
        timeout=60,
    )
    create_resp.raise_for_status()
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
        poll = requests.get(f"{base}/v1/tasks/runs/{run_id}", headers=headers, timeout=30)
        poll.raise_for_status()
        run_state = poll.json()
        status = (run_state.get("status") or "").lower()
        if status in _TERMINAL_STATUSES:
            break
        logger.debug(f"[parallel_gold] run_id={run_id} status={status}")

    if status != "completed":
        logger.error(f"[parallel_gold] task terminated with status={status} run_id={run_id}")
        return None

    # 3) Fetch result
    res = requests.get(f"{base}/v1/tasks/runs/{run_id}/result", headers=headers, timeout=60)
    res.raise_for_status()
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
