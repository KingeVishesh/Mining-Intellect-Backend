"""
LLM-as-judge for the analog finder. Standalone CLI you can run against
any subset of projects to get a quality score per project and an
aggregate digest of systemic issues.

Usage
-----
    # Score a single project
    python3 scripts/analog_quality_judge.py --project-id <uuid>

    # Random sample of 10 gold projects
    python3 scripts/analog_quality_judge.py --material gold --sample 10

    # Score every gold project (expensive — ~5 min, ~$1 in Grok credits)
    python3 scripts/analog_quality_judge.py --material gold --all

    # Score the projects with the lowest cascade quality (low_confidence
    # or thin library) first so the digest surfaces actionable gaps fast.
    python3 scripts/analog_quality_judge.py --material gold --sample 20 \
        --prioritize-gaps

Output
------
    1. Per-project JSON results in reports/judge/<timestamp>/scores.json
    2. Markdown digest in reports/judge/<timestamp>/digest.md
    3. Top-N suggested code/data fixes at the bottom of the digest

Reads cascade output FRESH each run (re-runs library_search + Exa via
the production cascade) — so the score reflects what the system would
return RIGHT NOW for that project, not whatever's stale in the DB.

What it checks (per project, 0-100):
    1. Geological similarity        (0-25 pts): subtype, pattern, mode, host
    2. Tectonic context            (0-25 pts): belt, sub-trend, age, structure
    3. Scale & stage similarity    (0-25 pts): tonnage, grade, mining
    4. Cohort coherence            (0-25 pts): all picks mutually consistent

Plus open-ended fields:
    - missing_canonicals: which analog(s) SHOULD be in the set but aren't
    - systemic_issue: pattern this reveals about the cascade's behavior
    - taxonomy_gap: keyword/slug to add to geo_taxonomy if applicable

The judge prompt is intentionally strict — we want it to surface
gaps, not validate weak picks. If the model agrees the set is great,
the score should clear 85.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from config import settings

from graphs.analog_finder import (
    load_project_and_rule_node,
    build_target_profile_node,
    library_search_node,
    exa_search_node,
    combine_filter_score_node,
)
from nodes.supabase_ops import get_client

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-3"


# ── Judge prompt ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """\
You are a senior mining-finance geologist evaluating the analog set an
automated system selected for a target mineral project. The system uses a
cascading geological-similarity match (commodity → deposit family →
subtype → pattern → mode → mining method → belt → sub-trend → tonnage →
grade), then ranks survivors.

Your job: score how well the chosen analog set fits the target, on four
dimensions of 25 points each (100 total). Be strict — analogs are used to
back the project's tonnage/grade model, so a wrong analog cascades into
wrong financial inferences.

Score rubric:
  1. Geological similarity (0-25)
       - Subtype match (alkalic vs Laramide vs IOCG ...)
       - Pattern match (vein vs disseminated_bulk vs stockwork ...)
       - Mode match (primary_sulfide vs supergene_oxide vs refractory ...)
       - Host rock class
       - Recovery / processing method
  2. Tectonic context (0-25)
       - Same tectonic belt (Abitibi / Yilgarn / Lachlan / ...)
       - Same SUB-TREND / SUB-CAMP within the belt (Cortez vs Carlin vs
         Battle Mountain-Eureka all inside great_basin_carlin; Cadillac
         Break vs Bousquet vs Casa Berardi all inside abitibi). Sub-trend
         match matters as much as belt match — different sub-trends have
         different host stratigraphy, age, and structural setting.
       - Cratonic age / orogen
       - Structural plumbing (shear zone vs unconformity vs detachment)
  3. Scale & stage similarity (0-25)
       - Tonnage in same order of magnitude
       - Grade in same band
       - Same mining method (open pit bulk vs underground vein)
       - Comparable project stage (explore / PEA / PFS / FS / production)
  4. Cohort coherence (0-25)
       - All chosen analogs are mutually consistent (not a mix of
         different styles)
       - No outlier that drags the cohort's tonnage/grade average wrong
       - At least one in-camp / in-trend canonical present

Also identify, in plain English:
  - missing_canonicals: list up to 3 deposit names that SHOULD be in the
    top-4 for this target but aren't. Include district / belt.
  - systemic_issue: if you spot a pattern (e.g. "cascade returned every
    Carlin in Nevada but missed all Cortez Trend canonicals"), describe
    it briefly. Otherwise null.
  - taxonomy_gap: if the target's location should have resolved to a
    specific sub-trend / sub-camp slug but didn't, name the slug or
    suggest one. Otherwise null.

Output STRICT JSON only, no commentary, matching this schema:
{
  "geology_pts": 0-25,
  "tectonic_pts": 0-25,
  "scale_pts": 0-25,
  "cohort_pts": 0-25,
  "total": 0-100,
  "verdict": "excellent|good|mediocre|poor",
  "rationale": "1-2 sentences",
  "missing_canonicals": [{"name": "...", "district": "...", "why": "..."}],
  "systemic_issue": "..." or null,
  "taxonomy_gap": "..." or null
}
"""


def _grok(prompt: str, timeout: int = 90) -> Optional[Dict[str, Any]]:
    """Call Grok-3 with the judge system prompt + the case-specific user
    prompt. Returns the parsed JSON dict or None on failure."""
    headers = {
        "Authorization": f"Bearer {settings.grok_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(GROK_API_URL, headers=headers, json=payload,
                                timeout=timeout)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Grok request failed: {e}")
        return None
    if resp.status_code != 200:
        logger.warning(f"Grok HTTP {resp.status_code}: {resp.text[:300]}")
        return None
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning(f"Grok returned non-JSON: {content[:300]}")
        return None


def _format_analog_brief(a: Dict[str, Any]) -> str:
    """Compact one-line representation of an analog for the judge prompt."""
    parts = [a.get("name") or "(unknown)"]
    if a.get("country") or a.get("district"):
        loc = " / ".join(p for p in (a.get("district"), a.get("country")) if p)
        parts.append(f"[{loc}]")
    geol_bits = []
    for k in ("deposit_subtype", "mineralization_pattern", "mineralization_mode",
              "tectonic_belt", "mining_method_class"):
        v = a.get(k)
        if v:
            geol_bits.append(f"{k}={v}")
    if geol_bits:
        parts.append("(" + ", ".join(geol_bits) + ")")
    if a.get("tonnage_mt") and a.get("grade_value"):
        parts.append(
            f"{a['tonnage_mt']} Mt @ {a['grade_value']} {a.get('grade_unit') or ''}"
        )
    parts.append(f"source={a.get('source')}")
    return " ".join(parts)


def _build_user_prompt(project: Dict[str, Any], target_profile: Dict[str, Any],
                       top: List[Dict[str, Any]]) -> str:
    """Render the case-specific prompt for one project."""
    p = project
    tp = target_profile or {}
    target_block = (
        f"TARGET PROJECT\n"
        f"  Name:             {p.get('name')}\n"
        f"  Material:         {p.get('material')}\n"
        f"  Country / Region: {p.get('country')} / {p.get('region')}\n"
        f"  District:         {p.get('district')}\n"
        f"  Location name:    {p.get('location_name')}\n"
        f"  Deposit type:     {p.get('deposit_type')}\n"
        f"  Deposit subtype:  {tp.get('deposit_subtype')}\n"
        f"  Pattern:          {tp.get('mineralization_pattern')}\n"
        f"  Mode:             {tp.get('mineralization_mode')}\n"
        f"  Host rock class:  {tp.get('host_rock_class')}\n"
        f"  Tectonic belt:    {tp.get('tectonic_belt')}\n"
        f"  Sub-trend:        {tp.get('sub_trend')}\n"
        f"  Metal suite:      {tp.get('metal_suite')}\n"
        f"  Mining method:    {tp.get('mining_method_class')}\n"
        f"  Project stage:    {tp.get('project_stage_class')}\n"
        f"  Tonnage / Grade:  {p.get('tonnage_mt')} Mt @ "
        f"{p.get('grade_value')} {p.get('grade_unit')}\n"
    )
    if top:
        ana_block = "SELECTED ANALOG SET (in rank order):\n"
        for i, a in enumerate(top, 1):
            ana_block += f"  {i}. {_format_analog_brief(a)}\n"
    else:
        ana_block = "SELECTED ANALOG SET: (empty — cascade returned no analogs)\n"
    instr = (
        "\nScore this set against the target on the four dimensions of the "
        "rubric. Return strict JSON only."
    )
    return target_block + "\n" + ana_block + instr


# ── Run the cascade fresh for each project ───────────────────────────────────

def _cascade_for(project_id: str, with_exa: bool = True) -> Dict[str, Any]:
    """Run the production cascade for one project. Returns a dict with
    project, target_profile, scored_analogs, low_confidence."""
    state: Dict[str, Any] = {"project_id": project_id}
    state.update(load_project_and_rule_node(state))
    if state.get("error"):
        return {"error": state["error"], "project": state.get("project")}
    state.update(build_target_profile_node(state))
    state.update(library_search_node(state))
    if with_exa:
        try:
            state.update(exa_search_node(state))
        except Exception as e:
            logger.warning(f"[exa] failed for {project_id}: {e}")
            state["exa_analogs"] = []
    else:
        state["exa_analogs"] = []
    state.update(combine_filter_score_node(state))
    return state


# ── Project listing / sampling ───────────────────────────────────────────────

def _list_projects(material: str, prioritize_gaps: bool = False) -> List[Dict[str, Any]]:
    client = get_client()
    q = (
        client.table("projects")
        .select("id,name,deposit_type,deposit_subtype,tectonic_belt")
        .ilike("material", material)
    )
    res = q.order("name").execute()
    rows = res.data or []
    if prioritize_gaps:
        # Sort projects with thinner structured data first so the digest
        # surfaces actionable gaps quickly.
        def gap_score(r: Dict[str, Any]) -> int:
            score = 0
            if not r.get("deposit_type"): score += 4
            if not r.get("deposit_subtype"): score += 3
            if not r.get("tectonic_belt"): score += 2
            return score
        rows.sort(key=lambda r: (-gap_score(r), r.get("name") or ""))
    return rows


# ── Aggregation + digest ─────────────────────────────────────────────────────

def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in results if r.get("score")]
    if not scored:
        return {"n": 0}
    totals = [r["score"]["total"] for r in scored]
    geos = [r["score"]["geology_pts"] for r in scored]
    tects = [r["score"]["tectonic_pts"] for r in scored]
    scales = [r["score"]["scale_pts"] for r in scored]
    cohorts = [r["score"]["cohort_pts"] for r in scored]
    verdict_counts: Dict[str, int] = {}
    for r in scored:
        v = r["score"].get("verdict", "unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    return {
        "n": len(scored),
        "average_total": round(sum(totals) / len(totals), 1),
        "average_geology": round(sum(geos) / len(geos), 1),
        "average_tectonic": round(sum(tects) / len(tects), 1),
        "average_scale": round(sum(scales) / len(scales), 1),
        "average_cohort": round(sum(cohorts) / len(cohorts), 1),
        "min_total": min(totals),
        "max_total": max(totals),
        "below_70": sum(1 for t in totals if t < 70),
        "above_85": sum(1 for t in totals if t >= 85),
        "verdicts": verdict_counts,
    }


def _digest_markdown(
    results: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
) -> str:
    """Render a human-readable digest grouping projects by failure mode."""
    lines: List[str] = []
    lines.append(f"# Analog Quality Judge — Digest\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")

    lines.append(f"## Aggregate\n")
    for k, v in aggregate.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Per-project rows, sorted by score ascending so worst surface first
    scored = [r for r in results if r.get("score")]
    scored.sort(key=lambda r: r["score"]["total"])

    lines.append(f"## Worst 10 projects\n")
    lines.append("| Score | Verdict | Project | Top picks | Missing canonicals |")
    lines.append("|---|---|---|---|---|")
    for r in scored[:10]:
        s = r["score"]
        top = " · ".join(a.get("name") or "?" for a in (r.get("top") or [])[:3])
        missing = "; ".join(
            (m.get("name") or "?") + (f" ({m['district']})" if m.get("district") else "")
            for m in (s.get("missing_canonicals") or [])
        )
        lines.append(
            f"| **{s['total']}** | {s.get('verdict','?')} | "
            f"{r['name']} | {top or '(empty)'} | {missing or '—'} |"
        )
    lines.append("")

    # Aggregate systemic issues across projects
    issues: Dict[str, List[str]] = {}
    for r in scored:
        si = r["score"].get("systemic_issue")
        if si:
            issues.setdefault(si.strip(), []).append(r["name"])
    if issues:
        lines.append(f"## Recurring systemic issues\n")
        for issue, projects in sorted(issues.items(), key=lambda x: -len(x[1])):
            lines.append(f"- **[{len(projects)} project(s)]** {issue}")
            for p in projects[:5]:
                lines.append(f"    - {p}")
        lines.append("")

    # Taxonomy gap suggestions
    gaps: Dict[str, List[str]] = {}
    for r in scored:
        tg = r["score"].get("taxonomy_gap")
        if tg:
            gaps.setdefault(tg.strip(), []).append(r["name"])
    if gaps:
        lines.append(f"## Taxonomy gap suggestions\n")
        for gap, projects in sorted(gaps.items(), key=lambda x: -len(x[1])):
            lines.append(f"- **[{len(projects)} project(s)]** {gap}")
            for p in projects[:5]:
                lines.append(f"    - {p}")
        lines.append("")

    # Canonical analog suggestions (most-named missing analogs across all
    # projects — high-impact library seeding suggestions)
    name_freq: Dict[str, int] = {}
    name_context: Dict[str, set] = {}
    for r in scored:
        for m in (r["score"].get("missing_canonicals") or []):
            n = (m.get("name") or "").strip()
            if not n:
                continue
            name_freq[n] = name_freq.get(n, 0) + 1
            ctx = (m.get("district") or "") + " | " + (m.get("why") or "")
            name_context.setdefault(n, set()).add(ctx.strip(" |"))
    if name_freq:
        lines.append(f"## Most-requested missing analogs\n")
        for name, count in sorted(name_freq.items(), key=lambda x: -x[1])[:20]:
            ctxs = " ; ".join(sorted(c for c in name_context[name] if c))
            lines.append(f"- **{count}×** {name} — {ctxs}")
        lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="gold")
    parser.add_argument("--project-id", help="Score a single specific project")
    parser.add_argument("--sample", type=int, default=None,
                          help="Random sample of N projects")
    parser.add_argument("--all", action="store_true",
                          help="Score every project of the material")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-exa", action="store_true",
                          help="Library-only (faster, deterministic, but Exa-side gaps invisible)")
    parser.add_argument("--prioritize-gaps", action="store_true",
                          help="Sort projects so thin-metadata ones go first")
    parser.add_argument("--out", default=None,
                          help="Output directory (default: reports/judge/<ts>)")
    parser.add_argument("--rate-limit-sleep", type=float, default=0.6,
                          help="Seconds between Grok calls")
    args = parser.parse_args()

    if args.project_id:
        projects = [{"id": args.project_id, "name": "(specified)"}]
    else:
        projects = _list_projects(args.material, args.prioritize_gaps)
        if args.sample:
            if not args.prioritize_gaps:
                random.Random(args.seed).shuffle(projects)
            projects = projects[: args.sample]
        elif not args.all:
            print("Specify --project-id, --sample N, or --all", file=sys.stderr)
            return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else Path("reports/judge") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[judge] Scoring {len(projects)} project(s); output in {out_dir}/")
    if not getattr(settings, "grok_api_key", None):
        print("ERROR: GROK_API_KEY not configured in settings", file=sys.stderr)
        return 2

    results: List[Dict[str, Any]] = []
    for i, p in enumerate(projects, 1):
        name = p.get("name") or p["id"]
        print(f"  [{i:3}/{len(projects)}] {name[:60]}", flush=True)
        try:
            state = _cascade_for(p["id"], with_exa=not args.no_exa)
        except Exception as e:
            logger.warning(f"  ! cascade failed: {e}")
            results.append({"project_id": p["id"], "name": name, "error": str(e)})
            continue
        if state.get("error"):
            results.append({"project_id": p["id"], "name": name,
                              "error": state["error"]})
            continue

        target_profile = state.get("target_profile") or {}
        top = (state.get("scored_analogs") or [])[:4]
        prompt = _build_user_prompt(state["project"], target_profile, top)
        score = _grok(prompt)
        if not score:
            results.append({"project_id": p["id"], "name": name,
                              "error": "grok call failed",
                              "top": top, "target_profile": target_profile})
        else:
            print(f"      → {score.get('total')}/100  ({score.get('verdict','?')})")
            results.append({
                "project_id": p["id"],
                "name": state["project"].get("name") or name,
                "target_profile": target_profile,
                "top": top,
                "score": score,
                "low_confidence": state.get("low_confidence", False),
                "profile_warning": state.get("profile_warning"),
            })
        time.sleep(args.rate_limit_sleep)

    # Persist
    scores_path = out_dir / "scores.json"
    scores_path.write_text(json.dumps({
        "ts": ts,
        "n_projects": len(results),
        "aggregate": _aggregate(results),
        "results": results,
    }, indent=2, default=str))

    digest_path = out_dir / "digest.md"
    digest_path.write_text(_digest_markdown(results, _aggregate(results)))

    print()
    print(f"[judge] Done.")
    print(f"  Scores  : {scores_path}")
    print(f"  Digest  : {digest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
