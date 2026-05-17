"""
Gap detector — deterministic, in-cascade diagnostics that flag systemic
weaknesses in the analog finder for a single project run.

Emits structured records to the `analog_quality_gaps` table. The goal is
not to BLOCK the cascade (it already runs to completion); it's to give
the weekly quality digest concrete, queryable evidence of WHY a project
ended up with mediocre analogs — so the next fix is targeted at the
right vocabulary / library / ranking gap instead of being a one-off
patch.

Gap types
---------
- taxonomy_sub_trend_missing
    Target has a tectonic_belt but detect_sub_trend() returned None.
    Means SUB_TRENDS doesn't cover this project's location yet — every
    other project in the same neighborhood will also miss sub-trend
    ranking. Fix: add keywords to nodes/geo_taxonomy.SUB_TRENDS.

- library_coverage_thin
    Library returned <N candidates for the rule's required_subtypes,
    indicating insufficient canonical analogs seeded for this geology.
    Fix: seed missing analogs OR rely entirely on Exa.

- ranking_out_of_trend_leader
    Top-1 scored analog doesn't share the target's sub_trend even
    though the target HAS a sub_trend. Means the L6.5 bonus didn't win
    — either because no in-sub-trend candidate exists in library/Exa
    OR because other ranking signals overrode it.

- library_metadata_incomplete
    A library candidate that passed the cascade is missing one of the
    structured fields the cascade scores on (e.g. analog_district is
    null, so sub_trend can't resolve). Each missing field on a passing
    candidate is a small ranking weakness compounded across the run.

- low_profile_strength
    Target's profile strength is below the rule's min_profile_strength
    and the cascade fell back to relaxed mode. Means the project
    research step left key fields empty; analog quality will be poor
    regardless of cascade tuning.

- relaxed_mode_fallback
    Cascade ran in relaxed mode for any reason. Surfaces alongside
    low_profile_strength so the user can see which path was taken.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Thresholds — adjust here if a category becomes too noisy or too quiet.
# Each is the boundary BELOW which we emit the gap.
LIBRARY_THIN_THRESHOLD = 3       # < this many library candidates → thin
LIBRARY_THIN_SEVERITY  = "high"  # missing library coverage matters a lot


def detect_gaps(
    project: Dict[str, Any],
    target_profile: Dict[str, Any],
    analog_rule: Optional[Dict[str, Any]],
    library_analogs: List[Dict[str, Any]],
    scored_analogs: List[Dict[str, Any]],
    low_confidence: bool = False,
    relaxed_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return a list of gap records for this cascade run. Each record has:
        project_id   — the target project's ID
        rule_id      — the matched analog_selection rule (or None)
        gap_type     — one of the slugs in the module docstring
        severity     — low / medium / high
        details      — JSON blob with the specific evidence
        suggestion   — short text the human can act on
    Records are returned, not written — the caller (combine_filter_score_node)
    decides whether to persist them.
    """
    gaps: List[Dict[str, Any]] = []
    project_id = project.get("id") or project.get("project_id") or ""
    rule_id = (analog_rule or {}).get("rule_id")

    gaps.extend(_taxonomy_sub_trend(project_id, rule_id, target_profile))
    gaps.extend(_library_coverage(project_id, rule_id, target_profile, library_analogs))
    gaps.extend(_ranking_out_of_trend(project_id, rule_id, target_profile, scored_analogs))
    gaps.extend(_library_metadata(project_id, rule_id, target_profile, scored_analogs))
    if relaxed_mode:
        gaps.extend(_relaxed_mode(project_id, rule_id, target_profile))
    if low_confidence and not relaxed_mode:
        # low_confidence + not relaxed = the strict no-rule path; surfaces
        # missing structured data rather than fallback behavior.
        gaps.extend(_low_profile_strength(project_id, rule_id, target_profile))

    return gaps


# ── Individual detectors ─────────────────────────────────────────────────────

def _taxonomy_sub_trend(
    project_id: str,
    rule_id: Optional[str],
    target: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Target has a belt but no sub_trend resolved → SUB_TRENDS is missing
    coverage for this project's geological neighborhood."""
    belt = target.get("tectonic_belt")
    sub_trend = target.get("sub_trend")
    if not belt or sub_trend:
        return []
    return [{
        "project_id": project_id,
        "rule_id": rule_id,
        "gap_type": "taxonomy_sub_trend_missing",
        "severity": "medium",
        "details": {
            "tectonic_belt": belt,
            "district": target.get("district"),
            "country": target.get("country"),
        },
        "suggestion": (
            f"Add district/sub-camp keywords for belt={belt!r} to "
            f"nodes.geo_taxonomy.SUB_TRENDS so the cascade can use the "
            f"L6.5 sub-trend ranking signal for this and similar "
            f"projects in the same neighborhood."
        ),
    }]


def _library_coverage(
    project_id: str,
    rule_id: Optional[str],
    target: Dict[str, Any],
    library_analogs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Library returned fewer than LIBRARY_THIN_THRESHOLD candidates for
    the matched rule. Means we lean entirely on Exa for this geology;
    if Exa flakes, the project gets nothing."""
    n = len(library_analogs or [])
    if n >= LIBRARY_THIN_THRESHOLD:
        return []
    return [{
        "project_id": project_id,
        "rule_id": rule_id,
        "gap_type": "library_coverage_thin",
        "severity": LIBRARY_THIN_SEVERITY,
        "details": {
            "library_candidate_count": n,
            "threshold": LIBRARY_THIN_THRESHOLD,
            "subtype": target.get("deposit_subtype"),
            "tectonic_belt": target.get("tectonic_belt"),
            "sub_trend": target.get("sub_trend"),
        },
        "suggestion": (
            f"Library has only {n} approved analog(s) for "
            f"rule={rule_id!r}, subtype={target.get('deposit_subtype')!r}, "
            f"belt={target.get('tectonic_belt')!r}. Seed canonical "
            f"examples for this geology, or improve the Exa query hint "
            f"so the search reliably surfaces them."
        ),
    }]


def _ranking_out_of_trend(
    project_id: str,
    rule_id: Optional[str],
    target: Dict[str, Any],
    scored: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Top-1 scored analog doesn't share the target's sub_trend even
    though both should. Diagnostic for sub-trend ranking failure."""
    if not scored:
        return []
    target_st = target.get("sub_trend")
    if not target_st:
        return []
    # Detect the top analog's sub_trend from its district/region text.
    # We can't trust an already-stored sub_trend field on the scored
    # dict because it may not exist; re-derive deterministically.
    from nodes.geo_taxonomy import detect_sub_trend
    top = scored[0]
    top_st = detect_sub_trend(
        top.get("district"), top.get("region"), top.get("location_name"),
    )
    if top_st == target_st:
        return []
    return [{
        "project_id": project_id,
        "rule_id": rule_id,
        "gap_type": "ranking_out_of_trend_leader",
        "severity": "high",
        "details": {
            "target_sub_trend": target_st,
            "top_analog_name": top.get("name"),
            "top_analog_district": top.get("district"),
            "top_analog_sub_trend": top_st,
        },
        "suggestion": (
            f"Top-1 analog {top.get('name')!r} resolved to "
            f"sub_trend={top_st!r} but target is {target_st!r}. Either "
            f"library lacks in-sub-trend canonicals (seed them or "
            f"improve the Exa hint) OR the L6.5 +25 bonus isn't enough "
            f"to override other ranking signals."
        ),
    }]


def _library_metadata(
    project_id: str,
    rule_id: Optional[str],
    target: Dict[str, Any],
    scored: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Each passing library candidate missing a structured field that the
    cascade scores on is a small ranking weakness. Most-impactful gaps
    are belt and sub-trend (via district)."""
    if not scored:
        return []
    incomplete = []
    for a in scored:
        if a.get("source") != "library":
            continue
        missing = []
        if not a.get("tectonic_belt"):
            missing.append("tectonic_belt")
        if not (a.get("district") or "").strip():
            missing.append("district")
        if not a.get("mineralization_pattern"):
            missing.append("mineralization_pattern")
        if not a.get("mining_method_class"):
            missing.append("mining_method_class")
        if missing:
            incomplete.append({
                "name": a.get("name"),
                "missing_fields": missing,
            })
    if not incomplete:
        return []
    return [{
        "project_id": project_id,
        "rule_id": rule_id,
        "gap_type": "library_metadata_incomplete",
        "severity": "low",
        "details": {"incomplete_candidates": incomplete},
        "suggestion": (
            f"{len(incomplete)} library candidate(s) that passed the "
            f"cascade are missing structured fields the ranker uses. "
            f"Backfill via scripts/backfill_geological_profiles.py or "
            f"update the seed entries directly."
        ),
    }]


def _low_profile_strength(
    project_id: str,
    rule_id: Optional[str],
    target: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Cascade returned low_confidence without falling into relaxed mode
    — means the no-rule path fired because deposit_type and
    deposit_subtype are both empty on the target. Upstream fix needed:
    re-run project_research or set the fields manually."""
    return [{
        "project_id": project_id,
        "rule_id": rule_id,
        "gap_type": "low_profile_strength",
        "severity": "high",
        "details": {
            "deposit_type": target.get("deposit_type"),
            "deposit_subtype": target.get("deposit_subtype"),
            "material": target.get("material"),
        },
        "suggestion": (
            "Project has no deposit_type and no deposit_subtype; cascade "
            "refused to score. Re-run the project_research graph or set "
            "the fields manually so a rule can match."
        ),
    }]


def _relaxed_mode(
    project_id: str,
    rule_id: Optional[str],
    target: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Cascade ran in relaxed mode (profile strength below rule's min).
    Surfaces alongside data-gap issues so the user sees which path was
    taken and can decide whether to enrich the project."""
    return [{
        "project_id": project_id,
        "rule_id": rule_id,
        "gap_type": "relaxed_mode_fallback",
        "severity": "medium",
        "details": {
            "deposit_subtype": target.get("deposit_subtype"),
            "tectonic_belt": target.get("tectonic_belt"),
            "host_rock_class": target.get("host_rock_class"),
            "mining_method_class": target.get("mining_method_class"),
            "mineralization_pattern": target.get("mineralization_pattern"),
        },
        "suggestion": (
            "Cascade ran in relaxed mode (rule's required_* lists were "
            "dropped due to thin target profile). Enrich the project to "
            "raise confidence — fill in the missing structured fields "
            "above so the cascade can apply rule-level filters."
        ),
    }]


# ── Persistence ──────────────────────────────────────────────────────────────

def save_gaps(gaps: List[Dict[str, Any]]) -> None:
    """Bulk-insert gap records to analog_quality_gaps. Non-fatal — gap
    detection must never break the cascade itself, so this swallows
    errors and logs them."""
    if not gaps:
        return
    try:
        from nodes.supabase_ops import get_client
        rows = []
        for g in gaps:
            rows.append({
                "project_id": str(g.get("project_id") or ""),
                "rule_id": g.get("rule_id"),
                "gap_type": g["gap_type"],
                "severity": g.get("severity", "medium"),
                "details": g.get("details", {}),
                "suggestion": g.get("suggestion"),
            })
        # 50-row batches keep us well under the Supabase payload limit.
        BATCH = 50
        client = get_client()
        for i in range(0, len(rows), BATCH):
            client.table("analog_quality_gaps").insert(rows[i:i + BATCH]).execute()
        logger.info(f"[gap_detector] persisted {len(rows)} gap record(s)")
    except Exception as e:
        logger.warning(f"[gap_detector] persist failed (non-fatal): {e}")
