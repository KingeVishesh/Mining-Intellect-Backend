"""Inferred-resource extractor.

For each analog used in a model run, this module pulls the published
M&I + Inferred breakdown from the most recent technical report (NI 43-101,
JORC, SK-1300) via Exa's Answer API. The result is cached on the
`public.analogs` row (`analog_inferred_tonnage_mt`, `analog_inferred_grade`)
so subsequent runs reuse it.

Background: `build_model_1`'s Inferred axis predicts the project's
Inferred bucket independently of the M&I axis, using the analog pool's
`inferred_tonnage_mt` / `inferred_grade` values. Until this extractor
was added, those fields existed only on backtest fixture JSON files and
the production library returned 0 for every project's Inferred prediction.

Shape returned by `extract_inferred_breakdown()`:

    {
      "inferred_tonnage_mt": float | None,    # Inferred-bucket tonnage in Mt
      "inferred_grade":      float | None,    # Inferred-bucket grade (native unit)
      "mi_tonnage_mt":       float | None,    # M&I total for cross-validation
      "mi_grade":            float | None,
      "mi_category_basis":    str | None,      # measured_plus_indicated, etc.
      "as_of_year":          int | None,      # year of the MRE/PEA cited
      "source_url":          str | None,
      "confidence":          "high" | "medium" | "low",
      "source":              "exa",
      "extracted_at":        ISO 8601 timestamp,
    }

If the analog publishes ONLY Inferred (no M&I yet) or ONLY M&I (no
Inferred halo remaining), the corresponding field is None — that IS
valid data; the build_model_1 Inferred-axis just ignores rows where
`inferred_tonnage_mt` is None when computing its geometric mean.
"""
from __future__ import annotations
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

EXA_ANSWER_URL = "https://api.exa.ai/answer"

# Inferred breakdowns change only when the operator publishes a new MRE,
# which is typically annual. A 90-day cache is comfortable.
DEFAULT_MAX_AGE_DAYS = 90


def should_refetch(
    inferred_data: Optional[Dict],
    fetched_at: Optional[str],
    force: bool = False,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> bool:
    """Return True when the analog's Inferred breakdown should be refetched.

    Missing data → always refetch. Forced (caller passed an explicit flag)
    → always refetch. Otherwise refetch only when the cached timestamp is
    older than `max_age_days`.
    """
    if force:
        return True
    if not inferred_data and not fetched_at:
        return True
    if not fetched_at:
        return True
    try:
        ts = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return True
    age = datetime.now(timezone.utc) - ts
    return age > timedelta(days=max_age_days)


# Cross-validation tolerance for two-pass extraction. When the two passes
# both return numeric values, we compare them in log space. ≤5% relative
# difference is "high" agreement, ≤20% is "medium", >20% is "low" and the
# value is flagged for human review (the row is persisted with NULL so it
# isn't fed to the model until manually verified).
_CROSS_VAL_HIGH_RDIFF = 0.05
_CROSS_VAL_MED_RDIFF  = 0.20


def _rel_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Relative difference |a-b| / mean(|a|,|b|). Returns None when either
    is missing. Used by the cross-validation step to flag disagreements
    between the two independent Exa queries."""
    if a is None or b is None:
        return None
    aa, bb = abs(float(a)), abs(float(b))
    if aa + bb == 0:
        return 0.0
    return abs(float(a) - float(b)) / ((aa + bb) / 2.0)


def _consensus(
    primary: Optional[float], secondary: Optional[float],
) -> Tuple[Optional[float], str, Optional[float]]:
    """Pick a consensus value from the two extraction passes.

    Returns (value, confidence, rel_diff). When the passes agree closely
    take the mean; when they disagree return the primary but flag low
    confidence; when only one pass returned a value, use it but cap
    confidence at "medium".
    """
    if primary is None and secondary is None:
        return None, "none", None
    if primary is None:
        return float(secondary), "medium", None
    if secondary is None:
        return float(primary), "medium", None
    rd = _rel_diff(primary, secondary) or 0.0
    if rd <= _CROSS_VAL_HIGH_RDIFF:
        return (float(primary) + float(secondary)) / 2.0, "high", rd
    if rd <= _CROSS_VAL_MED_RDIFF:
        return (float(primary) + float(secondary)) / 2.0, "medium", rd
    return float(primary), "low", rd


def _normalise_mi_basis(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    key = key.replace("_+_", "_plus_").replace("+", "plus")
    aliases = {
        "m_i": "measured_plus_indicated",
        "m&i": "measured_plus_indicated",
        "measured_indicated": "measured_plus_indicated",
        "measured_and_indicated": "measured_plus_indicated",
        "measured_plus_indicated": "measured_plus_indicated",
        "indicated": "indicated_only",
        "indicated_only": "indicated_only",
        "measured": "measured_only",
        "measured_only": "measured_only",
        "unknown": "unknown",
    }
    return aliases.get(key, key if key else None)


def _consensus_mi_basis(pass_a: Dict, pass_b: Dict) -> Optional[str]:
    bases = [
        basis
        for basis in (
            _normalise_mi_basis(pass_a.get("mi_category_basis")),
            _normalise_mi_basis(pass_b.get("mi_category_basis")),
        )
        if basis
    ]
    if not bases:
        return None
    if "measured_plus_indicated" in bases:
        return "measured_plus_indicated"
    if "indicated_only" in bases:
        return "indicated_only"
    if "measured_only" in bases:
        return "measured_only"
    return "unknown"


def _single_query(
    api_key: str,
    query: str,
    system_prompt: str,
    analog_name: str,
) -> Optional[Dict]:
    """Single Exa Answer pass with the shared output schema. Returns the
    parsed JSON (or None on any failure). Errors are logged, not raised."""
    payload = {
        "query": query,
        "system_prompt": system_prompt,
        "output_schema": {
            "type": "object",
            "properties": {
                "mi_tonnage_mt":       {"type": ["number", "null"]},
                "mi_grade":            {"type": ["number", "null"]},
                "mi_category_basis":    {
                    "type": ["string", "null"],
                    "description": (
                        "Basis for mi_tonnage_mt/mi_grade. Use "
                        "measured_plus_indicated only when the source reports "
                        "a combined M&I row or you explicitly summed Measured "
                        "+ Indicated tonnage and tonnage-weighted the grade. "
                        "Use measured_only if the value is only a standalone "
                        "Measured row."
                    ),
                },
                "inferred_tonnage_mt": {"type": ["number", "null"]},
                "inferred_grade":      {"type": ["number", "null"]},
                "as_of_year":          {"type": ["integer", "null"]},
                "source_url":          {"type": ["string", "null"]},
                "publisher":           {"type": ["string", "null"],
                                        "description": "Name of the publishing operator/company."},
                "confidence":          {"type": "string",
                                        "enum": ["high", "medium", "low"]},
            },
            "required": ["confidence"],
        },
        "text": False,
    }
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    try:
        resp = requests.post(EXA_ANSWER_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        logger.warning(f"[InferredExtractor] Request error for '{analog_name}': {e}")
        return None
    if resp.status_code != 200:
        logger.warning(
            f"[InferredExtractor] HTTP {resp.status_code} for '{analog_name}': "
            f"{resp.text[:200]}"
        )
        return None
    raw = resp.json().get("answer")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


def extract_inferred_breakdown(
    analog_name: str,
    material: str,
    country: Optional[str] = None,
    region: Optional[str] = None,
    deposit_type: Optional[str] = None,
) -> Optional[Dict]:
    """Query Exa Answer TWICE with different prompts, cross-validate, and
    return the consensus M&I + Inferred breakdown.

    Why two passes: a single-query result from an LLM-backed extractor can
    hallucinate or pick stale figures. Issuing two independent queries
    (one company-filings-first, one technical-report-first) and comparing
    their numeric outputs surfaces disagreements that would otherwise slip
    through. The cross-validation produces a per-field confidence flag —
    "high" (both agree within 5%), "medium" (within 20% or only one
    returned), "low" (>20% disagreement; flagged for review).

    Returns None on total extraction failure. Errors are logged, never
    raised — the model graph tolerates a missing analog gracefully.
    """
    api_key = settings.exa_api_key
    if not api_key:
        logger.warning("[InferredExtractor] EXA_API_KEY not set, skipping fetch")
        return None

    loc_parts = [p for p in (region, country) if p and p.strip()]
    location = ", ".join(loc_parts) if loc_parts else ""
    deposit_clause = f"({deposit_type})" if deposit_type else ""

    # ── Pass 1: company-disclosure-first framing ──────────────────────────
    query_a = (
        f"Latest published Mineral Resource Estimate for the {analog_name} "
        f"{material} mining project in {location} {deposit_clause}. "
        f"From the operator's most recent annual MRE statement or NI 43-101 / "
        f"JORC / SK-1300 technical report, report SEPARATELY: "
        f"(a) the Measured + Indicated (M&I) tonnage in Mt and grade in g/t "
        f"(for Au/Ag/PGM) or % (for Cu/Zn/Pb/Ni/Mo/U), and "
        f"(b) the Inferred tonnage in Mt and grade in same units. "
        f"If Measured and Indicated are shown as separate rows, add their "
        f"tonnages and calculate a tonnage-weighted grade; never use only "
        f"the standalone Measured row as M&I. "
        f"Cite the source URL and the publication year. "
        f"If only Inferred is reported (early-stage) or only M&I/Reserves "
        f"(near-depleted producer), return null for the missing category."
    )
    prompt_a = (
        "You are a mining-industry analyst extracting Mineral Resource "
        "Estimate (MRE) figures from operator disclosures. Return values "
        "EXACTLY as published in the source. Do NOT estimate, interpolate, "
        "or fabricate. Prefer the most recent annual MRE update. If you "
        "cannot find a specific figure in the source you cite, return null "
        "for that field rather than guessing. For M&I, combine separate "
        "Measured and Indicated rows into one tonnage-weighted figure. "
        "Never treat a standalone Measured row as M&I when an Indicated row "
        "is also present."
    )
    # ── Pass 2: technical-report-first framing ────────────────────────────
    query_b = (
        f"In the most recent NI 43-101 (or JORC / SK-1300) technical report "
        f"for {analog_name} ({material}, {location}), what is the reported "
        f"Mineral Resource breakdown by category? "
        f"Specifically: Measured tonnage + grade, Indicated tonnage + grade, "
        f"Inferred tonnage + grade. Combine Measured + Indicated into a "
        f"single tonnage-weighted (M&I) figure if not already aggregated. "
        f"Cite the technical-report URL and the effective date."
    )
    prompt_b = (
        "You are a NI 43-101 / JORC compliance reviewer cross-checking MRE "
        "figures against the source technical report. Quote tonnage in Mt "
        "(NOT kt or t) and grade in g/t for Au/Ag/PGM or % for base metals. "
        "Combine Measured and Indicated into a single M&I total weighted by "
        "tonnage. Never return only the Measured row as M&I if an Indicated "
        "row is present. Return null for any category not reported in the technical "
        "report you cite."
    )

    pass_a = _single_query(api_key, query_a, prompt_a, analog_name)
    pass_b = _single_query(api_key, query_b, prompt_b, analog_name)

    if pass_a is None and pass_b is None:
        logger.warning(f"[InferredExtractor] Both passes failed for '{analog_name}'")
        return None

    pass_a = pass_a or {}
    pass_b = pass_b or {}

    # Cross-validate each numeric field independently.
    mi_t,  mi_t_conf,  mi_t_rd  = _consensus(pass_a.get("mi_tonnage_mt"),
                                             pass_b.get("mi_tonnage_mt"))
    mi_g,  mi_g_conf,  mi_g_rd  = _consensus(pass_a.get("mi_grade"),
                                             pass_b.get("mi_grade"))
    inf_t, inf_t_conf, inf_t_rd = _consensus(pass_a.get("inferred_tonnage_mt"),
                                             pass_b.get("inferred_tonnage_mt"))
    inf_g, inf_g_conf, inf_g_rd = _consensus(pass_a.get("inferred_grade"),
                                             pass_b.get("inferred_grade"))

    # Source URL: prefer pass_b (technical-report framing) since it's more
    # likely to cite the primary source. Fall back to pass_a, then to
    # whichever pass surfaced anything.
    source_url = (pass_b.get("source_url") or pass_a.get("source_url"))
    publisher  = (pass_b.get("publisher")  or pass_a.get("publisher"))
    as_of_year = (pass_b.get("as_of_year") or pass_a.get("as_of_year"))
    mi_basis = _consensus_mi_basis(pass_a, pass_b)

    # Overall confidence is the MIN across the four fields — one
    # disagreement on any field is enough to flag the row as low confidence.
    conf_ranks = {"high": 2, "medium": 1, "low": 0, "none": -1}
    field_confs = [c for c in (mi_t_conf, mi_g_conf, inf_t_conf, inf_g_conf)
                   if c != "none"]
    overall_conf = (
        min(field_confs, key=lambda c: conf_ranks[c])
        if field_confs else "low"
    )

    result = {
        "mi_tonnage_mt":       mi_t,
        "mi_grade":            mi_g,
        "inferred_tonnage_mt": inf_t,
        "inferred_grade":      inf_g,
        "as_of_year":          as_of_year,
        "source_url":          source_url,
        "publisher":           publisher,
        "mi_category_basis":    mi_basis,
        "confidence":          overall_conf,
        "cross_validation": {
            "mi_category_basis":    {"pass_a": pass_a.get("mi_category_basis"),
                                     "pass_b": pass_b.get("mi_category_basis"),
                                     "consensus": mi_basis},
            "mi_tonnage_mt":       {"pass_a": pass_a.get("mi_tonnage_mt"),
                                    "pass_b": pass_b.get("mi_tonnage_mt"),
                                    "consensus": mi_t,
                                    "rel_diff": mi_t_rd,
                                    "confidence": mi_t_conf},
            "mi_grade":            {"pass_a": pass_a.get("mi_grade"),
                                    "pass_b": pass_b.get("mi_grade"),
                                    "consensus": mi_g,
                                    "rel_diff": mi_g_rd,
                                    "confidence": mi_g_conf},
            "inferred_tonnage_mt": {"pass_a": pass_a.get("inferred_tonnage_mt"),
                                    "pass_b": pass_b.get("inferred_tonnage_mt"),
                                    "consensus": inf_t,
                                    "rel_diff": inf_t_rd,
                                    "confidence": inf_t_conf},
            "inferred_grade":      {"pass_a": pass_a.get("inferred_grade"),
                                    "pass_b": pass_b.get("inferred_grade"),
                                    "consensus": inf_g,
                                    "rel_diff": inf_g_rd,
                                    "confidence": inf_g_conf},
        },
        "source":              "exa_2pass",
        "extracted_at":        datetime.now(timezone.utc).isoformat(),
    }

    # Drop fields flagged "low" so the model doesn't ingest values that
    # disagreed by >20% between passes. The cross-validation block keeps
    # the raw pass_a / pass_b values for human review (via the verification
    # CLI or a database audit query).
    for key, conf in (
        ("mi_tonnage_mt", mi_t_conf), ("mi_grade", mi_g_conf),
        ("inferred_tonnage_mt", inf_t_conf), ("inferred_grade", inf_g_conf),
    ):
        if conf == "low":
            result[key] = None

    if mi_basis == "measured_only":
        result["mi_tonnage_mt"] = None
        result["mi_grade"] = None
        result["confidence"] = "low"
        result["cross_validation"]["mi_category_basis"]["confidence"] = "low"
        result["cross_validation"]["mi_category_basis"]["reason"] = (
            "Extractor captured only a standalone Measured row, not a combined "
            "Measured + Indicated M&I figure."
        )

    # Nothing usable at all? Mark as failed.
    if (
        result["mi_tonnage_mt"] is None
        and result["inferred_tonnage_mt"] is None
    ):
        logger.info(
            f"[InferredExtractor] No MRE breakdown found for '{analog_name}' "
            f"(2-pass; confidence={overall_conf})"
        )
        return None

    logger.info(
        f"[InferredExtractor] '{analog_name}' (2-pass): "
        f"M&I {result['mi_tonnage_mt']} Mt @ {result['mi_grade']}, "
        f"Inferred {result['inferred_tonnage_mt']} Mt @ {result['inferred_grade']}, "
        f"as_of={as_of_year}, overall_confidence={overall_conf}, "
        f"source={source_url}"
    )
    return result
