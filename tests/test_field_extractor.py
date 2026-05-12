"""
LLM-bounded extraction tests.

Validate that:
  1. Grok's controlled-vocab extractions are gated through `_validate()` and
     hallucinations get dropped (then heuristics fill the gap).
  2. The heuristic fallback identifies the right slug from realistic freeform
     deposit_type / mineralization_style text.
  3. _VALID_SUBTYPES is NOT re-defined locally (single source of truth lives
     in nodes/geo_taxonomy.py).

These tests do NOT call the real Grok API — they exercise the validators and
heuristics directly.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from nodes import geo_taxonomy
from nodes.field_extractor import _validate, _fill_geological_profile, _fill_analog_profile


# ── Validator: hallucination rejection ──────────────────────────────────────


def test_validate_rejects_hallucinated_subtype():
    """A made-up slug from Grok must be rejected."""
    result = _validate("unicorn_porphyry", geo_taxonomy.ALL_SUBTYPE_SLUGS)
    assert result is None


def test_validate_accepts_known_subtype():
    result = _validate("alkalic_porphyry", geo_taxonomy.ALL_SUBTYPE_SLUGS)
    assert result == "alkalic_porphyry"


def test_validate_normalizes_punctuation():
    """Grok occasionally returns 'iocg-oxide' or 'iocg oxide'; normalise."""
    assert _validate("iocg oxide", geo_taxonomy.ALL_SUBTYPE_SLUGS) == "iocg_oxide"
    assert _validate("iocg-oxide", geo_taxonomy.ALL_SUBTYPE_SLUGS) == "iocg_oxide"


def test_validate_rejects_mode_outside_vocab():
    assert _validate("hyper_supergene", geo_taxonomy.ALL_MODE_SLUGS) is None
    assert _validate("primary_sulfide", geo_taxonomy.ALL_MODE_SLUGS) == "primary_sulfide"


# ── Heuristic fallback when Grok returns null/invalid ───────────────────────


def test_fill_geological_profile_runs_heuristic_when_grok_blank():
    """All 6 fields null → heuristics fill from freeform text."""
    fields = {
        # Profile fields all null (Grok returned nothing useful)
        "deposit_subtype": None, "mineralization_mode": None,
        "tectonic_belt": None, "metal_suite": None,
        "alteration_signature": None, "recovery_method": None,
        # Freeform text the heuristic can use
        "deposit_type": "alkalic porphyry copper-gold",
        "mineralization_style": "stockwork sulphide",
        "country": "Canada", "region": "British Columbia",
        "district": "Stikine Golden Triangle",
        "processing_method": "flotation",
        "location_name": "Stikine BC",
        "by_product_commodities": ["Gold"],
    }
    result = _fill_geological_profile(fields, "copper")
    assert result["deposit_subtype"] == "alkalic_porphyry"
    assert result["mineralization_mode"] == "primary_sulfide"
    assert result["tectonic_belt"] == "bc_quesnel_stikine"
    assert result["recovery_method"] == "flotation"
    assert result["alteration_signature"] is None or isinstance(result["alteration_signature"], str)


def test_fill_geological_profile_drops_hallucinated_subtype():
    """When Grok returns 'unicorn_porphyry', validator nukes it AND heuristic still hits."""
    fields = {
        "deposit_subtype": "unicorn_porphyry",  # hallucination
        "mineralization_mode": "magic_phase",   # hallucination
        "tectonic_belt": "narnia_belt",          # hallucination
        "metal_suite": None, "alteration_signature": None, "recovery_method": None,
        "deposit_type": "alkalic porphyry copper-gold",
        "mineralization_style": "stockwork sulphide",
        "country": "Canada", "region": "British Columbia",
        "district": "Stikine", "processing_method": "flotation",
        "by_product_commodities": [],
    }
    result = _fill_geological_profile(fields, "copper")
    assert result["deposit_subtype"] == "alkalic_porphyry"
    assert result["mineralization_mode"] == "primary_sulfide"
    assert result["tectonic_belt"] == "bc_quesnel_stikine"


def test_fill_analog_profile_routes_florence_to_iscr():
    """Florence's deposit_type + processing route should map to oxide_iscr."""
    analog = {
        "name": "Florence", "deposit_type": "porphyry oxide ISCR",
        "mineralization_style": "supergene oxide blanket chrysocolla",
        "country": "USA", "region": "Arizona", "district": "Pinal",
        "processing_method": "in-situ copper recovery SX-EW",
    }
    _fill_analog_profile(analog, "copper")
    assert analog["deposit_subtype"] == "oxide_iscr_supergene_blanket"
    assert analog["mineralization_mode"] == "supergene_oxide"
    assert analog["recovery_method"] == "iscr"
    assert analog["tectonic_belt"] == "laramide_southwest"


def test_fill_analog_profile_routes_kamoa_to_sediment_hosted():
    """Stratabound + Katanga must hit Central African Copperbelt / sediment-hosted."""
    analog = {
        "name": "Kamoa-Kakula",
        "deposit_type": "stratabound high-grade underground copper",
        "country": "DR Congo", "district": "Katanga Copperbelt",
        "processing_method": "flotation",
    }
    _fill_analog_profile(analog, "copper")
    assert analog["deposit_subtype"] in {"kupferschiefer_style", "sediment_hosted_general"}
    assert analog["tectonic_belt"] == "central_african_copperbelt"


# ── Architecture guard ──────────────────────────────────────────────────────


def test_no_local_vocab_duplication():
    """Make sure nobody re-introduced a hardcoded _VALID_SUBTYPES (drift risk)."""
    import nodes.field_extractor as fe
    assert not hasattr(fe, "_VALID_SUBTYPES")
    # Both fill functions must reference the taxonomy-imported sets
    src = (Path(fe.__file__)).read_text()
    assert "ALL_SUBTYPE_SLUGS" in src
    assert "ALL_MODE_SLUGS" in src
    assert "ALL_BELT_SLUGS" in src
