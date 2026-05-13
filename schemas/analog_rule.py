"""
Pydantic schema for analog_selection rules.

Every rule defined in `scripts/seed_analog_rules.py` is constructed through this
schema at import time. A typo like `"alkalik_porphyry"` becomes an ImportError
on deploy, not a silent runtime miss. Slug fields are validated against the
flat constants in `nodes/geo_taxonomy.py`, so adding a new vocabulary slug in
one place propagates to schema validation automatically.
"""
from __future__ import annotations
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nodes.geo_taxonomy import (
    ALL_SUBTYPE_SLUGS,
    ALL_MODE_SLUGS,
    ALL_BELT_SLUGS,
    ALL_ALTERATION_SLUGS,
    ALL_RECOVERY_SLUGS,
    ALL_PATTERN_SLUGS,
    ALL_HOST_CLASS_SLUGS,
)


# Commodities the system knows about — must match _MATERIAL_TO_RULES_KEYS in
# nodes/supabase_ops.py. The Literal type ensures bad commodities are caught
# at rule-construction time, not at runtime when a project is loaded.
Commodity = Literal[
    "gold", "silver", "gold_silver", "copper", "nickel",
    "uranium", "pgm", "iron",
]


def _validate_slug_list(values: List[str], allowed: frozenset, label: str) -> List[str]:
    """Validate every entry in a slug list against an allowed-set. Raises on miss."""
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(
            f"Unknown {label} slug(s): {bad!r}. "
            f"Valid values come from nodes/geo_taxonomy.py. "
            f"Allowed: {sorted(allowed)}"
        )
    return values


class AnalogRule(BaseModel):
    """A single analog_selection rule.

    Slug fields (required_subtypes, excluded_subtypes, etc.) are validated
    against the geo_taxonomy single-source-of-truth constants. lesson IDs are
    validated against nodes/lessons.LESSONS when that module is available
    (lazy-imported in the validator to avoid circular imports).
    """

    model_config = ConfigDict(extra="forbid")  # reject typos like "required_subtype" (singular)

    # ── Identity ───────────────────────────────────────────────────────────
    rule_id: str = Field(..., min_length=3)
    source_material: Commodity
    deposit_type: str

    # ── Numeric ranges ─────────────────────────────────────────────────────
    grade_min: Optional[float] = None
    grade_max: Optional[float] = None
    grade_unit: Optional[str] = None
    tonnage_min_mt: Optional[float] = None
    tonnage_max_mt: Optional[float] = None
    drilling_stage: Optional[str] = None

    # ── Structured filter directives (used by analog_finder cascade) ───────
    required_subtypes: List[str] = Field(default_factory=list)
    excluded_subtypes: List[str] = Field(default_factory=list)
    required_modes:    List[str] = Field(default_factory=list)
    excluded_modes:    List[str] = Field(default_factory=list)
    required_recovery: List[str] = Field(default_factory=list)
    excluded_recovery: List[str] = Field(default_factory=list)
    preferred_belts:   List[str] = Field(default_factory=list)
    required_belts:    List[str] = Field(default_factory=list)
    preferred_alteration: List[str] = Field(default_factory=list)
    excluded_alteration:  List[str] = Field(default_factory=list)
    # Mineralization pattern (orebody geometry) — vein vs disseminated_bulk vs
    # stockwork vs breccia vs replacement. Drives L4.5 cascade filter.
    required_patterns: List[str] = Field(default_factory=list)
    excluded_patterns: List[str] = Field(default_factory=list)
    # Host rock class — gives carbonate-sediment Carlin vs gneiss-hosted gold
    # the right wedge between them. Drives L4.7 cascade filter.
    required_host_classes: List[str] = Field(default_factory=list)
    excluded_host_classes: List[str] = Field(default_factory=list)
    # Hard tonnage tolerance — drop candidates whose tonnage diverges by more
    # than this multiplicative ratio when both sides have data. None = no cap.
    # See Gold Lesson 136 (>20–25% mismatch penalised heavily).
    tonnage_match_max_ratio: Optional[float] = None
    applies_lessons:   List[str] = Field(default_factory=list)

    # ── Documentation and tuning ───────────────────────────────────────────
    title: str = ""
    description: str = ""
    analog_criteria: List[str] = Field(default_factory=list)
    tonnage_multiplier: float = 1.0
    grade_multiplier: float = 1.0
    confidence_modifier: float = 0.0

    # ── Slug validators ────────────────────────────────────────────────────

    @field_validator("required_subtypes", "excluded_subtypes")
    @classmethod
    def _validate_subtype_slugs(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_SUBTYPE_SLUGS, info.field_name)

    @field_validator("required_modes", "excluded_modes")
    @classmethod
    def _validate_mode_slugs(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_MODE_SLUGS, info.field_name)

    @field_validator("required_recovery", "excluded_recovery")
    @classmethod
    def _validate_recovery_slugs(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_RECOVERY_SLUGS, info.field_name)

    @field_validator("preferred_belts", "required_belts")
    @classmethod
    def _validate_belt_slugs(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_BELT_SLUGS, info.field_name)

    @field_validator("preferred_alteration", "excluded_alteration")
    @classmethod
    def _validate_alteration_slugs(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_ALTERATION_SLUGS, info.field_name)

    @field_validator("required_patterns", "excluded_patterns")
    @classmethod
    def _validate_pattern_slugs(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_PATTERN_SLUGS, info.field_name)

    @field_validator("required_host_classes", "excluded_host_classes")
    @classmethod
    def _validate_host_class_slugs(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_HOST_CLASS_SLUGS, info.field_name)

    @field_validator("tonnage_match_max_ratio")
    @classmethod
    def _validate_tonnage_ratio(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < 1.0:
            raise ValueError(f"tonnage_match_max_ratio must be ≥1.0 (got {v})")
        return v

    @field_validator("applies_lessons")
    @classmethod
    def _validate_lesson_ids(cls, v: List[str]) -> List[str]:
        """Best-effort validation against nodes/lessons.LESSONS. Lazy import so
        the validator doesn't break before the lessons module is built."""
        try:
            from nodes.lessons import LESSONS
        except ImportError:
            return v
        unknown = [lesson_id for lesson_id in v if lesson_id not in LESSONS]
        if unknown:
            raise ValueError(
                f"Unknown lesson ID(s): {unknown!r}. "
                f"Add them to nodes/lessons.LESSONS first."
            )
        return v

    @field_validator("rule_id")
    @classmethod
    def _validate_rule_id(cls, v: str) -> str:
        if not v.startswith(("analog_sel_", "conf_adj_", "model_adj_", "data_qual_")):
            raise ValueError(
                f"rule_id {v!r} must start with analog_sel_ / conf_adj_ / "
                f"model_adj_ / data_qual_"
            )
        return v
