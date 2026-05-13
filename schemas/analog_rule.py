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
    ALL_STAGE_SLUGS,
    ALL_MINING_METHOD_SLUGS,
    ALL_RESOURCE_CATEGORY_SLUGS,
    ALL_SUITE_SLUGS,
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
    grade_match_max_ratio:   Optional[float] = None
    # Project stage — L4.6 cascade gate. Resource-stage analogs for an
    # exploration-stage target give over-confident models.
    required_stages: List[str] = Field(default_factory=list)
    excluded_stages: List[str] = Field(default_factory=list)
    # Mining method — L4.8 cascade gate. Open-pit bulk analogs for an
    # underground-vein target gives wrong cut-off / dilution / recovery.
    required_mining_methods: List[str] = Field(default_factory=list)
    excluded_mining_methods: List[str] = Field(default_factory=list)
    # Resource category — L4.9 cascade gate. Inferred-only analogs for an
    # M&I-stage model are too weak.
    min_resource_category: Optional[str] = None     # e.g. "m_and_i"
    excluded_resource_categories: List[str] = Field(default_factory=list)
    # Resource vintage — L4.95 cascade gate. Historical / press-release / pre-2010
    # estimates are blocked when the rule sets min_resource_year.
    min_resource_year: Optional[int] = None
    # Metal suite gating (was rank-only; now optional hard filter)
    required_metal_suites: List[str] = Field(default_factory=list)
    excluded_metal_suites: List[str] = Field(default_factory=list)
    # Per-rule profile-strength minimum (default 4 of 10 dimensions).
    min_profile_strength: int = 4
    # Rule priority — higher = checked first when multiple rules match.
    # Sub-rules (alkalic_porphyry vs generic porphyry) get higher priority.
    rule_priority: int = 100
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

    @field_validator("tonnage_match_max_ratio", "grade_match_max_ratio")
    @classmethod
    def _validate_match_ratio(cls, v: Optional[float], info) -> Optional[float]:
        if v is None:
            return v
        if v < 1.0:
            raise ValueError(f"{info.field_name} must be ≥1.0 (got {v})")
        return v

    @field_validator("required_stages", "excluded_stages")
    @classmethod
    def _validate_stage_slugs(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_STAGE_SLUGS, info.field_name)

    @field_validator("required_mining_methods", "excluded_mining_methods")
    @classmethod
    def _validate_mining_method_slugs(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_MINING_METHOD_SLUGS, info.field_name)

    @field_validator("min_resource_category")
    @classmethod
    def _validate_min_resource_category(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in ALL_RESOURCE_CATEGORY_SLUGS:
            raise ValueError(
                f"Unknown min_resource_category slug: {v!r}. "
                f"Valid: {sorted(ALL_RESOURCE_CATEGORY_SLUGS)}"
            )
        return v

    @field_validator("excluded_resource_categories")
    @classmethod
    def _validate_excluded_categories(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_RESOURCE_CATEGORY_SLUGS, info.field_name)

    @field_validator("required_metal_suites", "excluded_metal_suites")
    @classmethod
    def _validate_metal_suite_slugs(cls, v: List[str], info) -> List[str]:
        return _validate_slug_list(v, ALL_SUITE_SLUGS, info.field_name)

    @field_validator("min_resource_year")
    @classmethod
    def _validate_min_year(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 1900 or v > 2100:
            raise ValueError(f"min_resource_year out of range (got {v})")
        return v

    @field_validator("min_profile_strength")
    @classmethod
    def _validate_min_strength(cls, v: int) -> int:
        if v < 0 or v > 10:
            raise ValueError(f"min_profile_strength must be in [0,10] (got {v})")
        return v

    @field_validator("rule_priority")
    @classmethod
    def _validate_priority(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"rule_priority must be ≥0 (got {v})")
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
