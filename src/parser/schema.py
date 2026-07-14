"""Pydantic models for the Resolution Spec (v0.5.0)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ENTITY_TYPES = Literal[
    "person",
    "organization",
    "location",
    "instrument",
    "team",
    "event",
    "other",
]

FUNCTION_TYPES = Literal["binary", "threshold", "conditional"]

MECHANISM_TYPES = Literal[
    "bellwether",
    "centralized_platform",
    "uma",
    "other",
    "unknown",
]

TIER_REQUIREMENTS = Literal[
    "tier_1_only",
    "tier_2_or_better",
    "any_credible",
    "unspecified",
]

# Heightened-risk categories per DMO Advisory 26-08 §II.B (sports-context
# examples, applied generally) and §III fn. 26, plus categories the June 2026
# NPRM (proposed Rule 40.11 amendments) would prohibit or treat as likely
# contrary to the public interest. speech-triggered resolution is NOT a
# regulatory category — it lives in the manipulation_susceptibility=3 anchor.
DMO_FLAG_CATEGORIES = Literal[
    # DMO 26-08 §II.B
    "single_individual_action",
    "participant_injury",
    "unsportsmanlike_conduct",
    "physical_altercation",
    "officiating_decision",
    # DMO 26-08 §III fn. 26
    "tbd_settlement_source",
    # June 2026 NPRM
    "war_terrorism_assassination",
    "discrete_in_game_event",
    "youth_sports",
    "pure_chance",
]

# Categories the June 2026 NPRM would prohibit outright or deems likely
# contrary to the public interest. A detected flag here floors the grade at D
# regardless of mitigation — the NPRM prohibits, it does not mitigate.
NPRM_PROHIBITED_CATEGORIES = {
    "war_terrorism_assassination",
    "participant_injury",
    "officiating_decision",
    "discrete_in_game_event",
    "youth_sports",
    "pure_chance",
}

ENGAGEMENT_EVIDENCE = Literal[
    "official_data_source",
    "information_sharing_referenced",
    "integrity_standards_referenced",
    "no_evidence",
    "not_applicable",
]

RATING = Literal["A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]

# Ordered best → worst for cap comparisons
TIER_ORDER: list[str] = ["A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]


def _worse_grade(a: str, b: str) -> str:
    """Return the worse (lower) of two ratings."""
    ia = TIER_ORDER.index(a)
    ib = TIER_ORDER.index(b)
    return a if ia > ib else b


def _notch_down(grade: str, notches: int) -> str:
    """Move a grade down by N notches in TIER_ORDER."""
    idx = TIER_ORDER.index(grade)
    new_idx = min(idx + notches, len(TIER_ORDER) - 1)
    return TIER_ORDER[new_idx]


PARSER_VERSION = "0.5.0"

# The 8 specification-quality axes (base grade staircase)
SPEC_AXES = [
    "predicate_ambiguity",
    "entity_ambiguity",
    "temporal_precision",
    "source_specification",
    "source_quality",
    "edge_case_coverage",
    "headline_rules_alignment",
    "governing_body_engagement",
]

# The 2 structural axes (notch adjustment, not in base grade)
STRUCTURAL_AXES = ["manipulation_susceptibility", "outcome_concentration"]


ALL_AXES = SPEC_AXES + STRUCTURAL_AXES  # all 10 scoring axes


def spec_score(scores: dict[str, int | None]) -> int:
    """Specification-quality score: the plain sum of all ten axes (0-30).

    0 = no concern on any axis; higher = more/worse defects. A simple sum ---
    no weights, staircases, notches, or caps. (The two Core Principle 3 axes
    are also reported on their own as structural_score for downstream analysis,
    but they are part of this total.)
    """
    return sum(scores.get(k) or 0 for k in ALL_AXES)


def structural_score(scores: dict[str, int | None]) -> int:
    """Core Principle 3 sub-score: manipulation susceptibility + outcome
    concentration (0-6). Reported for the dispute analysis."""
    return sum(scores.get(k) or 0 for k in STRUCTURAL_AXES)


# Presentation bands on the 0-30 score. A DISCLOSED display convention only; the
# substantive measure is the continuous score. B or above = investment grade.
SPEC_GRADE_BANDS: list[tuple[int, RATING]] = [
    (0, "A"), (2, "BBB"), (4, "BB"), (7, "B"), (11, "CCC"), (15, "CC"), (20, "C"),
]


def spec_grade(score: int) -> RATING:
    """Map a score to a presentation letter grade (disclosed bands)."""
    for cutoff, g in SPEC_GRADE_BANDS:
        if score <= cutoff:
            return g
    return "D"


class Predicate(BaseModel):
    event_type: str
    statement: str
    notes: str | None = None


class Entity(BaseModel):
    name: str
    canonical_name: str | None = None
    entity_type: ENTITY_TYPES
    disambiguation_note: str | None = None


class TemporalBounds(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    timezone: str = "UTC"
    inclusive: bool | None = True
    notes: str | None = None


class ResolutionFunction(BaseModel):
    function_type: FUNCTION_TYPES
    outcomes: list[str] | None = None
    condition: str | None = None


class ResolutionMechanism(BaseModel):
    mechanism: MECHANISM_TYPES = "unknown"
    notes: str | None = None


class SourceHierarchy(BaseModel):
    named_sources: list[str] | None = None
    tier_requirement: TIER_REQUIREMENTS | None = None
    multiple_sources_required: bool = False
    notes: str | None = None


class EdgeCase(BaseModel):
    scenario: str
    addressed: bool
    handling: str | None = None


class DMO2608Flag(BaseModel):
    category: DMO_FLAG_CATEGORIES
    detected: bool
    mitigation_present: bool = False
    mitigation_notes: str | None = None


class GoverningBodyEngagement(BaseModel):
    applicable: bool
    body_named: str | None = None
    engagement_evidence: ENGAGEMENT_EVIDENCE = "not_applicable"
    notes: str | None = None


class ContractRating(BaseModel):
    predicate_ambiguity: int = Field(ge=0, le=3)
    entity_ambiguity: int = Field(ge=0, le=3)
    temporal_precision: int = Field(ge=0, le=3)
    manipulation_susceptibility: int = Field(ge=0, le=3)
    outcome_concentration: int = Field(ge=0, le=3)
    source_specification: int = Field(ge=0, le=3)
    source_quality: int = Field(ge=0, le=3)
    edge_case_coverage: int = Field(ge=0, le=3)
    headline_rules_alignment: int = Field(ge=0, le=3)
    governing_body_engagement: int = Field(default=3, ge=0, le=3)
    # Computed: spec_score (sum of 8 spec axes, 0-24) is the substantive measure;
    # rating is a disclosed presentation band on it. structural_score and the
    # flag indicators are reported separately (weighted empirically downstream).
    spec_score: int = 0
    structural_score: int = 0
    rating: RATING = "A"
    proposed_prohibited: bool = False
    mitigation_present: bool = False
    dmo_26_08_alignment: list[str] = Field(default_factory=list)
    explanations: dict[str, str] = Field(default_factory=dict)
    methodology_version: str = "v0.8"

    @model_validator(mode="after")
    def _set_rating(self) -> ContractRating:
        scores = {k: getattr(self, k) for k in SPEC_AXES + STRUCTURAL_AXES}
        self.spec_score = spec_score(scores)
        self.structural_score = structural_score(scores)
        self.rating = spec_grade(self.spec_score)
        return self


class ResolutionSpec(BaseModel):
    contract_id: str
    contract_text: str
    platform: str | None = None
    platform_id: str | None = None
    predicate: Predicate
    entities: list[Entity]
    temporal_bounds: TemporalBounds
    resolution_function: ResolutionFunction
    resolution_mechanism: ResolutionMechanism = Field(
        default_factory=ResolutionMechanism
    )
    source_hierarchy: SourceHierarchy
    edge_cases: list[EdgeCase]
    dmo_26_08_flags: list[DMO2608Flag] = Field(default_factory=list)
    governing_body_engagement: GoverningBodyEngagement | None = None
    rating: ContractRating
    governing_template_id: str | None = None
    template_match_method: str | None = None
    parser_confidence: dict[str, float]
    parser_version: str = PARSER_VERSION
    parsed_at: datetime

    @model_validator(mode="after")
    def _apply_dmo_flags(self) -> ResolutionSpec:
        # DMO/NPRM flags are descriptive indicators only. They do NOT modify the
        # grade (which is a transparent band on spec_score); their predictive
        # role is estimated empirically in the dispute analysis. We record two
        # booleans for downstream use: whether a recommended mitigation is
        # present, and whether a NPRM proposed-prohibited category was detected.
        self.rating.mitigation_present = any(
            f.detected and f.mitigation_present for f in self.dmo_26_08_flags
        )
        self.rating.proposed_prohibited = any(
            f.detected and f.category in NPRM_PROHIBITED_CATEGORIES
            for f in self.dmo_26_08_flags
        )
        return self
