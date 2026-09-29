"""Models used by bibliographic evaluation loaders and schema-v2 artifacts."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EVALUATION_SCHEMA_VERSION = 2


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationPolicy(StrictModel):
    gold_source: Literal["main", "full", "derived"] = "main"
    field_policy_id: Literal["observable-fields-v1"] = "observable-fields-v1"


class GoldRecord(BaseModel):
    key: str
    entry_type: str | None = None
    fields: dict[str, Any]
    raw_person_fields: dict[str, str] = Field(default_factory=dict)


class GoldBundle(BaseModel):
    source: Literal["main", "full", "derived"]
    path: Path
    sha256: str
    records: dict[str, GoldRecord]


class ExpectedSample(BaseModel):
    sample_id: str
    key: str
    style: str
    entry_type: str = ""
    discipline: str = ""
    subfield: str = ""
    difficulty_flags: str = ""
    style_impl: str = ""
    generation_status: str = ""
    reference: str = ""


class FieldStatus(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    MISSING = "missing"
    EXTRA = "extra"
    NOT_APPLICABLE = "not_applicable"


class FieldScopeStatus(StrEnum):
    ALWAYS = "always"
    EXCLUDED = "excluded"


class FieldEligibility(StrEnum):
    ELIGIBLE = "eligible"
    OUT_OF_SCOPE = "out_of_scope"
    GOLD_ABSENT = "gold_absent"
    VISIBILITY_UNRESOLVED = "visibility_unresolved"
    CANDIDATE_UNRESOLVED = "candidate_unresolved"


class FieldComparison(StrictModel):
    field: str
    status: FieldStatus
    gold: Any = None
    predicted: Any = None
    exact_match: bool = False
    scope_status: FieldScopeStatus
    gold_present: bool
    observable: bool | None
    eligibility: FieldEligibility
    exclusion_reason: str | None = None
    candidate_path_count: int = 0
    selected_candidate_id: str | None = None
    selected_gold: str | None = None
    normalized_prediction: str | None = None
    distance_raw: int | None = None
    distance_normalized: float | None = None
    include_completion: bool = False
    include_available: bool = False


class SampleEvaluation(StrictModel):
    run_id: str
    parser: str
    model: str | None = None
    sample_id: str
    key: str
    style: str
    reference: str = ""
    parse_status: str
    attempt_count: int
    entry_type: str = ""
    discipline: str = ""
    difficulty_flags: str = ""
    style_impl: str = ""
    generation_status: str = ""
    error_stage: str | None = None
    error_kind: str | None = None
    error_detail: str | None = None
    fields: list[FieldComparison]
    missing_fields: list[str]
    extra_fields: list[str]
    mismatched_fields: list[str]
    record_exact_match: bool | None


class SourceProvenance(StrictModel):
    path: Path
    sha256: str


class ArtifactProvenance(StrictModel):
    run_id: str
    path: Path
    manifest_sha256: str
    items_sha256: str


class EvaluationAuditRecord(StrictModel):
    run_id: str
    sample_id: str
    style: str
    entry_type: str
    field: str
    gold: Any
    reference: str
    reason: str


class RunMetrics(StrictModel):
    run_id: str
    parser: str
    model: str | None = None
    gold_source: str
    field_policy_id: str
    item_count: int
    valid_output_count: int
    valid_output_rate: float
    record_exact_match_count: int
    record_exact_match_support: int
    record_exact_match_rate: float | None
    completion_support: int
    available_support: int
    per_field_metrics: dict[str, dict[str, Any]]
    eligibility_counts: dict[str, int]
    exclusion_reason_counts: dict[str, int]
    error_counts: dict[str, int]
