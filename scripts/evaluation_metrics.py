"""Field-policy integration and record/run aggregation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from scripts.evaluation_field_policy import (
    EVALUATION_FIELD_ORDER,
    evaluate_field_policy,
)
from scripts.evaluation_field_policy import (
    levenshtein_distance as levenshtein_distance,
)
from scripts.evaluation_models import (
    EvaluationAuditRecord,
    EvaluationPolicy,
    ExpectedSample,
    FieldComparison,
    FieldEligibility,
    FieldScopeStatus,
    FieldStatus,
    GoldRecord,
    RunMetrics,
    SampleEvaluation,
)
from scripts.models import NativeParseResult


def compare_fields(
    gold: GoldRecord,
    native: NativeParseResult | None,
    *,
    style: str,
    entry_type: str,
    reference: str,
) -> tuple[list[FieldComparison], list[EvaluationAuditRecord]]:
    comparisons: list[FieldComparison] = []
    audits: list[EvaluationAuditRecord] = []
    for field in EVALUATION_FIELD_ORDER:
        gold_value = gold.fields.get(field)
        prediction = None
        if native is not None:
            prediction = (
                native.unknown_fields.get("edition")
                if field == "edition"
                else getattr(native, field, None)
            )
        decision = evaluate_field_policy(
            field=field,
            style=style,
            entry_type=entry_type,
            gold=gold_value,
            raw_person_gold=gold.raw_person_fields.get(field),
            prediction=prediction,
            reference=reference,
        )
        comparisons.append(
            FieldComparison(
                field=field,
                gold=gold_value,
                predicted=prediction,
                **{key: value for key, value in asdict(decision).items() if key != "warnings"},
            )
        )
        audit_reasons = [warning.code for warning in decision.warnings]
        if decision.eligibility in {
            FieldEligibility.VISIBILITY_UNRESOLVED,
            FieldEligibility.CANDIDATE_UNRESOLVED,
        }:
            audit_reasons.insert(0, decision.exclusion_reason or "candidate:empty")
        for reason in dict.fromkeys(audit_reasons):
            audits.append(
                EvaluationAuditRecord(
                    run_id="",
                    sample_id="",
                    style=style,
                    entry_type=entry_type,
                    field=field,
                    gold=gold_value,
                    reference=reference,
                    reason=reason,
                )
            )
    return comparisons, audits


def evaluate_sample(
    sample: ExpectedSample,
    gold_record: GoldRecord,
    item_record: dict[str, Any],
) -> tuple[SampleEvaluation, list[EvaluationAuditRecord]]:
    raw_output = item_record.get("parsed_output")
    native = (
        raw_output
        if isinstance(raw_output, NativeParseResult)
        else NativeParseResult.model_validate(raw_output)
        if isinstance(raw_output, dict)
        else None
    )
    reference = str(item_record.get("reference", sample.reference))
    comparisons, audits = compare_fields(
        gold_record,
        native,
        style=sample.style,
        entry_type=sample.entry_type,
        reference=reference,
    )
    run_id = str(item_record["run_id"])
    audits = [
        audit.model_copy(update={"run_id": run_id, "sample_id": sample.sample_id})
        for audit in audits
    ]
    eligible = [row for row in comparisons if row.eligibility == FieldEligibility.ELIGIBLE]
    evaluation = SampleEvaluation(
        run_id=run_id,
        parser=str(item_record.get("parser", "")),
        model=item_record.get("model"),
        sample_id=sample.sample_id,
        key=sample.key,
        style=sample.style,
        reference=reference,
        parse_status=str(item_record.get("parse_status", "missing_artifact")),
        attempt_count=int(item_record.get("attempt_count", 0)),
        entry_type=sample.entry_type,
        discipline=sample.discipline,
        difficulty_flags=sample.difficulty_flags,
        style_impl=sample.style_impl,
        generation_status=sample.generation_status,
        error_stage=item_record.get("error_stage"),
        error_kind=item_record.get("error_kind"),
        error_detail=item_record.get("error_detail"),
        fields=comparisons,
        missing_fields=[row.field for row in eligible if row.status == FieldStatus.MISSING],
        extra_fields=[
            row.field
            for row in comparisons
            if row.scope_status == FieldScopeStatus.ALWAYS
            and row.eligibility == FieldEligibility.GOLD_ABSENT
            and row.status == FieldStatus.EXTRA
        ],
        mismatched_fields=[row.field for row in eligible if row.status == FieldStatus.MISMATCH],
        record_exact_match=(
            all(row.status == FieldStatus.MATCH for row in eligible) if eligible else None
        ),
    )
    return evaluation, audits


def _mean(values: list[int] | list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_run_metrics(
    evaluations: list[SampleEvaluation],
    policy: EvaluationPolicy,
) -> RunMetrics:
    if not evaluations:
        return RunMetrics(
            run_id="",
            parser="",
            gold_source=policy.gold_source,
            field_policy_id=policy.field_policy_id,
            item_count=0,
            valid_output_count=0,
            valid_output_rate=0.0,
            record_exact_match_count=0,
            record_exact_match_support=0,
            record_exact_match_rate=None,
            completion_support=0,
            available_support=0,
            per_field_metrics={},
            eligibility_counts={},
            exclusion_reason_counts={},
            error_counts={},
        )

    run_ids = {evaluation.run_id for evaluation in evaluations}
    if len(run_ids) != 1:
        raise ValueError(
            f"compute_run_metrics received mixed run_id values: {sorted(run_ids)}. "
            "All evaluations must share the same run_id."
        )
    parsers = {evaluation.parser for evaluation in evaluations}
    if len(parsers) != 1:
        raise ValueError(
            f"compute_run_metrics received mixed parser values: {sorted(parsers)}. "
            "All evaluations must share the same parser."
        )
    models = {evaluation.model for evaluation in evaluations}
    if len(models) != 1:
        raise ValueError(
            f"compute_run_metrics received mixed model values: {sorted(models, key=str)}. "
            "All evaluations must share the same model."
        )

    item_count = len(evaluations)
    valid_output_count = sum(evaluation.parse_status == "ok" for evaluation in evaluations)
    exact_values = [
        evaluation.record_exact_match
        for evaluation in evaluations
        if evaluation.record_exact_match is not None
    ]
    record_exact_match_count = sum(exact_values)
    record_exact_match_support = len(exact_values)
    all_comparisons = [comparison for evaluation in evaluations for comparison in evaluation.fields]

    eligibility_counts: dict[str, int] = {}
    exclusion_reason_counts: dict[str, int] = {}
    for comparison in all_comparisons:
        eligibility = comparison.eligibility.value
        eligibility_counts[eligibility] = eligibility_counts.get(eligibility, 0) + 1
        if comparison.exclusion_reason is not None:
            exclusion_reason_counts[comparison.exclusion_reason] = (
                exclusion_reason_counts.get(comparison.exclusion_reason, 0) + 1
            )

    error_counts: dict[str, int] = {}
    for evaluation in evaluations:
        error_counts[evaluation.parse_status] = error_counts.get(evaluation.parse_status, 0) + 1

    observed_fields = {comparison.field for comparison in all_comparisons}
    ordered_fields = [field for field in EVALUATION_FIELD_ORDER if field in observed_fields]
    ordered_fields.extend(sorted(observed_fields - set(ordered_fields)))
    per_field_metrics: dict[str, dict[str, Any]] = {}
    for field in ordered_fields:
        field_comparisons = [
            comparison for comparison in all_comparisons if comparison.field == field
        ]
        completion = [
            comparison for comparison in field_comparisons if comparison.include_completion
        ]
        available = [comparison for comparison in field_comparisons if comparison.include_available]
        completion_raw = [
            comparison.distance_raw
            for comparison in completion
            if comparison.distance_raw is not None
        ]
        completion_normalized = [
            comparison.distance_normalized
            for comparison in completion
            if comparison.distance_normalized is not None
        ]
        available_raw = [
            comparison.distance_raw
            for comparison in available
            if comparison.distance_raw is not None
        ]
        available_normalized = [
            comparison.distance_normalized
            for comparison in available
            if comparison.distance_normalized is not None
        ]
        exact_count = sum(comparison.exact_match for comparison in completion)
        per_field_metrics[field] = {
            "completion_support": len(completion),
            "available_support": len(available),
            "exact_count": exact_count,
            "exact_rate": exact_count / len(completion) if completion else None,
            "completion_distance_raw_mean": _mean(completion_raw),
            "completion_distance_normalized_mean": _mean(completion_normalized),
            "available_distance_raw_mean": _mean(available_raw),
            "available_distance_normalized_mean": _mean(available_normalized),
        }

    completion_support = sum(comparison.include_completion for comparison in all_comparisons)
    available_support = sum(comparison.include_available for comparison in all_comparisons)
    return RunMetrics(
        run_id=evaluations[0].run_id,
        parser=evaluations[0].parser,
        model=evaluations[0].model,
        gold_source=policy.gold_source,
        field_policy_id=policy.field_policy_id,
        item_count=item_count,
        valid_output_count=valid_output_count,
        valid_output_rate=valid_output_count / item_count,
        record_exact_match_count=record_exact_match_count,
        record_exact_match_support=record_exact_match_support,
        record_exact_match_rate=(
            record_exact_match_count / record_exact_match_support
            if record_exact_match_support
            else None
        ),
        completion_support=completion_support,
        available_support=available_support,
        per_field_metrics=per_field_metrics,
        eligibility_counts=eligibility_counts,
        exclusion_reason_counts=exclusion_reason_counts,
        error_counts=error_counts,
    )
