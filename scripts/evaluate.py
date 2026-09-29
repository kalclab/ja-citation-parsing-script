"""Strict schema-v2 offline evaluation CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.config import RunConfig
from scripts.evaluation_adapter import (
    LoadedArtifactRun,
    evaluation_code_hash,
    git_provenance,
    load_artifact_run,
    runtime_versions,
    validate_artifact_run,
)
from scripts.evaluation_field_policy import (
    FIELD_POLICY_ID,
    field_policy_hash,
    field_policy_payload,
)
from scripts.evaluation_gold import EvaluationSources, load_evaluation_sources
from scripts.evaluation_metrics import compute_run_metrics, evaluate_sample
from scripts.evaluation_models import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationAuditRecord,
    EvaluationPolicy,
    RunMetrics,
    SampleEvaluation,
)

_SCOPE_SOURCE = "docs/cite-parse-icadl2026-main/reports/style_field_matrix.md#always-output"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvaluationBuildResult:
    evaluations: tuple[SampleEvaluation, ...]
    metrics: tuple[RunMetrics, ...]
    audit_records: tuple[EvaluationAuditRecord, ...]
    manifest: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.evaluate",
        description="Evaluate parser artifacts with the observable field policy.",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        type=Path,
        action="append",
        required=True,
        help="Path to a run artifact directory (repeatable).",
    )
    parser.add_argument(
        "--gold-source",
        default="main",
        choices=["main", "full", "derived"],
        help="Gold source to evaluate (default: main).",
    )
    parser.add_argument(
        "--field-policy",
        dest="field_policy",
        required=True,
        choices=[FIELD_POLICY_ID],
        help="Required observable field policy.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/v0.2-20260717"),
        help="Dataset directory containing gold and reference sources.",
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=Path("data/v0.2-20260717/sample_100.txt"),
        help="Sample key file.",
    )
    parser.add_argument(
        "--styles",
        nargs="*",
        default=None,
        help="Styles to evaluate (default: all styles from RunConfig).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Final schema-v2 evaluation output directory.",
    )
    args = parser.parse_args(argv)
    if args.styles is None:
        args.styles = RunConfig(run_id="_default", parser="grobid").styles
    return args


def _manifest_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(_REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _source_provenance(path: Path, sha256: str) -> dict[str, str]:
    return {"path": _manifest_path(path), "sha256": sha256}


def _artifact_provenance(artifact: LoadedArtifactRun) -> dict[str, str]:
    provenance = artifact.provenance
    return {
        "run_id": provenance.run_id,
        "path": _manifest_path(provenance.path),
        "manifest_sha256": provenance.manifest_sha256,
        "items_sha256": provenance.items_sha256,
    }


def _scope_hash() -> str:
    canonical = json.dumps(
        field_policy_payload()["scope"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _preflight_artifacts(
    run_dirs: list[Path],
    sources: EvaluationSources,
    styles: list[str],
) -> tuple[LoadedArtifactRun, ...]:
    resolved_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for run_dir in run_dirs:
        resolved = Path(run_dir).resolve()
        if resolved in seen_dirs:
            raise ValueError(f"Duplicate run directory: {resolved}")
        seen_dirs.add(resolved)
        resolved_dirs.append(resolved)

    artifacts: list[LoadedArtifactRun] = []
    seen_run_ids: set[str] = set()
    selected_styles = set(styles)
    for run_dir in resolved_dirs:
        artifact = load_artifact_run(run_dir)
        run_id = artifact.provenance.run_id
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate artifact run_id: {run_id}")
        seen_run_ids.add(run_id)
        manifest_styles = artifact.manifest.get("styles")
        if (
            not isinstance(manifest_styles, list)
            or not all(isinstance(style, str) for style in manifest_styles)
            or set(manifest_styles) != selected_styles
        ):
            raise ValueError(
                f"artifact styles mismatch for {run_id}: "
                f"expected={sorted(selected_styles)}, actual={manifest_styles!r}"
            )
        validate_artifact_run(artifact, sources)
        artifacts.append(artifact)
    return tuple(artifacts)


def _aggregate_counts(metrics: list[RunMetrics], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for metric in metrics:
        counts.update(getattr(metric, field))
    return dict(sorted(counts.items()))


def build_evaluation_pipeline(
    *,
    run_dirs: list[Path],
    gold_source: str,
    field_policy_id: str,
    dataset_path: Path,
    sample_path: Path,
    styles: list[str],
) -> EvaluationBuildResult:
    created_at = datetime.now(UTC).isoformat()
    policy = EvaluationPolicy(
        gold_source=gold_source,
        field_policy_id=field_policy_id,
    )
    selected_styles = list(dict.fromkeys(styles))
    sources = load_evaluation_sources(
        dataset_path=dataset_path,
        sample_path=sample_path,
        gold_source=policy.gold_source,
        styles=selected_styles,
    )
    artifacts = _preflight_artifacts(run_dirs, sources, selected_styles)

    evaluations: list[SampleEvaluation] = []
    metrics: list[RunMetrics] = []
    audits: list[EvaluationAuditRecord] = []
    for artifact in artifacts:
        manifest = artifact.manifest
        items_by_sample = {item["sample_id"]: item for item in artifact.items}
        run_evaluations: list[SampleEvaluation] = []
        for sample in sources.samples:
            item = items_by_sample.get(sample.sample_id)
            if item is None:
                item_record: dict[str, Any] = {
                    "run_id": artifact.provenance.run_id,
                    "parser": manifest.get("parser", ""),
                    "model": manifest.get("model"),
                    "reference": sample.reference,
                    "parse_status": "missing_artifact",
                    "attempt_count": 0,
                    "error_stage": "artifact",
                    "error_kind": "missing_artifact",
                    "error_detail": None,
                    "parsed_output": None,
                }
            else:
                item_record = dict(item)
                item_record.setdefault("parser", manifest.get("parser", ""))
                item_record.setdefault("model", manifest.get("model"))
            evaluation, sample_audits = evaluate_sample(
                sample,
                sources.gold.records[sample.key],
                item_record,
            )
            run_evaluations.append(evaluation)
            audits.extend(sample_audits)
        evaluations.extend(run_evaluations)
        metrics.append(compute_run_metrics(run_evaluations, policy))

    revision, dirty = git_provenance()
    completed_at = datetime.now(UTC).isoformat()
    eligibility_counts = _aggregate_counts(metrics, "eligibility_counts")
    exclusion_reason_counts = _aggregate_counts(metrics, "exclusion_reason_counts")
    warning_count = sum(audit.reason.startswith("visibility:") for audit in audits)
    manifest = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "field_policy_id": policy.field_policy_id,
        "field_policy_hash": field_policy_hash(),
        "evaluation_code_hash": evaluation_code_hash(),
        "evaluation_code_revision": revision,
        "evaluation_code_dirty": dirty,
        "scope_source": _SCOPE_SOURCE,
        "scope_hash": _scope_hash(),
        "input_artifacts": [_artifact_provenance(artifact) for artifact in artifacts],
        "gold_source": policy.gold_source,
        "gold": _source_provenance(sources.gold.path, sources.gold.sha256),
        "metadata": _source_provenance(
            sources.metadata.path,
            sources.metadata.sha256,
        ),
        "references": _source_provenance(
            sources.references.path,
            sources.references.sha256,
        ),
        "sample": _source_provenance(sources.sample.path, sources.sample.sha256),
        "dataset_hash": sources.dataset_hash,
        "styles": selected_styles,
        "eligibility_counts": eligibility_counts,
        "exclusion_reason_counts": exclusion_reason_counts,
        "completion_support": sum(metric.completion_support for metric in metrics),
        "available_support": sum(metric.available_support for metric in metrics),
        "visibility_audit_warning_count": warning_count,
        "runtime_versions": runtime_versions(),
        "created_at": created_at,
        "completed_at": completed_at,
    }
    return EvaluationBuildResult(
        evaluations=tuple(evaluations),
        metrics=tuple(metrics),
        audit_records=tuple(audits),
        manifest=manifest,
    )


def _validated_json_lines(
    rows: tuple[SampleEvaluation, ...] | tuple[RunMetrics, ...] | tuple[EvaluationAuditRecord, ...],
) -> str:
    serialized: list[str] = []
    for row in rows:
        validated = type(row).model_validate(row.model_dump())
        serialized.append(validated.model_dump_json())
    return "".join(f"{line}\n" for line in serialized)


def write_evaluation_output(
    output_dir: Path,
    result: EvaluationBuildResult,
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        (temporary / "manifest.json").write_text(
            json.dumps(result.manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "items.jsonl").write_text(
            _validated_json_lines(result.evaluations),
            encoding="utf-8",
        )
        (temporary / "summary.jsonl").write_text(
            _validated_json_lines(result.metrics),
            encoding="utf-8",
        )
        (temporary / "audit.jsonl").write_text(
            _validated_json_lines(result.audit_records),
            encoding="utf-8",
        )
        result_payload: dict[str, Any] = {
            "runs": {metric.run_id: metric.model_dump(mode="json") for metric in result.metrics},
            "_meta": {
                "evaluation_schema_version": result.manifest["evaluation_schema_version"],
                "field_policy_id": result.manifest["field_policy_id"],
                "created_at": result.manifest["created_at"],
                "completed_at": result.manifest["completed_at"],
            },
        }
        (temporary / "result.json").write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        temporary.rename(output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    args = parse_args()
    result = build_evaluation_pipeline(
        run_dirs=args.run_dirs,
        gold_source=args.gold_source,
        field_policy_id=args.field_policy,
        dataset_path=args.dataset_path,
        sample_path=args.sample_path,
        styles=args.styles,
    )
    write_evaluation_output(args.output_dir, result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        raise
