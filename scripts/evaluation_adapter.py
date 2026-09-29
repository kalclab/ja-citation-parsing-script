"""Strict artifact preflight and evaluation provenance helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import unicodedata
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from scripts.evaluation_gold import EvaluationSources
from scripts.evaluation_models import (
    EVALUATION_SCHEMA_VERSION,
    ArtifactProvenance,
)
from scripts.models import NativeParseResult


@dataclass(frozen=True)
class LoadedArtifactRun:
    root: Path
    manifest: dict[str, Any]
    items: tuple[dict[str, Any], ...]
    provenance: ArtifactProvenance


def load_artifact_run(run_dir: Path) -> LoadedArtifactRun:
    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    items_path = root / "items.jsonl"
    for path in (manifest_path, items_path):
        if not path.is_file():
            raise FileNotFoundError(f"required artifact file not found: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid artifact manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("run_id"), str):
        raise ValueError(f"invalid artifact manifest: {manifest_path}")
    run_id = manifest["run_id"]
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        items_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed artifact JSON at {items_path}:{line_number}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"artifact item is not an object at {items_path}:{line_number}")
        if item.get("run_id") != run_id:
            raise ValueError(f"artifact item run_id mismatch at {items_path}:{line_number}")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"artifact sample_id missing at {items_path}:{line_number}")
        if sample_id in seen:
            raise ValueError(f"duplicate artifact sample_id: {sample_id}")
        seen.add(sample_id)
        parsed_output = item.get("parsed_output")
        if parsed_output is not None:
            if not isinstance(parsed_output, dict):
                raise ValueError(f"invalid parsed_output for sample {sample_id}")
            unknown_keys = set(parsed_output) - set(NativeParseResult.model_fields)
            if unknown_keys:
                raise ValueError(
                    f"unknown parsed_output fields for sample {sample_id}: {sorted(unknown_keys)}"
                )
            native = NativeParseResult.model_validate(parsed_output)
            edition = native.unknown_fields.get("edition")
            if edition is not None and not isinstance(edition, str):
                raise ValueError(f"invalid unknown_fields.edition for sample {sample_id}")
        items.append(item)
    return LoadedArtifactRun(
        root=root,
        manifest=manifest,
        items=tuple(items),
        provenance=ArtifactProvenance(
            run_id=run_id,
            path=root,
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            items_sha256=hashlib.sha256(items_path.read_bytes()).hexdigest(),
        ),
    )


def validate_artifact_run(artifact: LoadedArtifactRun, sources: EvaluationSources) -> None:
    expected = {sample.sample_id: sample for sample in sources.samples}
    for item in artifact.items:
        sample_id = item["sample_id"]
        sample = expected.get(sample_id)
        if sample is None:
            raise ValueError(f"unexpected artifact sample_id: {sample_id}")
        expected_identity = {
            "key": sample.key,
            "style": sample.style,
            "entry_type": sample.entry_type,
            "reference": sample.reference,
        }
        for field, expected_value in expected_identity.items():
            if item.get(field) != expected_value:
                raise ValueError(
                    f"artifact {field} mismatch for {sample_id}: "
                    f"expected={expected_value!r}, actual={item.get(field)!r}"
                )

    expected_hashes = {
        "dataset_hash": sources.dataset_hash,
        "sample_hash": sources.sample.sha256,
    }
    for field, expected_hash in expected_hashes.items():
        actual_hash = artifact.manifest.get(field)
        if actual_hash != "unavailable" and actual_hash != expected_hash:
            raise ValueError(
                f"artifact manifest {field} mismatch: "
                f"expected={expected_hash!r}, actual={actual_hash!r}"
            )


_EVALUATION_CODE_PATHS = (
    "scripts/models.py",
    "scripts/evaluation_adapter.py",
    "scripts/evaluation_field_policy.py",
    "scripts/evaluation_gold.py",
    "scripts/evaluation_metrics.py",
    "scripts/evaluation_models.py",
    "scripts/evaluate.py",
)


def evaluation_code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative_path in _EVALUATION_CODE_PATHS:
        path_bytes = relative_path.encode("utf-8")
        source_bytes = (root / relative_path).read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(source_bytes).to_bytes(8, "big"))
        digest.update(source_bytes)
    return digest.hexdigest()


def git_provenance() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision or None, bool(status.strip())


def runtime_versions() -> dict[str, str | int]:
    return {
        "python": platform.python_version(),
        "unicodedata": unicodedata.unidata_version,
        "bibtexparser": version("bibtexparser"),
        "pydantic": version("pydantic"),
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
    }


def load_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    """Read a manifest for non-evaluation inspection callers."""
    path = run_dir / "manifest.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_artifact_items(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read item rows permissively for non-evaluation inspection callers."""
    path = run_dir / "items.jsonl"
    if not path.is_file():
        return [], []

    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                warnings.append(f"JSONDecodeError parsing line in {path}: {exc}")
                continue
            if isinstance(item, dict):
                items.append(item)
            else:
                warnings.append(f"Non-object JSON item in {path}")
    return items, warnings
