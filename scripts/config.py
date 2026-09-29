"""Run configuration model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from scripts.models import ParserKind


class RunConfig(BaseModel):
    """Configuration for a single experimental run."""

    model_config = {"extra": "forbid"}

    run_id: str
    parser: ParserKind

    model: str | None = None
    batch_size: int | None = None

    order_mode: Literal["fixed", "shuffle"] = "fixed"
    shuffle_seed: int | None = None

    temperature: float | None = None
    seed: int | None = None

    grobid_base_url: str = "http://localhost:8070"
    grobid_consolidate: int = 0

    dataset_path: str = "data/v0.2-20260717"
    sample_path: str = "data/v0.2-20260717/sample_100.txt"
    styles: list[str] = [
        "jecon-kalc",
        "ipsjsort",
        "jsai",
        "jplain",
        "sist02",
        "jpa2022",
        "chicago-author-date",
    ]

    artifacts_dir: str = "artifacts"

    @model_validator(mode="after")
    def _validate_parser_requires_fields(self) -> RunConfig:
        if self.parser == ParserKind.LLM:
            if not self.model:
                raise ValueError("LLM parser requires 'model'")
            if not self.batch_size or self.batch_size < 1:
                raise ValueError("LLM parser requires 'batch_size' >= 1")
        return self

    @property
    def output_dir(self) -> Path:
        return Path(self.artifacts_dir) / self.run_id


def load_config(path: Path | str) -> RunConfig:
    """Load a RunConfig from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RunConfig.model_validate(data)


def config_to_manifest(config: RunConfig, **extra: Any) -> dict[str, Any]:
    """Convert a RunConfig to a flat manifest dict for artifacts."""
    manifest: dict[str, Any] = {
        "run_id": config.run_id,
        "parser": config.parser.value,
        "model": config.model,
        "batch_size": config.batch_size,
        "order_mode": config.order_mode,
        "shuffle_seed": config.shuffle_seed,
        "temperature": config.temperature,
        "seed": config.seed,
        "grobid_base_url": config.grobid_base_url,
        "grobid_consolidate": config.grobid_consolidate,
        "dataset_path": config.dataset_path,
        "sample_path": config.sample_path,
        "styles": config.styles,
    }
    manifest.update(extra)
    return manifest
