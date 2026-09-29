"""Artifact writer: manifest, calls, items, raw outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.models import CallLog, ItemLog


class ArtifactWriter:
    """Writes experiment artifacts to disk.

    Directory structure::

        <output_dir>/
          manifest.json
          calls.jsonl
          items.jsonl
          raw/
            <call_id>/
              request
              response
    """

    def __init__(self, run_id: str, output_dir: Path) -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        raw_dir = self.output_dir / "raw"
        if self.output_dir.exists() and raw_dir.exists():
            raise ValueError(f"Output directory already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir()

    def write_manifest(self, data: dict[str, Any]) -> None:
        """Write the run manifest as JSON."""
        path = self.output_dir / "manifest.json"
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_call(self, call: CallLog) -> None:
        """Append a call log to calls.jsonl."""
        path = self.output_dir / "calls.jsonl"
        line = call.model_dump_json(exclude_none=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_item(self, item: ItemLog) -> None:
        """Append an item log to items.jsonl."""
        path = self.output_dir / "items.jsonl"
        dumped = item.model_dump_json(exclude_none=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(dumped + "\n")

    def write_raw(self, call_id: str, kind: str, content: str) -> Path:
        """Write raw request/response content.

        Args:
            call_id: The call this content belongs to.
            kind: 'request' or 'response'.
            content: The raw string content.

        Returns:
            The path to the written file.
        """
        call_dir = self.output_dir / "raw" / call_id
        call_dir.mkdir(parents=True, exist_ok=True)
        path = call_dir / kind
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def file_hash(path: Path) -> str:
        """Compute SHA-256 of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def content_hash(content: str) -> str:
        """Compute SHA-256 of a string."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
