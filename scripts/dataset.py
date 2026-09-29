"""Dataset loading: ref_strings.jsonl + metadata.csv + sample key list."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Literal

from scripts.models import InputItem

SAMPLE_ID_SEP = "::"


def load_inputs(
    data_dir: Path | str,
    sample_file: Path | str,
    styles: list[str],
    *,
    order_mode: Literal["fixed", "shuffle"] = "fixed",
    shuffle_seed: int | None = None,
) -> list[InputItem]:
    data_dir = Path(data_dir)
    sample_file = Path(sample_file)

    sample_keys = _load_sample_keys(sample_file)
    ref_strings = _load_ref_strings(data_dir / "ref_strings.jsonl", sample_keys, styles)
    metadata = _load_metadata(data_dir / "metadata.csv")

    items: list[InputItem] = []
    opaque_counter = 0

    if order_mode == "shuffle" and shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
    else:
        rng = None

    for style in styles:
        style_items: list[InputItem] = []
        for key in sorted(sample_keys):
            ref_key = (key, style)
            if ref_key not in ref_strings:
                raise ValueError(
                    f"Reference for key={key}, style={style} not found in ref_strings.jsonl"
                )
            ref_str = ref_strings[ref_key]
            if not ref_str.strip():
                raise ValueError(f"Empty reference string for key={key}, style={style}")
            meta = metadata.get(key, {})
            opaque_counter += 1
            opaque_id = f"ref_{opaque_counter:04d}"
            style_items.append(
                InputItem(
                    sample_id=f"{key}{SAMPLE_ID_SEP}{style}",
                    key=key,
                    style=style,
                    opaque_input_id=opaque_id,
                    reference=ref_str,
                    entry_type=meta.get("entry_type", ""),
                    discipline=meta.get("discipline", ""),
                    difficulty_flags=meta.get("difficulty_flags", ""),
                )
            )
        if rng is not None:
            rng.shuffle(style_items)
        items.extend(style_items)

    _validate_opaque_ids(items)
    return items


def _load_sample_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                keys.add(stripped)
    return keys


def _load_ref_strings(
    path: Path,
    keys: set[str],
    styles: list[str],
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = obj["key"]
            style = obj["style"]
            if key in keys and style in styles:
                result[(key, style)] = obj["ref_string"]
    return result


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["key"]] = {
                "entry_type": row.get("entry_type", ""),
                "discipline": row.get("discipline", ""),
                "difficulty_flags": row.get("difficulty_flags", ""),
            }
    return result


def _validate_opaque_ids(items: list[InputItem]) -> None:
    seen: set[str] = set()
    for item in items:
        if item.opaque_input_id in seen:
            raise ValueError(f"Duplicate opaque_input_id: {item.opaque_input_id}")
        seen.add(item.opaque_input_id)


def batch_items(items: list[InputItem], batch_size: int) -> list[list[InputItem]]:
    """Split items into batches of batch_size.

    Items are already grouped by style. Batches preserve style boundaries.
    """
    batches: list[list[InputItem]] = []
    current_batch: list[InputItem] = []
    current_style: str | None = None

    for item in items:
        if current_style is not None and item.style != current_style:
            if current_batch:
                batches.append(current_batch)
            current_batch = []
            current_style = item.style

        if current_style is None:
            current_style = item.style

        current_batch.append(item)
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []

    if current_batch:
        batches.append(current_batch)

    return batches
