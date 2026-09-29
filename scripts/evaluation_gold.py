"""Load and preflight BibTeX gold records and expected evaluation samples."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bibtexparser

from scripts.evaluation_field_policy import observable_field_scope
from scripts.evaluation_models import (
    ExpectedSample,
    GoldBundle,
    GoldRecord,
    SourceProvenance,
)

_PERSON_FIELDS = frozenset({"author", "editor", "translator"})
_SPECIAL_ENTRY_TYPES = frozenset({"comment", "preamble", "string"})


@dataclass(frozen=True)
class EvaluationSources:
    gold: GoldBundle
    samples: tuple[ExpectedSample, ...]
    metadata: SourceProvenance
    references: SourceProvenance
    sample: SourceProvenance
    dataset_hash: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_escaped(source: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and source[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _delimited_end(source: str, opening_index: int) -> int:
    opening = source[opening_index]
    closing = "}" if opening == "{" else ")"
    brace_depth = 1 if opening == "{" else 0
    parenthesis_depth = 1 if opening == "(" else 0
    in_quote = False
    index = opening_index + 1
    while index < len(source):
        character = source[index]
        if character == '"' and not _is_escaped(source, index):
            in_quote = not in_quote
        elif not in_quote and not _is_escaped(source, index):
            if character == "{":
                brace_depth += 1
            elif character == "}" and brace_depth:
                brace_depth -= 1
                if opening == "{" and brace_depth == 0:
                    return index
            elif character == "(" and brace_depth == 0:
                parenthesis_depth += 1
            elif character == ")" and brace_depth == 0:
                parenthesis_depth -= 1
                if opening == "(" and parenthesis_depth == 0:
                    return index
        index += 1
    raise ValueError(f"unterminated BibTeX {opening}{closing} block at index {opening_index}")


def _top_level_comma(source: str, start: int = 0) -> int:
    brace_depth = 0
    in_quote = False
    for index in range(start, len(source)):
        character = source[index]
        if character == '"' and brace_depth == 0 and not _is_escaped(source, index):
            in_quote = not in_quote
        elif not in_quote and not _is_escaped(source, index):
            if character == "{":
                brace_depth += 1
            elif character == "}" and brace_depth:
                brace_depth -= 1
            elif character == "," and brace_depth == 0:
                return index
    return len(source)


def _braced_value_end(source: str, opening_index: int) -> int:
    depth = 1
    for index in range(opening_index + 1, len(source)):
        if _is_escaped(source, index):
            continue
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unterminated BibTeX field value at index {opening_index}")


def _quoted_value_end(source: str, opening_index: int) -> int:
    brace_depth = 0
    for index in range(opening_index + 1, len(source)):
        if _is_escaped(source, index):
            continue
        if source[index] == "{":
            brace_depth += 1
        elif source[index] == "}" and brace_depth:
            brace_depth -= 1
        elif source[index] == '"' and brace_depth == 0:
            return index
    raise ValueError(f"unterminated quoted BibTeX field value at index {opening_index}")


def _scan_entry_person_fields(body: str, key: str) -> dict[str, str]:
    first_comma = _top_level_comma(body)
    if first_comma == len(body):
        return {}
    result: dict[str, str] = {}
    index = first_comma + 1
    while index < len(body):
        while index < len(body) and (body[index].isspace() or body[index] == ","):
            index += 1
        if index >= len(body):
            break
        name_match = re.match(r"[A-Za-z][A-Za-z0-9_-]*", body[index:])
        if name_match is None:
            raise ValueError(f"invalid BibTeX field assignment for entry {key}")
        field = name_match.group().lower()
        index += len(name_match.group())
        while index < len(body) and body[index].isspace():
            index += 1
        if index >= len(body) or body[index] != "=":
            raise ValueError(f"invalid BibTeX field assignment for entry {key}.{field}")
        index += 1
        while index < len(body) and body[index].isspace():
            index += 1
        value_start = index
        if index < len(body) and body[index] == "{":
            value_end = _braced_value_end(body, index)
            raw_value = body[index + 1 : value_end]
            index = value_end + 1
        elif index < len(body) and body[index] == '"':
            value_end = _quoted_value_end(body, index)
            raw_value = body[index + 1 : value_end]
            index = value_end + 1
        else:
            value_end = _top_level_comma(body, index)
            hash_index = body.find("#", index, value_end)
            if hash_index >= 0:
                value_end = hash_index
            raw_value = body[index:value_end].strip()
            index = value_end
            if field in _PERSON_FIELDS:
                raise ValueError(f"unsupported person value for entry {key}.{field}: bare macro")

        after_value = index
        while after_value < len(body) and body[after_value].isspace():
            after_value += 1
        if after_value < len(body) and body[after_value] == "#":
            if field in _PERSON_FIELDS:
                raise ValueError(f"unsupported person value for entry {key}.{field}: concatenation")
            index = _top_level_comma(body, after_value)
        elif index == value_start:
            raise ValueError(f"empty BibTeX field value for entry {key}.{field}")
        if field in _PERSON_FIELDS:
            if field in result:
                raise ValueError(f"duplicate person field for entry {key}.{field}")
            result[field] = raw_value
    return result


def _scan_raw_person_fields(source: str) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    index = 0
    while True:
        while index < len(source) and source[index] != "@":
            if source[index] == "%":
                newline_index = source.find("\n", index + 1)
                if newline_index < 0:
                    return records
                index = newline_index + 1
            else:
                index += 1
        if index >= len(source):
            break
        opening_at = index
        type_match = re.match(r"@[A-Za-z]+", source[opening_at:])
        if type_match is None:
            index = opening_at + 1
            continue
        entry_type = type_match.group()[1:].lower()
        opening_index = opening_at + len(type_match.group())
        while opening_index < len(source) and source[opening_index].isspace():
            opening_index += 1
        if opening_index >= len(source) or source[opening_index] not in "{(":
            raise ValueError(f"invalid BibTeX entry at index {opening_at}")
        closing_index = _delimited_end(source, opening_index)
        if entry_type not in _SPECIAL_ENTRY_TYPES:
            body = source[opening_index + 1 : closing_index]
            first_comma = _top_level_comma(body)
            key = body[:first_comma].strip()
            if not key:
                raise ValueError(f"BibTeX entry at index {opening_at} has no key")
            if key in records:
                raise ValueError(f"Duplicate BibTeX key: {key}")
            records[key] = _scan_entry_person_fields(body, key)
        index = closing_index + 1
    return records


def load_gold(path: Path) -> GoldBundle:
    """Parse gold and retain exact lexical person-list values."""
    path = Path(path)
    source_kind = _gold_source(path)
    raw_bytes = path.read_bytes()
    source_text = raw_bytes.decode("utf-8")
    raw_person_fields = _scan_raw_person_fields(source_text)
    database = bibtexparser.loads(source_text)
    records: dict[str, GoldRecord] = {}

    for entry in database.entries:
        key = entry.get("ID")
        if not isinstance(key, str) or not key:
            raise ValueError(f"BibTeX entry in {path} has no ID")
        if key in records:
            raise ValueError(f"Duplicate BibTeX key: {key}")

        fields: dict[str, Any] = {}
        for field_name, value in entry.items():
            if field_name in {"ID", "ENTRYTYPE"}:
                continue
            normalized_name = field_name.lower()
            if normalized_name in _PERSON_FIELDS:
                fields[normalized_name] = (
                    list(value) if isinstance(value, list) else _split_person_list(str(value))
                )
            else:
                fields[normalized_name] = value

        records[key] = GoldRecord(
            key=key,
            entry_type=entry.get("ENTRYTYPE"),
            fields=fields,
            raw_person_fields=raw_person_fields.get(key, {}),
        )

    if set(raw_person_fields) != set(records):
        raise ValueError(
            "lexical and parsed BibTeX entry keys differ: "
            f"lexical={sorted(raw_person_fields)}, parsed={sorted(records)}"
        )
    return GoldBundle(
        source=source_kind,
        path=path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        records=records,
    )


def load_expected_samples(
    sample_path: Path,
    styles: list[str],
    dataset_path: Path,
) -> list[ExpectedSample]:
    """Load one expected sample for each unique key and requested style."""
    sample_path = Path(sample_path)
    dataset_path = Path(dataset_path)
    sample_keys = _load_sample_keys(sample_path)
    metadata = _load_metadata(dataset_path / "metadata.csv")
    requested_styles = list(dict.fromkeys(styles))
    references = _load_references(dataset_path / "ref_strings.jsonl", sample_keys, requested_styles)

    samples: list[ExpectedSample] = []
    for style in requested_styles:
        for key in sorted(sample_keys):
            reference = references.get((key, style))
            if reference is None:
                raise ValueError(
                    f"Reference for key={key}, style={style} not found in ref_strings.jsonl"
                )
            ref_string = reference["ref_string"]
            if not isinstance(ref_string, str) or not ref_string.strip():
                raise ValueError(f"Empty reference string for key={key}, style={style}")

            meta = metadata.get(key, {})
            samples.append(
                ExpectedSample(
                    sample_id=f"{key}::{style}",
                    key=key,
                    style=style,
                    entry_type=meta.get("entry_type", ""),
                    discipline=meta.get("discipline", ""),
                    subfield=meta.get("subfield", ""),
                    difficulty_flags=meta.get("difficulty_flags", ""),
                    style_impl=reference.get("style_impl", ""),
                    generation_status=reference.get("generation_status", ""),
                    reference=ref_string,
                )
            )
    return samples


def load_evaluation_sources(
    *,
    dataset_path: Path,
    sample_path: Path,
    gold_source: str,
    styles: list[str],
) -> EvaluationSources:
    dataset_path = Path(dataset_path)
    sample_path = Path(sample_path)
    if gold_source not in {"main", "full", "derived"}:
        raise ValueError(f"unsupported gold source: {gold_source}")
    gold_path = dataset_path / f"ja_bib_{gold_source}.bib"
    metadata_path = dataset_path / "metadata.csv"
    references_path = dataset_path / "ref_strings.jsonl"
    for path in (gold_path, metadata_path, references_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(f"required evaluation source not found: {path}")

    gold = load_gold(gold_path)
    metadata = _load_metadata(metadata_path)
    sample_keys = _load_sample_keys(sample_path)
    for key in sorted(sample_keys):
        if key not in metadata:
            raise ValueError(f"metadata key not found for sample: {key}")
        record = gold.records.get(key)
        if record is None:
            raise ValueError(f"gold key not found for sample: {key}")
        metadata_type = metadata[key].get("entry_type", "")
        if record.entry_type != metadata_type:
            raise ValueError(
                f"entry type mismatch for {key}: gold={record.entry_type!r}, "
                f"metadata={metadata_type!r}"
            )
        for style in dict.fromkeys(styles):
            observable_field_scope(style, metadata_type)

    samples = tuple(load_expected_samples(sample_path, styles, dataset_path))
    dataset_digest = hashlib.sha256()
    for path in sorted([references_path, metadata_path, sample_path]):
        dataset_digest.update(path.read_bytes())
    return EvaluationSources(
        gold=gold,
        samples=samples,
        metadata=SourceProvenance(path=metadata_path, sha256=_sha256(metadata_path)),
        references=SourceProvenance(
            path=references_path,
            sha256=_sha256(references_path),
        ),
        sample=SourceProvenance(path=sample_path, sha256=_sha256(sample_path)),
        dataset_hash=dataset_digest.hexdigest(),
    )


def load_gold_pair(dataset_path: Path) -> dict[str, GoldBundle]:
    """Load full and derived gold files and require matching key sets."""
    dataset_path = Path(dataset_path)
    full = load_gold(dataset_path / "ja_bib_full.bib")
    derived = load_gold(dataset_path / "ja_bib_derived.bib")
    if set(full.records) != set(derived.records):
        raise ValueError("Full and derived gold key sets differ")
    return {"full": full, "derived": derived}


def _gold_source(path: Path) -> str:
    stem = path.stem.lower()
    for source in ("main", "full", "derived"):
        if stem == source or stem.endswith(f"_{source}"):
            return source
    raise ValueError(f"Cannot determine gold source from path: {path}")


def _split_person_list(value: str) -> list[str]:
    result: list[str] = []
    depth = 0
    current: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "{":
            depth += 1
            current.append(char)
        elif char == "}":
            if depth > 0:
                depth -= 1
            current.append(char)
        elif (
            depth == 0
            and value[index : index + 3] == "and"
            and index > 0
            and value[index - 1].isspace()
            and index + 3 < len(value)
            and value[index + 3].isspace()
        ):
            person = "".join(current).strip()
            if person:
                result.append(person)
            current = []
            index += 3
            continue
        else:
            current.append(char)
        index += 1

    person = "".join(current).strip()
    if person:
        result.append(person)
    return result


def _load_sample_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            key = line.strip()
            if key:
                keys.add(key)
    return keys


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = row.get("key", "")
            if not key:
                raise ValueError(f"Metadata row in {path} has no key")
            if key in result:
                raise ValueError(f"Duplicate metadata key: {key}")
            result[key] = {
                "entry_type": row.get("entry_type", ""),
                "discipline": row.get("discipline", ""),
                "subfield": row.get("subfield", ""),
                "difficulty_flags": row.get("difficulty_flags", ""),
            }
    return result


def _load_references(
    path: Path,
    keys: set[str],
    styles: list[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    seen_rows: set[tuple[str, str]] = set()
    requested_styles = set(styles)
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                reference = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed reference JSON at {path}:{line_number}") from exc
            if not isinstance(reference, dict):
                raise ValueError(f"Reference row is not an object at {path}:{line_number}")
            key = reference.get("key")
            style = reference.get("style")
            if not isinstance(key, str) or not isinstance(style, str):
                raise ValueError(f"Reference identity missing at {path}:{line_number}")
            reference_key = (key, style)
            if reference_key in seen_rows:
                raise ValueError(f"Duplicate reference row: key={key}, style={style}")
            seen_rows.add(reference_key)
            if key in keys and style in requested_styles:
                result[reference_key] = reference
    return result
