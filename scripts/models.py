"""Data models for the bibliographic reference parsing experiment."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class BibliographicField(StrEnum):
    """Canonical bibliographic fields extractable from reference strings."""

    ENTRY_TYPE = "entry_type"
    AUTHOR = "author"
    EDITOR = "editor"
    TRANSLATOR = "translator"
    TITLE = "title"
    JOURNAL = "journal"
    BOOKTITLE = "booktitle"
    SERIES = "series"
    PUBLISHER = "publisher"
    ADDRESS = "address"
    DATE = "date"
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    VOLUME = "volume"
    NUMBER = "number"
    PAGES = "pages"
    DOI = "doi"
    EPRINT = "eprint"
    ISBN = "isbn"
    ISSN = "issn"
    URL = "url"
    URLDATE = "urldate"
    ORGANIZATION = "organization"
    NOTE = "note"


class ParserKind(StrEnum):
    GROBID = "grobid"
    LLM = "llm"


class ParseIssue(BaseModel):
    """A structured issue found while validating parser output."""

    kind: Literal["parse_error", "missing_id", "unknown_id", "duplicate_id"]
    opaque_input_id: str | None = None
    detail: str



class InputItem(BaseModel):
    """A single reference string with its metadata."""

    sample_id: str
    key: str
    style: str
    opaque_input_id: str
    reference: str
    entry_type: str = ""
    discipline: str = ""
    difficulty_flags: str = ""


class NativeParseResult(BaseModel):
    """Parsed output from a parser, before canonical normalization.

    All fields are optional. Parsers populate what they can extract.
    ``unknown_fields`` captures fields not covered by BibliographicField.
    """

    entry_type: str | None = None
    author: list[str] | None = None
    editor: list[str] | None = None
    translator: list[str] | None = None
    title: str | None = None
    journal: str | None = None
    booktitle: str | None = None
    series: str | None = None
    publisher: str | None = None
    address: str | None = None
    date: str | None = None
    year: str | None = None
    month: str | None = None
    day: str | None = None
    volume: str | None = None
    number: str | None = None
    pages: str | None = None
    doi: str | None = None
    eprint: str | None = None
    isbn: str | None = None
    issn: str | None = None
    url: str | None = None
    urldate: str | None = None
    organization: str | None = None
    note: str | None = None

    unknown_fields: dict[str, Any] = Field(default_factory=dict)


class CallLog(BaseModel):
    """One API call record, written to calls.jsonl."""

    run_id: str
    call_id: str
    style: str
    batch_index: int
    attempt_index: int = 0
    expected_input_ids: list[str] = Field(default_factory=list)
    request_path: str | None = None
    response_path: str | None = None
    status: str = "ok"  # "ok" | "error"
    error_stage: str | None = None
    error_kind: str | None = None
    error_detail: str | None = None
    issues: list[dict[str, Any]] | None = None
    latency_ms: int = 0
    token_usage: dict[str, int] | None = None
    finish_reason: str | None = None
    http_status: int | None = None
    content_type: str | None = None


class ItemLog(BaseModel):
    """One sample result, written to items.jsonl."""

    run_id: str
    sample_id: str
    key: str
    style: str
    opaque_input_id: str
    batch_index: int
    batch_position: int
    reference: str
    entry_type: str = ""
    discipline: str = ""
    difficulty_flags: str = ""
    call_id: str | None = None
    attempt_count: int = 1
    parse_status: str = "ok"  # "ok" | "call_error" | "retry_exhausted" | "empty_response" etc.
    parsed_output: NativeParseResult | None = None
    error_stage: str | None = None
    error_kind: str | None = None
    error_detail: str | None = None
