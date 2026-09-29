"""GROBID client: call /api/processCitation with native BibTeX output."""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from scripts.models import NativeParseResult


class GrobidClientConfig(BaseModel):
    """GROBID connection settings."""

    base_url: str = "http://localhost:8070"
    endpoint: str = "/api/processCitation"
    timeout: float = 30.0
    consolidate_citations: int = 0
    include_raw_citations: int = 0


def _split_author_list(val_str: str) -> list[str]:
    result: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(val_str):
        ch = val_str[i]
        if ch == "{":
            depth += 1
            current.append(ch)
        elif ch == "}":
            if depth > 0:
                depth -= 1
            current.append(ch)
        elif depth == 0 and val_str[i:].startswith(" and "):
            name = "".join(current).strip()
            if name:
                result.append(name)
            current = []
            i += 4
            continue
        else:
            current.append(ch)
        i += 1
    name = "".join(current).strip()
    if name:
        result.append(name)
    return result


def parse_bibtex_response(
    bibtex_str: str, input_id: str
) -> tuple[NativeParseResult, str, str | None]:
    """Parse a GROBID BibTeX response into a NativeParseResult.

    Returns (result, raw_bibtex, error_kind). error_kind is None on success,
    or one of 'bibtex_parse_error', 'no_entry'."""
    raw = bibtex_str
    result = NativeParseResult()
    try:
        import bibtexparser
    except ImportError:
        return result, raw, "bibtex_parse_error"

    try:
        library = bibtexparser.loads(bibtex_str)
    except Exception:
        return result, raw, "bibtex_parse_error"

    if not library.entries:
        return result, raw, "no_entry"

    entry = library.entries[0]
    result.entry_type = entry.get("ENTRYTYPE")

    field_map: dict[str, str] = {
        "author": "author",
        "editor": "editor",
        "translator": "translator",
        "title": "title",
        "journal": "journal",
        "booktitle": "booktitle",
        "series": "series",
        "publisher": "publisher",
        "address": "address",
        "year": "year",
        "month": "month",
        "volume": "volume",
        "number": "number",
        "pages": "pages",
        "doi": "doi",
        "eprint": "eprint",
        "isbn": "isbn",
        "issn": "issn",
        "url": "url",
        "urldate": "urldate",
        "organization": "organization",
        "note": "note",
    }

    for key, val_str in entry.items():
        key_lower = key.lower()
        if key_lower in ("entrytype", "id"):
            continue
        if key_lower in field_map:
            attr = field_map[key_lower]
            if attr in ("author", "editor", "translator"):
                names = _split_author_list(str(val_str))
                if names:
                    setattr(result, attr, names)
            else:
                setattr(result, attr, str(val_str))
        elif key_lower not in ("raw", "date", "day"):
            result.unknown_fields[key_lower] = str(val_str)

    return result, raw, None


class GrobidClient:
    """HTTP client for GROBID citation parsing with native BibTeX output."""

    def __init__(self, config: GrobidClientConfig | None = None) -> None:
        self.config = config or GrobidClientConfig()
        self._http = httpx.Client(timeout=self.config.timeout)

    def check_health(self) -> bool:
        try:
            resp = self._http.get(f"{self.config.base_url}/api/isalive")
            return resp.status_code == 200 and resp.text.strip().lower() == "true"
        except Exception:
            return False

    def get_version(self) -> str | None:
        try:
            resp = self._http.get(f"{self.config.base_url}/api/version")
            if resp.status_code == 200:
                return resp.text.strip()
        except Exception:
            pass
        return None

    def parse_bibtex(self, reference: str) -> tuple[NativeParseResult, str, str | None]:
        """Parse a reference string as BibTeX via GROBID.

        Args:
            reference: The reference string to parse.

        Returns:
            Tuple of (NativeParseResult, raw_response_body, error_kind).
            error_kind is None on success, or one of:
            'transport_error', 'http_error', 'empty_response',
            'bibtex_parse_error', 'no_entry'.
        """
        url = self.config.base_url.rstrip("/") + self.config.endpoint
        data = {
            "citations": reference,
            "consolidateCitations": str(self.config.consolidate_citations),
            "includeRawCitations": str(self.config.include_raw_citations),
        }
        headers = {"Accept": "application/x-bibtex"}

        try:
            response = self._http.post(url, data=data, headers=headers)
            raw_body = response.text
        except httpx.HTTPError as exc:
            raw_body = f"HTTP error: {type(exc).__name__}: {exc}"
            return NativeParseResult(), raw_body, "transport_error"

        try:
            if response.status_code == 204 or not raw_body.strip():
                return NativeParseResult(), raw_body, "empty_response"
            response.raise_for_status()
        except Exception as exc:
            raw_body = f"HTTP {response.status_code}: {exc}\n{raw_body}"
            return NativeParseResult(), raw_body, "http_error"

        return parse_bibtex_response(raw_body, "")

    def close(self) -> None:
        self._http.close()
