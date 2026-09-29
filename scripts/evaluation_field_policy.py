"""Canonical observable-field policy and deterministic comparison candidates."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from typing import Self

import regex

from scripts.evaluation_models import (
    FieldEligibility,
    FieldScopeStatus,
    FieldStatus,
)

FIELD_POLICY_ID = "observable-fields-v1"

STYLE_ORDER = (
    "jecon-kalc",
    "ipsjsort",
    "jsai",
    "jplain",
    "sist02",
    "jpa2022",
    "chicago-author-date",
)

ENTRY_TYPE_ORDER = ("article", "inproceedings", "book", "misc")

EVALUATION_FIELD_ORDER = (
    "author",
    "editor",
    "translator",
    "title",
    "journal",
    "booktitle",
    "series",
    "publisher",
    "address",
    "edition",
    "year",
    "volume",
    "number",
    "pages",
    "doi",
    "isbn",
    "url",
    "urldate",
    "organization",
)

ALWAYS_FIELD_SCOPES = {
    ("jecon-kalc", "article"): (
        "author",
        "title",
        "journal",
        "volume",
        "number",
        "pages",
        "year",
        "doi",
    ),
    ("ipsjsort", "article"): (
        "author",
        "title",
        "journal",
        "volume",
        "number",
        "pages",
        "year",
        "doi",
    ),
    ("jsai", "article"): (
        "author",
        "title",
        "journal",
        "volume",
        "number",
        "pages",
        "year",
    ),
    ("jplain", "article"): (
        "author",
        "title",
        "journal",
        "volume",
        "number",
        "pages",
        "year",
    ),
    ("sist02", "article"): (
        "author",
        "title",
        "journal",
        "volume",
        "number",
        "pages",
        "year",
        "url",
    ),
    ("jpa2022", "article"): (
        "author",
        "title",
        "journal",
        "volume",
        "number",
        "pages",
        "year",
        "doi",
    ),
    ("chicago-author-date", "article"): (
        "author",
        "title",
        "journal",
        "volume",
        "number",
        "pages",
        "year",
        "doi",
    ),
    ("jecon-kalc", "inproceedings"): (
        "author",
        "title",
        "booktitle",
        "volume",
        "number",
        "pages",
        "year",
        "publisher",
        "organization",
        "url",
        "doi",
    ),
    ("ipsjsort", "inproceedings"): (
        "author",
        "title",
        "booktitle",
        "volume",
        "number",
        "pages",
        "year",
        "publisher",
        "organization",
        "url",
        "doi",
    ),
    ("jsai", "inproceedings"): (
        "author",
        "title",
        "booktitle",
        "volume",
        "pages",
        "year",
        "publisher",
        "organization",
    ),
    ("jplain", "inproceedings"): (
        "author",
        "title",
        "booktitle",
        "volume",
        "pages",
        "year",
        "publisher",
        "organization",
    ),
    ("sist02", "inproceedings"): (
        "author",
        "title",
        "booktitle",
        "pages",
        "year",
        "publisher",
        "organization",
        "url",
    ),
    ("jpa2022", "inproceedings"): (
        "author",
        "title",
        "booktitle",
        "volume",
        "pages",
        "year",
        "url",
    ),
    ("chicago-author-date", "inproceedings"): (
        "author",
        "title",
        "booktitle",
        "volume",
        "pages",
        "year",
        "url",
        "doi",
    ),
    ("jecon-kalc", "book"): (
        "author",
        "translator",
        "title",
        "publisher",
        "year",
        "address",
        "series",
        "volume",
        "edition",
    ),
    ("ipsjsort", "book"): (
        "author",
        "title",
        "publisher",
        "year",
        "address",
        "edition",
    ),
    ("jsai", "book"): (
        "author",
        "title",
        "publisher",
        "year",
        "address",
        "series",
        "volume",
        "edition",
    ),
    ("jplain", "book"): (
        "author",
        "title",
        "publisher",
        "year",
        "address",
        "series",
        "volume",
        "edition",
    ),
    ("sist02", "book"): (
        "author",
        "translator",
        "title",
        "publisher",
        "year",
        "address",
        "series",
        "edition",
        "isbn",
    ),
    ("jpa2022", "book"): ("author", "title", "publisher", "year", "series"),
    ("chicago-author-date", "book"): (
        "author",
        "editor",
        "translator",
        "title",
        "publisher",
        "year",
        "series",
        "volume",
        "edition",
    ),
    ("jecon-kalc", "misc"): ("author", "title", "url", "year"),
    ("ipsjsort", "misc"): ("author", "title", "url", "urldate", "year"),
    ("jsai", "misc"): ("author", "title", "url", "urldate", "year"),
    ("jplain", "misc"): ("author", "title", "url", "urldate", "year"),
    ("sist02", "misc"): ("author", "title", "url", "urldate", "year"),
    ("jpa2022", "misc"): ("author", "title", "url", "urldate", "year"),
    ("chicago-author-date", "misc"): ("author", "title", "url", "year"),
}

PERSON_FIELDS = frozenset({"author", "editor", "translator"})

# The fields a style renders as running text, and so may recase or respace.
RENDERED_TEXT_FIELDS = frozenset(
    {"title", "booktitle", "journal", "series", "publisher", "organization", "address"}
)

_EXPECTED_SCOPE_KEYS = tuple(
    (style, entry_type) for entry_type in ENTRY_TYPE_ORDER for style in STYLE_ORDER
)
if tuple(ALWAYS_FIELD_SCOPES) != _EXPECTED_SCOPE_KEYS:
    raise RuntimeError("observable field scopes must contain every style/type pair exactly once")
for _scope_key, _scope_fields in ALWAYS_FIELD_SCOPES.items():
    if "entry_type" in _scope_fields:
        raise RuntimeError(f"entry_type cannot be evaluated in scope {_scope_key!r}")
    if unknown := set(_scope_fields) - set(EVALUATION_FIELD_ORDER):
        raise RuntimeError(f"unknown evaluation fields in scope {_scope_key!r}: {sorted(unknown)}")


@dataclass(frozen=True)
class PolicyWarning:
    code: str
    message: str


@dataclass(frozen=True)
class AcceptedCandidate:
    candidate_id: str
    text: str


@dataclass(frozen=True)
class CandidateBuildResult:
    candidates: tuple[AcceptedCandidate, ...]
    warnings: tuple[PolicyWarning, ...] = ()


@dataclass(frozen=True)
class TextNormalization:
    text: str
    warnings: tuple[PolicyWarning, ...] = ()


@dataclass(frozen=True)
class VisibilityResult:
    observable: bool | None
    reason: str | None
    visible_person_count: int | None = None
    warnings: tuple[PolicyWarning, ...] = ()


@dataclass(frozen=True)
class CandidateSelection:
    candidate_id: str
    selected_gold: str
    normalized_prediction: str
    distance_raw: int
    distance_normalized: float


@dataclass(frozen=True)
class PersonName:
    family: str | None
    suffix: str | None
    given: str | None
    opaque: str | None


@dataclass(frozen=True)
class PersonCandidateBranch:
    candidate_priority: tuple[int, int]
    mode: str
    text: str


@dataclass(frozen=True)
class PersonCandidateDag:
    layers: tuple[tuple[PersonCandidateBranch, ...], ...]
    marker_layer: tuple[PersonCandidateBranch, ...]
    path_count: int
    warnings: tuple[PolicyWarning, ...] = ()

    @classmethod
    def from_layers(
        cls,
        layers: tuple[tuple[PersonCandidateBranch, ...], ...],
        *,
        marker_layer: tuple[PersonCandidateBranch, ...] | None = None,
        warnings: tuple[PolicyWarning, ...] = (),
    ) -> Self:
        if marker_layer is None:
            marker_layer = (PersonCandidateBranch((len(layers), 0), "absent", ""),)
        sizes = [len(layer) for layer in layers]
        sizes.append(len(marker_layer))
        path_count = math.prod(sizes) if sizes and all(sizes) else 0
        return cls(
            layers=layers,
            marker_layer=marker_layer,
            path_count=path_count,
            warnings=warnings,
        )

    def brute_force_strings_for_test(self) -> tuple[str, ...]:
        if self.path_count > 10_000:
            raise ValueError("brute-force helper is limited to 10,000 candidate paths")
        if self.path_count == 0:
            return ()
        return tuple(
            "".join(branch.text for branch in path)
            for path in itertools.product(*self.layers, self.marker_layer)
        )


@dataclass(frozen=True)
class FieldPolicyResult:
    status: FieldStatus
    exact_match: bool
    scope_status: FieldScopeStatus
    gold_present: bool
    observable: bool | None
    eligibility: FieldEligibility
    exclusion_reason: str | None
    candidate_path_count: int
    selected_candidate_id: str | None
    selected_gold: str | None
    normalized_prediction: str | None
    distance_raw: int | None
    distance_normalized: float | None
    include_completion: bool
    include_available: bool
    warnings: tuple[PolicyWarning, ...] = ()


def observable_field_scope(style: str, entry_type: str) -> tuple[str, ...]:
    try:
        return ALWAYS_FIELD_SCOPES[(style, entry_type)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported observable field scope: style={style!r}, entry_type={entry_type!r}"
        ) from exc


def field_policy_payload() -> dict[str, object]:
    return {
        "field_policy_id": FIELD_POLICY_ID,
        "styles": list(STYLE_ORDER),
        "entry_types": list(ENTRY_TYPE_ORDER),
        "fields": list(EVALUATION_FIELD_ORDER),
        "scope": [
            {"style": style, "entry_type": kind, "fields": list(fields)}
            for (style, kind), fields in ALWAYS_FIELD_SCOPES.items()
        ],
        "visibility": {
            "rule": "gold-reference-v2",
            "normalization": [
                "NFKC",
                "casefold",
                "remove-whitespace",
                "unify-dashes",
                "remove-quotes",
                "remove-bibtex-grouping-braces",
            ],
            "year_match": {
                "base_year": "first-four-digit-sequence",
                "base_year_reference_match": "normalized-digit-bounded",
                "no_base_year": "normalized-gold-containment",
            },
            "person_match": {
                "personal": "all-visible-people-in-gold-order",
                "opaque": "exact-normalized-name-only",
                "distance_only_modes": ["opaque-drop-and"],
            },
        },
        "distance_normalization": "remove-bibtex-grouping-braces-and-special-escapes-v1",
        "bibtex_special_escapes": ["%", "&", "_", "#", "$"],
        "scalar_candidates": {
            "canonical": True,
            "edition": {
                "jecon-kalc": ["第", "{canonical}", "版"],
                "ipsjsort": ["{canonical}", " edition"],
                "chicago-author-date": ["{canonical}", " 版"],
            },
            "pages": {
                "rule": "reference-confirmed-range-capture-v3",
                "styles": "all",
                "end_shortening": "chicago-contextual",
                "separator_spacing": "capture-exact-reference-code-points",
                "full_field_capture": True,
                "separator_forms": [name for name, _ in _PAGE_SEPARATOR_FORMS],
            },
            "urldate": "reference-confirmed-date-variant-v1",
            "rendered_text": {
                "rule": "reference-confirmed-render-capture-v1",
                "fields": sorted(RENDERED_TEXT_FIELDS),
                "tolerated": ["case", "whitespace-width"],
                "capture": "exact-reference-code-points",
            },
        },
        "person_candidates": {
            "rule": "person-dag-v1",
            "grammar": {
                "zero-top-level-commas": "opaque",
                "one-top-level-comma": "family-given",
                "two-top-level-commas": "family-suffix-given",
                "three-or-more-top-level-commas": "candidate-unresolved",
            },
            "mode_order": [
                "full-family-given",
                "full-given-family",
                "all-initials-family-given",
                "all-initials-given-family",
                "first-initial-family-given",
                "first-initial-given-family",
                "opaque-exact",
                "opaque-drop-and",
            ],
            "truncation": {
                "chicago-author-date": {"threshold": 7, "visible": 3},
                "jecon-kalc": {"threshold": 9, "visible": 3},
            },
            "markers": ["ほか", "terminal-他", "et al.", "others"],
            "marker_candidate_order": ["absent", "present"],
        },
        "selection_order": {
            "scalar": ["distance_raw", "distance_normalized", "candidate_id"],
            "person": [
                "distance_raw",
                "distance_normalized",
                "candidate_priority_tuple",
            ],
        },
    }


def field_policy_hash() -> str:
    payload = json.dumps(
        field_policy_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def raw_value_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(raw_value_present(item) for item in value)
    if isinstance(value, dict):
        return bool(value)
    return True


def _preceding_backslashes(value: str, index: int) -> int:
    count = 0
    index -= 1
    while index >= 0 and value[index] == "\\":
        count += 1
        index -= 1
    return count


def remove_bibtex_grouping_braces(
    source: str,
) -> tuple[str, tuple[PolicyWarning, ...]]:
    stack: list[int] = []
    balanced: set[int] = set()
    unmatched: list[int] = []
    for index, character in enumerate(source):
        if character not in "{}" or _preceding_backslashes(source, index) % 2:
            continue
        if character == "{":
            stack.append(index)
        elif stack:
            opening = stack.pop()
            balanced.update((opening, index))
        else:
            unmatched.append(index)
    unmatched.extend(stack)

    output: list[str] = []
    for index, character in enumerate(source):
        if character not in "{}":
            output.append(character)
            continue
        backslashes = _preceding_backslashes(source, index)
        if backslashes % 2:
            output.pop()
            output.append(character)
        elif index not in balanced:
            output.append(character)

    warnings = tuple(
        PolicyWarning(
            code="unmatched-brace",
            message=f"unmatched brace at index {index}",
        )
        for index in sorted(unmatched)
    )
    return "".join(output), warnings


_QUOTES = regex.compile(r"\p{Quotation_Mark}")
_DASHES = regex.compile(r"\p{Dash_Punctuation}")
_WHITESPACE = regex.compile(r"\s+")
_PUNCTUATION = regex.compile(r"\p{Punctuation}")
_RANGE_DASH = r"(?:--|[-‐‑‒–—―])"
_RANGE_DASH_CHARACTER = regex.compile(r"[-‐‑‒–—―]")
# The three ways a page range may be written: the ASCII hyphen the gold records
# hold, the en dash every style prints, and the double hyphen BibTeX uses to
# encode that en dash. The prompt asks for valid BibTeX, so all three denote the
# same range and none of them is evidence about extraction.
_PAGE_SEPARATOR_FORMS = (
    ("ascii-hyphen", "-"),
    ("en-dash", "–"),
    ("double-hyphen", "--"),
)
_BIBTEX_SPECIAL_ESCAPE = regex.compile(r"\\([%&_#$])")


def _normalize_visibility_text(value: object) -> TextNormalization:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized, warnings = remove_bibtex_grouping_braces(normalized)
    normalized = _QUOTES.sub("", normalized)
    normalized = _WHITESPACE.sub("", normalized)
    normalized = _DASHES.sub("-", normalized)
    return TextNormalization(normalized, warnings)


def _normalize_identifier(value: object) -> TextNormalization:
    normalized = _normalize_visibility_text(value)
    return TextNormalization(
        _PUNCTUATION.sub("", normalized.text),
        normalized.warnings,
    )


def _bounded_contains(reference: str, candidate: str) -> bool:
    if not candidate:
        return True
    left = r"(?<!\d)" if candidate[0].isdigit() else ""
    right = r"(?!\d)" if candidate[-1].isdigit() else ""
    return regex.search(f"{left}{regex.escape(candidate)}{right}", reference) is not None


def _date_variants(value: str) -> tuple[str, ...]:
    parts = regex.fullmatch(r"\s*(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?\s*", value)
    if parts is None:
        return (value,)
    year, month_text, day_text = parts.groups()
    month = int(month_text)
    day = int(day_text)
    variants = (
        f"{year}-{month:02d}-{day:02d}",
        f"{year}-{month}-{day}",
        f"{year}/{month:02d}/{day:02d}",
        f"{year}/{month}/{day}",
        f"{year}.{month:02d}.{day:02d}",
        f"{year}.{month}.{day}",
        f"{year}年{month:02d}月{day:02d}日",
        f"{year}年{month}月{day}日",
    )
    return tuple(dict.fromkeys(variants))


def evaluate_scalar_visibility(
    *, field: str, gold: str, style: str, reference: str
) -> VisibilityResult:
    gold_normalized = _normalize_visibility_text(gold)
    reference_normalized = _normalize_visibility_text(reference)
    warnings = (*gold_normalized.warnings, *reference_normalized.warnings)
    if not gold_normalized.text:
        return VisibilityResult(True, None, warnings=warnings)

    if field in {"doi", "isbn"}:
        gold_identifier = _normalize_identifier(gold)
        reference_identifier = _normalize_identifier(reference)
        if gold_identifier.text in reference_identifier.text:
            return VisibilityResult(True, None, warnings=warnings)
        return VisibilityResult(None, "visibility:identifier-not-found", warnings=warnings)

    if field == "urldate":
        for variant in _date_variants(gold_normalized.text):
            candidate = _normalize_visibility_text(variant).text
            if _bounded_contains(reference_normalized.text, candidate):
                return VisibilityResult(True, None, warnings=warnings)
        return VisibilityResult(None, "visibility:plain-not-found", warnings=warnings)

    if field == "year":
        base_year = regex.search(r"\d{4}", gold_normalized.text)
        if base_year is not None:
            if _bounded_contains(reference_normalized.text, base_year.group()):
                return VisibilityResult(True, None, warnings=warnings)
            return VisibilityResult(None, "visibility:numeric-not-found", warnings=warnings)
        if gold_normalized.text in reference_normalized.text:
            return VisibilityResult(True, None, warnings=warnings)
        return VisibilityResult(None, "visibility:plain-not-found", warnings=warnings)

    candidates = (AcceptedCandidate("canonical", gold),)
    if field == "pages" and style == "chicago-author-date":
        candidates = build_scalar_candidates(
            field=field,
            style=style,
            gold=gold,
            reference=reference,
        ).candidates

    if field in {"volume", "number", "pages", "edition"}:
        for candidate in candidates:
            candidate_text = _normalize_visibility_text(candidate.text).text
            candidate_text = regex.sub(r"-+", "-", candidate_text)
            normalized_reference = regex.sub(r"-+", "-", reference_normalized.text)
            if _bounded_contains(normalized_reference, candidate_text):
                return VisibilityResult(True, None, warnings=warnings)
        return VisibilityResult(None, "visibility:numeric-not-found", warnings=warnings)

    if gold_normalized.text in reference_normalized.text:
        return VisibilityResult(True, None, warnings=warnings)
    return VisibilityResult(None, "visibility:plain-not-found", warnings=warnings)


def _final_numeric_segment(value: str) -> tuple[str, str, str] | None:
    matches = tuple(regex.finditer(r"\d+", value))
    if not matches:
        return None
    final = matches[-1]
    return value[: final.start()], final.group(), value[final.end() :]


def _chicago_shortened_digits(start_digits: str, end_digits: str) -> str:
    start = int(start_digits)
    end = int(end_digits)
    if end < start or start < 100 or start % 100 == 0:
        return end_digits

    common_length = 0
    for start_digit, end_digit in zip(start_digits, end_digits, strict=False):
        if start_digit != end_digit:
            break
        common_length += 1
    changed_length = len(end_digits) - common_length
    minimum_length = 1 if 1 <= start % 100 <= 9 else 2
    suffix_length = max(changed_length, minimum_length)
    return end_digits[-suffix_length:]


def _page_literal_pattern(value: str) -> str:
    parts: list[str] = []
    cursor = 0
    for separator in regex.finditer(r"\s+|,", value):
        parts.append(regex.escape(value[cursor : separator.start()]))
        parts.append(r"\s*,\s*" if separator.group() == "," else r"\s*")
        cursor = separator.end()
    parts.append(regex.escape(value[cursor:]))
    return "".join(parts)


def _page_range_captures(gold: str, reference: str) -> tuple[str, ...]:
    """Page forms the reference itself prints, captured with its own code points.

    Every range in the gold value is matched against the reference under either
    its full closing number or the Chicago-style shortened one, and the token is
    taken verbatim from the reference, so the separator, its spacing, and the
    dash character are the ones the style printed. A form is only ever returned
    when the reference confirms it.
    """
    range_pattern = regex.compile(
        rf"(?P<start>[\p{{L}}\p{{N}}_.:]+)\s*{_RANGE_DASH}\s*"
        r"(?P<end>[\p{L}\p{N}_.:]+)"
    )
    ranges: list[tuple[regex.Match[str], tuple[str, ...]]] = []
    for page_range in range_pattern.finditer(gold):
        start = page_range.group("start")
        end = page_range.group("end")
        start_numeric = _final_numeric_segment(start)
        end_numeric = _final_numeric_segment(end)
        if start_numeric is None or end_numeric is None:
            continue
        start_prefix, start_digits, _ = start_numeric
        end_prefix, end_digits, end_remainder = end_numeric
        rendered_prefix = "" if end_prefix and end_prefix == start_prefix else end_prefix
        rendered_ends = [f"{rendered_prefix}{end_digits}{end_remainder}"]
        shortened_digits = _chicago_shortened_digits(start_digits, end_digits)
        rendered_ends.append(f"{rendered_prefix}{shortened_digits}{end_remainder}")
        ranges.append((page_range, tuple(dict.fromkeys(rendered_ends))))

    captures: list[tuple[int, str]] = []
    for rendered_ends in itertools.product(*(options for _, options in ranges)):
        pattern_parts: list[str] = []
        cursor = 0
        for (page_range, _), rendered_end in zip(ranges, rendered_ends, strict=True):
            pattern_parts.append(_page_literal_pattern(gold[cursor : page_range.start()]))
            pattern_parts.append(regex.escape(page_range.group("start")))
            pattern_parts.append(rf"\s*{_RANGE_DASH}\s*")
            pattern_parts.append(regex.escape(rendered_end))
            cursor = page_range.end()
        pattern_parts.append(_page_literal_pattern(gold[cursor:]))
        reference_pattern = regex.compile(rf"(?<!\d)(?P<token>{''.join(pattern_parts)})(?!\d)")
        captures.extend(
            (match.start("token"), match.group("token"))
            for match in reference_pattern.finditer(reference)
        )
    return tuple(dict.fromkeys(token for _, token in sorted(captures, key=lambda item: item)))


def _rendered_literal_pattern(value: str) -> str:
    parts: list[str] = []
    cursor = 0
    for run in _WHITESPACE.finditer(value):
        parts.append(regex.escape(value[cursor : run.start()]))
        parts.append(r"\s+")
        cursor = run.end()
    parts.append(regex.escape(value[cursor:]))
    return "".join(parts)


def _reference_confirmed_captures(gold: str, reference: str) -> tuple[str, ...]:
    """The renderings of the gold value that the reference itself prints.

    A style recases what it renders, so jplain prints an initialism the gold
    holds in capitals with only its first letter capital, and the renderer
    narrows an ideographic space to an ordinary one. A parser that copies what
    it was shown was charged for the rendering rather than for what it read.
    The gold is located without regard to case or to the width of a space and
    the matching token is taken verbatim, so no rendering the reference does
    not print is admitted.
    """
    if not gold:
        return ()
    printed = regex.compile(_rendered_literal_pattern(gold), regex.IGNORECASE).finditer(reference)
    return tuple(dict.fromkeys(match.group() for match in printed if match.group() != gold))


def unescape_bibtex_specials(value: str) -> str:
    """Undo the backslash escaping BibTeX requires of its five special characters.

    An unescaped per cent sign opens a comment and an unescaped ampersand,
    underscore, hash or dollar is an error, so a parser asked for valid BibTeX
    must write them escaped. The gold records hold them bare, which makes the
    escaping a difference of encoding rather than of what was read. Grouping
    braces are left to remove_bibtex_grouping_braces, which already reads the
    backslashes that protect them.
    """
    return _BIBTEX_SPECIAL_ESCAPE.sub(r"\1", value)


def _page_separator_variants(
    candidates: list[AcceptedCandidate],
) -> list[AcceptedCandidate]:
    variants: list[AcceptedCandidate] = []
    for candidate in candidates:
        if _RANGE_DASH_CHARACTER.search(candidate.text) is None:
            continue
        for name, separator in _PAGE_SEPARATOR_FORMS:
            text = _RANGE_DASH_CHARACTER.sub(separator, candidate.text)
            if text == candidate.text:
                continue
            variants.append(
                AcceptedCandidate(f"pages:separator:{name}:{candidate.candidate_id}", text)
            )
    return variants


def build_scalar_candidates(
    *, field: str, style: str, gold: str, reference: str
) -> CandidateBuildResult:
    canonical, warnings = remove_bibtex_grouping_braces(gold)
    canonical = unescape_bibtex_specials(canonical)
    if not canonical:
        return CandidateBuildResult((), warnings)

    candidates = [AcceptedCandidate("canonical", canonical)]
    if field == "edition":
        rendered_by_style = {
            "jecon-kalc": ("edition:jecon-rendered", f"第{canonical}版"),
            "ipsjsort": ("edition:ipsjsort-rendered", f"{canonical} edition"),
            "chicago-author-date": (
                "edition:chicago-rendered",
                f"{canonical} 版",
            ),
        }
        rendered = rendered_by_style.get(style)
        if rendered is not None and rendered[1] in reference:
            candidates.append(AcceptedCandidate(*rendered))
    elif field == "pages":
        for index, rendered in enumerate(_page_range_captures(canonical, reference)):
            candidates.append(AcceptedCandidate(f"pages:reference-confirmed:{index}", rendered))
        candidates.extend(_page_separator_variants(candidates))
    elif field == "urldate":
        for index, variant in enumerate(_date_variants(canonical)):
            if variant != canonical and _bounded_contains(reference, variant):
                candidates.append(
                    AcceptedCandidate(f"urldate:reference-confirmed:{index}", variant)
                )
    elif field in RENDERED_TEXT_FIELDS:
        for index, printed in enumerate(_reference_confirmed_captures(canonical, reference)):
            candidates.append(AcceptedCandidate(f"{field}:reference-confirmed:{index}", printed))

    deduplicated: list[AcceptedCandidate] = []
    seen_text: set[str] = set()
    for candidate in candidates:
        if candidate.text in seen_text:
            continue
        seen_text.add(candidate.text)
        deduplicated.append(candidate)
    return CandidateBuildResult(tuple(deduplicated), warnings)


def normalize_scalar_prediction(value: object) -> TextNormalization:
    normalized, warnings = remove_bibtex_grouping_braces(str(value))
    return TextNormalization(unescape_bibtex_specials(normalized), warnings)


def levenshtein_distance(left: str, right: str) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            substitution = previous[column - 1] + (left_character != right_character)
            current.append(min(current[column - 1] + 1, previous[column] + 1, substitution))
        previous = current
    return previous[-1]


def select_candidates(
    candidates: tuple[AcceptedCandidate, ...], prediction: str
) -> CandidateSelection:
    ranked: list[tuple[int, float, str, AcceptedCandidate]] = []
    for candidate in candidates:
        raw = levenshtein_distance(candidate.text, prediction)
        denominator = max(len(candidate.text), len(prediction))
        normalized = raw / denominator if denominator else 0.0
        ranked.append((raw, normalized, candidate.candidate_id, candidate))
    raw, normalized, _, candidate = min(ranked)
    return CandidateSelection(
        candidate_id=candidate.candidate_id,
        selected_gold=candidate.text,
        normalized_prediction=prediction,
        distance_raw=raw,
        distance_normalized=normalized,
    )


def _is_unescaped_brace(value: str, index: int) -> bool:
    return value[index] in "{}" and _preceding_backslashes(value, index) % 2 == 0


def _split_top_level_and(value: str) -> tuple[str, ...]:
    result: list[str] = []
    current: list[str] = []
    depth = 0
    index = 0
    while index < len(value):
        character = value[index]
        if character == "{" and _is_unescaped_brace(value, index):
            depth += 1
        elif character == "}" and _is_unescaped_brace(value, index) and depth:
            depth -= 1
        if (
            depth == 0
            and value[index : index + 3].casefold() == "and"
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
        current.append(character)
        index += 1
    person = "".join(current).strip()
    if person:
        result.append(person)
    return tuple(result)


def _split_top_level_commas(value: str) -> tuple[str, ...]:
    result: list[str] = []
    current: list[str] = []
    depth = 0
    for index, character in enumerate(value):
        if character == "{" and _is_unescaped_brace(value, index):
            depth += 1
        elif character == "}" and _is_unescaped_brace(value, index) and depth:
            depth -= 1
        if character == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    result.append("".join(current).strip())
    return tuple(result)


def _fully_brace_protected(value: str) -> bool:
    value = value.strip()
    if len(value) < 2 or value[0] != "{" or value[-1] != "}":
        return False
    depth = 0
    for index, character in enumerate(value):
        if character == "{" and _is_unescaped_brace(value, index):
            depth += 1
        elif character == "}" and _is_unescaped_brace(value, index):
            depth -= 1
            if depth == 0 and index != len(value) - 1:
                return False
    return depth == 0


def _without_grouping_braces(value: str) -> str:
    return remove_bibtex_grouping_braces(value)[0].strip()


def parse_person_list(raw_gold: str) -> tuple[PersonName, ...]:
    people: list[PersonName] = []
    for raw_person in _split_top_level_and(raw_gold):
        if _fully_brace_protected(raw_person):
            people.append(
                PersonName(
                    family=None,
                    suffix=None,
                    given=None,
                    opaque=_without_grouping_braces(raw_person),
                )
            )
            continue
        comma_parts = _split_top_level_commas(raw_person)
        if len(comma_parts) > 3:
            raise ValueError("person names with three or more top-level commas are unsupported")
        parts = tuple(_without_grouping_braces(part) for part in comma_parts)
        if any(not part for part in parts):
            raise ValueError("person name contains an empty comma-delimited component")
        if len(parts) == 3:
            family, suffix, given = parts
        elif len(parts) == 2:
            family, given = parts
            suffix = None
        else:
            people.append(
                PersonName(
                    family=None,
                    suffix=None,
                    given=None,
                    opaque=parts[0],
                )
            )
            continue
        people.append(
            PersonName(
                family=family,
                suffix=suffix,
                given=given,
                opaque=None,
            )
        )
    return tuple(people)


def _structure_text(value: str) -> str:
    without_braces, _ = remove_bibtex_grouping_braces(value)
    return "".join(
        character
        for character in without_braces
        if character.isalnum() or unicodedata.category(character).startswith("M")
    )


def _normalize_person_visibility_structure(value: str) -> TextNormalization:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized, warnings = remove_bibtex_grouping_braces(normalized)
    return TextNormalization(
        "".join(character for character in normalized if character.isalnum()),
        warnings,
    )


_INITIAL_STRUCTURE = regex.compile(r"[.\p{Dash_Punctuation}'’ʼ]")


def _latin_initial(token: str) -> str | None:
    stripped = _INITIAL_STRUCTURE.sub("", token)
    alphabetic = [character for character in stripped if character.isalpha()]
    if not alphabetic:
        return None
    if any("LATIN" not in unicodedata.name(character, "") for character in alphabetic):
        return None
    return alphabetic[0]


# The second tuple coordinate is the approved mode priority. Serialized candidate-ID
# strings are output labels only and never lexical tie-break keys.
_PERSON_MODE_PRIORITY = (
    "full-family-given",
    "full-given-family",
    "all-initials-family-given",
    "all-initials-given-family",
    "first-initial-family-given",
    "first-initial-given-family",
    "opaque-exact",
    "opaque-drop-and",
)


def _person_layer(person: PersonName, person_index: int) -> tuple[PersonCandidateBranch, ...]:
    candidate_text: dict[str, str] = {}
    if person.opaque is not None:
        candidate_text["opaque-exact"] = _structure_text(person.opaque)
        dropped = regex.sub(r"(?i)(?<!\p{L})and(?!\p{L})", "", person.opaque)
        candidate_text["opaque-drop-and"] = _structure_text(dropped)
    else:
        family = _structure_text(f"{person.family or ''}{person.suffix or ''}")
        given = _structure_text(person.given or "")
        candidate_text["full-family-given"] = f"{family}{given}"
        candidate_text["full-given-family"] = f"{given}{family}"
        given_tokens = tuple(regex.split(r"\s+", person.given or ""))
        initials = tuple(_latin_initial(token) for token in given_tokens if token)
        if initials and all(initial is not None for initial in initials):
            all_initials = "".join(initial for initial in initials if initial is not None)
            first_initial = initials[0] or ""
            candidate_text["all-initials-family-given"] = f"{family}{all_initials}"
            candidate_text["all-initials-given-family"] = f"{all_initials}{family}"
            candidate_text["first-initial-family-given"] = f"{family}{first_initial}"
            candidate_text["first-initial-given-family"] = f"{first_initial}{family}"

    branches: list[PersonCandidateBranch] = []
    seen_text: set[str] = set()
    for mode_index, mode in enumerate(_PERSON_MODE_PRIORITY):
        text = candidate_text.get(mode, "")
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        branches.append(PersonCandidateBranch((person_index, mode_index), mode, text))
    return tuple(branches)


def _person_visibility_candidates(person: PersonName) -> tuple[str, ...]:
    if person.opaque is not None:
        raw_candidates = [person.opaque]
    else:
        family = f"{person.family or ''}{person.suffix or ''}"
        given = person.given or ""
        raw_candidates = [f"{family}{given}", f"{given}{family}"]
        given_tokens = tuple(regex.split(r"\s+", unicodedata.normalize("NFKC", given)))
        initials = tuple(_latin_initial(token) for token in given_tokens if token)
        if initials and all(initial is not None for initial in initials):
            all_initials = "".join(initial for initial in initials if initial is not None)
            first_initial = initials[0] or ""
            raw_candidates.extend(
                (
                    f"{family}{all_initials}",
                    f"{all_initials}{family}",
                    f"{family}{first_initial}",
                    f"{first_initial}{family}",
                )
            )

    candidates: list[str] = []
    seen: set[str] = set()
    for raw_candidate in raw_candidates:
        candidate = _normalize_person_visibility_structure(raw_candidate).text
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return tuple(candidates)


_TRUNCATION_RULES = {
    "chicago-author-date": (7, 3),
    "jecon-kalc": (9, 3),
}

_ET_AL_MARKER = regex.compile(r"(?i)(?<!\p{L})et\s+al\.(?!\p{L})")
_OTHERS_ELEMENT = regex.compile(r"(?i)^[\s\p{P}]*others[\s\p{P}]*$")
_TERMINAL_OTHER_MARKER = regex.compile(r"(?<![\p{L}\p{N}])他[\s\p{P}]*$")
_HOKA_MARKER = regex.compile(r"ほか")
_WORD_AND_DELIMITER = regex.compile(r"(?i)(?<=\s)and(?=\s)")


@dataclass(frozen=True)
class _PersonTextSpan:
    start: int
    end: int


def _is_others_person_element(value: str) -> bool:
    return not _fully_brace_protected(value) and _OTHERS_ELEMENT.fullmatch(value) is not None


def _opaque_person_structures(people: tuple[PersonName, ...]) -> tuple[str, ...]:
    return tuple(
        _normalize_person_visibility_structure(person.opaque).text
        for person in people
        if person.opaque is not None
    )


def _normalized_person_structure_map(
    value: str,
) -> tuple[str, tuple[_PersonTextSpan, ...]]:
    normalized = _normalize_person_visibility_structure(value).text
    mapped_parts: list[str] = []
    source_spans: list[_PersonTextSpan] = []
    for match in regex.finditer(r"\X", value):
        part = _normalize_person_visibility_structure(match.group()).text
        mapped_parts.append(part)
        source_spans.extend(_PersonTextSpan(match.start(), match.end()) for _ in part)

    # Grapheme mapping only locates source spans; the whole-string result remains authoritative.
    if "".join(mapped_parts) == normalized:
        return normalized, tuple(source_spans)

    boundaries = [0, *(match.end() for match in regex.finditer(r"\X", value))]
    prefix_lengths = [
        len(_normalize_person_visibility_structure(value[:boundary]).text)
        for boundary in boundaries
    ]
    source_spans = []
    monotonic = all(left <= right for left, right in itertools.pairwise(prefix_lengths))
    for index in range(len(normalized)):
        if monotonic:
            start_index = bisect_right(prefix_lengths, index) - 1
            end_index = bisect_right(prefix_lengths, index + 1) - 1
        else:
            start_index = max(
                position for position, length in enumerate(prefix_lengths) if length <= index
            )
            end_index = max(
                position for position, length in enumerate(prefix_lengths) if length <= index + 1
            )
        source_spans.append(_PersonTextSpan(boundaries[start_index], boundaries[end_index]))
    return normalized, tuple(source_spans)


def _expand_person_structure_span(value: str, span: _PersonTextSpan) -> _PersonTextSpan:
    start = span.start
    while start > 0:
        character = value[start - 1]
        if character in ";&" or _normalize_person_visibility_structure(character).text:
            break
        start -= 1

    end = span.end
    while end < len(value):
        character = value[end]
        if character in ";&" or _normalize_person_visibility_structure(character).text:
            break
        end += 1
    return _PersonTextSpan(start, end)


def _opaque_person_spans(
    value: str, opaque_person_structures: tuple[str, ...]
) -> tuple[_PersonTextSpan, ...]:
    opaque_structures = tuple(structure for structure in opaque_person_structures if structure)
    if not opaque_structures:
        return ()

    normalized, source_map = _normalized_person_structure_map(value)

    spans: list[_PersonTextSpan] = []
    search_start = 0
    for structure in opaque_structures:
        match_start = normalized.find(structure, search_start)
        if match_start >= 0:
            match_end = match_start + len(structure)
            spans.append(
                _expand_person_structure_span(
                    value,
                    _PersonTextSpan(
                        source_map[match_start].start,
                        source_map[match_end - 1].end,
                    ),
                )
            )
            search_start = match_end
    return tuple(sorted(spans, key=lambda span: (span.start, span.end)))


def _ordered_person_spans(
    value: str,
    people: tuple[PersonName, ...],
) -> tuple[tuple[_PersonTextSpan, ...], tuple[_PersonTextSpan, ...], bool]:
    normalized, source_map = _normalized_person_structure_map(value)
    previous_end = 0
    person_spans: list[_PersonTextSpan] = []
    opaque_spans: list[_PersonTextSpan] = []
    all_people_found = True
    for person in people:
        matches: list[tuple[int, int]] = []
        for candidate in _person_visibility_candidates(person):
            position = normalized.find(candidate, previous_end)
            if position >= 0:
                matches.append((position, position + len(candidate)))
        if not matches:
            all_people_found = False
            continue

        match_start, match_end = min(matches, key=lambda item: (item[0], -item[1]))
        source_span = _expand_person_structure_span(
            value,
            _PersonTextSpan(
                source_map[match_start].start,
                source_map[match_end - 1].end,
            ),
        )
        person_spans.append(source_span)
        if person.opaque is not None:
            opaque_spans.append(source_span)
        previous_end = match_end
    return tuple(person_spans), tuple(opaque_spans), all_people_found


def _delimiter_is_protected(
    start: int, end: int, protected_spans: tuple[_PersonTextSpan, ...]
) -> bool:
    return any(span.start <= start and end <= span.end for span in protected_spans)


def _trimmed_person_span(value: str, start: int, end: int) -> _PersonTextSpan | None:
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    return _PersonTextSpan(start, end) if start < end else None


def _person_element_spans(
    value: str, protected_spans: tuple[_PersonTextSpan, ...]
) -> tuple[_PersonTextSpan, ...]:
    result: list[_PersonTextSpan] = []
    depth = 0
    element_start = 0
    index = 0
    while index < len(value):
        character = value[index]
        if character == "{" and _is_unescaped_brace(value, index):
            depth += 1
        elif character == "}" and _is_unescaped_brace(value, index) and depth:
            depth -= 1

        delimiter_end = index
        if depth == 0 and character in ";&":
            delimiter_end = index + 1
        elif (
            depth == 0
            and value[index : index + 3].casefold() == "and"
            and index > 0
            and value[index - 1].isspace()
            and index + 3 < len(value)
            and value[index + 3].isspace()
        ):
            delimiter_end = index + 3

        if delimiter_end > index and not _delimiter_is_protected(
            index, delimiter_end, protected_spans
        ):
            if span := _trimmed_person_span(value, element_start, index):
                result.append(span)
            element_start = delimiter_end
            index = delimiter_end
            continue
        index += 1

    if span := _trimmed_person_span(value, element_start, len(value)):
        result.append(span)
    return tuple(result)


def _marker_spans(
    value: str,
    opaque_person_structures: tuple[str, ...] = (),
    *,
    protected_spans: tuple[_PersonTextSpan, ...] | None = None,
) -> tuple[_PersonTextSpan, ...]:
    if protected_spans is None:
        protected_spans = _opaque_person_spans(value, opaque_person_structures)
    spans = {
        span
        for span in _person_element_spans(value, protected_spans)
        if not _delimiter_is_protected(span.start, span.end, protected_spans)
        and _is_others_person_element(value[span.start : span.end])
    }
    for pattern in (_HOKA_MARKER, _TERMINAL_OTHER_MARKER, _ET_AL_MARKER):
        spans.update(
            _PersonTextSpan(match.start(), match.end())
            for match in pattern.finditer(value)
            if not _delimiter_is_protected(match.start(), match.end(), protected_spans)
        )
    return tuple(sorted(spans, key=lambda span: (span.start, span.end)))


def _marker_present(
    value: str,
    opaque_person_structures: tuple[str, ...] = (),
    *,
    protected_spans: tuple[_PersonTextSpan, ...] | None = None,
) -> bool:
    return bool(
        _marker_spans(
            value,
            opaque_person_structures,
            protected_spans=protected_spans,
        )
    )


def _visible_person_count(style: str, total: int) -> int:
    rule = _TRUNCATION_RULES.get(style)
    if rule is not None and total >= rule[0]:
        return rule[1]
    return total


def evaluate_person_visibility(raw_gold: str, *, style: str, reference: str) -> VisibilityResult:
    _, brace_warnings = remove_bibtex_grouping_braces(raw_gold)
    normalized_reference = _normalize_person_visibility_structure(reference)
    warnings = (*brace_warnings, *normalized_reference.warnings)
    try:
        people = parse_person_list(raw_gold)
    except ValueError:
        return VisibilityResult(
            None,
            "visibility:not-found",
            warnings=warnings,
        )
    visible_count = _visible_person_count(style, len(people))
    marker_expected = visible_count < len(people)
    visible_people = people[:visible_count]
    _, protected_spans, people_found = _ordered_person_spans(reference, visible_people)
    if _marker_present(reference, protected_spans=protected_spans) != marker_expected:
        return VisibilityResult(
            None,
            "visibility:truncation-marker-mismatch",
            visible_person_count=visible_count,
            warnings=warnings,
        )

    if not people_found:
        return VisibilityResult(
            None,
            "visibility:not-found",
            visible_person_count=visible_count,
            warnings=warnings,
        )
    return VisibilityResult(
        True,
        None,
        visible_person_count=visible_count,
        warnings=warnings,
    )


def build_person_candidate_dag(*, raw_gold: str, style: str, reference: str) -> PersonCandidateDag:
    _, brace_warnings = remove_bibtex_grouping_braces(raw_gold)
    try:
        people = parse_person_list(raw_gold)
    except ValueError as exc:
        warning = PolicyWarning("person-parse-failed", str(exc))
        return PersonCandidateDag((), (), 0, (*brace_warnings, warning))
    visible_count = _visible_person_count(style, len(people))
    layers = tuple(
        _person_layer(person, person_index)
        for person_index, person in enumerate(people[:visible_count])
    )
    marker_layer = [PersonCandidateBranch((visible_count, 0), "absent", "")]
    _, protected_spans, _ = _ordered_person_spans(
        reference,
        people[:visible_count],
    )
    if _marker_present(reference, protected_spans=protected_spans):
        marker_layer.append(PersonCandidateBranch((visible_count, 1), "present", "OTHERS"))
    if not layers or any(not layer for layer in layers):
        return PersonCandidateDag(layers, tuple(marker_layer), 0, brace_warnings)
    return PersonCandidateDag.from_layers(
        layers,
        marker_layer=tuple(marker_layer),
        warnings=brace_warnings,
    )


def _word_and_separator_spans(
    value: str,
    person_spans: tuple[_PersonTextSpan, ...],
    marker_spans: tuple[_PersonTextSpan, ...],
) -> set[_PersonTextSpan]:
    if not person_spans:
        return set()

    gaps = [
        _PersonTextSpan(left.end, right.start) for left, right in itertools.pairwise(person_spans)
    ]
    gaps.extend(
        _PersonTextSpan(person_spans[-1].end, marker.start)
        for marker in marker_spans
        if person_spans[-1].end <= marker.start
    )
    return {
        _PersonTextSpan(match.start(), match.end())
        for match in _WORD_AND_DELIMITER.finditer(value)
        if any(gap.start <= match.start() and match.end() <= gap.end for gap in gaps)
    }


def _strip_markers(
    value: str,
    opaque_person_structures: tuple[str, ...] = (),
    *,
    protected_spans: tuple[_PersonTextSpan, ...] | None = None,
    person_spans: tuple[_PersonTextSpan, ...] = (),
) -> tuple[str, bool]:
    if protected_spans is None:
        protected_spans = _opaque_person_spans(value, opaque_person_structures)
    marker_spans = _marker_spans(value, protected_spans=protected_spans)
    delimiter_spans = _word_and_separator_spans(value, person_spans, marker_spans)
    for span in sorted(
        {*marker_spans, *delimiter_spans}, reverse=True, key=lambda item: item.start
    ):
        value = f"{value[: span.start]}{value[span.end :]}"
    return value, bool(marker_spans)


def normalize_person_prediction(
    value: object,
    *,
    raw_gold: str | None = None,
    style: str | None = None,
) -> TextNormalization:
    has_person_boundaries = isinstance(value, (list, tuple))
    raw_items = value if has_person_boundaries else (value,)
    output: list[str] = []
    warnings: list[PolicyWarning] = []
    marker_present = False
    people: tuple[PersonName, ...] = ()
    opaque_structures: tuple[str, ...] = ()
    if raw_gold:
        try:
            people = parse_person_list(raw_gold)
            if style is not None:
                people = people[: _visible_person_count(style, len(people))]
            opaque_structures = _opaque_person_structures(people)
        except ValueError:
            pass
    ordered_person_spans: tuple[_PersonTextSpan, ...] = ()
    ordered_opaque_spans: tuple[_PersonTextSpan, ...] | None = None
    if not has_person_boundaries and people:
        ordered_person_spans, ordered_opaque_spans, _ = _ordered_person_spans(str(value), people)
    remaining_opaque_structures = list(opaque_structures)
    for raw_item in raw_items:
        if raw_item is None:
            continue
        person_item = str(raw_item)
        _, item_warnings = remove_bibtex_grouping_braces(person_item)
        warnings.extend(item_warnings)
        exact_opaque_element = False
        if has_person_boundaries:
            item_structure = _normalize_person_visibility_structure(person_item).text
            if item_structure in remaining_opaque_structures:
                remaining_opaque_structures.remove(item_structure)
                exact_opaque_element = True
            else:
                standalone_without_marker, standalone_marker = _strip_markers(person_item)
                if (
                    standalone_marker
                    and not _normalize_person_visibility_structure(standalone_without_marker).text
                ):
                    marker_present = True
                    continue
        if exact_opaque_element:
            without_marker, has_marker = person_item, False
        else:
            without_marker, has_marker = _strip_markers(
                person_item,
                opaque_structures,
                protected_spans=ordered_opaque_spans,
                person_spans=ordered_person_spans,
            )
        marker_present = marker_present or has_marker
        structured = _structure_text(without_marker)
        if structured:
            output.append(structured)
    if marker_present:
        output.append("OTHERS")
    return TextNormalization("".join(output), tuple(warnings))


@dataclass(frozen=True)
class _GraphEdge:
    target: int
    character: str | None
    candidate_priority: tuple[int, int] | None
    candidate_id: str | None


@dataclass(frozen=True)
class _DijkstraState:
    cost: int
    candidate_priority_tuple: tuple[tuple[int, int], ...]
    candidate_ids: tuple[str, ...]
    text: str


def _character_graph(dag: PersonCandidateDag) -> tuple[list[list[_GraphEdge]], int]:
    edges: list[list[_GraphEdge]] = [[]]
    boundary = 0
    all_layers = (*dag.layers, dag.marker_layer)
    for layer_index, layer in enumerate(all_layers):
        branch_nodes: list[tuple[PersonCandidateBranch, list[int]]] = []
        for branch in layer:
            internal_nodes: list[int] = []
            for _ in branch.text[:-1]:
                internal_nodes.append(len(edges))
                edges.append([])
            branch_nodes.append((branch, internal_nodes))
        next_boundary = len(edges)
        edges.append([])
        for branch, internal_nodes in branch_nodes:
            candidate_id = (
                f"person:{layer_index}:{branch.mode}"
                if layer_index < len(dag.layers)
                else f"marker:{branch.mode}"
            )
            if not branch.text:
                edges[boundary].append(
                    _GraphEdge(
                        next_boundary,
                        None,
                        branch.candidate_priority,
                        candidate_id,
                    )
                )
                continue
            targets = [*internal_nodes, next_boundary]
            source = boundary
            for character_index, (character, target) in enumerate(
                zip(branch.text, targets, strict=True)
            ):
                edges[source].append(
                    _GraphEdge(
                        target,
                        character,
                        branch.candidate_priority if character_index == 0 else None,
                        candidate_id if character_index == 0 else None,
                    )
                )
                source = target
        boundary = next_boundary
    return edges, boundary


def _extend_state(state: _DijkstraState, edge: _GraphEdge, cost: int) -> _DijkstraState:
    return _DijkstraState(
        cost=state.cost + cost,
        candidate_priority_tuple=(
            (*state.candidate_priority_tuple, edge.candidate_priority)
            if edge.candidate_priority is not None
            else state.candidate_priority_tuple
        ),
        candidate_ids=(
            (*state.candidate_ids, edge.candidate_id)
            if edge.candidate_id is not None
            else state.candidate_ids
        ),
        text=f"{state.text}{edge.character or ''}",
    )


def _relax(
    states: dict[tuple[int, int], _DijkstraState],
    key: tuple[int, int],
    candidate: _DijkstraState,
) -> None:
    current = states.get(key)
    if current is None or (candidate.cost, candidate.candidate_priority_tuple) < (
        current.cost,
        current.candidate_priority_tuple,
    ):
        states[key] = candidate


def select_person_candidate(dag: PersonCandidateDag, prediction: str) -> CandidateSelection:
    if dag.path_count == 0:
        raise ValueError("cannot select from an empty person candidate DAG")
    graph, terminal = _character_graph(dag)
    states: list[dict[tuple[int, int], _DijkstraState]] = [{} for _ in range(len(graph))]
    states[0][(0, 0)] = _DijkstraState(0, (), (), "")
    prediction_length = len(prediction)

    for node, outgoing in enumerate(graph):
        for prediction_index in range(prediction_length + 1):
            candidate_lengths = sorted(
                length for index, length in states[node] if index == prediction_index
            )
            for candidate_length in candidate_lengths:
                state = states[node][(prediction_index, candidate_length)]
                if prediction_index < prediction_length:
                    _relax(
                        states[node],
                        (prediction_index + 1, candidate_length),
                        _DijkstraState(
                            state.cost + 1,
                            state.candidate_priority_tuple,
                            state.candidate_ids,
                            state.text,
                        ),
                    )
                for edge in outgoing:
                    if edge.character is None:
                        _relax(
                            states[edge.target],
                            (prediction_index, candidate_length),
                            _extend_state(state, edge, 0),
                        )
                        continue
                    next_length = candidate_length + 1
                    _relax(
                        states[edge.target],
                        (prediction_index, next_length),
                        _extend_state(state, edge, 1),
                    )
                    if prediction_index < prediction_length:
                        substitution = edge.character != prediction[prediction_index]
                        _relax(
                            states[edge.target],
                            (prediction_index + 1, next_length),
                            _extend_state(state, edge, int(substitution)),
                        )

    terminal_states = [
        (candidate_length, state)
        for (prediction_index, candidate_length), state in states[terminal].items()
        if prediction_index == prediction_length
    ]
    candidate_length, selected = min(
        terminal_states,
        key=lambda item: (
            item[1].cost,
            item[1].cost / max(item[0], prediction_length)
            if max(item[0], prediction_length)
            else 0.0,
            item[1].candidate_priority_tuple,
        ),
    )
    denominator = max(candidate_length, prediction_length)
    normalized = selected.cost / denominator if denominator else 0.0
    return CandidateSelection(
        candidate_id="|".join(selected.candidate_ids),
        selected_gold=selected.text,
        normalized_prediction=prediction,
        distance_raw=selected.cost,
        distance_normalized=normalized,
    )


def _unscored_result(
    *,
    status: FieldStatus,
    scope_status: FieldScopeStatus,
    gold_present: bool,
    observable: bool | None,
    eligibility: FieldEligibility,
    reason: str,
    exact_match: bool = False,
    candidate_path_count: int = 0,
    warnings: tuple[PolicyWarning, ...] = (),
) -> FieldPolicyResult:
    return FieldPolicyResult(
        status=status,
        exact_match=exact_match,
        scope_status=scope_status,
        gold_present=gold_present,
        observable=observable,
        eligibility=eligibility,
        exclusion_reason=reason,
        candidate_path_count=candidate_path_count,
        selected_candidate_id=None,
        selected_gold=None,
        normalized_prediction=None,
        distance_raw=None,
        distance_normalized=None,
        include_completion=False,
        include_available=False,
        warnings=warnings,
    )


def evaluate_field_policy(
    *,
    field: str,
    style: str,
    entry_type: str,
    gold: object,
    raw_person_gold: str | None,
    prediction: object,
    reference: str,
) -> FieldPolicyResult:
    gold_is_present = raw_value_present(gold)
    prediction_is_present = raw_value_present(prediction)
    if field not in observable_field_scope(style, entry_type):
        return _unscored_result(
            status=FieldStatus.NOT_APPLICABLE,
            scope_status=FieldScopeStatus.EXCLUDED,
            gold_present=gold_is_present,
            observable=None,
            eligibility=FieldEligibility.OUT_OF_SCOPE,
            reason="scope:excluded",
        )
    if not gold_is_present:
        return _unscored_result(
            status=FieldStatus.EXTRA if prediction_is_present else FieldStatus.MATCH,
            scope_status=FieldScopeStatus.ALWAYS,
            gold_present=False,
            observable=None,
            eligibility=FieldEligibility.GOLD_ABSENT,
            reason="gold:absent",
            exact_match=not prediction_is_present,
        )

    if field in PERSON_FIELDS:
        visibility = evaluate_person_visibility(
            raw_person_gold or "", style=style, reference=reference
        )
    else:
        visibility = evaluate_scalar_visibility(
            field=field,
            gold=str(gold),
            style=style,
            reference=reference,
        )
    if visibility.observable is not True:
        return _unscored_result(
            status=FieldStatus.NOT_APPLICABLE,
            scope_status=FieldScopeStatus.ALWAYS,
            gold_present=True,
            observable=None,
            eligibility=FieldEligibility.VISIBILITY_UNRESOLVED,
            reason=visibility.reason or "visibility:not-found",
            warnings=visibility.warnings,
        )

    if field in PERSON_FIELDS:
        dag = build_person_candidate_dag(
            raw_gold=raw_person_gold or "", style=style, reference=reference
        )
        candidate_path_count = dag.path_count
        prediction_normalization = (
            normalize_person_prediction(
                prediction,
                raw_gold=raw_person_gold,
                style=style,
            )
            if prediction_is_present
            else TextNormalization("")
        )
        normalized_prediction = prediction_normalization.text
        if candidate_path_count == 0:
            return _unscored_result(
                status=FieldStatus.NOT_APPLICABLE,
                scope_status=FieldScopeStatus.ALWAYS,
                gold_present=True,
                observable=True,
                eligibility=FieldEligibility.CANDIDATE_UNRESOLVED,
                reason="candidate:parse-failed",
                warnings=(*visibility.warnings, *dag.warnings),
            )
        selection = select_person_candidate(dag, normalized_prediction)
        warnings = (
            *visibility.warnings,
            *dag.warnings,
            *prediction_normalization.warnings,
        )
    else:
        built = build_scalar_candidates(
            field=field,
            style=style,
            gold=str(gold),
            reference=reference,
        )
        candidate_path_count = len(built.candidates)
        prediction_normalization = (
            normalize_scalar_prediction(prediction)
            if prediction_is_present
            else TextNormalization("")
        )
        normalized_prediction = prediction_normalization.text
        if not built.candidates:
            return _unscored_result(
                status=FieldStatus.NOT_APPLICABLE,
                scope_status=FieldScopeStatus.ALWAYS,
                gold_present=True,
                observable=True,
                eligibility=FieldEligibility.CANDIDATE_UNRESOLVED,
                reason="candidate:empty",
                warnings=(*visibility.warnings, *built.warnings),
            )
        selection = select_candidates(built.candidates, normalized_prediction)
        warnings = (
            *visibility.warnings,
            *built.warnings,
            *prediction_normalization.warnings,
        )

    prediction_is_available = prediction_is_present and bool(normalized_prediction)
    if not prediction_is_available:
        status = FieldStatus.MISSING
    elif selection.distance_raw == 0:
        status = FieldStatus.MATCH
    else:
        status = FieldStatus.MISMATCH
    return FieldPolicyResult(
        status=status,
        exact_match=status == FieldStatus.MATCH,
        scope_status=FieldScopeStatus.ALWAYS,
        gold_present=True,
        observable=True,
        eligibility=FieldEligibility.ELIGIBLE,
        exclusion_reason=None,
        candidate_path_count=candidate_path_count,
        selected_candidate_id=selection.candidate_id,
        selected_gold=selection.selected_gold,
        normalized_prediction=normalized_prediction,
        distance_raw=selection.distance_raw,
        distance_normalized=selection.distance_normalized,
        include_completion=True,
        include_available=prediction_is_available,
        warnings=warnings,
    )
