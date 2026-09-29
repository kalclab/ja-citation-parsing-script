"""BibTeX prompt template for bibliographic reference extraction."""

from __future__ import annotations

from scripts.models import InputItem

BIBTEX_SYSTEM = """\
You are a bibliographic reference parser. Convert the provided reference strings \
into valid BibTeX entries.

For each reference, create a single valid BibTeX entry. Return ONLY the BibTeX \
entries, separated by blank lines.

Rules:
- Use ONLY values that appear in the reference string. Do not invent or guess.
- Use the reference's input_id as the BibTeX citation key.
- Infer the entry type (article, inproceedings, book, incollection, misc) from \
the reference content.
- For personal authors, use "family, given" format separated by " and " \
(e.g., "Yamada, Taro and Suzuki, Hanako"). Preserve the detail level \
(full names or initials) as found in the reference.
- For institutional authors (organizations, committees, agencies), keep the \
full name as one author element \
(e.g., author = {World Health Organization}).

- Do not add any explanatory text outside the BibTeX entries."""

BIBTEX_USER_TEMPLATE = """\
Convert each reference string to a BibTeX entry. Use the input_id as the \
citation key.

## References

{references}"""


# An earlier version of the prompt also asked the model to protect institutional
# authors with double braces (author = {{World Health Organization}}). The rule
# was dropped because the gold records do not brace institutional names either.


def _reference_block(items: list[InputItem]) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(f"[{item.opaque_input_id}] {item.reference}")
    return "\n".join(lines)


def build_bibtex_prompt(items: list[InputItem]) -> tuple[str, str]:
    """Build system and user prompts for BibTeX extraction."""
    user = BIBTEX_USER_TEMPLATE.format(references=_reference_block(items))
    return BIBTEX_SYSTEM, user
