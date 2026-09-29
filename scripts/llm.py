"""LLM client: stateless BibTeX completion."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from scripts.models import InputItem, NativeParseResult, ParseIssue


def build_bibtex_prompt(items: list[InputItem]) -> tuple[str, str]:
    from scripts.prompts import build_bibtex_prompt as _build

    return _build(items)


def _entry_to_bibtex(entry: dict[str, Any]) -> str:
    entry_type = entry.get("ENTRYTYPE", "misc")
    key = entry.get("ID", "")
    lines = [f"@{entry_type}{{{key},"]
    for field, value in entry.items():
        if field.lower() not in ("entrytype", "id"):
            lines.append(f"  {field} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


def parse_bibtex_str(
    response: str,
    expected_items: list[InputItem],
) -> tuple[dict[str, NativeParseResult], list[ParseIssue]]:
    results: dict[str, NativeParseResult] = {}
    issues: list[ParseIssue] = []

    try:
        import bibtexparser
    except ImportError:
        for item in expected_items:
            issues.append(
                ParseIssue(
                    kind="parse_error",
                    opaque_input_id=item.opaque_input_id,
                    detail="bibtexparser not available",
                )
            )
        return results, issues

    try:
        library = bibtexparser.loads(response)
    except Exception as e:
        for item in expected_items:
            issues.append(
                ParseIssue(
                    kind="parse_error",
                    opaque_input_id=item.opaque_input_id,
                    detail=f"BibTeX syntax error: {e}",
                )
            )
        return results, issues

    expected_ids = {item.opaque_input_id for item in expected_items}
    found_ids: set[str] = set()

    for entry in library.entries:
        key = entry.get("ID", "")
        if key in found_ids:
            if key in expected_ids:
                issues.append(
                    ParseIssue(
                        kind="duplicate_id",
                        opaque_input_id=key,
                        detail=f"duplicate BibTeX ID: {key}",
                    )
                )
            else:
                issues.append(
                    ParseIssue(
                        kind="unknown_id",
                        detail=f"unexpected BibTeX ID: {key}",
                    )
                )
            continue
        found_ids.add(key)

        if key in expected_ids:
            from scripts.grobid import parse_bibtex_response

            entry_str = _entry_to_bibtex(entry)
            result, _, _ = parse_bibtex_response(entry_str, key)
            results[key] = result
        else:
            issues.append(
                ParseIssue(
                    kind="unknown_id",
                    detail=f"unexpected BibTeX ID: {key}",
                )
            )

    for oid in expected_ids - found_ids:
        issues.append(
            ParseIssue(
                kind="missing_id",
                opaque_input_id=oid,
                detail=f"missing expected BibTeX ID: {oid}",
            )
        )

    return results, issues


class LLMClient:
    """Wrapper around OpenAI-compatible client for stateless BibTeX completion."""

    def __init__(
        self,
        model: str = "",
        *,
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.0,
        seed: int | None = None,
        # max_tokens: int = 8192,
        max_tokens: int = 32768,
        _client: Any = None,
    ) -> None:
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

        if _client is not None:
            self._client = _client
            self.model = model or getattr(_client, "model", "")
        else:
            self.model = model
            kwargs: dict[str, Any] = {"max_retries": 0}
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if key:
                kwargs["api_key"] = key
            url = base_url or os.environ.get("LLM_BASE_URL", "")
            if url:
                kwargs["base_url"] = url
            self._client = OpenAI(**kwargs)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Send a stateless completion request.

        Returns a dict with keys: content, model, usage, finish_reason.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        finish = response.choices[0].finish_reason or ""

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "content": content,
            "model": response.model,
            "usage": usage,
            "finish_reason": finish,
        }
