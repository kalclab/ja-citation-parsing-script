"""Experiment runner: orchestrate input loading, parsing, and artifact writing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.artifacts import ArtifactWriter
from scripts.config import RunConfig, config_to_manifest
from scripts.dataset import batch_items, load_inputs
from scripts.grobid import GrobidClient, GrobidClientConfig
from scripts.models import (
    CallLog,
    InputItem,
    ItemLog,
    NativeParseResult,
    ParseIssue,
    ParserKind,
)

MAX_ATTEMPTS = 20


def _hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes())
    return h.hexdigest()


def _order_hash(items: list[InputItem]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.sample_id.encode())
        h.update(b"\n")
    return h.hexdigest()


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _make_llm_client(config: RunConfig):
    from scripts.llm import LLMClient

    return LLMClient(
        model=config.model or "",
        temperature=config.temperature or 0.0,
        seed=config.seed,
    )


def _issue_list(issues: list[ParseIssue]) -> list[dict[str, Any]]:
    return [
        {
            "kind": i.kind,
            "opaque_input_id": i.opaque_input_id,
            "detail": i.detail,
        }
        for i in issues
    ]


def _issues_stage_kind(issues: list[ParseIssue]) -> tuple[str, str]:
    kinds = {i.kind for i in issues}
    if "parse_error" in kinds:
        return "response_parse", "bibtex_parse_error"
    if "missing_id" in kinds:
        return "response_validation", "missing_id"
    if "unknown_id" in kinds:
        return "response_validation", "unknown_id"
    if "duplicate_id" in kinds:
        return "response_validation", "duplicate_id"
    return "response_validation", "unknown"


def _grobid_error_info(grobid_error: str) -> tuple[str, str]:
    """Map grobid error kind to (error_stage, parse_status)."""
    transport_errors = {"transport_error", "http_error"}
    if grobid_error in transport_errors:
        return "transport", "call_error"
    return "response_parse", grobid_error


def _run_llm_experiment(config: RunConfig, writer: ArtifactWriter) -> None:
    from scripts.llm import build_bibtex_prompt, parse_bibtex_str

    client = _make_llm_client(config)

    items = load_inputs(
        config.dataset_path,
        config.sample_path,
        config.styles,
        order_mode=config.order_mode,
        shuffle_seed=config.shuffle_seed,
    )
    batch_size = config.batch_size or 1
    batches = batch_items(items, batch_size)

    call_idx = 0
    for batch_idx, batch in enumerate(batches):
        attempt = 0
        success = False
        last_stage: str | None = None
        last_kind: str | None = None

        while attempt < MAX_ATTEMPTS:
            attempt_start = time.monotonic()
            call_id = f"call-{call_idx:04d}"
            call_idx += 1

            system, user = build_bibtex_prompt(batch)

            try:
                resp = client.complete(system, user)
                content = resp["content"]
                results, issues = parse_bibtex_str(content, batch)
                call_error = None
            except Exception as exc:
                results = {}
                issues = []
                content = ""
                resp = {
                    "content": "",
                    "model": client.model,
                    "usage": {},
                    "finish_reason": "",
                }
                call_error = f"{type(exc).__name__}: {exc}"

            latency_ms = int((time.monotonic() - attempt_start) * 1000)

            request_path = writer.write_raw(
                call_id,
                "request",
                json.dumps(
                    {"system": system, "user": user},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            response_path = writer.write_raw(call_id, "response", content)

            if call_error is not None:
                last_stage = "transport"
                last_kind = call_error
                call = CallLog(
                    run_id=config.run_id,
                    call_id=call_id,
                    style=batch[0].style,
                    batch_index=batch_idx,
                    attempt_index=attempt,
                    expected_input_ids=[item.opaque_input_id for item in batch],
                    request_path=str(request_path.relative_to(writer.output_dir)),
                    response_path=str(response_path.relative_to(writer.output_dir)),
                    status="error",
                    error_stage=last_stage,
                    error_kind=last_kind,
                    latency_ms=latency_ms,
                    token_usage=resp.get("usage") if isinstance(resp.get("usage"), dict) else None,
                    finish_reason=resp.get("finish_reason"),
                )
                writer.write_call(call)
                attempt += 1
                continue

            if not issues:
                call = CallLog(
                    run_id=config.run_id,
                    call_id=call_id,
                    style=batch[0].style,
                    batch_index=batch_idx,
                    attempt_index=attempt,
                    expected_input_ids=[item.opaque_input_id for item in batch],
                    request_path=str(request_path.relative_to(writer.output_dir)),
                    response_path=str(response_path.relative_to(writer.output_dir)),
                    status="ok",
                    latency_ms=latency_ms,
                    token_usage=resp.get("usage") if isinstance(resp.get("usage"), dict) else None,
                    finish_reason=resp.get("finish_reason"),
                )
                writer.write_call(call)
                for pos, item in enumerate(batch):
                    oid = item.opaque_input_id
                    parsed = results[oid]
                    item_log = ItemLog(
                        run_id=config.run_id,
                        sample_id=item.sample_id,
                        key=item.key,
                        style=item.style,
                        opaque_input_id=oid,
                        batch_index=batch_idx,
                        batch_position=pos,
                        reference=item.reference,
                        entry_type=item.entry_type,
                        discipline=item.discipline,
                        difficulty_flags=item.difficulty_flags,
                        call_id=call_id,
                        attempt_count=attempt + 1,
                        parse_status="ok",
                        parsed_output=parsed,
                    )
                    writer.write_item(item_log)
                success = True
                break

            last_stage, last_kind = _issues_stage_kind(issues)
            error_detail = "; ".join(f"{i.kind}: {i.detail}" for i in issues)
            call = CallLog(
                run_id=config.run_id,
                call_id=call_id,
                style=batch[0].style,
                batch_index=batch_idx,
                attempt_index=attempt,
                expected_input_ids=[item.opaque_input_id for item in batch],
                request_path=str(request_path.relative_to(writer.output_dir)),
                response_path=str(response_path.relative_to(writer.output_dir)),
                status="error",
                error_stage=last_stage,
                error_kind=last_kind,
                error_detail=error_detail,
                issues=_issue_list(issues),
                latency_ms=latency_ms,
                token_usage=resp.get("usage") if isinstance(resp.get("usage"), dict) else None,
                finish_reason=resp.get("finish_reason"),
            )
            writer.write_call(call)
            attempt += 1

        if not success:
            for pos, item in enumerate(batch):
                item_log = ItemLog(
                    run_id=config.run_id,
                    sample_id=item.sample_id,
                    key=item.key,
                    style=item.style,
                    opaque_input_id=item.opaque_input_id,
                    batch_index=batch_idx,
                    batch_position=pos,
                    reference=item.reference,
                    entry_type=item.entry_type,
                    discipline=item.discipline,
                    difficulty_flags=item.difficulty_flags,
                    call_id=None,
                    attempt_count=MAX_ATTEMPTS,
                    parse_status="retry_exhausted",
                    parsed_output=None,
                    error_stage=last_stage,
                    error_kind=last_kind,
                    error_detail="max attempts exceeded",
                )
                writer.write_item(item_log)


def _run_grobid_experiment(config: RunConfig, writer: ArtifactWriter) -> str | None:
    grobid_config = GrobidClientConfig(
        base_url=config.grobid_base_url,
        consolidate_citations=config.grobid_consolidate,
    )
    client = GrobidClient(grobid_config)

    if not client.check_health():
        raise RuntimeError(f"GROBID is not healthy at {config.grobid_base_url}")

    grobid_version = client.get_version()
    if grobid_version:
        print(f"[experiment] GROBID version: {grobid_version}", file=sys.stderr, flush=True)

    items = load_inputs(
        config.dataset_path,
        config.sample_path,
        config.styles,
        order_mode=config.order_mode,
        shuffle_seed=config.shuffle_seed,
    )

    call_idx = 0
    for batch_idx, item in enumerate(items):
        start_time = time.monotonic()
        call_id = f"call-{call_idx:04d}"
        call_idx += 1

        try:
            result, raw, grobid_error = client.parse_bibtex(item.reference)
        except Exception:
            result = NativeParseResult()
            raw = ""
            grobid_error = "transport_error"

        latency_ms = int((time.monotonic() - start_time) * 1000)

        request_path: str | None = None
        response_path: str | None = None
        if raw:
            writer.write_raw(call_id, "request", item.reference)
            request_path = f"raw/{call_id}/request"
            writer.write_raw(call_id, "response", raw)
            response_path = f"raw/{call_id}/response"

        if grobid_error is None:
            call = CallLog(
                run_id=config.run_id,
                call_id=call_id,
                style=item.style,
                batch_index=batch_idx,
                attempt_index=0,
                expected_input_ids=[item.opaque_input_id],
                request_path=request_path,
                response_path=response_path,
                status="ok",
                latency_ms=latency_ms,
            )
            writer.write_call(call)
            item_log = ItemLog(
                run_id=config.run_id,
                sample_id=item.sample_id,
                key=item.key,
                style=item.style,
                opaque_input_id=item.opaque_input_id,
                batch_index=batch_idx,
                batch_position=0,
                reference=item.reference,
                entry_type=item.entry_type,
                discipline=item.discipline,
                difficulty_flags=item.difficulty_flags,
                call_id=call_id,
                attempt_count=1,
                parse_status="ok",
                parsed_output=result,
            )
        else:
            error_stage, parse_status = _grobid_error_info(grobid_error)
            call = CallLog(
                run_id=config.run_id,
                call_id=call_id,
                style=item.style,
                batch_index=batch_idx,
                attempt_index=0,
                expected_input_ids=[item.opaque_input_id],
                request_path=request_path,
                response_path=response_path,
                status="error",
                error_stage=error_stage,
                error_kind=grobid_error,
                latency_ms=latency_ms,
            )
            writer.write_call(call)
            item_log = ItemLog(
                run_id=config.run_id,
                sample_id=item.sample_id,
                key=item.key,
                style=item.style,
                opaque_input_id=item.opaque_input_id,
                batch_index=batch_idx,
                batch_position=0,
                reference=item.reference,
                entry_type=item.entry_type,
                discipline=item.discipline,
                difficulty_flags=item.difficulty_flags,
                call_id=call_id,
                attempt_count=1,
                parse_status=parse_status,
                parsed_output=None,
                error_stage=error_stage,
                error_kind=grobid_error,
            )
        writer.write_item(item_log)

    client.close()
    return grobid_version


def run_experiment(
    config: RunConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> bool:
    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"[experiment] {msg}", file=sys.stderr, flush=True)

    try:
        writer = ArtifactWriter(config.run_id, config.output_dir)
    except ValueError as exc:
        raise exc

    start_time = datetime.now(UTC)

    try:
        dataset_files = [
            Path(config.dataset_path) / "ref_strings.jsonl",
            Path(config.dataset_path) / "metadata.csv",
            Path(config.sample_path),
        ]
        dataset_hash = _hash_files(dataset_files)
        sample_hash = ArtifactWriter.file_hash(Path(config.sample_path))
    except Exception:
        dataset_hash = "unavailable"
        sample_hash = "unavailable"

    items = load_inputs(
        config.dataset_path,
        config.sample_path,
        config.styles,
        order_mode=config.order_mode,
        shuffle_seed=config.shuffle_seed,
    )

    manifest = config_to_manifest(
        config,
        code_commit=_git_revision(),
        order_hash=_order_hash(items),
        item_count=len(items),
        dataset_hash=dataset_hash,
        sample_hash=sample_hash,
        start_time=start_time.isoformat(),
        end_time=None,
        status="running",
    )

    if config.parser == ParserKind.GROBID:
        _log(f"GROBID start: {config.run_id}")
        try:
            grobid_version = _run_grobid_experiment(config, writer)
            manifest["status"] = "completed"
            manifest["grobid_version"] = grobid_version
        except Exception as exc:
            _log(f"GROBID failed: {exc}")
            manifest["status"] = "failed"
    else:
        _log(f"LLM start: {config.run_id} model={config.model} batch={config.batch_size}")
        try:
            _run_llm_experiment(config, writer)
            manifest["status"] = "completed"
        except Exception as exc:
            _log(f"LLM failed: {exc}")
            manifest["status"] = "failed"

    end_time = datetime.now(UTC)
    manifest["end_time"] = end_time.isoformat()
    writer.write_manifest(manifest)
    _log(f"done: {config.run_id} status={manifest['status']}")
    return manifest["status"] == "completed"
