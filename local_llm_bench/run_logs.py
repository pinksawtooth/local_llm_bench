from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from .config import OutputSettings
from .error_utils import annotate_error_info, merge_excerpts

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _relative_to_history(output: OutputSettings, target: Path) -> str:
    return os.path.relpath(target, output.history_json.parent).replace(os.sep, "/")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sanitize_name(value: str) -> str:
    normalized = _SAFE_NAME_RE.sub("-", value.strip())
    normalized = normalized.strip("-.")
    return normalized or "item"


def _attempt_filename(phase: str, iteration: Any) -> str:
    return f"{_sanitize_name(str(phase))}-{int(iteration)}.json"


def _question_filename(phase: str, iteration: Any, question_id: str) -> str:
    return f"{_sanitize_name(str(phase))}-{int(iteration)}-{_sanitize_name(question_id)}.json"


def _question_results(record: Dict[str, Any]) -> list[Dict[str, Any]]:
    raw = record.get("question_results")
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _normalize_tool_name_counts(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, int] = {}
    for key, raw_count in value.items():
        name = str(key or "").strip()
        if not name:
            continue
        if isinstance(raw_count, bool):
            count = int(raw_count)
        elif isinstance(raw_count, (int, float)):
            count = int(raw_count)
        else:
            continue
        if count <= 0:
            continue
        normalized[name] = normalized.get(name, 0) + count
    return normalized


def _merge_tool_name_counts(*values: Any) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for value in values:
        for name, count in _normalize_tool_name_counts(value).items():
            merged[name] = merged.get(name, 0) + count
    return merged


def _tool_metrics_from_trace(trace: Any) -> tuple[int | None, Dict[str, int]]:
    if not isinstance(trace, dict):
        return None, {}
    total = 0
    name_counts: Dict[str, int] = {}
    raw_turns = trace.get("turns")
    if not isinstance(raw_turns, list):
        return 0, {}
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        raw_events = turn.get("tool_events")
        if not isinstance(raw_events, list):
            continue
        for event in raw_events:
            if not isinstance(event, dict):
                continue
            total += 1
            tool_name = str(event.get("tool_name") or "").strip()
            if tool_name:
                name_counts[tool_name] = name_counts.get(tool_name, 0) + 1
    return total, name_counts


def _tool_metrics_from_question_payload(question_payload: Dict[str, Any]) -> tuple[int | None, Dict[str, int]]:
    parsed_worker_result = question_payload.get("parsed_worker_result")
    if isinstance(parsed_worker_result, dict):
        count, name_counts = _tool_metrics_from_trace(parsed_worker_result.get("trace"))
        if count is not None:
            return count, name_counts
    return _tool_metrics_from_trace(question_payload.get("trace"))


def _apply_tool_metrics_from_question_payload(question_result: Dict[str, Any], question_payload: Dict[str, Any]) -> None:
    tool_call_count, tool_name_counts = _tool_metrics_from_question_payload(question_payload)
    if isinstance(tool_call_count, int):
        question_result["tool_call_count"] = tool_call_count
        question_result["tool_name_counts"] = tool_name_counts


def _apply_record_tool_metrics(record: Dict[str, Any]) -> None:
    question_results = _question_results(record)
    if not question_results:
        record.setdefault("tool_call_count", 0)
        record.setdefault("tool_name_counts", {})
        return

    total: int | None = None
    merged_name_counts: Dict[str, int] = {}
    for question_result in question_results:
        raw_count = question_result.get("tool_call_count")
        if isinstance(raw_count, bool):
            question_count = int(raw_count)
        elif isinstance(raw_count, (int, float)):
            question_count = int(raw_count)
        else:
            question_count = None
        if question_count is not None:
            total = (total or 0) + question_count
        merged_name_counts = _merge_tool_name_counts(merged_name_counts, question_result.get("tool_name_counts"))

    if total is not None:
        record["tool_call_count"] = total
        record["tool_name_counts"] = merged_name_counts


def persist_run_logs(output: OutputSettings, run_data: Dict[str, Any]) -> Dict[str, Any]:
    bundle = run_data.pop("_log_bundle", None)
    records = [item for item in run_data.get("records", []) if isinstance(item, dict)]

    for record in records:
        annotate_error_info(record)
        for question_result in _question_results(record):
            annotate_error_info(question_result)
        _apply_record_tool_metrics(record)

    if not isinstance(bundle, dict):
        return run_data

    run_id = str(run_data.get("run_id") or "unknown")
    run_dir = output.run_logs_dir / run_id
    attempts_dir = run_dir / "attempts"
    questions_dir = run_dir / "questions"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)

    console_lines = [str(item) for item in bundle.get("console_lines", []) if str(item).strip()]
    console_path = run_dir / "console.log"
    _ensure_parent(console_path)
    console_path.write_text("\n".join(console_lines) + ("\n" if console_lines else ""), encoding="utf-8")

    manifest_attempts: list[Dict[str, Any]] = []
    for attempt in bundle.get("attempts", []):
        if not isinstance(attempt, dict):
            continue
        record_index = attempt.get("record_index")
        if not isinstance(record_index, int) or not (0 <= record_index < len(records)):
            continue
        record = records[record_index]
        phase = str(record.get("phase") or attempt.get("phase") or "phase")
        iteration = int(record.get("iteration") or attempt.get("iteration") or 0)

        question_entries: list[Dict[str, Any]] = []
        question_logs = attempt.get("question_logs")
        if isinstance(question_logs, list):
            for question_log in question_logs:
                if not isinstance(question_log, dict):
                    continue
                question_index = question_log.get("question_index")
                question_results = _question_results(record)
                if not isinstance(question_index, int) or not (0 <= question_index < len(question_results)):
                    continue
                question_result = question_results[question_index]
                question_id = str(question_result.get("question_id") or question_log.get("question_id") or f"q{question_index + 1}")
                question_path = questions_dir / _question_filename(phase, iteration, question_id)
                question_payload = dict(question_log.get("payload") or {})
                _write_json(question_path, question_payload)
                question_rel = _relative_to_history(output, question_path)
                question_result["log_path"] = question_rel
                _apply_tool_metrics_from_question_payload(question_result, question_payload)
                annotate_error_info(
                    question_result,
                    stderr_text=question_payload.get("stderr"),
                )
                question_entries.append(
                    {
                        "question_id": question_id,
                        "status": question_result.get("status"),
                        "log_path": question_rel,
                        "tool_call_count": question_result.get("tool_call_count"),
                    }
                )

        attempt_payload = dict(attempt.get("payload") or {})
        attempt_payload["question_logs"] = question_entries
        attempt_path = attempts_dir / _attempt_filename(phase, iteration)
        _write_json(attempt_path, attempt_payload)
        attempt_rel = _relative_to_history(output, attempt_path)
        record["log_path"] = attempt_rel
        _apply_record_tool_metrics(record)
        stderr_excerpt = attempt_payload.get("stderr")
        if not stderr_excerpt and question_entries:
            stderr_excerpt = merge_excerpts(
                [result.get("stderr_excerpt") for result in _question_results(record)],
            )
        annotate_error_info(record, stderr_text=stderr_excerpt)
        manifest_attempts.append(
            {
                "phase": phase,
                "iteration": iteration,
                "status": record.get("status"),
                "log_path": attempt_rel,
                "tool_call_count": record.get("tool_call_count"),
                "question_logs": question_entries,
            }
        )

    manifest = {
        "version": 1,
        "run_id": run_id,
        "model": run_data.get("model"),
        "api_model": run_data.get("api_model"),
        "benchmark_mode": run_data.get("benchmark_mode"),
        "benchmark_id": run_data.get("benchmark_id"),
        "benchmark_title": run_data.get("benchmark_title"),
        "started_at": run_data.get("started_at"),
        "ended_at": run_data.get("ended_at"),
        "duration_sec": run_data.get("duration_sec"),
        "console_log": _relative_to_history(output, console_path),
        "attempts": manifest_attempts,
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    run_data["log_dir"] = _relative_to_history(output, run_dir)
    run_data["log_manifest_path"] = _relative_to_history(output, manifest_path)
    return run_data
