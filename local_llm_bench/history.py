from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .error_utils import annotate_error_info


PROMPT_TOKEN_FIELDS = (
    "prompt_tokens",
    "initial_prompt_tokens",
    "conversation_prompt_tokens",
)
PROMPT_PEER_KEY_SEPARATOR = "::prompt::"

PROMPT_TOKEN_FALLBACK_FIELDS = {
    "prompt_tokens": ("prompt_tokens", "initial_prompt_tokens", "conversation_prompt_tokens"),
    "initial_prompt_tokens": ("initial_prompt_tokens", "prompt_tokens", "conversation_prompt_tokens"),
    "conversation_prompt_tokens": ("conversation_prompt_tokens", "prompt_tokens", "initial_prompt_tokens"),
}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _first_text(values: list[Any]) -> str:
    for value in values:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return ""


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


def _tool_call_count(value: Any) -> int | None:
    if isinstance(value, bool):
        count = int(value)
    elif isinstance(value, (int, float)):
        count = int(value)
    else:
        return None
    return count if count >= 0 else None


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _compute_tps(token_count: float | None, latency_ms: float | None) -> float | None:
    if token_count is None or token_count <= 0 or latency_ms is None or latency_ms <= 0:
        return None
    return float(token_count) / (float(latency_ms) / 1000.0)


def _positive_token_count(value: Any) -> int | None:
    numeric = _numeric_value(value)
    if numeric is None:
        return None
    count = int(numeric)
    return count if count > 0 else None


def _median_int(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(value for value in values if value > 0)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return int(round((ordered[middle - 1] + ordered[middle]) / 2.0))


def _prompt_tokens_from_totals(payload: Dict[str, Any]) -> int | None:
    total_tokens = _numeric_value(payload.get("total_tokens"))
    completion_tokens = _numeric_value(payload.get("completion_tokens"))
    if total_tokens is None or completion_tokens is None or total_tokens < completion_tokens:
        return None
    prompt_tokens = int(total_tokens - completion_tokens)
    return prompt_tokens if prompt_tokens > 0 else None


def _prompt_metrics_from_trace(trace: Any) -> Dict[str, float | int]:
    if not isinstance(trace, dict):
        return {}
    raw_turns = trace.get("turns")
    if not isinstance(raw_turns, list):
        return {}

    prompt_tokens_total = 0
    prompt_latency_ms_total = 0.0
    has_prompt_tokens = False
    has_prompt_latency = False
    initial_prompt_tokens: int | None = None
    initial_prompt_latency_ms: float | None = None
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        usage = turn.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = _numeric_value(usage.get("prompt_tokens"))
            if prompt_tokens is not None and prompt_tokens > 0:
                prompt_tokens_total += int(prompt_tokens)
                has_prompt_tokens = True
                if initial_prompt_tokens is None:
                    initial_prompt_tokens = int(prompt_tokens)
        request_latency_ms = _numeric_value(turn.get("request_latency_ms"))
        if request_latency_ms is not None and request_latency_ms > 0:
            prompt_latency_ms_total += request_latency_ms
            has_prompt_latency = True
            if initial_prompt_latency_ms is None:
                initial_prompt_latency_ms = request_latency_ms

    metrics: Dict[str, float | int] = {}
    if has_prompt_tokens:
        metrics["prompt_tokens"] = prompt_tokens_total
    if has_prompt_latency:
        metrics["prompt_latency_ms"] = prompt_latency_ms_total
    if has_prompt_tokens and has_prompt_latency and prompt_latency_ms_total > 0:
        metrics["approx_prompt_tps"] = prompt_tokens_total / (prompt_latency_ms_total / 1000.0)
    if initial_prompt_tokens is not None:
        metrics["initial_prompt_tokens"] = initial_prompt_tokens
    if initial_prompt_latency_ms is not None and initial_prompt_latency_ms > 0:
        metrics["initial_prompt_latency_ms"] = initial_prompt_latency_ms
    initial_prompt_tps = _compute_tps(initial_prompt_tokens, initial_prompt_latency_ms)
    if initial_prompt_tps is not None:
        metrics["initial_prompt_tps"] = initial_prompt_tps
    if has_prompt_tokens:
        metrics["conversation_prompt_tokens"] = prompt_tokens_total
    if has_prompt_latency:
        metrics["conversation_prompt_latency_ms"] = prompt_latency_ms_total
    conversation_prompt_tps = _compute_tps(
        float(prompt_tokens_total) if has_prompt_tokens else None,
        prompt_latency_ms_total if has_prompt_latency else None,
    )
    if conversation_prompt_tps is not None:
        metrics["conversation_prompt_tps"] = conversation_prompt_tps
    return metrics


def _prompt_metrics_from_entry_payload(payload: Any) -> Dict[str, float | int]:
    if not isinstance(payload, dict):
        return {}

    metrics: Dict[str, float | int] = {}
    benchmark_mode = str(payload.get("benchmark_mode") or "").strip().lower()
    ttft_ms = _numeric_value(payload.get("ttft_ms"))
    total_latency_ms = _numeric_value(payload.get("total_latency_ms"))

    for field in (
        "prompt_tokens",
        "prompt_latency_ms",
        "approx_prompt_tps",
        "initial_prompt_tokens",
        "initial_prompt_latency_ms",
        "initial_prompt_tps",
        "conversation_prompt_tokens",
        "conversation_prompt_latency_ms",
        "conversation_prompt_tps",
    ):
        value = _numeric_value(payload.get(field))
        if value is None:
            continue
        metrics[field] = int(value) if field.endswith("_tokens") or field == "prompt_tokens" else float(value)

    if "prompt_tokens" not in metrics:
        derived_prompt_tokens = _prompt_tokens_from_totals(payload)
        if derived_prompt_tokens is not None:
            metrics["prompt_tokens"] = derived_prompt_tokens

    legacy_prompt_tokens = _numeric_value(metrics.get("prompt_tokens"))
    legacy_prompt_latency_ms = _numeric_value(metrics.get("prompt_latency_ms"))
    legacy_prompt_tps = _numeric_value(metrics.get("approx_prompt_tps"))

    initial_prompt_tokens = _numeric_value(metrics.get("initial_prompt_tokens"))
    initial_prompt_latency_ms = _numeric_value(metrics.get("initial_prompt_latency_ms"))
    if initial_prompt_tokens is None and benchmark_mode != "docker_task" and legacy_prompt_tokens is not None:
        initial_prompt_tokens = legacy_prompt_tokens
    if initial_prompt_latency_ms is None and benchmark_mode != "docker_task" and ttft_ms is not None and ttft_ms > 0:
        initial_prompt_latency_ms = ttft_ms
    initial_prompt_tps = _numeric_value(metrics.get("initial_prompt_tps"))
    if initial_prompt_tps is None:
        initial_prompt_tps = _compute_tps(initial_prompt_tokens, initial_prompt_latency_ms)
    if initial_prompt_tps is None and benchmark_mode != "docker_task" and legacy_prompt_tps is not None:
        initial_prompt_tps = legacy_prompt_tps

    conversation_prompt_tokens = _numeric_value(metrics.get("conversation_prompt_tokens"))
    conversation_prompt_latency_ms = _numeric_value(metrics.get("conversation_prompt_latency_ms"))
    if conversation_prompt_tokens is None:
        if benchmark_mode == "docker_task" and legacy_prompt_tokens is not None:
            conversation_prompt_tokens = legacy_prompt_tokens
        elif benchmark_mode != "docker_task":
            conversation_prompt_tokens = legacy_prompt_tokens if legacy_prompt_tokens is not None else initial_prompt_tokens
    if conversation_prompt_latency_ms is None:
        if benchmark_mode == "docker_task" and legacy_prompt_latency_ms is not None and legacy_prompt_latency_ms > 0:
            conversation_prompt_latency_ms = legacy_prompt_latency_ms
        elif benchmark_mode != "docker_task":
            if total_latency_ms is not None and total_latency_ms > 0:
                conversation_prompt_latency_ms = total_latency_ms
            elif legacy_prompt_latency_ms is not None and legacy_prompt_latency_ms > 0:
                conversation_prompt_latency_ms = legacy_prompt_latency_ms
    conversation_prompt_tps = _numeric_value(metrics.get("conversation_prompt_tps"))
    if conversation_prompt_tps is None:
        conversation_prompt_tps = _compute_tps(conversation_prompt_tokens, conversation_prompt_latency_ms)
    if conversation_prompt_tps is None and benchmark_mode == "docker_task" and legacy_prompt_tps is not None:
        conversation_prompt_tps = legacy_prompt_tps

    if initial_prompt_tokens is not None:
        metrics["initial_prompt_tokens"] = int(initial_prompt_tokens)
    if initial_prompt_latency_ms is not None and initial_prompt_latency_ms > 0:
        metrics["initial_prompt_latency_ms"] = float(initial_prompt_latency_ms)
    if initial_prompt_tps is not None:
        metrics["initial_prompt_tps"] = float(initial_prompt_tps)
    if conversation_prompt_tokens is not None:
        metrics["conversation_prompt_tokens"] = int(conversation_prompt_tokens)
    if conversation_prompt_latency_ms is not None and conversation_prompt_latency_ms > 0:
        metrics["conversation_prompt_latency_ms"] = float(conversation_prompt_latency_ms)
    if conversation_prompt_tps is not None:
        metrics["conversation_prompt_tps"] = float(conversation_prompt_tps)
    return metrics


def _prompt_metrics_from_log_payload(payload: Any) -> Dict[str, float | int]:
    if not isinstance(payload, dict):
        return {}
    parsed_worker_result = payload.get("parsed_worker_result")
    if isinstance(parsed_worker_result, dict):
        metrics = _prompt_metrics_from_entry_payload(parsed_worker_result)
        trace_metrics = _prompt_metrics_from_trace(parsed_worker_result.get("trace"))
        if trace_metrics:
            metrics.update(trace_metrics)
        if metrics:
            return metrics
    response_payload = payload.get("response")
    if isinstance(response_payload, dict):
        metrics = _prompt_metrics_from_entry_payload(response_payload)
        if metrics:
            return metrics
    metrics = _prompt_metrics_from_entry_payload(payload)
    if metrics:
        return metrics
    return _prompt_metrics_from_trace(payload.get("trace"))


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


def _tool_metrics_from_log_payload(payload: Any) -> tuple[int | None, Dict[str, int]]:
    if not isinstance(payload, dict):
        return None, {}
    parsed_worker_result = payload.get("parsed_worker_result")
    if isinstance(parsed_worker_result, dict):
        count, name_counts = _tool_metrics_from_trace(parsed_worker_result.get("trace"))
        if count is not None:
            return count, name_counts
    return _tool_metrics_from_trace(payload.get("trace"))


def _merge_tool_name_counts(*values: Any) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for value in values:
        for name, count in _normalize_tool_name_counts(value).items():
            merged[name] = merged.get(name, 0) + count
    return merged


def _tool_metrics_from_question_log_entries(question_logs: Any, history_dir: Path | None) -> tuple[int | None, Dict[str, int]]:
    if not isinstance(question_logs, list):
        return None, {}

    total = 0
    has_total = False
    merged_name_counts: Dict[str, int] = {}
    for question_log in question_logs:
        if not isinstance(question_log, dict):
            continue
        count = _tool_call_count(question_log.get("tool_call_count"))
        name_counts = _normalize_tool_name_counts(question_log.get("tool_name_counts"))

        if history_dir is not None and (count is None or not name_counts):
            nested_log_path = question_log.get("log_path")
            if isinstance(nested_log_path, str) and nested_log_path.strip():
                candidate = history_dir / nested_log_path
                try:
                    nested_payload = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    nested_payload = None
                nested_count, nested_name_counts = _tool_metrics_from_log_payload(nested_payload)
                if count is None and nested_count is not None:
                    count = nested_count
                if not name_counts and nested_name_counts:
                    name_counts = nested_name_counts

        if count is not None:
            total += count
            has_total = True
        merged_name_counts = _merge_tool_name_counts(merged_name_counts, name_counts)

    return (total if has_total else None), merged_name_counts


def _apply_tool_metrics_from_log_path(entry: Dict[str, Any], history_dir: Path | None) -> None:
    if history_dir is None:
        return
    if _tool_call_count(entry.get("tool_call_count")) is not None and _normalize_tool_name_counts(entry.get("tool_name_counts")):
        return
    log_path = entry.get("log_path")
    if not isinstance(log_path, str) or not log_path.strip():
        return
    candidate = history_dir / log_path
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    count, name_counts = _tool_metrics_from_log_payload(payload)
    nested_count, nested_name_counts = _tool_metrics_from_question_log_entries(
        payload.get("question_logs") if isinstance(payload, dict) else None,
        history_dir,
    )
    if count is None:
        count = nested_count
    if not name_counts and nested_name_counts:
        name_counts = nested_name_counts
    if _tool_call_count(entry.get("tool_call_count")) is None and count is not None:
        entry["tool_call_count"] = count
    if not _normalize_tool_name_counts(entry.get("tool_name_counts")) and name_counts:
        entry["tool_name_counts"] = name_counts


def _apply_prompt_metrics_from_log_path(entry: Dict[str, Any], history_dir: Path | None) -> None:
    if history_dir is None:
        return
    log_path = entry.get("log_path")
    if not isinstance(log_path, str) or not log_path.strip():
        return
    candidate = history_dir / log_path
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    metrics = _prompt_metrics_from_log_payload(payload)
    if not metrics:
        return
    if "prompt_tokens" in metrics:
        entry["prompt_tokens"] = int(metrics["prompt_tokens"])
    if "prompt_latency_ms" in metrics:
        entry["prompt_latency_ms"] = float(metrics["prompt_latency_ms"])
    if "approx_prompt_tps" in metrics:
        entry["approx_prompt_tps"] = float(metrics["approx_prompt_tps"])
    for field in (
        "initial_prompt_tokens",
        "initial_prompt_latency_ms",
        "initial_prompt_tps",
        "conversation_prompt_tokens",
        "conversation_prompt_latency_ms",
        "conversation_prompt_tps",
    ):
        if field not in metrics:
            continue
        value = metrics[field]
        entry[field] = int(value) if field.endswith("_tokens") else float(value)


def _apply_tool_metrics(
    entry: Dict[str, Any],
    *,
    question_results: list[Dict[str, Any]] | None = None,
    default_zero: bool = False,
) -> None:
    count = _tool_call_count(entry.get("tool_call_count"))
    name_counts = _normalize_tool_name_counts(entry.get("tool_name_counts"))

    if question_results is not None:
        derived_count = 0
        has_derived_count = False
        derived_name_counts: Dict[str, int] = {}
        for question_result in question_results:
            question_count = _tool_call_count(question_result.get("tool_call_count"))
            if question_count is not None:
                derived_count += question_count
                has_derived_count = True
            for name, name_count in _normalize_tool_name_counts(question_result.get("tool_name_counts")).items():
                derived_name_counts[name] = derived_name_counts.get(name, 0) + name_count

        if count is None and has_derived_count:
            count = derived_count
        if not name_counts and derived_name_counts:
            name_counts = derived_name_counts

    if count is None and name_counts:
        count = sum(name_counts.values())
    if count is None and default_zero:
        count = 0

    if count is not None:
        entry["tool_call_count"] = count
    elif "tool_call_count" in entry:
        entry.pop("tool_call_count", None)

    if name_counts:
        entry["tool_name_counts"] = name_counts
    elif "tool_name_counts" in entry:
        entry.pop("tool_name_counts", None)


def _apply_prompt_metrics(
    entry: Dict[str, Any],
    *,
    question_results: list[Dict[str, Any]] | None = None,
) -> None:
    benchmark_mode = str(entry.get("benchmark_mode") or "").strip().lower()
    ttft_ms = _numeric_value(entry.get("ttft_ms"))
    total_latency_ms = _numeric_value(entry.get("total_latency_ms"))
    prompt_tokens = _numeric_value(entry.get("prompt_tokens"))
    prompt_latency_ms = _numeric_value(entry.get("prompt_latency_ms"))
    initial_prompt_tokens = _numeric_value(entry.get("initial_prompt_tokens"))
    initial_prompt_latency_ms = _numeric_value(entry.get("initial_prompt_latency_ms"))
    initial_prompt_tps = _numeric_value(entry.get("initial_prompt_tps"))
    conversation_prompt_tokens = _numeric_value(entry.get("conversation_prompt_tokens"))
    conversation_prompt_latency_ms = _numeric_value(entry.get("conversation_prompt_latency_ms"))
    conversation_prompt_tps = _numeric_value(entry.get("conversation_prompt_tps"))

    if question_results is not None:
        derived_prompt_tokens = 0
        has_derived_prompt_tokens = False
        derived_prompt_latency_ms = 0.0
        has_derived_prompt_latency = False
        derived_initial_prompt_tokens = 0
        has_derived_initial_prompt_tokens = False
        derived_initial_prompt_latency_ms = 0.0
        has_derived_initial_prompt_latency = False
        derived_conversation_prompt_tokens = 0
        has_derived_conversation_prompt_tokens = False
        derived_conversation_prompt_latency_ms = 0.0
        has_derived_conversation_prompt_latency = False
        for question_result in question_results:
            question_prompt_tokens = _numeric_value(question_result.get("prompt_tokens"))
            if question_prompt_tokens is not None and question_prompt_tokens >= 0:
                derived_prompt_tokens += int(question_prompt_tokens)
                has_derived_prompt_tokens = True
            question_prompt_latency_ms = _numeric_value(question_result.get("prompt_latency_ms"))
            if question_prompt_latency_ms is not None and question_prompt_latency_ms > 0:
                derived_prompt_latency_ms += question_prompt_latency_ms
                has_derived_prompt_latency = True
            question_initial_prompt_tokens = _numeric_value(question_result.get("initial_prompt_tokens"))
            if question_initial_prompt_tokens is not None and question_initial_prompt_tokens >= 0:
                derived_initial_prompt_tokens += int(question_initial_prompt_tokens)
                has_derived_initial_prompt_tokens = True
            question_initial_prompt_latency_ms = _numeric_value(question_result.get("initial_prompt_latency_ms"))
            if question_initial_prompt_latency_ms is not None and question_initial_prompt_latency_ms > 0:
                derived_initial_prompt_latency_ms += question_initial_prompt_latency_ms
                has_derived_initial_prompt_latency = True
            question_conversation_prompt_tokens = _numeric_value(question_result.get("conversation_prompt_tokens"))
            if question_conversation_prompt_tokens is not None and question_conversation_prompt_tokens >= 0:
                derived_conversation_prompt_tokens += int(question_conversation_prompt_tokens)
                has_derived_conversation_prompt_tokens = True
            question_conversation_prompt_latency_ms = _numeric_value(question_result.get("conversation_prompt_latency_ms"))
            if question_conversation_prompt_latency_ms is not None and question_conversation_prompt_latency_ms > 0:
                derived_conversation_prompt_latency_ms += question_conversation_prompt_latency_ms
                has_derived_conversation_prompt_latency = True

        if prompt_tokens is None and has_derived_prompt_tokens:
            prompt_tokens = float(derived_prompt_tokens)
        if prompt_latency_ms is None and has_derived_prompt_latency:
            prompt_latency_ms = derived_prompt_latency_ms
        if initial_prompt_tokens is None and has_derived_initial_prompt_tokens:
            initial_prompt_tokens = float(derived_initial_prompt_tokens)
        if initial_prompt_latency_ms is None and has_derived_initial_prompt_latency:
            initial_prompt_latency_ms = derived_initial_prompt_latency_ms
        if conversation_prompt_tokens is None and has_derived_conversation_prompt_tokens:
            conversation_prompt_tokens = float(derived_conversation_prompt_tokens)
        if conversation_prompt_latency_ms is None and has_derived_conversation_prompt_latency:
            conversation_prompt_latency_ms = derived_conversation_prompt_latency_ms

    if prompt_tokens is None:
        prompt_tokens = _prompt_tokens_from_totals(entry)

    if prompt_tokens is None and benchmark_mode != "docker_task":
        if initial_prompt_tokens is not None and initial_prompt_tokens > 0:
            prompt_tokens = initial_prompt_tokens
        elif conversation_prompt_tokens is not None and conversation_prompt_tokens > 0:
            prompt_tokens = conversation_prompt_tokens

    if prompt_latency_ms is None and benchmark_mode != "docker_task":
        if initial_prompt_latency_ms is not None and initial_prompt_latency_ms > 0:
            prompt_latency_ms = initial_prompt_latency_ms
        elif conversation_prompt_latency_ms is not None and conversation_prompt_latency_ms > 0:
            prompt_latency_ms = conversation_prompt_latency_ms

    if prompt_tokens is not None:
        entry["prompt_tokens"] = int(prompt_tokens)
    elif "prompt_tokens" in entry:
        entry.pop("prompt_tokens", None)

    if prompt_latency_ms is not None and prompt_latency_ms > 0:
        entry["prompt_latency_ms"] = float(prompt_latency_ms)
    elif "prompt_latency_ms" in entry:
        entry.pop("prompt_latency_ms", None)

    if prompt_tokens is not None and prompt_latency_ms is not None and prompt_latency_ms > 0:
        entry["approx_prompt_tps"] = float(prompt_tokens) / (prompt_latency_ms / 1000.0)
    elif "approx_prompt_tps" in entry:
        entry.pop("approx_prompt_tps", None)

    if initial_prompt_tokens is None and benchmark_mode != "docker_task" and prompt_tokens is not None:
        initial_prompt_tokens = prompt_tokens
    if initial_prompt_latency_ms is None and benchmark_mode != "docker_task" and ttft_ms is not None and ttft_ms > 0:
        initial_prompt_latency_ms = ttft_ms
    initial_prompt_tps = initial_prompt_tps or _compute_tps(initial_prompt_tokens, initial_prompt_latency_ms)
    if initial_prompt_tokens is not None:
        entry["initial_prompt_tokens"] = int(initial_prompt_tokens)
    elif "initial_prompt_tokens" in entry:
        entry.pop("initial_prompt_tokens", None)
    if initial_prompt_latency_ms is not None and initial_prompt_latency_ms > 0:
        entry["initial_prompt_latency_ms"] = float(initial_prompt_latency_ms)
    elif "initial_prompt_latency_ms" in entry:
        entry.pop("initial_prompt_latency_ms", None)
    if initial_prompt_tps is not None:
        entry["initial_prompt_tps"] = float(initial_prompt_tps)
    elif "initial_prompt_tps" in entry:
        entry.pop("initial_prompt_tps", None)

    if conversation_prompt_tokens is None:
        if benchmark_mode == "docker_task" and prompt_tokens is not None:
            conversation_prompt_tokens = prompt_tokens
        elif benchmark_mode != "docker_task":
            conversation_prompt_tokens = prompt_tokens if prompt_tokens is not None else initial_prompt_tokens
    if conversation_prompt_latency_ms is None:
        if benchmark_mode == "docker_task" and prompt_latency_ms is not None and prompt_latency_ms > 0:
            conversation_prompt_latency_ms = prompt_latency_ms
        elif benchmark_mode != "docker_task":
            if total_latency_ms is not None and total_latency_ms > 0:
                conversation_prompt_latency_ms = total_latency_ms
            elif prompt_latency_ms is not None and prompt_latency_ms > 0:
                conversation_prompt_latency_ms = prompt_latency_ms
    conversation_prompt_tps = conversation_prompt_tps or _compute_tps(
        conversation_prompt_tokens,
        conversation_prompt_latency_ms,
    )
    if conversation_prompt_tokens is not None:
        entry["conversation_prompt_tokens"] = int(conversation_prompt_tokens)
    elif "conversation_prompt_tokens" in entry:
        entry.pop("conversation_prompt_tokens", None)
    if conversation_prompt_latency_ms is not None and conversation_prompt_latency_ms > 0:
        entry["conversation_prompt_latency_ms"] = float(conversation_prompt_latency_ms)
    elif "conversation_prompt_latency_ms" in entry:
        entry.pop("conversation_prompt_latency_ms", None)
    if conversation_prompt_tps is not None:
        entry["conversation_prompt_tps"] = float(conversation_prompt_tps)
    elif "conversation_prompt_tps" in entry:
        entry.pop("conversation_prompt_tps", None)


def _backfill_prompt_metrics_from_peer_records(history_runs: list[Dict[str, Any]]) -> None:
    by_model_prompt: Dict[str, Dict[str, list[int]]] = {}
    by_prompt: Dict[str, Dict[str, list[int]]] = {}

    def bucket_for(store: Dict[str, Dict[str, list[int]]], key: str) -> Dict[str, list[int]]:
        bucket = store.get(key)
        if bucket is None:
            bucket = {field: [] for field in PROMPT_TOKEN_FIELDS}
            store[key] = bucket
        return bucket

    for run in history_runs:
        run_prompt_text = _first_text([run.get("prompt_text")])
        run_model = _first_text([run.get("model")]) or "(unknown)"
        raw_records = run.get("records")
        if not isinstance(raw_records, list):
            continue
        for record in raw_records:
            if not isinstance(record, dict):
                continue
            benchmark_mode = str(record.get("benchmark_mode") or run.get("benchmark_mode") or "").strip().lower()
            if benchmark_mode == "docker_task":
                continue
            prompt_text = _first_text([record.get("prompt_text"), run_prompt_text])
            if not prompt_text:
                continue
            model = _first_text([record.get("model"), run_model]) or "(unknown)"
            model_bucket = bucket_for(by_model_prompt, f"{model}{PROMPT_PEER_KEY_SEPARATOR}{prompt_text}")
            prompt_bucket = bucket_for(by_prompt, prompt_text)
            for field in PROMPT_TOKEN_FIELDS:
                token_count = _positive_token_count(record.get(field))
                if token_count is None:
                    continue
                model_bucket[field].append(token_count)
                prompt_bucket[field].append(token_count)

    for run in history_runs:
        run_prompt_text = _first_text([run.get("prompt_text")])
        run_model = _first_text([run.get("model")]) or "(unknown)"
        raw_records = run.get("records")
        if not isinstance(raw_records, list):
            continue
        for record in raw_records:
            if not isinstance(record, dict):
                continue
            benchmark_mode = str(record.get("benchmark_mode") or run.get("benchmark_mode") or "").strip().lower()
            if benchmark_mode == "docker_task":
                continue
            prompt_text = _first_text([record.get("prompt_text"), run_prompt_text])
            if not prompt_text:
                continue
            model = _first_text([record.get("model"), run_model]) or "(unknown)"
            model_bucket = by_model_prompt.get(f"{model}{PROMPT_PEER_KEY_SEPARATOR}{prompt_text}", {})
            prompt_bucket = by_prompt.get(prompt_text, {})
            changed = False
            for field in PROMPT_TOKEN_FIELDS:
                if _positive_token_count(record.get(field)) is not None:
                    continue
                prior_value = None
                for candidate_field in PROMPT_TOKEN_FALLBACK_FIELDS[field]:
                    prior_value = _median_int(model_bucket.get(candidate_field, []))
                    if prior_value is not None:
                        break
                    prior_value = _median_int(prompt_bucket.get(candidate_field, []))
                    if prior_value is not None:
                        break
                if prior_value is None:
                    continue
                record[field] = prior_value
                changed = True
            if changed:
                raw_question_results = record.get("question_results")
                _apply_prompt_metrics(
                    record,
                    question_results=raw_question_results if isinstance(raw_question_results, list) else None,
                )


def normalize_run_entry(run_data: Dict[str, Any], history_dir: Path | None = None) -> Dict[str, Any]:
    normalized = dict(run_data)
    raw_records = normalized.get("records")
    records = [dict(record) for record in raw_records if isinstance(record, dict)] if isinstance(raw_records, list) else []

    model_candidates: list[Any] = [normalized.get("model")]
    models_value = normalized.get("models")
    if isinstance(models_value, list):
        model_candidates.extend(models_value)
    model_candidates.extend(record.get("model") for record in records)
    model = _first_text(model_candidates) or "(unknown)"

    prompt_candidates: list[Any] = [normalized.get("prompt_text")]
    prompt_candidates.extend(record.get("prompt_text") for record in records)
    prompt_text = _first_text(prompt_candidates)

    enriched_records: list[Dict[str, Any]] = []
    for record in records:
        enriched = dict(record)
        enriched.setdefault("model", model)
        enriched.setdefault("prompt_text", prompt_text)
        enriched.setdefault("run_id", normalized.get("run_id"))
        enriched.setdefault("run_started_at", normalized.get("started_at"))
        enriched.setdefault("benchmark_mode", normalized.get("benchmark_mode") or "prompt")
        enriched.setdefault("benchmark_id", normalized.get("benchmark_id"))
        enriched.setdefault("benchmark_title", normalized.get("benchmark_title") or normalized.get("benchmark_id"))
        enriched.setdefault("question_count", normalized.get("question_count"))
        annotate_error_info(enriched)
        raw_question_results = enriched.get("question_results")
        if isinstance(raw_question_results, list):
            enriched_question_results: list[Dict[str, Any]] = []
            for question_result in raw_question_results:
                if not isinstance(question_result, dict):
                    continue
                enriched_question_result = dict(question_result)
                annotate_error_info(enriched_question_result)
                _apply_tool_metrics_from_log_path(enriched_question_result, history_dir)
                _apply_prompt_metrics_from_log_path(enriched_question_result, history_dir)
                _apply_tool_metrics(enriched_question_result)
                _apply_prompt_metrics(enriched_question_result)
                enriched_question_results.append(enriched_question_result)
            enriched["question_results"] = enriched_question_results
            _apply_tool_metrics(
                enriched,
                question_results=enriched_question_results,
                default_zero=str(enriched.get("benchmark_mode") or "").strip().lower() != "docker_task",
            )
            _apply_prompt_metrics(enriched, question_results=enriched_question_results)
        else:
            _apply_tool_metrics_from_log_path(enriched, history_dir)
            _apply_prompt_metrics_from_log_path(enriched, history_dir)
            _apply_tool_metrics(
                enriched,
                default_zero=str(enriched.get("benchmark_mode") or "").strip().lower() != "docker_task",
            )
            _apply_prompt_metrics(enriched)
        enriched_records.append(enriched)

    normalized["model"] = model
    normalized["prompt_text"] = prompt_text
    normalized["records"] = enriched_records
    normalized.pop("models", None)
    normalized.pop("config", None)
    normalized.pop("summary", None)
    return normalized


def compact_run_entry(run_data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_run_entry(run_data)
    compact = dict(normalized)
    model = normalized.get("model")
    prompt_text = normalized.get("prompt_text")

    compact_records: list[Dict[str, Any]] = []
    for record in normalized.get("records", []):
        if not isinstance(record, dict):
            continue
        reduced = dict(record)
        if reduced.get("model") == model:
            reduced.pop("model", None)
        if reduced.get("prompt_text") == prompt_text:
            reduced.pop("prompt_text", None)
        if reduced.get("run_id") == normalized.get("run_id"):
            reduced.pop("run_id", None)
        if reduced.get("run_started_at") == normalized.get("started_at"):
            reduced.pop("run_started_at", None)
        compact_records.append(reduced)

    compact["records"] = compact_records
    compact.pop("models", None)
    compact.pop("config", None)
    compact.pop("summary", None)
    return compact


def load_history_entries(history_path: Path) -> List[Dict[str, Any]]:
    if not history_path.exists():
        return []
    content = history_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    loaded = json.loads(content)
    if isinstance(loaded, list):
        history = [normalize_run_entry(item, history_path.parent) for item in loaded if isinstance(item, dict)]
        _backfill_prompt_metrics_from_peer_records(history)
        return history
    if isinstance(loaded, dict):
        history = [normalize_run_entry(loaded, history_path.parent)]
        _backfill_prompt_metrics_from_peer_records(history)
        return history
    return []


def update_history(history_path: Path, run_data: Dict[str, Any]) -> None:
    history = load_history_entries(history_path)
    normalized_run = normalize_run_entry(run_data)
    run_id = str(normalized_run.get("run_id") or "")
    replaced = False
    for index, entry in enumerate(history):
        if entry.get("run_id") == run_id:
            history[index] = normalized_run
            replaced = True
            break
    if not replaced:
        history.append(normalized_run)
    _backfill_prompt_metrics_from_peer_records(history)
    _ensure_parent(history_path)
    history_path.write_text(
        json.dumps([compact_run_entry(entry) for entry in history], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_latest(latest_path: Path, run_data: Dict[str, Any]) -> None:
    normalized_run = normalize_run_entry(run_data)
    _backfill_prompt_metrics_from_peer_records([normalized_run])
    _ensure_parent(latest_path)
    latest_path.write_text(
        json.dumps(compact_run_entry(normalized_run), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
