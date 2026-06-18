from __future__ import annotations

import sys
import time
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict

try:
    import resource
except ImportError:  # pragma: no cover - resource is available on the supported local targets.
    resource = None  # type: ignore[assignment]


TELEMETRY_VERSION = 1

TELEMETRY_METRIC_FIELDS = (
    "ttft_ms",
    "total_latency_ms",
    "completion_window_ms",
    "attempt_wall_ms",
    "host_wall_ms",
    "prompt_latency_ms",
    "initial_prompt_latency_ms",
    "conversation_prompt_latency_ms",
    "prompt_tokens",
    "initial_prompt_tokens",
    "conversation_prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "decode_tps",
    "end_to_end_tps",
    "approx_prompt_tps",
    "initial_prompt_tps",
    "conversation_prompt_tps",
    "benchmark_score",
    "benchmark_correct_count",
    "benchmark_incorrect_count",
    "benchmark_error_count",
    "tool_call_count",
)


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            normalized = _jsonable(item)
            if normalized is not None:
                cleaned[str(key)] = normalized
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _clean_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        cleaned[str(key)] = _jsonable(value)
    return cleaned


def telemetry_metrics_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for field in TELEMETRY_METRIC_FIELDS:
        value = record.get(field)
        if isinstance(value, bool):
            metrics[field] = int(value)
        elif isinstance(value, (int, float)):
            metrics[field] = value
    return metrics


def _safe_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return int(value)
    return default


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _safe_positive_float(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _safe_nonnegative_float(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _content_char_len(value: Any, *, _seen: set[int] | None = None) -> int:
    if value is None:
        return 0
    if _seen is None:
        _seen = set()
    if isinstance(value, (list, dict)):
        marker = id(value)
        if marker in _seen:
            return 0
        _seen.add(marker)
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_content_char_len(item, _seen=_seen) for item in value)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return len(text)
        return sum(_content_char_len(item, _seen=_seen) for item in value.values())
    return len(str(value))


def estimated_tokens_from_chars(char_count: int) -> int:
    if char_count <= 0:
        return 0
    return max(1, int(math.ceil(char_count / 4)))


def _add_prompt_breakdown_category(
    categories: Dict[str, Dict[str, Any]],
    name: str,
    *,
    content_chars: int = 0,
    messages: int = 0,
    tool_calls: int = 0,
) -> None:
    item = categories.setdefault(
        name,
        {
            "messages": 0,
            "content_chars": 0,
            "estimated_tokens": 0,
        },
    )
    item["messages"] = int(item.get("messages", 0)) + messages
    item["content_chars"] = int(item.get("content_chars", 0)) + content_chars
    if tool_calls:
        item["tool_calls"] = int(item.get("tool_calls", 0)) + tool_calls
    item["estimated_tokens"] = estimated_tokens_from_chars(int(item["content_chars"]))


def prompt_breakdown_from_messages(
    messages: list[Dict[str, Any]],
    tools_schema: list[Dict[str, Any]] | None = None,
    *,
    tool_call_names_by_id: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    tools_schema = tools_schema or []
    categories: Dict[str, Dict[str, Any]] = {}
    role_counts: Dict[str, int] = {}
    tool_results_by_tool: Dict[str, Dict[str, Any]] = {}
    first_user_seen = False
    total_content_chars = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        content_chars = _content_char_len(message.get("content"))
        total_content_chars += content_chars

        if role == "system":
            category = "system"
        elif role == "user" and not first_user_seen:
            category = "task"
            first_user_seen = True
        elif role == "tool":
            category = "tool_results"
        else:
            category = "conversation_history"

        _add_prompt_breakdown_category(
            categories,
            category,
            content_chars=content_chars,
            messages=1,
        )

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            tool_name = "unknown"
            if isinstance(tool_call_id, str) and tool_call_names_by_id:
                tool_name = tool_call_names_by_id.get(tool_call_id, "unknown")
            tool_item = tool_results_by_tool.setdefault(
                tool_name,
                {
                    "messages": 0,
                    "content_chars": 0,
                    "estimated_tokens": 0,
                },
            )
            tool_item["messages"] = int(tool_item.get("messages", 0)) + 1
            tool_item["content_chars"] = int(tool_item.get("content_chars", 0)) + content_chars
            tool_item["estimated_tokens"] = estimated_tokens_from_chars(int(tool_item["content_chars"]))

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            tool_call_chars = _content_char_len(tool_calls)
            total_content_chars += tool_call_chars
            _add_prompt_breakdown_category(
                categories,
                "assistant_tool_calls",
                content_chars=tool_call_chars,
                tool_calls=len(tool_calls),
            )

    tool_schema_chars = _content_char_len(tools_schema)
    if tool_schema_chars:
        _add_prompt_breakdown_category(
            categories,
            "tool_schema",
            content_chars=tool_schema_chars,
            tool_calls=len(tools_schema),
        )

    return {
        "message_count": len(messages),
        "role_counts": role_counts,
        "tool_schema_count": len(tools_schema),
        "total_content_chars": total_content_chars,
        "tool_schema_chars": tool_schema_chars,
        "estimated_total_tokens": estimated_tokens_from_chars(total_content_chars + tool_schema_chars),
        "categories": categories,
        "tool_results_by_tool": tool_results_by_tool,
        "note": "estimated_tokens are character-based estimates, not provider tokenizer counts",
    }


def normalize_turn_usage_records(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    records: list[Dict[str, Any]] = []
    for raw_index, raw_record in enumerate(value, start=1):
        if not isinstance(raw_record, dict):
            continue

        record: Dict[str, Any] = {}
        source = raw_record.get("source")
        if isinstance(source, str) and source.strip():
            record["source"] = source.strip()

        turn_index = raw_record.get("turn_index")
        if not isinstance(turn_index, (int, float)) or isinstance(turn_index, bool):
            turn_index = raw_index
        record["turn_index"] = int(turn_index)

        for key in (
            "prompt_tokens",
            "cached_prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cumulative_prompt_tokens",
            "cumulative_completion_tokens",
            "cumulative_total_tokens",
        ):
            record[key] = _safe_int(raw_record.get(key), default=0)

        for key in (
            "elapsed_sec",
            "completion_tokens_per_sec",
            "total_tokens_per_sec",
            "cost",
            "ttft_sec",
            "first_chunk_sec",
            "prefill_sec",
            "decode_sec",
            "post_first_token_sec",
        ):
            raw_value = _safe_nonnegative_float(raw_record.get(key))
            if raw_value is not None:
                record[key] = raw_value

        for key in ("success", "timed_out"):
            raw_value = raw_record.get(key)
            if isinstance(raw_value, bool):
                record[key] = raw_value

        for key in ("error_type", "error_message", "question_id"):
            raw_value = raw_record.get(key)
            if isinstance(raw_value, str) and raw_value:
                record[key] = raw_value

        prompt_breakdown = raw_record.get("prompt_breakdown")
        if isinstance(prompt_breakdown, dict):
            record["prompt_breakdown"] = prompt_breakdown

        timing_sources = raw_record.get("timing_sources")
        if isinstance(timing_sources, dict):
            record["timing_sources"] = {
                str(key): str(item)
                for key, item in timing_sources.items()
                if isinstance(key, str) and isinstance(item, str)
            }

        records.append(record)
    return records


def build_turn_usage_record(
    *,
    source: str,
    turn_index: int,
    prompt_tokens: Any = None,
    cached_prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
    cumulative_prompt_tokens: Any = None,
    cumulative_completion_tokens: Any = None,
    elapsed_sec: Any = None,
    success: bool = True,
    cost: Any = None,
    ttft_sec: Any = None,
    first_chunk_sec: Any = None,
    prefill_sec: Any = None,
    decode_sec: Any = None,
    post_first_token_sec: Any = None,
    prompt_breakdown: Dict[str, Any] | None = None,
    timing_sources: Dict[str, str] | None = None,
    question_id: str | None = None,
) -> Dict[str, Any]:
    prompt_count = max(_safe_int(prompt_tokens), 0)
    cached_count = max(_safe_int(cached_prompt_tokens), 0)
    completion_count = max(_safe_int(completion_tokens), 0)
    total_count = max(_safe_int(total_tokens, default=prompt_count + completion_count), 0)
    cumulative_prompt = max(_safe_int(cumulative_prompt_tokens, default=prompt_count), 0)
    cumulative_completion = max(_safe_int(cumulative_completion_tokens, default=completion_count), 0)

    record: Dict[str, Any] = {
        "source": source,
        "turn_index": max(int(turn_index), 1),
        "prompt_tokens": prompt_count,
        "cached_prompt_tokens": cached_count,
        "completion_tokens": completion_count,
        "total_tokens": total_count,
        "cumulative_prompt_tokens": cumulative_prompt,
        "cumulative_completion_tokens": cumulative_completion,
        "cumulative_total_tokens": cumulative_prompt + cumulative_completion,
        "success": bool(success),
    }
    if question_id:
        record["question_id"] = str(question_id)

    elapsed = _safe_positive_float(elapsed_sec)
    if elapsed is not None:
        record["elapsed_sec"] = elapsed
        record["completion_tokens_per_sec"] = completion_count / elapsed if completion_count else 0.0
        record["total_tokens_per_sec"] = total_count / elapsed if total_count else 0.0

    timing_values = {
        "ttft_sec": ttft_sec,
        "first_chunk_sec": first_chunk_sec,
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "post_first_token_sec": post_first_token_sec,
    }
    for key, value in timing_values.items():
        parsed = _safe_nonnegative_float(value)
        if parsed is not None:
            record[key] = parsed

    parsed_cost = _safe_nonnegative_float(cost)
    if parsed_cost is not None:
        record["cost"] = parsed_cost
    if prompt_breakdown:
        record["prompt_breakdown"] = prompt_breakdown
    if timing_sources:
        record["timing_sources"] = dict(timing_sources)
    return normalize_turn_usage_records([record])[0]


def build_failed_turn_usage_record(
    *,
    source: str,
    turn_index: int,
    error_type: str,
    error_message: str,
    cumulative_prompt_tokens: Any = None,
    cumulative_completion_tokens: Any = None,
    elapsed_sec: Any = None,
    timed_out: bool = False,
    prompt_breakdown: Dict[str, Any] | None = None,
    question_id: str | None = None,
) -> Dict[str, Any]:
    record = build_turn_usage_record(
        source=source,
        turn_index=turn_index,
        cumulative_prompt_tokens=cumulative_prompt_tokens,
        cumulative_completion_tokens=cumulative_completion_tokens,
        elapsed_sec=elapsed_sec,
        success=False,
        prompt_breakdown=prompt_breakdown,
        question_id=question_id,
    )
    record["timed_out"] = timed_out
    record["error_type"] = error_type
    record["error_message"] = str(error_message or "")[:500]
    return normalize_turn_usage_records([record])[0]


def _maxrss_bytes(maxrss: int) -> int | None:
    if maxrss <= 0:
        return None
    if sys.platform == "darwin":
        return int(maxrss)
    return int(maxrss) * 1024


def _resource_sample() -> Dict[str, Any]:
    if resource is None:
        return {}
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:
        return {}
    maxrss = _maxrss_bytes(int(getattr(usage, "ru_maxrss", 0) or 0))
    sample: Dict[str, Any] = {
        "cpu_user_ms": float(getattr(usage, "ru_utime", 0.0) or 0.0) * 1000.0,
        "cpu_system_ms": float(getattr(usage, "ru_stime", 0.0) or 0.0) * 1000.0,
        "max_rss_bytes": maxrss,
    }
    return _clean_dict(sample)


@dataclass
class TelemetrySpan:
    recorder: "TelemetryRecorder"
    name: str
    started_offset_ms: float
    attrs: Dict[str, Any]

    def finish(self, *, status: str = "success", metrics: Dict[str, Any] | None = None, **attrs: Any) -> float:
        return self.recorder.finish_span(
            self,
            status=status,
            metrics=metrics,
            **attrs,
        )


class TelemetryRecorder:
    def __init__(
        self,
        *,
        run_id: str,
        started_at: str,
        now_fn: Callable[[], float] = time.perf_counter,
        wall_time_fn: Callable[[], str] = _utc_iso_now,
        origin_perf: float | None = None,
        source: str = "local_llm_bench",
    ) -> None:
        self.run_id = run_id
        self.started_at = started_at
        self.now_fn = now_fn
        self.wall_time_fn = wall_time_fn
        self.source = source
        self.origin_perf = now_fn() if origin_perf is None else origin_perf
        self.events: list[Dict[str, Any]] = []
        self.spans: list[Dict[str, Any]] = []
        self.resource_samples: list[Dict[str, Any]] = []
        self.mark("run_start")
        self.sample_resource("run_start")

    def offset_ms(self) -> float:
        return max((self.now_fn() - self.origin_perf) * 1000.0, 0.0)

    def mark(self, name: str, **attrs: Any) -> Dict[str, Any]:
        event = {
            "name": name,
            "timestamp": self.wall_time_fn(),
            "offset_ms": self.offset_ms(),
            **_clean_dict(attrs),
        }
        self.events.append(event)
        return event

    def sample_resource(self, label: str, **attrs: Any) -> Dict[str, Any]:
        sample = {
            "label": label,
            "timestamp": self.wall_time_fn(),
            "offset_ms": self.offset_ms(),
            **_resource_sample(),
            **_clean_dict(attrs),
        }
        self.resource_samples.append(sample)
        return sample

    def start_span(self, name: str, **attrs: Any) -> TelemetrySpan:
        event = self.mark(f"{name}_start", **attrs)
        return TelemetrySpan(
            recorder=self,
            name=name,
            started_offset_ms=float(event.get("offset_ms") or 0.0),
            attrs=_clean_dict(attrs),
        )

    def finish_span(
        self,
        span: TelemetrySpan,
        *,
        status: str = "success",
        metrics: Dict[str, Any] | None = None,
        **attrs: Any,
    ) -> float:
        ended_offset_ms = self.offset_ms()
        duration_ms = max(ended_offset_ms - span.started_offset_ms, 0.0)
        clean_metrics = telemetry_metrics_from_record(metrics or {})
        span_payload = {
            "name": span.name,
            **span.attrs,
            **_clean_dict(attrs),
            "status": status,
            "start_offset_ms": span.started_offset_ms,
            "end_offset_ms": ended_offset_ms,
            "duration_ms": duration_ms,
            "metrics": clean_metrics,
        }
        self.spans.append(span_payload)
        self.mark(
            f"{span.name}_end",
            **span.attrs,
            **_clean_dict(attrs),
            status=status,
            duration_ms=duration_ms,
            metrics=clean_metrics,
        )
        self.sample_resource(f"{span.name}_end", status=status, **span.attrs)
        return duration_ms

    def build(self, *, ended_at: str, duration_ms: float | None = None) -> Dict[str, Any]:
        final_duration_ms = self.offset_ms() if duration_ms is None else max(float(duration_ms), 0.0)
        self.mark("run_end", duration_ms=final_duration_ms)
        self.sample_resource("run_end")
        return {
            "version": TELEMETRY_VERSION,
            "source": self.source,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "duration_ms": final_duration_ms,
            "events": self.events,
            "spans": self.spans,
            "resources": {
                "samples": self.resource_samples,
                "summary": self._resource_summary(),
            },
            "summary": self._summary(final_duration_ms),
        }

    def _resource_summary(self) -> Dict[str, Any]:
        if not self.resource_samples:
            return {}
        first = self.resource_samples[0]
        last = self.resource_samples[-1]
        peak_rss_values = [
            sample.get("max_rss_bytes")
            for sample in self.resource_samples
            if isinstance(sample.get("max_rss_bytes"), (int, float))
        ]
        return _clean_dict(
            {
                "sample_count": len(self.resource_samples),
                "cpu_user_delta_ms": _numeric_delta(first.get("cpu_user_ms"), last.get("cpu_user_ms")),
                "cpu_system_delta_ms": _numeric_delta(first.get("cpu_system_ms"), last.get("cpu_system_ms")),
                "peak_rss_bytes": max(peak_rss_values) if peak_rss_values else None,
            }
        )

    def _summary(self, duration_ms: float) -> Dict[str, Any]:
        status_counts: Dict[str, int] = {}
        for span in self.spans:
            status = str(span.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        return {
            "duration_ms": duration_ms,
            "event_count": len(self.events),
            "span_count": len(self.spans),
            "status_counts": status_counts,
            "attempt_count": sum(1 for span in self.spans if span.get("name") == "attempt"),
            "question_count": sum(1 for span in self.spans if span.get("name") == "question"),
        }


def _numeric_delta(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return max(float(right) - float(left), 0.0)
