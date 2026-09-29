from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from .config import BenchmarkConfig
from .conditions import evaluation_identity
from .lmstudio_api import LMStudioAPIError, stream_chat_completion
from .stats import compute_run_summary
from .metrics import measured_metrics, inference_observation
from .telemetry import (
    TelemetryRecorder,
    build_failed_turn_usage_record,
    build_turn_usage_record,
    prompt_breakdown_from_messages,
)


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_from_exception(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, LMStudioAPIError) and "HTTPError" in str(exc):
        return "http_error"
    return "error"


def _empty_record(
    *,
    phase: str,
    iteration: int,
    started_at: str,
    status: str,
    error: str,
    prompt_text: str,
) -> Dict[str, Any]:
    return {
        "metrics": measured_metrics({}),
        "inference_requests": [inference_observation({}, status=status, turn=1, error=error)],
        "phase": phase,
        "iteration": iteration,
        "started_at": started_at,
        "ttft_ms": None,
        "total_latency_ms": None,
        "completion_window_ms": None,
        "prompt_tokens": None,
        "initial_prompt_tokens": None,
        "initial_prompt_latency_ms": None,
        "initial_prompt_tps": None,
        "conversation_prompt_tokens": None,
        "conversation_prompt_latency_ms": None,
        "conversation_prompt_tps": None,
        "completion_tokens": None,
        "total_tokens": None,
        "decode_tps": None,
        "end_to_end_tps": None,
        "approx_prompt_tps": None,
        "finish_reason": None,
        "status": status,
        "error": error,
        "reasoning_text": "",
        "response_text": "",
        "turn_usage": [
            build_failed_turn_usage_record(
                source="prompt",
                turn_index=1,
                error_type=status,
                error_message=error,
                timed_out=status == "timeout",
                prompt_breakdown=prompt_breakdown_from_messages([{"role": "user", "content": prompt_text}]),
            )
        ],
    }


def _turn_usage_from_result(result: Any, *, prompt_text: str) -> list[Dict[str, Any]]:
    metrics = measured_metrics(result.to_dict())
    prompt_breakdown = prompt_breakdown_from_messages([{"role": "user", "content": prompt_text}])
    elapsed_sec = (
        float(result.total_latency_ms) / 1000.0
        if isinstance(result.total_latency_ms, (int, float)) and result.total_latency_ms > 0
        else None
    )
    return [
        build_turn_usage_record(
            source="prompt",
            turn_index=1,
            prompt_tokens=result.prompt_tokens,
            cached_prompt_tokens=metrics["cached_prompt_tokens"],
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cumulative_prompt_tokens=result.prompt_tokens,
            cumulative_completion_tokens=result.completion_tokens,
            elapsed_sec=elapsed_sec,
            ttft_sec=(
                float(result.ttft_ms) / 1000.0
                if isinstance(result.ttft_ms, (int, float)) and result.ttft_ms >= 0
                else None
            ),
            first_chunk_sec=(
                float(result.ttft_ms) / 1000.0
                if isinstance(result.ttft_ms, (int, float)) and result.ttft_ms >= 0
                else None
            ),
            prefill_sec=(
                metrics["prefill_ms"] / 1000.0
                if metrics["prefill_ms"] is not None
                else None
            ),
            decode_sec=(
                metrics["generation_ms"] / 1000.0
                if metrics["generation_ms"] is not None
                else None
            ),
            post_first_token_sec=(
                float(result.completion_window_ms) / 1000.0
                if isinstance(result.completion_window_ms, (int, float)) and result.completion_window_ms >= 0
                else None
            ),
            prompt_breakdown=prompt_breakdown,
            timing_sources={"ttft_sec": metrics["sources"]["ttft_ms"],
                            "prefill_sec": metrics["sources"]["prefill_ms"],
                            "decode_sec": "api.timings.predicted_ms" if metrics["generation_ms"] is not None else "unavailable"},
            metrics=metrics,
        )
    ]


def run_benchmark(
    config: BenchmarkConfig,
    *,
    model: str | None = None,
    requested_model: str | None = None,
    client: Callable[..., Any] = stream_chat_completion,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.perf_counter,
    execution: Any = None,
) -> Dict[str, Any]:
    selected_model = model or (config.models[0] if len(config.models) == 1 else None)
    if not selected_model:
        raise ValueError("run_benchmark には単一モデルを渡してください。")
    display_model = requested_model or selected_model
    from .inspect_harness import require_inspect
    require_inspect()

    records: list[Dict[str, Any]] = []
    phases = [("cold", idx + 1) for idx in range(config.runs.cold_runs)] + [
        ("warm", idx + 1) for idx in range(config.runs.warm_runs)
    ]
    run_id = execution.run_id if execution else uuid.uuid4().hex[:8]
    started_at = execution.started_at if execution else _utc_iso_now()
    started_perf = now_fn()
    telemetry = TelemetryRecorder(
        run_id=run_id,
        started_at=started_at,
        now_fn=now_fn,
        origin_perf=started_perf,
        source="prompt",
    )

    console_lines: list[str] = []
    attempt_logs: list[Dict[str, Any]] = []

    def emit(line: str) -> None:
        console_lines.append(line)
        print(line)

    emit(
        f"[Run {run_id}] model={display_model} "
        f"{f'api_model={selected_model} ' if display_model != selected_model else ''}"
        f"provider={config.provider} "
        f"prompt_len={len(config.prompt_text)} "
        f"cold={config.runs.cold_runs} warm={config.runs.warm_runs}"
    )

    emit(f"[Model] {selected_model}")

    def warmup(api_model: str) -> dict:
        result = client(api_base=config.api_base, model=api_model, prompt_text=config.prompt_text,
                        **config.request_parameters(),
                        timeout_sec=config.runs.timeout_sec, now_fn=now_fn)
        return {"status": "success", "ttft_ms": result.ttft_ms, "total_latency_ms": result.total_latency_ms}

    for phase, iteration in phases:
        if execution:
            cached = execution.cached(phase, iteration, "prompt")
            if cached is not None:
                records.append(cached["result"])
                attempt_logs.append({"record_index": len(records) - 1, "phase": phase, "iteration": iteration, "payload": cached["log"]})
                continue
            selected_model = execution.before_attempt(phase, iteration, warmup)
        attempt_started_at = _utc_iso_now()
        emit(f"  - {phase} #{iteration} ...")
        attempt_span = telemetry.start_span(
            "attempt",
            phase=phase,
            iteration=iteration,
            benchmark_mode=config.mode,
        )
        inspect_info: dict = {}
        try:
            def invoke():
                return client(
                    api_base=config.api_base,
                    model=selected_model,
                    prompt_text=config.prompt_text,
                    **config.request_parameters(),
                    timeout_sec=config.runs.timeout_sec,
                    now_fn=now_fn,
                )
            from .inspect_harness import run_unit, unit_log_directory
            result = run_unit(config=config, model=selected_model, operation=invoke,
                              log_dir=unit_log_directory(config, run_id, phase, iteration, "prompt"),
                              metadata={"run_id": run_id, "phase": phase, "iteration": iteration}, info=inspect_info)
            record = {
                "metrics": measured_metrics(result.to_dict()),
                "inference_requests": [inference_observation(result.to_dict(), turn=1)],
                "raw_usage": result.raw_usage,
                "raw_timings": result.raw_timings,
                "raw_stats": result.raw_stats,
                "phase": phase,
                "iteration": iteration,
                "started_at": attempt_started_at,
                "ttft_ms": result.ttft_ms,
                "total_latency_ms": result.total_latency_ms,
                "completion_window_ms": result.completion_window_ms,
                "prompt_tokens": result.prompt_tokens,
                "initial_prompt_tokens": result.initial_prompt_tokens,
                "initial_prompt_latency_ms": result.initial_prompt_latency_ms,
                "initial_prompt_tps": result.initial_prompt_tps,
                "conversation_prompt_tokens": result.conversation_prompt_tokens,
                "conversation_prompt_latency_ms": result.conversation_prompt_latency_ms,
                "conversation_prompt_tps": result.conversation_prompt_tps,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "decode_tps": result.decode_tps,
                "end_to_end_tps": result.end_to_end_tps,
                "approx_prompt_tps": result.approx_prompt_tps,
                "finish_reason": result.finish_reason,
                "status": "success",
                "error": None,
                "reasoning_text": result.reasoning_text,
                "response_text": result.response_text,
                "turn_usage": _turn_usage_from_result(result, prompt_text=config.prompt_text),
            }
            attempt_wall_ms = attempt_span.finish(status="success", metrics=record)
            record["attempt_wall_ms"] = attempt_wall_ms
            records.append(record)
            attempt_logs.append(
                {
                    "record_index": len(records) - 1,
                    "phase": phase,
                    "iteration": iteration,
                    "payload": {
                        "kind": "prompt_attempt",
                        "run_id": run_id,
                        "model": display_model,
                        "api_model": selected_model,
                        "phase": phase,
                        "iteration": iteration,
                        "started_at": attempt_started_at,
                        "request": {
                            "provider": config.provider,
                            "api_base": config.api_base,
                            "model": selected_model,
                            "prompt_text": config.prompt_text,
                            **config.recorded_request_parameters(),
                            "timeout_sec": config.runs.timeout_sec,
                        },
                        "response": dict(record),
                    },
                }
            )
        except Exception as exc:  # pragma: no cover - covered via tests using fake clients
            status = _status_from_exception(exc)
            emit(f"    failed: {status}: {exc}")
            record = _empty_record(
                phase=phase,
                iteration=iteration,
                started_at=attempt_started_at,
                status=status,
                error=str(exc),
                prompt_text=config.prompt_text,
            )
            attempt_wall_ms = attempt_span.finish(status=status, metrics=record)
            record["attempt_wall_ms"] = attempt_wall_ms
            records.append(record)
            attempt_logs.append(
                {
                    "record_index": len(records) - 1,
                    "phase": phase,
                    "iteration": iteration,
                    "payload": {
                        "kind": "prompt_attempt",
                        "run_id": run_id,
                        "model": display_model,
                        "api_model": selected_model,
                        "phase": phase,
                        "iteration": iteration,
                        "started_at": attempt_started_at,
                        "request": {
                            "provider": config.provider,
                            "api_base": config.api_base,
                            "model": selected_model,
                            "prompt_text": config.prompt_text,
                            **config.recorded_request_parameters(),
                            "timeout_sec": config.runs.timeout_sec,
                        },
                        "response": dict(record),
                    },
                }
            )
        if inspect_info:
            records[-1]["inspect"] = inspect_info
            attempt_logs[-1]["payload"]["inspect"] = inspect_info
        if execution:
            execution.save_unit(phase, iteration, "prompt", records[-1], attempt_logs[-1]["payload"])
        if config.runs.cooldown_sec > 0:
            cooldown_span = telemetry.start_span(
                "cooldown",
                phase=phase,
                iteration=iteration,
                cooldown_sec=config.runs.cooldown_sec,
            )
            sleep_fn(config.runs.cooldown_sec)
            cooldown_span.finish(status="success", metrics={"cooldown_sec": config.runs.cooldown_sec})

    ended_at = _utc_iso_now()
    duration_sec = max(now_fn() - started_perf, 0.0)
    run_data: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": duration_sec,
        "provider": config.provider,
        "evaluation": evaluation_identity(config),
        "api_base": config.api_base,
        "model": display_model,
        "api_model": selected_model,
        "prompt_text": config.prompt_text,
        "request": {
            **config.recorded_request_parameters(),
        },
        "runs": {
            "cold_runs": config.runs.cold_runs,
            "warm_runs": config.runs.warm_runs,
            "timeout_sec": config.runs.timeout_sec,
            "cooldown_sec": config.runs.cooldown_sec,
        },
        "config_path": str(config.config_path) if config.config_path else None,
        "benchmark_mode": config.mode,
        "benchmark_id": None,
        "benchmark_title": None,
        "question_count": 1,
        "records": records,
    }
    run_data["telemetry"] = telemetry.build(ended_at=ended_at, duration_ms=duration_sec * 1000.0)
    run_data["summary"] = compute_run_summary(run_data)
    run_data["_log_bundle"] = {
        "console_lines": console_lines,
        "attempts": attempt_logs,
    }
    return run_data
