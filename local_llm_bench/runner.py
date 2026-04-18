from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from .config import BenchmarkConfig
from .lmstudio_api import LMStudioAPIError, stream_chat_completion
from .stats import compute_run_summary


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
) -> Dict[str, Any]:
    return {
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
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    model: str | None = None,
    requested_model: str | None = None,
    client: Callable[..., Any] = stream_chat_completion,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.perf_counter,
) -> Dict[str, Any]:
    selected_model = model or (config.models[0] if len(config.models) == 1 else None)
    if not selected_model:
        raise ValueError("run_benchmark には単一モデルを渡してください。")
    display_model = requested_model or selected_model

    records: list[Dict[str, Any]] = []
    phases = [("cold", idx + 1) for idx in range(config.runs.cold_runs)] + [
        ("warm", idx + 1) for idx in range(config.runs.warm_runs)
    ]
    run_id = uuid.uuid4().hex[:8]
    started_at = _utc_iso_now()
    started_perf = now_fn()

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
    for phase, iteration in phases:
        attempt_started_at = _utc_iso_now()
        emit(f"  - {phase} #{iteration} ...")
        try:
            result = client(
                api_base=config.api_base,
                model=selected_model,
                prompt_text=config.prompt_text,
                temperature=config.request.temperature,
                max_tokens=config.request.max_tokens,
                timeout_sec=config.runs.timeout_sec,
                now_fn=now_fn,
              )
            record = {
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
            }
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
                            "temperature": config.request.temperature,
                            "max_tokens": config.request.max_tokens,
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
            )
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
                            "temperature": config.request.temperature,
                            "max_tokens": config.request.max_tokens,
                            "timeout_sec": config.runs.timeout_sec,
                        },
                        "response": dict(record),
                    },
                }
            )
        if config.runs.cooldown_sec > 0:
            sleep_fn(config.runs.cooldown_sec)

    ended_at = _utc_iso_now()
    duration_sec = max(now_fn() - started_perf, 0.0)
    run_data: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": duration_sec,
        "provider": config.provider,
        "api_base": config.api_base,
        "model": display_model,
        "api_model": selected_model,
        "prompt_text": config.prompt_text,
        "request": {
            "temperature": config.request.temperature,
            "max_tokens": config.request.max_tokens,
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
    run_data["summary"] = compute_run_summary(run_data)
    run_data["_log_bundle"] = {
        "console_lines": console_lines,
        "attempts": attempt_logs,
    }
    return run_data
