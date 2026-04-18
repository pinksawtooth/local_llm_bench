from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, Optional

from .history import normalize_run_entry


COMMON_METRICS = (
    "ttft_ms",
    "total_latency_ms",
    "completion_window_ms",
    "prompt_tokens",
    "completion_tokens",
    "decode_tps",
    "end_to_end_tps",
    "approx_prompt_tps",
    "initial_prompt_tps",
    "conversation_prompt_tps",
)
BENCHMARK_METRICS = ("benchmark_score",)
ALL_METRICS = COMMON_METRICS + BENCHMARK_METRICS


def _numeric_values(records: Iterable[Dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _metric_stats(values: list[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "stddev": None,
            "cv": None,
        }
    current_mean = mean(values)
    current_stddev = pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": current_mean,
        "median": median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "stddev": current_stddev,
        "cv": (current_stddev / current_mean) if current_mean else None,
    }


def _phase_summary(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    success_records = [record for record in records if record.get("status") == "success"]
    status_counts = Counter(str(record.get("status") or "unknown") for record in records)
    finish_reasons = Counter(
        str(record.get("finish_reason") or "unknown")
        for record in success_records
    )
    metrics = {
        metric: _metric_stats(_numeric_values(success_records, metric))
        for metric in ALL_METRICS
    }
    benchmark_correct_count = sum(int(record.get("benchmark_correct_count") or 0) for record in records)
    benchmark_incorrect_count = sum(int(record.get("benchmark_incorrect_count") or 0) for record in records)
    benchmark_error_count = sum(int(record.get("benchmark_error_count") or 0) for record in records)
    benchmark_total = benchmark_correct_count + benchmark_incorrect_count + benchmark_error_count
    return {
        "samples": len(records),
        "success_count": len(success_records),
        "error_count": max(len(records) - len(success_records), 0),
        "success_rate": (len(success_records) / len(records)) if records else 0.0,
        "status_counts": dict(status_counts),
        "finish_reasons": dict(finish_reasons),
        "metrics": metrics,
        "benchmark": {
            "correct_count": benchmark_correct_count,
            "incorrect_count": benchmark_incorrect_count,
            "error_count": benchmark_error_count,
            "total_count": benchmark_total,
            "correct_rate": (
                benchmark_correct_count / benchmark_total
                if benchmark_total > 0
                else None
            ),
            "error_rate": (
                benchmark_error_count / benchmark_total
                if benchmark_total > 0
                else None
            ),
        },
    }


def _metric_mean(summary: Dict[str, Any], phase: str, metric: str) -> Optional[float]:
    phase_block = summary.get("phases", {}).get(phase, {})
    return phase_block.get("metrics", {}).get(metric, {}).get("mean")


def _flatten_phase_metrics(summary: Dict[str, Any], phase_name: str) -> None:
    phase = summary["phases"][phase_name]
    summary[f"{phase_name}_samples"] = phase["samples"]
    summary[f"{phase_name}_success_rate"] = phase["success_rate"]
    benchmark = phase.get("benchmark", {})
    summary[f"{phase_name}_benchmark_correct_count"] = benchmark.get("correct_count", 0)
    summary[f"{phase_name}_benchmark_incorrect_count"] = benchmark.get("incorrect_count", 0)
    summary[f"{phase_name}_benchmark_error_count"] = benchmark.get("error_count", 0)
    summary[f"{phase_name}_benchmark_total_count"] = benchmark.get("total_count", 0)
    summary[f"{phase_name}_benchmark_correct_rate"] = benchmark.get("correct_rate")
    summary[f"{phase_name}_benchmark_error_rate"] = benchmark.get("error_rate")
    for metric in ALL_METRICS:
        metric_summary = phase["metrics"][metric]
        summary[f"{phase_name}_mean_{metric}"] = metric_summary["mean"]
        summary[f"{phase_name}_p95_{metric}"] = metric_summary["p95"]
        summary[f"{phase_name}_stddev_{metric}"] = metric_summary["stddev"]
        summary[f"{phase_name}_cv_{metric}"] = metric_summary["cv"]


def _build_model_summary(model: str, records: list[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("phase") or "unknown")].append(record)

    summary = {
        "model": model,
        "total_samples": len(records),
        "phases": {
            "cold": _phase_summary(grouped.get("cold", [])),
            "warm": _phase_summary(grouped.get("warm", [])),
            "overall": _phase_summary(records),
        },
    }
    summary["success_count"] = summary["phases"]["overall"]["success_count"]
    summary["error_count"] = summary["phases"]["overall"]["error_count"]
    summary["success_rate"] = summary["phases"]["overall"]["success_rate"]
    summary["finish_reasons"] = summary["phases"]["overall"]["finish_reasons"]
    summary["status_counts"] = summary["phases"]["overall"]["status_counts"]
    summary["benchmark_correct_rate"] = summary["phases"]["overall"]["benchmark"]["correct_rate"]
    summary["benchmark_error_rate"] = summary["phases"]["overall"]["benchmark"]["error_rate"]
    summary["delta"] = {
        "ttft_ms": _delta(_metric_mean(summary, "cold", "ttft_ms"), _metric_mean(summary, "warm", "ttft_ms")),
        "total_latency_ms": _delta(
            _metric_mean(summary, "cold", "total_latency_ms"),
            _metric_mean(summary, "warm", "total_latency_ms"),
        ),
        "decode_tps": _delta(
            _metric_mean(summary, "cold", "decode_tps"),
            _metric_mean(summary, "warm", "decode_tps"),
        ),
    }
    for phase_name in ("cold", "warm", "overall"):
        _flatten_phase_metrics(summary, phase_name)
    return summary


def _delta(cold_value: Optional[float], warm_value: Optional[float]) -> Optional[float]:
    if cold_value is None or warm_value is None:
        return None
    return warm_value - cold_value


def _best_model(
    model_summaries: list[Dict[str, Any]],
    field: str,
    *,
    maximize: bool = False,
) -> Optional[Dict[str, Any]]:
    candidates = [
        summary for summary in model_summaries
        if isinstance(summary.get(field), (int, float))
    ]
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda item: float(item[field]),
        reverse=maximize,
    )
    best = ordered[0]
    return {"model": best["model"], "value": float(best[field])}


def _summary_payload_from_records(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("model") or "(unknown)")].append(record)

    model_summaries = [
        _build_model_summary(model, model_records)
        for model, model_records in sorted(grouped.items(), key=lambda item: item[0])
    ]

    cards_source = []
    for summary in model_summaries:
        cards_source.append({
            "model": summary["model"],
            "warm_mean_ttft_ms": summary["warm_mean_ttft_ms"],
            "warm_mean_total_latency_ms": summary["warm_mean_total_latency_ms"],
            "warm_mean_decode_tps": summary["warm_mean_decode_tps"],
            "warm_mean_benchmark_score": summary["warm_mean_benchmark_score"],
            "warm_benchmark_error_rate": summary["warm_benchmark_error_rate"],
        })

    fastest_ttft = _best_model(cards_source, "warm_mean_ttft_ms")
    fastest_warm_latency = _best_model(cards_source, "warm_mean_total_latency_ms")
    fastest_decode_speed = _best_model(cards_source, "warm_mean_decode_tps", maximize=True)
    best_benchmark_score = _best_model(cards_source, "warm_mean_benchmark_score", maximize=True)
    lowest_benchmark_error_rate = _best_model(cards_source, "warm_benchmark_error_rate")

    successful_samples = sum(1 for record in records if record.get("status") == "success")
    return {
        "total_models": len(model_summaries),
        "total_samples": len(records),
        "successful_samples": successful_samples,
        "failed_samples": max(len(records) - successful_samples, 0),
        "cards": {
            "fastest_ttft": fastest_ttft,
            "fastest_warm_latency": fastest_warm_latency,
            "fastest_decode_speed": fastest_decode_speed,
            "best_benchmark_score": best_benchmark_score,
            "lowest_benchmark_error_rate": lowest_benchmark_error_rate,
            "total_samples": {"value": len(records)},
        },
        "models": model_summaries,
    }


def compute_run_summary(run_data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_run_entry(run_data)
    records = [
        record for record in normalized.get("records", [])
        if isinstance(record, dict)
    ]
    return _summary_payload_from_records(records)


def compute_history_summary(history_entries: list[Dict[str, Any]]) -> Dict[str, Any]:
    normalized_runs = [
        normalize_run_entry(entry)
        for entry in history_entries
        if isinstance(entry, dict)
    ]
    flat_records: list[Dict[str, Any]] = []
    for run in normalized_runs:
        flat_records.extend(
            record for record in run.get("records", [])
            if isinstance(record, dict)
        )

    summary = _summary_payload_from_records(flat_records)
    summary["total_runs"] = len(normalized_runs)
    summary["latest_run_id"] = normalized_runs[-1].get("run_id") if normalized_runs else None
    summary["latest_started_at"] = normalized_runs[-1].get("started_at") if normalized_runs else None
    return summary
