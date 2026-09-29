"""Input-length and concurrent-request benchmarks, evaluated by Inspect AI."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import time
import uuid

from .conditions import evaluation_identity
from .inspect_harness import run_samples, run_unit, unit_log_directory
from .memory_measurement import MemorySampler
from .metrics import measured_metrics, number, rate, inference_observation
from .runner import _empty_record, _status_from_exception, _turn_usage_from_result, _utc_iso_now
from .stats import compute_run_summary

WORKLOAD_VERSION = "python-corpus-chars4-v1"


def scenario_units(settings):
    return [f"pp{length}-c{concurrency}-r{slot}"
            for length in settings.input_tokens for concurrency in settings.concurrency
            for slot in range(1, concurrency + 1)]


def make_prompt(target: int) -> str:
    """Stable shared text; target is a chars/4 estimate, NOT a tokenizer count.

    Identical text permits cross-runtime comparison without a tokenizer download.
    Actual API usage (including chat templates) is the plotted input length.
    """
    prefix = "Read this Python module and summarize its behavior in a short paragraph.\n\n"
    lines = []
    size = len(prefix)
    i = 0
    while size < target * 4:
        line = f"def scale_{i}(value):\n    return value * {i % 97 + 1} + {i % 31}\n\n"
        lines.append(line)
        size += len(line)
        i += 1
    return (prefix + "".join(lines))[:target * 4]


def cohort_metrics(records: list[dict], expected: int) -> dict:
    success = [r for r in records if r.get("status") == "success"]
    complete = len(records) == expected
    starts = [r.get("request_started_perf") for r in records]
    ends = [r.get("request_ended_perf") for r in records]
    elapsed = (max(ends) - min(starts)) if starts and all(number(x) is not None for x in starts + ends) else None
    peak = None
    spread = None
    if elapsed is not None and all(end > start for start, end in zip(starts, ends)):
        active, peak = 0, 0
        for _, delta in sorted([(start, 1) for start in starts] + [(end, -1) for end in ends]):
            active += delta
            peak = max(peak, active)
        spread = (max(starts) - min(starts)) * 1000
    def total(field):
        values = [(r.get("metrics") or {}).get(field) for r in success]
        return sum(values) if values and all(number(x) is not None for x in values) else None
    all_success = complete and len(success) == expected
    return {"complete": complete, "requests": expected, "completed": len(records),
            "success_count": len(success), "error_count": len(records) - len(success),
            "elapsed_sec": elapsed, "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
            "peak_inflight": peak, "dispatch_spread_ms": spread,
            "input_tps": rate(total("input_tokens"), elapsed) if all_success else None,
            "output_tps": rate(total("output_tokens"), elapsed) if all_success else None,
            "throughput_definition": "sum of tokens / cohort first request start to last request end (includes prefill)",
            "eligible_for_comparison": all_success and peak == expected and all(r.get("concurrency_actual") == r.get("concurrency_requested") for r in records)}


def run_performance_benchmark(config, *, model, requested_model=None, client, execution=None,
                              now_fn=time.perf_counter, sleep_fn=time.sleep, memory_factory=MemorySampler):
    run_id = execution.run_id if execution else uuid.uuid4().hex
    started_at = execution.started_at if execution else _utc_iso_now()
    began = now_fn()
    records, attempts = [], []
    selected_model = model
    phases = [(phase, i) for phase, count in (("cold", config.runs.cold_runs), ("warm", config.runs.warm_runs))
              for i in range(1, count + 1)]

    for length in config.performance.input_tokens:
        prompt = make_prompt(length)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        for concurrency in config.performance.concurrency:
            for phase, iteration in phases:
                units = [f"pp{length}-c{concurrency}-r{slot}" for slot in range(1, concurrency + 1)]
                pending = []
                for key in units:
                    cached = execution.cached(phase, iteration, key) if execution else None
                    if cached:
                        records.append(cached["result"])
                        attempts.append({"record_index": len(records) - 1, "payload": cached["log"]})
                    else:
                        pending.append(key)
                if not pending:
                    continue

                def warmup(api_model):
                    result = run_unit(config=replace(config, prompt_text=prompt), model=api_model,
                        operation=lambda: client(api_base=config.api_base, model=api_model, prompt_text=prompt,
                            **config.request_parameters(), timeout_sec=config.runs.timeout_sec, now_fn=now_fn),
                        log_dir=unit_log_directory(config, run_id, phase, iteration, units[0] + "-warmup"),
                        metadata={"run_id": run_id, "excluded_from_statistics": True})
                    return {"status": "success", "ttft_ms": result.ttft_ms, "total_latency_ms": result.total_latency_ms}

                if execution:
                    selected_model = execution.before_attempt(phase, iteration, warmup)
                stage = "repeat" if execution and execution.has_inference else "unknown_before_attach" if execution and execution.externally_managed else "first_group_after_load" if execution else "unknown"
                batch_id = uuid.uuid4().hex
                cohort = []
                info = {}
                samples = [{"id": key, "prompt": prompt, "metadata": {"target_input_tokens": length,
                           "input_length_method": WORKLOAD_VERSION, "prompt_sha256": prompt_hash,
                           "concurrency_requested": concurrency, "concurrency_actual": len(pending),
                           "batch_id": batch_id, "phase": phase, "iteration": iteration}} for key in pending]
                print(f"  - input≈{length} tok / concurrency={len(pending)}/{concurrency} / {phase} #{iteration}")

                def operation(sample):
                    sample["started_at"] = _utc_iso_now()
                    sample["start"] = now_fn()
                    try:
                        return client(api_base=config.api_base, model=selected_model, prompt_text=prompt,
                                      **config.request_parameters(), timeout_sec=config.runs.timeout_sec, now_fn=now_fn)
                    finally:
                        sample["end"] = now_fn()

                def completed(sample, result, error, audit):
                    if error:
                        record = _empty_record(phase=phase, iteration=iteration,
                            started_at=sample["started_at"], status=_status_from_exception(error), error=str(error), prompt_text=prompt)
                    else:
                        record = {**result.to_dict(), "status": "success", "error": None,
                                  "turn_usage": _turn_usage_from_result(result, prompt_text=prompt)}
                    record.update(sample["metadata"], phase=phase, iteration=iteration,
                                  sample_id=sample["id"], started_at=sample["started_at"],
                                  request_started_perf=sample["start"], request_ended_perf=sample["end"],
                                  attempt_wall_ms=(sample["end"] - sample["start"]) * 1000,
                                  inspect={**audit, "status": "error" if error else "success"})
                    record["metrics"] = measured_metrics(record)
                    record["inference_requests"] = [inference_observation(record, status=record["status"], turn=1)]
                    record["memory"] = {**memory.data, "sampling_complete": False}
                    record["measurement"] = {"inference_stage": stage, "cache_state": record["metrics"]["cache_state"]}
                    log = {"kind": "performance_sample", "run_id": run_id,
                           "request": {"model": selected_model, "prompt_text": prompt, **config.recorded_request_parameters()},
                           "response": dict(record), "inspect": record["inspect"]}
                    if execution:
                        execution.save_unit(phase, iteration, sample["id"], record, log)
                    records.append(record)
                    attempts.append({"record_index": len(records) - 1, "payload": log})
                    cohort.append(record)

                with memory_factory(config.performance.memory_pid) as memory:
                    info = run_samples(config=config, model=selected_model, samples=samples,
                        operation=operation, on_complete=completed,
                        log_dir=unit_log_directory(config, run_id, phase, iteration, batch_id),
                        metadata={"run_id": run_id, "batch_id": batch_id, "phase": phase, "iteration": iteration})
                batch = cohort_metrics(cohort, len(pending))
                for record in cohort:
                    updates = {"batch_metrics": batch, "memory": {**memory.data, "sampling_complete": True},
                               "inspect": {**info, "sample_id": record["sample_id"], "status": "success" if record["status"] == "success" else "error"}}
                    record.update(updates)
                    if execution:
                        execution.annotate_unit(phase, iteration, record["sample_id"], updates)
                if len(cohort) != len(pending):
                    raise RuntimeError("Inspect left uncompleted samples; resume this run")
                if config.runs.cooldown_sec:
                    sleep_fn(config.runs.cooldown_sec)

    # Re-read durable annotations into the exported attempt payloads.
    for attempt in attempts:
        record = records[attempt["record_index"]]
        attempt.update(phase=record["phase"], iteration=record["iteration"])
        attempt["payload"]["response"] = dict(record)
        attempt["payload"]["inspect"] = record.get("inspect")
    result = {"run_id": run_id, "started_at": started_at, "ended_at": _utc_iso_now(),
              "duration_sec": now_fn() - began, "model": requested_model or model, "api_model": selected_model,
              "provider": config.provider, "api_base": config.api_base, "evaluation": evaluation_identity(config),
              "benchmark_mode": "performance", "benchmark_id": "runtime-performance-v1",
              "benchmark_title": "入力長別・同時実行性能", "prompt_text": "入力長別の共通Pythonコーパス",
              "performance": {**asdict(config.performance), "input_length_method": WORKLOAD_VERSION},
              "request": config.recorded_request_parameters(), "runs": asdict(config.runs),
              "records": records, "_log_bundle": {"console_lines": [], "attempts": attempts}}
    result["summary"] = compute_run_summary(result)
    return result
