"""Versioned metric definitions. Missing observations are never zero-filled."""
from __future__ import annotations

import math
import statistics
from typing import Any

METRICS_VERSION = 1


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def rate(count, seconds):
    n, duration = number(count), number(seconds)
    return n / duration if n is not None and duration and duration > 0 else None


def mapping(value):
    return value if isinstance(value, dict) else {}


def measured_metrics(raw: dict) -> dict:
    """Normalize client clocks and documented API fields, retaining provenance.

    API prompt_eval_duration is provider-defined (oMLX can use server TTFT).
    It must not silently become an independently measured prefill interval.
    """
    usage = mapping(raw.get("raw_usage"))
    timings = mapping(raw.get("raw_timings"))
    prompt = number(raw.get("prompt_tokens"))
    output = number(raw.get("completion_tokens"))
    elapsed_ms = number(raw.get("total_latency_ms"))
    ttft = number(raw.get("ttft_ms"))
    window_ms = elapsed_ms - ttft if elapsed_ms is not None and ttft is not None and elapsed_ms > ttft else None
    cached = number(mapping(usage.get("prompt_tokens_details")).get("cached_tokens"))
    if cached is not None and (prompt is None or cached > prompt):
        cached = None
    sources = {}
    pp = number(timings.get("prompt_per_second"))
    prefill_ms = number(timings.get("prompt_ms"))
    if pp is None:
        pp = rate(timings.get("prompt_n"), prefill_ms / 1000 if prefill_ms is not None else None)
    if pp is not None:
        sources["pp_tps"] = "api.timings.prompt (processed tokens / prompt interval)"
    else:
        pp = number(usage.get("prompt_tokens_per_second"))
        if pp is not None:
            sources["pp_tps"] = "api.usage.prompt_tokens_per_second (provider-defined interval; may include server TTFT/cache)"
        else:
            pp = rate(prompt, ttft / 1000 if ttft is not None else None)
            sources["pp_tps"] = "client_ttft_estimate (all input tokens / first output chunk latency)" if pp is not None else "unavailable"
    tg = number(timings.get("predicted_per_second"))
    if tg is None:
        duration = number(timings.get("predicted_ms"))
        tg = rate(timings.get("predicted_n"), duration / 1000 if duration is not None else None)
    if tg is not None:
        sources["tg_tps"] = "api.timings.predicted (generated tokens / generation interval)"
    else:
        tg = number(usage.get("generation_tokens_per_second"))
        if tg is not None:
            sources["tg_tps"] = "api.usage.generation_tokens_per_second"
        else:
            tg = rate(output - 1 if output is not None and output > 1 else None,
                      window_ms / 1000 if window_ms is not None else None)
            sources["tg_tps"] = "client_estimate ((output tokens - 1) / post-first-chunk interval)" if tg is not None else "unavailable"
    tpot = window_ms / (output - 1) if window_ms is not None and output is not None and output > 1 else None
    sources.update(ttft_ms="client_first_content_reasoning_or_tool_chunk" if ttft is not None else "unavailable",
                   tpot_ms="client_estimate (chunks are not individual tokens)" if tpot is not None else "unavailable",
                   prefill_ms="api.timings.prompt_ms" if prefill_ms is not None else "unavailable",
                   e2e_ms="client_request_to_stream_end",
                   total_tps="client (input + output) / E2E",
                   input_tokens="api.usage.prompt_tokens" if prompt is not None else "unavailable",
                   output_tokens="api.usage.completion_tokens" if output is not None else "unavailable",
                   cached_prompt_tokens="api.usage.prompt_tokens_details.cached_tokens" if cached is not None else "unavailable")
    return {
        "version": METRICS_VERSION, "scope": "request", "pp_tps": pp, "tg_tps": tg,
        "ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": elapsed_ms,
        "post_first_token_ms": window_ms,
        "prefill_ms": prefill_ms,
        "reported_prompt_eval_ms": number(usage.get("prompt_eval_duration")) * 1000 if number(usage.get("prompt_eval_duration")) is not None else None,
        "reported_generation_ms": number(usage.get("generation_duration")) * 1000 if number(usage.get("generation_duration")) is not None else None,
        "generation_ms": number(timings.get("predicted_ms")),
        "input_tokens": prompt, "output_tokens": output,
        "cached_prompt_tokens": cached,
        "cache_state": "unknown" if cached is None else "hit" if cached else "miss",
        "reasoning_tokens": number(mapping(usage.get("completion_tokens_details")).get("reasoning_tokens")),
        "total_tps": rate(prompt + output if prompt is not None and output is not None else None,
                          elapsed_ms / 1000 if elapsed_ms is not None else None),
        "sources": sources,
    }


def inference_observation(raw: dict, *, status: str = "success", **identity) -> dict:
    """Durable transport observation; excludes prompt text and tool results."""
    return {**identity, "status": status, "metrics": measured_metrics(raw),
            **{key: mapping(raw.get(key)) for key in ("raw_usage", "raw_timings", "raw_stats")}}


def aggregate_inference_metrics(requests: list[dict]) -> dict:
    """Summarize a trial from its requests, never from tool/task wall time.

    Rates and first-token/inter-token latency use per-request medians. Token
    counts and inference E2E are sums. Missing requests/fields stay missing;
    coverage and original sources remain inspectable alongside raw requests.
    Always flatten requests across questions before calling (no nested medians).
    """
    result = measured_metrics({})
    result.update(scope="inference_requests", request_count=len(requests))
    medians = {"pp_tps", "tg_tps", "ttft_ms", "tpot_ms"}
    fields = medians | {"input_tokens", "output_tokens", "e2e_ms", "prefill_ms", "post_first_token_ms",
        "generation_ms", "reported_prompt_eval_ms", "reported_generation_ms",
        "cached_prompt_tokens", "reasoning_tokens"}
    coverage, source_details = {}, {}
    metrics = [mapping(r.get("metrics")) for r in requests]
    for field in sorted(fields):
        values = [value for m in metrics if (value := number(m.get(field))) is not None]
        coverage[field] = len(values)
        source_details[field] = sorted({str(mapping(m.get("sources")).get(field, "unavailable")) for m in metrics})
        complete = bool(requests) and len(values) == len(requests)
        method = "request_median" if field in medians else "request_sum"
        result[field] = (statistics.median(values) if field in medians else sum(values)) if complete else None
        result["sources"][field] = method if complete else "unavailable"
    input_tokens, output_tokens = result["input_tokens"], result["output_tokens"]
    result["total_tps"] = rate(input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None,
                               result["e2e_ms"] / 1000 if result["e2e_ms"] is not None else None)
    result["sources"]["e2e_ms"] = "request_sum (inference only; excludes tool execution)" if result["e2e_ms"] is not None else "unavailable"
    result["sources"]["total_tps"] = "request_totals (input + output) / inference E2E" if result["total_tps"] is not None else "unavailable"
    cached = result["cached_prompt_tokens"]
    result["cache_state"] = "unknown" if cached is None else "hit" if cached else "miss"
    result["aggregation"] = {"latency_and_rate": "request_median", "tokens_and_e2e": "request_sum",
                             "observed_requests": coverage}
    result["source_details"] = source_details
    return result
