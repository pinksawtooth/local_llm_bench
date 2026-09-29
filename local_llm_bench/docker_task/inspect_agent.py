"""Inspect ReAct over the existing provider transports, without an answer key."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import time
from typing import Any

from ..inspect_harness import (PROFILE_VERSION, eval_options, model_usage,
                               require_inspect)
from ..inspect_log import LOG_FORMAT, completed_log_info
from ..metrics import aggregate_inference_metrics, measured_metrics
from ..persistence import atomic_write_json
from ..telemetry import build_turn_usage_record, prompt_breakdown_from_messages


_SAVED_SETTINGS_PROVIDERS = {"omlx_saved": ("omlx", "oMLX"), "mlx_serve_saved": ("mlx_serve", "mlx-serve")}


def request_parameters(payload: dict) -> dict:
    source = payload.get("settings_source")
    if source in _SAVED_SETTINGS_PROVIDERS:
        provider, label = _SAVED_SETTINGS_PROVIDERS[source]
        if payload.get("provider") != provider:
            raise ValueError(f"{label} saved settings require provider={provider}")
        return {}
    if source == "lmstudio_saved":
        if payload.get("provider") in {provider for provider, _ in _SAVED_SETTINGS_PROVIDERS.values()}:
            raise ValueError("oMLX / mlx-serve cannot use LM Studio saved settings")
        return {}
    values = {key: payload[key] for key in
              ("temperature", "max_tokens", "top_p", "top_k", "min_p", "seed", "reasoning_effort")
              if payload.get(key) is not None}
    if payload.get("provider") == "ds4":
        from ..ds4 import apply_ds4_sampling
        return apply_ds4_sampling(values, payload.get("ds4_sampling"))
    if payload.get("provider") in {"omlx", "mlx_serve"}:
        if source not in (None, ""):
            raise ValueError("Invalid settings_source for " + payload["provider"])
        if payload["provider"] == "omlx":
            from ..omlx import validate_omlx_request
            validate_omlx_request(values)
        else:
            from ..mlx_serve import validate_mlx_serve_request
            validate_mlx_serve_request(values)
    return values


def wire_messages(messages, originals: dict) -> list[dict]:
    """Preserve raw assistant reasoning and tool arguments across DS4 turns."""
    result = []
    for message in messages:
        if message.id in originals:
            original = copy.deepcopy(originals[message.id])
            # Some compatible APIs omit role in their response message. The
            # next request still needs the role tracked by Inspect.
            original["role"] = message.role
            result.append(original)
        elif message.role == "tool":
            result.append({"role": "tool", "tool_call_id": message.tool_call_id,
                           "content": message.error.message if message.error else message.text})
        else:
            result.append({"role": message.role, "content": message.text})
    return result


async def run_agent(*, payload: dict, messages: list[dict], tools_schema: list[dict],
                    tool_sessions: dict, completion, trace: dict, log_dir: Path) -> dict:
    require_inspect()
    from inspect_ai import Task, eval_async
    from inspect_ai.agent import react
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ContentReasoning, ContentText
    from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem, ChatMessageUser,
                                 ChatCompletionChoice, GenerateConfig, Model, ModelAPI, ModelOutput, modelapi)
    from inspect_ai.tool import ToolCall, ToolDef, ToolParams
    from .container_worker import (_extract_final_answer, _extract_message_text,
                                   _extract_reasoning_text, _tool_result_to_text, _error_payload)

    limits = payload["inspect"]
    max_turns, max_tools = limits.get("max_turns"), limits.get("max_tool_calls")
    started = time.perf_counter()
    deadline = started + float(payload["timeout_sec"])
    originals: dict = {}
    responses: list[str] = []
    reasoning: list[str] = []
    usages: list[dict] = []
    observations: list[dict] = []
    calls = 0
    total_tools = 0
    prediction = None
    finish_reason = None
    fatal = None
    trace["turn_usage"] = []
    trace["turn_limit"] = max_turns
    trace["harness"] = "inspect"
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    inspect_info = {"profile": PROFILE_VERSION, "log_format": LOG_FORMAT, "log_dir": str(log_dir)}

    def persist():
        # Survives process termination while the next request is in flight.
        atomic_write_json(log_dir / "trace.json", trace)

    def remaining():
        value = deadline - time.perf_counter()
        if value <= 0:
            raise TimeoutError("question time budget exhausted")
        return value

    @modelapi(name="local-bench-provider")
    class ProviderAPI(ModelAPI):
        async def generate(self, input, tools, tool_choice, config):
            nonlocal calls, total_tools, prediction, finish_reason, fatal
            if fatal:
                raise fatal
            if max_turns is not None and calls >= max_turns:
                raise RuntimeError(f"max_turns exhausted ({max_turns})")
            calls += 1
            wire = wire_messages(input, originals)
            body = {"model": payload["model"], "messages": wire, "stream": True,
                    "stream_options": {"include_usage": True},
                    **request_parameters(payload)}
            if tools_schema:
                body.update(tools=tools_schema, tool_choice="auto")
            turn = {"turn": calls, "request": copy.deepcopy(body), "tool_events": []}
            if payload.get("settings_source"):
                turn["request"]["settings_source"] = payload["settings_source"]
            if payload.get("sampling_source"):
                turn["request"]["sampling_source"] = payload["sampling_source"]
            trace["turns"].append(turn)
            observation = {"turn": calls, "question_id": str(payload["question"]["id"]),
                           "status": "running", "metrics": measured_metrics({})}
            observations.append(observation)
            persist()
            try:
                raw, latency = completion(body=body, timeout_sec=remaining())
                turn["request_latency_ms"] = latency
                usage = raw.get("usage") or {}
                usages.append(usage)
                metrics = raw.get("metrics") or measured_metrics({**usage, "raw_usage": usage,
                    "raw_timings": raw.get("timings") or {}, "total_latency_ms": latency})
                observation.update(status="success", metrics=metrics, raw_usage=usage,
                                   raw_timings=raw.get("timings") or {}, raw_stats=raw.get("stats") or {})
                turn.update(usage=usage, metrics=metrics, raw_timings=observation["raw_timings"],
                            raw_stats=observation["raw_stats"])
                persist()
                remaining()
                choice = raw["choices"][0]
                message = choice["message"]
                text = _extract_message_text(message)
                reason = _extract_reasoning_text(message)
                tool_calls = []
                for call in message.get("tool_calls") or []:
                    arguments = call["function"]["arguments"]
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    tool_calls.append(ToolCall(id=call["id"], function=call["function"]["name"], arguments=arguments))
                total_tools += len(tool_calls)
                if max_tools is not None and total_tools > max_tools:
                    raise RuntimeError(f"max_tool_calls exhausted ({max_tools})")
                finish_reason = choice.get("finish_reason")
                if not text and not tool_calls:
                    raise RuntimeError("model returned an empty assistant response")
                content = [ContentReasoning(reasoning=reason), ContentText(text=text)] if reason else text
                assistant = ChatMessageAssistant(content=content, tool_calls=tool_calls or None)
                originals[assistant.id] = copy.deepcopy(message)
                responses.append(text)
                reasoning.append(reason)
                turn["response"] = {"assistant_text": text, "reasoning_text": reason,
                                    "tool_calls": message.get("tool_calls", []), "finish_reason": finish_reason}
                trace["turn_usage"].append(build_turn_usage_record(
                    source="inspect", turn_index=calls, prompt_tokens=usage.get("prompt_tokens"),
                    cached_prompt_tokens=turn["metrics"]["cached_prompt_tokens"],
                    completion_tokens=usage.get("completion_tokens"), total_tokens=usage.get("total_tokens"),
                    cumulative_prompt_tokens=sum(u.get("prompt_tokens") or 0 for u in usages),
                    cumulative_completion_tokens=sum(u.get("completion_tokens") or 0 for u in usages),
                    ttft_sec=metrics["ttft_ms"] / 1000 if metrics["ttft_ms"] is not None else None,
                    first_chunk_sec=metrics["ttft_ms"] / 1000 if metrics["ttft_ms"] is not None else None,
                    prefill_sec=metrics["prefill_ms"] / 1000 if metrics["prefill_ms"] is not None else None,
                    decode_sec=metrics["generation_ms"] / 1000 if metrics["generation_ms"] is not None else None,
                    post_first_token_sec=(latency - metrics["ttft_ms"]) / 1000 if metrics["ttft_ms"] is not None else None,
                    timing_sources=metrics["sources"], metrics=metrics,
                    elapsed_sec=latency / 1000, prompt_breakdown=prompt_breakdown_from_messages(wire, tools_schema),
                    question_id=str(payload["question"]["id"])))
                if not tool_calls:
                    prediction = _extract_final_answer(text)
                    if finish_reason == "length":
                        raise RuntimeError("output limit reached before a complete final answer")
                persist()
                return ModelOutput(model=payload["model"], choices=[ChatCompletionChoice(
                    message=assistant, stop_reason="tool_calls" if tool_calls else "stop")], usage=model_usage(usage),
                    metadata=copy.deepcopy(observation))
            except Exception as exc:
                fatal = exc
                observation.update(status="error", error=str(exc))
                turn["error"] = str(exc)
                persist()
                raise

    def wrap_tool(schema):
        function = schema["function"]
        name = function["name"]

        async def execute(**kwargs: Any) -> str:
            nonlocal fatal
            if fatal:
                raise fatal
            event = {"tool_name": name, "arguments": kwargs, "status": "running"}
            trace["turns"][-1]["tool_events"].append(event)
            persist()
            try:
                response = await asyncio.wait_for(tool_sessions[name].call_tool(name, arguments=kwargs),
                                                  min(float(limits["tool_timeout_sec"]), remaining()))
                text = _tool_result_to_text(response)
                event.update(status="error" if getattr(response, "isError", False) else "success", result=text)
                return text
            except Exception as exc:
                fatal = TimeoutError(f"tool '{name}' timed out") if isinstance(exc, TimeoutError) else exc
                event.update(status="error", error=str(fatal))
                raise fatal
            finally:
                persist()

        parameters = ToolParams.model_validate(function["parameters"])
        for key, parameter in parameters.properties.items():
            parameter.description = parameter.description or key
        return ToolDef(execute, name=name, description=function.get("description", ""),
                       parameters=parameters, parallel=False, max_output=0)

    async def on_continue(state):
        if fatal:
            raise fatal
        if prediction is not None:
            return False
        return "回答が確定したら `FINAL_ANSWER: <answer>` の形式で1行だけ返してください。"

    generation = GenerateConfig(max_retries=0, max_connections=1, cache=False)
    model = Model(ProviderAPI(model_name=payload["model"]), generation)
    inputs = [ChatMessageSystem(content=m["content"]) if m["role"] == "system"
              else ChatMessageUser(content=m["content"]) for m in messages]
    task = Task(name="local_bench_react", version=PROFILE_VERSION,
                dataset=[Sample(id=str(payload["question"]["id"]), input=inputs)],
                solver=react(tools=[wrap_tool(s) for s in tools_schema], model=model,
                             prompt=None, submit=False, attempts=1, retry_refusals=0,
                             compaction=None, truncation="disabled",
                             on_continue=on_continue),
                metadata={"profile": PROFILE_VERSION, "limits": limits,
                          "provider": payload["provider"], "request": request_parameters(payload)})
    try:
        logs = await eval_async(task, model=model, score=False, **eval_options(log_dir))
        error = logs[0].error if logs else None
        if logs:
            inspect_info.update(completed_log_info(logs[0], log_dir))
            trace["inspect"] = inspect_info
            persist()
            error = error or next((s.error for s in logs[0].samples or [] if s.error), None)
        if fatal:
            raise fatal
        if not logs or logs[0].status != "success" or error or prediction is None:
            raise RuntimeError(error.message if error else "Inspect did not complete a final answer")
        latency = (time.perf_counter() - started) * 1000
        def total(key):
            return sum(u[key] for u in usages) if usages and all(isinstance(u.get(key), int) for u in usages) else None
        completion_tokens = total("completion_tokens")
        metrics = aggregate_inference_metrics(observations)
        result = {"status": "success", "error": None, "predicted_answer": prediction,
                  "response_text": "\n\n".join(filter(None, responses)),
                  "reasoning_text": "\n\n".join(filter(None, reasoning)),
                  "finish_reason": finish_reason, "total_latency_ms": latency,
                  "prompt_tokens": total("prompt_tokens"), "completion_tokens": completion_tokens,
                  "total_tokens": total("total_tokens"), "initial_prompt_tokens": usages[0].get("prompt_tokens") if usages else None,
                  "conversation_prompt_tokens": total("prompt_tokens"),
                  "end_to_end_tps": completion_tokens / (latency / 1000) if completion_tokens is not None and latency else None,
                  "ttft_ms": metrics["ttft_ms"], "decode_tps": metrics["tg_tps"],
                  "completion_window_ms": metrics["post_first_token_ms"],
                  "metrics": metrics, "inference_requests": observations,
                  "turn_usage": trace["turn_usage"], "trace": trace,
                  "inspect": {**inspect_info, "turns": calls, "tool_calls": total_tools}}
        return result
    except Exception as exc:
        return {**_error_payload("timeout" if isinstance(exc, TimeoutError) else "error", exc, trace=trace),
                "metrics": aggregate_inference_metrics(observations), "inference_requests": observations,
                "inspect": {**inspect_info, "turns": calls, "tool_calls": total_tools}}
