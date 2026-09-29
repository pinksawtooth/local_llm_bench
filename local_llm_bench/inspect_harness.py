"""Inspect evaluation at the durable trial/question boundary.

Lifecycle, streaming clocks and same-run recovery stay outside Inspect. Only
explicit provider request parameters reach the existing transport adapters.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any, Callable
import uuid

from .inspect_log import LOG_FORMAT, completed_log_info

INSPECT_VERSION = "0.3.263"
PROFILE_VERSION = "local-bench-inspect-v1"


def dependency_lock_hash() -> str:
    return hashlib.sha256((Path(__file__).resolve().parent.parent / "requirements.lock").read_bytes()).hexdigest()


def verify_worker_lock(expected: str) -> None:
    if not expected or expected != dependency_lock_hash():
        raise RuntimeError("Inspect dependency lock differs from the worker image; rebuild with a new tag")


def require_inspect() -> None:
    try:
        version = importlib.metadata.version("inspect-ai")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError("Inspect AI が必要です。専用venvで pip install --require-hashes -r requirements.lock を実行してください。") from None
    if version != INSPECT_VERSION:
        raise RuntimeError(f"inspect-ai=={INSPECT_VERSION} が必要です（現在: {version}）。")


def unit_log_directory(config, run_id: str, phase: str, iteration: int, question_id: str) -> Path:
    # IDs are untrusted dataset input. Hash them instead of using them as paths.
    key = hashlib.sha256(f"{phase}:{iteration}:{question_id}".encode()).hexdigest()[:20]
    directory = config.output.run_logs_dir / run_id / "inspect" / key / uuid.uuid4().hex
    directory.mkdir(parents=True, mode=0o700)
    return directory.resolve()


def eval_options(directory: Path) -> dict:
    # An explicit format overrides INSPECT_LOG_FORMAT, including in Docker.
    # The pinned native recorder compresses and deduplicates incrementally.
    return dict(log_dir=str(directory), log_format=LOG_FORMAT, log_model_api=False,
                log_realtime=False, log_buffer=1,
                max_samples=1, max_tasks=1, max_connections=1, epochs=1,
                retry_on_error=0, fail_on_error=False, ctl_server=False)


def model_usage(raw: dict):
    from inspect_ai.model import ModelUsage
    raw = {**(raw.get("raw_usage") or {}), **raw}
    if raw.get("prompt_tokens") is None or raw.get("completion_tokens") is None:
        return None
    cached = (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
    return ModelUsage(input_tokens=max(raw["prompt_tokens"] - (cached or 0), 0),
                      input_tokens_cache_read=cached, output_tokens=raw["completion_tokens"],
                      total_tokens=raw.get("total_tokens") or raw["prompt_tokens"] + raw["completion_tokens"],
                      reasoning_tokens=(raw.get("completion_tokens_details") or {}).get("reasoning_tokens"))


def run_samples(*, config, model: str, samples: list[dict], operation: Callable,
                on_complete: Callable, log_dir: Path, metadata: dict) -> dict:
    """One synchronized cohort, scheduled and logged as Inspect samples.

    The blocking streaming adapter runs in an AnyIO worker thread. Its cancellation
    is shielded until the socket call finishes, so model cleanup cannot race it.
    Completion callbacks run serially on the evaluation loop and fsync each unit.
    """
    require_inspect()
    import anyio
    log_dir = log_dir.resolve()
    from contextvars import ContextVar
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageAssistant, ChatCompletionChoice, GenerateConfig, Model, ModelAPI, ModelOutput, modelapi
    from inspect_ai.solver import solver
    from .conditions import evaluation_identity

    active_sample = ContextVar("performance_sample")
    by_id = {sample["id"]: sample for sample in samples}
    called = set()
    ready = None
    info = {**evaluation_identity(config), "log_dir": str(log_dir),
            "max_samples": len(samples), "max_connections": len(samples)}
    interrupted = False

    @modelapi(name="local-bench-performance")
    class PerformanceAPI(ModelAPI):
        async def generate(self, input, tools, tool_choice, config):
            nonlocal ready, interrupted
            state = active_sample.get()
            key = str(state.sample_id)
            if key in called:
                raise RuntimeError("Inspect attempted to repeat a measured sample")
            called.add(key)
            if ready is None:
                ready = anyio.Event()
            if len(called) == len(samples):
                ready.set()
            await ready.wait()
            result, error = None, None
            try:
                result = await anyio.to_thread.run_sync(operation, by_id[key])
                raw = result.to_dict()
                from .metrics import measured_metrics
                state.metadata["metrics"] = measured_metrics(raw)
                state.metadata["raw_usage"] = raw.get("raw_usage") or {}
                state.metadata["raw_timings"] = raw.get("raw_timings") or {}
                stop = "max_tokens" if raw.get("finish_reason") == "length" else "stop"
                usage = {**(raw.get("raw_usage") or {}), **{k: raw.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}}
                return ModelOutput(model=model, metadata={"metrics": state.metadata["metrics"],
                    "raw_usage": state.metadata["raw_usage"], "raw_timings": state.metadata["raw_timings"],
                    "raw_stats": raw.get("raw_stats") or {}}, choices=[ChatCompletionChoice(
                    message=ChatMessageAssistant(content=raw.get("response_text") or raw.get("reasoning_text") or ""),
                    stop_reason=stop)], usage=model_usage(usage))
            except KeyboardInterrupt as exc:
                interrupted = True
                error = exc
                raise RuntimeError("Benchmark interrupted") from None
            except Exception as exc:
                error = exc
                raise
            finally:
                # A cancelled in-flight request stays pending; successful siblings
                # remain durable and will not be sent again during recovery.
                if result is not None or (error is not None and not isinstance(error, KeyboardInterrupt)):
                    on_complete(by_id[key], result, error, {**info, "sample_id": key})

    @solver
    def measured_generate():
        async def solve(state, generate):
            token = active_sample.set(state)
            try:
                state = await generate(state)
                return state
            finally:
                active_sample.reset(token)
        return solve

    native = Model(PerformanceAPI(model_name=model), GenerateConfig(
        max_retries=0, max_connections=len(samples), cache=False))
    task = Task(name="local_llm_performance", version=PROFILE_VERSION,
                dataset=[Sample(id=s["id"], input=s["prompt"], metadata=s["metadata"]) for s in samples],
                solver=measured_generate(), metadata={**metadata, "provider": config.provider,
                    "request": config.recorded_request_parameters(), "evaluation": info})
    options = {**eval_options(log_dir), "max_samples": len(samples), "max_connections": len(samples)}
    logs = eval(task, model=native, score=False, display="none", **options)
    if not logs:
        if interrupted:
            raise KeyboardInterrupt("Inspect evaluation interrupted")
        raise RuntimeError("Inspect returned no evaluation log")
    log = logs[0]
    info.update(completed_log_info(log, log_dir))
    if interrupted or log.status == "cancelled":
        raise KeyboardInterrupt("Inspect evaluation interrupted")
    if len(called) != len(samples):
        raise RuntimeError("Inspect did not execute all planned samples")
    return info


def run_unit(*, config, model: str, operation: Callable[[], Any], log_dir: Path,
             metadata: dict, question=None, info: dict | None = None):
    """One Inspect sample, one provider trial; gold never enters the worker.

    `info` is populated even on failure so the checkpoint keeps its audit path.
    Transport timing is returned untouched (Inspect overhead is host wall time).
    """
    require_inspect()
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import (ChatMessageAssistant, ChatCompletionChoice,
                                 GenerateConfig, Model, ModelAPI, ModelOutput, modelapi)
    from inspect_ai.scorer import Score, mean, scorer
    from inspect_ai.solver import solver
    from .conditions import evaluation_identity
    from .docker_task.scorer import score_answer

    info = info if info is not None else {}
    info.update(evaluation_identity(config), log_dir=str(log_dir), max_samples=1, max_connections=1)
    result = None
    failure = None
    calls = 0

    def measurement_metadata(raw):
        from .metrics import measured_metrics
        return {"metrics": raw.get("metrics") or measured_metrics(raw),
                **{key: raw[key] for key in ("inference_requests", "raw_usage", "raw_timings", "raw_stats") if key in raw}}

    @solver
    def measured_trial():
        async def solve(state, generate):
            try:
                return await generate(state)
            finally:
                if result is not None:
                    raw = result if isinstance(result, dict) else result.to_dict()
                    state.metadata.update(measurement_metadata(raw))
        return solve

    @modelapi(name="local-bench-trial")
    class TrialAPI(ModelAPI):
        async def generate(self, input, tools, tool_choice, config):
            nonlocal result, failure, calls
            calls += 1
            if calls != 1:
                raise RuntimeError("Inspect attempted to repeat a measured trial")
            try:
                result = operation()
                raw = result if isinstance(result, dict) else result.to_dict()
                if raw.get("status", "success") != "success":
                    raise RuntimeError(str(raw.get("error") or "worker failed"))
                output = str(raw.get("predicted_answer") if question else raw.get("response_text", ""))
                reason = raw.get("finish_reason")
                stop_reason = "max_tokens" if reason == "length" else reason if reason in {"stop", "content_filter", "model_length"} else "unknown"
                return ModelOutput(model=model, metadata=measurement_metadata(raw), choices=[ChatCompletionChoice(
                    message=ChatMessageAssistant(content=output), stop_reason=stop_reason)],
                    usage=model_usage(raw))
            except KeyboardInterrupt as exc:
                # Unwind Inspect's async task group as an ordinary sample
                # failure before propagating interruption to the run manager.
                # Raising KeyboardInterrupt inside a nested event loop can
                # leave the next evaluation waiting on cancelled tasks.
                failure = exc
                raise RuntimeError("Benchmark interrupted") from None
            except Exception as exc:
                failure = exc
                raise

    @scorer(metrics=[mean()])
    def host_typed():
        async def score(state, target):
            # A closure, not Sample.target or worker request metadata. The
            # answer key is available only after the Docker process has exited.
            scored = score_answer(question.answer_type, result.get("predicted_answer"), question.gold_answer)
            return Score(value=scored.score, answer=str(result.get("predicted_answer")),
                         explanation=scored.reason or "host deterministic scorer")
        return score

    generation = GenerateConfig(max_retries=0, max_connections=1, cache=False)
    native_model = Model(TrialAPI(model_name=model), generation)
    task = Task(name="local_llm_bench", version=PROFILE_VERSION,
                dataset=[Sample(id=question.id if question else "prompt",
                                input=question.prompt if question else config.prompt_text)],
                solver=measured_trial(), scorer=host_typed() if question else None,
                metadata={**metadata, "provider": config.provider,
                          "evaluation": evaluation_identity(config),
                          "request": config.recorded_request_parameters()})
    logs = eval(task, model=native_model, score=question is not None, display="none", **eval_options(log_dir))
    if not logs:
        raise RuntimeError("Inspect returned no evaluation log")
    log = logs[0]
    info.update(completed_log_info(log, log_dir))
    if isinstance(failure, KeyboardInterrupt):
        info.update(status="cancelled", error="Benchmark interrupted")
        raise failure
    if log.status == "cancelled":
        raise KeyboardInterrupt("Inspect evaluation interrupted")
    samples = log.samples or []
    error = log.error or next((sample.error for sample in samples if sample.error), None)
    if error:
        info["error"] = error.message
    if failure is not None:
        info.update(status="error", error=str(failure))
    if question and samples and samples[0].scores:
        info["score"] = samples[0].scores["host_typed"].value
    if failure is not None and not isinstance(result, dict):
        raise failure
    if result is None or (error and failure is None) or log.status != "success":
        if isinstance(result, dict) and result.get("status") != "success":
            return result
        raise RuntimeError(info.get("error") or f"Inspect evaluation {log.status}")
    return result
