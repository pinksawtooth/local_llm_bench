from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..config import BenchmarkConfig
from ..error_utils import merge_excerpts
from ..stats import compute_run_summary
from .scorer import score_answer
from .spec import BenchmarkSpec, Question, load_spec
from .targets import build_shared_system_prompt, build_task_prompt, resolve_native_binary_target

_DOCKER_GHIDRA_MCP_SOURCE_PATH_ENV = "LOCAL_LLM_BENCH_DOCKER_GHIDRA_MCP_SOURCE_PATH"
_DOCKER_GHIDRA_MCP_SOURCE_ENV = "LOCAL_LLM_BENCH_DOCKER_GHIDRA_MCP_SOURCE_ROOT"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARCH_ALIASES = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "arm64/v8": "arm64",
}


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_ghidra_mcp_source_root(path: Path) -> Optional[Path]:
    candidate = path.expanduser().resolve()
    if candidate.is_dir() and (candidate / "ghidra_mcp").is_dir():
        return candidate
    if candidate.is_dir() and candidate.name == "ghidra_mcp" and (candidate / "__init__.py").is_file():
        return candidate.parent
    return None


def _resolve_local_ghidra_mcp_source_root() -> Optional[Path]:
    for env_name in (_DOCKER_GHIDRA_MCP_SOURCE_PATH_ENV, "REV_BENCH_DOCKER_GHIDRA_MCP_SOURCE_PATH"):
        by_env = os.getenv(env_name)
        if by_env and by_env.strip():
            normalized = _normalize_ghidra_mcp_source_root(Path(by_env.strip()))
            if normalized is not None:
                return normalized

    try:
        import ghidra_mcp.presentation.cli as ghidra_cli
    except Exception:
        return None

    source_file = inspect.getsourcefile(ghidra_cli)
    if not source_file:
        return None
    return _normalize_ghidra_mcp_source_root(Path(source_file).resolve().parents[2])


def _docker_binary() -> str:
    docker = shutil.which("docker")
    if docker:
        return docker
    for candidate in ("/usr/local/bin/docker", "/opt/homebrew/bin/docker", "/usr/bin/docker"):
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("docker コマンドが見つかりません。")


def _normalize_docker_arch(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return _ARCH_ALIASES.get(normalized, normalized)


def _parse_platform(value: str) -> tuple[str, str] | None:
    normalized = str(value or "").strip().lower()
    if not normalized or "/" not in normalized:
        return None
    os_name, arch, *_ = normalized.split("/")
    os_name = os_name.strip()
    arch = _normalize_docker_arch(arch)
    if not os_name or not arch:
        return None
    return os_name, arch


def _inspect_local_docker_image_platform(image: str) -> tuple[str, str] | None:
    image_name = str(image or "").strip()
    if not image_name:
        return None
    completed = subprocess.run(
        [
            _docker_binary(),
            "image",
            "inspect",
            image_name,
            "--format",
            "{{.Os}}/{{.Architecture}}",
        ],
        capture_output=True,
        text=True,
    )
    if getattr(completed, "returncode", 0) != 0:
        return None
    return _parse_platform(completed.stdout.strip())


def _docker_platform_mismatch_error(image: str, requested_platform: str | None) -> str | None:
    requested = _parse_platform(requested_platform or "")
    if requested is None:
        return None
    actual = _inspect_local_docker_image_platform(image)
    if actual is None or actual == requested:
        return None

    requested_text = f"{requested[0]}/{requested[1]}"
    actual_text = f"{actual[0]}/{actual[1]}"
    message = (
        f"docker image '{image}' はローカルに {actual_text} として存在しますが、"
        f"設定は {requested_text} を要求しています。"
    )
    if actual == ("linux", "arm64") and requested == ("linux", "amd64"):
        return (
            message
            + " Apple Silicon 環境なら bench_d_compile_arm64.yaml を使うか、"
            + "同じタグの image を linux/amd64 で再ビルドしてください。"
        )
    return message + " docker.platform をローカル image と合わせるか、指定 platform で image を再ビルドしてください。"


def _stage_docker_ghidra_mcp_source(bootstrap_dir: Path) -> Path:
    source_root = _resolve_local_ghidra_mcp_source_root()
    if source_root is None:
        raise RuntimeError(
            "ghidra_mcp source root を解決できませんでした。"
            f" {_DOCKER_GHIDRA_MCP_SOURCE_PATH_ENV} を設定してください。"
        )
    destination_root = bootstrap_dir / "ghidra_mcp_src"
    shutil.copytree(source_root, destination_root, dirs_exist_ok=True)
    return destination_root


def _copy_question_binary(question: Question, bundle_dir: Path) -> str | None:
    if question.binary_path is None:
        return None
    relative_path = Path(question.binary_ref or Path("data") / question.binary_path.name)
    destination = (bundle_dir / relative_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(question.binary_path, destination)
    return str(relative_path)


def _write_request_payload(
    *,
    bundle_dir: Path,
    config: BenchmarkConfig,
    selected_model: str,
    question: Question,
    staged_binary_ref: str | None,
) -> Path:
    payload = {
        "provider": config.provider,
        "api_base": config.docker_api_base,
        "model": selected_model,
        "temperature": config.request.temperature,
        "max_tokens": config.request.max_tokens,
        "timeout_sec": config.benchmark_question_timeout_sec or config.runs.timeout_sec,
        "ghidra_tool_mode": config.benchmark_ghidra_tool_mode,
        "system_prompt": build_shared_system_prompt(),
        "task_prompt": build_task_prompt(question.prompt, Path(staged_binary_ref) if staged_binary_ref else None),
        "question": {
            "id": question.id,
            "prompt": question.prompt,
            "answer_type": question.answer_type,
            "binary_path": staged_binary_ref,
        },
    }
    request_path = bundle_dir / "request.json"
    request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return request_path


def _should_stage_ghidra_source(question: Question) -> bool:
    resolved = resolve_native_binary_target(question.binary_path)
    try:
        return resolved.path is not None and resolved.resolution in {"direct", "auto_resolved"}
    finally:
        if resolved.cleanup_dir is not None:
            shutil.rmtree(resolved.cleanup_dir, ignore_errors=True)


def _safe_sum(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if isinstance(record.get(field), (int, float))]
    if not values:
        return None
    return sum(values)


def _coerce_question_result(
    question: Question,
    worker_result: dict[str, Any],
) -> dict[str, Any]:
    status = str(worker_result.get("status") or "error")
    predicted_answer = worker_result.get("predicted_answer")
    score = 0.0
    correct = False
    incorrect_count = 0
    error_count = 0
    if status == "success":
        score_result = score_answer(question.answer_type, predicted_answer, question.gold_answer)
        score = score_result.score
        correct = score_result.correct
        incorrect_count = 0 if correct else 1
    else:
        error_count = 1

    return {
        "question_id": question.id,
        "status": status,
        "predicted_answer": predicted_answer,
        "response_text": str(worker_result.get("response_text") or ""),
        "reasoning_text": str(worker_result.get("reasoning_text") or ""),
        "finish_reason": worker_result.get("finish_reason"),
        "error": worker_result.get("error"),
        "ttft_ms": worker_result.get("ttft_ms"),
        "total_latency_ms": worker_result.get("total_latency_ms"),
        "completion_window_ms": worker_result.get("completion_window_ms"),
        "prompt_tokens": worker_result.get("prompt_tokens"),
        "prompt_latency_ms": worker_result.get("prompt_latency_ms"),
        "initial_prompt_tokens": worker_result.get("initial_prompt_tokens"),
        "initial_prompt_latency_ms": worker_result.get("initial_prompt_latency_ms"),
        "initial_prompt_tps": worker_result.get("initial_prompt_tps"),
        "conversation_prompt_tokens": worker_result.get("conversation_prompt_tokens"),
        "conversation_prompt_latency_ms": worker_result.get("conversation_prompt_latency_ms"),
        "conversation_prompt_tps": worker_result.get("conversation_prompt_tps"),
        "completion_tokens": worker_result.get("completion_tokens"),
        "total_tokens": worker_result.get("total_tokens"),
        "decode_tps": worker_result.get("decode_tps"),
        "end_to_end_tps": worker_result.get("end_to_end_tps"),
        "approx_prompt_tps": worker_result.get("approx_prompt_tps"),
        "benchmark_score": score,
        "benchmark_correct": correct,
        "benchmark_correct_count": 1 if correct else 0,
        "benchmark_incorrect_count": incorrect_count,
        "benchmark_error_count": error_count,
        "stderr_excerpt": worker_result.get("stderr_excerpt"),
        "log_path": None,
    }


def _run_question_in_docker(
    *,
    config: BenchmarkConfig,
    selected_model: str,
    question: Question,
    docker_executor: Callable[..., Any],
    docker_env: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="local-llm-bench-docker-") as tmpdir:
        bundle_dir = Path(tmpdir)
        bootstrap_dir = bundle_dir / ".local_llm_bench_bootstrap"
        bootstrap_dir.mkdir(parents=True, exist_ok=True)

        staged_binary_ref = _copy_question_binary(question, bundle_dir)
        request_path = _write_request_payload(
            bundle_dir=bundle_dir,
            config=config,
            selected_model=selected_model,
            question=question,
            staged_binary_ref=staged_binary_ref,
        )
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))

        env = os.environ.copy()
        if _should_stage_ghidra_source(question):
            staged_source_root = _stage_docker_ghidra_mcp_source(bootstrap_dir)
            env[_DOCKER_GHIDRA_MCP_SOURCE_ENV] = "/work/.local_llm_bench_bootstrap/ghidra_mcp_src"
            if not staged_source_root.exists():
                raise RuntimeError("ghidra_mcp source staging に失敗しました。")

        cmd = [
            _docker_binary(),
            "run",
            "--rm",
            "-v",
            f"{bundle_dir}:/work",
            "-v",
            f"{_REPO_ROOT}:/opt/local_llm_bench:ro",
            "-w",
            "/work",
            "-e",
            "HOME=/work/home",
            "-e",
            f"{_DOCKER_GHIDRA_MCP_SOURCE_ENV}={env.get(_DOCKER_GHIDRA_MCP_SOURCE_ENV, '')}",
        ]
        for key, value in sorted((docker_env or {}).items()):
            if not str(key).strip():
                continue
            env[str(key)] = str(value)
            cmd.extend(["-e", str(key)])
        if config.docker_platform:
            cmd.extend(["--platform", config.docker_platform])
        cmd.extend(
            [
                config.docker_image or "",
                "python",
                "-m",
                "local_llm_bench.docker_task.container_worker",
                str(Path("/work") / request_path.name),
            ]
        )
        docker_command = [str(item) for item in cmd]

        if docker_executor is subprocess.run:
            platform_error = _docker_platform_mismatch_error(config.docker_image or "", config.docker_platform)
            if platform_error:
                result = {
                    "status": "error",
                    "predicted_answer": None,
                    "response_text": "",
                    "reasoning_text": "",
                    "finish_reason": None,
                    "error": platform_error,
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
                    "stderr_excerpt": platform_error,
                }
                result["_question_log"] = {
                    "kind": "docker_question",
                    "question_id": question.id,
                    "request": request_payload,
                    "docker_command": docker_command,
                    "stdout": "",
                    "stderr": platform_error,
                    "parsed_worker_result": dict(result),
                }
                return result

        timeout_sec = float(config.benchmark_question_timeout_sec or config.runs.timeout_sec)
        try:
            completed = docker_executor(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec + 10.0,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            result = {
                "status": "timeout",
                "predicted_answer": None,
                "response_text": "",
                "reasoning_text": "",
                "finish_reason": None,
                "error": str(exc),
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
            }
            result["stderr_excerpt"] = None
            result["_question_log"] = {
                "kind": "docker_question",
                "question_id": question.id,
                "request": request_payload,
                "docker_command": docker_command,
                "stdout": "",
                "stderr": "",
                "parsed_worker_result": dict(result),
            }
            return result

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        parsed: dict[str, Any] | None = None
        if stdout:
            try:
                parsed = json.loads(stdout.splitlines()[-1])
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            result = {
                "status": "error",
                "predicted_answer": None,
                "response_text": stdout,
                "reasoning_text": "",
                "finish_reason": None,
                "error": stderr or "worker returned malformed JSON",
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
            }
            result["stderr_excerpt"] = merge_excerpts([stderr])
            result["_question_log"] = {
                "kind": "docker_question",
                "question_id": question.id,
                "request": request_payload,
                "docker_command": docker_command,
                "stdout": stdout,
                "stderr": stderr,
                "parsed_worker_result": dict(result),
            }
            return result
        if getattr(completed, "returncode", 0) != 0 and parsed.get("status") == "success":
            parsed["status"] = "error"
            parsed["error"] = parsed.get("error") or stderr or f"docker exited with {completed.returncode}"
        elif getattr(completed, "returncode", 0) != 0 and not parsed.get("error"):
            parsed["error"] = stderr or f"docker exited with {completed.returncode}"
        parsed["stderr_excerpt"] = merge_excerpts([stderr])
        parsed["_question_log"] = {
            "kind": "docker_question",
            "question_id": question.id,
            "request": request_payload,
            "docker_command": docker_command,
            "stdout": stdout,
            "stderr": stderr,
            "parsed_worker_result": {
                key: value
                for key, value in parsed.items()
                if not str(key).startswith("_")
            },
        }
        return parsed


def _aggregate_attempt_record(
    *,
    phase: str,
    iteration: int,
    started_at: str,
    prompt_text: str,
    question_results: list[dict[str, Any]],
) -> dict[str, Any]:
    correct_count = sum(int(result.get("benchmark_correct_count") or 0) for result in question_results)
    incorrect_count = sum(int(result.get("benchmark_incorrect_count") or 0) for result in question_results)
    error_count = sum(int(result.get("benchmark_error_count") or 0) for result in question_results)
    total_questions = len(question_results)
    statuses = [str(result.get("status") or "error") for result in question_results]

    if "timeout" in statuses:
        status = "timeout"
    elif any(item != "success" for item in statuses):
        status = "error"
    else:
        status = "success"

    predicted_payload = {
        result["question_id"]: result.get("predicted_answer")
        for result in question_results
    }
    predicted_answer = (
        next(iter(predicted_payload.values()), None)
        if len(predicted_payload) == 1
        else json.dumps(predicted_payload, ensure_ascii=False)
    )
    reasoning_text = "\n\n".join(
        f"[{result['question_id']}]\n{result.get('reasoning_text', '')}".strip()
        for result in question_results
        if str(result.get("reasoning_text") or "").strip()
    )
    response_text = "\n\n".join(
        f"[{result['question_id']}]\n{result.get('response_text', '')}".strip()
        for result in question_results
        if str(result.get("response_text") or "").strip()
    )
    error_text = "\n\n".join(
        f"[{result['question_id']}] {result.get('error')}".strip()
        for result in question_results
        if str(result.get("error") or "").strip()
    ) or None
    stderr_excerpt = merge_excerpts([result.get("stderr_excerpt") for result in question_results])

    total_latency_ms = _safe_sum(question_results, "total_latency_ms")
    ttft_ms_values = [float(result["ttft_ms"]) for result in question_results if isinstance(result.get("ttft_ms"), (int, float))]
    ttft_ms = ttft_ms_values[0] if ttft_ms_values else None
    completion_window_ms = (
        max(total_latency_ms - ttft_ms, 0.0)
        if isinstance(total_latency_ms, (int, float)) and isinstance(ttft_ms, (int, float))
        else None
    )
    prompt_tokens = _safe_sum(question_results, "prompt_tokens")
    prompt_latency_ms = _safe_sum(question_results, "prompt_latency_ms")
    initial_prompt_tokens = _safe_sum(question_results, "initial_prompt_tokens")
    initial_prompt_latency_ms = _safe_sum(question_results, "initial_prompt_latency_ms")
    conversation_prompt_tokens = _safe_sum(question_results, "conversation_prompt_tokens")
    conversation_prompt_latency_ms = _safe_sum(question_results, "conversation_prompt_latency_ms")
    completion_tokens = _safe_sum(question_results, "completion_tokens")
    total_tokens = _safe_sum(question_results, "total_tokens")
    decode_tps = (
        completion_tokens / (completion_window_ms / 1000.0)
        if isinstance(completion_tokens, (int, float))
        and isinstance(completion_window_ms, (int, float))
        and completion_window_ms > 0
        else None
    )
    end_to_end_tps = (
        completion_tokens / (total_latency_ms / 1000.0)
        if isinstance(completion_tokens, (int, float))
        and isinstance(total_latency_ms, (int, float))
        and total_latency_ms > 0
        else None
    )
    approx_prompt_tps = (
        prompt_tokens / (prompt_latency_ms / 1000.0)
        if isinstance(prompt_tokens, (int, float))
        and isinstance(prompt_latency_ms, (int, float))
        and prompt_latency_ms > 0
        else None
    )
    initial_prompt_tps = (
        initial_prompt_tokens / (initial_prompt_latency_ms / 1000.0)
        if isinstance(initial_prompt_tokens, (int, float))
        and isinstance(initial_prompt_latency_ms, (int, float))
        and initial_prompt_latency_ms > 0
        else None
    )
    conversation_prompt_tps = (
        conversation_prompt_tokens / (conversation_prompt_latency_ms / 1000.0)
        if isinstance(conversation_prompt_tokens, (int, float))
        and isinstance(conversation_prompt_latency_ms, (int, float))
        and conversation_prompt_latency_ms > 0
        else None
    )
    benchmark_score = (
        sum(float(result.get("benchmark_score") or 0.0) for result in question_results) / total_questions
        if total_questions
        else 0.0
    )

    return {
        "phase": phase,
        "iteration": iteration,
        "started_at": started_at,
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "completion_window_ms": completion_window_ms,
        "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None,
        "prompt_latency_ms": prompt_latency_ms,
        "initial_prompt_tokens": int(initial_prompt_tokens) if isinstance(initial_prompt_tokens, (int, float)) else None,
        "initial_prompt_latency_ms": initial_prompt_latency_ms,
        "initial_prompt_tps": initial_prompt_tps,
        "conversation_prompt_tokens": int(conversation_prompt_tokens) if isinstance(conversation_prompt_tokens, (int, float)) else None,
        "conversation_prompt_latency_ms": conversation_prompt_latency_ms,
        "conversation_prompt_tps": conversation_prompt_tps,
        "completion_tokens": int(completion_tokens) if isinstance(completion_tokens, (int, float)) else None,
        "total_tokens": int(total_tokens) if isinstance(total_tokens, (int, float)) else None,
        "decode_tps": decode_tps,
        "end_to_end_tps": end_to_end_tps,
        "approx_prompt_tps": approx_prompt_tps,
        "finish_reason": next(
            (result.get("finish_reason") for result in reversed(question_results) if result.get("finish_reason")),
            None,
        ),
        "status": status,
        "error": error_text,
        "stderr_excerpt": stderr_excerpt,
        "reasoning_text": reasoning_text,
        "response_text": response_text,
        "prompt_text": prompt_text,
        "predicted_answer": predicted_answer,
        "benchmark_score": benchmark_score,
        "benchmark_correct": correct_count == total_questions and error_count == 0,
        "benchmark_correct_count": correct_count,
        "benchmark_incorrect_count": incorrect_count,
        "benchmark_error_count": error_count,
        "log_path": None,
        "question_results": question_results,
    }


def run_docker_task_benchmark(
    config: BenchmarkConfig,
    *,
    model: str | None = None,
    requested_model: str | None = None,
    docker_executor: Callable[..., Any] = subprocess.run,
    docker_env: Optional[dict[str, str]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.perf_counter,
    spec: BenchmarkSpec | None = None,
) -> Dict[str, Any]:
    selected_model = model or (config.models[0] if len(config.models) == 1 else None)
    if not selected_model:
        raise ValueError("run_docker_task_benchmark には単一モデルを渡してください。")
    display_model = requested_model or selected_model

    benchmark_spec = spec or load_spec(config.benchmark_spec_path, config.benchmark_answer_key_path)
    prompt_text = "\n\n".join(question.prompt for question in benchmark_spec.questions)
    phases = [("cold", idx + 1) for idx in range(config.runs.cold_runs)] + [
        ("warm", idx + 1) for idx in range(config.runs.warm_runs)
    ]

    run_id = uuid.uuid4().hex[:8]
    started_at = _utc_iso_now()
    started_perf = now_fn()
    records: list[dict[str, Any]] = []
    console_lines: list[str] = []
    attempt_logs: list[dict[str, Any]] = []

    def emit(line: str) -> None:
        console_lines.append(line)
        print(line)

    emit(
        f"[Run {run_id}] model={display_model} "
        f"{f'api_model={selected_model} ' if display_model != selected_model else ''}"
        f"provider={config.provider} "
        f"benchmark={benchmark_spec.id} questions={len(benchmark_spec.questions)} "
        f"cold={config.runs.cold_runs} warm={config.runs.warm_runs}"
    )

    emit(f"[Model] {selected_model}")
    for phase, iteration in phases:
        attempt_started_at = _utc_iso_now()
        emit(f"  - {phase} #{iteration} ...")
        question_results: list[dict[str, Any]] = []
        question_log_entries: list[dict[str, Any]] = []
        for question in benchmark_spec.questions:
            emit(f"    * question={question.id}")
            raw_result = _run_question_in_docker(
                config=config,
                selected_model=selected_model,
                question=question,
                docker_executor=docker_executor,
                docker_env=docker_env,
            )
            question_log_entries.append(
                {
                    "question_index": len(question_results),
                    "question_id": question.id,
                    "payload": raw_result.get("_question_log") or {},
                }
            )
            question_results.append(_coerce_question_result(question, raw_result))
        attempt_record = _aggregate_attempt_record(
            phase=phase,
            iteration=iteration,
            started_at=attempt_started_at,
            prompt_text=prompt_text,
            question_results=question_results,
        )
        records.append(attempt_record)
        attempt_logs.append(
            {
                "record_index": len(records) - 1,
                "phase": phase,
                "iteration": iteration,
                "question_logs": question_log_entries,
                "payload": {
                    "kind": "docker_attempt",
                    "run_id": run_id,
                    "model": display_model,
                    "api_model": selected_model,
                    "benchmark_id": benchmark_spec.id,
                    "benchmark_title": benchmark_spec.title,
                    "phase": phase,
                    "iteration": iteration,
                    "started_at": attempt_started_at,
                    "attempt_record": {
                        key: value
                        for key, value in attempt_record.items()
                        if key != "question_results"
                    },
                    "question_results": question_results,
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
        "api_base": config.docker_api_base,
        "model": display_model,
        "api_model": selected_model,
        "prompt_text": prompt_text,
        "benchmark_mode": "docker_task",
        "benchmark_id": benchmark_spec.id,
        "benchmark_title": benchmark_spec.title,
        "question_count": len(benchmark_spec.questions),
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
        "docker": {
            "image": config.docker_image,
            "platform": config.docker_platform,
            "api_base": config.docker_api_base,
            "lmstudio_base_url": config.docker_api_base,
        },
        "config_path": str(config.config_path) if config.config_path else None,
        "records": records,
    }
    run_data["summary"] = compute_run_summary(run_data)
    run_data["_log_bundle"] = {
        "console_lines": console_lines,
        "attempts": attempt_logs,
    }
    return run_data
