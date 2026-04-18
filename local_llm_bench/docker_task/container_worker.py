from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from ..config import UNSLOTH_STUDIO_PROVIDER
from ..unsloth_api import UnslothStudioAuthSession, load_unsloth_auth_from_env
from .ghidra_tool_mode import DEFAULT_GHIDRA_TOOL_MODE, normalize_ghidra_tool_mode
from .targets import (
    build_shared_system_prompt,
    build_task_prompt,
    resolve_native_binary_target,
)


_WORK_DIR = Path(os.getenv("LOCAL_LLM_BENCH_WORK_DIR", "/work")).resolve()
_HOME_DIR = _WORK_DIR / "home"
_BOOTSTRAP_DIR = _WORK_DIR / ".local_llm_bench_bootstrap"
_DOCKER_GHIDRA_MCP_SOURCE_ENV = "LOCAL_LLM_BENCH_DOCKER_GHIDRA_MCP_SOURCE_ROOT"
_DOCKER_GHIDRA_SERVER_NAME = "mecha_ghidra"
_DOCKER_PYTHON_SERVER_NAME = "python"
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER:\s*(.+)", re.IGNORECASE)
_DEFAULT_MAX_TURNS = 24


def _ensure_worker_environment() -> None:
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    _HOME_DIR.mkdir(parents=True, exist_ok=True)
    _BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(_HOME_DIR)
    os.chdir(_WORK_DIR)


def _completion_url(api_base: str) -> str:
    trimmed = str(api_base or "").rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def _path_from_payload(raw_value: Any) -> Optional[Path]:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    candidate = Path(raw_value.strip())
    if candidate.is_absolute():
        return candidate
    return (_WORK_DIR / candidate).resolve()


def _load_request(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    return payload


def _read_file_prefix(path: Path, size: int = 4096) -> bytes:
    try:
        if not path.is_file():
            return b""
        with path.open("rb") as handle:
            return handle.read(size)
    except OSError:
        return b""


def _has_native_binary_header(candidate: Path) -> bool:
    prefix = _read_file_prefix(candidate)
    return prefix.startswith(
        (
            b"\x7fELF",
            b"MZ",
            b"\xfe\xed\xfa\xce",
            b"\xce\xfa\xed\xfe",
            b"\xfe\xed\xfa\xcf",
            b"\xcf\xfa\xed\xfe",
            b"\xca\xfe\xba\xbe",
            b"\xbe\xba\xfe\xca",
        )
    )


def _import_binary_into_project_with_analyze_headless(
    *,
    project_location: Path,
    project_name: str,
    binary_path: Path,
    ghidra_install_dir: Path,
) -> str:
    if not binary_path.exists():
        raise FileNotFoundError(f"binary_path が存在しません: {binary_path}")

    support_dir = ghidra_install_dir / "support"
    launcher_name = "analyzeHeadless.bat" if os.name == "nt" else "analyzeHeadless"
    launcher = support_dir / launcher_name
    if not launcher.exists():
        raise FileNotFoundError(f"analyzeHeadless が見つかりません: {launcher}")

    import_path = binary_path
    imported_name = binary_path.name
    if not imported_name.isascii():
        normalized_stem = unicodedata.normalize("NFKD", binary_path.stem)
        ascii_stem = normalized_stem.encode("ascii", "ignore").decode("ascii")
        ascii_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_stem).strip("._-") or "import_target"
        hash_suffix = hashlib.sha256(imported_name.encode("utf-8")).hexdigest()[:8]
        safe_name = f"{ascii_stem}-{hash_suffix}{binary_path.suffix}"
        import_dir = project_location / ".imports"
        import_dir.mkdir(parents=True, exist_ok=True)
        import_path = import_dir / safe_name
        shutil.copy2(binary_path, import_path)
        imported_name = safe_name

    timeout_sec = float(os.getenv("LOCAL_LLM_BENCH_GHIDRA_IMPORT_TIMEOUT_SEC", "300"))
    cmd = [
        str(launcher),
        str(project_location),
        project_name,
        "-import",
        str(import_path),
        "-overwrite",
        "-noanalysis",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            "analyzeHeadless でバイナリのインポートに失敗しました"
            f" (code={result.returncode})\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    return f"/{imported_name}"


def _extract_text_parts(value: Any, *, _seen: Optional[set[int]] = None) -> list[str]:
    if _seen is None:
        _seen = set()
    if isinstance(value, (list, dict)):
        marker = id(value)
        if marker in _seen:
            return []
        _seen.add(marker)

    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_extract_text_parts(item, _seen=_seen))
        return parts
    if isinstance(value, dict):
        parts: list[str] = []
        for field_name in (
            "text",
            "content",
            "value",
            "reasoning",
            "reasoning_content",
            "reasoning_text",
            "output_text",
            "output",
            "summary",
            "summary_text",
        ):
            if field_name not in value:
                continue
            parts.extend(_extract_text_parts(value.get(field_name), _seen=_seen))
        return parts
    return []


def _extract_message_text(message: dict[str, Any]) -> str:
    return "".join(_extract_text_parts(message.get("content")))


def _extract_reasoning_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    for field_name in ("reasoning", "reasoning_content", "reasoning_text", "summary", "summary_text"):
        parts.extend(_extract_text_parts(message.get(field_name)))
    return "".join(parts)


def _extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw = message.get("tool_calls")
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _assistant_message_payload(*, assistant_text: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_text if assistant_text else "",
    }
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _extract_final_answer(text: str) -> Optional[str]:
    matches = _FINAL_ANSWER_RE.findall(text or "")
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer.splitlines()[0].strip() if answer else None


def _tool_result_to_text(result_obj: Any) -> str:
    parts: list[str] = []
    content = getattr(result_obj, "content", None)
    if isinstance(content, list):
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                parts.append(getattr(item, "text", ""))
            elif item_type == "image":
                parts.append(f"[Image: {getattr(item, 'mimeType', 'unknown')}]")
            elif item_type:
                parts.append(f"[{item_type}]")
    if not parts:
        structured = getattr(result_obj, "structuredContent", None)
        if structured is not None:
            try:
                return json.dumps(structured, ensure_ascii=False)
            except TypeError:
                return str(structured)
    text = "".join(parts).strip()
    return text or "Success (no output)"


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except TypeError:
        return value


def _drop_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _drop_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value if item is not None]
    return value


def _flatten_exception_messages(exc: BaseException) -> list[str]:
    if isinstance(exc, BaseExceptionGroup):
        messages: list[str] = []
        for child in exc.exceptions:
            messages.extend(_flatten_exception_messages(child))
        return messages
    text = str(exc).strip()
    if not text:
        return [exc.__class__.__name__]
    if text == exc.__class__.__name__:
        return [text]
    return [f"{exc.__class__.__name__}: {text}"]


def _format_exception_text(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        summary = str(exc).strip() or exc.__class__.__name__
        messages: list[str] = []
        for item in _flatten_exception_messages(exc):
            if item not in messages:
                messages.append(item)
        if len(messages) == 1:
            return messages[0]
        if messages:
            return f"{summary}: {'; '.join(messages[:4])}"
        return summary
    text = str(exc).strip()
    return text or exc.__class__.__name__


async def _close_async_stack_safely(stack: contextlib.AsyncExitStack) -> None:
    try:
        await stack.aclose()
    except BaseException:  # noqa: BLE001
        return


def _tool_specs_to_openai(mcp_tools: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    sessions_by_name: dict[str, Any] = {}
    tool_list = getattr(mcp_tools, "tools", [])
    for tool in tool_list:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name.strip():
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": getattr(tool, "description", None) or "",
                    "parameters": getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}},
                },
            }
        )
    return tools, sessions_by_name


async def _open_mcp_stdio_session(
    *,
    server_command: str,
    args_list: list[str],
    server_env: Optional[dict[str, str]],
    timeout_sec: float,
) -> tuple[contextlib.AsyncExitStack, Any, Any]:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    try:
        server = StdioServerParameters(command=server_command, args=args_list, env=server_env)
        streams = await stack.enter_async_context(stdio_client(server))
        session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
        await asyncio.wait_for(session.initialize(), timeout=timeout_sec)
        tool_list = await asyncio.wait_for(session.list_tools(), timeout=timeout_sec)
        return stack, session, tool_list
    except Exception:
        await _close_async_stack_safely(stack)
        raise


def _chat_completion(
    *,
    api_base: str,
    body: dict[str, Any],
    timeout_sec: float,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    sanitized_body = _drop_none_values(body)
    request = urllib.request.Request(
        _completion_url(api_base),
        data=json.dumps(sanitized_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTPError {exc.code}: {body_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URLError: {exc}") from exc
    latency_ms = max((time.perf_counter() - started) * 1000.0, 0.0)
    if not isinstance(payload, dict):
        raise RuntimeError("chat completion response is not a JSON object")
    return payload, latency_ms


async def _run_question(payload: Dict[str, Any]) -> Dict[str, Any]:
    trace: Dict[str, Any] = {
        "turn_limit": _DEFAULT_MAX_TURNS,
        "server_names": [],
        "system_prompt": "",
        "task_prompt": "",
        "turns": [],
    }
    try:
        question = payload.get("question") or {}
        if not isinstance(question, dict):
            raise ValueError("question must be an object")

        binary_path = _path_from_payload(question.get("binary_path"))
        resolved_target = resolve_native_binary_target(binary_path)
        target_path = resolved_target.path if resolved_target.path is not None else binary_path

        timeout_sec = float(payload.get("timeout_sec") or 300.0)
        api_base = str(payload.get("api_base") or "").rstrip("/")
        if not api_base:
            raise ValueError("api_base is required")
        provider = str(payload.get("provider") or "").strip().lower()
        trace["provider"] = provider
        chat_urlopen: Callable[..., Any] = urllib.request.urlopen
        if provider == UNSLOTH_STUDIO_PROVIDER:
            auth_session = UnslothStudioAuthSession(
                load_unsloth_auth_from_env(),
                openai_api_base=api_base,
            )
            chat_urlopen = auth_session.urlopen

        worker_env = {str(key): str(value) for key, value in os.environ.items()}
        python_server_spec = {
            "server_name": _DOCKER_PYTHON_SERVER_NAME,
            "command": sys.executable,
            "args": ["-m", "local_llm_bench.docker_task.python_mcp_server"],
            "env": dict(worker_env),
        }

        server_specs = [python_server_spec]
        if target_path is not None:
            ghidra_install_dir = Path(os.getenv("GHIDRA_INSTALL_DIR", "/opt/ghidra")).expanduser()
            project_location = _BOOTSTRAP_DIR / "ghidra_project"
            project_name = "local-llm-bench"
            project_location.mkdir(parents=True, exist_ok=True)
            (project_location / f"{project_name}.gpr").touch(exist_ok=True)
            (project_location / f"{project_name}.rep" / "data").mkdir(parents=True, exist_ok=True)
            domain_path = _import_binary_into_project_with_analyze_headless(
                project_location=project_location,
                project_name=project_name,
                binary_path=target_path,
                ghidra_install_dir=ghidra_install_dir,
            )
            ghidra_env = dict(worker_env)
            ghidra_src = os.getenv(_DOCKER_GHIDRA_MCP_SOURCE_ENV, "").strip()
            if ghidra_src:
                existing_pythonpath = ghidra_env.get("PYTHONPATH", "").strip()
                ghidra_env["PYTHONPATH"] = ghidra_src if not existing_pythonpath else f"{ghidra_src}:{existing_pythonpath}"
            server_specs.append(
                {
                    "server_name": _DOCKER_GHIDRA_SERVER_NAME,
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "local_llm_bench.docker_task.ghidra_mcp_server",
                        "--ghidra-tool-mode",
                        normalize_ghidra_tool_mode(payload.get("ghidra_tool_mode"), default=DEFAULT_GHIDRA_TOOL_MODE),
                        "--project-location",
                        str(project_location),
                        "--project-name",
                        project_name,
                        "--domain-path",
                        domain_path,
                        "--target-name",
                        "default",
                        "--ghidra-path",
                        str(ghidra_install_dir),
                        "--transport",
                        "stdio",
                    ],
                    "env": ghidra_env,
                }
            )

        trace["server_names"] = [str(spec["server_name"]) for spec in server_specs]

        tools: list[dict[str, Any]] = []
        tool_sessions: dict[str, Any] = {}
        result: Dict[str, Any] | None = None
        pending_error: Exception | None = None
        try:
            async with contextlib.AsyncExitStack() as stack:
                try:
                    for spec in server_specs:
                        session_stack, session, mcp_tools = await _open_mcp_stdio_session(
                            server_command=str(spec["command"]),
                            args_list=[str(arg) for arg in spec["args"]],
                            server_env=spec["env"],
                            timeout_sec=min(timeout_sec, 60.0),
                        )
                        await stack.enter_async_context(session_stack)
                        raw_tools, _ = _tool_specs_to_openai(mcp_tools)
                        for tool in raw_tools:
                            tools.append(tool)
                            tool_sessions[tool["function"]["name"]] = session

                    system_prompt = str(payload.get("system_prompt") or build_shared_system_prompt())
                    task_prompt = str(
                        payload.get("task_prompt")
                        or build_task_prompt(str(question.get("prompt") or ""), binary_path)
                    )
                    trace["system_prompt"] = system_prompt
                    trace["task_prompt"] = task_prompt
                    messages: list[dict[str, Any]] = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task_prompt},
                    ]

                    total_prompt_tokens = 0
                    total_completion_tokens = 0
                    total_tokens = 0
                    total_request_latency_ms = 0.0
                    initial_prompt_tokens: int | None = None
                    finish_reason: str | None = None
                    reasoning_parts: list[str] = []
                    response_parts: list[str] = []
                    first_response_latency_ms: float | None = None
                    started = time.perf_counter()

                    for turn_index in range(_DEFAULT_MAX_TURNS):
                        body: dict[str, Any] = {
                            "model": payload["model"],
                            "messages": messages,
                            "temperature": float(payload.get("temperature") or 0.0),
                            "max_tokens": int(payload.get("max_tokens") or 1024),
                            "stream": False,
                        }
                        if tools:
                            body["tools"] = tools
                            body["tool_choice"] = "auto"

                        turn_trace: dict[str, Any] = {
                            "turn": turn_index + 1,
                            "request": {
                                "model": body["model"],
                                "temperature": body["temperature"],
                                "max_tokens": body["max_tokens"],
                                "messages": _json_clone(messages),
                                "tool_names": [tool["function"]["name"] for tool in tools],
                            },
                            "tool_events": [],
                        }
                        response_payload, request_latency_ms = _chat_completion(
                            api_base=api_base,
                            body=body,
                            timeout_sec=timeout_sec,
                            urlopen=chat_urlopen,
                        )
                        turn_trace["request_latency_ms"] = request_latency_ms
                        if isinstance(request_latency_ms, (int, float)) and request_latency_ms > 0:
                            total_request_latency_ms += float(request_latency_ms)
                        if first_response_latency_ms is None:
                            first_response_latency_ms = request_latency_ms

                        usage = response_payload.get("usage")
                        if isinstance(usage, dict):
                            current_prompt_tokens = int(usage.get("prompt_tokens") or 0)
                            total_prompt_tokens += current_prompt_tokens
                            if initial_prompt_tokens is None and current_prompt_tokens > 0:
                                initial_prompt_tokens = current_prompt_tokens
                            total_completion_tokens += int(usage.get("completion_tokens") or 0)
                            total_tokens += int(usage.get("total_tokens") or 0)
                        turn_trace["usage"] = _json_clone(usage)

                        choices = response_payload.get("choices")
                        if not isinstance(choices, list) or not choices:
                            raise RuntimeError("chat completion response does not include choices")
                        choice = choices[0]
                        if not isinstance(choice, dict):
                            raise RuntimeError("chat completion choice must be an object")
                        message = choice.get("message")
                        if not isinstance(message, dict):
                            raise RuntimeError("chat completion choice.message must be an object")

                        finish_reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else finish_reason
                        assistant_text = _extract_message_text(message)
                        reasoning_text = _extract_reasoning_text(message)
                        if reasoning_text:
                            reasoning_parts.append(reasoning_text)
                        if assistant_text:
                            response_parts.append(assistant_text)

                        tool_calls = _extract_tool_calls(message)
                        turn_trace["response"] = {
                            "finish_reason": finish_reason,
                            "assistant_text": assistant_text,
                            "reasoning_text": reasoning_text,
                            "tool_calls": _json_clone(tool_calls),
                        }
                        messages.append(
                            _assistant_message_payload(
                                assistant_text=assistant_text,
                                tool_calls=tool_calls,
                            )
                        )

                        if tool_calls:
                            for tool_call in tool_calls:
                                function_block = tool_call.get("function") or {}
                                tool_name = str(function_block.get("name") or "").strip()
                                raw_arguments = function_block.get("arguments")
                                try:
                                    tool_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) and raw_arguments.strip() else {}
                                except json.JSONDecodeError:
                                    tool_arguments = {}
                                tool_event = {
                                    "tool_call_id": tool_call.get("id"),
                                    "tool_name": tool_name,
                                    "arguments": _json_clone(tool_arguments),
                                }
                                session = tool_sessions.get(tool_name)
                                if session is None:
                                    tool_text = f"Tool '{tool_name}' is not available."
                                    tool_event["status"] = "missing"
                                    tool_event["result"] = tool_text
                                else:
                                    try:
                                        tool_result = await asyncio.wait_for(
                                            session.call_tool(tool_name, arguments=tool_arguments),
                                            timeout=timeout_sec,
                                        )
                                        tool_text = _tool_result_to_text(tool_result)
                                        tool_event["status"] = "success"
                                        tool_event["result"] = tool_text
                                    except asyncio.TimeoutError as exc:
                                        tool_event["status"] = "timeout"
                                        tool_event["error"] = f"tool '{tool_name}' timed out"
                                        turn_trace["tool_events"].append(tool_event)
                                        trace["turns"].append(turn_trace)
                                        raise TimeoutError(f"tool '{tool_name}' timed out") from exc
                                turn_trace["tool_events"].append(tool_event)
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.get("id"),
                                        "content": tool_text,
                                    }
                                )
                            trace["turns"].append(turn_trace)
                            continue

                        final_answer = _extract_final_answer(assistant_text)
                        turn_trace["final_answer"] = final_answer
                        trace["turns"].append(turn_trace)
                        if final_answer is not None:
                            total_latency_ms = max((time.perf_counter() - started) * 1000.0, 0.0)
                            completion_window_ms = (
                                max(total_latency_ms - first_response_latency_ms, 0.0)
                                if isinstance(first_response_latency_ms, (int, float))
                                else None
                            )
                            decode_tps = (
                                total_completion_tokens / (completion_window_ms / 1000.0)
                                if completion_window_ms and completion_window_ms > 0 and total_completion_tokens > 0
                                else None
                            )
                            end_to_end_tps = (
                                total_completion_tokens / (total_latency_ms / 1000.0)
                                if total_latency_ms > 0 and total_completion_tokens > 0
                                else None
                            )
                            approx_prompt_tps = (
                                total_prompt_tokens / (total_request_latency_ms / 1000.0)
                                if total_request_latency_ms > 0 and total_prompt_tokens > 0
                                else None
                            )
                            initial_prompt_tps = (
                                initial_prompt_tokens / (first_response_latency_ms / 1000.0)
                                if initial_prompt_tokens is not None
                                and isinstance(first_response_latency_ms, (int, float))
                                and first_response_latency_ms > 0
                                else None
                            )
                            result = {
                                "status": "success",
                                "predicted_answer": final_answer,
                                "response_text": "\n\n".join(part for part in response_parts if part).strip(),
                                "reasoning_text": "\n\n".join(part for part in reasoning_parts if part).strip(),
                                "finish_reason": finish_reason,
                                "error": None,
                                "ttft_ms": first_response_latency_ms,
                                "total_latency_ms": total_latency_ms,
                                "completion_window_ms": completion_window_ms,
                                "prompt_tokens": total_prompt_tokens or None,
                                "prompt_latency_ms": total_request_latency_ms or None,
                                "initial_prompt_tokens": initial_prompt_tokens,
                                "initial_prompt_latency_ms": first_response_latency_ms,
                                "initial_prompt_tps": initial_prompt_tps,
                                "conversation_prompt_tokens": total_prompt_tokens or None,
                                "conversation_prompt_latency_ms": total_request_latency_ms or None,
                                "conversation_prompt_tps": approx_prompt_tps,
                                "completion_tokens": total_completion_tokens or None,
                                "total_tokens": total_tokens or None,
                                "decode_tps": decode_tps,
                                "end_to_end_tps": end_to_end_tps,
                                "approx_prompt_tps": approx_prompt_tps,
                                "trace": trace,
                            }
                            break

                        if turn_index == _DEFAULT_MAX_TURNS - 1:
                            break
                        messages.append(
                            {
                                "role": "user",
                                "content": "回答が確定したら `FINAL_ANSWER: <answer>` の形式で1行だけ返してください。",
                            }
                        )
                    if result is None:
                        pending_error = RuntimeError("model did not return FINAL_ANSWER within the maximum number of turns")
                except TimeoutError as exc:
                    pending_error = exc
                except Exception as exc:  # noqa: BLE001
                    pending_error = exc
        except BaseException as exc:  # noqa: BLE001
            trace["cleanup_error"] = _format_exception_text(exc)

        if result is not None:
            return result
        if pending_error is not None:
            raise pending_error
        if trace.get("cleanup_error"):
            raise RuntimeError(str(trace["cleanup_error"]))
        raise RuntimeError("model did not return FINAL_ANSWER within the maximum number of turns")
    except TimeoutError as exc:
        return _error_payload("timeout", exc, trace=trace)
    except Exception as exc:  # noqa: BLE001
        return _error_payload("error", exc, trace=trace)


async def _run_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return await _run_question(payload)
    finally:
        if _BOOTSTRAP_DIR.exists():
            shutil.rmtree(_BOOTSTRAP_DIR, ignore_errors=True)


def _error_payload(status: str, exc: Exception, *, trace: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = {
        "status": status,
        "predicted_answer": None,
        "response_text": "",
        "reasoning_text": "",
        "finish_reason": None,
        "error": _format_exception_text(exc),
        "ttft_ms": None,
        "total_latency_ms": None,
        "completion_window_ms": None,
        "prompt_tokens": None,
        "prompt_latency_ms": None,
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
    if trace:
        payload["trace"] = trace
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="local_llm_bench docker worker")
    parser.add_argument("request_path", type=Path, help="Request JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _ensure_worker_environment()
    try:
        payload = _load_request(args.request_path)
        result = asyncio.run(_run_request(payload))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result.get("status") == "timeout":
            return 124
        if result.get("status") != "success":
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps(_error_payload("error", exc), ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
