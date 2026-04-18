from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Optional


class LMStudioAPIError(RuntimeError):
    """Raised when the LM Studio API call or stream parsing fails."""


CONTINUATION_PROMPT = (
    "直前の回答の続きを出力してください。"
    "すでに出した文章やコードは繰り返さず、途切れた箇所からそのまま続けてください。"
    "コードブロックが開いている場合は閉じるところまで出力してください。"
)
MAX_CONTINUATION_ROUNDS = 8


@dataclass
class StreamResult:
    response_text: str
    reasoning_text: str
    ttft_ms: Optional[float]
    total_latency_ms: float
    completion_window_ms: Optional[float]
    prompt_tokens: Optional[int]
    initial_prompt_tokens: Optional[int]
    initial_prompt_latency_ms: Optional[float]
    initial_prompt_tps: Optional[float]
    conversation_prompt_tokens: Optional[int]
    conversation_prompt_latency_ms: Optional[float]
    conversation_prompt_tps: Optional[float]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    decode_tps: Optional[float]
    end_to_end_tps: Optional[float]
    approx_prompt_tps: Optional[float]
    finish_reason: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _completion_url(api_base: str) -> str:
    trimmed = api_base.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def _safe_token_count(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized >= 0 else None


def _compute_tps(token_count: Optional[int], window_ms: Optional[float]) -> Optional[float]:
    if token_count is None or token_count <= 0 or window_ms is None or window_ms <= 0:
        return None
    return token_count / (window_ms / 1000.0)


def _sum_optional_int(current: Optional[int], value: Optional[int]) -> Optional[int]:
    if value is None:
        return current
    return value if current is None else current + value


def _sum_optional_float(current: Optional[float], value: Optional[float]) -> Optional[float]:
    if value is None:
        return current
    return value if current is None else current + value


TEXT_FIELD_NAMES = (
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
)
METADATA_FIELD_NAMES = {
    "id",
    "object",
    "index",
    "model",
    "created",
    "role",
    "type",
    "status",
    "finish_reason",
    "logprobs",
    "usage",
    "tool_calls",
    "function_call",
    "arguments",
    "name",
}


def _merge_text(existing: str, addition: str) -> str:
    if not addition:
        return existing
    if not existing:
        return addition
    max_overlap = min(len(existing), len(addition), 512)
    for overlap in range(max_overlap, 0, -1):
        if existing.endswith(addition[:overlap]):
            return existing + addition[overlap:]
    return existing + addition


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
        for field_name in TEXT_FIELD_NAMES:
            if field_name not in value:
                continue
            candidate = value.get(field_name)
            if candidate is None:
                continue
            parts.extend(_extract_text_parts(candidate, _seen=_seen))
        if parts:
            return parts
        for key, candidate in value.items():
            if key in METADATA_FIELD_NAMES or candidate is None:
                continue
            if isinstance(candidate, (dict, list)):
                parts.extend(_extract_text_parts(candidate, _seen=_seen))
        return parts
    return []


def _append_parts(parts: list[str], bucket: list[str], *, now_ts: float, state: dict[str, Any]) -> None:
    if not parts:
        return
    bucket.extend(part for part in parts if part)
    if state["first_output_ts"] is None and any(part for part in parts):
        state["first_output_ts"] = now_ts


def _extract_choice_text(choice: dict[str, Any], field_name: str) -> list[str]:
    field_candidates = [field_name]
    if field_name == "content":
        field_candidates.extend(["text", "output_text", "output"])
    if field_name == "reasoning":
        field_candidates.extend(["reasoning_content", "reasoning_text", "summary"])

    delta = choice.get("delta")
    if isinstance(delta, dict):
        for candidate_name in field_candidates:
            parts = _extract_text_parts(delta.get(candidate_name))
            if parts:
                return parts
    message = choice.get("message")
    if isinstance(message, dict):
        for candidate_name in field_candidates:
            parts = _extract_text_parts(message.get(candidate_name))
            if parts:
                return parts
    for candidate_name in field_candidates:
        parts = _extract_text_parts(choice.get(candidate_name))
        if parts:
            return parts
    return []


def _payload_debug_summary(payload: dict[str, Any]) -> str:
    keys = ",".join(sorted(str(key) for key in payload.keys())[:12])
    excerpt = json.dumps(payload, ensure_ascii=False)[:240]
    return f"keys=[{keys}] payload={excerpt}"


def _process_payload(
    payload: dict[str, Any],
    *,
    now_ts: float,
    content_chunks: list[str],
    reasoning_chunks: list[str],
    state: dict[str, Any],
) -> None:
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            _append_parts(
                _extract_choice_text(choice, "content"),
                content_chunks,
                now_ts=now_ts,
                state=state,
            )
            _append_parts(
                _extract_choice_text(choice, "reasoning"),
                reasoning_chunks,
                now_ts=now_ts,
                state=state,
            )
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                state["finish_reason"] = finish_reason

    if not content_chunks:
        _append_parts(
            _extract_text_parts(payload.get("output_text")),
            content_chunks,
            now_ts=now_ts,
            state=state,
        )
        _append_parts(
            _extract_text_parts(payload.get("output")),
            content_chunks,
            now_ts=now_ts,
            state=state,
        )
        _append_parts(
            _extract_text_parts(payload.get("response")),
            content_chunks,
            now_ts=now_ts,
            state=state,
        )
    if not reasoning_chunks:
        _append_parts(
            _extract_text_parts(payload.get("reasoning_content")),
            reasoning_chunks,
            now_ts=now_ts,
            state=state,
        )
        _append_parts(
            _extract_text_parts(payload.get("reasoning")),
            reasoning_chunks,
            now_ts=now_ts,
            state=state,
        )

    usage = payload.get("usage")
    if isinstance(usage, dict):
        state["prompt_tokens"] = _safe_token_count(usage.get("prompt_tokens"))
        state["completion_tokens"] = _safe_token_count(usage.get("completion_tokens"))
        state["total_tokens"] = _safe_token_count(usage.get("total_tokens"))


def consume_sse_stream(
    lines: Iterable[bytes | str],
    *,
    started_at: float,
    now_fn: Callable[[], float],
) -> StreamResult:
    data_lines: list[str] = []
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    state: dict[str, Any] = {
        "first_output_ts": None,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    def flush_event() -> bool:
        if not data_lines:
            return False
        raw_event = "\n".join(data_lines).strip()
        data_lines.clear()
        if not raw_event:
            return False
        if raw_event == "[DONE]":
            return True
        now_ts = now_fn()
        try:
            payload = json.loads(raw_event)
        except json.JSONDecodeError as exc:
            raise LMStudioAPIError(f"SSE JSONの解析に失敗しました: {raw_event[:200]}") from exc
        if not isinstance(payload, dict):
            raise LMStudioAPIError("SSEイベントがJSONオブジェクトではありません。")
        _process_payload(
            payload,
            now_ts=now_ts,
            content_chunks=content_chunks,
            reasoning_chunks=reasoning_chunks,
            state=state,
        )
        return False

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\r\n")
        if not line:
            if flush_event():
                break
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        flush_event()

    ended_at = now_fn()
    if not content_chunks and not reasoning_chunks:
        raise LMStudioAPIError("empty streamed response")

    ttft_ms = None
    if state["first_output_ts"] is not None:
        ttft_ms = max((state["first_output_ts"] - started_at) * 1000.0, 0.0)
    total_latency_ms = max((ended_at - started_at) * 1000.0, 0.0)

    completion_window_ms = None
    if ttft_ms is not None:
        completion_window_ms = max(total_latency_ms - ttft_ms, 0.0)

    prompt_tokens = state["prompt_tokens"]
    completion_tokens = state["completion_tokens"]
    reasoning_text = "".join(reasoning_chunks)
    response_text = "".join(content_chunks) or reasoning_text

    return StreamResult(
        response_text=response_text,
        reasoning_text=reasoning_text,
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        completion_window_ms=completion_window_ms,
        prompt_tokens=prompt_tokens,
        initial_prompt_tokens=prompt_tokens,
        initial_prompt_latency_ms=ttft_ms,
        initial_prompt_tps=_compute_tps(prompt_tokens, ttft_ms),
        conversation_prompt_tokens=prompt_tokens,
        conversation_prompt_latency_ms=total_latency_ms,
        conversation_prompt_tps=_compute_tps(prompt_tokens, total_latency_ms),
        completion_tokens=completion_tokens,
        total_tokens=state["total_tokens"],
        decode_tps=_compute_tps(completion_tokens, completion_window_ms),
        end_to_end_tps=_compute_tps(completion_tokens, total_latency_ms),
        approx_prompt_tps=_compute_tps(prompt_tokens, ttft_ms),
        finish_reason=state["finish_reason"],
    )


def _non_stream_payload_to_result(
    payload: dict[str, Any],
    *,
    started_at: float,
    ended_at: float,
) -> StreamResult:
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    state: dict[str, Any] = {
        "first_output_ts": started_at,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    _process_payload(
        payload,
        now_ts=ended_at,
        content_chunks=content_chunks,
        reasoning_chunks=reasoning_chunks,
        state=state,
    )
    if not content_chunks and not reasoning_chunks:
        raise LMStudioAPIError(f"empty completion response: {_payload_debug_summary(payload)}")
    total_latency_ms = max((ended_at - started_at) * 1000.0, 0.0)
    response_text = "".join(content_chunks) or "".join(reasoning_chunks)
    reasoning_text = "".join(reasoning_chunks)
    return StreamResult(
        response_text=response_text,
        reasoning_text=reasoning_text,
        ttft_ms=None,
        total_latency_ms=total_latency_ms,
        completion_window_ms=None,
        prompt_tokens=state["prompt_tokens"],
        initial_prompt_tokens=state["prompt_tokens"],
        initial_prompt_latency_ms=None,
        initial_prompt_tps=None,
        conversation_prompt_tokens=state["prompt_tokens"],
        conversation_prompt_latency_ms=total_latency_ms,
        conversation_prompt_tps=_compute_tps(state["prompt_tokens"], total_latency_ms),
        completion_tokens=state["completion_tokens"],
        total_tokens=state["total_tokens"],
        decode_tps=None,
        end_to_end_tps=_compute_tps(state["completion_tokens"], total_latency_ms),
        approx_prompt_tps=None,
        finish_reason=state["finish_reason"],
    )


def _request_json(
    request: urllib.request.Request,
    *,
    timeout_sec: float,
    now_fn: Callable[[], float],
    urlopen: Callable[..., Any],
) -> StreamResult:
    started_at = now_fn()
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            status = getattr(response, "status", None)
            if status is None and hasattr(response, "getcode"):
                status = response.getcode()
            if isinstance(status, int) and status >= 400:
                raise LMStudioAPIError(f"HTTP {status}")
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        message = body[:240] if body else str(exc)
        raise LMStudioAPIError(f"HTTPError {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise LMStudioAPIError(f"接続に失敗しました: {reason}") from exc
    ended_at = now_fn()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LMStudioAPIError(f"JSONの解析に失敗しました: {raw[:200]}") from exc
    if not isinstance(payload, dict):
        raise LMStudioAPIError("completion response がJSONオブジェクトではありません。")
    return _non_stream_payload_to_result(payload, started_at=started_at, ended_at=ended_at)


def _request_once(
    *,
    api_base: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    now_fn: Callable[[], float],
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> StreamResult:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        _completion_url(api_base),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    start_ts = now_fn()
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            status = getattr(response, "status", None)
            if status is None and hasattr(response, "getcode"):
                status = response.getcode()
            if isinstance(status, int) and status >= 400:
                raise LMStudioAPIError(f"HTTP {status}")
            try:
                return consume_sse_stream(response, started_at=start_ts, now_fn=now_fn)
            except LMStudioAPIError as exc:
                if str(exc) != "empty streamed response":
                    raise
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        message = body[:240] if body else str(exc)
        raise LMStudioAPIError(f"HTTPError {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise LMStudioAPIError(f"接続に失敗しました: {reason}") from exc

    fallback_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    fallback_request = urllib.request.Request(
        _completion_url(api_base),
        data=json.dumps(fallback_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    return _request_json(
        fallback_request,
        timeout_sec=timeout_sec,
        now_fn=now_fn,
        urlopen=urlopen,
    )


def stream_chat_completion(
    *,
    api_base: str,
    model: str,
    prompt_text: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    now_fn: Callable[[], float],
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> StreamResult:
    messages = [{"role": "user", "content": prompt_text}]
    result = _request_once(
        api_base=api_base,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        now_fn=now_fn,
        urlopen=urlopen,
    )

    total_response_text = result.response_text
    total_reasoning_text = result.reasoning_text
    total_latency_ms = result.total_latency_ms
    ttft_ms = result.ttft_ms
    prompt_tokens = result.prompt_tokens
    approx_prompt_tps = result.approx_prompt_tps
    initial_prompt_tokens = result.initial_prompt_tokens
    initial_prompt_latency_ms = result.initial_prompt_latency_ms
    initial_prompt_tps = result.initial_prompt_tps
    conversation_prompt_tokens = result.conversation_prompt_tokens
    conversation_prompt_latency_ms = result.conversation_prompt_latency_ms
    completion_tokens = result.completion_tokens
    final_finish_reason = result.finish_reason
    current_completion_tokens = result.completion_tokens or 0

    for _ in range(MAX_CONTINUATION_ROUNDS):
        if final_finish_reason != "length":
            break
        if completion_tokens is None:
            break
        remaining_tokens = max_tokens - current_completion_tokens
        if remaining_tokens <= 0:
            break
        continuation_source = total_response_text or total_reasoning_text
        if not continuation_source.strip():
            break

        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": continuation_source},
            {"role": "user", "content": CONTINUATION_PROMPT},
        ]
        next_result = _request_once(
            api_base=api_base,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=remaining_tokens,
            timeout_sec=timeout_sec,
            now_fn=now_fn,
            urlopen=urlopen,
        )
        total_response_text = _merge_text(total_response_text, next_result.response_text)
        total_reasoning_text = _merge_text(total_reasoning_text, next_result.reasoning_text)
        total_latency_ms += next_result.total_latency_ms
        conversation_prompt_tokens = _sum_optional_int(
            conversation_prompt_tokens,
            next_result.conversation_prompt_tokens,
        )
        conversation_prompt_latency_ms = _sum_optional_float(
            conversation_prompt_latency_ms,
            next_result.conversation_prompt_latency_ms,
        )
        current_completion_tokens += next_result.completion_tokens or 0
        completion_tokens = current_completion_tokens if next_result.completion_tokens is not None else None
        final_finish_reason = next_result.finish_reason

    completion_window_ms = None
    if ttft_ms is not None:
        completion_window_ms = max(total_latency_ms - ttft_ms, 0.0)

    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return StreamResult(
        response_text=total_response_text,
        reasoning_text=total_reasoning_text,
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        completion_window_ms=completion_window_ms,
        prompt_tokens=prompt_tokens,
        initial_prompt_tokens=initial_prompt_tokens,
        initial_prompt_latency_ms=initial_prompt_latency_ms,
        initial_prompt_tps=initial_prompt_tps,
        conversation_prompt_tokens=conversation_prompt_tokens,
        conversation_prompt_latency_ms=conversation_prompt_latency_ms,
        conversation_prompt_tps=_compute_tps(conversation_prompt_tokens, conversation_prompt_latency_ms),
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        decode_tps=_compute_tps(completion_tokens, completion_window_ms),
        end_to_end_tps=_compute_tps(completion_tokens, total_latency_ms),
        approx_prompt_tps=approx_prompt_tps,
        finish_reason=final_finish_reason,
    )
