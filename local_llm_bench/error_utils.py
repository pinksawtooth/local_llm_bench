from __future__ import annotations

import re
from typing import Any, MutableMapping

_QUESTION_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
_WHITESPACE_RE = re.compile(r"\s+")
_ABS_PATH_RE = re.compile(
    r"("
    r"/private/tmp/[^\s:]+"
    r"|/tmp/[^\s:]+"
    r"|/var/folders/[^\s:]+"
    r"|/Users/[^\s:]+"
    r"|[A-Za-z]:\\\\[^\s:]+"
    r")"
)


def excerpt_text(value: Any, *, max_length: int = 400) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def merge_excerpts(values: list[Any], *, max_length: int = 400) -> str | None:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        excerpt = excerpt_text(value, max_length=max_length)
        if not excerpt or excerpt in seen:
            continue
        seen.add(excerpt)
        unique.append(excerpt)
    if not unique:
        return None
    merged = "\n\n".join(unique)
    return excerpt_text(merged, max_length=max_length)


def normalize_error_signature(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    first_line = ""
    for line in text.replace("\r\n", "\n").splitlines():
        candidate = line.strip()
        if candidate:
            first_line = candidate
            break
    if not first_line:
        return None
    normalized = _QUESTION_PREFIX_RE.sub("", first_line)
    normalized = _ABS_PATH_RE.sub("<path>", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return excerpt_text(normalized, max_length=240)


def categorize_error(value: Any, *, status: Any = None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        if str(status or "").strip().lower() == "success":
            return None
        if str(status or "").strip().lower() == "timeout":
            return "timeout"
        return None
    lowered = f"{status or ''} {value}".lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if any(token in lowered for token in ("httperror", "urlerror", "api_base is required", "chat completion response")):
        return "api"
    if any(token in lowered for token in ("tool '", 'tool "', "call_tool", "mcp")):
        return "tool"
    if any(
        token in lowered
        for token in (
            "docker exited",
            "docker コマンド",
            "no such container",
            "unable to find image",
            "pull access denied",
            "no matching manifest",
            "manifest for",
            "docker image '",
        )
    ):
        return "docker"
    if any(
        token in lowered
        for token in (
            "worker returned malformed json",
            "final_answer",
            "analyzeheadless",
            "ghidra",
            "unhandled errors in a taskgroup",
            "binary_path",
        )
    ):
        return "worker"
    return "other"


def annotate_error_info(
    target: MutableMapping[str, Any],
    *,
    error: Any = None,
    status: Any = None,
    stderr_text: Any = None,
) -> MutableMapping[str, Any]:
    source_error = error if isinstance(error, str) and error.strip() else target.get("error")
    source_status = status if status is not None else target.get("status")
    source_stderr = stderr_text if isinstance(stderr_text, str) else target.get("stderr_excerpt")

    signature_source = source_error if isinstance(source_error, str) and source_error.strip() else source_stderr
    target["error_signature"] = normalize_error_signature(signature_source)
    target["error_category"] = categorize_error(signature_source or source_error or "", status=source_status)
    target["stderr_excerpt"] = excerpt_text(source_stderr)
    target.setdefault("log_path", None)
    return target
