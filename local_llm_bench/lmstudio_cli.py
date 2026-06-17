from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional


@dataclass
class ModelInfo:
    requested_model: str
    identifier: str
    model_key: str
    display_name: str
    format: str
    quantization: str
    quantization_name: str
    quantization_bits: Optional[int]
    publisher: str
    architecture: str
    selected_variant: str
    indexed_model_identifier: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnloadResult:
    requested_model: str
    target: str
    status: str
    message: str


def _run_cli(
    args: list[str],
    *,
    run: Callable[..., Any],
    timeout_sec: float,
) -> subprocess.CompletedProcess[str]:
    return run(
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _load_loaded_entries(
    *,
    run: Callable[..., Any],
    timeout_sec: float,
) -> list[dict[str, Any]]:
    try:
        completed = _run_cli(["lms", "ps", "--json"], run=run, timeout_sec=timeout_sec)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if completed.returncode != 0:
        return []

    try:
        loaded = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return []

    if not isinstance(loaded, list):
        return []
    return [entry for entry in loaded if isinstance(entry, dict)]


def _matching_entries(
    requested_model: str,
    *,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_model = requested_model.strip()
    if not normalized_model:
        return []

    candidate_names: list[str] = []
    for candidate in (
        normalized_model,
        normalized_model.split("@", 1)[0] if "@" in normalized_model else "",
        normalized_model.rsplit("-", 1)[0] if normalized_model.endswith(("-gguf", "-mlx")) else "",
    ):
        normalized_candidate = candidate.strip()
        if normalized_candidate and normalized_candidate not in candidate_names:
            candidate_names.append(normalized_candidate)

    base_model = candidate_names[1] if len(candidate_names) > 1 else candidate_names[0]
    tail_name = base_model.rsplit("/", 1)[-1].strip()
    if tail_name and tail_name not in candidate_names:
        candidate_names.append(tail_name)

    variant_token = normalized_model.split("@", 1)[1].strip() if "@" in normalized_model else ""
    if variant_token:
        candidate_names.append(variant_token)
        if tail_name:
            candidate_names.append(f"{tail_name}-{variant_token}")
    candidate_names = [candidate for candidate in candidate_names if candidate]

    exact_identifier_matches: list[dict[str, Any]] = []
    resolved_matches: list[dict[str, Any]] = []
    fuzzy_matches: list[dict[str, Any]] = []

    for entry in entries:
        identifier = _normalize_text(entry.get("identifier"))
        model_key = _normalize_text(entry.get("modelKey"))
        indexed_identifier = _normalize_text(entry.get("indexedModelIdentifier"))
        path = _normalize_text(entry.get("path"))
        selected_variant = _normalize_text(entry.get("selectedVariant"))
        display_name = _normalize_text(entry.get("displayName"))

        if identifier and identifier == normalized_model:
            exact_identifier_matches.append(entry)
            continue

        match_fields = {model_key, indexed_identifier, path, selected_variant}
        if any(candidate in match_fields for candidate in candidate_names):
            resolved_matches.append(entry)
            continue

        haystack = " ".join(part for part in (identifier, model_key, indexed_identifier, path, selected_variant, display_name) if part)
        haystack_lower = haystack.lower()
        if any(candidate.lower() in haystack_lower for candidate in candidate_names if len(candidate) >= 4):
            fuzzy_matches.append(entry)

    return exact_identifier_matches or resolved_matches or fuzzy_matches


def _safe_bits(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized >= 0 else None


def _build_quantization_fields(entry: dict[str, Any]) -> tuple[str, str, Optional[int]]:
    quantization = entry.get("quantization")
    if isinstance(quantization, dict):
        name = _normalize_text(quantization.get("name"))
        bits = _safe_bits(quantization.get("bits"))
        if bits is None:
            bits = _safe_bits(quantization.get("bits_per_weight"))
        if name and bits is not None:
            return (f"{name} ({bits}-bit)", name, bits)
        if name:
            return (name, name, bits)
        if bits is not None:
            return (f"{bits}-bit", "", bits)
    if isinstance(quantization, str):
        name = quantization.strip()
        if not name:
            return ("", "", None)
        match = re.search(r"(\d+)", name)
        bits = _safe_bits(match.group(1)) if match else None
        if bits is not None:
            return (f"{name} ({bits}-bit)", name, bits)
        return (name, name, None)
    return ("", "", None)


def _normalize_api_base(api_base: str) -> str:
    trimmed = str(api_base or "").rstrip("/")
    for suffix in ("/api/v1", "/api/v0", "/v1"):
        if trimmed.endswith(suffix):
            trimmed = trimmed[: -len(suffix)]
            break
    return trimmed.rstrip("/")


def _json_request(
    *,
    api_base: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    timeout_sec: float,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    root = _normalize_api_base(api_base)
    if not root:
        raise ValueError("LM Studio API base URL が空です。")
    data = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{root}{endpoint}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = ""
        if exc.fp is not None:
            try:
                body = exc.fp.read().decode("utf-8", errors="replace")
            except OSError:
                body = ""
        message = body.strip() or str(exc)
        raise RuntimeError(f"LM Studio API {endpoint} failed: HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio API {endpoint} failed: {exc}") from exc

    if not raw.strip():
        return {}
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"LM Studio API {endpoint} のレスポンスが不正です。")
    return loaded


def _load_v1_model_entries(
    *,
    api_base: str,
    timeout_sec: float,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> list[dict[str, Any]]:
    try:
        payload = _json_request(
            api_base=api_base,
            endpoint="/api/v1/models",
            timeout_sec=timeout_sec,
            urlopen=urlopen,
        )
    except RuntimeError:
        return []
    raw_entries = payload.get("models")
    if not isinstance(raw_entries, list):
        return []
    return [entry for entry in raw_entries if isinstance(entry, dict)]


def _matching_loaded_instance_ids(
    requested_model: str,
    *,
    api_base: str,
    timeout_sec: float,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> list[str]:
    raw_entries = _load_v1_model_entries(
        api_base=api_base,
        timeout_sec=timeout_sec,
        urlopen=urlopen,
    )
    normalized_entries: list[dict[str, Any]] = []
    loaded_by_model_key: dict[str, list[str]] = {}
    for entry in raw_entries:
        model_key = _normalize_text(entry.get("key")) or _normalize_text(entry.get("id"))
        loaded_instances = entry.get("loaded_instances")
        instance_ids: list[str] = []
        if isinstance(loaded_instances, list):
            for instance in loaded_instances:
                if not isinstance(instance, dict):
                    continue
                instance_id = _normalize_text(instance.get("id"))
                if instance_id:
                    instance_ids.append(instance_id)
        if instance_ids:
            loaded_by_model_key[model_key] = instance_ids
        normalized_entries.append(
            {
                "identifier": instance_ids[0] if instance_ids else model_key,
                "modelKey": model_key,
                "displayName": _normalize_text(entry.get("display_name")) or model_key,
                "format": _normalize_text(entry.get("format")),
                "quantization": entry.get("quantization"),
                "publisher": _normalize_text(entry.get("publisher")),
                "architecture": _normalize_text(entry.get("architecture")),
                "selectedVariant": _normalize_text(entry.get("selected_variant")),
                "indexedModelIdentifier": model_key,
                "path": model_key,
            }
        )

    matches = _matching_entries(requested_model, entries=normalized_entries)
    instance_ids: list[str] = []
    for match in matches:
        model_key = _normalize_text(match.get("modelKey"))
        for instance_id in loaded_by_model_key.get(model_key, []):
            if instance_id not in instance_ids:
                instance_ids.append(instance_id)
    return instance_ids


def unload_matching_models_via_api(
    requested_model: str,
    *,
    api_base: str,
    timeout_sec: float = 60.0,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> list[UnloadResult]:
    instance_ids = _matching_loaded_instance_ids(
        requested_model,
        api_base=api_base,
        timeout_sec=timeout_sec,
        urlopen=urlopen,
    )
    results: list[UnloadResult] = []
    for instance_id in instance_ids:
        try:
            payload = _json_request(
                api_base=api_base,
                endpoint="/api/v1/models/unload",
                payload={"instance_id": instance_id},
                timeout_sec=timeout_sec,
                urlopen=urlopen,
            )
        except RuntimeError as exc:
            results.append(
                UnloadResult(
                    requested_model=requested_model,
                    target=instance_id,
                    status="error",
                    message=str(exc),
                )
            )
            continue
        results.append(
            UnloadResult(
                requested_model=requested_model,
                target=_normalize_text(payload.get("instance_id")) or instance_id,
                status="unloaded",
                message="unloaded",
            )
        )
    return results


def load_model_with_config(
    requested_model: str,
    *,
    api_base: str,
    parallelism: int | None = None,
    context_length: int | None = None,
    eval_batch_size: int | None = None,
    flash_attention: bool | None = None,
    num_experts: int | None = None,
    offload_kv_cache_to_gpu: bool | None = None,
    timeout_sec: float = 300.0,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": requested_model,
        "echo_load_config": True,
    }
    if parallelism is not None:
        payload["parallel"] = parallelism
    if context_length is not None:
        payload["context_length"] = context_length
    if eval_batch_size is not None:
        payload["eval_batch_size"] = eval_batch_size
    if flash_attention is not None:
        payload["flash_attention"] = flash_attention
    if num_experts is not None:
        payload["num_experts"] = num_experts
    if offload_kv_cache_to_gpu is not None:
        payload["offload_kv_cache_to_gpu"] = offload_kv_cache_to_gpu

    return _json_request(
        api_base=api_base,
        endpoint="/api/v1/models/load",
        payload=payload,
        timeout_sec=timeout_sec,
        urlopen=urlopen,
    )


def _load_http_model_entries(
    *,
    api_base: str,
    timeout_sec: float,
) -> list[dict[str, Any]]:
    root = _normalize_api_base(api_base)
    if not root:
        return []

    for endpoint in ("/api/v1/models", "/api/v0/models"):
        try:
            with urllib.request.urlopen(f"{root}{endpoint}", timeout=timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (
            OSError,
            ValueError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ):
            continue

        if endpoint == "/api/v1/models":
            raw_entries = payload.get("models")
        else:
            raw_entries = payload.get("data")
        if not isinstance(raw_entries, list):
            continue

        normalized_entries: list[dict[str, Any]] = []
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue

            loaded_instances = entry.get("loaded_instances")
            loaded_identifier = ""
            if isinstance(loaded_instances, list):
                for instance in loaded_instances:
                    if isinstance(instance, dict):
                        loaded_identifier = _normalize_text(instance.get("id"))
                        if loaded_identifier:
                            break

            quantization = entry.get("quantization")
            if isinstance(quantization, str):
                quantization = {"name": quantization}

            normalized_entries.append(
                {
                    "identifier": loaded_identifier or _normalize_text(entry.get("id")) or _normalize_text(entry.get("key")),
                    "modelKey": _normalize_text(entry.get("key")) or _normalize_text(entry.get("id")),
                    "displayName": _normalize_text(entry.get("display_name")) or _normalize_text(entry.get("id")) or _normalize_text(entry.get("key")),
                    "format": _normalize_text(entry.get("format")) or _normalize_text(entry.get("compatibility_type")),
                    "quantization": quantization,
                    "publisher": _normalize_text(entry.get("publisher")),
                    "architecture": _normalize_text(entry.get("architecture")) or _normalize_text(entry.get("arch")),
                    "selectedVariant": _normalize_text(entry.get("selected_variant")),
                    "indexedModelIdentifier": _normalize_text(entry.get("id")) or _normalize_text(entry.get("key")),
                    "path": _normalize_text(entry.get("path")) or _normalize_text(entry.get("key")) or _normalize_text(entry.get("id")),
                }
            )
        if normalized_entries:
            return normalized_entries

    return []


def describe_loaded_model(
    requested_model: str,
    *,
    run: Callable[..., Any] = subprocess.run,
    timeout_sec: float = 15.0,
    api_base: str | None = None,
) -> Optional[dict[str, Any]]:
    entries = _load_loaded_entries(run=run, timeout_sec=timeout_sec)
    matches = _matching_entries(requested_model, entries=entries)
    if not matches and api_base:
        entries = _load_http_model_entries(api_base=api_base, timeout_sec=timeout_sec)
        matches = _matching_entries(requested_model, entries=entries)
    if not matches:
        return None

    entry = matches[0]
    quantization, quantization_name, quantization_bits = _build_quantization_fields(entry)
    info = ModelInfo(
        requested_model=requested_model,
        identifier=_normalize_text(entry.get("identifier")),
        model_key=_normalize_text(entry.get("modelKey")),
        display_name=_normalize_text(entry.get("displayName")) or _normalize_text(entry.get("modelKey")) or requested_model,
        format=_normalize_text(entry.get("format")),
        quantization=quantization,
        quantization_name=quantization_name,
        quantization_bits=quantization_bits,
        publisher=_normalize_text(entry.get("publisher")),
        architecture=_normalize_text(entry.get("architecture")),
        selected_variant=_normalize_text(entry.get("selectedVariant")),
        indexed_model_identifier=_normalize_text(entry.get("indexedModelIdentifier")),
        path=_normalize_text(entry.get("path")),
    )
    return info.to_dict()


def _resolve_unload_targets(
    requested_model: str,
    *,
    run: Callable[..., Any],
    timeout_sec: float,
) -> list[str]:
    loaded_entries = _load_loaded_entries(run=run, timeout_sec=timeout_sec)
    matches = _matching_entries(requested_model, entries=loaded_entries)
    if not matches:
        return [requested_model]

    targets = [
        _normalize_text(entry.get("identifier"))
        or _normalize_text(entry.get("modelKey"))
        or _normalize_text(entry.get("indexedModelIdentifier"))
        or _normalize_text(entry.get("path"))
        or requested_model
        for entry in matches
    ]
    unique_targets: list[str] = []
    seen: set[str] = set()
    for target in targets:
        normalized_target = target.strip()
        if not normalized_target or normalized_target in seen:
            continue
        seen.add(normalized_target)
        unique_targets.append(normalized_target)
    return unique_targets or [requested_model]


def unload_matching_models(
    requested_model: str,
    *,
    run: Callable[..., Any] = subprocess.run,
    timeout_sec: float = 15.0,
) -> list[UnloadResult]:
    targets = _resolve_unload_targets(
        requested_model,
        run=run,
        timeout_sec=timeout_sec,
    )
    results: list[UnloadResult] = []
    for target in targets:
        try:
            completed = _run_cli(["lms", "unload", target], run=run, timeout_sec=timeout_sec)
        except FileNotFoundError:
            results.append(
                UnloadResult(
                    requested_model=requested_model,
                    target=target,
                    status="error",
                    message="lms コマンドが見つかりません。",
                )
            )
            continue
        except subprocess.TimeoutExpired:
            results.append(
                UnloadResult(
                    requested_model=requested_model,
                    target=target,
                    status="timeout",
                    message="lms unload がタイムアウトしました。",
                )
            )
            continue

        if completed.returncode == 0:
            message = (completed.stdout or completed.stderr or "unloaded").strip()
            results.append(
                UnloadResult(
                    requested_model=requested_model,
                    target=target,
                    status="unloaded",
                    message=message or "unloaded",
                )
            )
        else:
            message = (completed.stderr or completed.stdout or "lms unload failed").strip()
            results.append(
                UnloadResult(
                    requested_model=requested_model,
                    target=target,
                    status="error",
                    message=message,
                )
            )
    return results
