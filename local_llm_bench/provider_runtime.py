from __future__ import annotations

from dataclasses import dataclass, field
import copy
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Optional

from .config import (
    BenchmarkConfig,
    LMSTUDIO_PROVIDER,
    UNSLOTH_STUDIO_PROVIDER,
    DS4_PROVIDER,
    OMLX_PROVIDER,
    MLX_SERVE_PROVIDER,
)
from .ds4 import (
    DS4Session, DS4_API_KEY_ENV, ManagedDS4Server, ds4_environment_info,
    ds4_launch_config_sha256, normalize_ds4_launch_config, read_ds4_model_info,
    validate_ds4_launch_files, validate_ds4_request, validate_managed_ds4_base_url,
    apply_ds4_sampling,
)
from .lmstudio_api import stream_chat_completion
from .lmstudio_cli import (
    ModelInfo,
    UnloadResult,
    _build_quantization_fields,
    _matching_entries,
    _json_request,
    describe_loaded_model,
    load_model_with_config,
    unload_matching_models_via_api,
)
from .unsloth_api import UnslothStudioAuthSession
from .omlx import OMLXSession, OMLX_API_KEY_ENV, OMLX_REQUEST_KEYS, validate_omlx_request
from .mlx_serve import (
    MLXServeSession, MLX_SERVE_API_KEY_ENV, MLX_SERVE_REQUEST_KEYS,
    normalize_mlx_serve_config, validate_mlx_serve_request,
)

_MODEL_PREPARE_TIMEOUT_SEC = 300.0
_QUANTIZATION_TOKEN_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"((?:IQ|Q)\d+(?:_[A-Za-z0-9]+)*|(?:BF|FP|F)\d+|MXFP\d+|\d+BIT)"
    r"(?![A-Za-z0-9])"
)
_AUXILIARY_GGUF_MARKERS = ("mmproj", "mm-proj", "projector")
_DRAFTER_GGUF_PREFIXES = ("mtp-", "dspark-", "dflash-")
_DRAFTER_GGUF_DIRECTORIES = {"mtp", "dspark"}
_IMATRIX_GGUF_RE = re.compile(r"^imatrix(?:[._-]|$)|[._-]imatrix$", re.IGNORECASE)


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _looks_like_filesystem_path(value: Any) -> bool:
    text = _normalize_text(value)
    if not text:
        return False
    return text.startswith(("/", "~/")) or bool(re.match(r"^[A-Za-z]:[\\/]", text))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _first_text(*values: Any) -> str:
    for value in values:
        normalized = _normalize_text(value)
        if normalized:
            return normalized
    return ""


def _extract_model_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "models", "items", "loaded_models", "results", "local_models", "cached"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _infer_publisher(entry: dict[str, Any]) -> str:
    for candidate in (
        entry.get("publisher"),
        entry.get("owned_by"),
        entry.get("owner"),
    ):
        normalized = _normalize_text(candidate)
        if normalized:
            return normalized
    for candidate in (
        entry.get("model_path"),
        entry.get("path"),
        entry.get("id"),
        entry.get("name"),
    ):
        normalized = _normalize_text(candidate)
        if "/" in normalized:
            return normalized.split("/", 1)[0]
    return ""


def _infer_format(entry: dict[str, Any]) -> str:
    explicit = _first_text(
        entry.get("format"),
        entry.get("backend"),
        entry.get("engine"),
        entry.get("compatibility_type"),
    )
    if explicit:
        return explicit
    for candidate in (
        entry.get("model_path"),
        entry.get("path"),
        entry.get("id"),
        entry.get("model_id"),
        entry.get("name"),
        entry.get("display_name"),
        entry.get("identifier"),
    ):
        text = _normalize_text(candidate)
        if not text:
            continue
        lowered_text = text.lower()
        if lowered_text.endswith(".gguf") or "gguf" in lowered_text or _coerce_bool(entry.get("is_gguf")):
            return "gguf"
        if "mlx" in lowered_text:
            return "mlx"
    path = _first_text(entry.get("model_path"), entry.get("path"), entry.get("id"))
    if _coerce_bool(entry.get("is_gguf")) or path.lower().endswith(".gguf"):
        return "gguf"
    return "transformers"


def _normalize_available_model_entries(payload: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in _extract_model_entries(payload):
        identifier = _first_text(
            entry.get("id"),
            entry.get("model_id"),
            entry.get("key"),
            entry.get("name"),
            entry.get("model"),
            entry.get("identifier"),
            entry.get("model_path"),
        )
        model_path = _first_text(
            entry.get("path"),
            entry.get("model_path"),
            entry.get("id"),
            entry.get("model_id"),
        )
        display_name = _first_text(
            entry.get("display_name"),
            entry.get("name"),
            entry.get("title"),
            entry.get("model_name"),
            entry.get("id"),
            identifier,
            model_path,
        )
        quantization = entry.get("quantization")
        if quantization in (None, ""):
            quantization = _first_text(entry.get("gguf_variant"), entry.get("variant"))
        entries.append(
            {
                "identifier": identifier,
                "modelKey": _first_text(entry.get("model_id"), entry.get("key"), entry.get("id"), identifier, model_path),
                "displayName": display_name,
                "format": _infer_format(entry),
                "quantization": quantization,
                "publisher": _infer_publisher(entry),
                "architecture": _first_text(entry.get("architecture"), entry.get("arch"), entry.get("family")),
                "selectedVariant": _first_text(entry.get("selected_variant"), entry.get("gguf_variant"), entry.get("variant")),
                "indexedModelIdentifier": _first_text(entry.get("id"), entry.get("model_id"), identifier, model_path, entry.get("key")),
                "path": model_path,
                "source": _first_text(entry.get("source")),
            }
        )
    return entries


def _normalize_loaded_model_entries(payload: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in _extract_model_entries(payload):
        identifier = _first_text(entry.get("id"), entry.get("model"), entry.get("name"))
        entries.append(
            {
                "identifier": identifier,
                "modelKey": _first_text(entry.get("id"), entry.get("model"), identifier),
                "displayName": _first_text(entry.get("display_name"), entry.get("name"), identifier),
                "format": _first_text(entry.get("format")),
                "quantization": entry.get("quantization"),
                "publisher": _first_text(entry.get("owned_by"), entry.get("publisher")),
                "architecture": _first_text(entry.get("architecture"), entry.get("arch")),
                "selectedVariant": _first_text(entry.get("selected_variant"), entry.get("variant")),
                "indexedModelIdentifier": _first_text(entry.get("id"), identifier),
                "path": _first_text(entry.get("id"), identifier),
            }
        )
    return entries


def _merge_entries(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    placeholder_display_names = {
        _first_text(primary.get("identifier")),
        _first_text(primary.get("path")),
        _first_text(primary.get("modelKey")),
    }
    for key, value in secondary.items():
        if key == "displayName" and _first_text(value) and _first_text(merged.get(key)) in placeholder_display_names:
            merged[key] = value
            continue
        if key not in merged or merged[key] in ("", None, {}):
            merged[key] = value
    return merged


def _extract_quantization_token(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""

    if "@" in text:
        variant = text.rsplit("@", 1)[-1].strip()
        _, name, _ = _build_quantization_fields({"quantization": variant})
        if name:
            return name

    candidates = [text]
    try:
        path = Path(text)
    except OSError:
        path = None
    if path is not None:
        for candidate in (path.name, path.stem):
            normalized = _normalize_text(candidate)
            if normalized and normalized not in candidates:
                candidates.append(normalized)

    for candidate in candidates:
        match = _QUANTIZATION_TOKEN_RE.search(candidate)
        if match:
            token = match.group(1).replace("-", "_").upper()
            _, name, _ = _build_quantization_fields({"quantization": token})
            if name:
                return name
    return ""


def _safe_file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return -1


def _is_auxiliary_gguf(path: Path) -> bool:
    name = path.name.lower()
    return (
        any(marker in name for marker in _AUXILIARY_GGUF_MARKERS)
        or name.startswith(_DRAFTER_GGUF_PREFIXES)
        or any(part.lower() in _DRAFTER_GGUF_DIRECTORIES for part in path.parts[:-1])
        or bool(_IMATRIX_GGUF_RE.search(path.stem))
    )


def _resolve_primary_gguf_artifact(model_path: str) -> str:
    normalized_path = _normalize_text(model_path)
    if not normalized_path:
        return ""
    try:
        root = Path(normalized_path)
    except OSError:
        return ""
    if not root.exists():
        return ""
    if root.is_file():
        relative_path = Path(root.parent.name) / root.name
        return str(root) if root.suffix.lower() == ".gguf" and not _is_auxiliary_gguf(relative_path) else ""
    if not root.is_dir():
        return ""

    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".gguf"
        and not _is_auxiliary_gguf(path.relative_to(root))
    ]
    if not candidates:
        return ""

    def sort_key(path: Path) -> tuple[int, int, str]:
        has_quantization = bool(_extract_quantization_token(path.name))
        return (
            0 if has_quantization else 1,
            -_safe_file_size(path),
            str(path),
        )

    candidates.sort(key=sort_key)
    return str(candidates[0])


def _build_unsloth_model_info(
    requested_model: str,
    *,
    entry: dict[str, Any],
    load_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = load_response or {}
    model_identifier = _first_text(
        entry.get("identifier"),
        requested_model,
        payload.get("model"),
    )
    payload_model_identifier = _first_text(payload.get("model"))
    if _looks_like_filesystem_path(model_identifier) and payload_model_identifier and not _looks_like_filesystem_path(payload_model_identifier):
        model_identifier = payload_model_identifier
    model_path = _first_text(
        entry.get("path"),
        payload.get("model"),
        entry.get("identifier"),
        requested_model,
    )
    format_label = _first_text(payload.get("format"))
    if not format_label:
        if _coerce_bool(payload.get("is_gguf")) or "gguf" in str(model_identifier).lower() or str(model_path).lower().endswith(".gguf"):
            format_label = "gguf"
        else:
            format_label = _first_text(entry.get("format")) or "transformers"

    artifact_path = ""
    if format_label.lower() == "gguf":
        artifact_path = _resolve_primary_gguf_artifact(model_path)

    quantization_source: Any = payload.get("quantization")
    if quantization_source in (None, ""):
        quantization_source = _first_text(payload.get("gguf_variant")) or entry.get("quantization")
    if quantization_source in (None, ""):
        for candidate in (
            payload.get("selected_variant"),
            entry.get("selectedVariant"),
            payload.get("display_name"),
            entry.get("displayName"),
            payload.get("model"),
            entry.get("modelKey"),
            entry.get("indexedModelIdentifier"),
            artifact_path,
            model_path,
            model_identifier,
            requested_model,
        ):
            inferred_quantization = _extract_quantization_token(candidate)
            if inferred_quantization:
                quantization_source = inferred_quantization
                break
    quantization, quantization_name, quantization_bits = _build_quantization_fields(
        {"quantization": quantization_source}
    )

    info = ModelInfo(
        requested_model=requested_model,
        identifier=model_identifier,
        model_key=_first_text(entry.get("modelKey"), payload.get("model"), model_identifier, model_path),
        display_name=_first_text(payload.get("display_name"), entry.get("displayName"), model_identifier, requested_model),
        format=format_label,
        quantization=quantization,
        quantization_name=quantization_name,
        quantization_bits=quantization_bits,
        publisher=_first_text(entry.get("publisher")),
        architecture=_first_text(entry.get("architecture")),
        selected_variant=_first_text(entry.get("selectedVariant"), payload.get("gguf_variant"), quantization_name),
        indexed_model_identifier=_first_text(entry.get("indexedModelIdentifier"), payload.get("model"), model_identifier, model_path),
        path=artifact_path or model_path,
    )
    return info.to_dict()


def _preferred_unsloth_chat_model(
    requested_model: str,
    *,
    entry: dict[str, Any],
    load_response: dict[str, Any],
    load_target: str,
) -> str:
    for candidate in (
        requested_model,
        entry.get("identifier"),
        entry.get("indexedModelIdentifier"),
        load_response.get("model"),
        entry.get("modelKey"),
        load_target,
    ):
        normalized = _normalize_text(candidate)
        if normalized and not _looks_like_filesystem_path(normalized):
            return normalized
    return _first_text(load_response.get("model"), load_target, requested_model)


def _preferred_unsloth_unload_target(requested_model: str, *, entry: dict[str, Any] | None = None) -> str:
    if _looks_like_filesystem_path(requested_model):
        return _resolve_primary_gguf_artifact(requested_model) or requested_model
    if entry and _normalize_text(entry.get("format")).lower() == "gguf":
        path = _first_text(entry.get("path"), requested_model)
        return _resolve_primary_gguf_artifact(path) or path
    if entry:
        return _first_text(
            entry.get("path"),
            entry.get("indexedModelIdentifier"),
            entry.get("identifier"),
            entry.get("modelKey"),
            requested_model,
        )
    return requested_model


def _preferred_unsloth_load_target(entry: dict[str, Any], requested_model: str) -> str:
    load_target = _first_text(
        entry.get("path"),
        entry.get("identifier"),
        entry.get("modelKey"),
        entry.get("indexedModelIdentifier"),
        requested_model,
    )
    if _normalize_text(entry.get("format")).lower() == "gguf":
        gguf_artifact = _resolve_primary_gguf_artifact(load_target)
        if gguf_artifact:
            return gguf_artifact
        if Path(load_target).exists():
            raise RuntimeError(f"Unsloth Studio のロード対象 '{load_target}' にモデル本体の GGUF が見つかりません。")
    return load_target


def _prefer_unsloth_available_entry(
    requested_model: str,
    *,
    entries: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    if not matches:
        raise RuntimeError("available entry candidates are required")
    preferred = matches[0]
    preferred_format = _normalize_text(preferred.get("format")).lower()
    requested_lower = requested_model.strip().lower()
    if preferred_format == "gguf" or "gguf" in requested_lower or "@" in requested_lower:
        return preferred

    model_tail = requested_lower.rsplit("/", 1)[-1]
    if model_tail.endswith(("-gguf", "-mlx")):
        model_tail = model_tail.rsplit("-", 1)[0]
    tail_tokens = [token for token in re.split(r"[-_/]+", model_tail) if token]

    alternatives: list[dict[str, Any]] = []
    for entry in entries:
        entry_format = _normalize_text(entry.get("format")).lower()
        if entry_format != "gguf":
            continue
        haystack = " ".join(
            _normalize_text(entry.get(key)).lower()
            for key in ("identifier", "modelKey", "displayName", "indexedModelIdentifier", "path")
        )
        if model_tail and model_tail in haystack:
            alternatives.append(entry)
            continue
        if tail_tokens and all(token in haystack for token in tail_tokens):
            alternatives.append(entry)

    if not alternatives:
        return preferred

    def sort_key(entry: dict[str, Any]) -> tuple[int, int, int, str]:
        source = _normalize_text(entry.get("source")).lower()
        identifier = _normalize_text(entry.get("identifier"))
        return (
            0 if source == "lmstudio" else 1,
            0 if not _looks_like_filesystem_path(identifier) else 1,
            0 if model_tail and model_tail in " ".join(
                _normalize_text(entry.get(key)).lower()
                for key in ("identifier", "modelKey", "displayName", "indexedModelIdentifier", "path")
            ) else 1,
            identifier or _normalize_text(entry.get("path")) or _normalize_text(entry.get("modelKey")),
        )

    alternatives.sort(key=sort_key)
    return alternatives[0]


class ProviderRuntime:
    def begin_run(self, directory: Path, *, ds4_launch_config=None) -> None:
        """Bind provider-owned artifacts to the current run, without loading."""

    def measurement_metadata(self, model_info: dict[str, Any]) -> dict[str, Any]:
        return model_info

    def inspect_model(self, requested_model: str) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_model(
        self,
        requested_model: str,
        *,
        lmstudio_parallelism: int | None = None,
    ) -> tuple[str, Optional[dict[str, Any]]]:
        raise NotImplementedError

    def describe_model(self, requested_model: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    def unload_model(self, requested_model: str) -> list[UnloadResult]:
        raise NotImplementedError

    def chat_client(self) -> Callable[..., Any]:
        raise NotImplementedError

    def docker_environment(self) -> dict[str, str]:
        return {}


@dataclass
class LMStudioProviderRuntime(ProviderRuntime):
    config: BenchmarkConfig

    def inspect_model(self, requested_model: str) -> dict[str, Any]:
        payload = _json_request(api_base=self.config.api_base, endpoint="/api/v1/models", timeout_sec=15.0)
        entries = _normalize_available_model_entries(payload)
        matches = _matching_entries(requested_model, entries=entries)
        if not matches:
            raise RuntimeError(f"LM Studio に model '{requested_model}' が見つかりません。")
        return matches[0]

    def prepare_model(
        self,
        requested_model: str,
        *,
        lmstudio_parallelism: int | None = None,
    ) -> tuple[str, Optional[dict[str, Any]]]:
        load_settings = self.config.lmstudio_load
        requested_parallelism = (
            lmstudio_parallelism
            if lmstudio_parallelism is not None
            else load_settings.parallelism
        )
        load_response: dict[str, Any] | None = None
        load_response = load_model_with_config(
            requested_model,
            api_base=self.config.api_base,
            parallelism=requested_parallelism,
            context_length=load_settings.context_length,
            eval_batch_size=load_settings.eval_batch_size,
            flash_attention=load_settings.flash_attention,
            num_experts=load_settings.num_experts,
            offload_kv_cache_to_gpu=load_settings.offload_kv_cache_to_gpu,
            timeout_sec=_MODEL_PREPARE_TIMEOUT_SEC,
        )

        model_info = describe_loaded_model(requested_model, api_base=self.config.api_base)
        if load_response is not None:
            if model_info is None:
                model_info = {}
            instance_id = _first_text(load_response.get("instance_id"))
            if instance_id and not _first_text(model_info.get("identifier")):
                model_info["identifier"] = instance_id
            if isinstance(load_response.get("load_config"), dict):
                model_info["load_config"] = load_response["load_config"]
            model_info["load_time_seconds"] = load_response.get("load_time_seconds")
            model_info["runtime"] = load_response.get("runtime") or model_info.get("runtime") or {}
            model_info["load_status"] = load_response.get("status")
            reported = dict(model_info.get("reported_inference") or {})
            reported.update({key: load_response[key] for key in ("sampling", "thinking", "speculative_decoding") if key in load_response})
            model_info["reported_inference"] = reported
            if requested_parallelism is not None:
                model_info["lmstudio_parallelism"] = requested_parallelism
        api_model = requested_model
        if model_info:
            api_model = _first_text(
                model_info.get("identifier"),
                model_info.get("model_key"),
                model_info.get("indexed_model_identifier"),
                model_info.get("path"),
                requested_model,
            )
        return api_model, model_info

    def describe_model(self, requested_model: str) -> Optional[dict[str, Any]]:
        return describe_loaded_model(requested_model, api_base=self.config.api_base)

    def unload_model(self, requested_model: str) -> list[UnloadResult]:
        return unload_matching_models_via_api(
            requested_model,
            api_base=self.config.api_base,
            timeout_sec=60.0,
        )

    def chat_client(self) -> Callable[..., Any]:
        return stream_chat_completion


@dataclass
class UnslothStudioProviderRuntime(ProviderRuntime):
    config: BenchmarkConfig
    session: UnslothStudioAuthSession
    _load_targets: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def inspect_model(self, requested_model: str) -> dict[str, Any]:
        return self._match_available_entry(requested_model, timeout_sec=15.0)

    def _available_entries(self, *, timeout_sec: float) -> list[dict[str, Any]]:
        payload = self.session.request_json("/api/models/local", timeout_sec=timeout_sec)
        return _normalize_available_model_entries(payload)

    def _loaded_entries(self, *, timeout_sec: float) -> list[dict[str, Any]]:
        payload = self.session.request_json("/v1/models", timeout_sec=timeout_sec)
        return _normalize_loaded_model_entries(payload)

    def _match_available_entry(self, requested_model: str, *, timeout_sec: float) -> dict[str, Any]:
        entries = self._available_entries(timeout_sec=timeout_sec)
        matches = _matching_entries(requested_model, entries=entries)
        if matches:
            return _prefer_unsloth_available_entry(requested_model, entries=entries, matches=matches)
        raise RuntimeError(f"Unsloth Studio で model '{requested_model}' が /api/models/local に見つかりません。")

    def prepare_model(
        self,
        requested_model: str,
        *,
        lmstudio_parallelism: int | None = None,
    ) -> tuple[str, Optional[dict[str, Any]]]:
        timeout_sec = max(float(self.config.runs.timeout_sec), _MODEL_PREPARE_TIMEOUT_SEC)
        entry = self._match_available_entry(requested_model, timeout_sec=timeout_sec)
        load_target = _preferred_unsloth_load_target(entry, requested_model)
        load_response = self.session.request_json(
            "/api/inference/load",
            method="POST",
            payload={"model_path": load_target},
            timeout_sec=timeout_sec,
        )
        if not isinstance(load_response, dict):
            raise RuntimeError("Unsloth Studio の model load レスポンスが不正です。")
        api_model = _preferred_unsloth_chat_model(
            requested_model,
            entry=entry,
            load_response=load_response,
            load_target=load_target,
        )
        model_info = _build_unsloth_model_info(
            requested_model,
            entry=entry,
            load_response=load_response,
        )
        model_info["load_status"] = load_response.get("status")
        model_info["load_time_seconds"] = load_response.get("load_time_seconds")
        model_info["load_config"] = load_response.get("load_config")
        model_info["runtime"] = load_response.get("runtime") or {}
        model_info["reported_inference"] = {
            key: load_response[key] for key in ("sampling", "thinking", "speculative_decoding") if key in load_response
        }
        effective_fields = ("context_length", "context_length_enforced", "cache_type_kv", "parallel_slots", "tensor_parallel", "gpu_ids", "mlx_kv_bits", "chat_template")
        observed_load = {key: load_response[key] for key in effective_fields if key in load_response}
        if observed_load and not model_info["load_config"]:
            model_info["load_config"] = observed_load
        # The API calls these recommended parameters and requested speculative mode.
        # Keep them as evidence; they are not per-request effective telemetry.
        model_info["inference_defaults"] = load_response.get("inference")
        model_info["reported_speculation"] = {key: load_response[key] for key in ("speculative_type", "spec_draft_n_max", "spec_drafter_kind", "spec_fallback_reason") if key in load_response}
        model_info["reasoning_capabilities"] = {key: load_response[key] for key in ("supports_reasoning", "reasoning_style", "reasoning_always_on", "preserve_thinking_default") if key in load_response}
        # Callers may unload using the requested path, chat alias, or metadata ID.
        # Keep the exact load request instead of resolving the inventory again.
        for alias in (requested_model, api_model, model_info.get("identifier"), load_target):
            if normalized_alias := _normalize_text(alias):
                self._load_targets[normalized_alias] = load_target
        return api_model, model_info

    def measurement_metadata(self, model_info: dict[str, Any]) -> dict[str, Any]:
        info = dict(model_info)
        try:
            status = self.session.request_json("/api/inference/status", timeout_sec=15.0)
        except Exception:
            return info
        if not isinstance(status, dict):
            return info
        identities = {str(info.get(key) or "") for key in ("requested_model", "identifier", "path", "model_key", "indexed_model_identifier")}
        identities.update(self._load_targets.values())
        active = _first_text(status.get("model_identifier"), status.get("active_model"))
        if not active or active not in identities:
            return info
        runtime = dict(info.get("runtime") or {})
        if status.get("is_gguf"):
            runtime.setdefault("engine", "llama.cpp")
            # An installed binary tag need not identify a running process after an update.
            runtime["installed_version"] = status.get("llama_cpp_installed_tag")
            runtime.setdefault("source", "provider_status")
        info["runtime"] = runtime
        for key in ("spec_drafter_kind", "spec_fallback_reason"):
            if key in status:
                info["reported_speculation"][key] = status[key]
        return info

    def describe_model(self, requested_model: str) -> Optional[dict[str, Any]]:
        timeout_sec = min(max(float(self.config.runs.timeout_sec), 15.0), 60.0)
        loaded_entries = self._loaded_entries(timeout_sec=timeout_sec)
        available_entries = self._available_entries(timeout_sec=timeout_sec)
        loaded_matches = _matching_entries(requested_model, entries=loaded_entries)
        available_matches = _matching_entries(requested_model, entries=available_entries)
        if loaded_matches and available_matches:
            return _build_unsloth_model_info(
                requested_model,
                entry=_merge_entries(loaded_matches[0], available_matches[0]),
            )
        if available_matches:
            return _build_unsloth_model_info(requested_model, entry=available_matches[0])
        if loaded_matches:
            return _build_unsloth_model_info(requested_model, entry=loaded_matches[0])
        return None

    def unload_model(self, requested_model: str) -> list[UnloadResult]:
        timeout_sec = max(float(self.config.runs.timeout_sec), 30.0)
        unload_target = self._load_targets.get(_normalize_text(requested_model))
        if unload_target is None:
            unload_target = requested_model
            try:
                if not _looks_like_filesystem_path(requested_model):
                    available_matches = _matching_entries(requested_model, entries=self._available_entries(timeout_sec=timeout_sec))
                    loaded_matches = _matching_entries(requested_model, entries=self._loaded_entries(timeout_sec=timeout_sec))
                    matched_entry = available_matches[0] if available_matches else (loaded_matches[0] if loaded_matches else None)
                    unload_target = _preferred_unsloth_unload_target(requested_model, entry=matched_entry)
            except Exception:
                unload_target = requested_model
        try:
            payload = self.session.request_json(
                "/api/inference/unload",
                method="POST",
                payload={"model_path": unload_target},
                timeout_sec=timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001
            return [
                UnloadResult(
                    requested_model=requested_model,
                    target=unload_target,
                    status="error",
                    message=str(exc),
                )
            ]

        target = unload_target
        status = "unloaded"
        message = "unloaded"
        if isinstance(payload, dict):
            target = _first_text(payload.get("model"), target)
            status = _first_text(payload.get("status")) or status
            message = _first_text(payload.get("status"), payload.get("message")) or message
        if status == "unloaded":
            self._load_targets = {
                alias: path for alias, path in self._load_targets.items() if path != unload_target
            }
        return [
            UnloadResult(
                requested_model=requested_model,
                target=target,
                status=status,
                message=message,
            )
        ]

    def chat_client(self) -> Callable[..., Any]:
        def client(
            *,
            api_base: str,
            model: str,
            prompt_text: str,
            temperature: float,
            max_tokens: int,
            timeout_sec: float,
            now_fn: Callable[[], float],
            urlopen: Callable[..., Any] | None = None,
            top_p: float | None = None,
            reasoning_effort: str | None = None,
        ) -> Any:
            return stream_chat_completion(
                api_base=self.config.api_base,
                model=model,
                prompt_text=prompt_text,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                now_fn=now_fn,
                urlopen=self.session.urlopen,
                top_p=top_p,
                reasoning_effort=reasoning_effort,
            )

        return client

    def docker_environment(self) -> dict[str, str]:
        return self.session.export_environment()


@dataclass
class DS4ProviderRuntime(ProviderRuntime):
    config: BenchmarkConfig

    def __post_init__(self):
        directory = self.config.config_path.parent if self.config.config_path else Path.cwd()
        self.launch_config = normalize_ds4_launch_config(self.config.ds4, directory=directory)
        if self.config.lmstudio_load.has_load_overrides() or self.config.lmstudio_load.parallelism_sweep:
            raise ValueError("ds4 のロード設定は ds4 ブロックに指定してください。")
        if self.managed:
            validate_managed_ds4_base_url(self.config.api_base)
        elif self.config.runs.cold_runs:
            raise ValueError("ds4 の外部管理モードでは cold_runs: 0 を指定してください。")
        validate_ds4_request(self.config.request_parameters(), self.launch_config.get("context_length"))
        self.session = DS4Session(self.config.api_base, self.config.auth.bearer_token)
        self._server: ManagedDS4Server | None = None
        self._run_directory: Path | None = None

    @property
    def managed(self) -> bool:
        return self.launch_config["management"] == "managed"

    def begin_run(self, directory: Path, *, ds4_launch_config=None) -> None:
        if self._server is not None:
            raise RuntimeError("前のrunのds4プロセスを停止してから次のrunを開始してください。")
        if ds4_launch_config is not None:
            self.launch_config = copy.deepcopy(ds4_launch_config)
        self._run_directory = directory

    def _environment(self, requested_model: str, *, probe: bool = False) -> dict[str, Any]:
        info = ds4_environment_info(requested_model, base_url=self.session.base_url,
                                    api_key=self.session.api_key, probe=probe)
        info.update(requested_model=requested_model, server_management=self.launch_config["management"],
                    launch_config_sha256=ds4_launch_config_sha256(self.launch_config),
                    runtime={"engine": "ds4", "version": None, "source": "explicit_provider"})
        if self.managed:
            info["configured_context_length"] = self.launch_config["context_length"]
        return info

    def inspect_model(self, requested_model: str) -> dict[str, Any]:
        if self.managed and (self._server is None or not self._server.ready):
            # Before launch, validate files only. An unrelated listener must
            # never supply model evidence for a run that has not loaded yet.
            validate_ds4_launch_files(self.launch_config)
            return self._environment(requested_model)
        return self._loaded_model_info(requested_model)

    def _loaded_model_info(self, requested_model: str) -> dict[str, Any]:
        if self.managed and (self._server is None or not self._server.ready):
            raise RuntimeError("このrunが起動したds4サーバーは準備完了していません。")
        info = read_ds4_model_info(requested_model, base_url=self.session.base_url, api_key=self.session.api_key)
        context = info.get("context_length")
        if type(context) is not int or context <= 0:
            raise RuntimeError("DS4 model API did not report a valid context_length")
        validate_ds4_request(self.config.request_parameters(), context)
        if self.managed and (not self._server.ready or context != self.launch_config["context_length"]):
            raise RuntimeError("DS4 loaded context_length or owned process state changed after launch")
        # Compatibility aliases are not evidence of a GGUF, quantization,
        # sampler, binary version, or even a distinct physical model.
        return {
            **self._environment(requested_model),
            "provider": DS4_PROVIDER, "requested_model": requested_model,
            "identifier": info["id"], "model_id": info["id"], "state": "loaded",
            "model_api_status": "matched", "capture_status": "partial",
            "model_identity_scope": "server_compatibility_alias",
            "local_artifact_identity_status": "unavailable", "process_config_status": "unavailable",
            "loaded_context_length": context, "load_config": {"context_length": context},
            "load_config_scope": "context_length_only",
            "runtime": {"engine": "ds4", "version": None, "source": "explicit_provider"},
        }

    def prepare_model(self, requested_model: str, *, lmstudio_parallelism=None):
        if lmstudio_parallelism is not None:
            raise ValueError("ds4 の並列数はサーバー側で管理してください。")
        if not self.managed:
            info = self.inspect_model(requested_model)
            return info["identifier"], info
        if self._run_directory is None:
            raise RuntimeError("ds4 のロード前にrunのログディレクトリを指定してください。")
        if self._server is not None:
            raise RuntimeError("ds4 のロード前にこのrunの前回プロセスを停止してください。")
        self._server = ManagedDS4Server(
            requested_model.removeprefix("openai/"), base_url=self.session.base_url,
            api_key=self.session.api_key, config=self.launch_config,
            log_path=self._run_directory / "ds4-server.log",
            lock_directory=Path(tempfile.gettempdir()) / f"local-llm-bench-{os.getuid()}" / "ds4-port-locks",
            docker_access=self.config.mode == "docker_task",
        )
        began = time.perf_counter()
        try:
            self._server.start()
            info = self._loaded_model_info(requested_model)
        except BaseException:
            self._server.stop()
            raise
        info.update(load_status="loaded", load_time_seconds=time.perf_counter() - began)
        return info["identifier"], info

    def describe_model(self, requested_model: str):
        return self.inspect_model(requested_model)

    def unload_model(self, requested_model: str) -> list[UnloadResult]:
        if not self.managed:
            raise RuntimeError("ds4 のモデル操作は外部管理です。アンロードは実行できません。")
        if self._server is None:
            return []
        target = self._server.model
        self._server.stop()
        self._server = None
        return [UnloadResult(requested_model=requested_model, target=target, status="unloaded", message="owned ds4 process stopped")]

    def chat_client(self) -> Callable[..., Any]:
        def client(**kwargs):
            if self.managed and (self._server is None or not self._server.ready):
                raise RuntimeError("このrunが起動したds4サーバーは停止しています。")
            kwargs["api_base"] = self.session.base_url
            kwargs["model"] = kwargs["model"].removeprefix("openai/")
            kwargs["urlopen"] = self.session.urlopen
            kwargs.update(self.config.request.optional_parameters())
            kwargs = apply_ds4_sampling(kwargs, self.config.ds4_sampling)
            validate_ds4_request(kwargs)
            return stream_chat_completion(**kwargs)
        return client

    def docker_environment(self) -> dict[str, str]:
        # Empty explicitly overrides any unrelated ambient host credential.
        return {DS4_API_KEY_ENV: self.session.api_key or ""}


@dataclass
class OMLXProviderRuntime(ProviderRuntime):
    config: BenchmarkConfig

    def __post_init__(self):
        validate_omlx_request(self.config.request_parameters())
        self.session = OMLXSession(self.config.api_base, self.config.auth.bearer_token)

    def inspect_model(self, requested_model: str) -> dict[str, Any]:
        return self.session.model_info(requested_model)

    def prepare_model(self, requested_model: str, *, lmstudio_parallelism=None):
        if lmstudio_parallelism is not None:
            raise ValueError("oMLX load settings are managed in oMLX")
        info = self.inspect_model(requested_model)
        if info["state"] != "unloaded":
            raise RuntimeError("oMLX model is already loaded; cold timing cannot be verified")
        began = time.perf_counter()
        info = self.session.change_model(info, "load", timeout_sec=max(self.config.runs.timeout_sec, _MODEL_PREPARE_TIMEOUT_SEC))
        info.update(load_status="loaded", load_time_seconds=time.perf_counter() - began)
        return info["identifier"], info

    def describe_model(self, requested_model: str):
        return self.inspect_model(requested_model)

    def measurement_metadata(self, model_info):
        observed = self.inspect_model(model_info["requested_model"])
        if observed["state"] != "loaded" or any(observed[key] != model_info[key] for key in ("identifier", "model_key", "path")):
            raise RuntimeError("oMLX model identity or load state changed before measurement")
        return {**model_info, **observed}

    def unload_model(self, requested_model: str) -> list[UnloadResult]:
        info = self.inspect_model(requested_model)
        if info["state"] == "unloaded":
            return []
        self.session.change_model(info, "unload", timeout_sec=max(self.config.runs.timeout_sec, 30.0))
        return [UnloadResult(requested_model=requested_model, target=info["model_key"], status="unloaded", message="oMLX model unloaded")]

    def chat_client(self) -> Callable[..., Any]:
        def client(**kwargs):
            kwargs["api_base"] = self.session.base_url
            kwargs["urlopen"] = self.session.urlopen
            for key in OMLX_REQUEST_KEYS:
                kwargs.pop(key, None)
            kwargs.update(self.config.request_parameters())
            validate_omlx_request(kwargs)
            return stream_chat_completion(**kwargs)
        return client

    def docker_environment(self) -> dict[str, str]:
        return {OMLX_API_KEY_ENV: self.session.api_key or ""}


@dataclass
class MLXServeProviderRuntime(ProviderRuntime):
    """mlx-serve (MLX Core) over its public model API; the server itself is never started."""

    config: BenchmarkConfig

    def __post_init__(self):
        validate_mlx_serve_request(self.config.request_parameters())
        settings = self.config.mlx_serve or normalize_mlx_serve_config(None, directory=None)
        self.session = MLXServeSession(self.config.api_base, self.config.auth.bearer_token,
                                       model_dirs=settings["model_dirs"],
                                       uses_server_defaults=self.config.request.use_mlx_serve_defaults)

    def inspect_model(self, requested_model: str) -> dict[str, Any]:
        return self.session.model_info(requested_model)

    def prepare_model(self, requested_model: str, *, lmstudio_parallelism=None):
        if lmstudio_parallelism is not None:
            raise ValueError("mlx-serve load settings are managed by the mlx-serve process options")
        info = self.inspect_model(requested_model)
        if info["state"] != "unloaded":
            raise RuntimeError("mlx-serve model is already loaded; cold timing cannot be verified")
        began = time.perf_counter()
        info = self.session.change_model(info, "load", timeout_sec=max(self.config.runs.timeout_sec, _MODEL_PREPARE_TIMEOUT_SEC))
        info.update(load_status="loaded", load_time_seconds=time.perf_counter() - began)
        return info["identifier"], info

    def describe_model(self, requested_model: str):
        return self.inspect_model(requested_model)

    def measurement_metadata(self, model_info):
        observed = self.inspect_model(model_info["requested_model"])
        if observed["state"] != "loaded" or any(observed[key] != model_info[key] for key in ("identifier", "model_key", "path")):
            raise RuntimeError("mlx-serve model identity or load state changed before measurement")
        return {**model_info, **observed}

    def unload_model(self, requested_model: str) -> list[UnloadResult]:
        # The last SSE event can arrive before mlx-serve releases its request slot.
        # Wait outside inference timing, and still refuse to unload a busy server.
        info = self.session.model_info(requested_model, idle_timeout_sec=30.0)
        if info["state"] == "unloaded":
            return []
        self.session.change_model(info, "unload", timeout_sec=max(self.config.runs.timeout_sec, 30.0))
        return [UnloadResult(requested_model=requested_model, target=info["model_key"], status="unloaded", message="mlx-serve model unloaded")]

    def chat_client(self) -> Callable[..., Any]:
        def client(**kwargs):
            kwargs["api_base"] = self.session.base_url
            kwargs["urlopen"] = self.session.urlopen
            for key in MLX_SERVE_REQUEST_KEYS | {"min_p", "seed"}:
                kwargs.pop(key, None)
            kwargs.update(self.config.request_parameters())
            validate_mlx_serve_request(kwargs)
            return stream_chat_completion(**kwargs)
        return client

    def docker_environment(self) -> dict[str, str]:
        return {MLX_SERVE_API_KEY_ENV: self.session.api_key or ""}


def build_provider_runtime(config: BenchmarkConfig) -> ProviderRuntime:
    if config.provider == MLX_SERVE_PROVIDER:
        return MLXServeProviderRuntime(config=config)
    if config.provider == OMLX_PROVIDER:
        return OMLXProviderRuntime(config=config)
    if config.provider == DS4_PROVIDER:
        return DS4ProviderRuntime(config=config)
    if config.provider == LMSTUDIO_PROVIDER:
        return LMStudioProviderRuntime(config=config)
    if config.provider == UNSLOTH_STUDIO_PROVIDER:
        return UnslothStudioProviderRuntime(
            config=config,
            session=UnslothStudioAuthSession(config.auth, openai_api_base=config.api_base),
        )
    raise ValueError(f"Unsupported provider: {config.provider!r}")
