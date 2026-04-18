from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Optional

from .config import (
    BenchmarkConfig,
    LMSTUDIO_PROVIDER,
    UNSLOTH_STUDIO_PROVIDER,
)
from .lmstudio_api import stream_chat_completion
from .lmstudio_cli import (
    ModelInfo,
    UnloadResult,
    _build_quantization_fields,
    _matching_entries,
    describe_loaded_model,
    unload_matching_models,
)
from .unsloth_api import UnslothStudioAuthSession

_MODEL_PREPARE_TIMEOUT_SEC = 300.0
_QUANTIZATION_TOKEN_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"((?:IQ|Q)\d+(?:_[A-Za-z0-9]+)*|(?:BF|FP|F)\d+|MXFP\d+|\d+BIT)"
    r"(?![A-Za-z0-9])"
)
_AUXILIARY_GGUF_MARKERS = ("mmproj", "mm-proj", "projector")


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
        return str(root) if root.suffix.lower() == ".gguf" else ""
    if not root.is_dir():
        return ""

    candidates = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".gguf"]
    if not candidates:
        return ""

    def sort_key(path: Path) -> tuple[int, int, int, str]:
        lowered_name = path.name.lower()
        is_auxiliary = any(marker in lowered_name for marker in _AUXILIARY_GGUF_MARKERS)
        has_quantization = bool(_extract_quantization_token(path.name))
        return (
            1 if is_auxiliary else 0,
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
        return requested_model
    if entry:
        return _first_text(
            entry.get("path"),
            entry.get("indexedModelIdentifier"),
            entry.get("identifier"),
            entry.get("modelKey"),
            requested_model,
        )
    return requested_model


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
    def prepare_model(self, requested_model: str) -> tuple[str, Optional[dict[str, Any]]]:
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

    def prepare_model(self, requested_model: str) -> tuple[str, Optional[dict[str, Any]]]:
        model_info = describe_loaded_model(requested_model, api_base=self.config.api_base)
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
        return unload_matching_models(requested_model)

    def chat_client(self) -> Callable[..., Any]:
        return stream_chat_completion


@dataclass
class UnslothStudioProviderRuntime(ProviderRuntime):
    config: BenchmarkConfig
    session: UnslothStudioAuthSession

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

    def prepare_model(self, requested_model: str) -> tuple[str, Optional[dict[str, Any]]]:
        timeout_sec = max(float(self.config.runs.timeout_sec), _MODEL_PREPARE_TIMEOUT_SEC)
        entry = self._match_available_entry(requested_model, timeout_sec=timeout_sec)
        load_target = _first_text(
            entry.get("path"),
            entry.get("identifier"),
            entry.get("modelKey"),
            entry.get("indexedModelIdentifier"),
            requested_model,
        )
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
        return api_model, model_info

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
        unload_target = requested_model
        try:
            if _looks_like_filesystem_path(requested_model):
                unload_target = requested_model
            else:
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
            )

        return client

    def docker_environment(self) -> dict[str, str]:
        return self.session.export_environment()


def build_provider_runtime(config: BenchmarkConfig) -> ProviderRuntime:
    if config.provider == LMSTUDIO_PROVIDER:
        return LMStudioProviderRuntime(config=config)
    if config.provider == UNSLOTH_STUDIO_PROVIDER:
        return UnslothStudioProviderRuntime(
            config=config,
            session=UnslothStudioAuthSession(config.auth, openai_api_base=config.api_base),
        )
    raise ValueError(f"Unsupported provider: {config.provider!r}")
