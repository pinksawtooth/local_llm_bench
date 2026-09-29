"""mlx-serve (MLX Core) public API transport and model identity; never starts the server.

Based on the HTTP surface of the installed ``mlx-serve 26.9.4``:
``GET /health``, ``GET /api/version``, ``GET /v1/models``, ``GET /props``,
``GET /metrics.json`` (only with ``--metrics``), ``POST /v1/load-model`` and
``POST /v1/unload-model`` with ``{"model": "<discovered id>"}``.
"""
from __future__ import annotations

import http.client
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .http_boundary import (
    HTTPBoundaryError, MAX_HTTP_ERROR_BYTES, MAX_HTTP_RESPONSE_BYTES, join_http_url,
    open_url_no_redirect, parse_content_length, read_bounded_response,
    strict_json_loads, validate_bearer_token, validate_http_base_url,
    validate_json_content_type,
)

DEFAULT_MLX_SERVE_API_BASE = "http://127.0.0.1:11234/v1"
MLX_SERVE_API_KEY_ENV = "MLX_SERVE_API_KEY"
# Chat fields documented by the server console: max_tokens, temperature, top_p,
# top_k, reasoning_effort. min_p / seed are not documented, so they are rejected.
MLX_SERVE_REQUEST_KEYS = frozenset({"temperature", "max_tokens", "top_p", "top_k", "reasoning_effort"})
DEFAULT_MLX_SERVE_MODEL_DIRS = ("~/.mlx-serve/models",)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_MAX_MODEL_DIRS = 8
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./:-]*\Z")
_GET_ENDPOINTS = frozenset({"/health", "/api/version", "/v1/models", "/props", "/metrics.json"})
_POST_ENDPOINTS = frozenset({"/v1/load-model", "/v1/unload-model"})
_PROPS_SETTING_KEYS = ("version", "engine", "kv_quant", "kv_attn_mode", "decode_attn_quant",
                       "prefill_chunk", "max_concurrent", "mtp", "pld", "drafter", "prefix_cache")
# Fixed descriptions only: server error messages can contain paths, prompts or
# credentials. These error types are emitted by the installed mlx-serve 26.9.4.
_API_ERROR_DETAILS = {
    "out_of_memory": "モデル用メモリが不足しています。mlx-serveのサーバーログで必要量・利用可能量・常駐メモリ上限を確認し、ほかのモデルやアプリの使用量を減らしてから再実行してください。",
    "model_load_failed": "モデルのロードに失敗しました。mlx-serveのサーバーログで原因を確認してください。",
    "model_unload_failed": "モデルのアンロードに失敗しました。mlx-serveのサーバーログで原因を確認してください。",
}


def _api_error_detail(response) -> str | None:
    """Extract a known error classification without exposing the response body."""
    try:
        validate_json_content_type(response.headers)
        payload = strict_json_loads(read_bounded_response(response, max_bytes=MAX_HTTP_ERROR_BYTES),
                                    max_bytes=MAX_HTTP_ERROR_BYTES)
    except (HTTPBoundaryError, OSError, http.client.HTTPException):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None
    for field in ("code", "type"):
        code = error.get(field)
        if isinstance(code, str) and code in _API_ERROR_DETAILS:
            return f"{code}: {_API_ERROR_DETAILS[code]}"
    return None


class MLXServeAPIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def mlx_serve_base_url(value: str) -> str:
    base = validate_http_base_url(value)
    if not urlsplit(base).path:
        return join_http_url(base, "/v1")
    if urlsplit(base).path != "/v1":
        raise ValueError("mlx-serve api_base must end in /v1 (without a proxy path prefix)")
    return base


def docker_mlx_serve_base_url(value: str) -> str:
    parsed = urlsplit(mlx_serve_base_url(value))
    if parsed.hostname in _LOCAL_HOSTS:
        parsed = parsed._replace(netloc="host.docker.internal" + (f":{parsed.port}" if parsed.port else ""))
    return urlunsplit(parsed)


def validate_mlx_serve_model_id(value: str) -> str:
    """A discovered id as listed by /v1/models; never a filesystem path or a LAN peer."""
    if (not isinstance(value, str) or not _MODEL_ID_RE.match(value) or "@" in value or "//" in value
            or any(segment in {".", ".."} for segment in value.split("/")) or value.endswith("/")):
        raise ValueError(f"mlx-serve model は /v1/models に表示されるIDを指定してください: {value!r}")
    return value


def validate_mlx_serve_request(parameters: dict) -> None:
    unsupported = {"min_p", "seed"} & {key for key, value in parameters.items() if value is not None}
    if unsupported:
        raise ValueError("mlx-serve request では " + ", ".join(sorted(unsupported)) + " を指定できません。")
    for key in MLX_SERVE_REQUEST_KEYS:
        value = parameters.get(key)
        if value is None:
            continue
        if key == "reasoning_effort":
            valid = (isinstance(value, str) and 0 < len(value) <= 64
                     and value == value.strip() and all(32 <= ord(char) < 127 for char in value))
        elif key in {"max_tokens", "top_k"}:
            valid = type(value) is int and (1 if key == "max_tokens" else 0) <= value <= 2**31 - 1
        else:
            valid = (type(value) in {int, float} and math.isfinite(value)
                     and 0 <= value <= (2 if key == "temperature" else 1)
                     and (key != "top_p" or value > 0))
        if not valid:
            raise ValueError(f"Invalid mlx-serve request.{key}")


def normalize_mlx_serve_config(block, *, directory: Path | None) -> dict:
    """Optional ``mlx_serve`` YAML block. Only ``model_dirs`` (for local weight hashing) is accepted."""
    if block is None:
        block = {}
    if not isinstance(block, dict) or set(block) - {"model_dirs"}:
        raise ValueError("mlx_serve は model_dirs のマッピングです。")
    raw_dirs = block.get("model_dirs", list(DEFAULT_MLX_SERVE_MODEL_DIRS))
    if (not isinstance(raw_dirs, list) or len(raw_dirs) > _MAX_MODEL_DIRS
            or any(not isinstance(item, str) or not item.strip() or item != item.strip() for item in raw_dirs)):
        raise ValueError(f"mlx_serve.model_dirs は空でない文字列の配列です（最大{_MAX_MODEL_DIRS}件）。")
    resolved = []
    for item in raw_dirs:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = (directory or Path.cwd()) / path
        resolved.append(str(path))
    return {"model_dirs": resolved}


def resolve_local_model_path(model_id: str, model_dirs, base_url: str) -> str:
    """Infer the served directory as mlx-serve's discovery would (``<model-dir>/<id>``).

    This is an inference from the configured folders, not a value reported by
    the API, so callers record it as such. Remote servers are never hashed.
    """
    if urlsplit(base_url).hostname not in _LOCAL_HOSTS or not model_dirs:
        return ""
    candidates = []
    for root in model_dirs:
        candidate = Path(root) / model_id
        if (candidate.is_dir() and (candidate / "config.json").is_file()) or (
                candidate.is_file() and candidate.suffix == ".gguf"):
            candidates.append(candidate)
    if len(candidates) > 1:
        raise RuntimeError(f"mlx-serve model {model_id!r} が複数の model_dirs に存在します。mlx_serve.model_dirs を絞ってください。")
    return str(candidates[0]) if candidates else ""


class _BoundedStream:
    def __init__(self, response):
        self.response = response
        self.headers = response.headers
        self.status = getattr(response, "status", 200)
        self.remaining = MAX_HTTP_RESPONSE_BYTES

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.response.close()

    def __iter__(self):
        return self

    def __next__(self):
        raw = self.response.readline(self.remaining + 1)
        if not raw:
            raise StopIteration
        self.remaining -= len(raw)
        if self.remaining < 0:
            raise HTTPBoundaryError("mlx-serve streaming response is too large")
        return raw


class MLXServeSession:
    def __init__(self, base_url: str, api_key: str | None = None, *, model_dirs=None, uses_server_defaults: bool = False):
        self.base_url = mlx_serve_base_url(base_url)
        self.api_key = validate_bearer_token(api_key)
        self.origin = self.base_url.removesuffix("/v1")
        self.model_dirs = list(model_dirs or [])
        self.uses_server_defaults = uses_server_defaults

    def _open(self, request, timeout):
        # Always replace caller credentials; never forward credentials across redirects.
        request.remove_header("Authorization")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            return open_url_no_redirect(request, timeout_sec=timeout)
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                detail = _api_error_detail(exc) if status in {500, 503} else None
            finally:
                exc.close()
            if status == 401:
                detail = "APIキーが受け付けられませんでした" if self.api_key else "APIキーが送信されていません"
                raise MLXServeAPIError(f"mlx-serve API returned HTTP 401: {detail}。mlx-serveの --api-key と同じ値を {MLX_SERVE_API_KEY_ENV} に設定してください。", 401) from None
            if detail:
                raise MLXServeAPIError(f"mlx-serve API returned HTTP {status}: {detail}", status) from None
            if status == 503 and request.full_url == join_http_url(self.origin, "/v1/load-model"):
                raise MLXServeAPIError("mlx-serve API returned HTTP 503: モデルをロードできませんでした。mlx-serveのサーバーログでメモリ不足やモデルのロードエラーを確認してください。", status) from None
            raise MLXServeAPIError(f"mlx-serve API returned HTTP {status}; check authentication, model state and public API support", status) from None
        except TimeoutError:
            raise TimeoutError("mlx-serve API timed out") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("mlx-serve API timed out") from None
            raise MLXServeAPIError("mlx-serve API is unavailable") from None
        except OSError:
            raise MLXServeAPIError("mlx-serve API is unavailable") from None

    def request_json(self, path: str, *, method="GET", body: dict | None = None, timeout_sec=15.0):
        if not ((method == "GET" and body is None and path in _GET_ENDPOINTS)
                or (method == "POST" and isinstance(body, dict) and path in _POST_ENDPOINTS)):
            raise HTTPBoundaryError("Unsupported mlx-serve public API endpoint")
        headers = {"Accept": "application/json"}
        data = None
        if method == "POST":
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(join_http_url(self.origin, path), method=method, data=data, headers=headers)
        with self._open(request, timeout_sec) as response:
            validate_json_content_type(response.headers)
            result = strict_json_loads(read_bounded_response(response))
        if not isinstance(result, dict):
            raise RuntimeError("mlx-serve API response must be an object")
        return result

    def urlopen(self, request: urllib.request.Request, *, timeout: float):
        if request.full_url != join_http_url(self.base_url, "/chat/completions") or request.get_method() != "POST":
            raise HTTPBoundaryError("mlx-serve completion endpoint does not match the configured server")
        response = self._open(request, timeout)
        try:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type == "text/event-stream":
                length = parse_content_length(response.headers)
                if length is not None and length > MAX_HTTP_RESPONSE_BYTES:
                    raise HTTPBoundaryError("mlx-serve streaming response is too large")
                return _BoundedStream(response)
            validate_json_content_type(response.headers)
            raw = read_bounded_response(response)
            strict_json_loads(raw)
            buffered = io.BytesIO(raw)
            buffered.headers = response.headers
            buffered.status = getattr(response, "status", 200)
            response.close()
            return buffered
        except BaseException:
            response.close()
            raise

    def server_status(self) -> dict:
        health = self.request_json("/health")
        if health.get("status") != "ok":
            raise RuntimeError("mlx-serve did not report a ready status")
        version = self.request_json("/api/version").get("version")
        if not isinstance(version, str) or not version:
            raise RuntimeError("mlx-serve did not report its running version")
        return {"version": version}

    def activity(self, *, wait_timeout_sec: float = 0.0) -> dict | None:
        """Require idle metrics; optionally allow server cleanup after the last SSE event."""
        deadline = time.monotonic() + wait_timeout_sec
        busy_message = None
        while True:
            remaining = deadline - time.monotonic()
            if wait_timeout_sec > 0 and remaining <= 0 and busy_message:
                raise RuntimeError(f"{busy_message}; idle wait timed out after {wait_timeout_sec:g}s")
            timeout = min(15.0, max(0.001, remaining)) if wait_timeout_sec else 15.0
            try:
                metrics = self.request_json("/metrics.json", timeout_sec=timeout)
            except MLXServeAPIError as exc:
                if exc.status == 404:
                    return None
                raise
            gauges = metrics.get("gauges")
            if not isinstance(gauges, dict):
                raise RuntimeError("mlx-serve metrics did not report request gauges")
            counts = {}
            for key in ("requests_running", "requests_waiting"):
                count = gauges.get(key)
                if type(count) is not int or count < 0:
                    raise RuntimeError(f"mlx-serve did not report {key}; model operations cannot be verified")
                counts[key] = count
            if not any(counts.values()):
                return counts
            key = next(key for key, count in counts.items() if count)
            message = f"mlx-serve is busy ({key}={counts[key]}); model operations were not performed"
            busy_message = message
            remaining = deadline - time.monotonic()
            if wait_timeout_sec <= 0 or remaining <= 0:
                if wait_timeout_sec > 0:
                    message += f"; idle wait timed out after {wait_timeout_sec:g}s"
                raise RuntimeError(message)
            time.sleep(min(0.25, remaining))

    def _props_for(self, model_id: str) -> dict:
        props = self.request_json("/props")
        defaults = props.get("default_generation_settings")
        settings = props.get("settings")
        if not isinstance(defaults, dict) or defaults.get("model") != model_id or not isinstance(settings, dict):
            return {"props_scope": "other_default_model"}
        features = {key: settings.get(key) for key in _PROPS_SETTING_KEYS if key in settings}
        features["props_scope"] = "default_model"
        if type(defaults.get("n_ctx")) is int:
            features["n_ctx"] = defaults["n_ctx"]
        return features

    def model_info(self, requested_model: str, *, idle_timeout_sec: float = 0.0) -> dict:
        validate_mlx_serve_model_id(requested_model)
        status = self.server_status()
        activity = self.activity(wait_timeout_sec=idle_timeout_sec)
        inventory = self.request_json("/v1/models").get("data")
        if not isinstance(inventory, list) or any(not isinstance(item, dict) for item in inventory):
            raise RuntimeError("mlx-serve returned an invalid model inventory")
        matches = [item for item in inventory if item.get("id") == requested_model]
        if len(matches) != 1:
            raise RuntimeError(f"mlx-serve model must match exactly one /v1/models id: {requested_model}")
        entry = matches[0]
        if entry.get("owned_by") != "mlx-serve":
            raise RuntimeError("The selected model is not served by mlx-serve itself")
        capabilities = entry.get("capabilities")
        if capabilities is not None and (not isinstance(capabilities, list) or "chat" not in capabilities):
            raise RuntimeError("mlx-serve benchmark requires a chat-capable model")
        if entry.get("error"):
            raise RuntimeError("mlx-serve reported a model error; resolve it in mlx-serve first")
        loaded = entry.get("loaded")
        if type(loaded) is not bool:
            raise RuntimeError("mlx-serve did not report the model load state")
        state = entry.get("state")
        if loaded and state != "ready":
            raise RuntimeError("mlx-serve model is loading or its loading state is unknown")
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        engine = meta.get("engine") if isinstance(meta.get("engine"), str) else None
        is_gguf = requested_model.lower().endswith(".gguf") or engine in {"llama", "llama.cpp", "ds4"}
        path = resolve_local_model_path(requested_model, self.model_dirs, self.base_url)
        context_length = meta.get("context_length") if type(meta.get("context_length")) is int else (
            meta.get("max_model_len") if type(meta.get("max_model_len")) is int else None)
        defaults = {key: meta.get(f"gen_{key}") for key in ("temperature", "top_p", "top_k")}
        defaults["max_tokens"] = meta.get("model_max_tokens")
        features = self._props_for(requested_model) if loaded else {}
        reported = {}
        if loaded and self.uses_server_defaults and all(defaults[key] is not None for key in ("temperature", "top_p", "top_k")):
            # Server-reported defaults for requests that omit sampling fields; the
            # benchmark omits them, so these are the effective values.
            reported["sampling"] = {key: defaults[key] for key in ("temperature", "top_p", "top_k")}
        mtp, pld = features.get("mtp"), features.get("pld")
        if (loaded and isinstance(mtp, dict) and isinstance(pld, dict)
                and type(mtp.get("default_on")) is bool and type(pld.get("default_on")) is bool
                and type(meta.get("mtp_loaded")) is bool):
            reported["speculative_decoding"] = {"mtp_loaded": meta["mtp_loaded"], "mtp_default_on": mtp["default_on"],
                                                "pld_default_on": pld["default_on"], "drafter": features.get("drafter")}
        return {
            "provider": "mlx_serve", "requested_model": requested_model, "identifier": requested_model,
            "model_key": requested_model, "display_name": requested_model,
            "format": "GGUF" if is_gguf else "MLX",
            "quantization_name": meta.get("quantization") if isinstance(meta.get("quantization"), str) else None,
            "path": path, "reported_model_path": None,
            "local_artifact_identity_status": "inferred_from_model_dirs" if path else "unavailable_from_api",
            "model_identity_scope": "server_model_id",
            "architecture": meta.get("architecture"),
            "state": "loaded" if loaded else "unloaded",
            "runtime": {"engine": "mlx-serve", "version": status["version"], "source": "provider_status"},
            "load_config": {"context_length": context_length, "engine": engine, "is_moe": meta.get("is_moe"),
                            "mtp_available": meta.get("mtp_available")},
            "load_config_scope": "context_length_only", "server_management": "external_model_api",
            "inference_defaults": defaults,
            "runtime_features": features,
            "reported_inference": reported,
            "server_activity": activity,
        }

    def change_model(self, info: dict, action: str, *, timeout_sec: float) -> dict:
        self.activity()
        target = info["model_key"]
        result = self.request_json(f"/v1/{action}-model", method="POST", body={"model": target}, timeout_sec=timeout_sec)
        model = result.get("model")
        if (not isinstance(model, dict) or model.get("loaded") is not (action == "load")
                or model.get("id") not in ({target} if action == "load" else {target, ""})):
            raise RuntimeError(f"mlx-serve did not confirm model {action}")
        observed = self.model_info(info["requested_model"])
        if (observed["model_key"] != target or observed["identifier"] != info["identifier"]
                or observed["path"] != info["path"] or observed["state"] != ("loaded" if action == "load" else "unloaded")):
            raise RuntimeError(f"mlx-serve model identity or state changed during {action}")
        return observed
