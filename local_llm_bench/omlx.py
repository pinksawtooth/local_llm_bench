"""oMLX public API transport and model identity; never starts the server."""
from __future__ import annotations

import io
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .http_boundary import (
    HTTPBoundaryError, MAX_HTTP_RESPONSE_BYTES, join_http_url,
    open_url_no_redirect, parse_content_length, read_bounded_response,
    strict_json_loads, validate_bearer_token, validate_http_base_url,
    validate_json_content_type,
)

DEFAULT_OMLX_API_BASE = "http://127.0.0.1:8000/v1"
OMLX_API_KEY_ENV = "OMLX_API_KEY"
OMLX_REQUEST_KEYS = frozenset({"temperature", "max_tokens", "top_p", "top_k", "min_p", "seed", "reasoning_effort"})
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_MAX_SETTINGS_BYTES = 1024 * 1024


def _omlx_settings_path() -> Path:
    """Follow oMLX's data-root precedence without importing the native runtime."""
    base = os.environ.get("OMLX_BASE_PATH")
    if not base:
        bootstrap = Path.home() / "Library/Application Support/oMLX/base-path"
        try:
            with bootstrap.open("r", encoding="utf-8") as stream:
                base = stream.read(4097).strip()
            if len(base) > 4096:
                raise ValueError("oMLX data-root path is too long")
        except FileNotFoundError:
            base = None
        except (OSError, UnicodeError):
            raise ValueError("oMLXの保存先を読み込めません。OMLX_BASE_PATH または OMLX_API_KEY を指定してください。") from None
    return (Path(base).expanduser() if base else Path.home() / ".omlx") / "settings.json"


def resolve_omlx_api_key(base_url: str, explicit_key: str | None = None) -> str | None:
    """Read a saved key only for this user's matching local oMLX endpoint.

    Resolve once on the host; workers receive the selected key via their existing
    environment. A failed explicit credential never falls back to another key.
    """
    supplied = explicit_key if explicit_key is not None else os.environ.get(OMLX_API_KEY_ENV)
    if supplied is not None:
        return validate_bearer_token(supplied)
    parsed = urlsplit(omlx_base_url(base_url))
    if parsed.scheme != "http" or parsed.hostname not in _LOCAL_HOSTS:
        return None
    path = _omlx_settings_path()
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_SETTINGS_BYTES + 1)
        settings = strict_json_loads(raw, max_bytes=_MAX_SETTINGS_BYTES)
        if not isinstance(settings, dict):
            raise ValueError("invalid settings")
        server = settings.get("server", {})
        auth = settings.get("auth", {})
        if not isinstance(server, dict) or not isinstance(auth, dict):
            raise ValueError("invalid settings sections")
        port = server.get("port", 8000)
        host = server.get("host", "127.0.0.1")
        if type(port) is not int or not 1 <= port <= 65535 or not isinstance(host, str):
            raise ValueError("invalid server settings")
        if host not in (_LOCAL_HOSTS | {"0.0.0.0", "::"}) or port != (parsed.port or 80):
            return None
        return validate_bearer_token(auth.get("api_key"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError):
        # JSON parse errors may contain arbitrary settings values, including keys.
        raise ValueError("oMLXの保存済み認証設定を読み込めません。OMLX_API_KEY を明示してください。") from None


def omlx_base_url(value: str) -> str:
    base = validate_http_base_url(value)
    if urlsplit(base).path != "/v1":
        raise ValueError("oMLX api_base must end in /v1 (without a proxy path prefix)")
    return base


def docker_omlx_base_url(value: str) -> str:
    parsed = urlsplit(omlx_base_url(value))
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        parsed = parsed._replace(netloc="host.docker.internal" + (f":{parsed.port}" if parsed.port else ""))
    return urlunsplit(parsed)


def validate_omlx_request(parameters: dict) -> None:
    for key in OMLX_REQUEST_KEYS:
        value = parameters.get(key)
        if value is None:
            continue
        if key == "reasoning_effort":
            valid = (isinstance(value, str) and 0 < len(value) <= 64
                     and value == value.strip() and all(32 <= ord(char) < 127 for char in value))
        elif key in {"max_tokens", "top_k", "seed"}:
            valid = type(value) is int and (1 if key == "max_tokens" else 0) <= value <= 2**31 - 1
        else:
            valid = (type(value) in {int, float} and math.isfinite(value)
                     and 0 <= value <= (2 if key == "temperature" else 1)
                     and (key != "top_p" or value > 0))
        if not valid:
            raise ValueError(f"Invalid oMLX request.{key}")


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
            raise HTTPBoundaryError("oMLX streaming response is too large")
        return raw


class OMLXSession:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = omlx_base_url(base_url)
        self.api_key = validate_bearer_token(api_key)
        self.origin = self.base_url.removesuffix("/v1")

    def _open(self, request, timeout):
        # Always replace caller credentials; never forward credentials across redirects.
        request.remove_header("Authorization")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            return open_url_no_redirect(request, timeout_sec=timeout)
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            if status == 401:
                detail = "APIキーが受け付けられませんでした" if self.api_key else "APIキーが送信されていません"
                raise RuntimeError(f"oMLX API returned HTTP 401: {detail}。oMLXで有効なAPIキーを OMLX_API_KEY に設定してください。") from None
            raise RuntimeError(f"oMLX API returned HTTP {status}; check authentication, model state and public API support") from None
        except TimeoutError:
            raise TimeoutError("oMLX API timed out") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("oMLX API timed out") from None
            raise RuntimeError("oMLX API is unavailable") from None
        except OSError:
            raise RuntimeError("oMLX API is unavailable") from None

    def request_json(self, path: str, *, method="GET", timeout_sec=15.0):
        allowed = (method == "GET" and path in {"/api/status", "/v1/models", "/v1/models/status"}) or (
            method == "POST" and re.fullmatch(r"/v1/models/[A-Za-z0-9_][A-Za-z0-9_.:-]*/(?:load|unload)", path)
        )
        if not allowed:
            raise HTTPBoundaryError("Unsupported oMLX public API endpoint")
        request = urllib.request.Request(join_http_url(self.origin, path), method=method,
                                         data=b"" if method == "POST" else None,
                                         headers={"Accept": "application/json"})
        with self._open(request, timeout_sec) as response:
            validate_json_content_type(response.headers)
            result = strict_json_loads(read_bounded_response(response))
        if not isinstance(result, dict):
            raise RuntimeError("oMLX API response must be an object")
        return result

    def urlopen(self, request: urllib.request.Request, *, timeout: float):
        if request.full_url != join_http_url(self.base_url, "/chat/completions") or request.get_method() != "POST":
            raise HTTPBoundaryError("oMLX completion endpoint does not match the configured server")
        response = self._open(request, timeout)
        try:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type == "text/event-stream":
                length = parse_content_length(response.headers)
                if length is not None and length > MAX_HTTP_RESPONSE_BYTES:
                    raise HTTPBoundaryError("oMLX streaming response is too large")
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

    def idle_status(self) -> dict:
        status = self.request_json("/api/status")
        if status.get("status") != "ok" or not isinstance(status.get("version"), str) or not status["version"]:
            raise RuntimeError("oMLX did not report its running version and ready status")
        for key in ("active_requests", "waiting_requests", "models_loading"):
            count = status.get(key)
            if type(count) is not int or count < 0:
                raise RuntimeError(f"oMLX did not report {key}; model operations cannot be verified")
            if count:
                raise RuntimeError(f"oMLX is busy ({key}={count}); model operations were not performed")
        return status

    def model_info(self, requested_model: str) -> dict:
        from pathlib import Path

        status = self.idle_status()
        inventory = self.request_json("/v1/models").get("data")
        models = self.request_json("/v1/models/status").get("models")
        if (not isinstance(inventory, list) or not isinstance(models, list)
                or any(not isinstance(item, dict) for item in [*inventory, *models])):
            raise RuntimeError("oMLX returned an invalid model inventory")
        matches = [item for item in models if requested_model in {item.get("id"), item.get("model_alias")}]
        if len(matches) != 1:
            raise RuntimeError(f"oMLX model must match one exact ID or alias: {requested_model}")
        entry = matches[0]
        source = entry.get("source_model_id") or entry.get("id")
        physical = [item for item in models if item.get("id") == source and not item.get("source_model_id")]
        if len(physical) != 1 or not isinstance(source, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*", source):
            raise RuntimeError("oMLX did not report an unambiguous model ID supported by its load API")
        target = physical[0]
        api_model = entry.get("model_alias") or entry.get("id")
        exposed = [item for item in inventory if item.get("id") == api_model and item.get("owned_by") == "omlx"]
        if len(exposed) != 1:
            raise RuntimeError("The selected model is not exposed by oMLX /v1/models")
        if target.get("is_loading") is not False:
            raise RuntimeError("oMLX model is loading or its loading state is unknown")
        if target.get("distributed") or target.get("model_type") not in {"llm", "vlm"}:
            raise RuntimeError("oMLX benchmark requires a local LLM or VLM")
        if type(target.get("loaded")) is not bool:
            raise RuntimeError("oMLX did not report the model load state")
        path = target.get("model_path")
        local = urlsplit(self.base_url).hostname in {"localhost", "127.0.0.1", "::1"}
        local_path = path if local and isinstance(path, str) and Path(path).is_absolute() else ""
        return {
            "provider": "omlx", "requested_model": requested_model, "identifier": api_model,
            "model_key": source, "display_name": api_model, "format": "MLX",
            "path": local_path, "reported_model_path": path,
            "architecture": target.get("config_model_type"),
            "state": "loaded" if target["loaded"] else "unloaded",
            "runtime": {"engine": "oMLX", "version": status["version"], "source": "provider_status"},
            "load_config": {"context_length": entry.get("max_context_window"), "pinned": target.get("pinned")},
            "load_config_scope": "context_length_only", "server_management": "external_model_api",
            "inference_defaults": {"max_tokens": entry.get("max_tokens")},
            "runtime_features": {key: status.get(key) for key in ("custom_kernels", "ane_prefill")},
            # Architecture defaults are not the active request's thinking or MTP state.
            "reasoning_capabilities": {key: target.get(key) for key in ("thinking_default", "preserve_thinking_default")},
            "reported_inference": {},
        }

    def change_model(self, info: dict, action: str, *, timeout_sec: float) -> dict:
        self.idle_status()
        target = info["model_key"]
        result = self.request_json(f"/v1/models/{target}/{action}", method="POST", timeout_sec=timeout_sec)
        if result.get("status") != "ok" or result.get("model_id") != target:
            raise RuntimeError(f"oMLX did not confirm model {action}")
        if action == "load" and str(result.get("message", "")).startswith("Already loaded:"):
            raise RuntimeError("oMLX model was already loaded; cold timing cannot be verified")
        observed = self.model_info(info["requested_model"])
        if (observed["model_key"] != target or observed["identifier"] != info["identifier"]
                or observed["path"] != info["path"] or observed["state"] != ("loaded" if action == "load" else "unloaded")):
            raise RuntimeError(f"oMLX model identity or state changed during {action}")
        return observed
