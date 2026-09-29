"""Adapted from RevBench: explicit DS4 transport and bounded HTTP metadata."""

from __future__ import annotations

import urllib.error
import urllib.request
import io
import math
import hashlib
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .http_boundary import (
    HTTPBoundaryError,
    MAX_HTTP_RESPONSE_BYTES,
    join_http_url,
    open_url_no_redirect,
    read_bounded_response,
    strict_json_loads,
    validate_bearer_token,
    validate_http_base_url,
    validate_json_content_type,
    parse_content_length,
)


DEFAULT_DS4_API_BASE = "http://127.0.0.1:8000/v1"
DS4_API_KEY_ENV = "DS4_API_KEY"
DS4_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}
DS4_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "min_p", "seed"})


def normalize_ds4_sampling_config(value: Any) -> dict[str, Any]:
    """Match RevBench's model policy without filling in server defaults."""
    if value is None:
        return {"source": "server_defaults"}
    if not isinstance(value, dict) or set(value) - ({"source"} | DS4_SAMPLING_KEYS):
        raise ValueError("ds4_sampling contains invalid fields")
    source = value.get("source")
    if source not in ("server_defaults", "model_config", "run_config"):
        raise ValueError("ds4_sampling.source must be server_defaults, model_config, or run_config")
    if source != "model_config":
        if set(value) != {"source"}:
            raise ValueError("Only model_config may contain sampling values")
        return {"source": source}
    if not DS4_SAMPLING_KEYS.intersection(value):
        raise ValueError("model_config requires at least one sampling value")
    for key in DS4_SAMPLING_KEYS.intersection(value):
        number = value[key]
        if key in {"top_k", "seed"}:
            maximum = 1024 if key == "top_k" else 2**53 - 1
            if type(number) is not int or not 0 <= number <= maximum:
                raise ValueError(f"Invalid DS4 sampling {key}")
        elif (
            type(number) not in {int, float} or not math.isfinite(number)
            or not 0 <= number <= (2 if key == "temperature" else 1)
            or (key == "top_p" and number == 0)
        ):
            raise ValueError(f"Invalid DS4 sampling {key}")
    return dict(value)


def apply_ds4_sampling(parameters: dict[str, Any], policy: Any) -> dict[str, Any]:
    """Resolve sampler precedence while preserving output budget and reasoning."""
    policy = normalize_ds4_sampling_config(policy)
    result = {key: value for key, value in parameters.items() if key not in DS4_SAMPLING_KEYS}
    selected = parameters if policy["source"] == "run_config" else policy if policy["source"] == "model_config" else {}
    sampling = {key: selected[key] for key in DS4_SAMPLING_KEYS if selected.get(key) is not None}
    if sampling:
        normalize_ds4_sampling_config({"source": "model_config", **sampling})
    return {**result, **sampling}


def docker_ds4_base_url(base_url: str) -> str:
    parsed = urlsplit(_ds4_base_url(base_url))
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        netloc = "host.docker.internal" + (f":{parsed.port}" if parsed.port else "")
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    return urlunsplit(parsed)


def validate_ds4_request(parameters: dict, context_length: int | None = None) -> None:
    effort = parameters.get("reasoning_effort")
    if effort is not None and effort not in DS4_REASONING_EFFORTS:
        raise ValueError("ds4 の reasoning_effort は minimal / low / medium / high / xhigh / max を指定してください。ultra は未対応です。")
    sampling = {key: parameters[key] for key in DS4_SAMPLING_KEYS if parameters.get(key) is not None}
    if sampling:
        normalize_ds4_sampling_config({"source": "model_config", **sampling})
    maximum = parameters.get("max_tokens")
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("ds4 の max_tokens は正の整数である必要があります。")
    # Input token counts are not available from /models; do not invent an input
    # allowance or silently clamp the operator's requested output budget.
    if context_length is not None and maximum >= context_length:
        raise ValueError("request.max_tokens は ds4 が報告した context_length より小さくしてください（入力にも領域が必要です）。")


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
            raise HTTPBoundaryError("DS4 streaming response is too large")
        return raw


class DS4Session:
    """Authenticated completions on one explicit endpoint, with no redirects."""

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = _ds4_base_url(base_url)
        self.api_key = validate_bearer_token(api_key)

    def urlopen(self, request: urllib.request.Request, *, timeout: float):
        if request.full_url != join_http_url(self.base_url, "/chat/completions") or request.get_method() != "POST":
            raise HTTPBoundaryError("DS4 completion endpoint does not match the configured server")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            response = open_url_no_redirect(request, timeout_sec=timeout)
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise RuntimeError(f"DS4 completion API returned HTTP {status}") from None
        except TimeoutError:
            raise TimeoutError("DS4 completion API timed out") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("DS4 completion API timed out") from None
            raise RuntimeError("DS4 completion API is unavailable") from None
        except OSError:
            raise RuntimeError("DS4 completion API is unavailable") from None
        try:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type == "text/event-stream":
                length = parse_content_length(response.headers)
                if length is not None and length > MAX_HTTP_RESPONSE_BYTES:
                    raise HTTPBoundaryError("DS4 streaming response is too large")
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


def normalize_ds4_launch_config(value: Any, *, directory: Path) -> dict[str, Any]:
    """Validate host-only launch settings without opening model files."""
    if not isinstance(value, dict):
        raise ValueError("ds4 must be an object")
    allowed = {"management", "server_path", "model_path", "context_length", "startup_timeout_sec", "server_args"}
    if set(value) - allowed:
        raise ValueError("ds4 contains unknown launch settings")
    management = value.get("management", "managed")
    if management == "external":
        if set(value) != {"management"}:
            raise ValueError("external DS4 may not declare managed launch settings")
        return {"management": "external"}
    if management != "managed":
        raise ValueError("DS4 management must be managed or external")
    result: dict[str, Any] = {"management": "managed"}
    for key in ("server_path", "model_path"):
        raw = value.get(key)
        if not isinstance(raw, str) or not raw or raw != raw.strip() or any(ord(c) < 32 for c in raw):
            raise ValueError(f"ds4.{key} must be a nonempty path")
        path = Path(raw).expanduser()
        result[key] = str(path if path.is_absolute() else directory.absolute() / path)
    context = value.get("context_length", 32768)
    if type(context) is not int or not 1 <= context <= 1_000_000:
        raise ValueError("DS4 context_length must be an integer from 1 to 1000000")
    timeout = value.get("startup_timeout_sec", 1200)
    if type(timeout) not in {float, int} or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise ValueError("DS4 startup_timeout_sec must be finite and in (0, 86400]")
    args = value.get("server_args", [])
    reserved = {"-m", "--model", "-c", "--ctx", "--host", "--port", "--chdir", "--help", "-h", "--version", "--log-level"}
    if (
        not isinstance(args, list) or len(args) > 128
        or any(not isinstance(arg, str) or not arg or len(arg) > 8192 or any(ord(c) < 32 for c in arg) for arg in args)
        or any(arg.split("=", 1)[0] in reserved for arg in args)
    ):
        raise ValueError("DS4 server_args must be an argv list without model/context/listen overrides")
    result.update(context_length=context, startup_timeout_sec=timeout, server_args=list(args))
    return result


def ds4_launch_config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_managed_ds4_base_url(base_url: str) -> None:
    parsed = urlsplit(_ds4_base_url(base_url))
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"} or parsed.path != "/v1":
        raise ValueError("Managed DS4 requires a local HTTP /v1 endpoint; use management=external for remote servers")


def validate_ds4_launch_files(config: dict[str, Any]) -> None:
    """Check local paths without opening GGUF contents or starting anything."""
    executable = Path(config["server_path"])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError("DS4 server_path is not an executable file")
    if not Path(config["model_path"]).is_file():
        raise RuntimeError("DS4 model_path does not exist; configure the downloaded GGUF in ds4")


def _acquire_port_lock(directory: Path, port: int):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle = (directory / f"ds4-{port}.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.write(b"\0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ImportError):
        handle.close()
        raise RuntimeError(f"DS4 port {port} is already reserved by another benchmark") from None
    return handle


class ManagedDS4Server:
    """Own exactly one child process for a run, including startup cancellation."""

    def __init__(self, model: str, *, base_url: str, api_key: str | None,
                 config: dict[str, Any], log_path: Path, lock_directory: Path,
                 docker_access: bool = False) -> None:
        self.model = model
        self.base_url = _ds4_base_url(base_url)
        validate_managed_ds4_base_url(self.base_url)
        parsed = urlsplit(self.base_url)
        self.port = parsed.port or 80
        self.bind_host = "0.0.0.0" if docker_access else "127.0.0.1"
        self.api_key = api_key
        self.config = normalize_ds4_launch_config(config, directory=Path.cwd())
        if self.config["management"] != "managed":
            raise ValueError("ManagedDS4Server requires management=managed")
        self.log_path = log_path
        self.lock_directory = lock_directory
        self.process: subprocess.Popen | None = None
        self._log = None
        self._port_lock = None
        self._log_start = 0
        self._mutex = threading.RLock()
        self._stopped = threading.Event()
        self._ready = threading.Event()

    @property
    def ready(self) -> bool:
        return bool(
            self._ready.is_set() and not self._stopped.is_set()
            and self.process is not None and self.process.poll() is None
        )

    def start(self) -> None:
        try:
            with self._mutex:
                if self._stopped.is_set():
                    raise RuntimeError("DS4 startup was cancelled")
                if self.process is not None:
                    raise RuntimeError("DS4 server was already started")
                executable = Path(self.config["server_path"])
                model_path = Path(self.config["model_path"])
                validate_ds4_launch_files(self.config)
                self._port_lock = _acquire_port_lock(self.lock_directory, self.port)
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                        # Match DS4's listener so TIME_WAIT connections from
                        # the previous run do not block the next model load.
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        # On macOS a reusable wildcard bind may coexist with
                        # a loopback listener. Probe the address we will use.
                        probe.bind((self.bind_host, self.port))
                        probe.listen(1)
                except OSError:
                    raise RuntimeError(f"DS4 port {self.port} is already in use; the existing server was left untouched") from None
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log = self.log_path.open("a+b")
                os.chmod(self.log_path, 0o600)
                self._log_start = self._log.seek(0, os.SEEK_END)
                command = [
                    str(executable), "-m", str(model_path),
                    "--ctx", str(self.config["context_length"]),
                    "--host", self.bind_host, "--port", str(self.port),
                    *self.config["server_args"],
                ]
                self.process = subprocess.Popen(
                    command, cwd=executable.parent, stdin=subprocess.DEVNULL,
                    stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            self._wait_ready()
            with self._mutex:
                if self._stopped.is_set():
                    raise RuntimeError("DS4 startup was cancelled")
                self._ready.set()
        except BaseException:
            self.stop()
            raise

    def _has_listen_banner(self) -> bool:
        # Require this child's post-bind log, not just an API answer from a
        # process that happened to take the port while the GGUF was loading.
        with self.log_path.open("rb") as stream:
            end = stream.seek(0, os.SEEK_END)
            stream.seek(max(self._log_start, end - 8192))
            tail = stream.read(8192)
        marker = f"ds4-server: listening on http://{self.bind_host}:{self.port}".encode()
        return marker + b"\n" in tail or marker + b"\r\n" in tail

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.config["startup_timeout_sec"]
        while not self._stopped.is_set():
            assert self.process is not None
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(f"DS4 exited during model loading (exit code {code}); see ds4-server.log")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("DS4 model loading timed out; see ds4-server.log")
            if self._has_listen_banner():
                try:
                    context = ds4_context_length(
                        self.model, base_url=self.base_url, api_key=self.api_key,
                        timeout_sec=min(10.0, remaining),
                    )
                except RuntimeError:
                    context = None
                if context is not None:
                    if context != self.config["context_length"]:
                        raise RuntimeError("DS4 loaded context_length differs from the launch configuration")
                    if self.process.poll() is None and not self._stopped.is_set():
                        return
            self._stopped.wait(min(0.2, remaining))
        raise RuntimeError("DS4 startup was cancelled")

    def stop(self) -> None:
        self._stopped.set()
        with self._mutex:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=5.0)
                except ProcessLookupError:
                    self.process.wait(timeout=5.0)
            # Release the reservation only after our child has exited. No PID
            # discovery, pkill, API unload, or operation on external servers.
            if self._log is not None:
                self._log.close()
                self._log = None
            if self._port_lock is not None:
                self._port_lock.close()
                self._port_lock = None


class DS4ModelNotFoundError(RuntimeError):
    pass


def _ds4_base_url(base_url: str | None = None) -> str:
    base = validate_http_base_url(base_url or DEFAULT_DS4_API_BASE)
    if not urlsplit(base).path:
        return join_http_url(base, "/v1")
    if not base.endswith("/v1"):
        raise HTTPBoundaryError("DS4 API base path must end in /v1")
    return base


def read_ds4_model_info(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    """Read an advertised alias; never start a server, load weights, or infer."""

    base = _ds4_base_url(base_url)
    token = validate_bearer_token(api_key)
    headers = {"Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    def read_metadata(path: str) -> Any:
        request = urllib.request.Request(join_http_url(base, path), headers=headers, method="GET")
        try:
            with open_url_no_redirect(request, timeout_sec=timeout_sec) as response:
                raw = read_bounded_response(response, max_bytes=MAX_HTTP_RESPONSE_BYTES)
                validate_json_content_type(response.headers)
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise RuntimeError(f"DS4 model API returned HTTP {status}") from None
        except OSError:
            raise RuntimeError("DS4 model API is unavailable") from None
        return strict_json_loads(raw, max_bytes=MAX_HTTP_RESPONSE_BYTES)

    payload = read_metadata("/models")
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError("DS4 model API returned an invalid model list")
    entries = payload["data"]
    if not all(
        isinstance(item, dict) and isinstance(item.get("id"), str)
        for item in entries
    ):
        raise RuntimeError("DS4 model API returned an invalid model entry")
    # LiteLLM may carry its routing prefix; no family or quantization guessing.
    alias = model.removeprefix("openai/")
    matches = [item for item in entries if item["id"] == alias]
    if not matches and alias == "glm-5.3-flash" and any(item["id"] == "glm-5.2" for item in entries):
        # DS4 may advertise 5.2 aliases for GLM 5.3. Require the individual
        # endpoint's exact ID; never relabel the first model in the list.
        direct = read_metadata("/models/glm-5.3-flash")
        if isinstance(direct, dict) and direct.get("id") == alias:
            matches = [direct]
    if not matches:
        raise DS4ModelNotFoundError("Requested model alias is not advertised by DS4")
    if len(matches) != 1:
        raise RuntimeError("DS4 model API returned an ambiguous model alias")
    return dict(matches[0])


def _positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def ds4_context_length(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_sec: float = 10.0,
) -> int:
    info = read_ds4_model_info(model, base_url=base_url, api_key=api_key, timeout_sec=timeout_sec)
    context = _positive_int(info.get("context_length"))
    if context is None:
        raise RuntimeError("DS4 model API did not report a valid context_length")
    return context


def ds4_environment_info(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    fresh: bool = False,
    probe: bool = True,
) -> dict[str, Any]:
    # A server can restart with another GGUF under the same alias. Always read
    # the current advertisement, including when callers allow cached probes.
    del fresh
    result: dict[str, Any] = {
        "provider": "ds4",
        "engine": "ds4",
        "api_base": _ds4_base_url(base_url),
        "capture_status": "partial",
        "local_artifact_identity_status": "unavailable",
        "process_config_status": "unavailable",
        "model_identity_scope": "server_compatibility_alias",
    }
    if not probe:
        result["model_api_status"] = "not_loaded_by_run"
        return result
    try:
        info = read_ds4_model_info(model, base_url=base_url, api_key=api_key)
    except DS4ModelNotFoundError:
        result["model_api_status"] = "model_unmatched"
        return result
    except Exception:
        result["model_api_status"] = "unavailable"
        return result
    result.update(model_api_status="matched", model_id=info["id"], state="loaded")
    context = _positive_int(info.get("context_length"))
    if context is not None:
        result["loaded_context_length"] = context
        result["load_config"] = {"context_length": context}
    # /v1/models lists compatibility names, not distinct weight artifacts. Do
    # not promote API names/hashes or server sampler defaults into verified
    # GGUF identity, engine version, or effective sampling telemetry.
    return result
