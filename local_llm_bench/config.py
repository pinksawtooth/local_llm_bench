from __future__ import annotations

import os
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import yaml

from .docker_task.ghidra_tool_mode import (
    DEFAULT_GHIDRA_TOOL_MODE,
    normalize_ghidra_tool_mode,
)
from .ds4 import (
    DEFAULT_DS4_API_BASE, DS4_API_KEY_ENV, _ds4_base_url, docker_ds4_base_url,
    normalize_ds4_launch_config, validate_ds4_request, validate_managed_ds4_base_url,
    DS4_SAMPLING_KEYS, apply_ds4_sampling, normalize_ds4_sampling_config,
)
from .http_boundary import validate_bearer_token
from .omlx import DEFAULT_OMLX_API_BASE, OMLX_API_KEY_ENV, omlx_base_url, docker_omlx_base_url, validate_omlx_request, resolve_omlx_api_key
from .mlx_serve import (
    DEFAULT_MLX_SERVE_API_BASE, MLX_SERVE_API_KEY_ENV, MLX_SERVE_REQUEST_KEYS, mlx_serve_base_url,
    docker_mlx_serve_base_url, normalize_mlx_serve_config, validate_mlx_serve_request,
)

DEFAULT_PROMPT = "pythonでライブラリを使わずにRC4を実装して"
DEFAULT_CONFIG_PATH = Path("configs/bench.yaml")
DEFAULT_MODE = "prompt"
DOCKER_TASK_MODE = "docker_task"
DEFAULT_QUESTION_TIMEOUT_SEC = 3600.0
PERFORMANCE_MODE = "performance"
SUPPORTED_MODES = {DEFAULT_MODE, DOCKER_TASK_MODE, PERFORMANCE_MODE}
LMSTUDIO_PROVIDER = "lmstudio"
UNSLOTH_STUDIO_PROVIDER = "unsloth_studio"
DS4_PROVIDER = "ds4"
OMLX_PROVIDER = "omlx"
MLX_SERVE_PROVIDER = "mlx_serve"
DEFAULT_PROVIDER = LMSTUDIO_PROVIDER
SUPPORTED_PROVIDERS = {LMSTUDIO_PROVIDER, UNSLOTH_STUDIO_PROVIDER, DS4_PROVIDER, OMLX_PROVIDER, MLX_SERVE_PROVIDER}
# Providers whose saved/server-side generation settings are used by omitting request fields.
SAVED_DEFAULT_FLAGS = {OMLX_PROVIDER: "use_omlx_defaults", MLX_SERVE_PROVIDER: "use_mlx_serve_defaults"}
DEFAULT_LMSTUDIO_API_BASE = "http://localhost:1234/v1"
DEFAULT_DOCKER_API_BASE = "http://host.docker.internal:1234/v1"
DEFAULT_UNSLOTH_STUDIO_API_BASE = "http://127.0.0.1:8888/v1"
DEFAULT_DOCKER_UNSLOTH_STUDIO_API_BASE = "http://host.docker.internal:8888/v1"
UNSLOTH_STUDIO_BEARER_TOKEN_ENV = "UNSLOTH_STUDIO_BEARER_TOKEN"
UNSLOTH_STUDIO_USERNAME_ENV = "UNSLOTH_STUDIO_USERNAME"
UNSLOTH_STUDIO_PASSWORD_ENV = "UNSLOTH_STUDIO_PASSWORD"


@dataclass
class RequestSettings:
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    use_lmstudio_defaults: bool = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    seed: Optional[int] = None
    use_omlx_defaults: bool = False
    use_mlx_serve_defaults: bool = False

    @property
    def uses_saved_defaults(self) -> bool:
        return self.use_lmstudio_defaults or self.use_omlx_defaults or self.use_mlx_serve_defaults

    def optional_parameters(self) -> dict[str, Any]:
        if self.uses_saved_defaults:
            return {}
        return {key: value for key, value in {"top_p": self.top_p, "reasoning_effort": self.reasoning_effort,
                                            "top_k": self.top_k, "min_p": self.min_p, "seed": self.seed}.items() if value is not None}

    def api_parameters(self) -> dict[str, Any]:
        if self.uses_saved_defaults:
            return {}
        return {key: value for key, value in {"temperature": self.temperature, "max_tokens": self.max_tokens, **self.optional_parameters()}.items() if value is not None}

    def recorded_parameters(self) -> dict[str, Any]:
        if self.use_mlx_serve_defaults:
            return {"settings_source": "mlx_serve_saved"}
        if self.use_omlx_defaults:
            return {"settings_source": "omlx_saved"}
        if self.use_lmstudio_defaults:
            return {"settings_source": "lmstudio_saved"}
        return self.api_parameters()


@dataclass
class RunSettings:
    cold_runs: int = 1
    warm_runs: int = 3
    timeout_sec: float = 120.0
    cooldown_sec: float = 0.0


@dataclass
class InspectSettings:
    max_turns: Optional[int] = None
    max_tool_calls: Optional[int] = None
    tool_timeout_sec: float = 120.0


@dataclass
class PerformanceSettings:
    input_tokens: list[int] = field(default_factory=lambda: [1024, 4096, 8192])
    concurrency: list[int] = field(default_factory=lambda: [1, 2, 4])
    memory_pid: int | None = None


def _performance_settings(block: Any) -> PerformanceSettings:
    if not isinstance(block, dict) or set(block) - {"input_tokens", "concurrency", "memory_pid"}:
        raise ValueError("performance は input_tokens / concurrency / memory_pid のマッピングです。")
    def values(key, default, limit):
        items = block.get(key, default)
        if (not isinstance(items, list) or not items or len(items) > 16
                or any(type(item) is not int or not 1 <= item <= limit for item in items)
                or len(set(items)) != len(items)):
            raise ValueError(f"performance.{key} は重複のない正の整数配列です（最大値 {limit}）。")
        return sorted(items)
    pid = block.get("memory_pid")
    if pid is not None and (type(pid) is not int or pid < 1):
        raise ValueError("performance.memory_pid は推論サーバーの正のPIDです。")
    return PerformanceSettings(values("input_tokens", [1024, 4096, 8192], 131072),
                               values("concurrency", [1, 2, 4], 32), pid)


@dataclass
class LMStudioLoadSettings:
    parallelism: Optional[int] = None
    parallelism_sweep: list[int] = field(default_factory=list)
    context_length: Optional[int] = None
    eval_batch_size: Optional[int] = None
    flash_attention: Optional[bool] = None
    num_experts: Optional[int] = None
    offload_kv_cache_to_gpu: Optional[bool] = None

    def has_load_overrides(self) -> bool:
        return any(
            value is not None
            for value in (
                self.parallelism,
                self.context_length,
                self.eval_batch_size,
                self.flash_attention,
                self.num_experts,
                self.offload_kv_cache_to_gpu,
            )
        )


@dataclass
class OutputSettings:
    history_json: Path
    latest_json: Path
    report_html: Path
    run_logs_dir: Path


@dataclass
class AuthSettings:
    bearer_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    def to_safe_dict(self) -> Dict[str, Any]:
        return {
            "bearer_token_present": bool(self.bearer_token),
            "username": self.username,
            "password_present": bool(self.password),
        }


@dataclass
class BenchmarkConfig:
    api_base: str
    models: list[str]
    prompt_text: str
    request: RequestSettings
    runs: RunSettings
    output: OutputSettings
    config_path: Optional[Path] = None
    mode: str = DEFAULT_MODE
    provider: str = DEFAULT_PROVIDER
    auth: AuthSettings = field(default_factory=AuthSettings)
    lmstudio_load: LMStudioLoadSettings = field(default_factory=LMStudioLoadSettings)
    ds4: dict[str, Any] | None = None
    ds4_sampling: dict[str, Any] | None = None
    mlx_serve: dict[str, Any] | None = None
    benchmark_spec_path: Optional[Path] = None
    benchmark_answer_key_path: Optional[Path] = None
    benchmark_question_timeout_sec: Optional[float] = DEFAULT_QUESTION_TIMEOUT_SEC
    benchmark_ghidra_tool_mode: str = DEFAULT_GHIDRA_TOOL_MODE
    docker_image: Optional[str] = None
    docker_platform: Optional[str] = None
    docker_api_base: str = DEFAULT_DOCKER_API_BASE
    inspect: InspectSettings = field(default_factory=InspectSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)

    @property
    def docker_lmstudio_base_url(self) -> str:
        return self.docker_api_base

    def request_parameters(self) -> dict[str, Any]:
        parameters = self.request.api_parameters()
        if self.provider == DS4_PROVIDER:
            return apply_ds4_sampling(parameters, self.ds4_sampling)
        return parameters

    def recorded_request_parameters(self) -> dict[str, Any]:
        if self.provider == DS4_PROVIDER:
            policy = normalize_ds4_sampling_config(self.ds4_sampling)
            return {**self.request_parameters(), "sampling_source": "ds4_" + policy["source"]}
        return self.request.recorded_parameters()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["output"] = {
            "history_json": str(self.output.history_json),
            "latest_json": str(self.output.latest_json),
            "report_html": str(self.output.report_html),
            "run_logs_dir": str(self.output.run_logs_dir),
        }
        payload["config_path"] = str(self.config_path) if self.config_path else None
        payload["benchmark_spec_path"] = (
            str(self.benchmark_spec_path) if self.benchmark_spec_path else None
        )
        payload["benchmark_answer_key_path"] = (
            str(self.benchmark_answer_key_path) if self.benchmark_answer_key_path else None
        )
        payload["auth"] = self.auth.to_safe_dict()
        return payload


def _ensure_positive_int(value: Any, field_name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} は整数である必要があります: {value!r}") from exc
    if normalized < 1:
        raise ValueError(f"{field_name} は1以上である必要があります: {normalized}")
    return normalized


def _ensure_non_negative_int(value: Any, field_name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} は整数である必要があります: {value!r}") from exc
    if normalized < 0:
        raise ValueError(f"{field_name} は0以上である必要があります: {normalized}")
    return normalized


def _ensure_non_negative_float(value: Any, field_name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} は数値である必要があります: {value!r}") from exc
    if normalized < 0:
        raise ValueError(f"{field_name} は0以上である必要があります: {normalized}")
    return normalized


def _ensure_positive_int_list(value: Any, field_name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items: list[Any] = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raise ValueError(f"{field_name} は配列かカンマ区切り文字列である必要があります: {value!r}")

    normalized_values: list[int] = []
    for index, item in enumerate(raw_items):
        if item is None or (isinstance(item, str) and not item.strip()):
            continue
        normalized = _ensure_positive_int(item, f"{field_name}[{index}]")
        if normalized not in normalized_values:
            normalized_values.append(normalized)
    return normalized_values


def _ensure_optional_positive_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    return _ensure_positive_int(value, field_name)


def _ensure_optional_bool(value: Any, field_name: str) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} は真偽値である必要があります: {value!r}")


def _resolve_output_path(base_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_optional_path(base_dir: Path, raw_value: Any) -> Optional[Path]:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _normalize_mode(raw_value: Any) -> str:
    normalized = str(raw_value or DEFAULT_MODE).strip().lower()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(
            f"mode は {sorted(SUPPORTED_MODES)} のいずれかである必要があります: {raw_value!r}"
        )
    return normalized


def _normalize_provider(raw_value: Any) -> str:
    normalized = str(raw_value or DEFAULT_PROVIDER).strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"provider は {sorted(SUPPORTED_PROVIDERS)} のいずれかである必要があります: {raw_value!r}"
        )
    return normalized


def _normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_auth_settings(raw_value: Any, provider: str = DEFAULT_PROVIDER) -> AuthSettings:
    if raw_value is None:
        block: dict[str, Any] = {}
    elif isinstance(raw_value, dict):
        block = raw_value
    else:
        raise ValueError("auth はマッピングである必要があります。")

    if provider in {DS4_PROVIDER, OMLX_PROVIDER, MLX_SERVE_PROVIDER}:
        key_env = {OMLX_PROVIDER: OMLX_API_KEY_ENV, MLX_SERVE_PROVIDER: MLX_SERVE_API_KEY_ENV}.get(provider, DS4_API_KEY_ENV)
        if block.get("username") or block.get("password"):
            raise ValueError(f"provider={provider} の認証は auth.bearer_token または {key_env} で指定してください。")
        return AuthSettings(bearer_token=validate_bearer_token(block.get("bearer_token") or os.getenv(key_env)))

    bearer_token = _normalize_optional_text(block.get("bearer_token")) or _normalize_optional_text(
        os.getenv(UNSLOTH_STUDIO_BEARER_TOKEN_ENV)
    )
    username = _normalize_optional_text(block.get("username")) or _normalize_optional_text(
        os.getenv(UNSLOTH_STUDIO_USERNAME_ENV)
    )
    password = _normalize_optional_text(block.get("password")) or _normalize_optional_text(
        os.getenv(UNSLOTH_STUDIO_PASSWORD_ENV)
    )
    return AuthSettings(
        bearer_token=bearer_token,
        username=username,
        password=password,
    )


def load_config(
    config_path: Optional[Path],
    *,
    cli_models: Optional[list[str]] = None,
    cli_prompt_text: Optional[str] = None,
    cli_api_base: Optional[str] = None,
    cli_cold_runs: Optional[int] = None,
    cli_warm_runs: Optional[int] = None,
    cli_timeout_sec: Optional[float] = None,
    cli_parallelism: Optional[int] = None,
    cli_parallelism_sweep: Any = None,
    cli_max_tokens: Optional[int] = None,
    cli_temperature: Optional[float] = None,
    cli_out_dir: Optional[Path] = None,
) -> BenchmarkConfig:
    resolved_config_path = (config_path or DEFAULT_CONFIG_PATH).resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {resolved_config_path}")

    loaded = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("設定ファイルのトップレベルはマッピングである必要があります。")

    base_dir = resolved_config_path.parent
    request_block = loaded.get("request") or {}
    runs_block = loaded.get("runs") or {}
    prompt_block = loaded.get("prompt") or {}
    output_block = loaded.get("output") or {}
    benchmark_block = loaded.get("benchmark") or {}
    docker_block = loaded.get("docker") or {}
    lmstudio_block = loaded.get("lmstudio") or {}
    if not isinstance(lmstudio_block, dict):
        raise ValueError("lmstudio はマッピングである必要があります。")
    mode = _normalize_mode(loaded.get("mode"))
    performance = _performance_settings(loaded.get("performance", {}))
    if "performance" in loaded and mode != PERFORMANCE_MODE:
        raise ValueError("performance 設定には mode: performance が必要です。")
    if "harness" in loaded:
        raise ValueError("評価基盤はInspect AIに固定されています。harness設定を削除してください。")
    inspect_block = loaded.get("inspect", {})
    if not isinstance(inspect_block, dict) or set(inspect_block) - {"max_turns", "max_tool_calls", "tool_timeout_sec"}:
        raise ValueError("inspect は max_turns / max_tool_calls / tool_timeout_sec のマッピングです。")
    def inspect_positive(name, default, *, integer=True, optional=False):
        value = inspect_block.get(name, default)
        if optional and value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or (integer and int(value) != value):
            raise ValueError(f"inspect.{name} は有限の正の数値で指定してください。")
        return int(value) if integer else float(value)
    inspect_settings = InspectSettings(
        max_turns=inspect_positive("max_turns", None, optional=True),
        max_tool_calls=inspect_positive("max_tool_calls", None, optional=True),
        tool_timeout_sec=inspect_positive("tool_timeout_sec", 120.0, integer=False),
    )
    provider = _normalize_provider(loaded.get("provider"))
    auth = _load_auth_settings(loaded.get("auth"), provider)
    ds4_launch = None
    if provider == DS4_PROVIDER:
        if loaded.get("ds4") is None:
            raise ValueError("provider=ds4 では ds4.server_path / model_path、または ds4.management: external を指定してください。")
        ds4_launch = normalize_ds4_launch_config(loaded["ds4"], directory=base_dir)
    elif "ds4" in loaded:
        raise ValueError("ds4 のロード設定には provider: ds4 が必要です。")
    ds4_sampling = None
    if "ds4_sampling" in loaded and (provider != DS4_PROVIDER or loaded["ds4_sampling"] is None):
        raise ValueError("ds4_sampling は provider=ds4 専用のマッピングです。")
    if provider == DS4_PROVIDER:
        ds4_sampling = normalize_ds4_sampling_config(loaded.get("ds4_sampling"))
    mlx_serve_settings = None
    if provider == MLX_SERVE_PROVIDER:
        mlx_serve_settings = normalize_mlx_serve_config(loaded.get("mlx_serve"), directory=base_dir)
    elif "mlx_serve" in loaded:
        raise ValueError("mlx_serve の設定には provider: mlx_serve が必要です。")

    raw_models = cli_models if cli_models else loaded.get("models", [])
    if not isinstance(raw_models, list):
        raise ValueError("models は配列である必要があります。")
    models = [str(item).strip() for item in raw_models if str(item).strip()]
    if not models:
        raise ValueError("比較対象モデルが1件もありません。設定ファイルの models か --model で指定してください。")

    raw_api_base = cli_api_base if cli_api_base is not None else loaded.get("api_base")
    if provider == UNSLOTH_STUDIO_PROVIDER:
        if _normalize_optional_text(raw_api_base):
            raise ValueError("provider=unsloth_studio では api_base を指定できません。")
        api_base = DEFAULT_UNSLOTH_STUDIO_API_BASE
    elif provider == DS4_PROVIDER:
        api_base = _ds4_base_url(raw_api_base or DEFAULT_DS4_API_BASE)
        if ds4_launch["management"] == "managed":
            validate_managed_ds4_base_url(api_base)
    elif provider == OMLX_PROVIDER:
        api_base = omlx_base_url(raw_api_base or DEFAULT_OMLX_API_BASE)
        auth.bearer_token = resolve_omlx_api_key(api_base, auth.bearer_token)
    elif provider == MLX_SERVE_PROVIDER:
        api_base = mlx_serve_base_url(raw_api_base or DEFAULT_MLX_SERVE_API_BASE)
    else:
        api_base = str(raw_api_base or DEFAULT_LMSTUDIO_API_BASE).rstrip("/")
    if mode == DOCKER_TASK_MODE:
        prompt_text = str(cli_prompt_text if cli_prompt_text is not None else prompt_block.get("text") or "")
    else:
        prompt_text = str(cli_prompt_text if cli_prompt_text is not None else prompt_block.get("text") or DEFAULT_PROMPT)

    if not isinstance(request_block, dict):
        raise ValueError("request はマッピングである必要があります。")
    saved_default_flags = {}
    for saved_provider, flag in SAVED_DEFAULT_FLAGS.items():
        value = request_block.get(flag, provider == saved_provider)
        if not isinstance(value, bool):
            raise ValueError(f"request.{flag} は真偽値で指定してください。")
        if flag in request_block and provider != saved_provider:
            raise ValueError(f"request.{flag} は provider={saved_provider} 専用です。")
        saved_default_flags[saved_provider] = value
    use_omlx_defaults = saved_default_flags[OMLX_PROVIDER]
    use_mlx_serve_defaults = saved_default_flags[MLX_SERVE_PROVIDER]
    if provider in SAVED_DEFAULT_FLAGS:
        from .omlx import OMLX_REQUEST_KEYS
        request_keys = OMLX_REQUEST_KEYS if provider == OMLX_PROVIDER else MLX_SERVE_REQUEST_KEYS
        validate_request = validate_omlx_request if provider == OMLX_PROVIDER else validate_mlx_serve_request
        label = "oMLX" if provider == OMLX_PROVIDER else "mlx-serve"
        if set(request_block) - (request_keys | {SAVED_DEFAULT_FLAGS[provider]}):
            raise ValueError(f"{label} request に未対応の設定があります。")
        if saved_default_flags[provider] and (any(request_block.get(key) is not None for key in request_keys)
                                              or cli_temperature is not None or cli_max_tokens is not None):
            raise ValueError(f"{label} の保存設定を使用するときは推論設定や --temperature / --max-tokens を指定できません。")
        validate_request({**request_block,
                          **({"temperature": cli_temperature} if cli_temperature is not None else {}),
                          **({"max_tokens": cli_max_tokens} if cli_max_tokens is not None else {})})
    use_lmstudio_defaults = request_block.get("use_lmstudio_defaults", provider == LMSTUDIO_PROVIDER)
    if not isinstance(use_lmstudio_defaults, bool):
        raise ValueError("request.use_lmstudio_defaults は真偽値で指定してください。")
    if use_lmstudio_defaults:
        if provider != LMSTUDIO_PROVIDER:
            raise ValueError("request.use_lmstudio_defaults は provider=lmstudio 専用です。")
        overrides = [key for key in request_block if key != "use_lmstudio_defaults" and request_block[key] is not None]
        if overrides or cli_temperature is not None or cli_max_tokens is not None:
            raise ValueError("LM Studio の保存設定を使用するときは request の推論設定や --temperature / --max-tokens を指定できません。")

    uses_saved_defaults = use_lmstudio_defaults or use_omlx_defaults or use_mlx_serve_defaults
    request = RequestSettings(
        temperature=float(
            cli_temperature
            if cli_temperature is not None
            else request_block.get("temperature", 0.0)
        ) if not uses_saved_defaults else None,
        max_tokens=_ensure_positive_int(
            cli_max_tokens if cli_max_tokens is not None else request_block.get("max_tokens", 512),
            "request.max_tokens",
        ) if not uses_saved_defaults else None,
        top_p=float(request_block["top_p"]) if request_block.get("top_p") is not None else None,
        reasoning_effort=_normalize_optional_text(request_block.get("reasoning_effort")),
        use_lmstudio_defaults=use_lmstudio_defaults,
        use_omlx_defaults=use_omlx_defaults,
        use_mlx_serve_defaults=use_mlx_serve_defaults,
        top_k=request_block.get("top_k"),
        min_p=request_block.get("min_p"),
        seed=request_block.get("seed"),
    )
    if request.top_p is not None and (not math.isfinite(request.top_p) or not 0 < request.top_p <= 1):
        raise ValueError("request.top_p は 0 より大きく 1 以下である必要があります。")
    if provider == DS4_PROVIDER:
        raw_sampling = {key: request_block[key] for key in DS4_SAMPLING_KEYS if request_block.get(key) is not None}
        if cli_temperature is not None:
            raw_sampling["temperature"] = cli_temperature
        if raw_sampling:
            normalize_ds4_sampling_config({"source": "model_config", **raw_sampling})
        validate_ds4_request(apply_ds4_sampling(request.api_parameters(), ds4_sampling), ds4_launch.get("context_length"))
    elif provider == OMLX_PROVIDER:
        validate_omlx_request(request.api_parameters())
    elif provider == MLX_SERVE_PROVIDER:
        validate_mlx_serve_request(request.api_parameters())
    elif any(request_block.get(key) is not None for key in ("top_k", "min_p", "seed")):
        raise ValueError("request.top_k / min_p / seed は provider=ds4 / omlx / mlx_serve 専用です。")

    runs = RunSettings(
        cold_runs=_ensure_non_negative_int(
            cli_cold_runs if cli_cold_runs is not None else runs_block.get("cold_runs", 0 if ds4_launch and ds4_launch["management"] == "external" else 1),
            "runs.cold_runs",
        ),
        warm_runs=_ensure_non_negative_int(
            cli_warm_runs if cli_warm_runs is not None else runs_block.get("warm_runs", 3),
            "runs.warm_runs",
        ),
        timeout_sec=_ensure_non_negative_float(
            cli_timeout_sec if cli_timeout_sec is not None else runs_block.get("timeout_sec", 120.0),
            "runs.timeout_sec",
        ),
        cooldown_sec=_ensure_non_negative_float(
            runs_block.get("cooldown_sec", 0.0),
            "runs.cooldown_sec",
        ),
    )
    if runs.cold_runs + runs.warm_runs < 1:
        raise ValueError("runs.cold_runs と runs.warm_runs の合計は1以上である必要があります。")
    if ds4_launch and ds4_launch["management"] == "external" and runs.cold_runs:
        raise ValueError("ds4 は外部管理のサーバーです。再ロードを伴う cold 測定はできません。runs.cold_runs: 0 を指定してください。")

    lmstudio_load = LMStudioLoadSettings(
        parallelism=_ensure_optional_positive_int(
            cli_parallelism
            if cli_parallelism is not None
            else lmstudio_block.get("parallelism", lmstudio_block.get("parallel")),
            "lmstudio.parallelism",
        ),
        parallelism_sweep=_ensure_positive_int_list(
            cli_parallelism_sweep
            if cli_parallelism_sweep is not None
            else [] if cli_parallelism is not None
            else lmstudio_block.get(
                "parallelism_sweep",
                lmstudio_block.get("parallel_sweep", lmstudio_block.get("parallelism_values")),
            ),
            "lmstudio.parallelism_sweep",
        ),
        context_length=_ensure_optional_positive_int(
            lmstudio_block.get("context_length"),
            "lmstudio.context_length",
        ),
        eval_batch_size=_ensure_optional_positive_int(
            lmstudio_block.get("eval_batch_size"),
            "lmstudio.eval_batch_size",
        ),
        flash_attention=_ensure_optional_bool(
            lmstudio_block.get("flash_attention"),
            "lmstudio.flash_attention",
        ),
        num_experts=_ensure_optional_positive_int(
            lmstudio_block.get("num_experts"),
            "lmstudio.num_experts",
        ),
        offload_kv_cache_to_gpu=_ensure_optional_bool(
            lmstudio_block.get("offload_kv_cache_to_gpu"),
            "lmstudio.offload_kv_cache_to_gpu",
        ),
    )
    if provider == DS4_PROVIDER and (lmstudio_load.has_load_overrides() or lmstudio_load.parallelism_sweep):
        raise ValueError("ds4 のロード設定は ds4 ブロックに指定してください。lmstudio 設定や --parallelism は使用できません。")
    if provider == OMLX_PROVIDER and (lmstudio_block or lmstudio_load.has_load_overrides() or lmstudio_load.parallelism_sweep):
        raise ValueError("oMLX のロード設定はoMLX側で設定してください。lmstudio 設定や --parallelism は使用できません。")
    if provider == MLX_SERVE_PROVIDER and (lmstudio_block or lmstudio_load.has_load_overrides() or lmstudio_load.parallelism_sweep):
        raise ValueError("mlx-serve のロード設定はmlx-serveの起動オプションで設定してください。lmstudio 設定や --parallelism は使用できません。")

    if cli_out_dir is not None:
        out_dir = cli_out_dir.resolve()
        output = OutputSettings(
            history_json=out_dir / "history.json",
            latest_json=out_dir / "latest_run.json",
            report_html=out_dir / "index.html",
            run_logs_dir=out_dir / "logs",
        )
    else:
        output = OutputSettings(
            history_json=_resolve_output_path(base_dir, str(output_block.get("history_json", "runs/history.json"))),
            latest_json=_resolve_output_path(base_dir, str(output_block.get("latest_json", "runs/latest_run.json"))),
            report_html=_resolve_output_path(base_dir, str(output_block.get("report_html", "docs/index.html"))),
            run_logs_dir=_resolve_output_path(base_dir, str(output_block.get("run_logs_dir", "runs/logs"))),
        )

    benchmark_spec_path = _resolve_optional_path(base_dir, benchmark_block.get("spec"))
    benchmark_answer_key_path = _resolve_optional_path(base_dir, benchmark_block.get("answer_key"))
    benchmark_question_timeout_sec_raw = benchmark_block.get("question_timeout_sec")
    benchmark_question_timeout_sec = (
        _ensure_non_negative_float(
            benchmark_question_timeout_sec_raw,
            "benchmark.question_timeout_sec",
        )
        if benchmark_question_timeout_sec_raw is not None
        else (DEFAULT_QUESTION_TIMEOUT_SEC if mode == DOCKER_TASK_MODE else None)
    )
    benchmark_ghidra_tool_mode = normalize_ghidra_tool_mode(
        benchmark_block.get("ghidra_tool_mode"),
        default=DEFAULT_GHIDRA_TOOL_MODE,
    )
    docker_image = str(docker_block.get("image") or "").strip() or None
    docker_platform = str(docker_block.get("platform") or "").strip() or None
    raw_docker_api_base = docker_block.get("api_base")
    if raw_docker_api_base is None:
        raw_docker_api_base = docker_block.get("lmstudio_base_url")
    if provider == UNSLOTH_STUDIO_PROVIDER:
        if _normalize_optional_text(raw_docker_api_base):
            raise ValueError(
                "provider=unsloth_studio では docker.api_base / docker.lmstudio_base_url を指定できません。"
            )
        docker_api_base = DEFAULT_DOCKER_UNSLOTH_STUDIO_API_BASE
    elif provider in {DS4_PROVIDER, OMLX_PROVIDER, MLX_SERVE_PROVIDER}:
        normalize_base, docker_base = {
            OMLX_PROVIDER: (omlx_base_url, docker_omlx_base_url),
            MLX_SERVE_PROVIDER: (mlx_serve_base_url, docker_mlx_serve_base_url),
        }.get(provider, (_ds4_base_url, docker_ds4_base_url))
        docker_api_base = normalize_base(raw_docker_api_base) if raw_docker_api_base else docker_base(api_base)
        from .persistence import server_identity
        if (server_identity(docker_api_base) != server_identity(api_base)
                or urlsplit(docker_api_base).path != urlsplit(api_base).path):
            raise ValueError(f"{provider} の docker.api_base は api_base と同じ推論サーバーを指定してください。")
        if mode == DOCKER_TASK_MODE and urlsplit(docker_api_base).hostname in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("docker.api_base の loopback はコンテナ自身を指します。host.docker.internal を使用してください。")
    else:
        docker_api_base = str(raw_docker_api_base or DEFAULT_DOCKER_API_BASE).rstrip("/")

    if mode == DOCKER_TASK_MODE:
        if benchmark_spec_path is None:
            raise ValueError("mode=docker_task では benchmark.spec が必須です。")
        if docker_image is None:
            raise ValueError("mode=docker_task では docker.image が必須です。")

    return BenchmarkConfig(
        api_base=api_base,
        models=models,
        prompt_text=prompt_text,
        request=request,
        runs=runs,
        output=output,
        config_path=resolved_config_path,
        mode=mode,
        provider=provider,
        auth=auth,
        lmstudio_load=lmstudio_load,
        ds4=ds4_launch,
        ds4_sampling=ds4_sampling,
        mlx_serve=mlx_serve_settings,
        benchmark_spec_path=benchmark_spec_path,
        benchmark_answer_key_path=benchmark_answer_key_path,
        benchmark_question_timeout_sec=benchmark_question_timeout_sec,
        benchmark_ghidra_tool_mode=benchmark_ghidra_tool_mode,
        docker_image=docker_image,
        docker_platform=docker_platform,
        docker_api_base=docker_api_base,
        inspect=inspect_settings,
        performance=performance,
    )
