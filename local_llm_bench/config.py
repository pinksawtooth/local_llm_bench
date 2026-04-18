from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .docker_task.ghidra_tool_mode import (
    DEFAULT_GHIDRA_TOOL_MODE,
    normalize_ghidra_tool_mode,
)

DEFAULT_PROMPT = "pythonでライブラリを使わずにRC4を実装して"
DEFAULT_CONFIG_PATH = Path("bench.yaml")
DEFAULT_MODE = "prompt"
DOCKER_TASK_MODE = "docker_task"
SUPPORTED_MODES = {DEFAULT_MODE, DOCKER_TASK_MODE}
LMSTUDIO_PROVIDER = "lmstudio"
UNSLOTH_STUDIO_PROVIDER = "unsloth_studio"
DEFAULT_PROVIDER = LMSTUDIO_PROVIDER
SUPPORTED_PROVIDERS = {LMSTUDIO_PROVIDER, UNSLOTH_STUDIO_PROVIDER}
DEFAULT_LMSTUDIO_API_BASE = "http://localhost:1234/v1"
DEFAULT_DOCKER_API_BASE = "http://host.docker.internal:1234/v1"
DEFAULT_UNSLOTH_STUDIO_API_BASE = "http://127.0.0.1:8888/v1"
DEFAULT_DOCKER_UNSLOTH_STUDIO_API_BASE = "http://host.docker.internal:8888/v1"
UNSLOTH_STUDIO_BEARER_TOKEN_ENV = "UNSLOTH_STUDIO_BEARER_TOKEN"
UNSLOTH_STUDIO_USERNAME_ENV = "UNSLOTH_STUDIO_USERNAME"
UNSLOTH_STUDIO_PASSWORD_ENV = "UNSLOTH_STUDIO_PASSWORD"


@dataclass
class RequestSettings:
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass
class RunSettings:
    cold_runs: int = 1
    warm_runs: int = 3
    timeout_sec: float = 120.0
    cooldown_sec: float = 0.0


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
    benchmark_spec_path: Optional[Path] = None
    benchmark_answer_key_path: Optional[Path] = None
    benchmark_question_timeout_sec: Optional[float] = None
    benchmark_ghidra_tool_mode: str = DEFAULT_GHIDRA_TOOL_MODE
    docker_image: Optional[str] = None
    docker_platform: Optional[str] = None
    docker_api_base: str = DEFAULT_DOCKER_API_BASE

    @property
    def docker_lmstudio_base_url(self) -> str:
        return self.docker_api_base

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


def _load_auth_settings(raw_value: Any) -> AuthSettings:
    if raw_value is None:
        block: dict[str, Any] = {}
    elif isinstance(raw_value, dict):
        block = raw_value
    else:
        raise ValueError("auth はマッピングである必要があります。")

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
    mode = _normalize_mode(loaded.get("mode"))
    provider = _normalize_provider(loaded.get("provider"))
    auth = _load_auth_settings(loaded.get("auth"))

    raw_models = cli_models if cli_models else loaded.get("models", [])
    if not isinstance(raw_models, list):
        raise ValueError("models は配列である必要があります。")
    models = [str(item).strip() for item in raw_models if str(item).strip()]
    if not models:
        raise ValueError("比較対象モデルが1件もありません。bench.yaml か --model で指定してください。")

    raw_api_base = cli_api_base if cli_api_base is not None else loaded.get("api_base")
    if provider == UNSLOTH_STUDIO_PROVIDER:
        if _normalize_optional_text(raw_api_base):
            raise ValueError("provider=unsloth_studio では api_base を指定できません。")
        api_base = DEFAULT_UNSLOTH_STUDIO_API_BASE
    else:
        api_base = str(raw_api_base or DEFAULT_LMSTUDIO_API_BASE).rstrip("/")
    if mode == DOCKER_TASK_MODE:
        prompt_text = str(cli_prompt_text if cli_prompt_text is not None else prompt_block.get("text") or "")
    else:
        prompt_text = str(cli_prompt_text if cli_prompt_text is not None else prompt_block.get("text") or DEFAULT_PROMPT)

    request = RequestSettings(
        temperature=float(
            cli_temperature
            if cli_temperature is not None
            else request_block.get("temperature", 0.0)
        ),
        max_tokens=_ensure_positive_int(
            cli_max_tokens if cli_max_tokens is not None else request_block.get("max_tokens", 512),
            "request.max_tokens",
        ),
    )

    runs = RunSettings(
        cold_runs=_ensure_non_negative_int(
            cli_cold_runs if cli_cold_runs is not None else runs_block.get("cold_runs", 1),
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
        else None
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
        benchmark_spec_path=benchmark_spec_path,
        benchmark_answer_key_path=benchmark_answer_key_path,
        benchmark_question_timeout_sec=benchmark_question_timeout_sec,
        benchmark_ghidra_tool_mode=benchmark_ghidra_tool_mode,
        docker_image=docker_image,
        docker_platform=docker_platform,
        docker_api_base=docker_api_base,
    )
