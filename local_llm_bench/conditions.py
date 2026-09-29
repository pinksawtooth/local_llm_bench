from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .persistence import atomic_write_json, server_identity

PROTOCOL = "load-first-repeat-v1"
EXTERNAL_PROTOCOL = "external-server-repeat-v1"
_SPLIT = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path, cache_dir: Path | None = None) -> str:
    path = path.expanduser().resolve()
    before = path.stat()
    identity = [str(path), before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino]
    cache = cache_dir / f"{digest(identity)}.json" if cache_dir else None
    if cache and cache.exists():
        data = json.loads(cache.read_text())
        if data.get("identity") == identity and re.fullmatch(r"[0-9a-f]{64}", data.get("sha256", "")):
            return data["sha256"]
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError(f"ハッシュ計算中にファイルが変更されました: {path}")
    result = hasher.hexdigest()
    if cache:
        atomic_write_json(cache, {"identity": identity, "sha256": result})
    return result


def model_artifact(model_info: dict, cache_dir: Path) -> dict:
    path = Path(str(model_info.get("path") or "")).expanduser()
    if model_info.get("path") and model_info.get("format") == "MLX" and path.is_dir():
        return mlx_artifact(path, cache_dir)
    if not path.is_file():
        return {"status": "unavailable", "sha256": None, "files": []}
    paths = [path]
    split = _SPLIT.match(path.name)
    if split:
        prefix, _, count = split.groups()
        paths = [path.with_name(f"{prefix}-{i:05d}-of-{count}.gguf") for i in range(1, int(count) + 1)]
        if not all(item.is_file() for item in paths):
            raise RuntimeError(f"GGUFの分割ファイルが揃っていません: {path}")
    files = [{"name": item.name, "size_bytes": item.stat().st_size, "sha256": file_hash(item, cache_dir)} for item in paths]
    return {"status": "measured", "files": files, "sha256": files[0]["sha256"] if len(files) == 1 else digest([item["sha256"] for item in files])}


def mlx_artifact(directory: Path, cache_dir: Path) -> dict:
    """Fingerprint local weights plus model/tokenizer configuration, not a model alias."""
    weights = set(directory.glob("*.safetensors"))
    if not weights or not (directory / "config.json").is_file():
        return {"status": "unavailable", "sha256": None, "files": []}
    for index in directory.glob("*.safetensors.index.json"):
        from .http_boundary import MAX_HTTP_RESPONSE_BYTES, strict_json_loads
        with index.open("rb") as stream:
            raw = stream.read(MAX_HTTP_RESPONSE_BYTES + 1)
        if len(raw) > MAX_HTTP_RESPONSE_BYTES:
            raise ValueError("MLX weight index is too large")
        value = strict_json_loads(raw)
        mapping = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("MLX weight index has no weight_map")
        for name in mapping.values():
            if (not isinstance(name, str) or Path(name).name != name or "\\" in name
                    or not name.endswith(".safetensors") or directory / name not in weights):
                raise ValueError("MLX weight index references a missing or invalid shard")
    paths = set(weights)
    for pattern in ("config.json", "*_config.json", "*.safetensors.index.json", "tokenizer.json",
                    "special_tokens_map.json", "vocab.json", "added_tokens.json", "*.model",
                    "*.tiktoken", "*.jinja", "chat_templates/*.jinja", "merges.txt", "vocab.txt"):
        paths.update(directory.glob(pattern))
    if not all(item.is_file() for item in paths):
        raise ValueError("MLX model artifact contains an unreadable file")
    files = [{"name": str(item.relative_to(directory)), "size_bytes": item.stat().st_size,
              "sha256": file_hash(item, cache_dir)} for item in sorted(paths)]
    return {"status": "measured", "format": "MLX", "files": files, "sha256": digest(files)}


def workload(config) -> dict:
    if config.mode == "performance":
        from .performance import WORKLOAD_VERSION
        return {"generator": WORKLOAD_VERSION, "performance": asdict(config.performance),
                "output_policy": "saved_or_explicit_request_settings"}
    if config.mode != "docker_task":
        return {"prompt": config.prompt_text}
    from .docker_task.spec import load_spec

    spec = load_spec(config.benchmark_spec_path, config.benchmark_answer_key_path)
    questions = []
    for question in spec.questions:
        entry = asdict(question)
        entry["binary_path"] = file_hash(question.binary_path) if question.binary_path else None
        questions.append(entry)
    return {"id": spec.id, "questions": questions}


def execution_contract(config, model: str, parallelism: int | None) -> dict:
    ds4_launch = None
    if config.provider == "ds4":
        from .ds4 import normalize_ds4_launch_config
        ds4_launch = normalize_ds4_launch_config(config.ds4, directory=config.config_path.parent if config.config_path else Path.cwd())
    package = Path(__file__).resolve().parent
    sources = [*package.rglob("*.py"), package.parent / "benchmark.py"]
    harness = digest({str(path.relative_to(package.parent)): file_hash(path) for path in sorted(sources)})
    override = os.environ.get("LOCAL_LLM_BENCH_DOCKER_GHIDRA_MCP_SOURCE_PATH") or os.environ.get("REV_BENCH_DOCKER_GHIDRA_MCP_SOURCE_PATH")
    override_sources = None
    if config.mode == "docker_task" and override:
        root = Path(override).expanduser().resolve()
        override_sources = {"path": str(root), "python_sources_sha256": digest({str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*.py")) if ".git" not in path.parts and "__pycache__" not in path.parts})}
    return {
        "harness_sha256": harness, "ghidra_source_override": override_sources,
        "evaluation": evaluation_identity(config),
        "provider": config.provider, "server": server_identity(config.api_base), "model": model,
        "mode": config.mode, "request": config.recorded_request_parameters(), "runs": asdict(config.runs),
        "load": {**{key: value for key, value in asdict(config.lmstudio_load).items() if key != "parallelism_sweep"}, "parallelism": parallelism if parallelism is not None else config.lmstudio_load.parallelism},
        "workload_hash": digest(workload(config)), "docker_image": config.docker_image,
        "docker_platform": config.docker_platform, "docker_api_base": config.docker_api_base,
        "question_timeout_sec": config.benchmark_question_timeout_sec,
        "ghidra_tool_mode": config.benchmark_ghidra_tool_mode,
        "protocol": EXTERNAL_PROTOCOL if ds4_launch and ds4_launch["management"] == "external" else PROTOCOL,
        **({"ds4_launch_config": ds4_launch} if ds4_launch else {}),
    }


def evaluation_identity(config) -> dict:
    from .inspect_harness import INSPECT_VERSION, PROFILE_VERSION, dependency_lock_hash
    from .inspect_log import LOG_FORMAT
    return {"name": "inspect", "version": INSPECT_VERSION, "profile": PROFILE_VERSION,
            "log_format": LOG_FORMAT,
            "dependency_lock_sha256": dependency_lock_hash(),
            "limits": asdict(config.inspect), "epochs": 1,
            "max_samples": max(config.performance.concurrency) if config.mode == "performance" else 1,
            "model_retries": 0, "sample_retries": 0, "cache": False,
            "scorer": "host_typed_v1" if config.mode == "docker_task" else None}


def capture_conditions(config, model_info: dict, contract: dict, preflight: dict) -> dict:
    runtime = model_info.get("runtime") or {}
    effective = model_info.get("load_config")
    reported = model_info.get("reported_inference") or {}
    def observed(name):
        return {"value": reported.get(name), "source": "provider_response" if name in reported else "unavailable"}
    host = preflight.get("host", {})
    return {
        "schema_version": 1, "protocol": contract["protocol"],
        "hardware": {"scope": "benchmark_host", "system": platform.system(), "machine": platform.machine(), "os_release": platform.release(), **{key: host.get(key) for key in ("cpu_count", "cpu_model", "hardware_model", "memory_total_bytes")}},
        "provider": config.provider, "server": contract["server"],
        "model": {"identifier": contract["model"], "format": model_info.get("format"), "quantization": model_info.get("quantization_name"), "artifact": model_artifact(model_info, config.output.run_logs_dir.parent / ".hash-cache")},
        "runtime": {"engine": runtime.get("engine"), "version": runtime.get("version"), "installed_version": runtime.get("installed_version"), "source": runtime.get("source", "provider_response" if runtime else "unavailable")},
        "request": config.recorded_request_parameters(),
        "load": {"requested": contract.get("ds4_launch_config", contract["load"]), "effective": effective},
        "sampling": observed("sampling"), "thinking": observed("thinking"), "speculative_decoding": observed("speculative_decoding"),
        "provider_evidence": {key: model_info.get(key) for key in ("inference_defaults", "reported_speculation", "reasoning_capabilities", "model_identity_scope", "capture_status", "local_artifact_identity_status", "process_config_status", "load_config_scope", "server_management", "launch_config_sha256", "configured_context_length", "runtime_features", "reported_model_path")},
        "harness": {"sha256": contract["harness_sha256"], "python_version": platform.python_version(), "ghidra_source_override": contract["ghidra_source_override"], **contract["evaluation"]},
        "measurement": {key: contract[key] for key in ("mode", "runs", "question_timeout_sec", "ghidra_tool_mode")},
        "cache": {"state": "unknown", "source": "not_observed"},
        "workload_hash": contract["workload_hash"], "docker": preflight.get("docker"),
    }


def unknown_conditions(conditions: dict) -> list[str]:
    missing = []
    for name, value in (
        ("model.artifact.sha256", conditions.get("model", {}).get("artifact", {}).get("sha256")),
        ("runtime.engine", conditions.get("runtime", {}).get("engine")),
        ("runtime.version", conditions.get("runtime", {}).get("version")),
        ("load.effective", conditions.get("load", {}).get("effective")),
        ("sampling", conditions.get("sampling", {}).get("value")),
        ("thinking", conditions.get("thinking", {}).get("value")),
        ("speculative_decoding", conditions.get("speculative_decoding", {}).get("value")),
    ):
        if value is None or value == "" or (isinstance(value, (dict, list)) and not value):
            missing.append(name)
    if conditions.get("cache", {}).get("state") in (None, "unknown"):
        missing.append("cache.state")
    if conditions.get("provider_evidence", {}).get("load_config_scope") == "context_length_only":
        missing.append("load.other_effective_settings")
    if conditions.get("server") and not conditions["server"].startswith(("http://localhost:", "https://localhost:")):
        missing.append("inference_host")
    return missing


def comparison_metadata(conditions: dict, run_id: str, *, recovered: bool = False) -> dict:
    unknown = unknown_conditions(conditions)
    fingerprint = digest(conditions)
    # Equal missing values are not evidence that two executions were equivalent.
    group = f"{fingerprint[:12]}:{run_id}" if unknown or recovered else fingerprint[:12]
    return {"fingerprint": fingerprint, "group_id": group, "unknown_fields": unknown, "verification": "incomplete" if unknown else "recorded", "recovered": recovered}


def condition_differences(left: Any, right: Any, prefix: str = "") -> list[dict]:
    if isinstance(left, dict) and isinstance(right, dict):
        return [difference for key in sorted(left.keys() | right.keys()) for difference in condition_differences(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else key)]
    return [] if left == right else [{"field": prefix, "left": left, "right": right}]
