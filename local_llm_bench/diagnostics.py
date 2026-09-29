from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .conditions import digest
from .persistence import server_identity


def _read_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def host_snapshot() -> dict:
    result = {"captured_at": datetime.now(timezone.utc).isoformat(), "scope": "benchmark_host", "cpu_count": os.cpu_count(), "memory_total_bytes": None, "memory_available_bytes": None, "swap_used_bytes": None}
    if platform.system() == "Darwin":
        result["hardware_model"] = _read_command(["sysctl", "-n", "hw.model"])
        result["cpu_model"] = _read_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        total = _read_command(["sysctl", "-n", "hw.memsize"])
        result["memory_total_bytes"] = int(total) if total and total.isdigit() else None
        swap = _read_command(["sysctl", "-n", "vm.swapusage"]) or ""
        match = re.search(r"used\s*=\s*([\d.]+)([KMG])", swap)
        if match:
            result["swap_used_bytes"] = int(float(match[1]) * 1024 ** ("KMG".index(match[2]) + 1))
        pressure = _read_command(["memory_pressure", "-Q"]) or ""
        match = re.search(r"free percentage:\s*(\d+)%", pressure)
        result["memory_pressure_free_percent"] = int(match[1]) if match else None
    elif Path("/proc/meminfo").is_file():
        entries = {match[1]: int(match[2]) * 1024 for match in re.finditer(r"^(\w+):\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.MULTILINE)}
        result.update(memory_total_bytes=entries.get("MemTotal"), memory_available_bytes=entries.get("MemAvailable"), swap_used_bytes=entries.get("SwapTotal", 0) - entries.get("SwapFree", 0))
        cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
        match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        result["cpu_model"] = match[1] if match else platform.machine()
    # Only command names are collected, never command lines containing credentials.
    processes = _read_command(["ps", "-axo", "pid=,rss=,comm="]) or ""
    result["inference_processes"] = []
    for line in processes.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and any(name in fields[2].lower() for name in ("llama-server", "llmster", "lm studio", "unsloth", "ds4-server", "omlx", "mlx-serve", "mlxcore")):
            if fields[0].isdigit() and fields[1].isdigit():
                result["inference_processes"].append({"pid": int(fields[0]), "rss_bytes": int(fields[1]) * 1024, "executable": fields[2]})
    return result


def run_preflight(config, runtime, model: str) -> dict:
    from .inspect_harness import require_inspect
    require_inspect()
    result = {"status": "passed", "model": runtime.inspect_model(model), "host": host_snapshot(), "docker": None}
    if config.mode == "docker_task":
        from .docker_task.runner import _docker_binary, _parse_platform
        from .docker_task.spec import load_spec

        load_spec(config.benchmark_spec_path, config.benchmark_answer_key_path)
        docker = _docker_binary()
        version = _read_command([docker, "info", "--format", "{{.ServerVersion}}"])
        if not version:
            raise RuntimeError("Docker daemon に接続できません。")
        active = _read_command([docker, "ps", "--filter", f"label=local-llm-bench.server={digest(server_identity(config.api_base))}", "--format", "{{.ID}}"])
        if active is None:
            raise RuntimeError("Docker の実行中コンテナを確認できません。")
        if active:
            raise RuntimeError("同じ推論サーバーを使うベンチ用コンテナが残っています: " + active.replace("\n", ", "))
        raw = _read_command([docker, "image", "inspect", config.docker_image])
        if not raw:
            build_command = [str(Path(__file__).resolve().parents[1] / "build_bench_image.sh")]
            if config.docker_platform:
                build_command.extend(["--platform", config.docker_platform])
            build_command.extend(["--tag", config.docker_image])
            raise RuntimeError(
                f"Docker image がローカルにありません: {config.docker_image}\n"
                "実行中のベンチが終了してから、以下のコマンドでイメージを作成し、ベンチを再実行してください。\n"
                + shlex.join(build_command)
            )
        entry = json.loads(raw)[0]
        from .inspect_harness import INSPECT_VERSION, PROFILE_VERSION, dependency_lock_hash
        labels = (entry.get("Config") or {}).get("Labels") or {}
        if (labels.get("local-llm-bench.inspect.version") != INSPECT_VERSION or
                labels.get("local-llm-bench.worker.profile") != PROFILE_VERSION or
                labels.get("local-llm-bench.dependencies.sha256") != dependency_lock_hash()):
            raise RuntimeError("Inspect対応のDocker imageが必要です。実行中のベンチ終了後に新しいタグで再ビルドしてください。")
        actual = f"{entry['Os']}/{entry['Architecture']}"
        if config.docker_platform and _parse_platform(actual) != _parse_platform(config.docker_platform):
            raise RuntimeError(f"Docker platform が一致しません: requested={config.docker_platform}, actual={actual}")
        result["docker"] = {"image_id": entry["Id"], "platform": actual, "server_version": version}
    return result
