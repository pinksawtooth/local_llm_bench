from pathlib import Path
from dataclasses import fields
from unittest.mock import MagicMock

from local_llm_bench.config import BenchmarkConfig, OutputSettings, RequestSettings, RunSettings
from local_llm_bench.lmstudio_api import StreamResult


def make_config(root: Path, *, cold=1, warm=2):
    return BenchmarkConfig(
        api_base="http://localhost:1234/v1", models=["model-a"], prompt_text="hello",
        request=RequestSettings(temperature=0, max_tokens=128),
        runs=RunSettings(cold_runs=cold, warm_runs=warm, timeout_sec=10, cooldown_sec=0),
        output=OutputSettings(root / "runs/history.json", root / "runs/latest.json", root / "docs/index.html", root / "runs/logs"),
    )


def response(text="answer"):
    values = {f.name: None for f in fields(StreamResult)}
    values.update(response_text=text, reasoning_text="", total_latency_ms=100.0, ttft_ms=10.0, finish_reason="stop")
    return StreamResult(**values)


def runtime_for(root: Path):
    path = root / "model-Q4_K_M.gguf"
    path.write_bytes(b"GGUF fixture")
    runtime = MagicMock()
    runtime.measurement_metadata.side_effect = lambda info: info
    runtime.inspect_model.return_value = {"identifier": "model-a", "path": str(path)}
    runtime.prepare_model.side_effect = lambda model, **kwargs: (model + "-loaded", {
        "identifier": model + "-loaded", "format": "gguf", "quantization_name": "Q4_K_M", "path": str(path),
        "runtime": {"engine": "llama.cpp", "version": "fixture-1"},
        "load_config": {"context_length": 4096, "parallel": kwargs.get("lmstudio_parallelism")},
        "reported_inference": {"thinking": False, "speculative_decoding": False, "sampling": {"temperature": 0}},
    })
    runtime.unload_model.return_value = []
    runtime.chat_client.return_value = MagicMock(side_effect=lambda **kwargs: response())
    runtime.docker_environment.return_value = {}
    return runtime
