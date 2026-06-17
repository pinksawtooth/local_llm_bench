from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

from local_llm_bench.lmstudio_cli import (
    describe_loaded_model,
    load_model_with_config,
    unload_matching_models,
    unload_matching_models_via_api,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class LMStudioCLITests(unittest.TestCase):
    def test_load_model_with_config_sends_parallel_as_load_parameter(self) -> None:
        requests: list[tuple[str, dict[str, object]]] = []

        def fake_urlopen(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            requests.append((request.full_url, payload))
            return _FakeHTTPResponse(
                {
                    "type": "llm",
                    "instance_id": "bench-model",
                    "status": "loaded",
                    "load_config": {"parallel": 3},
                }
            )

        response = load_model_with_config(
            "bench-model",
            api_base="http://localhost:1234/v1",
            parallelism=3,
            context_length=4096,
            flash_attention=True,
            urlopen=fake_urlopen,
        )

        self.assertEqual(response["instance_id"], "bench-model")
        self.assertEqual(requests[0][0], "http://localhost:1234/api/v1/models/load")
        self.assertEqual(
            requests[0][1],
            {
                "model": "bench-model",
                "echo_load_config": True,
                "parallel": 3,
                "context_length": 4096,
                "flash_attention": True,
            },
        )

    def test_unload_matching_models_via_api_unloads_loaded_instance(self) -> None:
        calls: list[tuple[str, dict[str, object] | None]] = []

        def fake_urlopen(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8")) if request.data else None
            calls.append((request.full_url, payload))
            if request.full_url.endswith("/api/v1/models"):
                return _FakeHTTPResponse(
                    {
                        "models": [
                            {
                                "key": "bench-model",
                                "display_name": "Bench Model",
                                "loaded_instances": [
                                    {"id": "bench-model-instance", "config": {"parallel": 2}}
                                ],
                            }
                        ]
                    }
                )
            if request.full_url.endswith("/api/v1/models/unload"):
                return _FakeHTTPResponse({"instance_id": payload["instance_id"]})
            raise AssertionError(f"unexpected url: {request.full_url}")

        results = unload_matching_models_via_api(
            "bench-model",
            api_base="http://localhost:1234/v1",
            urlopen=fake_urlopen,
        )

        self.assertEqual(results[0].status, "unloaded")
        self.assertEqual(results[0].target, "bench-model-instance")
        self.assertEqual(calls[-1][1], {"instance_id": "bench-model-instance"})

    def test_describe_loaded_model_reads_format_and_quantization(self) -> None:
        def fake_run(args, **kwargs):
            self.assertEqual(args, ["lms", "ps", "--json"])
            payload = [
                {
                    "identifier": "gpt-oss-20b-mlx",
                    "modelKey": "openai/gpt-oss-20b",
                    "displayName": "GPT-OSS 20B",
                    "format": "mlx",
                    "quantization": {"name": "MXFP4", "bits": 4},
                    "selectedVariant": "openai/gpt-oss-20b@mlx-mxfp4",
                    "publisher": "openai",
                    "architecture": "gpt-oss",
                }
            ]
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")

        info = describe_loaded_model("gpt-oss-20b-mlx", run=fake_run)

        assert info is not None
        self.assertEqual(info["display_name"], "GPT-OSS 20B")
        self.assertEqual(info["format"], "mlx")
        self.assertEqual(info["quantization"], "MXFP4 (4-bit)")
        self.assertEqual(info["quantization_bits"], 4)

    def test_describe_loaded_model_matches_variant_style_requested_name(self) -> None:
        def fake_run(args, **kwargs):
            self.assertEqual(args, ["lms", "ps", "--json"])
            payload = [
                {
                    "identifier": "gpt-oss-20b-f16",
                    "modelKey": "unsloth/gpt-oss-20b",
                    "displayName": "gpt-oss-20b-GGUF",
                    "path": "unsloth/GPT-OSS-20B-GGUF/gpt-oss-20b-F16.gguf",
                    "format": "gguf",
                    "quantization": {"name": "F16", "bits": 16},
                }
            ]
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")

        info = describe_loaded_model("unsloth/gpt-oss-20b@f16", run=fake_run)

        assert info is not None
        self.assertEqual(info["display_name"], "gpt-oss-20b-GGUF")
        self.assertEqual(info["format"], "gguf")
        self.assertEqual(info["quantization"], "F16 (16-bit)")

    def test_describe_loaded_model_falls_back_to_http_api_when_lms_is_missing(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = payload

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

        def fake_run(args, **kwargs):
            raise FileNotFoundError("lms not found")

        payload = {
            "models": [
                {
                    "key": "nvidia/nemotron-3-super",
                    "display_name": "Nemotron 3 Super",
                    "publisher": "nvidia",
                    "architecture": "nemotron_h_moe",
                    "format": "gguf",
                    "quantization": {"name": "Q4_K_M", "bits_per_weight": 4},
                    "selected_variant": "nvidia/nemotron-3-super@q4_k_m",
                    "loaded_instances": [],
                }
            ]
        }

        with patch("local_llm_bench.lmstudio_cli.urllib.request.urlopen", return_value=FakeResponse(payload)) as urlopen_mock:
            info = describe_loaded_model(
                "nvidia/nemotron-3-super",
                run=fake_run,
                api_base="http://localhost:1234/v1",
            )

        assert info is not None
        urlopen_mock.assert_called_once_with("http://localhost:1234/api/v1/models", timeout=15.0)
        self.assertEqual(info["display_name"], "Nemotron 3 Super")
        self.assertEqual(info["format"], "gguf")
        self.assertEqual(info["quantization"], "Q4_K_M (4-bit)")
        self.assertEqual(info["selected_variant"], "nvidia/nemotron-3-super@q4_k_m")

    def test_unload_matching_models_prefers_exact_identifier(self) -> None:
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            if args == ["lms", "ps", "--json"]:
                payload = [
                    {
                        "identifier": "gpt-oss-20b-gguf",
                        "modelKey": "openai/gpt-oss-20b",
                        "selectedVariant": "openai/gpt-oss-20b@gguf-mxfp4",
                    },
                    {
                        "identifier": "gpt-oss-20b-mlx",
                        "modelKey": "openai/gpt-oss-20b",
                        "selectedVariant": "openai/gpt-oss-20b@mlx-mxfp4",
                    },
                ]
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")
            if args == ["lms", "unload", "gpt-oss-20b-mlx"]:
                return subprocess.CompletedProcess(args, 0, stdout="unloaded mlx", stderr="")
            raise AssertionError(f"unexpected args: {args}")

        results = unload_matching_models("gpt-oss-20b-mlx", run=fake_run)

        self.assertEqual(calls, [["lms", "ps", "--json"], ["lms", "unload", "gpt-oss-20b-mlx"]])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "unloaded")
        self.assertEqual(results[0].target, "gpt-oss-20b-mlx")

    def test_unload_matching_models_resolves_identifier_from_model_key(self) -> None:
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            if args == ["lms", "ps", "--json"]:
                payload = [
                    {
                        "identifier": "bench-model-instance",
                        "modelKey": "bench-model",
                    }
                ]
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")
            if args == ["lms", "unload", "bench-model-instance"]:
                return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")
            raise AssertionError(f"unexpected args: {args}")

        results = unload_matching_models("bench-model", run=fake_run)

        self.assertEqual(calls[-1], ["lms", "unload", "bench-model-instance"])
        self.assertEqual(results[0].target, "bench-model-instance")

    def test_unload_matching_models_falls_back_when_ps_fails(self) -> None:
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            if args == ["lms", "ps", "--json"]:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom")
            if args == ["lms", "unload", "bench-model"]:
                return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")
            raise AssertionError(f"unexpected args: {args}")

        results = unload_matching_models("bench-model", run=fake_run)

        self.assertEqual(calls, [["lms", "ps", "--json"], ["lms", "unload", "bench-model"]])
        self.assertEqual(results[0].status, "unloaded")
