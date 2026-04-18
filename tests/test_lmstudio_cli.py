from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

from local_llm_bench.lmstudio_cli import describe_loaded_model, unload_matching_models


class LMStudioCLITests(unittest.TestCase):
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
