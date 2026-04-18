from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import benchmark
from local_llm_bench.config import BenchmarkConfig, OutputSettings, RequestSettings, RunSettings


def _make_config(root: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        api_base="http://localhost:1234/v1",
        models=["model-a", "model-b"],
        prompt_text="pythonでライブラリを使わずにRC4を実装して",
        request=RequestSettings(temperature=0.0, max_tokens=128),
        runs=RunSettings(cold_runs=1, warm_runs=1, timeout_sec=10.0, cooldown_sec=0.0),
        output=OutputSettings(
            history_json=root / "runs" / "history.json",
            latest_json=root / "runs" / "latest_run.json",
            report_html=root / "docs" / "index.html",
            run_logs_dir=root / "runs" / "logs",
        ),
    )


def _run_data(model: str) -> dict[str, object]:
    return {
        "run_id": f"{model}-run",
        "summary": {
            "models": [
                {
                    "model": model,
                    "warm_mean_total_latency_ms": 100.0,
                    "warm_mean_ttft_ms": 20.0,
                    "warm_mean_decode_tps": 30.0,
                }
            ]
        },
    }


class BenchmarkMainTests(unittest.TestCase):
    @patch("benchmark.ensure_report_html", return_value=False)
    @patch("benchmark.build_provider_runtime")
    @patch("benchmark.write_latest")
    @patch("benchmark.update_history")
    @patch("benchmark.run_benchmark")
    @patch("benchmark.load_config")
    def test_main_unloads_each_model_by_default(
        self,
        load_config_mock,
        run_benchmark_mock,
        update_history_mock,
        write_latest_mock,
        build_provider_runtime_mock,
        ensure_report_html_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            load_config_mock.return_value = config
            run_benchmark_mock.side_effect = [_run_data("model-a"), _run_data("model-b")]
            runtime = MagicMock()
            runtime.prepare_model.side_effect = [
                (
                    "model-a-id",
                    {
                        "identifier": "model-a-id",
                        "display_name": "Model A",
                        "format": "gguf",
                        "quantization": "Q4_K_M",
                    },
                ),
                (
                    "model-b-id",
                    {
                        "identifier": "model-b-id",
                        "display_name": "Model B",
                        "format": "mlx",
                        "quantization": "4-bit",
                    },
                ),
            ]
            runtime.unload_model.side_effect = [[], []]
            runtime.chat_client.return_value = "client-token"
            build_provider_runtime_mock.return_value = runtime

            exit_code = benchmark.main(["--config", "bench.yaml"])

        self.assertEqual(exit_code, 0)
        build_provider_runtime_mock.assert_called_once_with(config)
        self.assertEqual(
            run_benchmark_mock.call_args_list,
            [
                call(config, model="model-a-id", requested_model="model-a", client="client-token"),
                call(config, model="model-b-id", requested_model="model-b", client="client-token"),
            ],
        )
        self.assertEqual(
            runtime.unload_model.call_args_list,
            [call("model-a-id"), call("model-b-id")],
        )
        self.assertEqual(
            update_history_mock.call_args_list[0].args[1]["model_info"]["format"],
            "gguf",
        )
        self.assertEqual(
            update_history_mock.call_args_list[1].args[1]["model_info"]["display_name"],
            "Model B",
        )
        self.assertEqual(
            update_history_mock.call_args_list[0].args[1]["api_model"],
            "model-a-id",
        )
        self.assertEqual(update_history_mock.call_count, 2)
        self.assertEqual(write_latest_mock.call_count, 2)
        ensure_report_html_mock.assert_called_once()

    @patch("benchmark.ensure_report_html", return_value=False)
    @patch("benchmark.build_provider_runtime")
    @patch("benchmark.write_latest")
    @patch("benchmark.update_history")
    @patch("benchmark.run_benchmark")
    @patch("benchmark.load_config")
    def test_main_skips_unload_when_keep_loaded_is_set(
        self,
        load_config_mock,
        run_benchmark_mock,
        update_history_mock,
        write_latest_mock,
        build_provider_runtime_mock,
        ensure_report_html_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            load_config_mock.return_value = config
            run_benchmark_mock.side_effect = [_run_data("model-a"), _run_data("model-b")]
            runtime = MagicMock()
            runtime.prepare_model.side_effect = [("model-a", None), ("model-b", None)]
            runtime.describe_model.return_value = None
            runtime.chat_client.return_value = "client-token"
            build_provider_runtime_mock.return_value = runtime

            exit_code = benchmark.main(["--config", "bench.yaml", "--keep-loaded"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run_benchmark_mock.call_args_list,
            [
                call(config, model="model-a", requested_model="model-a", client="client-token"),
                call(config, model="model-b", requested_model="model-b", client="client-token"),
            ],
        )
        runtime.unload_model.assert_not_called()
        self.assertEqual(update_history_mock.call_count, 2)
        self.assertEqual(write_latest_mock.call_count, 2)
        ensure_report_html_mock.assert_called_once()

    @patch("benchmark.ensure_report_html", return_value=False)
    @patch("benchmark.build_provider_runtime")
    @patch("benchmark.write_latest")
    @patch("benchmark.update_history")
    @patch("benchmark.run_benchmark")
    @patch("benchmark.load_config")
    def test_main_resolves_model_info_after_run_when_not_loaded_beforehand(
        self,
        load_config_mock,
        run_benchmark_mock,
        update_history_mock,
        write_latest_mock,
        build_provider_runtime_mock,
        ensure_report_html_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            load_config_mock.return_value = config
            run_benchmark_mock.side_effect = [_run_data("model-a"), _run_data("model-b")]
            runtime = MagicMock()
            runtime.prepare_model.side_effect = [("model-a", None), ("model-b", None)]
            runtime.describe_model.side_effect = [
                {"identifier": "model-a-loaded", "format": "gguf", "quantization": "F16"},
                {"identifier": "model-b-loaded", "format": "mlx", "quantization": "4bit"},
            ]
            runtime.unload_model.side_effect = [[], []]
            runtime.chat_client.return_value = "client-token"
            build_provider_runtime_mock.return_value = runtime

            exit_code = benchmark.main(["--config", "bench.yaml"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run_benchmark_mock.call_args_list,
            [
                call(config, model="model-a", requested_model="model-a", client="client-token"),
                call(config, model="model-b", requested_model="model-b", client="client-token"),
            ],
        )
        self.assertEqual(
            runtime.unload_model.call_args_list,
            [call("model-a-loaded"), call("model-b-loaded")],
        )
        self.assertEqual(update_history_mock.call_args_list[0].args[1]["api_model"], "model-a-loaded")
        self.assertEqual(update_history_mock.call_args_list[1].args[1]["api_model"], "model-b-loaded")
        self.assertEqual(update_history_mock.call_args_list[0].args[1]["model_info"]["quantization"], "F16")
        self.assertEqual(update_history_mock.call_args_list[1].args[1]["model_info"]["format"], "mlx")
        self.assertEqual(write_latest_mock.call_count, 2)
        ensure_report_html_mock.assert_called_once()

    @patch("benchmark.ensure_report_html", return_value=False)
    @patch("benchmark.build_provider_runtime")
    @patch("benchmark.write_latest")
    @patch("benchmark.update_history")
    @patch("benchmark.run_docker_task_benchmark")
    @patch("benchmark.run_benchmark")
    @patch("benchmark.load_config")
    def test_main_dispatches_docker_task_mode(
        self,
        load_config_mock,
        run_benchmark_mock,
        run_docker_task_benchmark_mock,
        update_history_mock,
        write_latest_mock,
        build_provider_runtime_mock,
        ensure_report_html_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            config.mode = "docker_task"
            config.docker_image = "local-llm-bench:bench"
            config.benchmark_spec_path = Path(tmpdir) / "spec.yaml"
            load_config_mock.return_value = config
            run_docker_task_benchmark_mock.side_effect = [_run_data("model-a"), _run_data("model-b")]
            runtime = MagicMock()
            runtime.prepare_model.side_effect = [("model-a", None), ("model-b", None)]
            runtime.describe_model.return_value = None
            runtime.unload_model.return_value = []
            runtime.docker_environment.return_value = {"TOKEN": "secret"}
            build_provider_runtime_mock.return_value = runtime

            exit_code = benchmark.main(["--config", "bench.yaml"])

        self.assertEqual(exit_code, 0)
        run_benchmark_mock.assert_not_called()
        self.assertEqual(
            run_docker_task_benchmark_mock.call_args_list,
            [
                call(config, model="model-a", requested_model="model-a", docker_env={"TOKEN": "secret"}),
                call(config, model="model-b", requested_model="model-b", docker_env={"TOKEN": "secret"}),
            ],
        )
        self.assertEqual(update_history_mock.call_count, 2)
        self.assertEqual(write_latest_mock.call_count, 2)
        ensure_report_html_mock.assert_called_once()

    @patch("benchmark.ensure_report_html", return_value=False)
    @patch("benchmark.build_provider_runtime")
    @patch("benchmark.load_config")
    def test_main_does_not_unload_when_prepare_model_fails(
        self,
        load_config_mock,
        build_provider_runtime_mock,
        ensure_report_html_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            config.models = ["model-a"]
            load_config_mock.return_value = config
            runtime = MagicMock()
            runtime.prepare_model.side_effect = RuntimeError("HTTPError 401: invalid token")
            build_provider_runtime_mock.return_value = runtime

            with self.assertRaisesRegex(RuntimeError, "invalid token"):
                benchmark.main(["--config", "bench.yaml"])

        runtime.unload_model.assert_not_called()
        ensure_report_html_mock.assert_not_called()
