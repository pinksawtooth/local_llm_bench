from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_llm_bench.config import BenchmarkConfig, OutputSettings, RequestSettings, RunSettings
from local_llm_bench.history import load_history_entries, update_history, write_latest
from local_llm_bench.lmstudio_api import LMStudioAPIError, StreamResult
from local_llm_bench.report import write_report_html
from local_llm_bench.runner import run_benchmark
from local_llm_bench.run_logs import persist_run_logs


class RunnerIntegrationTests(unittest.TestCase):
    def test_runner_persists_errors_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = BenchmarkConfig(
                api_base="http://localhost:1234/v1",
                models=["bench-model"],
                prompt_text="pythonでライブラリを使わずにRC4を実装して",
                request=RequestSettings(temperature=0.0, max_tokens=128),
                runs=RunSettings(cold_runs=1, warm_runs=2, timeout_sec=10.0, cooldown_sec=0.0),
                output=OutputSettings(
                    history_json=root / "runs" / "history.json",
                    latest_json=root / "runs" / "latest_run.json",
                    report_html=root / "docs" / "index.html",
                    run_logs_dir=root / "runs" / "logs",
                ),
            )

            calls = {"count": 0}

            def fake_client(**_: object) -> StreamResult:
                calls["count"] += 1
                if calls["count"] == 1:
                    return StreamResult(
                        response_text="ok",
                        reasoning_text="thinking",
                        ttft_ms=120.0,
                        total_latency_ms=800.0,
                        completion_window_ms=680.0,
                        prompt_tokens=40,
                        initial_prompt_tokens=40,
                        initial_prompt_latency_ms=120.0,
                        initial_prompt_tps=333.33,
                        conversation_prompt_tokens=40,
                        conversation_prompt_latency_ms=800.0,
                        conversation_prompt_tps=50.0,
                        completion_tokens=20,
                        total_tokens=60,
                        decode_tps=29.41,
                        end_to_end_tps=25.0,
                        approx_prompt_tps=333.33,
                        finish_reason="stop",
                    )
                if calls["count"] == 2:
                    raise TimeoutError("timed out")
                raise LMStudioAPIError("empty streamed response")

            run_data = run_benchmark(config, client=fake_client, sleep_fn=lambda _: None, now_fn=lambda: 1.0)
            run_data["model_info"] = {
                "display_name": "Bench Model",
                "format": "gguf",
                "quantization": "Q4_K_M",
            }
            run_data = persist_run_logs(config.output, run_data)
            update_history(config.output.history_json, run_data)
            write_latest(config.output.latest_json, run_data)
            history = load_history_entries(config.output.history_json)
            write_report_html(config.output.report_html, config.output.history_json)
            latest = json.loads(config.output.latest_json.read_text(encoding="utf-8"))
            history_raw = json.loads(config.output.history_json.read_text(encoding="utf-8"))
            report_text = config.output.report_html.read_text(encoding="utf-8")

            self.assertEqual(len(run_data["records"]), 3)
            self.assertEqual([record["status"] for record in run_data["records"]], ["success", "timeout", "error"])
            self.assertEqual(run_data["records"][0]["reasoning_text"], "thinking")
            self.assertAlmostEqual(run_data["records"][0]["initial_prompt_tps"], 333.33, places=2)
            self.assertAlmostEqual(run_data["records"][0]["conversation_prompt_tps"], 50.0, places=2)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["run_id"], run_data["run_id"])
            self.assertEqual(history[0]["model"], "bench-model")
            self.assertEqual(history[0]["model_info"]["display_name"], "Bench Model")
            self.assertNotIn("model", history_raw[0]["records"][0])
            self.assertTrue((root / "runs" / "logs" / run_data["run_id"] / "manifest.json").exists())
            self.assertTrue(history[0]["log_manifest_path"].endswith(f"{run_data['run_id']}/manifest.json"))
            self.assertTrue(history[0]["records"][0]["log_path"].endswith("cold-1.json"))
            self.assertEqual(latest["run_id"], run_data["run_id"])
            self.assertIn("Local LLM Bench", report_text)
            self.assertIn("../runs/history.json", report_text)
            self.assertIn("Quantization", report_text)
