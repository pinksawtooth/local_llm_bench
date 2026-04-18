from __future__ import annotations

import unittest

from local_llm_bench.stats import compute_run_summary


class StatsTests(unittest.TestCase):
    def test_compute_run_summary_builds_cold_warm_deltas(self) -> None:
        run_data = {
            "records": [
                {
                    "model": "model-a",
                    "phase": "cold",
                    "status": "success",
                    "finish_reason": "stop",
                    "ttft_ms": 100.0,
                    "total_latency_ms": 1000.0,
                    "completion_window_ms": 900.0,
                    "prompt_tokens": 80,
                    "completion_tokens": 40,
                    "decode_tps": 44.4,
                    "end_to_end_tps": 40.0,
                    "approx_prompt_tps": 800.0,
                },
                {
                    "model": "model-a",
                    "phase": "warm",
                    "status": "success",
                    "finish_reason": "stop",
                    "ttft_ms": 50.0,
                    "total_latency_ms": 500.0,
                    "completion_window_ms": 450.0,
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "decode_tps": 55.0,
                    "end_to_end_tps": 50.0,
                    "approx_prompt_tps": 2000.0,
                },
                {
                    "model": "model-a",
                    "phase": "warm",
                    "status": "success",
                    "finish_reason": "stop",
                    "ttft_ms": 60.0,
                    "total_latency_ms": 520.0,
                    "completion_window_ms": 460.0,
                    "prompt_tokens": 110,
                    "completion_tokens": 25,
                    "decode_tps": 54.0,
                    "end_to_end_tps": 48.0,
                    "approx_prompt_tps": 1833.0,
                },
                {
                    "model": "model-b",
                    "phase": "cold",
                    "status": "success",
                    "finish_reason": "stop",
                    "ttft_ms": 200.0,
                    "total_latency_ms": 1200.0,
                    "completion_window_ms": 1000.0,
                    "prompt_tokens": 70,
                    "completion_tokens": 30,
                    "decode_tps": 30.0,
                    "end_to_end_tps": 25.0,
                    "approx_prompt_tps": 350.0,
                },
                {
                    "model": "model-b",
                    "phase": "warm",
                    "status": "error",
                    "finish_reason": None,
                    "ttft_ms": None,
                    "total_latency_ms": None,
                    "completion_window_ms": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "decode_tps": None,
                    "end_to_end_tps": None,
                    "approx_prompt_tps": None,
                },
            ]
        }

        summary = compute_run_summary(run_data)
        model_a = next(row for row in summary["models"] if row["model"] == "model-a")
        model_b = next(row for row in summary["models"] if row["model"] == "model-b")

        self.assertEqual(summary["total_models"], 2)
        self.assertEqual(summary["total_samples"], 5)
        self.assertAlmostEqual(model_a["warm_mean_total_latency_ms"], 510.0, places=4)
        self.assertAlmostEqual(model_a["delta"]["total_latency_ms"], -490.0, places=4)
        self.assertAlmostEqual(model_a["overall_mean_completion_tokens"], 30.0, places=4)
        self.assertAlmostEqual(model_b["success_rate"], 0.5, places=4)
        self.assertEqual(summary["cards"]["fastest_warm_latency"]["model"], "model-a")

    def test_compute_run_summary_includes_benchmark_metrics(self) -> None:
        run_data = {
            "records": [
                {
                    "model": "model-a",
                    "phase": "warm",
                    "status": "success",
                    "finish_reason": "stop",
                    "ttft_ms": 50.0,
                    "total_latency_ms": 500.0,
                    "completion_window_ms": 450.0,
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "decode_tps": 55.0,
                    "end_to_end_tps": 50.0,
                    "approx_prompt_tps": 2000.0,
                    "benchmark_score": 1.0,
                    "benchmark_correct_count": 1,
                    "benchmark_incorrect_count": 0,
                    "benchmark_error_count": 0,
                },
                {
                    "model": "model-a",
                    "phase": "warm",
                    "status": "success",
                    "finish_reason": "stop",
                    "ttft_ms": 55.0,
                    "total_latency_ms": 520.0,
                    "completion_window_ms": 465.0,
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "decode_tps": 53.0,
                    "end_to_end_tps": 48.0,
                    "approx_prompt_tps": 1800.0,
                    "benchmark_score": 0.0,
                    "benchmark_correct_count": 0,
                    "benchmark_incorrect_count": 1,
                    "benchmark_error_count": 0,
                },
                {
                    "model": "model-a",
                    "phase": "warm",
                    "status": "timeout",
                    "finish_reason": None,
                    "ttft_ms": None,
                    "total_latency_ms": None,
                    "completion_window_ms": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "decode_tps": None,
                    "end_to_end_tps": None,
                    "approx_prompt_tps": None,
                    "benchmark_score": 0.0,
                    "benchmark_correct_count": 0,
                    "benchmark_incorrect_count": 0,
                    "benchmark_error_count": 1,
                },
            ]
        }

        summary = compute_run_summary(run_data)
        model_a = summary["models"][0]

        self.assertAlmostEqual(model_a["warm_mean_benchmark_score"], 0.5, places=4)
        self.assertAlmostEqual(model_a["warm_benchmark_correct_rate"], 1 / 3, places=4)
        self.assertAlmostEqual(model_a["warm_benchmark_error_rate"], 1 / 3, places=4)
