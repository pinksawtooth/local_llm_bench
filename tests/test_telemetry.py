from __future__ import annotations

import unittest

from local_llm_bench.telemetry import (
    TelemetryRecorder,
    build_failed_turn_usage_record,
    build_turn_usage_record,
    prompt_breakdown_from_messages,
)


class FakeClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class TelemetryTests(unittest.TestCase):
    def test_recorder_captures_spans_events_and_numeric_metrics(self) -> None:
        clock = FakeClock(10.0)
        recorder = TelemetryRecorder(
            run_id="demo",
            started_at="2026-04-01T00:00:00+00:00",
            now_fn=clock,
            wall_time_fn=lambda: "2026-04-01T00:00:00+00:00",
            origin_perf=10.0,
            source="test",
        )

        span = recorder.start_span("attempt", phase="warm", iteration=1)
        clock.advance(0.25)
        duration_ms = span.finish(
            status="success",
            metrics={
                "total_latency_ms": 123.4,
                "completion_tokens": 12,
                "response_text": "not a metric",
            },
        )
        clock.advance(0.05)
        payload = recorder.build(
            ended_at="2026-04-01T00:00:01+00:00",
            duration_ms=300.0,
        )

        self.assertAlmostEqual(duration_ms, 250.0)
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["source"], "test")
        self.assertEqual(payload["summary"]["attempt_count"], 1)
        self.assertEqual(payload["summary"]["span_count"], 1)
        self.assertEqual(payload["spans"][0]["status"], "success")
        self.assertEqual(payload["spans"][0]["metrics"]["completion_tokens"], 12)
        self.assertNotIn("response_text", payload["spans"][0]["metrics"])
        self.assertEqual(
            [event["name"] for event in payload["events"]],
            ["run_start", "attempt_start", "attempt_end", "run_end"],
        )
        self.assertGreaterEqual(payload["resources"]["summary"]["sample_count"], 3)

    def test_prompt_breakdown_estimates_context_categories(self) -> None:
        breakdown = prompt_breakdown_from_messages(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task prompt"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "demo", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
            ],
            [{"type": "function", "function": {"name": "demo"}}],
            tool_call_names_by_id={"call-1": "demo"},
        )

        self.assertEqual(breakdown["role_counts"]["tool"], 1)
        self.assertIn("system", breakdown["categories"])
        self.assertIn("task", breakdown["categories"])
        self.assertIn("tool_results", breakdown["categories"])
        self.assertIn("tool_schema", breakdown["categories"])
        self.assertIn("assistant_tool_calls", breakdown["categories"])
        self.assertGreater(breakdown["estimated_total_tokens"], 0)
        self.assertIn("demo", breakdown["tool_results_by_tool"])

    def test_turn_usage_records_compute_rates_and_failures(self) -> None:
        success = build_turn_usage_record(
            source="test",
            turn_index=2,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cumulative_prompt_tokens=20,
            cumulative_completion_tokens=8,
            elapsed_sec=0.5,
            ttft_sec=0.1,
            prompt_breakdown={"categories": {}},
        )
        failed = build_failed_turn_usage_record(
            source="test",
            turn_index=3,
            error_type="timeout",
            error_message="timed out",
            cumulative_prompt_tokens=20,
            cumulative_completion_tokens=8,
            elapsed_sec=1.0,
            timed_out=True,
        )

        self.assertTrue(success["success"])
        self.assertEqual(success["turn_index"], 2)
        self.assertEqual(success["cumulative_total_tokens"], 28)
        self.assertAlmostEqual(success["completion_tokens_per_sec"], 10.0)
        self.assertAlmostEqual(success["ttft_sec"], 0.1)
        self.assertFalse(failed["success"])
        self.assertTrue(failed["timed_out"])
        self.assertEqual(failed["error_type"], "timeout")
        self.assertEqual(failed["completion_tokens_per_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
