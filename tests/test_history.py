from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_llm_bench.history import load_history_entries, update_history


class HistoryTests(unittest.TestCase):
    def test_load_history_entries_backfills_error_metadata_for_legacy_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "runs" / "history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "run_id": "legacy",
                            "started_at": "2026-04-01T00:00:00+00:00",
                            "model": "demo-model",
                            "records": [
                                {
                                    "phase": "warm",
                                    "iteration": 1,
                                    "status": "error",
                                    "error": "[q1] HTTPError 500: boom",
                                    "question_results": [
                                        {
                                            "question_id": "q1",
                                            "status": "error",
                                            "error": "worker returned malformed JSON",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            history = load_history_entries(history_path)

        record = history[0]["records"][0]
        self.assertEqual(record["error_signature"], "HTTPError 500: boom")
        self.assertEqual(record["error_category"], "api")
        self.assertEqual(record["question_results"][0]["error_category"], "worker")

    def test_load_history_entries_defaults_prompt_runs_to_zero_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "runs" / "history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "run_id": "legacy-prompt",
                            "started_at": "2026-04-01T00:00:00+00:00",
                            "model": "demo-model",
                            "records": [
                                {
                                    "phase": "warm",
                                    "iteration": 1,
                                    "status": "success",
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            history = load_history_entries(history_path)

        self.assertEqual(history[0]["records"][0]["tool_call_count"], 0)

    def test_load_history_entries_aggregates_tool_metrics_from_question_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "runs" / "history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "run_id": "legacy-tools",
                            "started_at": "2026-04-01T00:00:00+00:00",
                            "model": "demo-model",
                            "records": [
                                {
                                    "phase": "warm",
                                    "iteration": 1,
                                    "status": "success",
                                    "question_results": [
                                        {
                                            "question_id": "q1",
                                            "status": "success",
                                            "tool_call_count": 2,
                                            "tool_name_counts": {"mecha_ghidra.decompile_function": 2},
                                        },
                                        {
                                            "question_id": "q2",
                                            "status": "success",
                                            "tool_name_counts": {"run_python": 1},
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            history = load_history_entries(history_path)

        record = history[0]["records"][0]
        self.assertEqual(record["tool_call_count"], 3)
        self.assertEqual(
            record["tool_name_counts"],
            {"mecha_ghidra.decompile_function": 2, "run_python": 1},
        )

    def test_load_history_entries_backfills_tool_metrics_from_question_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "runs" / "history.json"
            question_log_path = Path(tmpdir) / "runs" / "logs" / "demo" / "questions" / "warm-1-q1.json"
            question_log_path.parent.mkdir(parents=True, exist_ok=True)
            question_log_path.write_text(
                json.dumps(
                    {
                        "parsed_worker_result": {
                            "trace": {
                                "turns": [
                                    {
                                        "turn": 1,
                                        "tool_events": [
                                            {"tool_name": "mecha_ghidra.list_functions"},
                                            {"tool_name": "run_python"},
                                            {"tool_name": "mecha_ghidra.list_functions"},
                                        ],
                                    }
                                ]
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "run_id": "legacy-log-tools",
                            "started_at": "2026-04-01T00:00:00+00:00",
                            "model": "demo-model",
                            "records": [
                                {
                                    "phase": "warm",
                                    "iteration": 1,
                                    "status": "success",
                                    "question_results": [
                                        {
                                            "question_id": "q1",
                                            "status": "success",
                                            "log_path": "logs/demo/questions/warm-1-q1.json",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            history = load_history_entries(history_path)

        question = history[0]["records"][0]["question_results"][0]
        self.assertEqual(question["tool_call_count"], 3)
        self.assertEqual(
            question["tool_name_counts"],
            {"mecha_ghidra.list_functions": 2, "run_python": 1},
        )

    def test_load_history_entries_recomputes_docker_prompt_speed_from_question_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "runs" / "history.json"
            question_log_path = Path(tmpdir) / "runs" / "logs" / "demo" / "questions" / "warm-1-q1.json"
            question_log_path.parent.mkdir(parents=True, exist_ok=True)
            question_log_path.write_text(
                json.dumps(
                    {
                        "parsed_worker_result": {
                            "trace": {
                                "turns": [
                                    {
                                        "request_latency_ms": 10.0,
                                        "usage": {"prompt_tokens": 100},
                                    },
                                    {
                                        "request_latency_ms": 40.0,
                                        "usage": {"prompt_tokens": 200},
                                    },
                                ]
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "run_id": "legacy-prompt-speed",
                            "started_at": "2026-04-01T00:00:00+00:00",
                            "model": "demo-model",
                            "benchmark_mode": "docker_task",
                            "records": [
                                {
                                    "phase": "warm",
                                    "iteration": 1,
                                    "status": "success",
                                    "prompt_tokens": 300,
                                    "approx_prompt_tps": 30000,
                                    "question_results": [
                                        {
                                            "question_id": "q1",
                                            "status": "success",
                                            "prompt_tokens": 300,
                                            "approx_prompt_tps": 30000,
                                            "log_path": "logs/demo/questions/warm-1-q1.json",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            history = load_history_entries(history_path)

        question = history[0]["records"][0]["question_results"][0]
        record = history[0]["records"][0]
        self.assertEqual(question["prompt_tokens"], 300)
        self.assertEqual(question["prompt_latency_ms"], 50.0)
        self.assertAlmostEqual(question["approx_prompt_tps"], 6000.0)
        self.assertEqual(question["initial_prompt_tokens"], 100)
        self.assertEqual(question["initial_prompt_latency_ms"], 10.0)
        self.assertAlmostEqual(question["initial_prompt_tps"], 10000.0)
        self.assertEqual(question["conversation_prompt_tokens"], 300)
        self.assertEqual(question["conversation_prompt_latency_ms"], 50.0)
        self.assertAlmostEqual(question["conversation_prompt_tps"], 6000.0)
        self.assertEqual(record["prompt_tokens"], 300)
        self.assertEqual(record["prompt_latency_ms"], 50.0)
        self.assertAlmostEqual(record["approx_prompt_tps"], 6000.0)
        self.assertEqual(record["initial_prompt_tokens"], 100)
        self.assertEqual(record["initial_prompt_latency_ms"], 10.0)
        self.assertAlmostEqual(record["initial_prompt_tps"], 10000.0)
        self.assertEqual(record["conversation_prompt_tokens"], 300)
        self.assertEqual(record["conversation_prompt_latency_ms"], 50.0)
        self.assertAlmostEqual(record["conversation_prompt_tps"], 6000.0)

    def test_load_history_entries_backfills_prompt_speed_from_same_prompt_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "runs" / "history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "run_id": "seed-run",
                            "started_at": "2026-04-01T00:00:00+00:00",
                            "model": "seed-model",
                            "prompt_text": "pythonでライブラリを使わずにRC4を実装して",
                            "benchmark_mode": "prompt",
                            "records": [
                                {
                                    "phase": "warm",
                                    "iteration": 1,
                                    "status": "success",
                                    "prompt_tokens": 27,
                                    "initial_prompt_tokens": 27,
                                    "conversation_prompt_tokens": 27,
                                    "ttft_ms": 225.0,
                                    "total_latency_ms": 25000.0,
                                }
                            ],
                        },
                        {
                            "run_id": "missing-run",
                            "started_at": "2026-04-02T00:00:00+00:00",
                            "model": "missing-model",
                            "prompt_text": "pythonでライブラリを使わずにRC4を実装して",
                            "benchmark_mode": "prompt",
                            "records": [
                                {
                                    "phase": "warm",
                                    "iteration": 1,
                                    "status": "success",
                                    "ttft_ms": 300.0,
                                    "total_latency_ms": 18000.0,
                                }
                            ],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            history = load_history_entries(history_path)

        record = history[1]["records"][0]
        self.assertEqual(record["prompt_tokens"], 27)
        self.assertEqual(record["initial_prompt_tokens"], 27)
        self.assertEqual(record["conversation_prompt_tokens"], 27)
        self.assertAlmostEqual(record["initial_prompt_tps"], 90.0)
        self.assertAlmostEqual(record["conversation_prompt_tps"], 1.5)

    def test_load_history_entries_backfills_record_tool_metrics_from_attempt_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "runs" / "history.json"
            attempt_log_path = Path(tmpdir) / "runs" / "logs" / "demo" / "attempts" / "warm-1.json"
            question_log_path = Path(tmpdir) / "runs" / "logs" / "demo" / "questions" / "warm-1-q1.json"
            question_log_path.parent.mkdir(parents=True, exist_ok=True)
            question_log_path.write_text(
                json.dumps(
                    {
                        "parsed_worker_result": {
                            "trace": {
                                "turns": [
                                    {
                                        "turn": 1,
                                        "tool_events": [
                                            {"tool_name": "mecha_ghidra.decompile_function"},
                                            {"tool_name": "run_python"},
                                        ],
                                    }
                                ]
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
            attempt_log_path.write_text(
                json.dumps(
                    {
                        "kind": "docker_attempt",
                        "question_logs": [
                            {
                                "question_id": "q1",
                                "log_path": "logs/demo/questions/warm-1-q1.json",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "run_id": "legacy-attempt-tools",
                            "started_at": "2026-04-01T00:00:00+00:00",
                            "model": "demo-model",
                            "records": [
                                {
                                    "phase": "warm",
                                    "iteration": 1,
                                    "status": "success",
                                    "log_path": "logs/demo/attempts/warm-1.json",
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            history = load_history_entries(history_path)

        record = history[0]["records"][0]
        self.assertEqual(record["tool_call_count"], 2)
        self.assertEqual(
            record["tool_name_counts"],
            {"mecha_ghidra.decompile_function": 1, "run_python": 1},
        )

    def test_update_history_preserves_log_paths_and_error_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "runs" / "history.json"
            run_data = {
                "run_id": "demo",
                "started_at": "2026-04-01T00:00:00+00:00",
                "model": "demo-model",
                "log_manifest_path": "logs/demo/manifest.json",
                "log_dir": "logs/demo",
                "records": [
                    {
                        "phase": "warm",
                        "iteration": 1,
                        "status": "error",
                        "error": "docker exited with 1",
                        "error_signature": "docker exited with 1",
                        "error_category": "docker",
                        "log_path": "logs/demo/attempts/warm-1.json",
                        "stderr_excerpt": "stderr",
                        "tool_call_count": 4,
                        "tool_name_counts": {"mecha_ghidra.list_functions": 3, "run_python": 1},
                    }
                ],
            }

            update_history(history_path, run_data)
            stored = json.loads(history_path.read_text(encoding="utf-8"))

        self.assertEqual(stored[0]["log_manifest_path"], "logs/demo/manifest.json")
        self.assertEqual(stored[0]["records"][0]["log_path"], "logs/demo/attempts/warm-1.json")
        self.assertEqual(stored[0]["records"][0]["error_category"], "docker")
        self.assertEqual(stored[0]["records"][0]["tool_call_count"], 4)
        self.assertEqual(
            stored[0]["records"][0]["tool_name_counts"],
            {"mecha_ghidra.list_functions": 3, "run_python": 1},
        )
