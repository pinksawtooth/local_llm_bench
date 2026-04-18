from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_llm_bench.config import OutputSettings
from local_llm_bench.run_logs import persist_run_logs


class RunLogsTests(unittest.TestCase):
    def test_persist_run_logs_writes_prompt_attempt_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = OutputSettings(
                history_json=root / "runs" / "history.json",
                latest_json=root / "runs" / "latest_run.json",
                report_html=root / "docs" / "index.html",
                run_logs_dir=root / "runs" / "logs",
            )
            run_data = {
                "run_id": "demo",
                "started_at": "2026-04-01T00:00:00+00:00",
                "ended_at": "2026-04-01T00:00:05+00:00",
                "model": "demo-model",
                "records": [
                    {
                        "phase": "warm",
                        "iteration": 1,
                        "started_at": "2026-04-01T00:00:00+00:00",
                        "status": "error",
                        "error": "URLError: refused",
                        "response_text": "",
                        "reasoning_text": "",
                    }
                ],
                "_log_bundle": {
                    "console_lines": ["[Run demo] model=demo-model", "  - warm #1 ..."],
                    "attempts": [
                        {
                            "record_index": 0,
                            "phase": "warm",
                            "iteration": 1,
                            "payload": {
                                "kind": "prompt_attempt",
                                "phase": "warm",
                                "iteration": 1,
                                "response": {"status": "error", "error": "URLError: refused"},
                            },
                        }
                    ],
                },
            }

            persisted = persist_run_logs(output, run_data)
            manifest_path = root / "runs" / "logs" / "demo" / "manifest.json"
            attempt_path = root / "runs" / "logs" / "demo" / "attempts" / "warm-1.json"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(attempt_path.exists())
            self.assertEqual(persisted["log_manifest_path"], "logs/demo/manifest.json")
            self.assertEqual(persisted["records"][0]["log_path"], "logs/demo/attempts/warm-1.json")
            self.assertEqual(persisted["records"][0]["error_category"], "api")
            self.assertEqual(persisted["records"][0]["tool_call_count"], 0)

    def test_persist_run_logs_writes_question_logs_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = OutputSettings(
                history_json=root / "runs" / "history.json",
                latest_json=root / "runs" / "latest_run.json",
                report_html=root / "docs" / "index.html",
                run_logs_dir=root / "runs" / "logs",
            )
            run_data = {
                "run_id": "docker-demo",
                "started_at": "2026-04-01T00:00:00+00:00",
                "ended_at": "2026-04-01T00:00:05+00:00",
                "model": "demo-model",
                "records": [
                    {
                        "phase": "warm",
                        "iteration": 1,
                        "started_at": "2026-04-01T00:00:00+00:00",
                        "status": "error",
                        "error": "[q1] worker returned malformed JSON",
                        "question_results": [
                            {
                                "question_id": "q1",
                                "status": "error",
                                "error": "worker returned malformed JSON",
                                "stderr_excerpt": "traceback line",
                            }
                        ],
                    }
                ],
                "_log_bundle": {
                    "console_lines": ["[Run docker-demo] model=demo-model"],
                    "attempts": [
                        {
                            "record_index": 0,
                            "phase": "warm",
                            "iteration": 1,
                            "payload": {"kind": "docker_attempt"},
                            "question_logs": [
                                {
                                    "question_index": 0,
                                    "question_id": "q1",
                                    "payload": {
                                        "kind": "docker_question",
                                        "stderr": "traceback line",
                                        "parsed_worker_result": {
                                            "status": "error",
                                            "error": "worker returned malformed JSON",
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
                                            },
                                        },
                                    },
                                }
                            ],
                        }
                    ],
                },
            }

            persisted = persist_run_logs(output, run_data)
            question_path = root / "runs" / "logs" / "docker-demo" / "questions" / "warm-1-q1.json"
            question_payload = json.loads(question_path.read_text(encoding="utf-8"))

            self.assertTrue(question_path.exists())
            self.assertEqual(persisted["records"][0]["question_results"][0]["log_path"], "logs/docker-demo/questions/warm-1-q1.json")
            self.assertEqual(persisted["records"][0]["question_results"][0]["error_category"], "worker")
            self.assertEqual(persisted["records"][0]["question_results"][0]["tool_call_count"], 3)
            self.assertEqual(
                persisted["records"][0]["question_results"][0]["tool_name_counts"],
                {"mecha_ghidra.list_functions": 2, "run_python": 1},
            )
            self.assertEqual(persisted["records"][0]["tool_call_count"], 3)
            self.assertEqual(question_payload["parsed_worker_result"]["trace"]["turns"][0]["turn"], 1)
