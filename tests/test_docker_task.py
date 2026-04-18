from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

from local_llm_bench.config import BenchmarkConfig, OutputSettings, RequestSettings, RunSettings
from local_llm_bench.docker_task.runner import _REPO_ROOT, _run_question_in_docker, run_docker_task_benchmark
from local_llm_bench.docker_task.scorer import score_answer
from local_llm_bench.docker_task.spec import load_spec


class DockerTaskTests(unittest.TestCase):
    def test_load_spec_resolves_answer_key_and_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data").mkdir()
            (root / "data" / "d-compile").write_bytes(b"\x7fELFdemo")
            spec_dir = root / "benchmarks" / "d_compile"
            spec_dir.mkdir(parents=True)
            spec_path = spec_dir / "spec.yaml"
            spec_path.write_text(
                textwrap.dedent(
                    """
                    id: d-compile
                    title: d-compile
                    questions:
                      - id: q1
                        prompt: question
                        answer_type: exact
                        binary_path: data/d-compile
                    """
                ),
                encoding="utf-8",
            )
            (spec_dir / "spec.answers.yaml").write_text(
                "answers:\n  q1: flag{demo}\n",
                encoding="utf-8",
            )

            spec = load_spec(spec_path)

        self.assertEqual(spec.id, "d-compile")
        self.assertEqual(spec.questions[0].gold_answer, "flag{demo}")
        self.assertTrue(str(spec.questions[0].binary_path).endswith("data/d-compile"))

    def test_score_answer_exact_is_case_insensitive_trimmed(self) -> None:
        result = score_answer("exact", "  FLAG{demo}  ", "flag{demo}")
        self.assertTrue(result.correct)
        self.assertEqual(result.score, 1.0)

    def test_run_docker_task_benchmark_keeps_wrong_answer_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            binary_path = root / "d-compile"
            binary_path.write_bytes(b"\x7fELFdemo")
            spec_path = root / "spec.yaml"
            spec_path.write_text(
                textwrap.dedent(
                    """
                    id: d-compile
                    title: d-compile
                    questions:
                      - id: q1
                        prompt: question
                        answer_type: exact
                        binary_path: d-compile
                    """
                ),
                encoding="utf-8",
            )
            answers_path = root / "spec.answers.yaml"
            answers_path.write_text("answers:\n  q1: flag{correct}\n", encoding="utf-8")
            config = BenchmarkConfig(
                api_base="http://localhost:1234/v1",
                models=["bench-model"],
                prompt_text="",
                request=RequestSettings(temperature=0.0, max_tokens=128),
                runs=RunSettings(cold_runs=1, warm_runs=0, timeout_sec=10.0, cooldown_sec=0.0),
                output=OutputSettings(
                    history_json=root / "runs" / "history.json",
                    latest_json=root / "runs" / "latest_run.json",
                    report_html=root / "docs" / "index.html",
                    run_logs_dir=root / "runs" / "logs",
                ),
                mode="docker_task",
                benchmark_spec_path=spec_path,
                benchmark_answer_key_path=answers_path,
                docker_image="local-llm-bench:bench",
            )

            def fake_executor(*args, **kwargs):
                return subprocess.CompletedProcess(
                    args[0],
                    0,
                    stdout=json.dumps(
                        {
                            "status": "success",
                            "predicted_answer": "flag{wrong}",
                            "response_text": "FINAL_ANSWER: flag{wrong}",
                            "reasoning_text": "trace",
                            "finish_reason": "stop",
                            "ttft_ms": 100.0,
                            "total_latency_ms": 500.0,
                            "completion_window_ms": 400.0,
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                            "decode_tps": 50.0,
                            "end_to_end_tps": 40.0,
                            "approx_prompt_tps": 100.0,
                        },
                        ensure_ascii=False,
                    ),
                    stderr="",
                )

            from unittest.mock import patch

            with patch("local_llm_bench.docker_task.runner._should_stage_ghidra_source", return_value=False):
                run_data = run_docker_task_benchmark(
                    config,
                    docker_executor=fake_executor,
                    now_fn=lambda: 1.0,
                    sleep_fn=lambda _: None,
                )

        record = run_data["records"][0]
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["benchmark_score"], 0.0)
        self.assertFalse(record["benchmark_correct"])
        self.assertEqual(record["benchmark_incorrect_count"], 1)
        self.assertEqual(record["benchmark_error_count"], 0)

    def test_run_docker_task_benchmark_marks_timeout_as_error_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            binary_path = root / "d-compile"
            binary_path.write_bytes(b"\x7fELFdemo")
            spec_path = root / "spec.yaml"
            spec_path.write_text(
                textwrap.dedent(
                    """
                    id: d-compile
                    title: d-compile
                    questions:
                      - id: q1
                        prompt: question
                        answer_type: exact
                        binary_path: d-compile
                    """
                ),
                encoding="utf-8",
            )
            answers_path = root / "spec.answers.yaml"
            answers_path.write_text("answers:\n  q1: flag{correct}\n", encoding="utf-8")
            config = BenchmarkConfig(
                api_base="http://localhost:1234/v1",
                models=["bench-model"],
                prompt_text="",
                request=RequestSettings(temperature=0.0, max_tokens=128),
                runs=RunSettings(cold_runs=0, warm_runs=1, timeout_sec=10.0, cooldown_sec=0.0),
                output=OutputSettings(
                    history_json=root / "runs" / "history.json",
                    latest_json=root / "runs" / "latest_run.json",
                    report_html=root / "docs" / "index.html",
                    run_logs_dir=root / "runs" / "logs",
                ),
                mode="docker_task",
                benchmark_spec_path=spec_path,
                benchmark_answer_key_path=answers_path,
                docker_image="local-llm-bench:bench",
            )

            def fake_executor(*args, **kwargs):
                raise subprocess.TimeoutExpired(kwargs.get("args", args[0]), timeout=10.0)

            from unittest.mock import patch

            with patch("local_llm_bench.docker_task.runner._should_stage_ghidra_source", return_value=False):
                run_data = run_docker_task_benchmark(
                    config,
                    docker_executor=fake_executor,
                    now_fn=lambda: 1.0,
                    sleep_fn=lambda _: None,
                )

        record = run_data["records"][0]
        self.assertEqual(record["status"], "timeout")
        self.assertEqual(record["benchmark_error_count"], 1)
        self.assertEqual(record["benchmark_incorrect_count"], 0)

    def test_run_docker_task_benchmark_mounts_repo_root_into_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            binary_path = root / "d-compile"
            binary_path.write_bytes(b"\x7fELFdemo")
            spec_path = root / "spec.yaml"
            spec_path.write_text(
                textwrap.dedent(
                    """
                    id: d-compile
                    title: d-compile
                    questions:
                      - id: q1
                        prompt: question
                        answer_type: exact
                        binary_path: d-compile
                    """
                ),
                encoding="utf-8",
            )
            answers_path = root / "spec.answers.yaml"
            answers_path.write_text("answers:\n  q1: flag{correct}\n", encoding="utf-8")
            config = BenchmarkConfig(
                api_base="http://localhost:1234/v1",
                models=["bench-model"],
                prompt_text="",
                request=RequestSettings(temperature=0.0, max_tokens=128),
                runs=RunSettings(cold_runs=1, warm_runs=0, timeout_sec=10.0, cooldown_sec=0.0),
                output=OutputSettings(
                    history_json=root / "runs" / "history.json",
                    latest_json=root / "runs" / "latest_run.json",
                    report_html=root / "docs" / "index.html",
                    run_logs_dir=root / "runs" / "logs",
                ),
                mode="docker_task",
                benchmark_spec_path=spec_path,
                benchmark_answer_key_path=answers_path,
                docker_image="local-llm-bench:bench",
            )

            seen_cmd: list[str] = []

            def fake_executor(*args, **kwargs):
                nonlocal seen_cmd
                seen_cmd = list(args[0])
                return subprocess.CompletedProcess(
                    args[0],
                    0,
                    stdout=json.dumps(
                        {
                            "status": "success",
                            "predicted_answer": "flag{correct}",
                            "response_text": "FINAL_ANSWER: flag{correct}",
                            "reasoning_text": "",
                            "finish_reason": "stop",
                            "ttft_ms": 100.0,
                            "total_latency_ms": 500.0,
                            "completion_window_ms": 400.0,
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                            "decode_tps": 50.0,
                            "end_to_end_tps": 40.0,
                            "approx_prompt_tps": 100.0,
                        },
                        ensure_ascii=False,
                    ),
                    stderr="",
                )

            from unittest.mock import patch

            with patch("local_llm_bench.docker_task.runner._should_stage_ghidra_source", return_value=False):
                run_docker_task_benchmark(
                    config,
                    docker_executor=fake_executor,
                    now_fn=lambda: 1.0,
                    sleep_fn=lambda _: None,
                )

        self.assertIn(f"{_REPO_ROOT}:/opt/local_llm_bench:ro", seen_cmd)

    def test_run_docker_task_benchmark_reports_platform_mismatch_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            binary_path = root / "d-compile"
            binary_path.write_bytes(b"\x7fELFdemo")
            spec_path = root / "spec.yaml"
            spec_path.write_text(
                textwrap.dedent(
                    """
                    id: d-compile
                    title: d-compile
                    questions:
                      - id: q1
                        prompt: question
                        answer_type: exact
                        binary_path: d-compile
                    """
                ),
                encoding="utf-8",
            )
            answers_path = root / "spec.answers.yaml"
            answers_path.write_text("answers:\n  q1: flag{correct}\n", encoding="utf-8")
            config = BenchmarkConfig(
                api_base="http://localhost:1234/v1",
                models=["bench-model"],
                prompt_text="",
                request=RequestSettings(temperature=0.0, max_tokens=128),
                runs=RunSettings(cold_runs=1, warm_runs=0, timeout_sec=10.0, cooldown_sec=0.0),
                output=OutputSettings(
                    history_json=root / "runs" / "history.json",
                    latest_json=root / "runs" / "latest_run.json",
                    report_html=root / "docs" / "index.html",
                    run_logs_dir=root / "runs" / "logs",
                ),
                mode="docker_task",
                benchmark_spec_path=spec_path,
                benchmark_answer_key_path=answers_path,
                docker_image="local-llm-bench:bench",
                docker_platform="linux/amd64",
            )

            mismatch_error = (
                "docker image 'local-llm-bench:bench' はローカルに linux/arm64 として存在しますが、"
                "設定は linux/amd64 を要求しています。"
            )

            from unittest.mock import patch

            with patch("local_llm_bench.docker_task.runner._should_stage_ghidra_source", return_value=False), patch(
                "local_llm_bench.docker_task.runner._docker_platform_mismatch_error",
                return_value=mismatch_error,
            ):
                run_data = run_docker_task_benchmark(
                    config,
                    docker_executor=subprocess.run,
                    now_fn=lambda: 1.0,
                    sleep_fn=lambda _: None,
                )

        record = run_data["records"][0]
        self.assertEqual(record["status"], "error")
        self.assertIn("linux/arm64", record["error"])
        self.assertIn("linux/amd64", record["error"])

    def test_run_question_in_docker_keeps_auth_secret_out_of_request_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = BenchmarkConfig(
                api_base="http://127.0.0.1:8888/v1",
                models=["bench-model"],
                prompt_text="",
                request=RequestSettings(temperature=0.0, max_tokens=128),
                runs=RunSettings(cold_runs=1, warm_runs=0, timeout_sec=10.0, cooldown_sec=0.0),
                output=OutputSettings(
                    history_json=root / "runs" / "history.json",
                    latest_json=root / "runs" / "latest_run.json",
                    report_html=root / "docs" / "index.html",
                    run_logs_dir=root / "runs" / "logs",
                ),
                provider="unsloth_studio",
                docker_image="local-llm-bench:bench",
                docker_api_base="http://host.docker.internal:8888/v1",
            )
            question = types.SimpleNamespace(
                id="q1",
                prompt="question",
                answer_type="exact",
                binary_path=None,
                binary_ref=None,
            )
            seen_cmd: list[str] = []
            seen_env: dict[str, str] = {}

            def fake_executor(*args, **kwargs):
                nonlocal seen_cmd, seen_env
                seen_cmd = list(args[0])
                seen_env = dict(kwargs.get("env") or {})
                return subprocess.CompletedProcess(
                    args[0],
                    0,
                    stdout=json.dumps(
                        {
                            "status": "success",
                            "predicted_answer": "flag{correct}",
                            "response_text": "FINAL_ANSWER: flag{correct}",
                            "reasoning_text": "",
                            "finish_reason": "stop",
                            "ttft_ms": 100.0,
                            "total_latency_ms": 500.0,
                            "completion_window_ms": 400.0,
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                            "decode_tps": 50.0,
                            "end_to_end_tps": 40.0,
                            "approx_prompt_tps": 100.0,
                        },
                        ensure_ascii=False,
                    ),
                    stderr="",
                )

            from unittest.mock import patch

            with patch("local_llm_bench.docker_task.runner._should_stage_ghidra_source", return_value=False):
                result = _run_question_in_docker(
                    config=config,
                    selected_model="bench-model",
                    question=question,
                    docker_executor=fake_executor,
                    docker_env={"UNSLOTH_STUDIO_BEARER_TOKEN": "super-secret"},
                )

        self.assertEqual(result["status"], "success")
        self.assertIn("UNSLOTH_STUDIO_BEARER_TOKEN", seen_cmd)
        self.assertNotIn("super-secret", " ".join(seen_cmd))
        self.assertEqual(seen_env["UNSLOTH_STUDIO_BEARER_TOKEN"], "super-secret")
        request_log = result["_question_log"]["request"]
        self.assertNotIn("super-secret", json.dumps(request_log, ensure_ascii=False))
        self.assertNotIn("super-secret", json.dumps(result["_question_log"]["docker_command"], ensure_ascii=False))
