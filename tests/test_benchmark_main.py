from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import benchmark
from bench_fakes import make_config, response, runtime_for
from local_llm_bench.persistence import server_lock


class BenchmarkMainTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        # Never contend with a real benchmark's endpoint lock during tests.
        self.stack.enter_context(patch("tempfile.tempdir", str(self.root)))
        self.stack.enter_context(patch.dict("os.environ", {"TMPDIR": str(self.root)}))
        self.config = make_config(self.root)
        self.runtime = runtime_for(self.root)
        self.stack.enter_context(patch("benchmark.load_config", return_value=self.config))
        self.stack.enter_context(patch("benchmark.build_provider_runtime", return_value=self.runtime))
        self.stack.enter_context(patch("local_llm_bench.diagnostics.host_snapshot", return_value={"cpu_count": 8}))
        self.stack.enter_context(patch("local_llm_bench.execution.host_snapshot", return_value={"cpu_count": 8, "swap_used_bytes": 0}))
        self.report = self.stack.enter_context(patch("benchmark.ensure_report_html", return_value=False))
        self.stack.enter_context(patch("builtins.print"))

    def history(self):
        return json.loads(self.config.output.history_json.read_text())

    def test_models_are_loaded_and_unloaded_individually(self):
        self.config.models = ["model-a", "model-b"]
        self.assertEqual(benchmark.main([]), 0)
        self.assertEqual([c.args[0] for c in self.runtime.unload_model.call_args_list], ["model-a", "model-a-loaded", "model-b", "model-b-loaded"])
        history = self.history()
        self.assertEqual(len(history), 2)
        self.assertTrue(all(run["status"] == "completed" for run in history))
        self.assertEqual(history[0]["model_info"]["runtime"]["version"], "fixture-1")
        self.assertEqual(len(history[0]["records"]), 3)
        self.report.assert_called_once()

    def test_keep_loaded_still_resets_before_cold_measurement(self):
        benchmark.main(["--keep-loaded"])
        self.runtime.unload_model.assert_called_once_with("model-a")

    def test_parallelism_sweep_records_effective_load_config(self):
        self.config.lmstudio_load.parallelism_sweep = [2, 3, 4]
        benchmark.main([])
        self.assertEqual([r["lmstudio_parallelism"] for r in self.history()], [2, 3, 4])
        self.assertEqual([r["conditions"]["load"]["effective"]["parallel"] for r in self.history()], [2, 3, 4])
        self.assertEqual(len({r["comparison"]["group_id"] for r in self.history()}), 3)

    def test_preflight_auth_error_never_loads_or_unloads(self):
        self.runtime.inspect_model.side_effect = RuntimeError("401 invalid token")
        with self.assertRaisesRegex(RuntimeError, "invalid token"):
            benchmark.main([])
        self.runtime.prepare_model.assert_not_called()
        self.runtime.unload_model.assert_not_called()
        self.assertEqual(self.history()[0]["status"], "failed")

    def test_check_and_concurrent_run_do_not_mutate_server(self):
        benchmark.main(["--check"])
        self.runtime.prepare_model.assert_not_called()
        self.runtime.unload_model.assert_not_called()
        self.assertFalse(self.config.output.history_json.exists())
        with server_lock("http://127.0.0.1:1234/api/v1"):
            with self.assertRaisesRegex(RuntimeError, "使用中"):
                benchmark.main([])
        self.runtime.prepare_model.assert_not_called()

    def test_interrupted_run_resumes_same_id_preserving_completed_answer(self):
        def omit_effective_sampling(info):
            info["reported_inference"].pop("sampling", None)
            return info
        self.runtime.measurement_metadata.side_effect = omit_effective_sampling
        client = self.runtime.chat_client.return_value
        client.side_effect = [response("original"), KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            benchmark.main([])
        run = self.history()[0]
        run_id = run["run_id"]
        self.assertEqual(run["status"], "interrupted")
        checkpoint = self.config.output.run_logs_dir / run_id / "checkpoint.json"
        original = json.loads(checkpoint.read_text())["units"]['["cold", 1, "prompt"]']
        prior_bytes = checkpoint.read_bytes()
        client.side_effect = lambda **kwargs: response("resumed")
        with self.assertRaisesRegex(ValueError, "allow-unverified-resume"):
            benchmark.main(["--resume-run-id", run_id])
        self.assertEqual(checkpoint.read_bytes(), prior_bytes)
        client.reset_mock()
        benchmark.main(["--resume-run-id", run_id, "--allow-unverified-resume"])
        self.assertEqual(client.call_count, 3)  # one excluded primer, two remaining attempts
        final = self.history()
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["status"], "completed")
        self.assertEqual(json.loads(checkpoint.read_text())["units"]['["cold", 1, "prompt"]'], original)
        self.assertTrue(final[0]["comparison"]["recovered"])
        self.assertEqual(len(list((checkpoint.parent / "revisions").glob("*.json"))), 1)

    def test_retry_errors_keeps_successes_and_original_revision(self):
        client = self.runtime.chat_client.return_value
        client.side_effect = [response("one"), TimeoutError("timeout"), response("three")]
        benchmark.main([])
        run_id = self.history()[0]["run_id"]
        checkpoint = self.config.output.run_logs_dir / run_id / "checkpoint.json"
        original = json.loads(checkpoint.read_text())
        client.reset_mock()
        client.side_effect = lambda **kwargs: response("repaired")
        benchmark.main(["--retry-errors-run-id", run_id, "--allow-unverified-resume"])
        self.assertEqual(client.call_count, 2)  # excluded primer + failed attempt only
        final = json.loads(checkpoint.read_text())
        self.assertEqual(len(final["unit_timings"]), 4)
        self.assertEqual(self.history()[0]["timings"]["inference_wall_sec"], sum(item["wall_sec"] for item in final["unit_timings"]))
        for key, unit in original["units"].items():
            if unit["result"]["status"] == "success":
                self.assertEqual(final["units"][key], unit)
        self.assertTrue(all(unit["result"]["status"] == "success" for unit in final["units"].values()))

    def test_changed_contract_is_rejected_without_modifying_checkpoint(self):
        self.runtime.chat_client.return_value.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            benchmark.main([])
        run_id = self.history()[0]["run_id"]
        checkpoint = self.config.output.run_logs_dir / run_id / "checkpoint.json"
        before = checkpoint.read_bytes()
        self.config.request.temperature = 0.5
        self.runtime.prepare_model.reset_mock()
        with self.assertRaisesRegex(ValueError, "request.temperature"):
            benchmark.main(["--resume-run-id", run_id])
        self.assertEqual(checkpoint.read_bytes(), before)
        self.runtime.prepare_model.assert_not_called()

    def test_interrupted_retry_resumes_remaining_retry_targets(self):
        client = self.runtime.chat_client.return_value
        client.side_effect = [response("original"), TimeoutError("one"), TimeoutError("two")]
        benchmark.main([])
        run_id = self.history()[0]["run_id"]
        client.side_effect = [response("primer"), response("repaired one"), KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            benchmark.main(["--retry-errors-run-id", run_id])
        checkpoint = self.config.output.run_logs_dir / run_id / "checkpoint.json"
        interrupted = json.loads(checkpoint.read_text())
        self.assertEqual(interrupted["pending_retries"], ['["warm", 2, "prompt"]'])
        preserved = interrupted["units"]['["warm", 1, "prompt"]']
        client.reset_mock()
        client.side_effect = lambda **kwargs: response("remaining")
        benchmark.main(["--resume-run-id", run_id])
        self.assertEqual(client.call_count, 2)
        finished = json.loads(checkpoint.read_text())
        self.assertEqual(finished["units"]['["warm", 1, "prompt"]'], preserved)
        self.assertEqual(finished["pending_retries"], [])

    def test_docker_dispatch_pins_verified_image(self):
        from local_llm_bench.inspect_harness import INSPECT_VERSION, PROFILE_VERSION, dependency_lock_hash
        labels = {"local-llm-bench.inspect.version": INSPECT_VERSION,
                  "local-llm-bench.worker.profile": PROFILE_VERSION,
                  "local-llm-bench.dependencies.sha256": dependency_lock_hash()}
        self.config.mode = "docker_task"
        self.config.docker_image = "mutable:tag"
        self.config.benchmark_spec_path = self.root / "spec.yaml"
        self.config.benchmark_spec_path.write_text("id: fixture\nquestions:\n  - {id: q1, prompt: question, answer_type: exact}\n")
        (self.root / "spec.answers.yaml").write_text("answers:\n  q1: ok\n")
        with patch("local_llm_bench.docker_task.runner._docker_binary", return_value="docker"), patch("local_llm_bench.diagnostics._read_command", side_effect=["28", "", json.dumps([{"Id": "sha256:fixed", "Os": "linux", "Architecture": "arm64", "Config": {"Labels": labels}}])]), patch("local_llm_bench.docker_task.runner._run_question_in_docker", return_value={"status": "success", "predicted_answer": "ok"}) as worker:
            benchmark.main([])
        self.assertEqual(worker.call_count, 3)
        self.assertTrue(all(call.kwargs["config"].docker_image == "sha256:fixed" for call in worker.call_args_list))
        self.assertEqual(self.history()[0]["conditions"]["docker"]["image_id"], "sha256:fixed")

    def test_docker_timeouts_complete_all_trials_and_models_without_extra_primer(self):
        self.config.models = ["model-a", "model-b", "model-c", "model-d"]
        self.config.mode = "docker_task"
        self.config.runs.cold_runs, self.config.runs.warm_runs = 1, 3
        self.config.benchmark_question_timeout_sec = 3600
        self.config.docker_image = "fixture"
        self.config.benchmark_spec_path = self.root / "spec.yaml"
        self.config.benchmark_spec_path.write_text("id: fixture\nquestions:\n  - {id: q1, prompt: question, answer_type: exact}\n")
        (self.root / "spec.answers.yaml").write_text("answers:\n  q1: ok\n")
        preflight = {"host": {"cpu_count": 8}, "docker": {"image_id": "sha256:fixed"}}
        with patch("benchmark.run_preflight", return_value=preflight), patch(
            "local_llm_bench.docker_task.runner._run_question_in_docker",
            return_value={"status": "timeout", "error": "question time budget exhausted"},
        ) as worker:
            self.assertEqual(benchmark.main([]), 0)
        self.assertEqual(worker.call_count, 16)
        self.assertEqual([call.args[0] for call in self.runtime.prepare_model.call_args_list], self.config.models)
        history = self.history()
        self.assertEqual([run["model"] for run in history], self.config.models)
        for run in history:
            self.assertEqual(run["status"], "completed")
            self.assertEqual([record["status"] for record in run["records"]], ["timeout"] * 4)
            self.assertFalse(any(item["event"] == "warmup" for item in run["lifecycle"]))
            checkpoint = json.loads((self.config.output.run_logs_dir / run["run_id"] / "checkpoint.json").read_text())
            self.assertEqual(len(checkpoint["units"]), 4)
            stages = [unit["result"]["measurement"]["inference_stage"] for unit in checkpoint["units"].values()]
            self.assertEqual(stages, ["first_after_load"] + ["unknown_after_error"] * 3)
        self.assertEqual(json.loads(self.config.output.latest_json.read_text())["model"], "model-d")
