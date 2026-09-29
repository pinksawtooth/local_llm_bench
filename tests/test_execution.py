from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bench_fakes import make_config, runtime_for
from local_llm_bench.conditions import capture_conditions, comparison_metadata, file_hash, model_artifact
from local_llm_bench.diagnostics import run_preflight
from local_llm_bench.docker_task.runner import run_docker_task_benchmark
from local_llm_bench.execution import RunExecution
from local_llm_bench.persistence import atomic_write_json, server_lock
from local_llm_bench.runner import run_benchmark
from local_llm_bench.stats import compute_history_summary


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Child lock probes inherit this private directory as well.
        locks = patch("tempfile.tempdir", str(self.root))
        locks.start()
        self.addCleanup(locks.stop)
        environment = patch.dict("os.environ", {"TMPDIR": str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)
        self.config = make_config(self.root)
        self.runtime = runtime_for(self.root)
        self.preflight = {"host": {"cpu_count": 8, "memory_total_bytes": 1024}, "docker": None}
        snapshot = patch("local_llm_bench.execution.host_snapshot", return_value={"memory_total_bytes": 1024, "swap_used_bytes": 0})
        snapshot.start()
        self.addCleanup(snapshot.stop)

    def execute(self, execution):
        execution.start(self.preflight)
        result = run_benchmark(self.config, model=execution.api_model, client=self.runtime.chat_client(), execution=execution)
        execution.close()
        execution.finish("completed")
        return execution.enrich(result)

    def test_each_cold_reloads_and_warm_only_has_excluded_primer(self):
        self.config.runs.cold_runs = 2
        self.config.runs.warm_runs = 1
        result = self.execute(RunExecution(self.config, self.runtime, "model-a"))
        self.assertEqual(self.runtime.prepare_model.call_count, 2)
        self.assertEqual(self.runtime.chat_client().call_count, 3)
        self.assertEqual([r["measurement"]["inference_stage"] for r in result["records"]], ["first_after_load", "first_after_load", "repeat"])
        self.assertEqual(result["conditions"]["cache"]["state"], "unknown")
        self.assertEqual(len([e for e in result["lifecycle"] if e["event"] == "load"]), 2)
        self.runtime.chat_client().reset_mock()
        self.config.runs.cold_runs, self.config.runs.warm_runs = 0, 2
        result = self.execute(RunExecution(self.config, self.runtime, "model-a"))
        self.assertEqual(self.runtime.chat_client().call_count, 3)
        self.assertTrue(all(r["measurement"]["inference_stage"] == "repeat" for r in result["records"]))
        self.assertTrue(next(e for e in result["lifecycle"] if e["event"] == "warmup")["excluded_from_statistics"])

    def test_fsynced_question_event_survives_checkpoint_failure(self):
        ex = RunExecution(self.config, self.runtime, "model-a")
        ex.start(self.preflight)
        unit = {"status": "success", "response_text": "durable"}
        with patch.object(ex, "_flush", side_effect=OSError("disk interrupted")):
            with self.assertRaises(OSError):
                ex.save_unit("cold", 1, "prompt", unit, {"response": "original"})
        ex.close()
        # Simulate a torn final event append after the durable result.
        with (ex.directory / "events.jsonl").open("ab") as stream:
            stream.write(b'{"event":')
        resumed = RunExecution(self.config, self.runtime, "model-a", resume_id=ex.run_id, allow_unverified=True)
        self.assertEqual(resumed.cached("cold", 1, "prompt")["result"]["response_text"], "durable")
        resumed.start(self.preflight)
        resumed.finish("interrupted")
        resumed.close()
        for line in (ex.directory / "events.jsonl").read_text().splitlines():
            json.loads(line)
        self.assertIsNone(resumed.enrich({"records": []})["timings"]["total_wall_sec"])

    def test_changed_model_bytes_refuse_resume_and_keep_checkpoint(self):
        ex = RunExecution(self.config, self.runtime, "model-a")
        ex.start(self.preflight)
        ex.finish("interrupted")
        ex.close()
        before = ex.checkpoint.read_bytes()
        (self.root / "model-Q4_K_M.gguf").write_bytes(b"different weights")
        resumed = RunExecution(self.config, self.runtime, "model-a", resume_id=ex.run_id, allow_unverified=True)
        try:
            with self.assertRaisesRegex(ValueError, "model.artifact"):
                resumed.start(self.preflight)
        finally:
            resumed.close()
        self.assertEqual(before, ex.checkpoint.read_bytes())

    def test_retry_incomplete_run_is_rejected_before_model_load(self):
        ex = RunExecution(self.config, self.runtime, "model-a")
        ex.start(self.preflight)
        ex.save_unit("cold", 1, "prompt", {"status": "error"}, {})
        ex.finish("interrupted")
        ex.close()
        self.runtime.prepare_model.reset_mock()
        with self.assertRaisesRegex(ValueError, "先に --resume-run-id"):
            RunExecution(self.config, self.runtime, "model-a", retry_id=ex.run_id)
        self.runtime.prepare_model.assert_not_called()

    def test_docker_question_resume_and_selected_retry_preserve_other_questions(self):
        self.config.mode = "docker_task"
        self.config.runs.cold_runs, self.config.runs.warm_runs = 1, 0
        self.config.docker_image = "fixture"
        self.config.benchmark_spec_path = self.root / "spec.yaml"
        self.config.benchmark_spec_path.write_text("id: fixture\nquestions:\n  - {id: q1, prompt: first, answer_type: exact, gold_answer: ok}\n  - {id: q2, prompt: second, answer_type: exact, gold_answer: ok}\n")
        def result(question, status="success"):
            return {"status": status, "predicted_answer": "ok", "response_text": question, "_question_log": {"original": question}}
        (self.root / "spec.answers.yaml").write_text("answers:\n  q1: ok\n  q2: ok\n")
        ex = RunExecution(self.config, self.runtime, "model-a")
        ex.start(self.preflight)
        with patch("local_llm_bench.docker_task.runner._run_question_in_docker", side_effect=[result("q1"), KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                run_docker_task_benchmark(self.config, model=ex.api_model, execution=ex)
        ex.finish("interrupted")
        ex.close()
        original = copy.deepcopy(ex.cached("cold", 1, "q1"))
        resumed = RunExecution(self.config, self.runtime, "model-a", resume_id=ex.run_id, allow_unverified=True)
        resumed.start(self.preflight)
        with patch("local_llm_bench.docker_task.runner._run_question_in_docker", return_value=result("q2", "timeout")) as worker:
            data = run_docker_task_benchmark(self.config, model=resumed.api_model, execution=resumed)
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(worker.call_args.kwargs["question"].id, "q2")
        resumed.finish("completed")
        resumed.close()
        self.assertEqual(len(data["records"][0]["question_results"]), 2)
        retried = RunExecution(self.config, self.runtime, "model-a", retry_id=ex.run_id, questions=["q2"], allow_unverified=True)
        retried.start(self.preflight)
        with patch("local_llm_bench.docker_task.runner._run_question_in_docker", return_value=result("q2")) as worker:
            run_docker_task_benchmark(self.config, model=retried.api_model, execution=retried)
            self.assertEqual(worker.call_count, 1)
        retried.finish("completed")
        retried.close()
        self.assertEqual(retried.cached("cold", 1, "q1"), original)
        self.assertEqual(retried.cached("cold", 1, "q2")["result"]["status"], "success")

    def test_hash_tracks_all_shards_and_detects_changes(self):
        first = self.root / "weights-00001-of-00002.gguf"
        second = self.root / "weights-00002-of-00002.gguf"
        first.write_bytes(b"one")
        with self.assertRaisesRegex(RuntimeError, "分割ファイル"):
            model_artifact({"path": str(first)}, self.root / "cache")
        second.write_bytes(b"two")
        initial = model_artifact({"path": str(first)}, self.root / "cache")
        second.write_bytes(b"new")
        updated = model_artifact({"path": str(first)}, self.root / "cache")
        self.assertNotEqual(initial["sha256"], updated["sha256"])
        self.assertEqual(initial["files"][0]["sha256"], file_hash(first))
        self.assertEqual(model_artifact({"path": "missing"}, self.root)["status"], "unavailable")

    def test_summary_separates_conditions_unknown_legacy_and_incomplete(self):
        ex = RunExecution(self.config, self.runtime, "model-a")
        ex.start(self.preflight)
        conditions = ex.state["conditions"]
        ex.finish("completed")
        ex.close()
        runs = []
        for run_id in ("one", "two"):
            runs.append({"run_id": run_id, "status": "completed", "model": "model-a", "conditions": conditions, "comparison": comparison_metadata(conditions, run_id), "records": [{"status": "success", "phase": "warm", "total_latency_ms": 10}]})
        self.assertNotEqual(runs[0]["comparison"]["group_id"], runs[1]["comparison"]["group_id"])
        runs += [{"run_id": "old1", "model": "model-a", "records": runs[0]["records"]}, {"run_id": "old2", "model": "model-a", "records": runs[0]["records"]}, {"run_id": "partial", "status": "interrupted", "model": "model-a", "records": runs[0]["records"]}]
        summary = compute_history_summary(runs)
        self.assertEqual(summary["total_models"], 4)
        self.assertEqual(summary["total_samples"], 4)

    def test_docker_preflight_requires_local_matching_image(self):
        from local_llm_bench.inspect_harness import INSPECT_VERSION, PROFILE_VERSION, dependency_lock_hash
        labels = {"local-llm-bench.inspect.version": INSPECT_VERSION,
                  "local-llm-bench.worker.profile": PROFILE_VERSION,
                  "local-llm-bench.dependencies.sha256": dependency_lock_hash()}
        self.config.mode = "docker_task"
        self.config.docker_image = "fixture"
        self.config.docker_platform = "linux/arm64"
        with patch("local_llm_bench.docker_task.spec.load_spec"), patch("local_llm_bench.docker_task.runner._docker_binary", return_value="docker"), patch("local_llm_bench.diagnostics.host_snapshot", return_value={}), patch("local_llm_bench.diagnostics._read_command", side_effect=["28.0", "", json.dumps([{"Id": "sha256:fixture", "Os": "linux", "Architecture": "arm64", "Config": {"Labels": labels}}])]):
            self.assertEqual(run_preflight(self.config, self.runtime, "model-a")["docker"]["image_id"], "sha256:fixture")
        self.runtime.prepare_model.assert_not_called()
        with patch("local_llm_bench.docker_task.spec.load_spec"), patch("local_llm_bench.docker_task.runner._docker_binary", return_value="docker"), patch("local_llm_bench.diagnostics.host_snapshot", return_value={}), patch("local_llm_bench.diagnostics._read_command", side_effect=["28.0", "", None]):
            with self.assertRaisesRegex(RuntimeError, "ローカルにありません") as error:
                run_preflight(self.config, self.runtime, "model-a")
        import shlex
        command = shlex.split(str(error.exception).splitlines()[-1])
        self.assertEqual(Path(command[0]).name, "build_bench_image.sh")
        self.assertTrue(Path(command[0]).is_file())
        self.assertEqual(command[1:], ["--platform", "linux/arm64", "--tag", "fixture"])
        self.runtime.prepare_model.assert_not_called()

    def test_server_lock_is_shared_across_processes_and_aliases(self):
        script = "from local_llm_bench.persistence import server_lock\nwith server_lock('http://127.0.0.1:1234/api/v1'):\n print('acquired')\n"
        with server_lock(self.config.api_base):
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("使用中", result.stderr)
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)

    def test_atomic_json_failure_preserves_previous_file(self):
        path = self.root / "atomic.json"
        atomic_write_json(path, {"original": 1})
        with patch("local_llm_bench.persistence.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                atomic_write_json(path, {"replacement": 2})
        self.assertEqual(json.loads(path.read_text()), {"original": 1})

    def test_sweep_run_contract_can_be_resumed_with_one_parallelism(self):
        self.config.lmstudio_load.parallelism_sweep = [2, 3, 4]
        ex = RunExecution(self.config, self.runtime, "model-a", 3)
        ex.start(self.preflight)
        ex.finish("interrupted")
        ex.close()
        self.config.lmstudio_load.parallelism_sweep = []
        self.config.lmstudio_load.parallelism = 3
        resumed = RunExecution(self.config, self.runtime, "model-a", 3, resume_id=ex.run_id)
        resumed.start(self.preflight)
        resumed.close()

    def test_lmstudio_unloaded_state_needs_no_cli_fallback(self):
        from local_llm_bench.provider_runtime import LMStudioProviderRuntime
        from local_llm_bench.lmstudio_cli import unload_matching_models_via_api
        with patch("local_llm_bench.provider_runtime.unload_matching_models_via_api", return_value=[]):
            self.assertEqual(LMStudioProviderRuntime(self.config).unload_model("model-a"), [])
        with patch("local_llm_bench.lmstudio_cli._json_request", side_effect=RuntimeError("401")):
            with self.assertRaisesRegex(RuntimeError, "401"):
                unload_matching_models_via_api("model-a", api_base=self.config.api_base)

    def test_unsloth_installed_engine_version_does_not_claim_running_version(self):
        from unittest.mock import MagicMock
        from local_llm_bench.provider_runtime import UnslothStudioProviderRuntime
        session = MagicMock()
        session.request_json.return_value = {"model_identifier": "model-a", "is_gguf": True, "llama_cpp_installed_tag": "b1234", "spec_fallback_reason": "binary_no_mtp"}
        runtime = UnslothStudioProviderRuntime(self.config, session)
        info = runtime.measurement_metadata({"requested_model": "model-a", "reported_speculation": {"speculative_type": "auto"}})
        self.assertEqual(info["runtime"]["installed_version"], "b1234")
        self.assertIsNone(info["runtime"].get("version"))
        self.assertEqual(info["reported_speculation"]["spec_fallback_reason"], "binary_no_mtp")
        self.assertNotIn("reported_inference", info)

    def test_orphan_docker_benchmark_blocks_another_run(self):
        self.config.mode = "docker_task"
        with patch("local_llm_bench.docker_task.spec.load_spec"), patch("local_llm_bench.docker_task.runner._docker_binary", return_value="docker"), patch("local_llm_bench.diagnostics.host_snapshot", return_value={}), patch("local_llm_bench.diagnostics._read_command", side_effect=["28", "orphan123"]):
            with self.assertRaisesRegex(RuntimeError, "orphan123"):
                run_preflight(self.config, self.runtime, "model-a")
        self.runtime.prepare_model.assert_not_called()

    def test_docker_timeout_and_interrupt_remove_only_owned_container(self):
        from local_llm_bench.docker_task.runner import _run_question_in_docker
        from local_llm_bench.docker_task.spec import Question
        question = Question("q1", "question", "exact", "answer")
        self.config.docker_image = "fixture"
        for error in (subprocess.TimeoutExpired("docker", 10), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), patch("local_llm_bench.docker_task.runner._docker_binary", return_value="docker"), patch("local_llm_bench.docker_task.runner._docker_platform_mismatch_error", return_value=None), patch("local_llm_bench.docker_task.runner.subprocess.run", side_effect=[error, subprocess.CompletedProcess([], 0)]) as executor:
                if isinstance(error, KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        _run_question_in_docker(config=self.config, selected_model="model-a", question=question, docker_executor=executor)
                else:
                    result = _run_question_in_docker(config=self.config, selected_model="model-a", question=question, docker_executor=executor)
                    self.assertEqual(result["status"], "timeout")
                command = executor.call_args_list[0].args[0]
                owned_name = command[command.index("--name") + 1]
                self.assertTrue(owned_name.startswith("local-llm-bench-"))
                self.assertEqual(executor.call_args_list[1].args[0], ["docker", "rm", "-f", owned_name])

    def test_inference_stage_after_failed_request_is_unknown(self):
        ex = RunExecution(self.config, self.runtime, "model-a")
        ex.start(self.preflight)
        ex.save_unit("cold", 1, "q1", {"status": "timeout"}, {})
        ex.save_unit("cold", 1, "q2", {"status": "success"}, {})
        self.assertEqual(ex.cached("cold", 1, "q2")["result"]["measurement"]["inference_stage"], "unknown_after_error")
        ex.close()

    def test_failed_cold_skips_primer_and_keeps_stage_unknown_until_success(self):
        ex = RunExecution(self.config, self.runtime, "model-a")
        ex.start(self.preflight)
        self.addCleanup(ex.close)
        primer = Mock(side_effect=AssertionError("A failed measured attempt must not trigger another primer"))
        ex.before_attempt("cold", 1, primer)
        ex.save_unit("cold", 1, "prompt", {"status": "timeout"}, {})
        for iteration, status in enumerate(("error", "success", "success"), 1):
            self.assertEqual(ex.before_attempt("warm", iteration, primer), ex.api_model)
            ex.save_unit("warm", iteration, "prompt", {"status": status}, {})
        primer.assert_not_called()
        stages = [ex.cached("warm", i, "prompt")["result"]["measurement"]["inference_stage"] for i in (1, 2, 3)]
        self.assertEqual(stages, ["unknown_after_error", "unknown_after_error", "repeat"])

    def test_finished_units_can_recover_interrupted_output_publication(self):
        self.config.runs.cold_runs, self.config.runs.warm_runs = 1, 0
        ex = RunExecution(self.config, self.runtime, "model-a")
        self.execute(ex)
        self.runtime.chat_client().reset_mock()
        resumed = RunExecution(self.config, self.runtime, "model-a", resume_id=ex.run_id)
        self.execute(resumed)
        self.runtime.chat_client().assert_not_called()
        resumed.mark_outputs_saved()
        with self.assertRaisesRegex(ValueError, "完了しています"):
            RunExecution(self.config, self.runtime, "model-a", resume_id=ex.run_id)
