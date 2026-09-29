import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from bench_fakes import make_config, response, runtime_for
from local_llm_bench.config import PerformanceSettings, RequestSettings, _performance_settings
from local_llm_bench.execution import RunExecution
from local_llm_bench.inspect_harness import run_samples
from local_llm_bench.lmstudio_api import consume_sse_stream
from local_llm_bench.metrics import measured_metrics
from local_llm_bench.memory_measurement import MemorySampler
from local_llm_bench.performance import cohort_metrics, make_prompt, run_performance_benchmark
from local_llm_bench.run_logs import persist_run_logs
from local_llm_bench.runner import _turn_usage_from_result
from local_llm_bench.telemetry import normalize_turn_usage_records


class NullMemory:
    def __init__(self, pid):
        self.data = {"peak_rss_bytes": None, "peak_host_used_bytes": None}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class MetricTests(unittest.TestCase):
    def test_memory_peak_scope_and_missing_pid(self):
        from types import SimpleNamespace
        import psutil
        process = Mock()
        process.is_running.return_value = True
        process.create_time.return_value = 10
        process.pid = 100
        process.memory_info.return_value = SimpleNamespace(rss=12)
        child = Mock(pid=101)
        child.memory_info.return_value = SimpleNamespace(rss=3)
        process.children.return_value = [child]
        with patch("psutil.Process", return_value=process), \
             patch("psutil.virtual_memory", side_effect=[SimpleNamespace(used=30), SimpleNamespace(used=40)]), \
             patch("psutil.swap_memory", side_effect=[SimpleNamespace(used=5), SimpleNamespace(used=8)]):
            with MemorySampler(100, interval=60) as memory:
                pass
        self.assertEqual(memory.data["peak_rss_bytes"], 15)
        self.assertEqual(memory.data["peak_host_used_bytes"], 40)
        self.assertEqual(memory.data["peak_swap_used_bytes"], 8)
        self.assertEqual(memory.data["rss_scope"], "selected_process_tree_rss")
        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(100)), \
             patch("psutil.virtual_memory", return_value=SimpleNamespace(used=40)), \
             patch("psutil.swap_memory", return_value=SimpleNamespace(used=8)):
            with MemorySampler(100, interval=60) as missing:
                pass
        self.assertIsNone(missing.data["peak_rss_bytes"])
        self.assertEqual(missing.data["rss_scope"], "unavailable")

    def test_stream_preserves_extensions_and_distinguishes_server_ttft(self):
        payloads = [{"choices": [{"delta": {"content": "a"}}]},
                    {"usage": {"prompt_tokens": 1024, "completion_tokens": 11, "total_tokens": 1035,
                               "prompt_eval_duration": .5, "prompt_tokens_per_second": 2048,
                               "generation_tokens_per_second": 40,
                               "prompt_tokens_details": {"cached_tokens": 1000}}},
                    {"usage": {"completion_tokens_details": {"reasoning_tokens": 4}}}]
        lines = [line for p in payloads for line in ["data: " + json.dumps(p), ""]] + ["data: [DONE]", ""]
        r = consume_sse_stream(lines, started_at=0, now_fn=Mock(side_effect=[1, 2, 3, 4]))
        m = measured_metrics(r.to_dict())
        self.assertEqual(r.prompt_tokens, 1024)
        self.assertEqual(m["pp_tps"], 2048)
        self.assertEqual(m["tg_tps"], 40)
        self.assertEqual(m["cached_prompt_tokens"], 1000)
        self.assertEqual(m["reasoning_tokens"], 4)
        self.assertIsNone(m["prefill_ms"])
        self.assertEqual(m["reported_prompt_eval_ms"], 500)
        turn = _turn_usage_from_result(r, prompt_text="hello")[0]
        self.assertNotIn("prefill_sec", turn)
        self.assertEqual(turn["cached_prompt_tokens"], 1000)

    def test_estimate_and_missing_data_are_explicit(self):
        raw = {"prompt_tokens": 21, "completion_tokens": 11, "ttft_ms": 500, "total_latency_ms": 1500}
        m = measured_metrics(raw)
        self.assertEqual(m["pp_tps"], 42)
        self.assertEqual(m["tg_tps"], 10)
        self.assertEqual(m["tpot_ms"], 100)
        self.assertIsNone(m["cached_prompt_tokens"])
        self.assertEqual(m["cache_state"], "unknown")
        self.assertIn("estimate", m["sources"]["pp_tps"])
        self.assertIsNone(measured_metrics({})["pp_tps"])
        self.assertIsNone(measured_metrics({**raw, "completion_tokens": 1})["tpot_ms"])
        self.assertIsNone(measured_metrics({**raw, "ttft_ms": float("nan")})["ttft_ms"])
        self.assertIsNone(normalize_turn_usage_records([{}])[0]["cached_prompt_tokens"])

    def test_backend_timings_use_processed_count(self):
        m = measured_metrics({"prompt_tokens": 1024, "completion_tokens": 10,
                              "raw_timings": {"prompt_n": 24, "prompt_ms": 120, "predicted_per_second": 35}})
        self.assertEqual(m["pp_tps"], 200)
        self.assertEqual(m["prefill_ms"], 120)

    def test_cohort_throughput_uses_wall_window_and_rejects_failures(self):
        a = {"status": "success", "request_started_perf": 1, "request_ended_perf": 4,
             "concurrency_actual": 2, "concurrency_requested": 2,
             "metrics": {"input_tokens": 100, "output_tokens": 10}}
        b = {**a, "request_started_perf": 2, "request_ended_perf": 6}
        result = cohort_metrics([a, b], 2)
        self.assertEqual(result["elapsed_sec"], 5)
        self.assertEqual(result["output_tps"], 4)
        self.assertEqual(result["peak_inflight"], 2)
        self.assertFalse(cohort_metrics([a, {**b, "request_started_perf": 4}], 2)["eligible_for_comparison"])
        self.assertTrue(result["eligible_for_comparison"])
        self.assertIsNone(cohort_metrics([a, {**b, "status": "error"}], 2)["output_tps"])
        self.assertFalse(cohort_metrics([a], 2)["eligible_for_comparison"])

    def test_profile_validation_and_reproducible_inputs(self):
        for block in ({"concurrency": [True]}, {"input_tokens": []}, {"concurrency": [1, 1]},
                      {"input_tokens": [float("inf")]}, {"temperature": 0}, {"memory_pid": -1}):
            with self.assertRaises(ValueError):
                _performance_settings(block)
        self.assertEqual(make_prompt(1024), make_prompt(1024))
        self.assertEqual(len(make_prompt(4096)), 4096 * 4)

    def test_yaml_profile_keeps_saved_runtime_parameters(self):
        from local_llm_bench.config import load_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench.yaml"
            path.write_text("provider: omlx\nmode: performance\nmodels: [fixture]\nperformance:\n  input_tokens: [1024, 4096]\n  concurrency: [1, 2]\n")
            with patch("local_llm_bench.config.resolve_omlx_api_key", return_value=None):
                config = load_config(path)
            self.assertEqual(config.mode, "performance")
            self.assertEqual(config.performance.input_tokens, [1024, 4096])
            self.assertEqual(config.request_parameters(), {})
            self.assertEqual(config.recorded_request_parameters(), {"settings_source": "omlx_saved"})


class PerformanceInspectTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config = replace(make_config(self.root, cold=0, warm=1), mode="performance",
                              performance=PerformanceSettings([32], [1, 2]),
                              request=RequestSettings(use_lmstudio_defaults=True))
        for name in ("user_data_path", "user_cache_path"):
            self.stack.enter_context(patch("inspect_ai._util.appdirs." + name, return_value=self.root / name))
        for name in ("connect", "connect_ex", "bind"):
            self.stack.enter_context(patch("socket.socket." + name, side_effect=AssertionError("Live socket forbidden")))
        self.stack.enter_context(patch("local_llm_bench.execution.host_snapshot", return_value={}))

    def test_inspect_runs_simultaneously_without_sampling_overrides(self):
        from inspect_ai.log import read_eval_log
        barrier = threading.Barrier(2, timeout=10)
        called, saved = [], []
        def operation(sample):
            called.append(sample["id"])
            barrier.wait()
            return response(sample["id"])
        info = run_samples(config=self.config, model="fixture", samples=[
            {"id": str(i), "prompt": "same input", "metadata": {}} for i in range(2)],
            operation=operation, on_complete=lambda *args: saved.append(args),
            log_dir=self.root / "eval", metadata={})
        self.assertEqual(sorted(called), ["0", "1"])
        self.assertEqual(len(saved), 2)
        log = read_eval_log(info["log_path"])
        self.assertEqual(len(log.samples), 2)
        self.assertEqual(log.status, "success")
        self.assertIn("metrics", log.samples[0].metadata)
        self.assertEqual(info["log_storage"]["compression"], ["zstd"])

    def execute(self, execution, client):
        execution.start({"host": {}, "docker": None})
        result = run_performance_benchmark(self.config, model=execution.api_model, client=client,
                                           execution=execution, memory_factory=NullMemory)
        execution.finish("completed")
        return persist_run_logs(self.config.output, execution.enrich(result))

    def test_matrix_saved_each_sample_with_distinct_logs_and_retry(self):
        execution = RunExecution(self.config, runtime_for(self.root), "model-a")
        lock = threading.Lock()
        count = 0
        def client(**kwargs):
            nonlocal count
            self.assertNotIn("max_tokens", kwargs)
            self.assertNotIn("temperature", kwargs)
            with lock:
                count += 1
                n = count
            if n == 4:
                raise TimeoutError("fixture")
            r = response(str(n))
            r.prompt_tokens, r.completion_tokens = 35, 11
            return r
        result = self.execute(execution, client)
        self.assertEqual(len(result["records"]), 3)
        self.assertEqual(len(execution.state["units"]), 3)
        self.assertEqual(len({r["log_path"] for r in result["records"]}), 3)
        self.assertEqual(len(result["summary"]["models"]), 3)  # failure has no observed metric source
        self.assertEqual(len(execution.partial_result()["records"]), 3)
        success = {k: v for k, v in execution.state["units"].items() if v["result"]["status"] == "success"}
        retry = RunExecution(self.config, runtime_for(self.root), "model-a", retry_id=execution.run_id)
        retried = self.execute(retry, Mock(return_value=response("retry")))
        for key, value in success.items():
            self.assertEqual(retry.state["units"][key], value)
        updated = next(r for r in retried["records"] if r["response_text"] == "retry")
        self.assertEqual(updated["concurrency_actual"], 1)
        self.assertEqual(updated["concurrency_requested"], 2)
        self.assertFalse(updated["batch_metrics"]["eligible_for_comparison"])

    def test_completed_sibling_is_durable_before_other_finishes(self):
        event = threading.Event()
        def operation(sample):
            if sample["id"] == "slow":
                if not event.wait(10):
                    raise AssertionError("Completion was buffered until the cohort finished")
            return response(sample["id"])
        def completed(sample, result, error, audit):
            if sample["id"] == "fast":
                event.set()
        info = run_samples(config=self.config, model="fixture",
            samples=[{"id": name, "prompt": "same", "metadata": {}} for name in ["fast", "slow"]],
            operation=operation, on_complete=completed, log_dir=self.root / "logs", metadata={})
        self.assertEqual(info["status"], "success")

    def test_interrupted_cohort_resumes_only_missing_sample(self):
        self.config.performance = PerformanceSettings([32], [2])
        execution = RunExecution(self.config, runtime_for(self.root), "model-a")
        execution.start({"host": {}, "docker": None})
        saved = threading.Event()
        save_unit = execution.save_unit
        def save(*args):
            save_unit(*args)
            saved.set()
        execution.save_unit = save
        counter = 0
        lock = threading.Lock()
        def client(**kwargs):
            nonlocal counter
            with lock:
                counter += 1
                n = counter
            if n == 3:
                self.assertTrue(saved.wait(10))
                raise KeyboardInterrupt()
            return response("original")
        with self.assertRaises(KeyboardInterrupt):
            run_performance_benchmark(self.config, model=execution.api_model, client=client,
                                      execution=execution, memory_factory=NullMemory)
        execution.finish("interrupted")
        self.assertEqual(len(execution.state["units"]), 1)
        original = next(iter(execution.state["units"].values()))
        resumed = RunExecution(self.config, runtime_for(self.root), "model-a", resume_id=execution.run_id)
        client = Mock(return_value=response("resumed"))
        result = self.execute(resumed, client)
        self.assertEqual(client.call_count, 2)  # excluded warmup + missing sample
        self.assertEqual(len(result["records"]), 2)
        self.assertIn(original, resumed.state["units"].values())
        self.assertTrue(all(not (r.get("batch_metrics") or {}).get("eligible_for_comparison") for r in result["records"]))

    def test_catalogue_finds_each_sample_in_multi_sample_eval(self):
        from local_llm_bench.dashboard import InspectCatalogue
        directory = self.config.output.run_logs_dir / "run1" / "inspect" / "cohort" / "attempt"
        info = run_samples(config=self.config, model="fixture",
            samples=[{"id": name, "prompt": "input", "metadata": {}} for name in ["pp32-c2-r1", "pp32-c2-r2"]],
            operation=lambda sample: response(), on_complete=lambda *args: None,
            log_dir=directory, metadata={"phase": "warm", "iteration": 1})
        catalogue = InspectCatalogue(self.config.output.run_logs_dir)
        for key in ["pp32-c2-r1", "pp32-c2-r2"]:
            entries = catalogue.list("run1", "warm", 1, key)
            self.assertEqual([entry["path"] for entry in entries], [info["log_path"]])
            self.assertEqual(entries[0]["log_storage"], info["log_storage"])
        self.assertEqual(catalogue.list("run1", "cold", 1, "pp32-c2-r2"), [])
