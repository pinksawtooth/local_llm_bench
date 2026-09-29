"""Real Inspect evaluation and ReAct, using only in-memory model/MCP fixtures."""
import asyncio
from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from bench_fakes import make_config, response, runtime_for
from local_llm_bench.inspect_harness import run_unit, unit_log_directory
from local_llm_bench.docker_task.inspect_agent import run_agent
from local_llm_bench.docker_task.spec import Question


class InspectContractTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config = make_config(self.root)
        self.stack.enter_context(patch("inspect_ai._util.appdirs.user_data_path", return_value=self.root / "app"))
        self.stack.enter_context(patch("inspect_ai._util.appdirs.user_cache_path", return_value=self.root / "cache"))
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.socket.bind"):
            self.stack.enter_context(patch(target, side_effect=AssertionError("Live network forbidden")))
        self.info = {}

    def unit(self, operation, question=None):
        return run_unit(config=self.config, model="fixture", operation=operation,
                        log_dir=self.root / "logs", metadata={"phase": "warm"},
                        question=question, info=self.info)

    def test_prompt_has_one_call_and_original_timing(self):
        expected = response("answer")
        operation = Mock(return_value=expected)
        with patch.dict(os.environ, {"INSPECT_LOG_FORMAT": "json"}):
            self.assertIs(self.unit(operation), expected)
        operation.assert_called_once_with()
        self.assertEqual(expected.ttft_ms, 10)
        self.assertEqual(self.info["status"], "success")
        self.assertTrue(Path(self.info["log_path"]).is_file())
        self.assertEqual(self.info["log_storage"]["compression"], ["zstd"])
        self.assertEqual(Path(self.info["log_path"]).suffix, ".eval")

    def test_error_logged_without_retry(self):
        operation = Mock(side_effect=TimeoutError("fixture timeout"))
        with self.assertRaisesRegex(TimeoutError, "fixture timeout"):
            self.unit(operation)
        operation.assert_called_once_with()
        self.assertTrue(Path(self.info["log_path"]).is_file())
        self.assertIn("fixture timeout", self.info["error"])
        self.assertEqual(self.info["log_storage"]["compression"], ["zstd"])

    def test_interruption_propagates_without_retry(self):
        operation = Mock(side_effect=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.unit(operation)
        operation.assert_called_once_with()
        self.assertEqual(self.info["status"], "cancelled")
        self.assertEqual(self.info["log_storage"]["compression"], ["zstd"])
        # A same-process resume must not inherit a cancelled Inspect task.
        resumed = Mock(return_value=response("resumed"))
        self.assertEqual(self.unit(resumed).response_text, "resumed")
        resumed.assert_called_once_with()

    def test_worker_error_is_not_a_scored_wrong_answer(self):
        from inspect_ai.log import read_eval_log
        q = Question(id="q1", prompt="compute", answer_type="number", gold_answer=42, binary_path=None)
        expected = {"status": "timeout", "error": "worker fixture timeout"}
        self.assertIs(self.unit(lambda: expected, q), expected)
        self.assertEqual(self.info["status"], "error")
        self.assertNotIn("score", self.info)
        log = read_eval_log(self.info["log_path"])
        self.assertFalse(log.samples[0].scores)

    def test_inspect_retry_preserves_successful_unit_and_log(self):
        from local_llm_bench.execution import RunExecution
        from local_llm_bench.runner import run_benchmark
        self.config.runs.cold_runs = 1
        self.config.runs.warm_runs = 1
        runtime = runtime_for(self.root)
        preflight = {"host": {}, "docker": None}
        with patch("local_llm_bench.execution.host_snapshot", return_value={}):
            execution = RunExecution(self.config, runtime, "model-a")
            execution.start(preflight)
            result = run_benchmark(self.config, model=execution.api_model, execution=execution,
                                   client=Mock(side_effect=[response("original"), TimeoutError("fixture")]))
            execution.finish("completed")
            execution.close()
            original = execution.cached("cold", 1, "prompt")
            old_log = Path(original["result"]["inspect"]["log_path"])
            original_bytes = old_log.read_bytes()
            self.assertEqual([r["status"] for r in result["records"]], ["success", "timeout"])
            retry = RunExecution(self.config, runtime, "model-a", retry_id=execution.run_id)
            retry.start(preflight)
            client = Mock(return_value=response("retried"))
            updated = run_benchmark(self.config, model=retry.api_model, execution=retry, client=client)
            retry.finish("completed")
            retry.close()
        self.assertEqual([r["response_text"] for r in updated["records"]], ["original", "retried"])
        self.assertEqual(client.call_count, 2)  # excluded primer, then failed unit only
        self.assertEqual(retry.cached("cold", 1, "prompt"), original)
        self.assertEqual(old_log.read_bytes(), original_bytes)
        self.assertNotIn("continue_on_length", client.call_args.kwargs)

    def test_typed_host_score_and_private_gold(self):
        from inspect_ai.log import read_eval_log
        q = Question(id="q1", prompt="compute", answer_type="number", gold_answer=42, binary_path=None)
        self.unit(lambda: {"status": "success", "predicted_answer": "42"}, q)
        self.assertEqual(self.info["score"], 1)
        log = read_eval_log(self.info["log_path"])
        self.assertEqual(log.samples[0].target, "")
        self.assertNotIn("gold_answer", log.model_dump_json())

    def test_report_uses_inspect_score_without_rescoring(self):
        from local_llm_bench.docker_task.runner import _coerce_question_result
        q = Question(id="q1", prompt="compute", answer_type="number", gold_answer=42, binary_path=None)
        raw = self.unit(lambda: {"status": "success", "predicted_answer": "41"}, q)
        with patch("local_llm_bench.docker_task.scorer.score_answer", side_effect=AssertionError("duplicate scoring")):
            record = _coerce_question_result(q, raw, self.info)
        self.assertEqual(record["benchmark_score"], 0)
        self.assertEqual(record["benchmark_incorrect_count"], 1)
        with self.assertRaisesRegex(RuntimeError, "Inspect Scorer"):
            _coerce_question_result(q, raw, {})

    def test_unique_safe_log_directory(self):
        a = unit_log_directory(self.config, "run", "warm", 1, "../../bad")
        b = unit_log_directory(self.config, "run", "warm", 1, "../../bad")
        self.assertNotEqual(a, b)
        self.assertTrue(a.is_relative_to(self.config.output.run_logs_dir.resolve()))

    def agent(self, responses, *, payload=None, session=None, completion=None):
        payload = {"provider": "lmstudio", "model": "fixture", "timeout_sec": 10,
                   "settings_source": "lmstudio_saved", "question": {"id": "q1"},
                   "inspect": {"max_turns": 3, "max_tool_calls": 2, "tool_timeout_sec": 1},
                   **(payload or {})}
        trace = {"turns": []}
        if completion is None:
            completion = Mock(side_effect=[(item, 20.0) if isinstance(item, dict) else item for item in responses])
        session = session or SimpleNamespace(call_tool=AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="42")], isError=False)))
        tools = [{"type": "function", "function": {"name": "calculate", "description": "calculate",
                  "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
        result = asyncio.run(run_agent(payload=payload,
            messages=[{"role": "system", "content": "Solve"}, {"role": "user", "content": "6 * 7"}],
            tools_schema=tools, tool_sessions={"calculate": session}, completion=completion,
            trace=trace, log_dir=self.root / "worker"))
        return result, completion, session, tools

    def answer(self, text="FINAL_ANSWER: 42", calls=None, **message):
        return {"choices": [{"message": {"role": "assistant", "content": text, **message,
                 **({"tool_calls": calls} if calls else {})}, "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                          "prompt_tokens_details": {"cached_tokens": 3}}}

    def tool_call(self, arguments='{"expression":"6*7"}'):
        return {"id": "call1", "type": "function", "function": {"name": "calculate", "arguments": arguments}}

    def test_react_mcp_saved_settings_and_exact_schema(self):
        with patch.dict(os.environ, {"INSPECT_LOG_FORMAT": "json"}):
            result, completion, session, tools = self.agent([
                self.answer("", [self.tool_call()]), self.answer()])
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertEqual(result["predicted_answer"], "42")
        session.call_tool.assert_awaited_once_with("calculate", arguments={"expression": "6*7"})
        self.assertEqual(completion.call_count, 2)
        for call in completion.call_args_list:
            self.assertEqual(set(call.kwargs["body"]), {"model", "messages", "stream", "stream_options", "tools", "tool_choice"})
            self.assertTrue(call.kwargs["body"]["stream"])
            self.assertEqual(call.kwargs["body"]["stream_options"], {"include_usage": True})
            self.assertEqual(call.kwargs["body"]["tools"], tools)
        self.assertEqual(result["prompt_tokens"], 20)
        self.assertIsNone(result["ttft_ms"])
        self.assertIsNone(result["decode_tps"])
        self.assertEqual(result["inspect"]["log_storage"]["compression"], ["zstd"])
        from inspect_ai.log import read_eval_log
        log = read_eval_log(result["inspect"]["log_path"], resolve_attachments=True)
        self.assertTrue(any(message.role == "tool" and message.text == "42" for message in log.samples[0].messages))
        self.assertTrue(any(event.event == "tool" for event in log.samples[0].events))
        self.assertEqual(log.samples[0].output.completion, "FINAL_ANSWER: 42")

    def test_ds4_controls_and_reasoning_survive_tool_turn(self):
        params = {"provider": "ds4", "settings_source": None, "max_tokens": 81,
                  "reasoning_effort": "high", "ds4_sampling": {"source": "model_config", "temperature": 0.7,
                  "top_p": 0.8, "top_k": 0, "min_p": 0}}
        result, completion, _, _ = self.agent([
            self.answer("", [self.tool_call()], reasoning_content="private reasoning"), self.answer()], payload=params)
        self.assertEqual(result["status"], "success", result.get("error"))
        for call in completion.call_args_list:
            body = call.kwargs["body"]
            self.assertEqual(body["max_tokens"], 81)
            self.assertEqual(body["top_k"], 0)
            self.assertEqual(body["min_p"], 0)
        self.assertEqual(completion.call_args_list[1].kwargs["body"]["messages"][2]["reasoning_content"], "private reasoning")

    def test_malformed_tool_arguments_never_execute(self):
        result, completion, session, _ = self.agent([self.answer("", [self.tool_call("{invalid")])])
        self.assertEqual(result["status"], "error")
        self.assertEqual(completion.call_count, 1)
        session.call_tool.assert_not_awaited()

    def test_turn_budget_and_provider_errors_do_not_retry(self):
        for responses, message in (([self.answer("analysis")] * 3, "max_turns"),
                                   ([RuntimeError("HTTP 503 fixture")], "HTTP 503")):
            with self.subTest(message=message):
                result, completion, _, _ = self.agent(responses)
                self.assertEqual(result["status"], "error")
                self.assertIn(message, result["error"])
                self.assertEqual(completion.call_count, len(responses))

    def test_tool_budget_is_checked_before_execution(self):
        result, _, session, _ = self.agent([self.answer("", [self.tool_call()] * 3)])
        self.assertEqual(result["status"], "error")
        self.assertIn("max_tool_calls", result["error"])
        session.call_tool.assert_not_awaited()

    def test_unlimited_agent_exceeds_previous_turn_and_tool_limits(self):
        for limits in ({"tool_timeout_sec": 1},
                       {"max_turns": None, "max_tool_calls": None, "tool_timeout_sec": 1}):
            with self.subTest(limits=limits):
                responses = [self.answer("", [self.tool_call()]) for _ in range(97)]
                result, completion, session, _ = self.agent(
                    [*responses, self.answer()], payload={"inspect": limits, "timeout_sec": 3600})
                self.assertEqual(result["status"], "success", result.get("error"))
                self.assertEqual(result["predicted_answer"], "42")
                self.assertEqual(completion.call_count, 98)
                self.assertEqual(session.call_tool.await_count, 97)
                self.assertIsNone(result["trace"]["turn_limit"])
                self.assertEqual(result["inspect"]["turns"], 98)
                self.assertEqual(result["inspect"]["tool_calls"], 97)

    def test_unlimited_agent_uses_remaining_hour_and_rejects_late_answer(self):
        clock = [0.0]
        responses = iter([self.answer("", [self.tool_call()]), self.answer()])

        def complete(**kwargs):
            clock[0] += 1800.0
            return next(responses), 1800000.0

        completion = Mock(side_effect=complete)
        with patch("local_llm_bench.docker_task.inspect_agent.time",
                   SimpleNamespace(perf_counter=lambda: clock[0])):
            result, _, session, _ = self.agent([], completion=completion,
                payload={"timeout_sec": 3600, "inspect": {"tool_timeout_sec": 120}})
        self.assertEqual(result["status"], "timeout", result.get("error"))
        self.assertIn("question time budget exhausted", result["error"])
        self.assertIsNone(result["predicted_answer"])
        self.assertEqual([call.kwargs["timeout_sec"] for call in completion.call_args_list], [3600, 1800])
        session.call_tool.assert_awaited_once()

    def test_unlimited_agent_stops_before_next_request_when_hour_expires(self):
        clock = [0.0]

        async def tool(*args, **kwargs):
            clock[0] = 3600.0
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="42")], isError=False)

        session = SimpleNamespace(call_tool=AsyncMock(side_effect=tool))
        with patch("local_llm_bench.docker_task.inspect_agent.time",
                   SimpleNamespace(perf_counter=lambda: clock[0])):
            result, completion, _, _ = self.agent([self.answer("", [self.tool_call()])], session=session,
                payload={"timeout_sec": 3600, "inspect": {"tool_timeout_sec": 120}})
        self.assertEqual(result["status"], "timeout", result.get("error"))
        self.assertIn("question time budget exhausted", result["error"])
        self.assertEqual(completion.call_count, 1)
        session.call_tool.assert_awaited_once()

    def test_tool_timeout_ends_sample_and_preserves_trace(self):
        session = SimpleNamespace(call_tool=AsyncMock(side_effect=TimeoutError()))
        result, completion, _, _ = self.agent([self.answer("", [self.tool_call()])], session=session)
        self.assertEqual(result["status"], "timeout", result.get("error"))
        self.assertEqual(completion.call_count, 1)
        self.assertIn("timed out", (self.root / "worker/trace.json").read_text())
        self.assertEqual(result["inspect"]["log_storage"]["compression"], ["zstd"])

    def test_length_is_not_automatically_continued(self):
        answer = self.answer("FINAL_ANSWER: partial")
        answer["choices"][0]["finish_reason"] = "length"
        result, completion, _, _ = self.agent([answer])
        self.assertEqual(result["status"], "error")
        self.assertIn("output limit", result["error"])
        self.assertEqual(completion.call_count, 1)


class InspectConfigurationTests(unittest.TestCase):
    def test_cli_has_no_harness_selector(self):
        from contextlib import redirect_stderr
        from io import StringIO
        import benchmark
        for value in ("legacy", "inspect"):
            with self.subTest(value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                benchmark.build_arg_parser().parse_args(["--harness", value])
            self.assertEqual(error.exception.code, 2)

    def test_missing_inspect_stops_all_entry_points_before_runtime_access(self):
        from importlib.metadata import PackageNotFoundError
        from local_llm_bench.runner import run_benchmark
        from local_llm_bench.docker_task.runner import run_docker_task_benchmark
        from local_llm_bench.diagnostics import run_preflight
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            operation, runtime = Mock(), Mock()
            with patch("local_llm_bench.inspect_harness.importlib.metadata.version", side_effect=PackageNotFoundError):
                for call in (
                    lambda: run_benchmark(config, client=operation),
                    lambda: run_docker_task_benchmark(config, docker_executor=operation),
                    lambda: run_preflight(config, runtime, "model-a"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "Inspect AI"):
                        call()
            operation.assert_not_called()
            runtime.inspect_model.assert_not_called()

    def test_wrong_inspect_version_cannot_fall_back(self):
        from local_llm_bench.inspect_harness import require_inspect
        with patch("local_llm_bench.inspect_harness.importlib.metadata.version", return_value="0.0.0"):
            with self.assertRaisesRegex(RuntimeError, "inspect-ai=="):
                require_inspect()

    def test_default_limits_and_strict_validation(self):
        from local_llm_bench.config import load_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench.yaml"
            path.write_text('models: [fixture]\n')
            config = load_config(path)
            self.assertFalse(hasattr(config, "harness"))
            self.assertIsNone(config.inspect.max_turns)
            self.assertIsNone(config.inspect.max_tool_calls)
            for block, expected in (("inspect: {max_turns: null, max_tool_calls: null}", (None, None)),
                                    ("inspect: {max_turns: 3, max_tool_calls: 7}", (3, 7))):
                path.write_text('models: [fixture]\n' + block)
                configured = load_config(path)
                self.assertEqual((configured.inspect.max_turns, configured.inspect.max_tool_calls), expected)
            for block in ('harness: legacy', 'harness: inspect', 'harness: typo', 'inspect: {max_turns: 0}', 'inspect: {max_tool_calls: true}',
                          'inspect: {tool_timeout_sec: .inf}', 'inspect: {tool_timeout_sec: null}', 'inspect: {max_turns: 2.5}',
                          'inspect: {retries: 3}'):
                path.write_text('models: [fixture]\n' + block)
                with self.subTest(block=block), self.assertRaises(ValueError):
                    load_config(path)

    def test_lock_mismatch_rejected(self):
        from local_llm_bench.inspect_harness import dependency_lock_hash, verify_worker_lock
        verify_worker_lock(dependency_lock_hash())
        with self.assertRaisesRegex(RuntimeError, "differs"):
            verify_worker_lock("0" * 64)

    def test_preflight_rejects_legacy_image_without_loading_model(self):
        from local_llm_bench.diagnostics import run_preflight
        with tempfile.TemporaryDirectory() as directory:
            config = replace(make_config(Path(directory)), mode="docker_task", docker_image="fixture")
            runtime = runtime_for(Path(directory))
            image = {"Id": "sha256:fixture", "Os": "linux", "Architecture": "arm64"}
            with patch("local_llm_bench.inspect_harness.require_inspect"), patch("local_llm_bench.diagnostics.host_snapshot", return_value={}), \
                 patch("local_llm_bench.docker_task.spec.load_spec"), patch("local_llm_bench.docker_task.runner._docker_binary", return_value="docker"), \
                 patch("local_llm_bench.diagnostics._read_command", side_effect=["28", "", json.dumps([image])]):
                with self.assertRaisesRegex(RuntimeError, "Inspect対応"):
                    run_preflight(config, runtime, "model-a")
            runtime.prepare_model.assert_not_called()
