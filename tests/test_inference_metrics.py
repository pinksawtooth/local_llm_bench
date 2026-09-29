"""Common request timing and durable task metrics, with scripted HTTP/MCP only."""
import asyncio
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from bench_fakes import make_config
from local_llm_bench.docker_task.container_worker import _chat_completion
from local_llm_bench.docker_task.inspect_agent import run_agent
from local_llm_bench.docker_task.runner import _aggregate_attempt_record, _coerce_question_result
from local_llm_bench.docker_task.spec import Question
from local_llm_bench.history import load_history_entries, update_history
from local_llm_bench.inspect_harness import run_unit
from local_llm_bench.lmstudio_api import LMStudioAPIError, consume_sse_stream, stream_chat_completion
from local_llm_bench.metrics import aggregate_inference_metrics, inference_observation, measured_metrics
from local_llm_bench.telemetry import normalize_turn_usage_records


def sse(events):
    response = io.BytesIO(("".join("data: " + json.dumps(e) + "\n\n" for e in events)
                           + "data: [DONE]\n\n").encode())
    response.headers = {"Content-Type": "text/event-stream"}
    return response


TOOL_EVENTS = [
    {"choices": [{"delta": {"role": "assistant", "tool_calls": [
        {"index": 0, "id": "call1", "type": "function", "function": {"name": "", "arguments": ""}}]}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "look", "arguments": '{"value":'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "up", "arguments": '"42"}'}}]}}]},
    {"choices": [{"finish_reason": "tool_calls"}],
     "usage": {"prompt_tokens": 100, "completion_tokens": 11, "total_tokens": 111},
     "timings": {"prompt_per_second": 800, "predicted_per_second": 40}, "stats": {"draft": 3}},
]
ANSWER_EVENTS = [
    {"choices": [{"delta": {"role": "assistant"}}]},
    {"choices": [{"delta": {"reasoning_content": "check"}}]},
    {"choices": [{"delta": {"content": "FINAL_ANSWER: ok"}}]},
    {"choices": [{"finish_reason": "stop"}],
     "usage": {"prompt_tokens": 200, "completion_tokens": 21, "total_tokens": 221,
               "prompt_tokens_per_second": 600, "generation_tokens_per_second": 20}},
]


class InferenceMetricsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config = make_config(self.root)
        for name in ("connect", "connect_ex", "bind"):
            self.enterContext(patch("socket.socket." + name, side_effect=AssertionError("Live sockets forbidden")))
        self.enterContext(patch("inspect_ai._util.appdirs.user_data_path", return_value=self.root / "app"))
        self.enterContext(patch("inspect_ai._util.appdirs.user_cache_path", return_value=self.root / "cache"))
        self.enterContext(patch("inspect_ai._eval.task.log.git_context", return_value=None))
        self.enterContext(patch.dict("os.environ", {"INSPECT_DISPLAY": "none"}))

    def test_tool_only_stream_measures_first_function_fragment_and_merges_arguments(self):
        ticks = iter([.05, .2, .5, 1, 1.2])
        result = consume_sse_stream(sse(TOOL_EVENTS), started_at=0, now_fn=lambda: next(ticks))
        self.assertEqual(result.message["tool_calls"], [
            {"id": "call1", "type": "function", "function": {"name": "lookup", "arguments": '{"value":"42"}'}}])
        self.assertEqual(result.message["content"], "")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.ttft_ms, 200)
        metrics = measured_metrics(result.to_dict())
        self.assertEqual([metrics[k] for k in ("pp_tps", "tg_tps", "ttft_ms", "tpot_ms")], [800, 40, 200, 100])
        self.assertEqual(result.raw_stats, {"draft": 3})

    def test_interleaved_tool_indexes_and_reasoning_are_preserved(self):
        events = [
            {"choices": [{"delta": {"reasoning_content": "first ", "tool_calls": [
                {"index": 1, "id": "b", "function": {"name": "second", "arguments": "{"}},
                {"index": 0, "id": "a", "function": {"name": "first", "arguments": "{"}}]}}]},
            {"choices": [{"delta": {"reasoning_content": "next", "tool_calls": [
                {"index": 0, "function": {"arguments": "}"}},
                {"index": 1, "function": {"arguments": "}"}}]}, "finish_reason": "tool_calls"}]},
        ]
        ticks = iter([.1, .3, .4])
        result = consume_sse_stream(sse(events), started_at=0, now_fn=lambda: next(ticks))
        self.assertEqual([c["id"] for c in result.message["tool_calls"]], ["a", "b"])
        self.assertTrue(all(c["function"]["arguments"] == "{}" for c in result.message["tool_calls"]))
        self.assertEqual(result.message["reasoning_content"], "first next")
        self.assertEqual(result.ttft_ms, 100)

    def test_prompt_and_task_transport_share_the_same_measurement(self):
        def clock():
            ticks = iter([3, 3.05, 3.3, 3.8, 4.4, 4.5])
            return lambda: next(ticks)
        prompt_open = Mock(return_value=sse(ANSWER_EVENTS))
        task_open = Mock(return_value=sse(ANSWER_EVENTS))
        prompt = stream_chat_completion(api_base="http://fixture/v1", model="fixture", prompt_text="answer",
            now_fn=clock(), timeout_sec=10, urlopen=prompt_open)
        raw, elapsed = _chat_completion(api_base="http://fixture/v1", body={"model": "fixture", "messages": []},
            now_fn=clock(), timeout_sec=10, urlopen=task_open)
        self.assertEqual(measured_metrics(prompt.to_dict()), raw["metrics"])
        self.assertEqual(prompt.total_latency_ms, elapsed)
        self.assertEqual(raw["choices"][0]["message"]["reasoning_content"], "check")
        for opened in (prompt_open, task_open):
            opened.assert_called_once()
            body = json.loads(opened.call_args.args[0].data)
            self.assertTrue(body["stream"])
            self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_empty_or_nonstreaming_response_is_not_silently_retried(self):
        json_response = io.BytesIO(b'{"choices":[{"message":{"content":"answer"}}]}')
        json_response.headers = {"Content-Type": "application/json"}
        for response, message in ((sse([]), "empty streamed response"), (json_response, "non-streaming"),
                                  (sse([{"error": {"message": "fixture error"}}]), "fixture error")):
            with self.subTest(message=message):
                opened = Mock(return_value=response)
                with self.assertRaisesRegex(LMStudioAPIError, message):
                    stream_chat_completion(api_base="http://fixture/v1", model="fixture", prompt_text="answer",
                        now_fn=lambda: 1, timeout_sec=10, urlopen=opened)
                opened.assert_called_once()

    def agent(self, fail_second=False):
        opened = Mock(side_effect=[sse(TOOL_EVENTS), TimeoutError("fixture interruption") if fail_second else sse(ANSWER_EVENTS)])
        ticks = iter([0, .05, .2, .5, 1, 1.2, 3, 3.05, 3.3, 3.8, 4.4, 4.5])
        def completion(**kwargs):
            return _chat_completion(api_base="http://fixture/v1", urlopen=opened, now_fn=lambda: next(ticks), **kwargs)
        session = SimpleNamespace(call_tool=AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="42")], isError=False)))
        payload = {"model": "fixture", "provider": "lmstudio", "timeout_sec": 10,
                   "settings_source": "lmstudio_saved", "question": {"id": "q1"},
                   "inspect": {"max_turns": 3, "max_tool_calls": 3, "tool_timeout_sec": 1}}
        tools = [{"type": "function", "function": {"name": "lookup", "description": "lookup",
                  "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}}]
        result = asyncio.run(run_agent(payload=payload, messages=[{"role": "user", "content": "answer"}],
            tools_schema=tools, tool_sessions={"lookup": session}, completion=completion,
            trace={"turns": []}, log_dir=self.root / "worker"))
        session.call_tool.assert_awaited_once_with("lookup", arguments={"value": "42"})
        self.assertEqual(opened.call_count, 2)
        return result

    def test_stream_metrics_survive_agent_inspect_question_trial_and_history(self):
        from inspect_ai.log import read_eval_log
        worker = self.agent()
        self.assertEqual(worker["status"], "success", worker.get("error"))
        metrics = worker["metrics"]
        for field, expected in {"pp_tps": 700, "tg_tps": 30, "ttft_ms": 250, "tpot_ms": 80,
                                "input_tokens": 300, "output_tokens": 32, "e2e_ms": 2700}.items():
            self.assertAlmostEqual(metrics[field], expected)
        self.assertAlmostEqual(metrics["total_tps"], 332 / 2.7)
        self.assertEqual(len(worker["inference_requests"]), 2)
        self.assertEqual(normalize_turn_usage_records(worker["turn_usage"])[0]["metrics"]["pp_tps"], 800)
        log = read_eval_log(worker["inspect"]["log_path"])
        observed = [event.output.metadata for event in log.samples[0].events if event.event == "model" and event.output]
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0]["metrics"]["pp_tps"], 800)
        self.assertEqual(observed[0]["raw_stats"], {"draft": 3})

        # A long tool/task duration must not replace the request clock.
        worker["total_latency_ms"] = 100000
        question = Question(id="q1", prompt="answer", answer_type="exact", gold_answer="ok")
        audit = {}
        raw = run_unit(config=self.config, model="fixture", operation=lambda: worker, question=question,
            log_dir=self.root / "host", metadata={}, info=audit)
        result = _coerce_question_result(question, raw, audit)
        record = _aggregate_attempt_record(phase="warm", iteration=1, started_at="2026-09-16T00:00:00Z",
            prompt_text="answer", question_results=[result])
        self.assertEqual(record["total_latency_ms"], 100000)
        self.assertEqual(record["metrics"], metrics)
        self.assertAlmostEqual(record["ttft_ms"], 250)
        host_log = read_eval_log(audit["log_path"])
        self.assertEqual(host_log.samples[0].metadata["metrics"], metrics)
        self.assertEqual(host_log.samples[0].output.metadata["metrics"], metrics)
        self.assertEqual(len(host_log.samples[0].metadata["inference_requests"]), 2)
        history = self.root / "history.json"
        update_history(history, {"run_id": "fixture", "model": "fixture", "benchmark_mode": "docker_task",
            "status": "completed", "records": [record]})
        saved = load_history_entries(history)[0]["records"][0]
        self.assertEqual(saved["metrics"], metrics)
        self.assertEqual(saved["question_results"][0]["metrics"], metrics)
        self.assertEqual(saved["inference_requests"][0]["raw_timings"]["prompt_per_second"], 800)

    def test_failed_task_retains_completed_requests_and_missing_coverage(self):
        worker = self.agent(fail_second=True)
        self.assertEqual(worker["status"], "timeout")
        self.assertEqual(len(worker["inference_requests"]), 2)
        self.assertEqual(worker["inference_requests"][0]["metrics"]["ttft_ms"], 200)
        self.assertEqual(worker["inference_requests"][1]["status"], "error")
        self.assertIsNone(worker["metrics"]["ttft_ms"])
        self.assertEqual(worker["metrics"]["aggregation"]["observed_requests"]["ttft_ms"], 1)

    def test_trial_flattens_requests_instead_of_taking_median_of_question_medians(self):
        def observation(ttft):
            return inference_observation({"prompt_tokens": 100, "completion_tokens": 10,
                                          "ttft_ms": ttft, "total_latency_ms": 1000})
        first, second = [observation(10)], [observation(100), observation(200), observation(300)]
        questions = [{"question_id": str(i), "status": "success", "inference_requests": requests,
                      "metrics": aggregate_inference_metrics(requests)} for i, requests in enumerate((first, second))]
        record = _aggregate_attempt_record(phase="warm", iteration=1, started_at="now", prompt_text="",
                                            question_results=questions)
        self.assertEqual(record["metrics"]["ttft_ms"], 150)
        self.assertEqual(record["metrics"]["request_count"], 4)
        questions.append({"question_id": "old", "status": "success", "ttft_ms": 999})
        partial = _aggregate_attempt_record(phase="warm", iteration=1, started_at="now", prompt_text="",
                                             question_results=questions)
        self.assertIsNone(partial["metrics"]["ttft_ms"])
        self.assertEqual(partial["metrics"]["aggregation"]["observed_requests"]["ttft_ms"], 4)

    def test_unknown_usage_and_single_output_token_are_not_zero_filled(self):
        raw = {"ttft_ms": 100, "total_latency_ms": 200}
        unknown = aggregate_inference_metrics([inference_observation(raw)])
        self.assertEqual(unknown["ttft_ms"], 100)
        self.assertIsNone(unknown["pp_tps"])
        self.assertIsNone(unknown["tpot_ms"])
        single = aggregate_inference_metrics([inference_observation({**raw, "prompt_tokens": 5, "completion_tokens": 1})])
        self.assertEqual(single["pp_tps"], 50)
        self.assertIsNone(single["tpot_ms"])
        self.assertIsNone(single["cached_prompt_tokens"])
