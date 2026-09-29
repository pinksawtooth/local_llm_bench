"""Saved-settings routing tests; all HTTP and MCP responses are local fixtures."""
import asyncio
import contextlib
import io
import itertools
import json
import tempfile
import types
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

from bench_fakes import make_config
from local_llm_bench.conditions import capture_conditions, execution_contract, unknown_conditions
from local_llm_bench.config import RequestSettings, load_config
from local_llm_bench.docker_task import container_worker
from local_llm_bench.docker_task.runner import _write_request_payload
from local_llm_bench.docker_task.spec import Question
from local_llm_bench.lmstudio_api import LMStudioAPIError, stream_chat_completion
from local_llm_bench.runner import run_benchmark

REPO = Path(__file__).resolve().parent.parent
SAVED_SOURCE = {"settings_source": "lmstudio_saved"}
EXPLICIT = {"temperature": 0.0, "max_tokens": 128, "top_p": 0.9, "reasoning_effort": "high"}


def completion(text="answer", *, finish="stop", tool_calls=None, stream=True):
    message = {"content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    payload = {
        "choices": [{"delta" if stream else "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }
    encoded = json.dumps(payload)
    body = f"data: {encoded}\n\ndata: [DONE]\n\n" if stream else encoded
    response = io.BytesIO(body.encode())
    response.headers = {"Content-Type": "text/event-stream" if stream else "application/json"}
    return response


class LMStudioSavedDefaultsTests(unittest.TestCase):
    def setUp(self):
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.socket.bind", "subprocess.Popen"):
            blocker = patch(target, side_effect=AssertionError("Real runtime access is forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(patch.dict("os.environ", {"LOCAL_BENCH_INSPECT_LOG_DIR": str(self.root / "worker")}))
        self.enterContext(patch("inspect_ai._util.appdirs.user_data_path", return_value=self.root / "app"))
        self.enterContext(patch("inspect_ai._util.appdirs.user_cache_path", return_value=self.root / "cache"))
        self.enterContext(patch("inspect_ai._eval.task.log.git_context", return_value=None))

    def assert_no_overrides(self, body):
        self.assertTrue(set(EXPLICIT).isdisjoint(body), body)
        self.assertNotIn("settings_source", body)

    def test_qwen_presets_use_saved_inference_and_load_settings(self):
        for filename in ("bench_qwen3_8_flash_next.yaml", "bench_d_compile_arm64_qwen3_8_flash_next.yaml"):
            with self.subTest(filename=filename):
                config = load_config(REPO / "configs" / filename)
                self.assertEqual(config.models, ["qwen3.8-flash-next"])
                self.assertTrue(config.request.use_lmstudio_defaults)
                self.assertIsNone(config.request.temperature)
                self.assertIsNone(config.request.max_tokens)
                self.assertEqual(config.request.api_parameters(), {})
                self.assertEqual(config.request.recorded_parameters(), SAVED_SOURCE)
                self.assertIsNone(config.lmstudio_load.parallelism)
                self.assertIsNone(config.lmstudio_load.context_length)

    def test_config_rejects_conflicting_overrides_and_non_lmstudio_provider(self):
        path = self.root / "bench.yaml"
        cases = [({"use_lmstudio_defaults": True, key: value}, {}, "lmstudio")
                 for key, value in EXPLICIT.items()]
        cases.extend([
            ({"use_lmstudio_defaults": True}, {"cli_temperature": 0.0}, "lmstudio"),
            ({"use_lmstudio_defaults": True}, {"cli_max_tokens": 128}, "lmstudio"),
            ({"use_lmstudio_defaults": True}, {}, "unsloth_studio"),
            ({"use_lmstudio_defaults": "true"}, {}, "lmstudio"),
        ])
        for request, cli, provider in cases:
            with self.subTest(request=request, cli=cli, provider=provider):
                path.write_text(json.dumps({"provider": provider, "models": ["fixture"], "request": request}))
                with self.assertRaises(ValueError):
                    load_config(path, **cli)
        path.write_text(json.dumps({"models": ["fixture"], "request": {"use_lmstudio_defaults": False, **EXPLICIT}}))
        self.assertEqual(load_config(path).request.api_parameters(), EXPLICIT)
        path.write_text(json.dumps({"models": ["fixture"]}))
        self.assertEqual(load_config(path).request.api_parameters(), {})
        self.assertEqual(load_config(path).recorded_request_parameters(), SAVED_SOURCE)

    def test_prompt_warmup_and_measurement_omit_overrides_and_do_not_continue_length(self):
        config = make_config(self.root, cold=0, warm=1)
        config.request = RequestSettings(use_lmstudio_defaults=True)
        execution = MagicMock(run_id="fixture", started_at="2026-09-12T00:00:00+00:00")
        execution.cached.return_value = None

        def before_attempt(phase, iteration, warmup):
            warmup("loaded-fixture")
            return "loaded-fixture"

        execution.before_attempt.side_effect = before_attempt
        opened = MagicMock(side_effect=[completion(finish="length", stream=True), completion(finish="length", stream=True)])
        with contextlib.redirect_stdout(io.StringIO()):
            run = run_benchmark(config, client=partial(stream_chat_completion, urlopen=opened),
                                execution=execution, now_fn=itertools.count().__next__)
        self.assertEqual(opened.call_count, 2)  # One primer and one measured response.
        for call in opened.call_args_list:
            body = json.loads(call.args[0].data)
            self.assert_no_overrides(body)
            self.assertEqual(body["model"], "loaded-fixture")
        self.assertEqual(run["records"][0]["status"], "success")
        self.assertEqual(run["records"][0]["finish_reason"], "length")
        self.assertEqual(run["request"], SAVED_SOURCE)
        saved_log = execution.save_unit.call_args.args[4]
        self.assertEqual(saved_log["request"]["settings_source"], "lmstudio_saved")
        self.assertTrue(set(EXPLICIT).isdisjoint(saved_log["request"]))

    def test_empty_stream_does_not_resend_or_change_saved_or_explicit_policy(self):
        for saved in (True, False):
            with self.subTest(saved=saved):
                empty_stream = io.BytesIO(b"data: [DONE]\n\n")
                opened = MagicMock(side_effect=[empty_stream, completion(finish="length" if saved else "stop")])
                settings = RequestSettings(**EXPLICIT, use_lmstudio_defaults=saved)
                with self.assertRaisesRegex(LMStudioAPIError, "empty streamed response"):
                    stream_chat_completion(
                        api_base="http://localhost:1234/v1", model="fixture", prompt_text="hello",
                        **settings.api_parameters(), timeout_sec=10, now_fn=itertools.count().__next__, urlopen=opened,
                    )
                self.assertEqual(opened.call_count, 1)
                bodies = [json.loads(call.args[0].data) for call in opened.call_args_list]
                self.assertEqual([body["stream"] for body in bodies], [True])
                for body in bodies:
                    if saved:
                        self.assert_no_overrides(body)
                    else:
                        self.assertEqual({key: body[key] for key in EXPLICIT}, EXPLICIT)

    def test_saved_policy_does_not_claim_effective_sampling(self):
        config = make_config(self.root)
        config.request = RequestSettings(use_lmstudio_defaults=True)
        contract = execution_contract(config, "fixture", None)
        conditions = capture_conditions(config, {}, contract, {})
        self.assertEqual(contract["request"], SAVED_SOURCE)
        self.assertEqual(conditions["request"], SAVED_SOURCE)
        self.assertEqual(conditions["sampling"], {"value": None, "source": "unavailable"})
        self.assertIn("sampling", unknown_conditions(conditions))

    def test_docker_payload_and_every_tool_turn_preserve_policy(self):
        tool = types.SimpleNamespace(name="lookup", description="lookup", inputSchema={"type": "object", "properties": {}})

        class ToolSession:
            async def call_tool(self, name, arguments):
                return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="found")])

        async def open_mcp(**kwargs):
            return contextlib.nullcontext(), ToolSession(), types.SimpleNamespace(tools=[tool])

        docker_bases = {"omlx": "http://host.docker.internal:8000/v1", "mlx_serve": "http://host.docker.internal:11234/v1"}
        for provider, saved in (("lmstudio", True), ("lmstudio", False), ("omlx", True), ("mlx_serve", True)):
            with self.subTest(provider=provider, saved=saved):
                config = make_config(self.root)
                config.provider = provider
                config.request = RequestSettings(**EXPLICIT, use_lmstudio_defaults=saved and provider == "lmstudio",
                                                 use_omlx_defaults=saved and provider == "omlx",
                                                 use_mlx_serve_defaults=saved and provider == "mlx_serve")
                config.docker_api_base = docker_bases.get(provider, "http://host.docker.internal:1234/v1")
                path = _write_request_payload(
                    bundle_dir=self.root, config=config, selected_model="fixture",
                    question=Question(id="q1", prompt="check", answer_type="exact", gold_answer="ok"), staged_binary_ref=None,
                )
                payload = json.loads(path.read_text())
                if saved:
                    self.assertEqual(payload["settings_source"], provider + "_saved")
                    self.assertTrue(set(EXPLICIT).isdisjoint(payload))
                    payload.update(EXPLICIT)  # Stale worker fields must not override saved policy.
                calls = [{"id": "call_fixture", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
                replies = [completion("", finish="tool_calls", tool_calls=calls), completion("FINAL_ANSWER: ok")]
                opened = MagicMock(side_effect=replies)
                with (
                    patch.object(container_worker, "_open_mcp_stdio_session", side_effect=open_mcp),
                    patch.object(container_worker, "resolve_native_binary_target", return_value=types.SimpleNamespace(path=None)),
                    patch.object(container_worker.urllib.request, "urlopen", opened),
                    patch("local_llm_bench.omlx.open_url_no_redirect", opened),
                    patch("local_llm_bench.mlx_serve.open_url_no_redirect", opened),
                ):
                    result = asyncio.run(container_worker._run_question(payload))
                self.assertEqual(result["status"], "success", result.get("error"))
                self.assertEqual(result["predicted_answer"], "ok")
                self.assertEqual(opened.call_count, 2)
                for call, turn in zip(opened.call_args_list, result["trace"]["turns"]):
                    body = json.loads(call.args[0].data)
                    if saved:
                        self.assert_no_overrides(body)
                        self.assertEqual(turn["request"]["settings_source"], provider + "_saved")
                        self.assertTrue(set(EXPLICIT).isdisjoint(turn["request"]))
                    else:
                        self.assertEqual({key: body[key] for key in EXPLICIT}, EXPLICIT)
                second_body = json.loads(opened.call_args_list[1].args[0].data)
                tool_results = [message for message in second_body["messages"] if message["role"] == "tool"]
                self.assertEqual(len(tool_results), 1)
                self.assertEqual(tool_results[0]["tool_call_id"], "call_fixture")
