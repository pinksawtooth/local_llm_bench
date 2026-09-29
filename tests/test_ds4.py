"""Offline DS4 contract tests, including the API cases ported from RevBench."""
import asyncio
import contextlib
import copy
import io
import itertools
import json
import tempfile
import subprocess
import types
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import yaml
import benchmark
from bench_fakes import make_config, response
from local_llm_bench import ds4
from local_llm_bench.config import AuthSettings, LMStudioLoadSettings, RequestSettings, load_config
from local_llm_bench.conditions import comparison_metadata
from local_llm_bench.docker_task import container_worker
from local_llm_bench.docker_task.runner import _write_request_payload
from local_llm_bench.docker_task.spec import Question
from local_llm_bench.execution import RunExecution
from local_llm_bench.http_boundary import HTTPBoundaryError, HTTPMessageTooLarge, _NoRedirectHandler, open_url_no_redirect
from local_llm_bench.provider_runtime import DS4ProviderRuntime, LMStudioProviderRuntime, build_provider_runtime
from local_llm_bench.runner import run_benchmark

API_MODEL = "deepseek-v4-flash"
BASE = "http://127.0.0.1:8000/v1"
REPO = Path(__file__).resolve().parent.parent
MODEL_INFO = {"id": API_MODEL, "object": "model", "context_length": 32768}

class _Response(io.BytesIO):
    def __init__(self, body: bytes, content_type: str = "application/json") -> None:
        super().__init__(body)
        self.headers = Message()
        self.headers["Content-Length"] = str(len(body))
        self.headers["Content-Type"] = content_type


def _models_response(*models):
    return _Response(json.dumps({"object": "list", "data": list(models)}).encode())


class _OfflineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict("os.environ", {"LOCAL_BENCH_INSPECT_LOG_DIR": str(root / "worker")}))
        self.enterContext(patch("inspect_ai._util.appdirs.user_data_path", return_value=root / "app"))
        self.enterContext(patch("inspect_ai._util.appdirs.user_cache_path", return_value=root / "cache"))
        self.enterContext(patch("inspect_ai._eval.task.log.git_context", return_value=None))
        # A regression must fail here instead of reaching a running benchmark,
        # inference server, CLI, Docker daemon, or model loader on the host.
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.socket.bind", "subprocess.Popen"):
            blocker = patch(target, side_effect=AssertionError("Real runtime access is forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)


class DS4APITests(_OfflineTestCase):
    def test_metadata_uses_only_authenticated_get(self) -> None:
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_models_response(MODEL_INFO)) as request:
            context = ds4.ds4_context_length(API_MODEL, base_url=BASE, api_key="test-key")
        self.assertEqual(context, 32768)
        sent = request.call_args.args[0]
        self.assertEqual(sent.full_url, BASE + "/models")
        self.assertEqual(sent.method, "GET")
        self.assertIsNone(sent.data)
        self.assertEqual(sent.get_header("Authorization"), "Bearer test-key")

    def test_base_urls_preserve_proxy_prefix_and_accept_root(self) -> None:
        self.assertEqual(ds4._ds4_base_url("http://localhost:8000/"), "http://localhost:8000/v1")
        self.assertEqual(ds4._ds4_base_url("https://ds4.example/proxy/v1/"), "https://ds4.example/proxy/v1")
        for base in ("ftp://localhost:8000", "http://user:key@localhost:8000/v1", BASE + "?key=secret", BASE + "/completions"):
            with self.subTest(base=base), self.assertRaises(HTTPBoundaryError):
                ds4.read_ds4_model_info(API_MODEL, base_url=base)
        with self.assertRaises(HTTPBoundaryError):
            ds4.read_ds4_model_info(API_MODEL, api_key="bad\r\nX-Leak: yes")

    def test_alias_matching_never_selects_first_model_or_guesses_family(self) -> None:
        for alias in ("deepseek-v4", "ds4/deepseek-v4-flash", "glm-5.2"):
            with self.subTest(alias=alias), patch(
                "local_llm_bench.ds4.open_url_no_redirect", return_value=_models_response(MODEL_INFO)
            ), self.assertRaises(ds4.DS4ModelNotFoundError):
                ds4.read_ds4_model_info(alias)
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_models_response(MODEL_INFO)):
            self.assertEqual(ds4.ds4_context_length("openai/" + API_MODEL), 32768)

    def test_ambiguous_alias_and_invalid_context_fail(self) -> None:
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_models_response(MODEL_INFO, MODEL_INFO)):
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                ds4.read_ds4_model_info(API_MODEL)
        for value in (None, True, 0, -1, "32768", 32768.0):
            with self.subTest(value=value), patch(
                "local_llm_bench.ds4.open_url_no_redirect",
                return_value=_models_response({**MODEL_INFO, "context_length": value}),
            ), self.assertRaisesRegex(RuntimeError, "context_length"):
                ds4.ds4_context_length(API_MODEL)

    def test_response_boundary_rejects_malformed_and_oversized_payloads(self) -> None:
        for raw in (b'{"data":[],"data":[]}', b'[]', b'{"data":[null]}', b'{"data":null}'):
            with self.subTest(raw=raw), patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_Response(raw)):
                with self.assertRaises((HTTPBoundaryError, RuntimeError)):
                    ds4.read_ds4_model_info(API_MODEL)
        oversized = _Response(b"")
        oversized.headers.replace_header("Content-Length", str(64 * 1024 * 1024 + 1))
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=oversized):
            with self.assertRaises(HTTPMessageTooLarge):
                ds4.read_ds4_model_info(API_MODEL)
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_Response(b"{}", "text/html")):
            with self.assertRaises(HTTPBoundaryError):
                ds4.read_ds4_model_info(API_MODEL)

    def test_http_and_connection_errors_do_not_expose_credentials_or_body(self) -> None:
        for error in (
            urllib.error.HTTPError(BASE, 401, "secret-key", {}, io.BytesIO(b"secret-body")),
            urllib.error.URLError("secret-connection-details"),
        ):
            with self.subTest(error=type(error)), patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=error):
                with self.assertRaises(RuntimeError) as caught:
                    ds4.read_ds4_model_info(API_MODEL)
            self.assertNotIn("secret", str(caught.exception))

    def test_environment_never_promotes_advertised_hash_or_sampler_to_identity(self) -> None:
        claimed = {**MODEL_INFO, "sha256": "a" * 64, "engine_version": "claimed", "temperature": 1.0}
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_models_response(claimed)):
            info = ds4.ds4_environment_info(API_MODEL, api_key="private-key")
        self.assertEqual(info["capture_status"], "partial")
        self.assertEqual(info["model_identity_scope"], "server_compatibility_alias")
        self.assertEqual(info["loaded_context_length"], 32768)
        self.assertEqual(info["local_artifact_identity_status"], "unavailable")
        for field in ("engine_version", "temperature", "local_artifact_sha256", "api_key"):
            self.assertNotIn(field, info)

    def test_environment_rechecks_restarted_server_and_reports_unavailable(self) -> None:
        with patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=[
            _models_response(MODEL_INFO),
            _models_response({**MODEL_INFO, "context_length": 65536}),
            _models_response(),
            urllib.error.URLError("offline"),
        ]):
            self.assertEqual(ds4.ds4_environment_info(API_MODEL)["loaded_context_length"], 32768)
            self.assertEqual(ds4.ds4_environment_info(API_MODEL)["loaded_context_length"], 65536)
            self.assertEqual(ds4.ds4_environment_info(API_MODEL)["model_api_status"], "model_unmatched")
            self.assertEqual(ds4.ds4_environment_info(API_MODEL)["model_api_status"], "unavailable")


def ds4_config(root):
    config = make_config(root, cold=0)
    return replace(config, provider="ds4", api_base=BASE, models=[API_MODEL],
                   request=RequestSettings(temperature=1.0, max_tokens=128, top_p=1.0, reasoning_effort="high"),
                   auth=AuthSettings(bearer_token="private-fixture-key"), ds4={"management": "external"},
                   ds4_sampling={"source": "run_config"},
                   docker_api_base="http://host.docker.internal:8000/v1")


def completion(content="answer", *, finish="stop", reasoning="checked", tool_calls=None):
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if tool_calls:
        message["tool_calls"] = tool_calls
    payload = {"choices": [{"delta": message, "finish_reason": finish}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
    return _Response(("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode(), "text/event-stream")


class DS4ConfigTests(_OfflineTestCase):
    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "bench.yaml"

    def config(self, extra=None, **kwargs):
        data = {"provider": "ds4", "models": [API_MODEL], "ds4": {"management": "external"}, **(extra or {})}
        self.path.write_text(yaml.safe_dump(data))
        return load_config(self.path, **kwargs)

    def test_explicit_provider_defaults_and_both_examples(self):
        config = self.config()
        self.assertEqual(config.api_base, BASE)
        self.assertEqual(config.docker_api_base, "http://host.docker.internal:8000/v1")
        self.assertEqual(config.runs.cold_runs, 0)
        self.assertIsInstance(build_provider_runtime(config), DS4ProviderRuntime)
        self.assertIsInstance(build_provider_runtime(replace(config, provider="lmstudio")), LMStudioProviderRuntime)
        for filename in ("bench_ds4.yaml", "bench_d_compile_arm64_ds4.yaml"):
            example = load_config(REPO / "configs" / filename)
            self.assertEqual(example.request.reasoning_effort, "high")
            self.assertEqual(example.request.max_tokens, 8192)
            self.assertEqual(example.runs.cold_runs, 1)
            self.assertEqual(example.ds4["management"], "managed")

    def test_credentials_do_not_fall_back_to_other_providers(self):
        with patch.dict("os.environ", {"UNSLOTH_STUDIO_BEARER_TOKEN": "wrong-secret", "UNSLOTH_STUDIO_PASSWORD": "wrong-password"}, clear=True):
            self.assertIsNone(self.config().auth.bearer_token)
            with patch.dict("os.environ", {"DS4_API_KEY": "right-secret"}):
                config = self.config()
                self.assertEqual(config.auth.bearer_token, "right-secret")
                self.assertNotIn("right-secret", json.dumps(config.to_dict()))
                self.assertEqual(build_provider_runtime(config).docker_environment(), {"DS4_API_KEY": "right-secret"})

    def test_invalid_lifecycle_and_request_settings_fail_before_runtime_access(self):
        cases = [
            {"runs": {"cold_runs": 1}}, {"lmstudio": {"parallelism": 2}},
            {"lmstudio": {"parallelism_sweep": [1, 2]}}, {"lmstudio": {"context_length": 4096}},
            {"request": {"reasoning_effort": "ultra"}}, {"request": {"top_p": float("nan")}},
            {"request": {"temperature": float("inf")}}, {"auth": {"username": "user", "password": "test"}},
        ]
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.config(extra)
        for override in ({"cli_cold_runs": 1}, {"cli_parallelism": 2}, {"cli_parallelism_sweep": "2,3"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.config(**override)
        for effort in sorted(ds4.DS4_REASONING_EFFORTS):
            self.assertEqual(self.config({"request": {"reasoning_effort": effort}}).request.reasoning_effort, effort)

    def test_docker_endpoint_tracks_port_and_proxy_path_without_network(self):
        config = self.config({"api_base": "http://localhost:9000/proxy/v1"})
        self.assertEqual(config.docker_api_base, "http://host.docker.internal:9000/proxy/v1")
        self.assertEqual(ds4.docker_ds4_base_url("http://[::1]:9000/v1"), "http://host.docker.internal:9000/v1")
        self.assertEqual(self.config({"api_base": "https://ds4.example/proxy/v1"}).docker_api_base, "https://ds4.example/proxy/v1")
        for endpoint in ("http://host.docker.internal:1234/v1", "http://host.docker.internal:8000/other/v1"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                self.config({"docker": {"api_base": endpoint}})

    def test_managed_launch_defaults_paths_and_cold_runs(self):
        settings = {"server_path": "engine/ds4-server", "model_path": "models/flash.gguf"}
        config = self.config({"ds4": settings})
        self.assertEqual(config.ds4, {
            "management": "managed", "server_path": str(self.path.resolve().parent / "engine/ds4-server"),
            "model_path": str(self.path.resolve().parent / "models/flash.gguf"), "context_length": 32768,
            "startup_timeout_sec": 1200, "server_args": [],
        })
        self.assertEqual(config.runs.cold_runs, 1)
        self.assertEqual(self.config({"ds4": settings}, cli_cold_runs=2).runs.cold_runs, 2)
        self.assertTrue(build_provider_runtime(config).managed)

    def test_launch_config_rejects_ambiguous_or_unsafe_settings_without_io(self):
        valid = {"server_path": "engine/ds4-server", "model_path": "models/flash.gguf"}
        invalid = [None, {}, {**valid, "management": "external"}, {**valid, "management": "guess"},
                   {**valid, "context_length": True}, {**valid, "context_length": 1_000_001},
                   {**valid, "startup_timeout_sec": 0}, {**valid, "startup_timeout_sec": float("nan")},
                   {**valid, "server_args": "--ssd-streaming"}, {**valid, "unknown": "x"}]
        invalid += [{**valid, "server_args": args} for args in (["--ctx", "4096"], ["--port=9000"], ["-m", "other.gguf"], ["--chdir", "/tmp"], ["--help"])]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.config({"ds4": settings})
        for endpoint in ("https://127.0.0.1:8000/v1", "http://remote.invalid/v1", "http://localhost:8000/proxy/v1"):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "local HTTP /v1"):
                self.config({"ds4": valid, "api_base": endpoint})
        with self.assertRaisesRegex(ValueError, "context_length"):
            self.config({"ds4": {**valid, "context_length": 512}})


class DS4RuntimeTests(_OfflineTestCase):
    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = ds4_config(self.root)
        self.runtime = build_provider_runtime(self.config)
        snapshot = patch("local_llm_bench.execution.host_snapshot", return_value={"cpu_count": 8})
        snapshot.start()
        self.addCleanup(snapshot.stop)
        self.preflight = {"host": {"cpu_count": 8}, "docker": None}

    def test_attachment_validates_context_and_never_uses_claimed_artifact_or_sampler(self):
        claimed = {**MODEL_INFO, "path": "/models/claimed.gguf", "sha256": "a" * 64,
                   "runtime": {"version": "claimed"}, "load_config": {"parallelism": 100},
                   "reported_inference": {"thinking": True, "sampling": {"temperature": 1}}}
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_models_response(claimed)) as opened:
            alias, info = self.runtime.prepare_model("openai/" + API_MODEL)
        self.assertEqual(alias, API_MODEL)
        self.assertEqual(info["load_config"], {"context_length": 32768})
        self.assertEqual(info["model_identity_scope"], "server_compatibility_alias")
        self.assertIsNone(info["runtime"]["version"])
        for field in ("path", "sha256", "reported_inference", "quantization"):
            self.assertNotIn(field, info)
        self.assertEqual(opened.call_args.args[0].method, "GET")
        with self.assertRaisesRegex(RuntimeError, "外部管理"):
            self.runtime.unload_model(API_MODEL)
        self.config.request.max_tokens = 32768
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_models_response(MODEL_INFO)), self.assertRaisesRegex(ValueError, "context_length"):
            self.runtime.inspect_model(API_MODEL)

    def test_stream_keeps_auth_and_stops_at_length(self):
        replies = [completion("part", finish="length"), completion("done")]
        with patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=replies) as opened:
            result = self.runtime.chat_client()(api_base=BASE, model=API_MODEL, prompt_text="hello",
                temperature=1.0, max_tokens=128, timeout_sec=10, now_fn=lambda: float(next(ticks)))
        bodies = [json.loads(call.args[0].data) for call in opened.call_args_list]
        self.assertEqual(len(bodies), 1)
        self.assertEqual([body["stream"] for body in bodies], [True])
        self.assertEqual(result.response_text, "part")
        self.assertEqual(result.finish_reason, "length")
        for call, body in zip(opened.call_args_list, bodies):
            self.assertEqual(call.args[0].get_header("Authorization"), "Bearer private-fixture-key")
            self.assertEqual(body["top_p"], 1.0)
            self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(result.reasoning_text, "checked")
        self.assertEqual(bodies[-1]["max_tokens"], 128)

    def test_sse_reasoning_and_visible_text_are_separate(self):
        events = [{"choices": [{"delta": {"reasoning_content": "thinking"}}]},
                  {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}]
        raw = "".join("data: " + json.dumps(item) + "\n\n" for item in events) + "data: [DONE]\n\n"
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_Response(raw.encode(), "text/event-stream")):
            result = self.runtime.chat_client()(api_base=BASE, model=API_MODEL, prompt_text="hello",
                temperature=1.0, max_tokens=128, timeout_sec=10, now_fn=lambda: float(next(ticks)))
        self.assertEqual(result.response_text, "answer")
        self.assertEqual(result.reasoning_text, "thinking")

    def test_external_lifecycle_is_saved_without_false_load_measurements(self):
        ex = RunExecution(self.config, self.runtime, API_MODEL)
        chat = Mock(return_value=response())
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_models_response(MODEL_INFO)), patch.object(self.runtime, "unload_model") as unload:
            ex.start(self.preflight)
            result = run_benchmark(self.config, model=ex.api_model, client=chat, execution=ex)
            ex.finish("completed")
            ex.close()
        result = ex.enrich(result)
        unload.assert_not_called()
        self.assertEqual(chat.call_count, 3)  # one excluded primer + two saved repeats
        self.assertEqual([entry["event"] for entry in result["lifecycle"]], ["attach", "warmup"])
        self.assertIsNone(result["timings"]["load_wall_sec"])
        self.assertIsNone(result["timings"]["first_after_load_wall_sec"])
        self.assertEqual(result["timings"]["first_after_load_count"], 0)
        self.assertEqual(result["conditions"]["protocol"], "external-server-repeat-v1")
        self.assertIsNone(result["conditions"]["model"]["artifact"]["sha256"])
        self.assertIn("load.other_effective_settings", result["comparison"]["unknown_fields"])
        self.assertIsNone(result["conditions"]["thinking"]["value"])
        self.assertNotEqual(result["comparison"]["group_id"], comparison_metadata(result["conditions"], "other")["group_id"])
        saved = json.loads(ex.checkpoint.read_text())
        self.assertEqual(len(saved["units"]), 2)
        self.assertNotIn("private-fixture-key", json.dumps(saved))
        self.assertEqual(result["records"][0]["measurement"]["inference_stage"], "repeat")

    def test_resume_requires_unknown_condition_override_and_preserves_finished_units(self):
        with patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=lambda *a, **k: _models_response(MODEL_INFO)):
            original = RunExecution(self.config, self.runtime, API_MODEL)
            original.start(self.preflight)
            original.before_attempt("warm", 1, lambda _: {"status": "success"})
            original.save_unit("warm", 1, "prompt", {"status": "success", "response_text": "preserved"}, {})
            original.finish("interrupted")
            original.close()
            rejected = RunExecution(self.config, self.runtime, API_MODEL, resume_id=original.run_id)
            with self.assertRaisesRegex(ValueError, "allow-unverified-resume"):
                rejected.start(self.preflight)
            resumed = RunExecution(self.config, self.runtime, API_MODEL, resume_id=original.run_id, allow_unverified=True)
            resumed.start(self.preflight)
            self.assertEqual(resumed.cached("warm", 1, "prompt")["result"]["response_text"], "preserved")
            chat = Mock(return_value=response())
            result = run_benchmark(self.config, model=API_MODEL, client=chat, execution=resumed)
            self.assertEqual(chat.call_count, 2)  # primer + missing unit only
            self.assertEqual(result["records"][0]["response_text"], "preserved")


    def test_main_uses_attachment_without_unloading(self):
        with (patch("benchmark.load_config", return_value=self.config),
              patch("benchmark.server_lock", side_effect=lambda _: contextlib.nullcontext()),
              patch("local_llm_bench.diagnostics.host_snapshot", return_value={}),
              patch("benchmark.ensure_report_html"), patch("builtins.print"),
              patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=lambda *a, **k: _models_response(MODEL_INFO)),
              patch.object(DS4ProviderRuntime, "chat_client", return_value=Mock(return_value=response())),
              patch.object(DS4ProviderRuntime, "unload_model", side_effect=AssertionError("Unloading is forbidden"))):
            self.assertEqual(benchmark.main([]), 0)
        history = json.loads(self.config.output.history_json.read_text())
        self.assertEqual(history[0]["request"]["reasoning_effort"], "high")
        self.assertIsNone(history[0]["timings"]["load_wall_sec"])
        self.assertNotIn("private-fixture-key", self.config.output.history_json.read_text())


class DS4BoundaryTests(_OfflineTestCase):
    def test_no_proxy_or_redirect_can_forward_credentials(self):
        request = urllib.request.Request(BASE + "/models", headers={"Authorization": "Bearer fixture"})
        with patch("local_llm_bench.http_boundary.urllib.request.build_opener") as build:
            open_url_no_redirect(request, timeout_sec=10)
        self.assertEqual(build.call_args.args[0].proxies, {})
        handler = build.call_args.args[1]
        self.assertIsInstance(handler, _NoRedirectHandler)
        self.assertIsNone(handler.redirect_request(request, None, 302, "redirect", {}, "https://other.invalid/"))

    def test_completion_errors_are_bounded_and_hide_private_responses(self):
        session = ds4.DS4Session(BASE, "private-fixture")
        request = urllib.request.Request(BASE + "/chat/completions", data=b"{}", method="POST")
        for error in (urllib.error.HTTPError(BASE, 401, "private-details", {}, io.BytesIO(b"private-body")),
                      urllib.error.URLError("private-details")):
            with patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=error), self.assertRaises(RuntimeError) as caught:
                session.urlopen(request, timeout=10)
            self.assertNotIn("private", str(caught.exception))
        for body in (_Response(b'{"a":1,"a":2}'), _Response(b"{}", "text/html")):
            with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=body), self.assertRaises(HTTPBoundaryError):
                session.urlopen(request, timeout=10)
        other = urllib.request.Request("http://other.invalid/v1/chat/completions", data=b"{}")
        with self.assertRaises(HTTPBoundaryError):
            session.urlopen(other, timeout=10)

    def test_stream_limit_applies_even_without_content_length(self):
        raw = _Response(b"data: too long\n", "text/event-stream")
        del raw.headers["Content-Length"]
        session = ds4.DS4Session(BASE)
        request = urllib.request.Request(BASE + "/chat/completions", data=b"{}")
        with patch("local_llm_bench.ds4.MAX_HTTP_RESPONSE_BYTES", 4), patch("local_llm_bench.ds4.open_url_no_redirect", return_value=raw):
            with session.urlopen(request, timeout=10) as stream, self.assertRaises(HTTPBoundaryError):
                list(stream)
        self.assertTrue(raw.closed)


class DS4DockerTests(_OfflineTestCase):
    def test_worker_tool_round_trip_preserves_reasoning_and_request_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ds4_config(root)
            question = Question(id="q1", prompt="check", answer_type="exact", gold_answer="ok")
            request_path = _write_request_payload(bundle_dir=root, config=config, selected_model=API_MODEL,
                                                  question=question, staged_binary_ref=None)
            payload = json.loads(request_path.read_text())
        self.assertNotIn("private-fixture-key", json.dumps(payload))
        self.assertEqual(payload["api_base"], "http://host.docker.internal:8000/v1")
        tool = types.SimpleNamespace(name="lookup", description="lookup", inputSchema={"type": "object", "properties": {}})
        calls = []

        class ToolSession:
            async def call_tool(self, name, arguments):
                calls.append((name, arguments))
                return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="found")])

        async def open_mcp(**kwargs):
            return contextlib.nullcontext(), ToolSession(), types.SimpleNamespace(tools=[tool])

        tool_calls = [{"id": "call_fixture", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
        replies = [completion("", finish="tool_calls", reasoning="inspect first", tool_calls=tool_calls),
                   completion("FINAL_ANSWER: ok")]
        with (patch("local_llm_bench.docker_task.container_worker._open_mcp_stdio_session", side_effect=open_mcp),
              patch("local_llm_bench.docker_task.container_worker.resolve_native_binary_target", return_value=types.SimpleNamespace(path=None)),
              patch.dict("os.environ", {"DS4_API_KEY": "private-fixture-key"}),
              patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=replies) as opened):
            result = asyncio.run(container_worker._run_question(payload))
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertEqual(result["predicted_answer"], "ok")
        self.assertEqual(calls, [("lookup", {})])
        requests = [json.loads(call.args[0].data) for call in opened.call_args_list]
        self.assertEqual(len(requests), 2)
        for call, sent in zip(opened.call_args_list, requests):
            self.assertEqual(call.args[0].get_header("Authorization"), "Bearer private-fixture-key")
            self.assertEqual(sent["top_p"], 1.0)
            self.assertEqual(sent["reasoning_effort"], "high")
        self.assertEqual(requests[1]["messages"][2]["reasoning_content"], "inspect first")
        self.assertEqual(requests[1]["messages"][3]["tool_call_id"], "call_fixture")
        self.assertNotIn("private-fixture-key", json.dumps(result))
        self.assertEqual(result["trace"]["turns"][1]["request"]["reasoning_effort"], "high")


ticks = itertools.count()


class DS4ManagedServerTests(_OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        executable = self.root / "engine with spaces" / "ds4-server"
        executable.parent.mkdir()
        executable.write_text("offline placeholder - never executable in tests")
        executable.chmod(0o700)
        model = self.root / "fake flash.gguf"
        model.write_bytes(b"offline test fixture")
        self.settings = ds4.normalize_ds4_launch_config({
            "server_path": str(executable), "model_path": str(model),
            "context_length": 32768, "server_args": ["--ssd-streaming"],
        }, directory=self.root)
        self.process = Mock()
        self.process.poll.return_value = None
        self.process.wait.side_effect = lambda **kwargs: setattr(self.process.poll, "return_value", 0)
        self.port_lock = Mock()
        self.socket = MagicMock()
        self.popen = self.enterContext(patch("local_llm_bench.ds4.subprocess.Popen", return_value=self.process))
        self.enterContext(patch("local_llm_bench.ds4._acquire_port_lock", return_value=self.port_lock))
        self.enterContext(patch("local_llm_bench.ds4.socket.socket", return_value=self.socket))
        self.metadata = self.enterContext(patch("local_llm_bench.ds4.ds4_context_length", return_value=32768))

    def server(self, **overrides):
        kwargs = {"base_url": BASE, "api_key": "ds4-local", "config": self.settings,
                  "log_path": self.root / "logs/ds4-server.log", "lock_directory": self.root / "locks"}
        kwargs.update(overrides)
        server = ds4.ManagedDS4Server(API_MODEL, **kwargs)
        self.addCleanup(server.stop)
        return server

    def start_ready(self, server):
        with patch.object(server, "_has_listen_banner", return_value=True):
            server.start()

    def test_start_passes_model_context_and_host_then_stops_owned_process(self) -> None:
        server = self.server(docker_access=True)
        self.start_ready(server)
        command = self.popen.call_args.args[0]
        self.assertEqual(command, [self.settings["server_path"], "-m", self.settings["model_path"],
                                 "--ctx", "32768", "--host", "0.0.0.0", "--port", "8000", "--ssd-streaming"])
        self.assertEqual(self.popen.call_args.kwargs["cwd"], Path(self.settings["server_path"]).parent)
        self.assertTrue(self.popen.call_args.kwargs["start_new_session"])
        self.socket.__enter__.return_value.setsockopt.assert_called_once_with(
            ds4.socket.SOL_SOCKET, ds4.socket.SO_REUSEADDR, 1
        )
        self.socket.__enter__.return_value.listen.assert_called_once_with(1)
        self.assertEqual(self.metadata.call_args.kwargs["base_url"], BASE)
        self.assertEqual(server.log_path.stat().st_mode & 0o777, 0o600)
        server.stop()
        server.stop()
        self.process.terminate.assert_called_once()
        self.process.kill.assert_not_called()
        self.port_lock.close.assert_called_once()
        self.assertTrue(Path(self.settings["model_path"]).is_file())

    def test_busy_port_never_spawns_or_contacts_existing_server(self) -> None:
        self.socket.__enter__.return_value.bind.side_effect = OSError("occupied")
        server = self.server()
        with self.assertRaisesRegex(RuntimeError, "existing server was left untouched"):
            server.start()
        self.popen.assert_not_called()
        self.metadata.assert_not_called()
        self.process.terminate.assert_not_called()
        self.port_lock.close.assert_called_once()

    def test_loopback_conflict_is_detected_even_if_wildcard_bind_would_succeed(self) -> None:
        def bind(address):
            if address == ("127.0.0.1", 8000):
                raise OSError("Address already in use")
        self.socket.__enter__.return_value.bind.side_effect = bind
        server = self.server()
        with self.assertRaisesRegex(RuntimeError, "port 8000 is already in use"):
            server.start()
        self.popen.assert_not_called()
        self.metadata.assert_not_called()
        self.port_lock.close.assert_called_once()

    def test_missing_model_fails_before_port_or_process_access(self) -> None:
        settings = {**self.settings, "model_path": str(self.root / "missing.gguf")}
        with self.assertRaisesRegex(RuntimeError, "model_path does not exist"):
            self.server(config=settings).start()
        self.popen.assert_not_called()
        self.socket.__enter__.assert_not_called()
        self.metadata.assert_not_called()

    def test_managed_remote_url_is_rejected_without_runtime_access(self) -> None:
        for url in ("https://127.0.0.1:8000/v1", "http://remote.example:8000/v1", "http://localhost:8000/proxy/v1"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "local HTTP /v1"):
                self.server(base_url=url)
        self.popen.assert_not_called()

    def test_cancelled_before_worker_starts_never_loads(self) -> None:
        server = self.server()
        server.stop()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            server.start()
        self.popen.assert_not_called()
        self.metadata.assert_not_called()

    def test_loading_cancellation_terminates_child(self) -> None:
        server = self.server()
        with patch.object(server, "_has_listen_banner", return_value=False), patch.object(
            server._stopped, "wait", side_effect=lambda timeout: server._stopped.set()
        ):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                server.start()
        self.process.terminate.assert_called_once()
        self.port_lock.close.assert_called_once()
        self.metadata.assert_not_called()

    def test_timeout_does_not_accept_another_servers_metadata(self) -> None:
        server = self.server()
        with patch.object(server, "_has_listen_banner", return_value=False), patch("local_llm_bench.ds4.time.monotonic", side_effect=[0, 1201]):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                server.start()
        self.metadata.assert_not_called()
        self.process.terminate.assert_called_once()
        self.port_lock.close.assert_called_once()

    def test_context_mismatch_and_early_exit_release_owned_resources(self) -> None:
        server = self.server()
        self.metadata.return_value = 65536
        with self.assertRaisesRegex(RuntimeError, "context_length differs"):
            self.start_ready(server)
        self.process.terminate.assert_called_once()
        self.port_lock.close.assert_called_once()
        self.process.reset_mock()
        self.process.poll.return_value = 2
        server = self.server()
        with self.assertRaisesRegex(RuntimeError, "exit code 2"):
            server.start()
        self.process.terminate.assert_not_called()

    def test_failed_process_creation_releases_lock(self) -> None:
        self.popen.side_effect = OSError("exec failed")
        with self.assertRaisesRegex(OSError, "exec failed"):
            self.server().start()
        self.port_lock.close.assert_called_once()
        self.process.terminate.assert_not_called()

    def test_shutdown_escalates_only_its_child_when_termination_times_out(self) -> None:
        server = self.server()
        self.start_ready(server)
        def wait(**kwargs):
            if not self.process.kill.called:
                raise subprocess.TimeoutExpired("offline-child", 5)
            self.process.poll.return_value = 0
        self.process.wait.side_effect = wait
        server.stop()
        self.process.terminate.assert_called_once()
        self.process.kill.assert_called_once()
        self.port_lock.close.assert_called_once()

    def test_readiness_log_ignores_previous_run_banner(self) -> None:
        server = self.server()
        server.log_path.parent.mkdir()
        banner = b"0912 12:00:00 ds4-server: listening on http://127.0.0.1:8000\n"
        server.log_path.write_bytes(banner)
        server._log_start = len(banner)
        self.assertFalse(server._has_listen_banner())
        with server.log_path.open("ab") as stream:
            stream.write(b"loading model\n" + banner)
        self.assertTrue(server._has_listen_banner())


class DS4ManagedExecutionTests(_OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        executable = self.root / "ds4-server"
        executable.write_text("offline fixture; never executed")
        executable.chmod(0o700)
        model = self.root / "flash.gguf"
        model.write_bytes(b"tiny offline GGUF fixture")
        settings = ds4.normalize_ds4_launch_config({"server_path": str(executable), "model_path": str(model)}, directory=self.root)
        self.config = replace(ds4_config(self.root), ds4=settings)
        self.config.runs.cold_runs = 1
        self.runtime = build_provider_runtime(self.config)
        self.preflight = {"host": {"cpu_count": 8}, "docker": None}
        self.servers = []

        def construct(model, **kwargs):
            server = Mock(model=model, ready=False)
            server.start.side_effect = lambda: setattr(server, "ready", True)
            server.stop.side_effect = lambda: setattr(server, "ready", False)
            self.servers.append(server)
            return server

        self.constructor = self.enterContext(patch("local_llm_bench.provider_runtime.ManagedDS4Server", side_effect=construct))
        self.metadata = self.enterContext(patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=lambda *a, **k: _models_response(MODEL_INFO)))
        self.enterContext(patch("local_llm_bench.execution.host_snapshot", return_value={"cpu_count": 8}))
        self.enterContext(patch("local_llm_bench.diagnostics.host_snapshot", return_value={"cpu_count": 8}))
        self.enterContext(patch("builtins.print"))

    def test_preflight_and_failed_launch_do_not_probe_external_listener(self):
        from local_llm_bench.diagnostics import run_preflight
        preflight = run_preflight(self.config, self.runtime, API_MODEL)
        self.assertEqual(preflight["model"]["model_api_status"], "not_loaded_by_run")
        self.assertEqual(preflight["model"]["configured_context_length"], 32768)
        self.assertNotIn("loaded_context_length", preflight["model"])
        self.constructor.assert_not_called()
        self.metadata.assert_not_called()
        ex = RunExecution(self.config, self.runtime, API_MODEL)
        server = Mock(model=API_MODEL, ready=False)
        server.start.side_effect = RuntimeError("occupied")
        self.constructor.side_effect = None
        self.constructor.return_value = server
        with self.assertRaisesRegex(RuntimeError, "occupied"):
            ex.start(preflight)
        ex.close()
        self.metadata.assert_not_called()
        server.stop.assert_called()
        self.assertIsNone(self.runtime._server)
        self.assertEqual(self.runtime.inspect_model(API_MODEL)["model_api_status"], "not_loaded_by_run")
        self.metadata.assert_not_called()

    def test_each_cold_restarts_owned_server_and_records_loading_separately(self):
        self.config.runs.cold_runs = 2
        ex = RunExecution(self.config, self.runtime, API_MODEL)
        ex.start(self.preflight)
        chat = Mock(return_value=response())
        result = run_benchmark(self.config, model=API_MODEL, client=chat, execution=ex)
        ex.close(keep_loaded=True)
        ex.finish("completed")
        result = ex.enrich(result)
        self.assertEqual(len(self.servers), 2)
        for server in self.servers:
            server.start.assert_called_once()
            server.stop.assert_called_once()
        self.assertEqual(chat.call_count, 4)
        self.assertEqual([r["measurement"]["inference_stage"] for r in result["records"]], ["first_after_load", "first_after_load", "repeat", "repeat"])
        self.assertEqual(result["conditions"]["protocol"], "load-first-repeat-v1")
        self.assertEqual([event["event"] for event in result["lifecycle"]], ["load", "load"])
        self.assertIsNotNone(result["timings"]["load_wall_sec"])
        self.assertEqual(result["timings"]["first_after_load_count"], 2)
        self.assertEqual(result["conditions"]["load"]["requested"], self.config.ds4)
        evidence = result["conditions"]["provider_evidence"]
        self.assertEqual(evidence["server_management"], "managed")
        self.assertEqual(evidence["launch_config_sha256"], ds4.ds4_launch_config_sha256(self.config.ds4))
        self.assertIsNone(result["conditions"]["model"]["artifact"]["sha256"])
        self.assertEqual(self.constructor.call_args.kwargs["log_path"], ex.directory / "ds4-server.log")
        self.assertNotIn("ds4", result["request"])

    def test_launch_commitment_is_frozen_and_resume_rejects_changes_before_launch(self):
        ex = RunExecution(self.config, self.runtime, API_MODEL)
        original = copy.deepcopy(self.config.ds4)
        self.config.ds4["server_args"].append("--ssd-streaming")
        ex.start(self.preflight)
        self.assertEqual(self.constructor.call_args.kwargs["config"], original)
        ex.finish("interrupted")
        ex.close()
        count = self.constructor.call_count
        with self.assertRaisesRegex(ValueError, "ds4_launch_config.server_args"):
            RunExecution(self.config, self.runtime, API_MODEL, resume_id=ex.run_id, allow_unverified=True)
        self.assertEqual(self.constructor.call_count, count)

    def test_managed_docker_only_receives_inference_settings(self):
        self.config.mode = "docker_task"
        self.runtime.begin_run(self.root / "run")
        self.runtime.prepare_model(API_MODEL)
        self.assertTrue(self.constructor.call_args.kwargs["docker_access"])
        question = Question(id="q", prompt="test", answer_type="exact", gold_answer="ok")
        payload_path = _write_request_payload(bundle_dir=self.root, config=self.config, selected_model=API_MODEL,
                                              question=question, staged_binary_ref=None)
        payload = payload_path.read_text()
        self.assertNotIn("server_path", payload)
        self.assertNotIn("model_path", payload)
        self.assertNotIn("startup_timeout_sec", payload)
        self.assertNotIn("private-fixture-key", payload)
        self.runtime.unload_model(API_MODEL)
        self.servers[0].stop.assert_called_once()

    def test_main_stops_owned_child_on_success_error_and_interrupt(self):
        for failure in (None, RuntimeError("offline failure"), KeyboardInterrupt(), SystemExit(143)):
            with self.subTest(failure=type(failure).__name__):
                client = Mock(return_value=response()) if failure is None else Mock(side_effect=failure)
                with (patch("benchmark.load_config", return_value=self.config),
                      patch("benchmark.server_lock", side_effect=lambda _: contextlib.nullcontext()),
                      patch("benchmark.ensure_report_html"),
                      patch.object(DS4ProviderRuntime, "chat_client", return_value=client)):
                    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                        with self.assertRaises(type(failure)):
                            benchmark.main([])
                    else:
                        self.assertEqual(benchmark.main([]), 0)
                        run = json.loads(self.config.output.latest_json.read_text())
                        self.assertEqual(run["status"], "completed")
                        expected_status = "error" if failure is not None else "success"
                        self.assertEqual([record["status"] for record in run["records"]],
                                         [expected_status] * (self.config.runs.cold_runs + self.config.runs.warm_runs))
                self.servers[-1].stop.assert_called_once()
        self.assertNotIn("private-fixture-key", self.config.output.history_json.read_text())

    def test_port_reservation_is_private_and_released(self):
        directory = self.root / "locks"
        first = ds4._acquire_port_lock(directory, 8000)
        try:
            with self.assertRaisesRegex(RuntimeError, "already reserved"):
                ds4._acquire_port_lock(directory, 8000)
        finally:
            first.close()
        ds4._acquire_port_lock(directory, 8000).close()
