"""mlx-serve API fixtures only: no native MLX import, sockets or model/server operation."""
import contextlib
import copy
import http.client
import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from bench_fakes import make_config
from local_llm_bench.conditions import unknown_conditions
from local_llm_bench.config import RequestSettings, load_config
from local_llm_bench.execution import RunExecution
from local_llm_bench.http_boundary import HTTPBoundaryError, MAX_HTTP_ERROR_BYTES
from local_llm_bench.mlx_serve import (
    MLXServeSession, MLX_SERVE_REQUEST_KEYS, normalize_mlx_serve_config,
    resolve_local_model_path, validate_mlx_serve_model_id, validate_mlx_serve_request,
)
from local_llm_bench.persistence import server_identity
from local_llm_bench.provider_runtime import MLXServeProviderRuntime, build_provider_runtime
from local_llm_bench.runner import run_benchmark

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "org/model-a"


def reply(value):
    response = io.BytesIO(json.dumps(value).encode())
    response.headers = {"Content-Type": "application/json"}
    response.status = 200
    return response


class ModelAPI:
    """Shapes taken from mlx-serve 26.9.4 responses (index page, /v1/models, /props)."""

    def __init__(self):
        self.version = "26.9.4-fixture"
        self.metrics_enabled = True
        self.gauges = {"requests_running": 0, "requests_waiting": 0}
        self.model = {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "mlx-serve",
                      "loaded": False, "state": "unloaded", "bytes_resident": 0, "bytes_on_disk": 4096,
                      "capabilities": ["chat", "tool_use", "streaming", "json_schema"],
                      "meta": {"quantization": "4-bit", "context_length": 8192, "model_max_tokens": 4096,
                               "engine": "mlx", "architecture": "qwen3_5", "is_moe": False, "mtp_available": True,
                               "mtp_loaded": False, "gen_temperature": 0.8, "gen_top_p": 0.95, "gen_top_k": 0}}
        self.inventory = [self.model]
        self.settings = {"version": self.version, "engine": "mlx", "kv_quant": "off", "prefill_chunk": 8192,
                         "max_concurrent": 1, "mtp": {"loaded": True, "default_on": True, "depth": 6},
                         "pld": {"default_on": True, "draft_len": 5, "key_len": 3}, "drafter": "<none>"}
        self.requests = []
        self.accept_mutation = True

    def __call__(self, request, *, timeout_sec):
        self.requests.append(request)
        path = urlsplit(request.full_url).path
        if path == "/health":
            return reply({"status": "ok"})
        if path == "/api/version":
            return reply({"version": self.version})
        if path == "/metrics.json":
            if not self.metrics_enabled:
                raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, io.BytesIO(b""))
            return reply({"gauges": self.gauges})
        if path == "/v1/models":
            return reply({"object": "list", "data": self.inventory})
        if path == "/props":
            return reply({"default_generation_settings": {"model": MODEL_ID if self.model["loaded"] else "", "n_ctx": 8192},
                          "settings": self.settings})
        if path in {"/v1/load-model", "/v1/unload-model"}:
            assert request.get_header("Content-type") == "application/json"
            assert json.loads(request.data) == {"model": MODEL_ID}, request.data
            load = path == "/v1/load-model"
            if self.accept_mutation:
                self.model.update(loaded=load, state="ready" if load else "unloaded", bytes_resident=1 if load else 0)
                self.model["meta"]["mtp_loaded"] = load
            return reply({"model": {"id": MODEL_ID if load else "", "object": "model", "loaded": load,
                                    "state": "ready" if load else "unloaded"}})
        if path == "/v1/chat/completions":
            payload = {"choices": [{"delta": {"content": "fixture answer"}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                       "timings": {"prompt_ms": 20.0, "predicted_ms": 30.0, "prompt_per_second": 250.0, "predicted_per_second": 100.0}}
            response = io.BytesIO(("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode())
            response.headers = {"Content-Type": "text/event-stream"}
            return response
        raise AssertionError("Unexpected fixture request: " + path)

    @property
    def mutations(self):
        return [urlsplit(request.full_url).path.rsplit("/", 1)[-1].removesuffix("-model")
                for request in self.requests if request.get_method() == "POST" and "-model" in request.full_url]


class MLXServeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {}))
        os.environ.pop("MLX_SERVE_API_KEY", None)
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.socket.bind", "subprocess.Popen"):
            self.enterContext(patch(target, side_effect=AssertionError("Real runtime access forbidden")))
        for name in ("user_data_path", "user_cache_path"):
            self.enterContext(patch("inspect_ai._util.appdirs." + name, return_value=self.root / name))
        self.enterContext(patch("inspect_ai._eval.task.log.git_context", return_value=None))
        self.enterContext(patch("local_llm_bench.execution.host_snapshot", return_value={}))
        self.enterContext(patch("tempfile.tempdir", str(self.root)))
        self.model_dir = self.root / "models"
        self.directory = self.model_dir / MODEL_ID
        self.directory.mkdir(parents=True)
        (self.directory / "config.json").write_text('{"model_type":"fixture"}')
        (self.directory / "tokenizer.json").write_text('{"tokenizer":"fixture"}')
        (self.directory / "model.safetensors").write_bytes(b"tiny fixture weights")
        self.api = ModelAPI()
        self.opened = self.enterContext(patch("local_llm_bench.mlx_serve.open_url_no_redirect", side_effect=self.api))
        self.config = make_config(self.root, cold=2, warm=1)
        self.config.provider = "mlx_serve"
        self.config.api_base = "http://127.0.0.1:11234/v1"
        self.config.docker_api_base = "http://host.docker.internal:11234/v1"
        self.config.request = RequestSettings(use_mlx_serve_defaults=True)
        self.config.mlx_serve = {"model_dirs": [str(self.model_dir)]}
        self.runtime = build_provider_runtime(self.config)

    def read_config(self, **values):
        path = self.root / "bench.yaml"
        path.write_text(json.dumps({"provider": "mlx_serve", "models": [MODEL_ID], **values}))
        return load_config(path)

    def test_defaults_auth_presets_and_same_server_lock(self):
        with patch.dict("os.environ", {"MLX_SERVE_API_KEY": "fixture-key", "OMLX_API_KEY": "other-key",
                                       "DS4_API_KEY": "other-key", "UNSLOTH_STUDIO_BEARER_TOKEN": "unrelated-key"}):
            config = self.read_config()
        self.assertEqual(config.auth.bearer_token, "fixture-key")
        self.assertEqual(config.api_base, "http://127.0.0.1:11234/v1")
        self.assertEqual(config.request_parameters(), {})
        self.assertEqual(config.recorded_request_parameters(), {"settings_source": "mlx_serve_saved"})
        self.assertNotIn("fixture-key", json.dumps(config.to_dict()))
        self.assertEqual(config.docker_api_base, "http://host.docker.internal:11234/v1")
        self.assertEqual(server_identity(config.api_base), server_identity(config.docker_api_base))
        self.assertEqual(config.mlx_serve, {"model_dirs": [str(Path.home() / ".mlx-serve/models")]})
        runtime = build_provider_runtime(config)
        self.assertIsInstance(runtime, MLXServeProviderRuntime)
        self.assertEqual(runtime.docker_environment(), {"MLX_SERVE_API_KEY": "fixture-key"})
        self.assertEqual(self.read_config(api_base="http://localhost:11234").api_base, "http://localhost:11234/v1")
        relative = self.read_config(mlx_serve={"model_dirs": ["weights", "~/other"]})
        self.assertEqual(relative.mlx_serve["model_dirs"],
                         [str(relative.config_path.parent / "weights"), str(Path.home() / "other")])
        for name in ("bench_mlx_serve.yaml", "bench_performance_mlx_serve.yaml",
                     "bench_d_compile_arm64_mlx_serve.yaml", "bench_mafc_arm64_mlx_serve.yaml"):
            with self.subTest(name=name):
                preset = load_config(REPO / "configs" / name)
                self.assertEqual(preset.provider, "mlx_serve")
                self.assertEqual(preset.request_parameters(), {})
                self.assertEqual(preset.recorded_request_parameters(), {"settings_source": "mlx_serve_saved"})
                self.assertFalse(preset.lmstudio_load.has_load_overrides())
                self.assertFalse(preset.lmstudio_load.parallelism_sweep)
                self.assertEqual(preset.docker_api_base, "http://host.docker.internal:11234/v1")
                self.assertTrue(all(Path(item).is_absolute() for item in preset.mlx_serve["model_dirs"]))
        self.opened.assert_not_called()

    def test_invalid_config_fails_without_io(self):
        for fields in (
            {"request": {"temperature": 0}}, {"request": {"max_tokens": 200}},
            {"request": {"use_mlx_serve_defaults": "true"}}, {"request": {"use_lmstudio_defaults": True}},
            {"request": {"use_omlx_defaults": True}},
            {"request": {"use_mlx_serve_defaults": False, "temperature": float("nan")}},
            {"request": {"use_mlx_serve_defaults": False, "min_p": 0.1}},
            {"request": {"use_mlx_serve_defaults": False, "seed": 7}},
            {"request": {"unknown_setting": True}},
            {"api_base": "http://user:secret@localhost:11234/v1"}, {"api_base": "http://localhost:11234/api/v1"},
            {"api_base": "http://localhost:11234/v1?secret=x"},
            {"docker": {"api_base": "http://host.docker.internal:11235/v1"}},
            {"auth": {"username": "someone"}}, {"lmstudio": {"context_length": 4096}},
            {"mlx_serve": {"model_dir": "x"}}, {"mlx_serve": {"model_dirs": "x"}},
            {"mlx_serve": {"model_dirs": [""]}}, {"mlx_serve": {"model_dirs": ["a"] * 9}},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.read_config(**fields)
        with self.assertRaises(ValueError):
            self.read_config(provider="lmstudio", request={"use_mlx_serve_defaults": True})
        with self.assertRaises(ValueError):
            self.read_config(provider="lmstudio", mlx_serve={"model_dirs": ["x"]})
        for key, value in (("top_k", -1), ("top_p", 0), ("temperature", 3), ("max_tokens", 0),
                           ("reasoning_effort", " high"), ("min_p", 0.1), ("seed", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_mlx_serve_request({key: value})
        for model in ("model@peer", "/abs/path", "../escape", "a//b", "a/", "", "with space"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                validate_mlx_serve_model_id(model)
        config = self.read_config(request={"use_mlx_serve_defaults": False, "temperature": .2, "top_k": 40, "max_tokens": 64})
        self.assertEqual(config.request_parameters(), {"temperature": .2, "max_tokens": 64, "top_k": 40})
        self.assertEqual(config.recorded_request_parameters(), config.request_parameters())
        self.opened.assert_not_called()

    def test_exact_id_lifecycle_saved_settings_and_reported_defaults(self):
        info = self.runtime.inspect_model(MODEL_ID)
        self.assertEqual(info["identifier"], MODEL_ID)
        self.assertEqual(info["path"], str(self.directory))
        self.assertEqual(info["local_artifact_identity_status"], "inferred_from_model_dirs")
        self.assertEqual(info["format"], "MLX")
        self.assertEqual(info["quantization_name"], "4-bit")
        self.assertEqual(info["runtime"], {"engine": "mlx-serve", "version": "26.9.4-fixture", "source": "provider_status"})
        self.assertEqual(info["load_config"]["context_length"], 8192)
        self.assertEqual(info["state"], "unloaded")
        self.assertEqual(info["reported_inference"], {})
        self.assertEqual(info["runtime_features"], {})
        self.assertEqual(info["server_activity"], {"requests_running": 0, "requests_waiting": 0})
        self.assertEqual(self.api.mutations, [])
        self.assertEqual(self.runtime.unload_model(MODEL_ID), [])
        api_model, loaded = self.runtime.prepare_model(MODEL_ID)
        self.assertEqual(api_model, MODEL_ID)
        self.assertEqual(loaded["load_status"], "loaded")
        self.assertGreaterEqual(loaded["load_time_seconds"], 0)
        measured = self.runtime.measurement_metadata(loaded)
        self.assertEqual(measured["state"], "loaded")
        self.assertEqual(measured["reported_inference"]["sampling"], {"temperature": 0.8, "top_p": 0.95, "top_k": 0})
        self.assertEqual(measured["reported_inference"]["speculative_decoding"],
                         {"mtp_loaded": True, "mtp_default_on": True, "pld_default_on": True, "drafter": "<none>"})
        self.assertEqual(measured["runtime_features"]["props_scope"], "default_model")
        self.assertEqual(measured["runtime_features"]["n_ctx"], 8192)
        self.assertEqual(measured["runtime_features"]["max_concurrent"], 1)
        self.assertNotIn("bytes_resident", json.dumps(measured))
        with self.assertRaisesRegex(RuntimeError, "already loaded"):
            self.runtime.prepare_model(MODEL_ID)
        self.assertEqual(self.runtime.unload_model(api_model)[0].status, "unloaded")
        self.assertEqual(self.api.mutations, ["load", "unload"])

    def test_explicit_request_values_are_not_reported_as_server_defaults(self):
        self.config.request = RequestSettings(use_mlx_serve_defaults=False, temperature=0.1, max_tokens=64, top_k=20)
        runtime = build_provider_runtime(self.config)
        _, loaded = runtime.prepare_model(MODEL_ID)
        self.assertNotIn("sampling", loaded["reported_inference"])
        self.assertEqual(loaded["inference_defaults"], {"temperature": 0.8, "top_p": 0.95, "top_k": 0, "max_tokens": 4096})
        client = runtime.chat_client()
        client(api_base="http://other.example/v1", model=MODEL_ID, prompt_text="hi", timeout_sec=5,
               now_fn=lambda: 0.0, min_p=0.5, seed=3, temperature=0.9)
        body = json.loads(self.api.requests[-1].data)
        self.assertEqual({key: body[key] for key in ("temperature", "max_tokens", "top_k")}, {"temperature": 0.1, "max_tokens": 64, "top_k": 20})
        self.assertTrue({"min_p", "seed", "top_p"}.isdisjoint(body))
        self.assertEqual(self.api.requests[-1].full_url, "http://127.0.0.1:11234/v1/chat/completions")

    def test_busy_missing_wrong_owner_peer_and_ambiguous_fail_before_mutation(self):
        for key in ("requests_running", "requests_waiting"):
            for value in (1, None, -1):
                with self.subTest(key=key, value=value), patch.dict(self.api.gauges, {key: value}):
                    with patch("local_llm_bench.mlx_serve.time.monotonic", side_effect=range(0, 10000, 60)), self.assertRaises(RuntimeError):
                        self.runtime.unload_model(MODEL_ID)
        for key, value in (("loaded", None), ("owned_by", "peer"), ("capabilities", ["embeddings"]), ("error", "broken")):
            with self.subTest(key=key), patch.dict(self.api.model, {key: value}):
                with self.assertRaises(RuntimeError):
                    self.runtime.prepare_model(MODEL_ID)
        with patch.dict(self.api.model, {"loaded": True, "state": "loading"}), self.assertRaisesRegex(RuntimeError, "loading"):
            self.runtime.inspect_model(MODEL_ID)
        with self.assertRaises(RuntimeError):
            self.runtime.inspect_model("org/model-b")
        with self.assertRaises(ValueError):
            self.runtime.inspect_model(MODEL_ID + "@peer")
        self.api.inventory.append(copy.deepcopy(self.api.model))
        with self.assertRaises(RuntimeError):
            self.runtime.inspect_model(MODEL_ID)
        self.assertEqual(self.api.mutations, [])

    def test_unload_waits_for_stream_cleanup_without_repeating_inference(self):
        self.api.model.update(loaded=True, state="ready")
        self.api.gauges.update(requests_running=1)
        clock = [0.0]

        def advance(seconds):
            self.assertEqual(self.api.mutations, [])
            clock[0] += seconds
            # Running and queued requests must both drain before unloading.
            self.api.gauges.update(requests_running=0, requests_waiting=1 if clock[0] < 0.5 else 0)

        with patch("local_llm_bench.mlx_serve.time.monotonic", side_effect=lambda: clock[0]), \
                patch("local_llm_bench.mlx_serve.time.sleep", side_effect=advance) as sleep:
            results = self.runtime.unload_model(MODEL_ID)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(results[0].status, "unloaded")
        self.assertEqual(self.api.mutations, ["unload"])
        self.assertFalse(any(urlsplit(r.full_url).path == "/v1/chat/completions" for r in self.api.requests))

    def test_unload_idle_wait_is_bounded_and_rejects_invalid_metrics(self):
        self.api.model.update(loaded=True, state="ready")
        for key in ("requests_running", "requests_waiting"):
            clock = [0.0]
            with self.subTest(key=key), patch.dict(self.api.gauges, {key: 1}), \
                    patch("local_llm_bench.mlx_serve.time.monotonic", side_effect=lambda: clock[0]), \
                    patch("local_llm_bench.mlx_serve.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
                with self.assertRaisesRegex(RuntimeError, "idle wait timed out after 30s"):
                    self.runtime.unload_model(MODEL_ID)
                self.assertEqual(clock[0], 30.0)
        with patch.dict(self.api.gauges, {"requests_running": 1, "requests_waiting": None}), \
                patch("local_llm_bench.mlx_serve.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "did not report requests_waiting"):
                self.runtime.unload_model(MODEL_ID)
            sleep.assert_not_called()
        self.assertEqual(self.api.mutations, [])

    def test_preflight_still_rejects_busy_server_immediately(self):
        self.api.gauges.update(requests_running=1)
        with patch("local_llm_bench.mlx_serve.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "mlx-serve is busy"):
                self.runtime.inspect_model(MODEL_ID)
            sleep.assert_not_called()
        self.assertEqual(self.api.mutations, [])

    def test_metrics_disabled_records_unknown_activity_without_blocking(self):
        self.api.metrics_enabled = False
        info = self.runtime.inspect_model(MODEL_ID)
        self.assertIsNone(info["server_activity"])
        _, loaded = self.runtime.prepare_model(MODEL_ID)
        self.assertEqual(loaded["state"], "loaded")
        self.assertEqual(self.api.mutations, ["load"])

    def test_model_dirs_resolution_is_local_only_and_unambiguous(self):
        self.assertEqual(resolve_local_model_path(MODEL_ID, [str(self.model_dir)], "http://inference.example:11234/v1"), "")
        self.assertEqual(resolve_local_model_path(MODEL_ID, [], self.config.api_base), "")
        self.assertEqual(resolve_local_model_path("org/missing", [str(self.model_dir)], self.config.api_base), "")
        other = self.root / "other" / MODEL_ID
        other.mkdir(parents=True)
        (other / "config.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "model_dirs"):
            resolve_local_model_path(MODEL_ID, [str(self.model_dir), str(self.root / "other")], self.config.api_base)
        gguf = self.model_dir / "weights.gguf"
        gguf.write_bytes(b"GGUF")
        self.assertEqual(resolve_local_model_path("weights.gguf", [str(self.model_dir)], self.config.api_base), str(gguf))
        self.config.mlx_serve = None
        self.assertEqual(build_provider_runtime(self.config).session.model_dirs, normalize_mlx_serve_config(None, directory=None)["model_dirs"])
        unresolved = MLXServeSession(self.config.api_base, model_dirs=[]).model_info(MODEL_ID)
        self.assertEqual(unresolved["path"], "")
        self.assertEqual(unresolved["local_artifact_identity_status"], "unavailable_from_api")

    def test_mutation_requires_confirmed_state(self):
        self.api.accept_mutation = False
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            self.runtime.prepare_model(MODEL_ID)
        self.assertEqual(self.api.mutations, ["load"])

    def test_authentication_error_is_actionable_and_never_retries_another_key(self):
        for key, expected in ((None, "送信されていません"), ("fixture-rejected-key", "受け付けられません")):
            with self.subTest(present=bool(key)):
                error = urllib.error.HTTPError(self.config.api_base, 401, "fixture-rejected-key", {}, io.BytesIO(b""))
                with patch("local_llm_bench.mlx_serve.open_url_no_redirect", side_effect=error) as opened:
                    with self.assertRaisesRegex(RuntimeError, expected) as caught:
                        MLXServeSession(self.config.api_base, key).request_json("/health")
                    self.assertIn("MLX_SERVE_API_KEY", str(caught.exception))
                    self.assertNotIn("fixture-rejected-key", str(caught.exception))
                    self.assertEqual(opened.call_count, 1)

    def test_transport_rejects_redirects_wrong_origin_and_redacts_errors(self):
        session = MLXServeSession(self.config.api_base, "fixture-secret")
        for target in ("http://other.example/v1/chat/completions", self.config.api_base + "/models"):
            with self.assertRaises(HTTPBoundaryError):
                session.urlopen(urllib.request.Request(target, data=b"{}"), timeout=1)
        for path, method, body in (("/v1/models/rescan", "POST", {}), ("/v1/load-model", "GET", None),
                                   ("/v1/load-model", "POST", None), ("/admin", "GET", None)):
            with self.assertRaises(HTTPBoundaryError):
                session.request_json(path, method=method, body=body)
        self.opened.assert_not_called()
        for code in (302, 401, 404, 409, 503):
            error = urllib.error.HTTPError("http://hidden/fixture-secret", code, "fixture-secret", {}, io.BytesIO(b"fixture-secret"))
            with patch("local_llm_bench.mlx_serve.open_url_no_redirect", side_effect=error) as opened:
                with self.assertRaises(RuntimeError) as caught:
                    session.request_json("/health")
                self.assertIn(str(code), str(caught.exception))
                self.assertNotIn("fixture-secret", str(caught.exception))
                self.assertEqual(opened.call_count, 1)
                self.assertEqual(opened.call_args.args[0].get_header("Authorization"), "Bearer fixture-secret")

    def test_model_load_reports_memory_error_without_retry_or_response_secrets(self):
        for field in ("code", "type"):
            with self.subTest(field=field):
                body = reply({"error": {field: "out_of_memory", "message": "fixture-secret /private/model"}})
                error = urllib.error.HTTPError(self.config.api_base, 503, "fixture-secret", body.headers, body)

                def fail_load(request, *, timeout_sec):
                    if urlsplit(request.full_url).path == "/v1/load-model":
                        raise error
                    return self.api(request, timeout_sec=timeout_sec)

                with patch("local_llm_bench.mlx_serve.open_url_no_redirect", side_effect=fail_load) as opened:
                    with self.assertRaisesRegex(RuntimeError, "out_of_memory.*メモリが不足") as caught:
                        self.runtime.prepare_model(MODEL_ID)
                self.assertEqual(caught.exception.status, 503)
                self.assertNotIn("fixture-secret", str(caught.exception))
                self.assertNotIn("/private/model", str(caught.exception))
                posts = [call.args[0] for call in opened.call_args_list if call.args[0].get_method() == "POST"]
                self.assertEqual([urlsplit(request.full_url).path for request in posts], ["/v1/load-model"])
                self.assertTrue(body.closed)
                self.assertFalse(self.api.model["loaded"])

    def test_load_error_fallback_preserves_status_and_bounds_untrusted_body(self):
        for raw, headers in (
            (b"fixture-secret", {"Content-Type": "text/plain"}),
            (b"{", {"Content-Type": "application/json"}),
            (b'{"error":{"type":{"secret":"fixture-secret"}}}', {"Content-Type": "application/json"}),
            (b'{"error":{"type":"fixture-secret","message":"out_of_memory"}}', {"Content-Type": "application/json"}),
            (b'{"error":{"type":"out_of_memory","type":"fixture-secret"}}', {"Content-Type": "application/json"}),
            (b" " * (MAX_HTTP_ERROR_BYTES + 1), {"Content-Type": "application/json"}),
            (b"fixture-secret", {"Content-Type": "application/json", "Content-Length": str(2**40)}),
        ):
            body = io.BytesIO(raw)
            error = urllib.error.HTTPError(self.config.api_base, 503, "fixture-secret", headers, body)
            with self.subTest(raw=raw[:80], headers=headers), \
                    patch("local_llm_bench.mlx_serve.open_url_no_redirect", side_effect=error) as opened:
                with self.assertRaisesRegex(RuntimeError, "HTTP 503.*サーバーログ") as caught:
                    self.runtime.session.request_json("/v1/load-model", method="POST", body={"model": MODEL_ID})
                self.assertEqual(caught.exception.status, 503)
                self.assertNotIn("fixture-secret", str(caught.exception))
                self.assertNotIn("out_of_memory:", str(caught.exception))
                self.assertTrue(body.closed)
                self.assertEqual(opened.call_count, 1)

    def test_error_body_read_failure_does_not_hide_http_status(self):
        body = reply({})
        error = urllib.error.HTTPError(self.config.api_base, 503, "fixture-secret", body.headers, body)
        with patch.object(error, "read", side_effect=http.client.IncompleteRead(b"fixture-secret")), \
                patch("local_llm_bench.mlx_serve.open_url_no_redirect", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503.*サーバーログ") as caught:
                self.runtime.session.request_json("/v1/load-model", method="POST", body={"model": MODEL_ID})
        self.assertEqual(caught.exception.status, 503)
        self.assertNotIn("fixture-secret", str(caught.exception))
        self.assertTrue(body.closed)

    def test_stream_transport_uses_only_configured_credentials_and_bounds_responses(self):
        session = MLXServeSession(self.config.api_base, "fixture-secret")
        request = urllib.request.Request(self.config.api_base + "/chat/completions", data=b"{}",
                                         headers={"Authorization": "Bearer unrelated-key"})
        response = io.BytesIO(b"data: {}\n\ndata: [DONE]\n\n")
        response.headers = {"Content-Type": "text/event-stream"}
        with patch("local_llm_bench.mlx_serve.open_url_no_redirect", return_value=response) as opened:
            with session.urlopen(request, timeout=1) as stream:
                self.assertEqual(list(stream)[0], b"data: {}\n")
            self.assertTrue(response.closed)
            self.assertEqual(opened.call_args.args[0].get_header("Authorization"), "Bearer fixture-secret")
        oversized = io.BytesIO(b"")
        oversized.headers = {"Content-Type": "text/event-stream", "Content-Length": str(2**40)}
        with patch("local_llm_bench.mlx_serve.open_url_no_redirect", return_value=oversized):
            with self.assertRaises(HTTPBoundaryError):
                session.urlopen(request, timeout=1)
        self.assertTrue(oversized.closed)
        invalid = io.BytesIO(b'{"status":"ok","status":"bad"}')
        invalid.headers = {"Content-Type": "application/json"}
        with patch("local_llm_bench.mlx_serve.open_url_no_redirect", return_value=invalid):
            with self.assertRaises(ValueError):
                session.request_json("/health")
        self.assertTrue(invalid.closed)

    def test_inspect_prompt_lifecycle_and_saved_policy_for_cold_warm_and_primer(self):
        for cold in (2, 0):
            with self.subTest(cold=cold), contextlib.redirect_stdout(io.StringIO()):
                self.config.runs.cold_runs = cold
                start = len(self.api.requests)
                ex = RunExecution(self.config, self.runtime, MODEL_ID)
                try:
                    ex.start({"host": {}, "docker": None})
                    result = run_benchmark(self.config, model=ex.api_model, client=self.runtime.chat_client(), execution=ex)
                    ex.finish("completed")
                finally:
                    ex.close()
                enriched = ex.enrich(result)
                conditions = enriched["conditions"]
                self.assertEqual(conditions["provider"], "mlx_serve")
                self.assertEqual(conditions["model"]["artifact"]["status"], "measured")
                self.assertEqual(conditions["runtime"]["engine"], "mlx-serve")
                self.assertEqual(conditions["sampling"]["value"], {"temperature": 0.8, "top_p": 0.95, "top_k": 0})
                unknown = unknown_conditions(conditions)
                self.assertNotIn("sampling", unknown)
                self.assertNotIn("speculative_decoding", unknown)
                self.assertIn("thinking", unknown)
                self.assertEqual(conditions["cache"]["state"], "unknown")
                self.assertEqual(enriched["request"], {"settings_source": "mlx_serve_saved"})
                self.assertTrue(all(record["status"] == "success" for record in result["records"]))
                requests = [r for r in self.api.requests[start:] if r.full_url.endswith("/chat/completions")]
                self.assertEqual(len(requests), cold + 1 if cold else 2)
                for request in requests:
                    body = json.loads(request.data)
                    self.assertTrue((MLX_SERVE_REQUEST_KEYS | {"min_p", "seed"}).isdisjoint(body), body)
                    self.assertEqual(body["model"], MODEL_ID)
                    self.assertTrue(body["stream"])


if __name__ == "__main__":
    unittest.main()
