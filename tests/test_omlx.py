"""oMLX API fixtures only: no native MLX import, sockets or model/server operation."""
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path
from unittest.mock import MagicMock, patch

from bench_fakes import make_config
from local_llm_bench.conditions import model_artifact, unknown_conditions
from local_llm_bench.config import RequestSettings, load_config
from local_llm_bench.execution import RunExecution
from local_llm_bench.http_boundary import HTTPBoundaryError
from local_llm_bench.omlx import OMLXSession, OMLX_REQUEST_KEYS, validate_omlx_request, resolve_omlx_api_key, _omlx_settings_path
from local_llm_bench.persistence import server_identity
from local_llm_bench.provider_runtime import OMLXProviderRuntime, build_provider_runtime
from local_llm_bench.runner import run_benchmark

REPO = Path(__file__).resolve().parents[1]


def reply(value):
    response = io.BytesIO(json.dumps(value).encode())
    response.headers = {"Content-Type": "application/json"}
    response.status = 200
    return response


class ModelAPI:
    def __init__(self, directory):
        self.status = {"status": "ok", "version": "fixture-0.7", "active_requests": 0,
                       "waiting_requests": 0, "models_loading": 0}
        self.model = {"id": "model-a", "model_path": str(directory), "loaded": False,
                      "is_loading": False, "pinned": False, "model_type": "llm", "distributed": False,
                      "model_alias": "chat-alias", "max_context_window": 8192, "max_tokens": 2048,
                      "thinking_default": True, "config_model_type": "qwen3_5"}
        self.models = [self.model]
        self.inventory = [{"id": "chat-alias", "owned_by": "omlx"}]
        self.requests = []
        self.accept_mutation = True

    def __call__(self, request, *, timeout_sec):
        self.requests.append(request)
        path = urlsplit(request.full_url).path
        if path == "/api/status":
            return reply(self.status)
        if path == "/v1/models/status":
            return reply({"models": self.models})
        if path == "/v1/models":
            return reply({"data": self.inventory})
        if path in {"/v1/models/model-a/load", "/v1/models/model-a/unload"}:
            if self.accept_mutation:
                self.model["loaded"] = path.endswith("/load")
            return reply({"status": "ok", "model_id": "model-a"})
        if path == "/v1/chat/completions":
            payload = {"choices": [{"delta": {"content": "fixture answer"}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
            response = io.BytesIO(("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode())
            response.headers = {"Content-Type": "text/event-stream"}
            return response
        raise AssertionError("Unexpected fixture request: " + path)

    @property
    def mutations(self):
        return [request.full_url.rsplit("/", 1)[-1] for request in self.requests
                if request.get_method() == "POST" and "/models/" in request.full_url]


class OMLXTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.settings_path = self.root / "settings.json"
        self.settings_lookup = self.enterContext(patch("local_llm_bench.omlx._omlx_settings_path", return_value=self.settings_path))
        self.enterContext(patch.dict(os.environ, {}))
        os.environ.pop("OMLX_API_KEY", None)
        os.environ.pop("OMLX_BASE_PATH", None)
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.socket.bind", "subprocess.Popen"):
            self.enterContext(patch(target, side_effect=AssertionError("Real runtime access forbidden")))
        for name in ("user_data_path", "user_cache_path"):
            self.enterContext(patch("inspect_ai._util.appdirs." + name, return_value=self.root / name))
        self.enterContext(patch("inspect_ai._eval.task.log.git_context", return_value=None))
        self.enterContext(patch("local_llm_bench.execution.host_snapshot", return_value={}))
        self.enterContext(patch("tempfile.tempdir", str(self.root)))
        self.directory = self.root / "model"
        self.directory.mkdir()
        (self.directory / "config.json").write_text('{"model_type":"fixture"}')
        (self.directory / "tokenizer.json").write_text('{"tokenizer":"fixture"}')
        (self.directory / "model.safetensors").write_bytes(b"tiny fixture weights")
        self.api = ModelAPI(self.directory)
        self.opened = self.enterContext(patch("local_llm_bench.omlx.open_url_no_redirect", side_effect=self.api))
        self.config = make_config(self.root, cold=2, warm=1)
        self.config.provider = "omlx"
        self.config.api_base = "http://127.0.0.1:8000/v1"
        self.config.docker_api_base = "http://host.docker.internal:8000/v1"
        self.config.request = RequestSettings(use_omlx_defaults=True)
        self.runtime = build_provider_runtime(self.config)

    def read_config(self, **values):
        path = self.root / "bench.yaml"
        path.write_text(json.dumps({"provider": "omlx", "models": ["model-a"], **values}))
        return load_config(path)

    def test_defaults_auth_presets_and_same_server_lock(self):
        with patch.dict("os.environ", {"OMLX_API_KEY": "fixture-key", "DS4_API_KEY": "other-key",
                                       "UNSLOTH_STUDIO_BEARER_TOKEN": "unrelated-key"}):
            config = self.read_config()
        self.assertEqual(config.auth.bearer_token, "fixture-key")
        self.assertEqual(config.request_parameters(), {})
        self.assertEqual(config.recorded_request_parameters(), {"settings_source": "omlx_saved"})
        self.assertNotIn("fixture-key", json.dumps(config.to_dict()))
        self.assertEqual(config.docker_api_base, "http://host.docker.internal:8000/v1")
        self.assertEqual(server_identity(config.api_base), server_identity(config.docker_api_base))
        self.assertEqual(server_identity(config.api_base), server_identity("http://localhost:8000/v1"))
        self.assertIsInstance(build_provider_runtime(config), OMLXProviderRuntime)
        self.assertEqual(build_provider_runtime(config).docker_environment(), {"OMLX_API_KEY": "fixture-key"})
        for name in ("bench_omlx.yaml", "bench_d_compile_arm64_omlx.yaml", "bench_mafc_arm64_omlx.yaml",
                     "bench.yaml", "bench_d_compile_arm64.yaml", "bench_lmstudio_4_models.yaml",
                     "bench_d_compile_arm64_lmstudio_4_models.yaml", "bench_mafc_arm64_lmstudio_4_models.yaml"):
            with self.subTest(name=name):
                preset = load_config(REPO / "configs" / name)
                self.assertEqual(preset.request_parameters(), {})
                self.assertFalse(preset.lmstudio_load.has_load_overrides())
                self.assertFalse(preset.lmstudio_load.parallelism_sweep)
        self.opened.assert_not_called()

    def test_saved_auth_is_selected_without_changing_settings_and_is_redacted(self):
        self.settings_path.write_text(json.dumps({"auth": {"api_key": "saved-fixture-key", "secret_key": "not-an-api-key"},
                                                 "server": {"host": "127.0.0.1", "port": 8000}}))
        original = self.settings_path.read_bytes()
        config = self.read_config()
        self.assertEqual(config.auth.bearer_token, "saved-fixture-key")
        self.assertEqual(config.request_parameters(), {})
        self.assertTrue(config.to_dict()["auth"]["bearer_token_present"])
        self.assertNotIn("saved-fixture-key", json.dumps(config.to_dict()))
        runtime = build_provider_runtime(config)
        self.assertEqual(runtime.docker_environment(), {"OMLX_API_KEY": "saved-fixture-key"})
        runtime.inspect_model("model-a")
        self.assertEqual(self.api.mutations, [])
        for request in self.api.requests:
            self.assertEqual(request.get_header("Authorization"), "Bearer saved-fixture-key")
        self.assertEqual(self.settings_path.read_bytes(), original)

    def test_explicit_key_and_environment_win_without_reading_saved_credentials(self):
        self.settings_path.write_text("invalid settings")
        with patch.dict(os.environ, {"OMLX_API_KEY": "environment-fixture-key"}):
            self.assertEqual(self.read_config().auth.bearer_token, "environment-fixture-key")
            self.assertEqual(self.read_config(auth={"bearer_token": "explicit-fixture-key"}).auth.bearer_token, "explicit-fixture-key")
        self.settings_lookup.assert_not_called()
        self.opened.assert_not_called()

    def test_saved_key_is_not_sent_to_another_endpoint(self):
        self.settings_path.write_text(json.dumps({"auth": {"api_key": "saved-fixture-key"},
                                                 "server": {"host": "0.0.0.0", "port": 8000}}))
        for url in ("http://example.com:8000/v1", "http://host.docker.internal:8000/v1", "https://127.0.0.1:8000/v1"):
            with self.subTest(url=url):
                self.assertIsNone(resolve_omlx_api_key(url))
        self.settings_lookup.assert_not_called()
        self.assertIsNone(resolve_omlx_api_key("http://127.0.0.1:8001/v1"))
        for host in ("localhost", "127.0.0.1", "[::1]"):
            self.assertEqual(resolve_omlx_api_key(f"http://{host}:8000/v1"), "saved-fixture-key")
        self.opened.assert_not_called()

    def test_saved_auth_missing_invalid_and_unrelated_settings_are_safe(self):
        self.assertIsNone(resolve_omlx_api_key(self.config.api_base))
        for settings in ({"auth": {"secret_key": "not-api"}}, {"auth": {}},
                         {"server": {"host": "other.example", "port": 8000}, "auth": {"api_key": "saved-fixture-key"}}):
            self.settings_path.write_text(json.dumps(settings))
            self.assertIsNone(resolve_omlx_api_key(self.config.api_base))
        for raw in ('{"auth":{"api_key":"private-fixture-value",}}',
                    '{"auth":{"api_key":"private-fixture-value\\nextra"}}',
                    '{"auth": "private-fixture-value"}',
                    '{"server": {"port": "private-fixture-value"}}'):
            self.settings_path.write_text(raw)
            with self.assertRaisesRegex(ValueError, "OMLX_API_KEY") as caught:
                resolve_omlx_api_key(self.config.api_base)
            self.assertNotIn("private-fixture-value", str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)
        self.opened.assert_not_called()

    def test_data_root_matches_omlx_precedence(self):
        home = self.root / "home"
        bootstrap = home / "Library/Application Support/oMLX/base-path"
        with patch("pathlib.Path.home", return_value=home):
            self.assertEqual(_omlx_settings_path(), home / ".omlx/settings.json")
            bootstrap.parent.mkdir(parents=True)
            moved = self.root / "moved"
            bootstrap.write_text(str(moved) + "\n")
            self.assertEqual(_omlx_settings_path(), moved / "settings.json")
            override = self.root / "override"
            with patch.dict(os.environ, {"OMLX_BASE_PATH": str(override)}):
                self.assertEqual(_omlx_settings_path(), override / "settings.json")
        self.opened.assert_not_called()

    def test_authentication_error_is_actionable_and_never_retries_another_key(self):
        for key, expected in ((None, "送信されていません"), ("fixture-rejected-key", "受け付けられません")):
            with self.subTest(present=bool(key)):
                error = urllib.error.HTTPError(self.config.api_base, 401, "fixture-rejected-key", {}, io.BytesIO(b""))
                with patch("local_llm_bench.omlx.open_url_no_redirect", side_effect=error) as opened:
                    with self.assertRaisesRegex(RuntimeError, expected) as caught:
                        OMLXSession(self.config.api_base, key).request_json("/api/status")
                    self.assertIn("OMLX_API_KEY", str(caught.exception))
                    self.assertNotIn("fixture-rejected-key", str(caught.exception))
                    self.assertEqual(opened.call_count, 1)
        self.settings_lookup.assert_not_called()

    def test_invalid_config_fails_without_io(self):
        for fields in (
            {"request": {"temperature": 0}}, {"request": {"max_tokens": 200}},
            {"request": {"use_omlx_defaults": "true"}}, {"request": {"use_lmstudio_defaults": True}},
            {"request": {"use_omlx_defaults": False, "temperature": float("nan")}},
            {"request": {"use_omlx_defaults": False, "max_tokens": True}},
            {"api_base": "http://user:secret@localhost:8000/v1"}, {"api_base": "http://localhost:8000/"},
            {"api_base": "http://localhost:8000/v1?secret=x"},
            {"docker": {"api_base": "http://host.docker.internal:8001/v1"}},
            {"auth": {"username": "someone"}}, {"lmstudio": {"context_length": 4096}},
            {"request": {"unknown_setting": True}},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.read_config(**fields)
        with self.assertRaises(ValueError):
            self.read_config(provider="lmstudio", request={"use_omlx_defaults": True})
        for key, value in (("top_k", -1), ("top_p", 0), ("min_p", 2), ("seed", True), ("temperature", float("inf"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_omlx_request({key: value})
        config = self.read_config(request={"use_omlx_defaults": False, "temperature": .2, "top_k": 40, "min_p": .1, "seed": 1})
        self.assertEqual(config.request_parameters()["top_k"], 40)
        self.opened.assert_not_called()

    def test_exact_alias_identity_lifecycle_and_saved_settings(self):
        info = self.runtime.inspect_model("model-a")
        self.assertEqual(info["identifier"], "chat-alias")
        self.assertEqual(info["path"], str(self.directory))
        self.assertEqual(info["runtime"]["version"], "fixture-0.7")
        self.assertEqual(info["reported_inference"], {})
        self.assertEqual(self.api.mutations, [])
        self.assertEqual(self.runtime.unload_model("model-a"), [])
        api_model, loaded = self.runtime.prepare_model("model-a")
        self.assertEqual(api_model, "chat-alias")
        self.assertEqual(loaded["load_status"], "loaded")
        self.assertEqual(self.runtime.measurement_metadata(loaded)["state"], "loaded")
        with self.assertRaisesRegex(RuntimeError, "already loaded"):
            self.runtime.prepare_model("model-a")
        self.assertEqual(self.runtime.unload_model(api_model)[0].status, "unloaded")
        self.assertEqual(self.api.mutations, ["load", "unload"])
        for request in self.api.requests:
            if request.get_method() == "POST":
                self.assertEqual(request.data, b"")  # No generation/load config overrides.

    def test_busy_missing_wrong_provider_and_ambiguous_fail_before_mutation(self):
        for key in ("active_requests", "waiting_requests", "models_loading"):
            for value in (1, None):
                with self.subTest(key=key, value=value), patch.dict(self.api.status, {key: value}):
                    with self.assertRaises(RuntimeError):
                        self.runtime.unload_model("model-a")
        for key, value in (("is_loading", True), ("loaded", None),
                           ("model_type", "embedding"), ("distributed", True)):
            with self.subTest(key=key), patch.dict(self.api.model, {key: value}):
                with self.assertRaises(RuntimeError):
                    self.runtime.prepare_model("model-a")
        with self.assertRaises(RuntimeError):
            self.runtime.inspect_model("model")
        with patch.dict(self.api.inventory[0], {"owned_by": "ds4"}), self.assertRaises(RuntimeError):
            self.runtime.inspect_model("model-a")
        self.api.models.append(copy.deepcopy(self.api.model))
        with self.assertRaises(RuntimeError):
            self.runtime.inspect_model("model-a")
        self.assertEqual(self.api.mutations, [])

    def test_pinned_setting_is_preserved_across_model_reload(self):
        self.api.model["pinned"] = True
        self.api.model["loaded"] = True
        self.runtime.unload_model("model-a")
        _, info = self.runtime.prepare_model("model-a")
        self.assertIs(info["load_config"]["pinned"], True)
        self.assertIs(self.api.model["pinned"], True)
        self.assertEqual(self.api.mutations, ["unload", "load"])

    def test_remote_paths_not_hashed_and_profiles_use_physical_model(self):
        remote = OMLXSession("http://inference.example:8000/v1").model_info("model-a")
        self.assertEqual(remote["path"], "")
        self.assertEqual(remote["reported_model_path"], str(self.directory))
        profile = {**self.api.model, "id": "profile", "model_alias": None, "source_model_id": "model-a", "max_tokens": 1234}
        self.api.models.append(profile)
        self.api.inventory.append({"id": "profile", "owned_by": "omlx"})
        api_model, info = self.runtime.prepare_model("profile")
        self.assertEqual(api_model, "profile")
        self.assertEqual(info["model_key"], "model-a")
        self.assertEqual(info["inference_defaults"], {"max_tokens": 1234})

    def test_mutation_requires_confirmed_state(self):
        self.api.accept_mutation = False
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            self.runtime.prepare_model("model-a")
        self.assertEqual(self.api.mutations, ["load"])

    def test_transport_rejects_redirects_wrong_origin_and_redacts_errors(self):
        session = OMLXSession(self.config.api_base, "fixture-secret")
        for target in ("http://other.example/v1/chat/completions", self.config.api_base + "/models"):
            with self.assertRaises(HTTPBoundaryError):
                session.urlopen(urllib.request.Request(target, data=b"{}"), timeout=1)
        with self.assertRaises(HTTPBoundaryError):
            session.request_json("/admin/api/models", method="POST")
        self.opened.assert_not_called()
        for code in (302, 401, 404, 409, 503):
            error = urllib.error.HTTPError("http://hidden/fixture-secret", code, "fixture-secret", {}, io.BytesIO(b"fixture-secret"))
            with patch("local_llm_bench.omlx.open_url_no_redirect", side_effect=error) as opened:
                with self.assertRaises(RuntimeError) as caught:
                    session.request_json("/api/status")
                self.assertIn(str(code), str(caught.exception))
                self.assertNotIn("fixture-secret", str(caught.exception))
                self.assertEqual(opened.call_count, 1)
                self.assertEqual(opened.call_args.args[0].get_header("Authorization"), "Bearer fixture-secret")

    def test_stream_transport_uses_only_configured_credentials_and_bounds_responses(self):
        session = OMLXSession(self.config.api_base, "fixture-secret")
        request = urllib.request.Request(self.config.api_base + "/chat/completions", data=b"{}",
                                         headers={"Authorization": "Bearer unrelated-key"})
        response = io.BytesIO(b"data: {}\n\ndata: [DONE]\n\n")
        response.headers = {"Content-Type": "text/event-stream"}
        with patch("local_llm_bench.omlx.open_url_no_redirect", return_value=response) as opened:
            with session.urlopen(request, timeout=1) as stream:
                self.assertEqual(list(stream)[0], b"data: {}\n")
            self.assertTrue(response.closed)
            self.assertEqual(opened.call_args.args[0].get_header("Authorization"), "Bearer fixture-secret")
        oversized = io.BytesIO(b"")
        oversized.headers = {"Content-Type": "text/event-stream", "Content-Length": str(2**40)}
        with patch("local_llm_bench.omlx.open_url_no_redirect", return_value=oversized):
            with self.assertRaises(HTTPBoundaryError):
                session.urlopen(request, timeout=1)
        self.assertTrue(oversized.closed)
        invalid = io.BytesIO(b'{"status":"ok","status":"bad"}')
        invalid.headers = {"Content-Type": "application/json"}
        with patch("local_llm_bench.omlx.open_url_no_redirect", return_value=invalid):
            with self.assertRaises(ValueError):
                session.request_json("/api/status")
        self.assertTrue(invalid.closed)

    def test_mlx_fingerprint_changes_with_weights_tokenizer_and_settings(self):
        info = {"path": str(self.directory), "format": "MLX"}
        cache = self.root / "hashes"
        previous = model_artifact(info, cache)
        self.assertEqual(previous["status"], "measured")
        for name in ("model.safetensors", "tokenizer.json", "config.json"):
            path = self.directory / name
            path.write_bytes(path.read_bytes() + b" ")
            current = model_artifact(info, cache)
            self.assertNotEqual(current["sha256"], previous["sha256"])
            previous = current
        (self.directory / "README.md").write_text("not inference data")
        self.assertEqual(model_artifact(info, cache)["sha256"], previous["sha256"])
        index = self.directory / "model.safetensors.index.json"
        for name in ("missing.safetensors", "../model.safetensors"):
            index.write_text(json.dumps({"weight_map": {"x": name}}))
            with self.assertRaisesRegex(ValueError, "shard"):
                model_artifact(info, cache)

    def test_inspect_prompt_lifecycle_and_saved_policy_for_cold_warm_and_primer(self):
        for cold in (2, 0):
            with self.subTest(cold=cold), contextlib.redirect_stdout(io.StringIO()):
                self.config.runs.cold_runs = cold
                start = len(self.api.requests)
                ex = RunExecution(self.config, self.runtime, "model-a")
                try:
                    ex.start({"host": {}, "docker": None})
                    result = run_benchmark(self.config, model=ex.api_model, client=self.runtime.chat_client(), execution=ex)
                    ex.finish("completed")
                finally:
                    ex.close()
                enriched = ex.enrich(result)
                self.assertEqual(enriched["conditions"]["model"]["artifact"]["status"], "measured")
                self.assertIn("sampling", unknown_conditions(enriched["conditions"]))
                self.assertIn("speculative_decoding", unknown_conditions(enriched["conditions"]))
                self.assertEqual(enriched["conditions"]["cache"]["state"], "unknown")
                self.assertEqual(enriched["request"], {"settings_source": "omlx_saved"})
                self.assertTrue(all(record["status"] == "success" for record in result["records"]))
                requests = [r for r in self.api.requests[start:] if r.full_url.endswith("/chat/completions")]
                self.assertEqual(len(requests), cold + 1 if cold else 2)
                for request in requests:
                    body = json.loads(request.data)
                    self.assertTrue(OMLX_REQUEST_KEYS.isdisjoint(body), body)
                    self.assertEqual(body["model"], "chat-alias")


if __name__ == "__main__":
    unittest.main()
