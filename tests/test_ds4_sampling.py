"""Offline regressions for RevBench's DS4 model policies and GLM alias lookup."""
import asyncio
import contextlib
import copy
import io
import itertools
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_llm_bench import ds4
from local_llm_bench.conditions import capture_conditions, execution_contract, unknown_conditions
from local_llm_bench.config import RequestSettings, load_config
from local_llm_bench.docker_task import container_worker
from local_llm_bench.docker_task.runner import _write_request_payload
from local_llm_bench.docker_task.spec import Question
from local_llm_bench.http_boundary import HTTPBoundaryError
from local_llm_bench.provider_runtime import build_provider_runtime
from local_llm_bench.runner import run_benchmark
from test_ds4 import _OfflineTestCase, _Response, _models_response, ds4_config, completion, BASE, MODEL_INFO, REPO

GLM = "glm-5.3-flash"
GLM_POLICY = {"source": "model_config", "temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0}
RUN_SAMPLING = {"temperature": 0.0, "top_p": 0.8, "top_k": 17, "min_p": 0.1, "seed": 9}
POLICIES = [
    (None, {}),
    ({"source": "server_defaults"}, {}),
    (GLM_POLICY, {key: value for key, value in GLM_POLICY.items() if key != "source"}),
    ({"source": "model_config", "temperature": 0.7, "seed": 0}, {"temperature": 0.7, "seed": 0}),
    ({"source": "run_config"}, RUN_SAMPLING),
]


class DS4SamplingTests(_OfflineTestCase):
    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = ds4_config(self.root)
        self.config.request = RequestSettings(**RUN_SAMPLING, max_tokens=128, reasoning_effort="high")

    def assert_sampler(self, body, expected):
        self.assertEqual({key: body[key] for key in ds4.DS4_SAMPLING_KEYS if key in body}, expected)
        self.assertEqual(body["reasoning_effort"], "high")
        for key in ("source", "sampling_source", "ds4_sampling", "ds4", "extra_body"):
            self.assertNotIn(key, body)

    def test_policy_validation_matches_revbench_ranges_and_zero_values(self):
        for policy, _ in POLICIES:
            self.assertEqual(ds4.normalize_ds4_sampling_config(policy), policy or {"source": "server_defaults"})
        maximum = {"source": "model_config", "temperature": 2, "top_p": 1, "top_k": 1024, "min_p": 1, "seed": 2**53 - 1}
        self.assertEqual(ds4.normalize_ds4_sampling_config(maximum), maximum)
        invalid = [{}, [], {"source": []}, {"source": "auto"}, {"source": "model_config"},
                   {"source": "model_config", "unknown": 1}, {"source": "server_defaults", "temperature": 1},
                   {"source": "run_config", "top_p": 0.95}]
        for key, values in {
            "temperature": [True, None, "1", -0.1, 2.1, float("nan"), float("inf")],
            "top_p": [0, 1.1, True], "top_k": [0.5, -1, 1025, True],
            "min_p": [-0.1, 1.1, True], "seed": [-1, 2**53, 1.5, True],
        }.items():
            invalid.extend({"source": "model_config", key: value} for value in values)
        for policy in invalid:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                ds4.normalize_ds4_sampling_config(policy)

    def test_config_is_ds4_only_and_validates_before_runtime_access(self):
        path = self.root / "bench.yaml"
        base = {"provider": "ds4", "models": [GLM], "ds4": {"management": "external"}}
        for override in (
            {"ds4_sampling": None}, {"ds4_sampling": GLM_POLICY, "provider": "lmstudio", "ds4": None},
            {"ds4_sampling": {"source": "run_config"}, "request": {"seed": True}},
            {"ds4_sampling": {"source": "run_config"}, "request": {"temperature": True}},
            {"ds4_sampling": {"source": "model_config", "min_p": -1}},
        ):
            document = {**base, **override}
            if document.get("ds4") is None:
                document.pop("ds4", None)
            path.write_text(json.dumps(document))
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(path)
        path.write_text(json.dumps({**base, "ds4_sampling": GLM_POLICY, "request": {"max_tokens": 8192, "reasoning_effort": "high"}}))
        loaded = load_config(path, cli_temperature=0.2)
        self.assertEqual(loaded.request.temperature, 0.2)
        self.assertEqual(loaded.request_parameters()["temperature"], 1.0)
        path.write_text(json.dumps({**base, "ds4_sampling": {"source": "server_defaults"}}))
        self.assertEqual(loaded.ds4_sampling, GLM_POLICY)  # No mid-run config-file reread.
        self.assertNotIn("temperature", load_config(path).request_parameters())
        path.write_text(json.dumps({**base, "ds4_sampling": {"source": "run_config"}, "request": {"top_k": 0, "min_p": 0, "seed": 0}}))
        self.assertEqual(load_config(path).request_parameters(), {"temperature": 0.0, "max_tokens": 512, "top_k": 0, "min_p": 0, "seed": 0})

    def test_presets_pin_distinct_weights_and_model_policies(self):
        for filename, model, weight, policy in (
            ("bench_ds4.yaml", "deepseek-v4-flash", "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf", {"source": "server_defaults"}),
            ("bench_ds4_glm_5_3_flash.yaml", GLM, "GLM-5.3-Flash-Q2.gguf", GLM_POLICY),
        ):
            for candidate in (filename, filename.replace("bench_", "bench_d_compile_arm64_", 1)):
                with self.subTest(filename=candidate):
                    config = load_config(REPO / "configs" / candidate)
                    self.assertEqual(config.models, [model])
                    self.assertEqual(Path(config.ds4["model_path"]).name, weight)
                    self.assertEqual(config.ds4_sampling, policy)
                    self.assertEqual(config.ds4["context_length"], 32768)
                    self.assertEqual(config.request_parameters()["max_tokens"], 8192)
                    self.assertEqual(config.request_parameters()["reasoning_effort"], "high")

    def test_stream_keeps_each_policy_without_continuation(self):
        for policy, expected in POLICIES:
            with self.subTest(policy=policy):
                self.config.ds4_sampling = policy
                runtime = build_provider_runtime(self.config)
                replies = [completion("part", finish="length"), completion("done")]
                with patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=replies) as opened:
                    result = runtime.chat_client()(
                        api_base=BASE, model=GLM, prompt_text="hello", **self.config.request.api_parameters(),
                        timeout_sec=10, now_fn=itertools.count().__next__,
                    )
                bodies = [json.loads(call.args[0].data) for call in opened.call_args_list]
                self.assertEqual([body["stream"] for body in bodies], [True])
                self.assertEqual([body["max_tokens"] for body in bodies], [128])
                self.assertEqual(result.response_text, "part")
                self.assertEqual(result.finish_reason, "length")
                for call, body in zip(opened.call_args_list, bodies):
                    self.assert_sampler(body, expected)
                    self.assertEqual(call.args[0].get_header("Authorization"), "Bearer private-fixture-key")
                self.assertEqual(result.reasoning_text, "checked")

    def test_warmup_and_prompt_logs_use_resolved_values(self):
        self.config.ds4_sampling = GLM_POLICY
        self.config.runs.warm_runs = 1
        runtime = build_provider_runtime(self.config)
        execution = MagicMock(run_id="fixture", started_at="2026-09-12T00:00:00+00:00")
        execution.cached.return_value = None

        def prepare(phase, iteration, warmup):
            warmup(GLM)
            return GLM

        execution.before_attempt.side_effect = prepare
        with patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=[completion(), completion()]) as opened, contextlib.redirect_stdout(io.StringIO()):
            run = run_benchmark(self.config, client=runtime.chat_client(), execution=execution)
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(run["records"][0]["status"], "success")
        expected = {key: value for key, value in GLM_POLICY.items() if key != "source"}
        for call in opened.call_args_list:
            self.assert_sampler(json.loads(call.args[0].data), expected)
        self.assertEqual(run["request"], {**expected, "max_tokens": 128, "reasoning_effort": "high", "sampling_source": "ds4_model_config"})
        self.assertEqual(execution.save_unit.call_args.args[4]["request"]["min_p"], 0.0)

    def test_contract_ignores_unused_run_controls_but_detects_policy_and_budget_changes(self):
        for policy, expected in POLICIES:
            with self.subTest(policy=policy):
                self.config.ds4_sampling = copy.deepcopy(policy)
                contract = execution_contract(self.config, GLM, None)
                changed = copy.deepcopy(self.config)
                changed.request.temperature = 1.7
                ignored = policy is None or policy["source"] != "run_config"
                self.assertEqual(contract == execution_contract(changed, GLM, None), ignored)
                changed = copy.deepcopy(self.config)
                changed.request.max_tokens = 64
                self.assertNotEqual(contract, execution_contract(changed, GLM, None))
                changed.ds4_sampling = {"source": "model_config", "min_p": 0.3}
                changed.request.max_tokens = 128
                self.assertNotEqual(contract, execution_contract(changed, GLM, None))
                conditions = capture_conditions(self.config, {}, contract, {})
                self.assertEqual({key: conditions["request"][key] for key in expected}, expected)
                self.assertEqual(conditions["sampling"], {"value": None, "source": "unavailable"})
                self.assertIn("sampling", unknown_conditions(conditions))

    def test_docker_frozen_policy_reaches_every_tool_turn(self):
        tool = types.SimpleNamespace(name="lookup", description="lookup", inputSchema={"type": "object", "properties": {}})

        class ToolSession:
            async def call_tool(self, name, arguments):
                return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="found")])

        async def open_mcp(**kwargs):
            return contextlib.nullcontext(), ToolSession(), types.SimpleNamespace(tools=[tool])

        for policy, expected in POLICIES:
            with self.subTest(policy=policy):
                self.config.ds4_sampling = copy.deepcopy(policy)
                payload_path = _write_request_payload(
                    bundle_dir=self.root, config=self.config, selected_model=GLM,
                    question=Question(id="q1", prompt="check", answer_type="exact", gold_answer="ok"), staged_binary_ref=None,
                )
                payload = json.loads(payload_path.read_text())
                self.config.ds4_sampling = {"source": "model_config", "temperature": 1.9}
                self.assertEqual(payload["ds4_sampling"], policy or {"source": "server_defaults"})
                self.assertNotIn("server_path", payload)
                payload.update(RUN_SAMPLING)  # Model/server policy must ignore stale run defaults.
                calls = [{"id": "call_fixture", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
                with (
                    patch.object(container_worker, "_open_mcp_stdio_session", side_effect=open_mcp),
                    patch.object(container_worker, "resolve_native_binary_target", return_value=types.SimpleNamespace(path=None)),
                    patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=[completion("", finish="tool_calls", tool_calls=calls), completion("FINAL_ANSWER: ok")]) as opened,
                ):
                    result = asyncio.run(container_worker._run_question(payload))
                self.assertEqual(result["status"], "success", result.get("error"))
                self.assertEqual(result["predicted_answer"], "ok")
                self.assertEqual(opened.call_count, 2)
                for call, turn in zip(opened.call_args_list, result["trace"]["turns"]):
                    self.assert_sampler(json.loads(call.args[0].data), expected)
                    self.assertEqual(turn["request"]["sampling_source"], "ds4_" + (policy or {"source": "server_defaults"})["source"])
                    self.assertEqual({key: turn["request"][key] for key in ds4.DS4_SAMPLING_KEYS if key in turn["request"]}, expected)


class DS4GLMAliasTests(_OfflineTestCase):
    def test_individual_endpoint_is_used_only_for_known_glm_list_gap(self):
        glm = {**MODEL_INFO, "id": GLM}
        for listed in (glm, {**glm, "id": "glm-5.2"}):
            replies = [_models_response(listed)]
            if listed["id"] != GLM:
                replies.append(_Response(json.dumps(glm).encode()))
            with self.subTest(listed=listed["id"]), patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=replies) as opened:
                self.assertEqual(ds4.ds4_context_length("openai/" + GLM, base_url=BASE, api_key="fixture-key"), 32768)
            self.assertEqual([call.args[0].full_url for call in opened.call_args_list],
                             [BASE + "/models"] + ([BASE + "/models/" + GLM] if listed["id"] != GLM else []))
            for call in opened.call_args_list:
                self.assertEqual(call.args[0].method, "GET")
                self.assertIsNone(call.args[0].data)
                self.assertEqual(call.args[0].get_header("Authorization"), "Bearer fixture-key")

    def test_direct_response_must_match_exactly_and_pass_json_boundary(self):
        listed = {**MODEL_INFO, "id": "glm-5.2"}
        for direct in ({**listed}, {"data": [{**listed, "id": GLM}]}, None):
            with self.subTest(direct=direct), patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=[_models_response(listed), _Response(json.dumps(direct).encode())]), self.assertRaises(ds4.DS4ModelNotFoundError):
                ds4.read_ds4_model_info(GLM)
        for body in (b'{"id":"glm-5.3-flash","id":"glm-5.2"}', b'NaN'):
            with self.subTest(body=body), patch("local_llm_bench.ds4.open_url_no_redirect", side_effect=[_models_response(listed), _Response(body)]), self.assertRaises(HTTPBoundaryError):
                ds4.read_ds4_model_info(GLM)
        with patch("local_llm_bench.ds4.open_url_no_redirect", return_value=_models_response(MODEL_INFO)) as opened, self.assertRaises(ds4.DS4ModelNotFoundError):
            ds4.read_ds4_model_info(GLM)
        opened.assert_called_once()
