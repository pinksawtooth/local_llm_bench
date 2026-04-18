from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from local_llm_bench.config import (
    DEFAULT_DOCKER_UNSLOTH_STUDIO_API_BASE,
    DEFAULT_UNSLOTH_STUDIO_API_BASE,
    load_config,
)


class ConfigTests(unittest.TestCase):
    def test_load_config_resolves_relative_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "bench.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    api_base: "http://localhost:1234/v1"
                    models: ["model-a"]
                    prompt:
                      text: "hello"
                    request:
                      temperature: 0.1
                      max_tokens: 128
                    runs:
                      cold_runs: 1
                      warm_runs: 2
                      timeout_sec: 30
                    output:
                      history_json: "runs/history.json"
                      latest_json: "runs/latest.json"
                      report_html: "docs/index.html"
                    """
                ),
                encoding="utf-8",
            )
            loaded = load_config(config_path)

        self.assertEqual(loaded.models, ["model-a"])
        self.assertEqual(loaded.prompt_text, "hello")
        self.assertEqual(loaded.output.history_json.name, "history.json")
        self.assertTrue(str(loaded.output.report_html).endswith("docs/index.html"))
        self.assertTrue(str(loaded.output.run_logs_dir).endswith("runs/logs"))

    def test_cli_models_override_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "api_base: http://localhost:1234/v1\nmodels: [old]\n",
                encoding="utf-8",
            )
            loaded = load_config(config_path, cli_models=["new-a", "new-b"])

        self.assertEqual(loaded.models, ["new-a", "new-b"])

    def test_cli_prompt_text_overrides_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "bench.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    api_base: "http://localhost:1234/v1"
                    models: ["model-a"]
                    prompt:
                      text: "old prompt"
                    """
                ),
                encoding="utf-8",
            )
            loaded = load_config(config_path, cli_prompt_text="new prompt")

        self.assertEqual(loaded.prompt_text, "new prompt")

    def test_load_config_supports_docker_task_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "benchmarks").mkdir()
            (root / "benchmarks" / "spec.yaml").write_text("id: demo\nquestions: [{id: q1, prompt: p, answer_type: exact}]\n", encoding="utf-8")
            config_path = root / "bench.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    mode: docker_task
                    api_base: http://localhost:1234/v1
                    models: ["model-a"]
                    benchmark:
                      spec: "benchmarks/spec.yaml"
                      question_timeout_sec: 45
                      ghidra_tool_mode: decompile-only
                    docker:
                      image: "local-llm-bench:bench"
                      platform: "linux/arm64"
                      lmstudio_base_url: "http://host.docker.internal:1234/v1"
                    """
                ),
                encoding="utf-8",
            )

            loaded = load_config(config_path)

        self.assertEqual(loaded.mode, "docker_task")
        self.assertEqual(loaded.prompt_text, "")
        self.assertEqual(loaded.benchmark_question_timeout_sec, 45.0)
        self.assertEqual(loaded.benchmark_ghidra_tool_mode, "decompile-only")
        self.assertEqual(loaded.docker_image, "local-llm-bench:bench")
        self.assertEqual(loaded.docker_platform, "linux/arm64")
        self.assertTrue(str(loaded.benchmark_spec_path).endswith("benchmarks/spec.yaml"))

    def test_cli_out_dir_routes_logs_under_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "api_base: http://localhost:1234/v1\nmodels: [model-a]\n",
                encoding="utf-8",
            )
            out_dir = root / "out"

            loaded = load_config(config_path, cli_out_dir=out_dir)

        self.assertEqual(loaded.output.history_json, out_dir.resolve() / "history.json")
        self.assertEqual(loaded.output.latest_json, out_dir.resolve() / "latest_run.json")
        self.assertEqual(loaded.output.report_html, out_dir.resolve() / "index.html")
        self.assertEqual(loaded.output.run_logs_dir, out_dir.resolve() / "logs")

    def test_load_config_supports_unsloth_provider_with_fixed_endpoints_and_env_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {
                "UNSLOTH_STUDIO_BEARER_TOKEN": "token-from-env",
                "UNSLOTH_STUDIO_PASSWORD": "password-from-env",
            },
            clear=False,
        ):
            root = Path(tmpdir)
            config_path = root / "bench.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    provider: unsloth_studio
                    models: ["unsloth/gpt-oss-20b"]
                    auth:
                      username: "alice"
                    """
                ),
                encoding="utf-8",
            )

            loaded = load_config(config_path)

        self.assertEqual(loaded.provider, "unsloth_studio")
        self.assertEqual(loaded.api_base, DEFAULT_UNSLOTH_STUDIO_API_BASE)
        self.assertEqual(loaded.docker_api_base, DEFAULT_DOCKER_UNSLOTH_STUDIO_API_BASE)
        self.assertEqual(loaded.auth.bearer_token, "token-from-env")
        self.assertEqual(loaded.auth.username, "alice")
        self.assertEqual(loaded.auth.password, "password-from-env")

    def test_load_config_rejects_api_base_for_unsloth_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "bench.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    provider: unsloth_studio
                    api_base: http://127.0.0.1:9999/v1
                    models: ["unsloth/gpt-oss-20b"]
                    """
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "provider=unsloth_studio では api_base を指定できません"):
                load_config(config_path)
