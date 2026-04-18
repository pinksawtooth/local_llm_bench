from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

from local_llm_bench.config import BenchmarkConfig, OutputSettings, RequestSettings, RunSettings, AuthSettings
from local_llm_bench.provider_runtime import UnslothStudioProviderRuntime
from local_llm_bench.unsloth_api import UnslothStudioAuthSession


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.status = 200

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_config(root: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        api_base="http://127.0.0.1:8888/v1",
        models=["unsloth/gpt-oss-20b"],
        prompt_text="hello",
        request=RequestSettings(temperature=0.0, max_tokens=128),
        runs=RunSettings(cold_runs=1, warm_runs=1, timeout_sec=10.0, cooldown_sec=0.0),
        output=OutputSettings(
            history_json=root / "runs" / "history.json",
            latest_json=root / "runs" / "latest_run.json",
            report_html=root / "docs" / "index.html",
            run_logs_dir=root / "runs" / "logs",
        ),
        provider="unsloth_studio",
        auth=AuthSettings(bearer_token="preset-token"),
        docker_api_base="http://host.docker.internal:8888/v1",
    )


class UnslothStudioAuthSessionTests(unittest.TestCase):
    def test_urlopen_retries_after_401_by_refreshing_token(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_urlopen(request, timeout=None):
            headers = dict(request.header_items())
            authorization = headers.get("Authorization", "")
            calls.append((request.full_url, authorization))
            if request.full_url.endswith("/api/auth/login"):
                return _FakeResponse(
                    {
                        "access_token": "token-1",
                        "refresh_token": "refresh-1",
                        "token_type": "bearer",
                        "must_change_password": False,
                    }
                )
            if request.full_url.endswith("/api/auth/refresh"):
                return _FakeResponse(
                    {
                        "access_token": "token-2",
                        "refresh_token": "refresh-2",
                        "token_type": "bearer",
                        "must_change_password": False,
                    }
                )
            if request.full_url.endswith("/v1/models"):
                if authorization == "Bearer token-1":
                    raise urllib.error.HTTPError(
                        request.full_url,
                        401,
                        "Unauthorized",
                        hdrs=None,
                        fp=io.BytesIO(b'{"detail":"expired"}'),
                    )
                return _FakeResponse({"data": [{"id": "unsloth/gpt-oss-20b"}]})
            raise AssertionError(f"unexpected url: {request.full_url}")

        session = UnslothStudioAuthSession(
            AuthSettings(username="alice", password="secret"),
            openai_api_base="http://127.0.0.1:8888/v1",
            urlopen=fake_urlopen,
        )

        payload = session.request_json("/v1/models", timeout_sec=15.0)

        self.assertEqual(payload["data"][0]["id"], "unsloth/gpt-oss-20b")
        self.assertEqual(session.export_environment()["UNSLOTH_STUDIO_BEARER_TOKEN"], "token-2")
        self.assertEqual(session.export_environment()["LOCAL_LLM_BENCH_UNSLOTH_STUDIO_REFRESH_TOKEN"], "refresh-2")
        self.assertEqual(
            calls,
            [
                ("http://127.0.0.1:8888/api/auth/login", ""),
                ("http://127.0.0.1:8888/v1/models", "Bearer token-1"),
                ("http://127.0.0.1:8888/api/auth/refresh", ""),
                ("http://127.0.0.1:8888/v1/models", "Bearer token-2"),
            ],
        )


class UnslothStudioProviderRuntimeTests(unittest.TestCase):
    def test_prepare_model_loads_unsloth_model_and_formats_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            session = MagicMock()
            session.request_json.side_effect = [
                {
                    "data": [
                        {
                            "id": "unsloth/gpt-oss-20b",
                            "display_name": "GPT OSS 20B",
                            "path": "/tmp/models/unsloth/gpt-oss-20b",
                            "source": "hf_cache",
                            "gguf_variant": "Q4_K_M",
                        }
                    ]
                },
                {
                    "status": "loaded",
                    "model": "unsloth/gpt-oss-20b",
                    "display_name": "GPT OSS 20B",
                    "is_gguf": True,
                    "gguf_variant": "Q4_K_M",
                    "inference": {},
                },
            ]
            session.export_environment.return_value = {"UNSLOTH_STUDIO_BEARER_TOKEN": "token"}

            runtime = UnslothStudioProviderRuntime(config=config, session=session)
            api_model, model_info = runtime.prepare_model("unsloth/gpt-oss-20b")

        self.assertEqual(api_model, "unsloth/gpt-oss-20b")
        assert model_info is not None
        self.assertEqual(model_info["display_name"], "GPT OSS 20B")
        self.assertEqual(model_info["format"], "gguf")
        self.assertEqual(model_info["quantization"], "Q4_K_M (4-bit)")
        self.assertEqual(model_info["path"], "/tmp/models/unsloth/gpt-oss-20b")
        self.assertEqual(
            session.request_json.call_args_list[1].kwargs["payload"],
            {"model_path": "/tmp/models/unsloth/gpt-oss-20b"},
        )
        self.assertEqual(runtime.docker_environment(), {"UNSLOTH_STUDIO_BEARER_TOKEN": "token"})

    def test_describe_model_merges_loaded_and_available_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            session = MagicMock()
            session.request_json.side_effect = [
                {"data": [{"id": "unsloth/gpt-oss-20b", "owned_by": "unsloth"}]},
                {
                    "data": [
                        {
                            "id": "unsloth/gpt-oss-20b",
                            "display_name": "GPT OSS 20B",
                            "path": "/tmp/models/unsloth/gpt-oss-20b",
                            "source": "hf_cache",
                            "gguf_variant": "Q4_K_M",
                        }
                    ]
                },
            ]

            runtime = UnslothStudioProviderRuntime(config=config, session=session)
            model_info = runtime.describe_model("unsloth/gpt-oss-20b")

        assert model_info is not None
        self.assertEqual(model_info["display_name"], "GPT OSS 20B")
        self.assertEqual(model_info["publisher"], "unsloth")
        self.assertEqual(model_info["quantization"], "Q4_K_M (4-bit)")

    def test_unload_model_returns_success_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            session = MagicMock()
            session.request_json.return_value = {"status": "unloaded", "model": "unsloth/gpt-oss-20b"}

            runtime = UnslothStudioProviderRuntime(config=config, session=session)
            results = runtime.unload_model("unsloth/gpt-oss-20b")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "unloaded")
        self.assertEqual(results[0].target, "unsloth/gpt-oss-20b")

    def test_prepare_model_falls_back_to_local_model_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            model_root = Path(tmpdir) / "hf-cache" / "models--unsloth--gemma-4-26b-a4b-it-gguf"
            snapshot_dir = model_root / "snapshots" / "2f6caf1733f31c87fdcfda391e978120033609a0"
            snapshot_dir.mkdir(parents=True)
            main_model_path = snapshot_dir / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
            mmproj_path = snapshot_dir / "mmproj-BF16.gguf"
            main_model_path.write_bytes(b"main-model")
            mmproj_path.write_bytes(b"mm")
            session = MagicMock()
            session.request_json.side_effect = [
                {
                    "data": [
                        {
                            "id": "unsloth/gemma-4-26b-a4b-it-gguf",
                            "display_name": "gemma-4-26b-a4b-it-gguf",
                            "path": str(model_root),
                            "source": "hf_cache",
                        }
                    ]
                },
                {
                    "status": "loaded",
                    "model": "unsloth/gemma-4-26b-a4b-it-gguf",
                    "display_name": "gemma-4-26b-a4b-it-gguf",
                    "is_gguf": True,
                    "inference": {},
                },
            ]

            runtime = UnslothStudioProviderRuntime(config=config, session=session)
            api_model, model_info = runtime.prepare_model("unsloth/gemma-4-26b-a4b-it-gguf")

        self.assertEqual(api_model, "unsloth/gemma-4-26b-a4b-it-gguf")
        assert model_info is not None
        self.assertEqual(model_info["format"], "gguf")
        self.assertEqual(model_info["quantization"], "Q4_K_XL (4-bit)")
        self.assertEqual(model_info["quantization_name"], "Q4_K_XL")
        self.assertEqual(model_info["quantization_bits"], 4)
        self.assertEqual(model_info["path"], str(main_model_path))
        self.assertEqual(session.request_json.call_args_list[0].args[0], "/api/models/local")
        self.assertEqual(
            session.request_json.call_args_list[1].kwargs["payload"],
            {"model_path": str(model_root)},
        )

    def test_prepare_model_infers_quantization_from_display_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            session = MagicMock()
            session.request_json.side_effect = [
                {
                    "data": [
                        {
                            "id": "unsloth/gemma-4-26b-a4b-it-gguf",
                            "display_name": "gemma-4-26b-a4b-it-gguf (UD-Q4_K_XL)",
                            "path": str(Path(tmpdir) / "models" / "gemma-4-26b-a4b-it-gguf"),
                            "source": "hf_cache",
                        }
                    ]
                },
                {
                    "status": "loaded",
                    "model": "unsloth/gemma-4-26b-a4b-it-gguf",
                    "display_name": "gemma-4-26b-a4b-it-gguf (UD-Q4_K_XL)",
                    "is_gguf": True,
                    "inference": {},
                },
            ]

            runtime = UnslothStudioProviderRuntime(config=config, session=session)
            _, model_info = runtime.prepare_model("unsloth/gemma-4-26b-a4b-it-gguf")

        assert model_info is not None
        self.assertEqual(model_info["quantization"], "Q4_K_XL (4-bit)")
        self.assertEqual(model_info["quantization_name"], "Q4_K_XL")
        self.assertEqual(model_info["quantization_bits"], 4)

    def test_prepare_model_prefers_requested_model_for_unsloth_chat_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            local_model_dir = Path(tmpdir) / "models" / "gemma-4-26B-A4B-it-GGUF"
            local_model_dir.mkdir(parents=True)
            session = MagicMock()
            session.request_json.side_effect = [
                {
                    "data": [
                        {
                            "id": "google/gemma-4-26b-a4b",
                            "display_name": "gemma-4-26B-A4B-it-Q4_K_M",
                            "path": str(local_model_dir),
                            "source": "lmstudio",
                        }
                    ]
                },
                {
                    "status": "loaded",
                    "model": str(local_model_dir),
                    "display_name": "gemma-4-26B-A4B-it-Q4_K_M",
                    "is_gguf": True,
                    "inference": {},
                },
            ]

            runtime = UnslothStudioProviderRuntime(config=config, session=session)
            api_model, model_info = runtime.prepare_model("google/gemma-4-26b-a4b")

        self.assertEqual(api_model, "google/gemma-4-26b-a4b")
        assert model_info is not None
        self.assertEqual(model_info["identifier"], "google/gemma-4-26b-a4b")
        self.assertEqual(
            session.request_json.call_args_list[1].kwargs["payload"],
            {"model_path": str(local_model_dir)},
        )

    def test_prepare_model_prefers_lmstudio_gguf_entry_over_transformers_hf_cache_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            hf_transformers_dir = Path(tmpdir) / "hf-cache" / "models--google--gemma-4-26b-a4b"
            lmstudio_dir = Path(tmpdir) / "lmstudio" / "gemma-4-26B-A4B-it-GGUF"
            hf_transformers_dir.mkdir(parents=True)
            lmstudio_dir.mkdir(parents=True)
            session = MagicMock()
            session.request_json.side_effect = [
                {
                    "models": [
                        {
                            "id": "google/gemma-4-26b-a4b",
                            "display_name": "gemma-4-26b-a4b",
                            "path": str(hf_transformers_dir),
                            "source": "hf_cache",
                        },
                        {
                            "id": str(lmstudio_dir),
                            "display_name": "gemma-4-26B-A4B-it-GGUF",
                            "path": str(lmstudio_dir),
                            "source": "lmstudio",
                            "model_id": "lmstudio-community/gemma-4-26B-A4B-it-GGUF",
                        },
                    ]
                },
                {
                    "status": "loaded",
                    "model": str(lmstudio_dir),
                    "display_name": "gemma-4-26B-A4B-it-Q4_K_M",
                    "is_gguf": True,
                    "inference": {},
                },
            ]

            runtime = UnslothStudioProviderRuntime(config=config, session=session)
            api_model, model_info = runtime.prepare_model("google/gemma-4-26b-a4b")

        self.assertEqual(api_model, "google/gemma-4-26b-a4b")
        assert model_info is not None
        self.assertEqual(model_info["quantization"], "Q4_K_M (4-bit)")
        self.assertEqual(
            session.request_json.call_args_list[1].kwargs["payload"],
            {"model_path": str(lmstudio_dir)},
        )

    def test_describe_model_infers_quantization_from_local_gguf_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            model_root = Path(tmpdir) / "hf-cache" / "models--unsloth--gemma-4-26b-a4b-it-gguf"
            snapshot_dir = model_root / "snapshots" / "2f6caf1733f31c87fdcfda391e978120033609a0"
            snapshot_dir.mkdir(parents=True)
            main_model_path = snapshot_dir / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
            mmproj_path = snapshot_dir / "mmproj-BF16.gguf"
            main_model_path.write_bytes(b"main-model")
            mmproj_path.write_bytes(b"mm")
            session = MagicMock()
            session.request_json.side_effect = [
                {"data": []},
                {
                    "data": [
                        {
                            "id": "unsloth/gemma-4-26b-a4b-it-gguf",
                            "display_name": "gemma-4-26b-a4b-it-gguf",
                            "path": str(model_root),
                            "source": "hf_cache",
                        }
                    ]
                },
            ]

            runtime = UnslothStudioProviderRuntime(config=config, session=session)
            model_info = runtime.describe_model("unsloth/gemma-4-26b-a4b-it-gguf")

        assert model_info is not None
        self.assertEqual(model_info["format"], "gguf")
        self.assertEqual(model_info["quantization"], "Q4_K_XL (4-bit)")
        self.assertEqual(model_info["path"], str(main_model_path))

    def test_unload_model_prefers_local_model_path_when_requested_model_is_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            local_model_dir = Path(tmpdir) / "models" / "gemma-4-26B-A4B-it-GGUF"
            local_model_dir.mkdir(parents=True)
            session = MagicMock()
            session.request_json.side_effect = [
                {
                    "data": [
                        {
                            "id": "google/gemma-4-26b-a4b",
                            "display_name": "gemma-4-26B-A4B-it-Q4_K_M",
                            "path": str(local_model_dir),
                            "source": "lmstudio",
                        }
                    ]
                },
                {"data": []},
                {"status": "unloaded", "model": str(local_model_dir)},
            ]

            runtime = UnslothStudioProviderRuntime(config=config, session=session)
            results = runtime.unload_model("google/gemma-4-26b-a4b")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "unloaded")
        self.assertEqual(results[0].target, str(local_model_dir))
        self.assertEqual(
            session.request_json.call_args_list[2].kwargs["payload"],
            {"model_path": str(local_model_dir)},
        )
