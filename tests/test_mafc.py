"""MAFC dataset, archive staging and scoring checks; never run the challenge."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import zipfile

from local_llm_bench.conditions import digest, workload
from local_llm_bench.config import load_config
from local_llm_bench.docker_task.runner import _coerce_question_result, _run_question_in_docker, _write_request_payload
from local_llm_bench.docker_task.scorer import score_answer
from local_llm_bench.docker_task.spec import load_spec
from local_llm_bench.docker_task.targets import resolve_native_binary_target

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "benchmarks/mafc/spec.yaml"
SOURCE = ROOT / "benchmarks/mafc/source.json"
ARCHIVE = ROOT / "data/MAFC.zip"
FLAG = "ctf4b{way_2_90!_y0u_suc3553d_2_ana1yz3_Ma1war3!!!}"
CONFIGS = ["bench_mafc_arm64_lmstudio_4_models.yaml", "bench_mafc_arm64_ds4.yaml",
           "bench_mafc_arm64_ds4_glm_5_3_flash.yaml"]


class MAFCDataTests(unittest.TestCase):
    def test_original_distribution_matches_provenance_and_contains_no_answer(self):
        source = json.loads(SOURCE.read_text())
        self.assertIn(source["commit"], source["archive"]["url"])
        self.assertEqual(ARCHIVE.stat().st_size, source["archive"]["size_bytes"])
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), source["archive"]["sha256"])
        with zipfile.ZipFile(ARCHIVE) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual({i.filename for i in z.infolist() if not i.is_dir()},
                             {item["member"] for item in source["files"]})
            for entry in source["files"]:
                name = PurePosixPath(entry["member"])
                self.assertFalse(name.is_absolute())
                self.assertNotIn("..", name.parts)
                content = z.read(entry["member"])
                self.assertEqual(len(content), entry["size_bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
                self.assertNotIn(FLAG.encode(), content)
                if name.suffix == ".exe":
                    pe = struct.unpack_from("<I", content, 0x3c)[0]
                    self.assertEqual(content[pe:pe + 4], b"PE\0\0")
                    self.assertEqual(struct.unpack_from("<H", content, pe + 4)[0], 0x8664)
                    self.assertEqual(struct.unpack_from("<H", content, pe + 24)[0], 0x20b)

    def test_zip_resolves_to_executable_with_encrypted_file_beside_it(self):
        target = resolve_native_binary_target(ARCHIVE)
        self.addCleanup(lambda: shutil.rmtree(target.cleanup_dir, ignore_errors=True) if target.cleanup_dir else None)
        self.assertEqual(target.resolution, "auto_resolved")
        self.assertEqual(target.path.name, "MalwareAnalysis-FirstChallenge.exe")
        self.assertEqual(target.path.read_bytes()[:2], b"MZ")
        self.assertEqual((target.path.parent / "flag.encrypted").stat().st_size, 64)
        self.assertEqual(resolve_native_binary_target(ROOT / "data/d-compile").resolution, "direct")

    def test_official_answer_is_case_sensitive_and_whole_string_only(self):
        question = load_spec(SPEC).questions[0]
        self.assertEqual(question.id, "mafc")
        self.assertEqual(question.binary_path, ARCHIVE)
        self.assertTrue(score_answer(question.answer_type, FLAG, question.gold_answer).correct)
        for wrong in (FLAG.lower(), "prefix " + FLAG, FLAG + " suffix", FLAG + "\n", ""):
            with self.subTest(wrong=wrong):
                self.assertFalse(score_answer(question.answer_type, wrong, question.gold_answer).correct)

    def test_configs_preserve_corresponding_runtime_settings(self):
        for name in CONFIGS:
            with self.subTest(name=name):
                config = load_config(ROOT / "configs" / name)
                baseline = load_config(ROOT / "configs" / name.replace("bench_mafc_", "bench_d_compile_"))
                self.assertFalse(hasattr(config, "harness"))
                self.assertEqual(config.mode, "docker_task")
                self.assertEqual(config.benchmark_spec_path, SPEC)
                self.assertEqual(config.benchmark_answer_key_path, SPEC.with_name("spec.answers.yaml"))
                self.assertEqual(config.docker_image, "local-llm-bench:inspect-v1")
                for field in ("models", "request", "runs", "inspect", "ds4", "ds4_sampling", "docker_platform", "api_base", "docker_api_base"):
                    self.assertEqual(getattr(config, field), getattr(baseline, field), field)
        self.assertEqual(len(load_config(ROOT / "configs" / CONFIGS[0]).models), 4)
        self.assertEqual(load_config(ROOT / "configs" / CONFIGS[0]).request_parameters(), {})

    def test_workload_identity_includes_the_complete_archive(self):
        config = load_config(ROOT / "configs" / CONFIGS[0])
        original = workload(config)
        source = json.loads(SOURCE.read_text())
        self.assertEqual(original["questions"][0]["binary_path"], source["archive"]["sha256"])
        # Alter just the cipher-text member of a temporary copy, never the fixture.
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "MAFC.zip"
            with zipfile.ZipFile(ARCHIVE) as src, zipfile.ZipFile(changed, "w") as dst:
                for member in src.infolist():
                    data = src.read(member)
                    dst.writestr(member, b"X" + data[1:] if member.filename.endswith("flag.encrypted") else data)
            spec = load_spec(SPEC)
            spec.questions[0].binary_path = changed
            with patch("local_llm_bench.docker_task.spec.load_spec", return_value=spec):
                self.assertNotEqual(digest(workload(config)), digest(original))

    def test_docker_stages_only_distribution_and_request_and_scores_on_host(self):
        question = load_spec(SPEC).questions[0]
        config = load_config(ROOT / "configs" / CONFIGS[0])

        def fake_docker(args, **kwargs):
            mounts = [args[i + 1] for i, token in enumerate(args) if token == "-v"]
            bundle = Path(next(mount[:-6] for mount in mounts if mount.endswith(":/work")))
            self.assertEqual({p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()},
                             {"request.json", "data/MAFC.zip"})
            self.assertEqual((bundle / "data/MAFC.zip").read_bytes(), ARCHIVE.read_bytes())
            request = json.loads((bundle / "request.json").read_text())
            self.assertEqual(request["question"]["binary_path"], "data/MAFC.zip")
            self.assertIn("Target file: data/MAFC.zip", request["task_prompt"])
            self.assertNotIn("gold_answer", request["question"])
            self.assertNotIn(FLAG, json.dumps(request))
            self.assertEqual(request["settings_source"], "lmstudio_saved")
            self.assertNotIn("max_tokens", request)
            return subprocess.CompletedProcess(args, 0, json.dumps({
                "status": "success", "predicted_answer": FLAG, "response_text": "FINAL_ANSWER: " + FLAG}), "")

        with patch("local_llm_bench.docker_task.runner._docker_binary", return_value="docker"), \
             patch("local_llm_bench.docker_task.runner._maybe_stage_docker_ghidra_mcp_source", return_value=None):
            from local_llm_bench.inspect_harness import run_unit
            with tempfile.TemporaryDirectory() as directory:
                info = {}
                raw = run_unit(config=config, model=config.models[0], question=question,
                    operation=lambda: _run_question_in_docker(config=config, selected_model=config.models[0],
                        question=question, docker_executor=fake_docker),
                    log_dir=Path(directory), metadata={}, info=info)
        self.assertEqual(_coerce_question_result(question, raw, info)["benchmark_score"], 1)
        self.assertEqual(_coerce_question_result(question, {"status": "error", "predicted_answer": FLAG}, {})["benchmark_error_count"], 1)


class MAFCWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_imports_the_pe_and_passes_only_public_input_to_inspect(self):
        from local_llm_bench.docker_task import container_worker as worker
        class SessionStack:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        async def fake_mcp(**kwargs):
            return SessionStack(), object(), SimpleNamespace(tools=[])

        config = load_config(ROOT / "configs" / CONFIGS[0])
        question = load_spec(SPEC).questions[0]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            # All imported-program scaffolding and archive extraction stay temporary.
            stack.enter_context(patch("tempfile.tempdir", str(root)))
            stack.enter_context(patch.object(worker, "_BOOTSTRAP_DIR", root / "bootstrap"))
            imported = stack.enter_context(patch.object(worker, "_import_binary_into_project_with_analyze_headless", return_value="/fixture"))
            stack.enter_context(patch.object(worker, "_open_mcp_stdio_session", side_effect=fake_mcp))
            agent = stack.enter_context(patch("local_llm_bench.docker_task.inspect_agent.run_agent", new_callable=AsyncMock,
                                             return_value={"status": "success", "predicted_answer": FLAG}))
            path = _write_request_payload(bundle_dir=root, config=config, selected_model=config.models[0],
                                          question=question, staged_binary_ref=str(ARCHIVE))
            payload = json.loads(path.read_text())
            result = await worker._run_question(payload)
            self.assertEqual(result["status"], "success", result.get("error"))
            target = imported.call_args.kwargs["binary_path"]
            self.assertEqual(target.name, "MalwareAnalysis-FirstChallenge.exe")
            self.assertEqual(target.read_bytes()[:2], b"MZ")
            self.assertTrue((target.parent / "flag.encrypted").is_file())
            self.assertNotIn(FLAG, json.dumps(agent.call_args.kwargs["payload"]))
            self.assertNotIn("gold_answer", agent.call_args.kwargs["payload"]["question"])
