"""Exercise the integrated viewer through ASGI; no server or model is started."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import quote

import httpx

from local_llm_bench.dashboard import create_app


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.runs = self.root / "runs"
        self.app = create_app(self.runs)

    def request(self, path, method="GET", **kwargs):
        async def call():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                         base_url="http://127.0.0.1:8080") as client:
                return await client.request(method, path, **kwargs)
        return asyncio.run(call())

    def fixture(self, *, run="run one", phase="warm", iteration=1, question="q/日本語",
                attempt="attempt-one", worker=False, provider="lmstudio", status="success"):
        from inspect_ai.log import EvalLog, EvalSpec, EvalDataset, EvalConfig, EvalSample, write_eval_log
        from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ModelOutput
        from inspect_ai.scorer import Score

        key = hashlib.sha256(f"{phase}:{iteration}:{question}".encode()).hexdigest()[:20]
        directory = self.runs / "logs" / run / "inspect" / key / attempt
        if worker:
            directory /= "worker"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "fixture 日本語 & %.eval"
        metadata = {"provider": provider}
        if not worker:
            metadata.update(run_id=run, phase=phase, iteration=iteration)
        log = EvalLog(status=status, eval=EvalSpec(
            created="2026-09-15T00:00:00Z", task="local_bench_react" if worker else "local_llm_bench",
            dataset=EvalDataset(samples=1, sample_ids=[question]), config=EvalConfig(),
            model="fixture-model", metadata=metadata), samples=[EvalSample(
                id=question, epoch=1, input="fixture question", target="",
                messages=[ChatMessageUser(content="fixture question"), ChatMessageAssistant(content="42")],
                output=ModelOutput.from_content("fixture-model", "42"),
                scores=None if worker else {"host_typed": Score(value=1, answer="42")})])
        write_eval_log(log, path)
        return path

    def test_empty_dashboard_and_bundled_viewer_share_origin(self):
        response = self.request("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('data-tab="inspect"', response.text)
        self.assertIn('const INSPECT_API_URL = "/dashboard-api/inspect-logs"', response.text)
        self.assertEqual(response.text, self.request("/docs/").text)
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        native = self.request("/inspect/")
        self.assertEqual(native.status_code, 200)
        self.assertIn("Inspect View", native.text)
        self.assertEqual(native.headers["x-frame-options"], "SAMEORIGIN")
        self.assertEqual(native.headers["content-security-policy"], "frame-ancestors 'self'")
        asset = self.request("/inspect/assets/index.js")
        self.assertEqual(asset.status_code, 200)
        self.assertIn('get("inspect_server") === "true"', asset.text)
        self.assertIn('params.get("log_file")', asset.text)
        self.assertEqual(self.request("/runs/history.json").json(), [])
        self.assertEqual(self.request("/dashboard-api/inspect-logs").json()["logs"], [])
        self.assertEqual(self.request("/api/log-files").json()["files"], [])
        self.assertEqual(self.request("/api/logs").status_code, 200)

    def test_native_inspect_reads_real_eval_and_scores(self):
        path = self.fixture()
        encoded = quote(str(path), safe="")
        response = self.request("/api/logs/" + encoded)
        self.assertEqual(response.status_code, 200, response.text)
        sample = response.json()["samples"][0]
        self.assertEqual(sample["messages"][1]["content"], "42")
        self.assertEqual(sample["scores"]["host_typed"]["value"], 1)
        self.assertEqual(self.request("/api/log-download/" + encoded).content, path.read_bytes())
        self.assertEqual(len(self.request("/api/log-files").json()["files"]), 1)
        links = self.request("/dashboard-api/inspect-logs").json()["logs"]
        self.assertEqual(links[0]["url"], "/inspect/?inspect_server=true&log_file=" + encoded)
        self.assertEqual(links[0]["log_storage"]["compression"], ["zstd"])
        self.assertEqual(links[0]["log_storage"]["file_bytes"], path.stat().st_size)

    def test_log_capacity_is_cached_and_failure_keeps_header_visible(self):
        self.fixture(status="started")
        from local_llm_bench.inspect_log import storage_metadata
        with patch("local_llm_bench.dashboard.storage_metadata", wraps=storage_metadata) as describe:
            self.request("/dashboard-api/inspect-logs")
            self.request("/dashboard-api/inspect-logs")
        self.assertEqual(describe.call_count, 1)
        self.fixture(status="success")
        with patch("local_llm_bench.dashboard.storage_metadata", side_effect=ValueError("unfinished archive")):
            entry = self.request("/dashboard-api/inspect-logs").json()["logs"][0]
        self.assertEqual(entry["status"], "success")
        self.assertEqual(entry["log_storage"], {"status": "unavailable"})

    def test_links_separate_retries_questions_and_roles(self):
        host = self.fixture()
        worker = self.fixture(worker=True)
        retry = self.fixture(attempt="attempt-two", provider="ds4")
        self.fixture(question="another")
        self.fixture(run="another-run")
        self.fixture(phase="cold")
        logs = self.request("/dashboard-api/inspect-logs", params={
            "run_id": "run one", "phase": "warm", "iteration": 1, "question_id": "q/日本語"}).json()["logs"]
        self.assertEqual({entry["path"] for entry in logs}, {str(host), str(worker), str(retry)})
        agent = next(entry for entry in logs if entry["role"] == "agent")
        self.assertEqual(agent["iteration"], 1)
        self.assertEqual(agent["phase"], "warm")
        self.assertEqual(agent["attempt_dir"], str(host.parent))
        self.assertEqual({entry["provider"] for entry in logs}, {"lmstudio", "ds4"})
        run_logs = self.request("/dashboard-api/inspect-logs", params={
            "run_id": "run one", "phase": "warm", "iteration": 1}).json()["logs"]
        self.assertEqual(len(run_logs), 4)

    def test_refresh_detects_new_changed_removed_and_incomplete_logs(self):
        path = self.fixture(status="started")
        self.assertEqual(self.request("/dashboard-api/inspect-logs").json()["logs"][0]["status"], "started")
        self.fixture(status="success")
        self.assertEqual(self.request("/dashboard-api/inspect-logs").json()["logs"][0]["status"], "success")
        unfinished = path.parent / "unfinished.eval"
        unfinished.write_bytes(b"incomplete zip")
        logs = self.request("/dashboard-api/inspect-logs").json()["logs"]
        self.assertEqual({entry["status"] for entry in logs}, {"success", "unavailable"})
        path.unlink()
        self.assertEqual(len(self.request("/dashboard-api/inspect-logs").json()["logs"]), 1)

    def test_read_only_paths_and_browser_boundaries(self):
        path = self.fixture()
        original = path.read_bytes()
        encoded = quote(str(path), safe="")
        for method, endpoint in (("DELETE", "/api/log-delete/"), ("POST", "/api/log-edit/")):
            self.assertEqual(self.request(endpoint + encoded, method,
                headers={"X-Inspect-View-Request": "true"}, json={}).status_code, 405)
        self.assertEqual(path.read_bytes(), original)
        outside = self.root / "outside.eval"
        outside.write_bytes(original)
        alias = path.parent / "alias.eval"
        alias.symlink_to(outside)
        for target in (outside, alias):
            self.assertEqual(self.request("/api/logs/" + quote(str(target), safe="")).status_code, 403)
        self.assertEqual(self.request("/api/log-files", params={"log_dir": str(self.root)}).status_code, 403)
        self.assertEqual(self.request("/", headers={"host": "untrusted.invalid:8080"}).status_code, 400)
        self.assertEqual(self.request("/runs/history.json", headers={"origin": "https://untrusted.invalid"}).status_code, 403)
        self.assertEqual(self.request("/bench_unsloth.yaml").status_code, 404)
        self.assertEqual(self.request("/configs/bench_unsloth.yaml").status_code, 404)
        self.assertEqual(self.request("/runs/logs/..%2f..%2foutside.eval").status_code, 404)

    def test_history_and_raw_logs_refresh_without_exposing_project(self):
        history = self.runs / "history.json"
        history.write_text(json.dumps([{"run_id": "fixture"}]))
        response = self.request("/runs/history.json")
        self.assertEqual(response.json()[0]["run_id"], "fixture")
        self.assertEqual(response.headers["cache-control"], "no-store")
        history.write_text("invalid json")
        self.assertEqual(self.request("/runs/history.json").status_code, 503)
        raw = self.runs / "logs" / "fixture.json"
        raw.write_text('{"status": "success"}')
        self.assertEqual(self.request("/runs/logs/fixture.json").json()["status"], "success")
        (self.runs / "logs" / "alias.json").symlink_to(history)
        self.assertEqual(self.request("/runs/logs/alias.json").status_code, 404)


if __name__ == "__main__":
    unittest.main()
