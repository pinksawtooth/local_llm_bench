import json
from pathlib import Path
import tempfile
import unittest

from local_llm_bench.config import OutputSettings
from local_llm_bench.error_utils import annotate_error_info
from local_llm_bench.history import load_history_entries
from local_llm_bench.run_logs import persist_run_logs


INFO_LOG = ("[09/17/26 07:13:25] INFO     Processing request of type            server.py:733\n"
            "                             ListToolsRequest")
INFO_SIGNATURE = "INFO Processing request of type server.py:733"


class ErrorMetadataTests(unittest.TestCase):
    def test_success_diagnostics_do_not_become_errors(self):
        for stderr in (INFO_LOG, "DEBUG request dispatched", "WARNING cleanup recovered",
                       "ERROR tool failed but the agent recovered"):
            for status in ("success", " SUCCESS ", None):
                with self.subTest(stderr=stderr, status=status):
                    record = {"status": status, "error": " ", "stderr_excerpt": stderr,
                              "error_signature": INFO_SIGNATURE, "error_category": "tool",
                              "log_path": "logs/original.json"}
                    annotate_error_info(record)
                    self.assertIsNone(record["error_signature"])
                    self.assertIsNone(record["error_category"])
                    self.assertEqual(record["stderr_excerpt"], stderr)
                    self.assertEqual(record["log_path"], "logs/original.json")

    def test_failures_and_explicit_errors_keep_their_diagnostics(self):
        for status, error, stderr, signature, category in (
            ("error", "[mafc] RuntimeError('max_turns exhausted (24)')", INFO_LOG,
             "RuntimeError('max_turns exhausted (24)')", "other"),
            ("timeout", "request timed out", INFO_LOG, "request timed out", "timeout"),
            ("error", None, "docker exited with 1", "docker exited with 1", "docker"),
            ("success", "HTTPError 503: unavailable", INFO_LOG, "HTTPError 503: unavailable", "api"),
            ("timeout", None, None, None, "timeout"),
        ):
            with self.subTest(status=status, error=error):
                record = {"status": status, "error": error, "stderr_excerpt": stderr}
                annotate_error_info(record)
                self.assertEqual(record["error_signature"], signature)
                self.assertEqual(record["error_category"], category)
                self.assertEqual(record["error"], error)

    def test_saved_history_clears_false_metadata_without_rewriting_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            success = {"status": "success", "error": None, "stderr_excerpt": INFO_LOG,
                       "error_signature": INFO_SIGNATURE, "error_category": "tool"}
            path.write_text(json.dumps([{"run_id": "old", "model": "fixture", "records": [
                {**success, "phase": "warm", "iteration": 1,
                 "question_results": [{**success, "question_id": "q1"}]},
                {"phase": "warm", "iteration": 2, "status": "error", "error": "HTTPError 503"},
            ]}]))
            original = path.read_bytes()
            runs = load_history_entries(path)
            record = runs[0]["records"][0]
            for entry in (record, record["question_results"][0]):
                self.assertIsNone(entry["error_signature"])
                self.assertIsNone(entry["error_category"])
                self.assertEqual(entry["stderr_excerpt"], INFO_LOG)
            self.assertEqual(runs[0]["records"][1]["error_signature"], "HTTPError 503")
            self.assertEqual(path.read_bytes(), original)

    def test_persisted_success_retains_raw_mcp_log_without_error_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = OutputSettings(history_json=root / "history.json", latest_json=root / "latest.json",
                                    report_html=root / "index.html", run_logs_dir=root / "logs")
            run = {"run_id": "success", "model": "fixture", "records": [
                {"phase": "warm", "iteration": 1, "status": "success", "error": None,
                 "question_results": [{"question_id": "q1", "status": "success", "error": None}]}],
                "_log_bundle": {"attempts": [{"record_index": 0, "payload": {"stderr": INFO_LOG},
                    "question_logs": [{"question_index": 0, "question_id": "q1",
                        "payload": {"stderr": INFO_LOG,
                                    "parsed_worker_result": {"status": "success", "error": None}}}]}]}}
            persisted = persist_run_logs(output, run)
            record = persisted["records"][0]
            question = record["question_results"][0]
            for entry in (record, question):
                self.assertIsNone(entry["error_signature"])
                self.assertIsNone(entry["error_category"])
                self.assertEqual(entry["stderr_excerpt"], INFO_LOG)
            raw = json.loads((root / question["log_path"]).read_text())
            self.assertEqual(raw["stderr"], INFO_LOG)
