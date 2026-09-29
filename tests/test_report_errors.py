from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from local_llm_bench.report import render_report_html


class ReportErrorTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js required for dashboard JavaScript tests")
    def test_error_analysis_excludes_success_logs_and_retains_failed_questions(self):
        script = render_report_html("history.json").split("<script>", 1)[1].split("</script>", 1)[0]
        functions = script.split('    document.getElementById("reload")', 1)[0]
        checks = r'''
const assert = require("node:assert/strict");
const info = "[09/17/26 07:13:25] INFO     Processing request of type            server.py:733\n                             ListToolsRequest";
const signature = "INFO Processing request of type server.py:733";
const success = {phase: "warm", status: "success", error: null, stderr_excerpt: info,
  error_signature: signature, error_category: "tool", log_path: "logs/original.json"};
const records = [
  {...success, iteration: 1},
  {...success, iteration: 2, question_results: [{...success, question_id: "correct", benchmark_score: 1}]},
  {...success, iteration: 3, question_results: [{...success, question_id: "wrong", benchmark_score: 0}]},
  {...success, iteration: 4, error: " ", stderr_excerpt: "WARNING cleanup recovered"},
  {phase: "warm", iteration: 5, status: "error", error: "RuntimeError('max_turns exhausted (24)')", stderr_excerpt: info},
  {phase: "warm", iteration: 6, status: "timeout"},
  {phase: "warm", iteration: 7, status: "error", question_results: [
    {...success, question_id: "passed"},
    {question_id: "failed", status: "error", error: "HTTPError 503: unavailable", stderr_excerpt: info},
  ]},
  {phase: "warm", iteration: 8, status: "error", stderr_excerpt: "docker exited with 1"},
  {phase: "warm", iteration: 9, status: "error"},
];
const original = JSON.stringify(records);
const run = {run_id: "fixture", model: "fixture", status: "completed", records};
const report = buildReportPayload([run]);
const events = flattenErrorEvents(report.records);
assert.deepEqual(events.map(event => event.iteration), [5, 6, 7, 8, 9]);
assert.equal(events[0].error_signature, "RuntimeError('max_turns exhausted (24)')");
assert.equal(events[1].error_category, "timeout");
assert.equal(events[2].question_id, "failed");
assert.equal(events[2].error_category, "api");
assert.equal(events[3].error_category, "docker");
assert.equal(events[4].status, "error");
assert.ok(events.every(event => event.error_signature !== signature));
const cleared = report.records[0];
assert.equal(cleared.error_signature, "");
assert.equal(cleared.error_category, "");
assert.equal(cleared.stderr_excerpt, info);
assert.equal(cleared.log_path, "logs/original.json");
assert.equal(report.records[1].question_results[0].error_signature, "");
assert.equal(JSON.stringify(records), original);
assert.equal(flattenErrorEvents([{...success, error: "HTTPError 500: explicit error"}]).length, 1);
assert.equal(flattenErrorEvents([{...success, status: undefined}]).length, 0);
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.js"
            path.write_text("const document = {getElementById: () => null};\n" + functions + checks)
            checked = subprocess.run([shutil.which("node"), str(path)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
