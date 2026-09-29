"""Task performance stays visible without inventing per-request telemetry."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from local_llm_bench.report import render_report_html


class TaskPerformanceReportTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js required for dashboard JavaScript tests")
    def test_task_medians_grouping_missing_metrics_and_inspect_links(self):
        html = render_report_html("history.json")
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        functions = script.split('    document.getElementById("reload")', 1)[0]
        checks = r'''
const assert = require("node:assert/strict");
const elements = new Map();
document.getElementById = id => {
  if (!elements.has(id)) elements.set(id, {innerHTML: "", textContent: "", querySelectorAll: () => []});
  return elements.get(id);
};
document.querySelectorAll = () => [];
function trial(iteration, input, output, elapsed, extra = {}) {
  return {phase: "warm", iteration, status: "success", prompt_tokens: input,
    completion_tokens: output, total_latency_ms: elapsed,
    benchmark_correct_count: 0, benchmark_incorrect_count: 1, benchmark_error_count: 0,
    started_at: `2026-09-15T0${iteration}:00:00Z`,
    question_results: [{question_id: "q1", status: "success", benchmark_correct: false,
      benchmark_correct_count: 0, benchmark_incorrect_count: 1, benchmark_error_count: 0,
      inspect: {log_path: `/logs/attempt-${iteration}.eval`}}], ...extra};
}
function task(id, records, extra = {}) {
  return {run_id: id, model: "fixture-model", provider: "lmstudio", status: "completed",
    benchmark_mode: "docker_task", benchmark_id: id, benchmark_title: id, question_count: 1,
    evaluation: {name: "inspect"}, comparison: {group_id: "shared"}, records, ...extra};
}
function display(runs) {
  reportData = buildReportPayload(runs); viewData = reportData;
  state.leaderboardFilterTouched = false;
  renderLeaderboard();
}
const records = [trial(1, 100, 10, 1000), trial(2, 300, 30, 3000), trial(3, 200, 20, 2000),
  trial(4, 999999, 999999, 999999, {status: "error"})];
const compile = task("d-compile", records);
const rc4 = task("rc4", [trial(1, 400, 40, 4000)]);
display([compile, rc4]);
let groups = performanceGroups(currentRecords());
assert.equal(groups.length, 2, "different tasks must never share a performance row");
const group = groups.find(g => g[0].benchmark_id === "d-compile");
assert.equal(group.length, 4);
assert.equal(performanceMedian(group, "input_tokens"), 200);
assert.equal(performanceMedian(group, "output_tokens"), 20);
assert.equal(performanceMedian(group, "e2e_ms"), 2000);
assert.equal(performanceMedian(group, "total_tps"), 110);
assert.equal(performanceSortRow(group).success_rate, 0.75);
assert.equal(performanceTestLabel(group[0]), "d-compile / 1問");
const body = elements.get("leaderboard-body").innerHTML;
assert.ok(body.includes("d-compile / 1問") && body.includes("rc4 / 1問"));
assert.ok(!body.includes("データはありません"));
assert.ok(body.includes("3/4") && body.includes("失敗 1"));
assert.ok(body.includes("2.000") && body.includes("110.0"));
assert.ok(body.includes("task_total") && body.includes("includes tools"));
assert.ok(body.includes("最新ログ") && body.includes('&quot;iteration&quot;:4'));
assert.equal(elements.get("quality-section").hidden, false);
assert.equal(elements.get("concurrency-section").hidden, true);
assert.ok(elements.get("leaderboard-note").textContent.includes("ツール実行"));

// Legacy conversation/first-response aliases must not masquerade as prefill
// or decode timings for an entire multi-turn task.
const legacy = {...group[0], ttft_ms: 500, prompt_latency_ms: 2000,
  approx_prompt_tps: 123, decode_tps: 456, initial_prompt_tps: 789};
for (const field of ["pp_tps", "tg_tps", "ttft_ms", "tpot_ms"]) {
  assert.equal(performanceMetrics(legacy)[field], null);
  assert.match(performanceCell([legacy], field), /未取得/);
  assert.match(performanceCell([legacy], field), />—</);
}
assert.equal(performanceMetrics({benchmark_mode: "docker_task"}).total_tps, null);
assert.equal(performanceMetrics({...legacy, completion_tokens: null}).total_tps, null);
assert.equal(performanceMetrics({...legacy, total_latency_ms: 0}).total_tps, null);
assert.equal(performanceMetrics({...legacy, prompt_tokens: 0, completion_tokens: 0}).total_tps, 0);
const measured = {...legacy, metrics: {version: 1, pp_tps: 42, sources: {pp_tps: "api.timings.prompt"}}};
assert.equal(performanceMetrics(measured).pp_tps, 42);
assert.match(performanceCell([measured], "pp_tps"), /API報告値/);

// New task observations are rendered in every metric column and use inference
// time, even when the task's total duration includes lengthy tool execution.
const inferred = {...baseTaskMetrics()};
function baseTaskMetrics() {
  return {version: 1, scope: "inference_requests", pp_tps: 700, tg_tps: 30, ttft_ms: 250, tpot_ms: 80,
    input_tokens: 300, output_tokens: 32, e2e_ms: 2700, total_tps: 332 / 2.7, cache_state: "unknown",
    sources: {pp_tps: "request_median", tg_tps: "request_median", ttft_ms: "request_median", tpot_ms: "request_median"},
    source_details: {pp_tps: ["api.timings.prompt"], tg_tps: ["client_estimate"]}};
}
display([task("d-compile", [trial(1, 300, 32, 100000, {metrics: inferred})])]);
const measuredBody = elements.get("leaderboard-body").innerHTML;
for (const value of ["700.0", "30.0", "250.0", "80.00", "2.700", "123.0"]) assert.ok(measuredBody.includes(value), value);
assert.ok(elements.get("leaderboard-note").textContent.includes("ツール実行時間を除きます"));
assert.ok(!measuredBody.includes("100.000"));
assert.match(performanceCell(currentRecords(), "tpot_ms"), /推論リクエストごとの計測値の中央値/);
renderDetailPane(currentRecords()[0]);
assert.ok(elements.get("detail-pane").innerHTML.includes("推論E2Eの合計"));
assert.ok(elements.get("detail-pane").innerHTML.includes("700.0"));
assert.equal(performanceGroups([currentRecords()[0], {...currentRecords()[0], metrics: undefined}]).length, 2);
assert.equal(performanceGroups([currentRecords()[0], {...currentRecords()[0], metrics: {...inferred,
  source_details: {pp_tps: ["client_ttft_estimate"]}}}]).length, 2);

// Benchmark subsets, phases, and prompt measurements remain separate.
const base = group[0];
assert.equal(performanceGroups([base, {...base, question_results: [{question_id: "q2"}]}]).length, 2);
assert.equal(performanceGroups([base, {...base, phase: "cold"}]).length, 2);
assert.equal(performanceGroups([base, {...base, benchmark_mode: "prompt"}]).length, 2);
const questions = [{question_id: "a"}, {question_id: "b"}];
assert.equal(performanceGroups([{...base, question_results: questions},
  {...base, question_results: [...questions].reverse()}]).length, 1);
assert.equal(performanceGroups([base, {...base, partial: true}]).length, 1);
assert.equal(performanceGroups([base, {...base, partial: true}])[0].length, 1);
display([task("partial", [trial(1, 100, 10, 1000, {partial: true})]),
  task("interrupted", records, {status: "interrupted"})]);
assert.ok(elements.get("leaderboard-body").innerHTML.includes("データはありません"));

display([task('<img src=x onerror="alert(1)">', records)]);
assert.ok(!elements.get("leaderboard-body").innerHTML.includes("<img"));
display([compile, rc4]);
state.leaderboardFilterTouched = true;
state.leaderboardSelectedModels = [];
renderLeaderboard();
assert.ok(elements.get("leaderboard-body").innerHTML.includes("データはありません"));
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.js"
            path.write_text("const document = {getElementById: () => null};\n" + functions + checks,
                            encoding="utf-8")
            result = subprocess.run([shutil.which("node"), str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
