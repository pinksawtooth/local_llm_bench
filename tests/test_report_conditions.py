import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from local_llm_bench.report import render_report_html


class ReportConditionsTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js required for dashboard JavaScript tests")
    def test_grouping_comparison_axes_and_missing_data(self):
        html = render_report_html("history.json")
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.js"
            path.write_text(script)
            checked = subprocess.run([shutil.which("node"), "--check", str(path)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            # Exercise the data functions without browser globals or DOM interaction.
            functions = script.split('    document.getElementById("reload")', 1)[0]
            checks = r'''
const assert = require("node:assert/strict");
assert.equal(formatRunSeconds(null), "不明");
assert.equal(formatRunSeconds(undefined), "不明");
assert.equal(formatRunSeconds(0), "0.000 s");
const ds4Info = normalizeModelInfo({model_identity_scope: "server_compatibility_alias"}, "deepseek-Q4_K_M.gguf");
assert.equal(ds4Info.quantization_name, undefined);
assert.equal(ds4Info.format, undefined);
assert.ok(modelIdentityText("deepseek", ds4Info).includes("互換エイリアス"));
const condition = {model: {artifact: {sha256: "weights"}}, runtime: {engine: "fixture", version: "1"}, request: {temperature: 0}, cache: {state: "unknown"}};
const first = {run_id: "a", status: "completed", model: "model", prompt_text: "prompt", conditions: condition, comparison: {group_id: "one", unknown_fields: ["cache.state"]}, records: [{phase: "warm", iteration: 1, status: "success", total_latency_ms: 100, completion_tokens: 4}]};
const second = JSON.parse(JSON.stringify(first));
second.run_id = "b"; second.comparison.group_id = "two"; second.conditions.request.temperature = 0.7;
const interrupted = {...first, run_id: "interrupted", status: "interrupted", comparison: {group_id: "partial"}};
const report = buildReportPayload([first, second, interrupted]);
const inspected = normalizeRunEntry({...first, evaluation: {name: "inspect", version: "0.3.263"}});
const original = normalizeRunEntry({...first, evaluation: {name: "legacy"}});
assert.equal(inspected.records[0].evaluation.name, "inspect");
assert.ok(comparisonModelKey("model", null, null, inspected).includes("Inspect AI"));
assert.notEqual(comparisonModelKey("model", null, null, inspected), comparisonModelKey("model", null, null, original));
assert.equal(report.summary.total_models, 2);
assert.equal(report.summary.total_samples, 2);
assert.equal(report.records.length, 3);
assert.equal(report.records[0].conditions.request.temperature, 0);
assert.notEqual(report.records[0].comparison_model, report.records[1].comparison_model);
assert.equal(assessComparison(first, second, "model").verdict, "比較軸以外の条件差あり");
assert.ok(assessComparison(first, second, "model").differences.some(item => item.field === "request.temperature" && !item.axis));
const otherRuntime = JSON.parse(JSON.stringify(first));
otherRuntime.conditions.runtime.version = "2";
const runtimeComparison = assessComparison(first, otherRuntime, "runtime");
assert.equal(runtimeComparison.differences.filter(item => !item.axis).length, 0);
assert.ok(runtimeComparison.unknown.includes("cache.state"));
const legacy = {model: "model", records: first.records};
const legacyReport = buildReportPayload([legacy, legacy]);
assert.equal(legacyReport.summary.total_models, 2);
assert.ok(assessComparison(legacy, first, "model").unknown.includes("旧形式: 測定条件なし"));
const peer = {...first, records: [{...first.records[0], prompt_tokens: 999}]};
const unknownTokens = buildReportPayload([first, peer]);
assert.ok(unknownTokens.records[0].prompt_tokens == null);
const normalizedRuns = report.history_runs;
assert.notEqual(telemetrySequencesForRun(normalizedRuns[0])[0].model, telemetrySequencesForRun(normalizedRuns[1])[0].model);

// Run-level providers must survive flattening, and equal model/group names
// from different runtimes must remain distinct throughout the dashboard.
const studio = {...first, run_id: "studio", provider: "lmstudio"};
const ds4 = {...first, run_id: "ds4", provider: "ds4"};
const providers = buildReportPayload([studio, ds4]);
assert.deepEqual(providers.records.map(record => record.provider), ["lmstudio", "ds4"]);
assert.equal(providers.summary.total_models, 2);
assert.deepEqual(new Set(providers.summary.models.map(row => row.provider)), new Set(["lmstudio", "ds4"]));
assert.ok(providers.records[0].comparison_model.includes("[LM Studio]"));
assert.ok(providers.records[1].comparison_model.includes("[ds4]"));
const omlx = {...first, run_id: "omlx", provider: "omlx"};
const fourProviders = buildReportPayload([studio, ds4, omlx, {...first, run_id: "unsloth", provider: "unsloth_studio"}]);
assert.equal(fourProviders.summary.total_models, 4);
assert.ok(fourProviders.records[2].comparison_model.includes("[oMLX]"));
assert.equal(providerLabel("omlx"), "oMLX");
assert.match(renderProviderBadge("omlx"), /provider-omlx[^>]*>oMLX<\/span>/);
assert.ok(inspectLogLabel({role: "agent", provider: "omlx"}).includes("oMLX"));
const fiveProviders = buildReportPayload([studio, ds4, omlx, {...first, run_id: "mlx", provider: "mlx_serve"}]);
assert.equal(fiveProviders.summary.total_models, 4);
assert.ok(fiveProviders.records[3].comparison_model.includes("[mlx-serve]"));
assert.equal(providerLabel("mlx_serve"), "mlx-serve");
assert.match(renderProviderBadge("mlx_serve"), /provider-mlx_serve[^>]*>mlx-serve<\/span>/);
assert.ok(inspectLogLabel({role: "agent", provider: "mlx_serve"}).includes("mlx-serve"));
assert.equal(normalizeRunEntry({...first, conditions: {provider: "ds4"}}).records[0].provider, "ds4");
assert.equal(normalizeRunEntry({...first, model_info: {provider: "unsloth_studio"}}).records[0].provider, "unsloth_studio");
assert.equal(normalizeRunEntry({...studio, records: [{...first.records[0], provider: "ds4"}]}).records[0].provider, "ds4");
assert.equal(recordedProvider({provider: " Unsloth-Studio "}), "unsloth_studio");
const unknownProvider = normalizeRunEntry({...first, model: "ds4-flash", api_base: "http://localhost:1234/v1", conditions: {runtime: {engine: "llama.cpp"}}});
assert.equal(unknownProvider.records[0].provider, "");
assert.equal(providerLabel(unknownProvider.provider), "不明");
assert.equal(providerLabel("unsloth_studio"), "Unsloth Studio");
assert.equal(providerLabel("__proto__"), "不明");
assert.ok(!renderProviderBadge('<img src=x onerror="alert(1)">').includes("<img"));

// Drilldown carries the exact result's audit path; retries stay independently selectable.
const audit = {...providers.records[0], evaluation: {name: "inspect"}, inspect: {log_path: "/logs/accepted.eval"}};
const button = inspectButton(audit);
assert.ok(button.includes("data-inspect-scope="));
assert.ok(button.includes("accepted.eval"));
assert.ok(button.includes("prompt"));
const questionButton = inspectButton(audit, {question_id: '<img src=x onerror="alert(1)">', inspect: {log_path: "/logs/question.eval"}});
assert.ok(!questionButton.includes("<img"));
assert.ok(questionButton.includes("question.eval"));
const catalogue = [
  {path: "/logs/retry.eval", url: "/inspect/?log_file=retry"},
  {path: "/logs/accepted.eval", url: "/inspect/?log_file=accepted"},
];
assert.equal(chooseInspectLog(catalogue, "/logs/accepted.eval", ""), catalogue[1].url);
assert.equal(chooseInspectLog(catalogue, "/logs/deleted.eval", ""), "");
assert.equal(chooseInspectLog(catalogue, "/logs/accepted.eval", catalogue[0].url), catalogue[0].url);
assert.ok(inspectLogLabel({role: "agent", provider: "ds4"}).includes("会話・ツール"));
assert.ok(inspectLogLabel({role: "evaluation", provider: "lmstudio"}).includes("LM Studio"));

// Render the actual view functions into a minimal DOM, without opening a
// browser, starting a server, fetching history, or contacting a model runtime.
const elements = new Map();
document.getElementById = (id) => {
  if (!elements.has(id)) elements.set(id, {innerHTML: "", textContent: "", value: "", querySelectorAll: () => []});
  return elements.get(id);
};
document.querySelectorAll = () => [];
reportData = providers;
viewData = providers;
renderLeaderboard();
assert.match(elements.get("leaderboard-body").innerHTML, /provider-lmstudio[^>]*>LM Studio<\/span>/);
assert.match(elements.get("leaderboard-body").innerHTML, /provider-ds4[^>]*>ds4<\/span>/);
const visibleText = html => html.replace(/<[^>]*>/g, "");
assert.ok(visibleText(elements.get("leaderboard-model-options").innerHTML).includes("model · LM Studio"));
assert.ok(visibleText(elements.get("leaderboard-model-options").innerHTML).includes("model · ds4"));
renderCompare();
assert.match(elements.get("compare-body").innerHTML, /実行元<\/td>\s*<td>(LM Studio|ds4)<\/td>\s*<td>(LM Studio|ds4)<\/td>/);
renderDetails();
assert.ok(elements.get("detail-body").innerHTML.includes("provider-lmstudio"));
assert.ok(elements.get("detail-body").innerHTML.includes("provider-ds4"));
renderDetailPane(fourProviders.records[2]);
assert.ok(elements.get("detail-pane").innerHTML.includes("provider-omlx"));
assert.ok(!elements.get("detail-pane").innerHTML.includes("LM Studio Parallelism"));
renderDetailPane(providers.records[1]);
assert.ok(elements.get("detail-pane").innerHTML.includes("provider-ds4"));
assert.ok(!elements.get("detail-pane").innerHTML.includes("LM Studio Model"));
assert.ok(!elements.get("detail-pane").innerHTML.includes("LM Studio Parallelism"));
renderDetailPane(providers.records[0]);
assert.ok(elements.get("detail-pane").innerHTML.includes("LM Studio Parallelism"));
renderDetailPane(unknownProvider.records[0]);
assert.match(elements.get("detail-pane").innerHTML, /provider-unknown[^>]*>不明<\/span>/);
filters.query = "LM Studio";
assert.deepEqual(filteredRecords().map(record => record.provider), ["lmstudio"]);
filters.query = "";
renderColdWarm();
renderStability();
assert.ok(visibleText(elements.get("coldwarm-body").innerHTML).includes("LM Studio"));
assert.ok(visibleText(elements.get("stability-body").innerHTML).includes("ds4"));
state.leaderboardFilterTouched = true;
state.leaderboardSelectedModels = [];
renderLeaderboard();
assert.ok(elements.get("leaderboard-body").innerHTML.includes('colspan="13"'));

// Visible names are independent of internal condition/provider/evaluation keys.
const friendlyName = "Qwopus3.8-27B-Flash-oQ4e-mtp";
const groupA = "aeb093b70b31:38e7da8d51844ee2be88e66a43bead82";
const groupB = "changed-conditions:another-run";
const cleanRun = {...first, model: friendlyName, provider: "omlx", run_id: "run-a",
  started_at: "2026-09-15T01:00:00Z", evaluation: {name: "inspect", version: "0.3.263"},
  model_info: {requested_model: friendlyName, identifier: friendlyName, display_name: friendlyName, format: "MLX"},
  comparison: {group_id: groupA, unknown_fields: ["cache.state"]}};
const cleanPeer = {...cleanRun, run_id: "run-b", started_at: "2026-09-15T02:00:00Z",
  comparison: {group_id: groupB, unknown_fields: ["cache.state"]}, conditions: second.conditions};
reportData = buildReportPayload([cleanRun, cleanPeer]);
viewData = reportData;
state.leaderboardFilterTouched = false;
const keys = reportData.records.map(record => record.comparison_model);
assert.notEqual(keys[0], keys[1]);
assert.ok(keys[0].includes(groupA));
assert.equal(reportData.summary.total_models, 2);
assert.equal(modelNameText(keys[0]), friendlyName);
assert.equal(modelNameText(keys[1]), friendlyName);
assert.equal(modelChoiceText(keys[0]), `${friendlyName} · oMLX · 測定 1`);
assert.equal(modelChoiceText(keys[1]), `${friendlyName} · oMLX · 測定 2`);
viewData = buildReportPayload([cleanPeer]);
assert.equal(modelPresentation(keys[1]).measurement, "測定 2");
viewData = reportData;
renderLeaderboard();
renderCompare();
renderColdWarm();
renderStability();
renderDetails();
buildDetailFilters();
for (const id of ["leaderboard-body", "leaderboard-model-options", "coldwarm-body", "stability-body",
                  "detail-body", "compare-left-model", "detail-model-filter"]) {
  const text = visibleText(elements.get(id).innerHTML);
  assert.ok(text.includes(friendlyName), id);
  assert.ok(!text.includes(`${friendlyName}-mlx`), id);
  assert.ok(!text.includes("Inspect AI"), id);
  assert.ok(!text.includes(groupA) && !text.includes(groupB), id);
}
assert.ok(visibleText(elements.get("leaderboard-body").innerHTML).includes("測定 1"));
assert.ok(visibleText(elements.get("leaderboard-body").innerHTML).includes("測定 2"));
assert.equal(elements.get("compare-left-heading").textContent, friendlyName);
assert.ok(elements.get("compare-left-model").innerHTML.includes(escapeHtml(keys[0])));
filters.model = keys[1];
assert.equal(filteredRecords().length, 1);
assert.equal(filteredRecords()[0].run_id, "run-b");
filters.model = "";
renderDetailPane(reportData.records[0]);
const detailHtml = elements.get("detail-pane").innerHTML;
assert.match(detailHtml, /<h2 class="detail-pane-title">Qwopus3.8-27B-Flash-oQ4e-mtp<\/h2>/);
assert.ok(detailHtml.includes("測定条件ID") && detailHtml.includes(groupA));
assert.ok(detailHtml.includes("評価基盤") && detailHtml.includes("Inspect AI"));
const transcriptRecord = {...reportData.records[0], inspect: {log_path: "/logs/private.eval",
  log_storage: {format: "eval", compression: ["zstd"], file_bytes: 2048}},
  prompt_text: "fixture-prompt-body", reasoning_text: "fixture-reasoning-body", response_text: "fixture-response-body"};
renderDetailPane(transcriptRecord);
const compactDetails = elements.get("detail-pane").innerHTML;
assert.ok(compactDetails.includes("計測サマリー"));
assert.ok(compactDetails.includes("data-inspect-scope="));
assert.ok(compactDetails.includes("Zstandard · 2.0 KiB"));
for (const text of ["fixture-prompt-body", "fixture-reasoning-body", "fixture-response-body", "Tool Activity", "Predicted Answer"]) {
  assert.ok(!compactDetails.includes(text), text);
}
const question = {question_id: "q-one", status: "success", benchmark_score: 1, total_latency_ms: 120,
  prompt_tokens: 20, completion_tokens: 4, tool_call_count: 2, inspect: {log_path: "/logs/q-one.eval"}};
renderDetailPane({...transcriptRecord, benchmark_mode: "docker_task", benchmark_correct_count: 1,
  benchmark_incorrect_count: 0, benchmark_error_count: 1,
  question_results: [question, {...question, question_id: "q-two", status: "timeout", error: "fixture timeout", benchmark_score: null}]});
const taskDetails = elements.get("detail-pane").innerHTML;
assert.ok(taskDetails.includes("50.0%"));
assert.ok(taskDetails.includes("問題別の成績"));
assert.ok(taskDetails.includes("fixture timeout"));
assert.ok(taskDetails.includes("q-one.eval"));
assert.ok(taskDetails.includes("Prefill"));
assert.ok(taskDetails.includes("TPOT"));
assert.ok(!taskDetails.includes("fixture-response-body"));
assert.ok(!telemetryRunLabel(reportData.history_runs[0]).includes(groupA));
assert.ok(!telemetryGroupSubtitle({model: keys[0], provider: "omlx"}).includes(groupA));
const averages = telemetryAverageTableRows([{model: keys[0], provider: "omlx", key: "fixture", executionCount: 1}], "fixture", "runs");
assert.ok(visibleText(averages).includes(friendlyName));
assert.ok(!visibleText(averages).includes(groupA));
reportData = buildReportPayload([cleanRun]);
viewData = reportData;
assert.equal(modelPresentation(keys[0]).measurement, "");
assert.ok(!renderModelName(keys[0]).includes("測定"));
const hostile = {...cleanRun, model: '<img src=x onerror="alert(1)">', model_info: null};
reportData = buildReportPayload([hostile]);
viewData = reportData;
assert.ok(!renderModelName(reportData.records[0].comparison_model).includes("<img"));
const measurement = {version: 1, pp_tps: 200, tg_tps: 40, ttft_ms: 100, tpot_ms: 25, e2e_ms: 1000,
  input_tokens: 1024, output_tokens: 40, cached_prompt_tokens: null, cache_state: "unknown", total_tps: 1064,
  sources: {pp_tps: "api.timings.prompt", tg_tps: "api.timings.predicted_per_second"}};
const sample = {...first.records[0], metrics: measurement, benchmark_mode: "performance", target_input_tokens: 1024,
  concurrency_actual: 1, concurrency_requested: 1, phase: "warm", sample_id: "pp1024-c1-r1", inspect: {}};
const performanceRun = {...first, benchmark_mode: "performance", records: [sample,
  {...sample, target_input_tokens: 4096}, {...sample, concurrency_actual: 2, concurrency_requested: 2},
  {...sample, metrics: {...measurement, sources: {...measurement.sources, pp_tps: "client_ttft_estimate"}}}]};
reportData = buildReportPayload([performanceRun]); viewData = reportData;
assert.equal(currentModelRows().length, 4);
assert.equal(performanceGroups(currentRecords()).length, 4);
assert.ok(assessComparison(currentRecords()[0], currentRecords()[1], "runtime").differences.some(d => d.field === "measurement.scenario.target_input_tokens" && !d.axis));
assert.match(performanceCell([sample], "pp_tps"), /API/);
assert.match(performanceCell([sample], "cached_prompt_tokens"), /未取得/);
assert.equal(performanceMetrics({prompt_tokens: 21, completion_tokens: 11, ttft_ms: 500, total_latency_ms: 1500}).pp_tps, 42);
assert.equal(performanceMetrics({}).pp_tps, null);
assert.notEqual(recordKey(sample), recordKey({...sample, sample_id: "pp1024-c1-r2"}));
assert.ok(inspectButton(sample).includes("pp1024-c1-r1"));
state.leaderboardFilterTouched = false;
renderLeaderboard();
const cells = elements.get("leaderboard-body").innerHTML;
assert.ok(cells.includes("client_ttft_estimate"));
assert.ok(!visibleText(cells).includes("概算"));
assert.ok(!visibleText(cells).includes("TTFT換算"));
assert.ok(!visibleText(cells).includes("cache"));
assert.ok(!visibleText(cells).includes("API"));
assert.ok(!cells.includes("Peak Mem"));
assert.ok(cells.includes("最新ログ"));
assert.equal(elements.get("concurrency-section").hidden, true);
assert.equal(elements.get("quality-section").hidden, true);
assert.equal(elements.get("concurrency-body").innerHTML, "");
assert.equal(elements.get("quality-body").innerHTML, "");
const previousSample = {...sample, started_at: "2026-09-15T00:00:00Z", inspect: {log_path: "/logs/previous.eval"}};
const latestSample = {...sample, started_at: "2026-09-15T01:00:00Z", inspect: {log_path: "/logs/latest.eval"}};
assert.ok(latestInspectButton([previousSample, latestSample]).includes("latest.eval"));
assert.ok(!latestInspectButton([latestSample, previousSample]).includes("previous.eval"));
const baseCohort = {...sample, run_id: "run", prompt_sha256: "same", batch_id: "base",
  batch_metrics: {eligible_for_comparison: true, requests: 1, output_tps: 10}};
const multi = {...baseCohort, batch_id: "multi", concurrency_actual: 2, concurrency_requested: 2,
  batch_metrics: {eligible_for_comparison: true, requests: 2, output_tps: 16}};
assert.equal(concurrencyRows([baseCohort, multi, {...multi, sample_id: "r2"}])[1].speedup, 1.6);
assert.equal(concurrencyRows([baseCohort, {...multi, concurrency_actual: 1,
  batch_metrics: {...multi.batch_metrics, requests: 1, eligible_for_comparison: false}}])[1].speedup, null);
assert.equal(concurrencyRows([baseCohort, {...multi, prompt_sha256: "different"},
  {...multi, prompt_sha256: "different", sample_id: "r2"}])[1].speedup, null);
// Dynamic links in the leaderboard and details use the same Inspect handler.
let inspectClick, openedTab, requestedScroll = false;
document.addEventListener = (event, callback) => { if (event === "click") inspectClick = callback; };
for (const id of ["inspect-all", "inspect-refresh", "inspect-log-select"]) elements.set(id, {addEventListener: () => {}});
elements.set("inspect-panel", {scrollIntoView: () => { requestedScroll = true; }});
displayInspectLog = () => {};
showTab = tab => { openedTab = tab; };
bindInspectControls();
const selectedScope = {run_id: "accepted-run", question_id: "q-one", preferred_path: "/logs/q-one.eval"};
inspectClick({target: {closest: () => ({dataset: {inspectScope: JSON.stringify(selectedScope)}})}});
assert.deepEqual(state.inspectScope, selectedScope);
assert.equal(openedTab, "inspect");
assert.equal(requestedScroll, true);
'''
            path.write_text("const document = {getElementById: () => null};\n" + functions + checks)
            checked = subprocess.run([shutil.which("node"), str(path)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
