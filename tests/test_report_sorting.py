"""Click the rendered table headers with fixture data, without a browser/server."""
import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from local_llm_bench.report import render_report_html


class TableHeaders(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tables = {}
        self.table = None
        self.header = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.table = attrs.get("id")
            if self.table:
                self.tables[self.table] = []
        if self.table and tag == "th":
            self.header = []
            self.tables[self.table].append(self.header)
        if self.header is not None and "data-sort" in attrs:
            self.header.append({"tagName": tag.upper(), "key": attrs["data-sort"]})

    def handle_endtag(self, tag):
        if tag == "th":
            self.header = None
        if tag == "table":
            self.table = None


class ReportSortingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js required for dashboard JavaScript tests")
    def test_header_clicks_sort_displayed_values_and_keep_missing_last(self):
        html = render_report_html("history.json")
        headers = TableHeaders(html).tables
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        functions, startup = script.split('    document.getElementById("reload")', 1)
        self.assertIn("bindTableSorters();", startup)
        checks = r'''
const assert = require("node:assert/strict");
const elements = new Map();
const tableHeaders = new Map();
function element(tagName = "DIV") {
  const classes = new Set(), attributes = {}, handlers = {};
  return {tagName, innerHTML: "", textContent: "", value: "", dataset: {}, attributes, handlers,
    querySelectorAll: () => [],
    classList: {toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name), contains: name => classes.has(name)},
    setAttribute: (name, value) => { attributes[name] = value; },
    addEventListener: (event, callback) => { handlers[event] = callback; }};
}
for (const [id, columns] of Object.entries(headerDefinitions)) {
  tableHeaders.set(id, columns.map(definitions => {
    const th = element("TH");
    th.controls = definitions.map(def => {
      const control = def.tagName === "TH" ? th : element(def.tagName);
      control.dataset.sort = def.key;
      control.closest = () => th;
      return control;
    });
    return th;
  }));
}
document.getElementById = id => {
  if (!elements.has(id)) elements.set(id, element());
  return elements.get(id);
};
document.querySelectorAll = selector => {
  const match = selector.match(/^#([\w-]+) (th|\[data-sort\])$/);
  if (!match) return [];
  const columns = tableHeaders.get(match[1]) || [];
  return match[2] === "th" ? columns : columns.flatMap(th => th.controls);
};
function header(table, key) {
  const found = (tableHeaders.get(table) || []).flatMap(th => th.controls).find(th => th.dataset.sort === key);
  assert.ok(found, `Missing sortable header: ${table}.${key}`);
  assert.equal(typeof found.handlers.click, "function", `Unbound header: ${table}.${key}`);
  return found;
}
function click(table, key) { header(table, key).handlers.click(); }
const names = ["Alpha", "Beta", "Zero", "Missing"];
function rowNames(body) {
  return (elements.get(body).innerHTML.match(/<tr>[\s\S]*?<\/tr>/g) || [])
    .map(row => names.find(name => row.includes(`>${name}<`)));
}
function record(pp, extra = {}) {
  return {phase: "warm", iteration: 1, status: "success", total_latency_ms: 2000,
    metrics: {version: 1, input_tokens: 100, output_tokens: 9, pp_tps: pp, tg_tps: pp,
      ttft_ms: pp, tpot_ms: pp, e2e_ms: 2000, total_tps: pp, cache_state: "unknown",
      sources: {pp_tps: "api.fixture", tg_tps: "api.fixture"}}, ...extra};
}
function run(name, records, extra = {}) {
  return {run_id: name, model: name, status: "completed", provider: "lmstudio", prompt_text: "fixture",
    conditions: {runtime: {engine: "fixture"}}, comparison: {group_id: name}, records, ...extra};
}
function display(runs) {
  reportData = buildReportPayload(runs); viewData = reportData;
  state.leaderboardFilterTouched = false;
  renderLeaderboard();
}
bindTableSorters();
const alpha = [record(10), record(100), record(20), record(9999, {status: "error"})];
alpha.forEach((r, i) => { r.iteration = i + 1; r.metrics.e2e_ms = 10000; r.metrics.output_tokens = 100; });
const missing = record(null);
Object.keys(missing.metrics).filter(key => typeof missing.metrics[key] === "number").forEach(key => {
  if (key !== "version") missing.metrics[key] = null;
});
const runs = [run("Beta", [record(18)], {provider: "omlx"}), run("Alpha", alpha),
  run("Missing", [missing]), run("Zero", [record(0)])];
display(runs);
const originalRecords = JSON.stringify(currentRecords());
click("leaderboard-table", "model");
assert.deepEqual(rowNames("leaderboard-body"), ["Alpha", "Beta", "Missing", "Zero"]);
click("leaderboard-table", "model");
assert.deepEqual(rowNames("leaderboard-body"), ["Zero", "Missing", "Beta", "Alpha"]);
for (const key of ["pp_tps", "tg_tps", "total_tps"]) {
  click("leaderboard-table", key);
  assert.deepEqual(rowNames("leaderboard-body"), ["Alpha", "Beta", "Zero", "Missing"]);
  assert.equal(header("leaderboard-table", key).attributes["aria-sort"], "descending");
  assert.ok(header("leaderboard-table", key).classList.contains("sort-desc"));
  click("leaderboard-table", key);
  assert.deepEqual(rowNames("leaderboard-body"), ["Zero", "Beta", "Alpha", "Missing"]);
  assert.equal(header("leaderboard-table", key).attributes["aria-sort"], "ascending");
}
for (const key of ["ttft_ms", "tpot_ms"]) {
  click("leaderboard-table", key);
  assert.deepEqual(rowNames("leaderboard-body"), ["Zero", "Beta", "Alpha", "Missing"]);
}
click("leaderboard-table", "e2e_ms");
assert.deepEqual(rowNames("leaderboard-body"), ["Beta", "Zero", "Alpha", "Missing"]);
click("leaderboard-table", "output_tokens");
assert.deepEqual(rowNames("leaderboard-body"), ["Beta", "Zero", "Alpha", "Missing"]);
assert.equal(header("leaderboard-table", "output_tokens").attributes["aria-pressed"], "true");
click("leaderboard-table", "input_tokens");
assert.equal(header("leaderboard-table", "output_tokens").attributes["aria-pressed"], "false");
const ppHeader = header("leaderboard-table", "pp_tps");
ppHeader.handlers.keydown({key: "Enter", preventDefault() {}});
assert.deepEqual(rowNames("leaderboard-body"), ["Alpha", "Beta", "Zero", "Missing"]);
ppHeader.handlers.keydown({key: " ", preventDefault() {}});
assert.deepEqual(rowNames("leaderboard-body"), ["Zero", "Beta", "Alpha", "Missing"]);
renderLeaderboard(); // Refresh/filter redraw preserves the selected sort.
assert.deepEqual(rowNames("leaderboard-body"), ["Zero", "Beta", "Alpha", "Missing"]);
assert.equal(JSON.stringify(currentRecords()), originalRecords);
for (const key of ["provider_sort", "test", "phase", "success_rate"]) click("leaderboard-table", key);

function cohort(name, output, elapsed) {
  return run(name, [record(output, {benchmark_mode: "performance", batch_id: name, sample_id: name,
    target_input_tokens: 1024, concurrency_requested: 1, concurrency_actual: 1,
    batch_metrics: {eligible_for_comparison: true, complete: true, requests: 1, output_tps: output,
      input_tps: output, elapsed_sec: elapsed}})], {benchmark_mode: "performance"});
}
display([cohort("Beta", 9, 2), cohort("Missing", null, null), cohort("Alpha", 100, 10), cohort("Zero", 0, 0)]);
click("concurrency-table", "output_tps");
assert.deepEqual(rowNames("concurrency-body"), ["Alpha", "Beta", "Zero", "Missing"]);
click("concurrency-table", "output_tps");
assert.deepEqual(rowNames("concurrency-body"), ["Zero", "Beta", "Alpha", "Missing"]);
assert.equal(header("concurrency-table", "output_tps").attributes["aria-sort"], "ascending");
click("concurrency-table", "elapsed_sec");
assert.deepEqual(rowNames("concurrency-body"), ["Zero", "Beta", "Alpha", "Missing"]);
for (const key of ["model", "provider_sort", "target_input_tokens", "trial", "concurrency_actual",
  "concurrency_requested", "input_tps", "speedup", "ttft_ms", "e2e_ms", "error_count"]) click("concurrency-table", key);

function task(name, correct, ms) {
  return run(name, [record(null, {benchmark_mode: "docker_task", total_latency_ms: ms,
    benchmark_total_count: 4, benchmark_correct_count: correct, benchmark_incorrect_count: 4 - correct,
    benchmark_error_count: 0, benchmark_score: correct / 4})], {benchmark_mode: "docker_task"});
}
display([task("Beta", 1, 2000), task("Zero", 0, 0), task("Alpha", 4, 10000)]);
assert.deepEqual(rowNames("quality-body"), ["Alpha", "Beta", "Zero"]);
click("quality-table", "benchmark_correct_rate");
assert.deepEqual(rowNames("quality-body"), ["Zero", "Beta", "Alpha"]);
click("quality-table", "warm_mean_total_latency_ms");
assert.deepEqual(rowNames("quality-body"), ["Zero", "Beta", "Alpha"]);
click("quality-table", "warm_mean_total_latency_ms");
assert.deepEqual(rowNames("quality-body"), ["Alpha", "Beta", "Zero"]);
for (const key of ["display_model", "provider_sort", "overall_benchmark_correct_count", "overall_benchmark_error_count"]) click("quality-table", key);
// Other existing table sorters keep missing/nonfinite values at the end as well.
for (const asc of [true, false]) {
  const rows = sortRows([{v: NaN}, {v: 9}, {v: null}, {v: 100}, {v: Infinity}, {v: 0}], {key: "v", asc});
  assert.deepEqual(rows.slice(0, 3).map(r => r.v), asc ? [0, 9, 100] : [100, 9, 0]);
}
display([]);
click("leaderboard-table", "pp_tps");
assert.ok(elements.get("leaderboard-body").innerHTML.includes("データはありません"));
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.js"
            path.write_text("const document = {getElementById: () => null};\n"
                            + "const headerDefinitions = " + json.dumps(headers) + ";\n"
                            + functions + checks, encoding="utf-8")
            checked = subprocess.run([shutil.which("node"), str(path)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
