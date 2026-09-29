// Performance tables use versioned observations, never the legacy prompt-speed aliases.
function performanceMetrics(record) {
  if (record.metrics?.version === 1) return record.metrics;
  const input = normalizeNumeric(record.prompt_tokens);
  const output = normalizeNumeric(record.completion_tokens);
  const ttft = normalizeNumeric(record.ttft_ms);
  const e2e = normalizeNumeric(record.total_latency_ms);
  if (record.benchmark_mode === "docker_task") {
    // Task totals cover every conversation and tool call in the attempt. They
    // cannot establish prefill/decode speed or per-request TTFT/TPOT.
    return {version: 0, input_tokens: input, output_tokens: output, e2e_ms: e2e,
      pp_tps: null, tg_tps: null, ttft_ms: null, tpot_ms: null,
      total_tps: input != null && output != null && e2e > 0 ? (input + output) / (e2e / 1000) : null,
      cache_state: "unknown", cached_prompt_tokens: null,
      sources: {input_tokens: "task_total (all questions and conversation turns)",
        output_tokens: "task_total (all questions and conversation turns)",
        e2e_ms: "task_elapsed (includes conversation and tool execution)",
        total_tps: "task_total (input + output) / task elapsed time (includes tools)",
        pp_tps: "unavailable", tg_tps: "unavailable", ttft_ms: "unavailable", tpot_ms: "unavailable"}};
  }
  const window = e2e != null && ttft != null && e2e > ttft ? e2e - ttft : null;
  return {version: 0, input_tokens: input, output_tokens: output, ttft_ms: ttft, e2e_ms: e2e,
    pp_tps: computeTps(input, ttft), tg_tps: output > 1 ? computeTps(output - 1, window) : null,
    tpot_ms: output > 1 && window != null ? window / (output - 1) : null,
    total_tps: input != null && output != null ? computeTps(input + output, e2e) : null,
    cache_state: "unknown", cached_prompt_tokens: null,
    sources: {pp_tps: "client_ttft_estimate (legacy input / TTFT)", tg_tps: "client_estimate (legacy output - 1 / post-first-chunk interval)"}};
}

function performanceIdentity(record) {
  if (record.benchmark_mode !== "performance") return "";
  const metrics = performanceMetrics(record);
  return JSON.stringify([record.target_input_tokens, record.concurrency_requested, record.concurrency_actual,
    record.phase, record.measurement?.inference_stage, record.measurement?.segment,
    metrics.version, metrics.cache_state, metrics.sources?.pp_tps, metrics.sources?.tg_tps]);
}

function performanceGroups(records) {
  const grouped = new Map();
  for (const record of records) {
    if (record.partial) continue;
    const m = performanceMetrics(record);
    const key = JSON.stringify([record.comparison_model, record.benchmark_mode,
      record.benchmark_id || record.benchmark_title, record.question_count,
      (record.question_results || []).map(q => q.question_id).sort(),
      record.phase, record.target_input_tokens,
      record.concurrency_actual || 1, record.concurrency_requested || 1,
      record.measurement?.inference_stage, record.measurement?.segment,
      m.version, m.scope, m.cache_state, m.sources?.pp_tps, m.sources?.tg_tps,
      m.source_details || null]);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(record);
  }
  return [...grouped.values()];
}

function performanceValues(records, field) {
  return records.filter(r => r.status === "success").map(r => performanceMetrics(r)[field])
    .filter(v => typeof v === "number" && Number.isFinite(v));
}

function performanceMedian(records, field) {
  return median(performanceValues(records, field));
}

function performanceCell(records, field, digits = 1, scale = 1) {
  const values = performanceValues(records, field).map(v => v / scale);
  if (!values.length) {
    const detail = ["pp_tps", "tg_tps", "ttft_ms", "tpot_ms"].includes(field)
      && records.some(r => r.benchmark_mode === "docker_task")
      ? "未取得。この履歴のタスク全体の所要時間から、個々の推論の速度・待ち時間は算出できません。"
      : "未取得";
    return `<span class="muted" title="${escapeHtml(detail)}">—</span>`;
  }
  const metrics = performanceMetrics(records[0]);
  const source = String(metrics.sources?.[field] || "client observation");
  const method = metrics.scope === "inference_requests"
    ? source.startsWith("request_median") ? "各試行内の推論リクエストごとの計測値の中央値"
      : field === "total_tps" ? "合計入出力トークン数 / 推論時間の合計（ツール実行を除く）"
      : "各試行内の推論リクエストの合計（ツール実行を除く）"
    : field === "pp_tps" ? source.startsWith("api.") ? "API報告値" : "入力トークン数 / TTFT（推定）"
    : field === "tg_tps" ? source.startsWith("api.") ? "API報告値" : "(出力トークン数−1) / 初回チャンク後の時間（推定）"
    : field === "tpot_ms" ? "初回チャンク後の時間 / (出力トークン数−1)（推定）" : "";
  const sourceDetails = metrics.source_details?.[field]?.join("; ") || "";
  const detail = `${method}; n=${values.length}; min=${Math.min(...values).toFixed(digits)}; max=${Math.max(...values).toFixed(digits)}; ${source}; ${sourceDetails}`;
  return `<span title="${escapeHtml(detail)}">${median(values).toFixed(digits)}</span>`;
}

function performanceTestLabel(record) {
  if (record.benchmark_mode === "docker_task") {
    const title = record.benchmark_title || record.benchmark_id || "タスク";
    const count = record.question_count || record.question_results?.length;
    return `${title}${count ? ` / ${count}問` : ""}`;
  }
  return record.target_input_tokens ? `≈${record.target_input_tokens} tok / ${record.concurrency_actual}並列` : "通常プロンプト";
}

function performanceSortRow(group) {
  const r = group[0];
  return {group, model: modelNameText(r.comparison_model, r), provider_sort: providerLabel(r.provider),
    test: r.benchmark_mode === "docker_task" ? performanceTestLabel(r) : r.target_input_tokens || 0, phase: r.phase,
    success_rate: group.filter(x => x.status === "success").length / group.length,
    ...Object.fromEntries(["input_tokens", "output_tokens", "pp_tps", "tg_tps", "ttft_ms", "tpot_ms", "e2e_ms", "total_tps"]
      .map(field => [field, performanceMedian(group, field)]))};
}

function concurrencyRows(records) {
  const cohorts = new Map();
  for (const r of records) {
    if (r.benchmark_mode === "performance" && r.batch_id) {
      const key = `${r.run_id}:${r.batch_id}`;
      if (!cohorts.has(key)) cohorts.set(key, []);
      cohorts.get(key).push(r);
    }
  }
  const groups = [...cohorts.values()];
  function signature(group) {
    const r = group[0];
    return JSON.stringify([r.run_id, r.target_input_tokens, r.prompt_sha256, r.phase,
      r.measurement?.inference_stage, r.measurement?.segment,
      [...new Set(group.map(x => performanceMetrics(x).cache_state))].sort(),
      [...new Set(group.map(x => JSON.stringify(performanceMetrics(x).sources)))].sort()]);
  }
  return groups.map(group => {
    const r = group[0], b = r.batch_metrics;
    const eligible = b?.eligible_for_comparison && b.requests === group.length;
    const baselines = groups.filter(g => g[0].concurrency_actual === 1 && g[0].concurrency_requested === 1
      && g[0].batch_metrics?.eligible_for_comparison && signature(g) === signature(group))
      .map(g => g[0].batch_metrics.output_tps).filter(x => typeof x === "number" && Number.isFinite(x) && x > 0);
    const baseline = median(baselines);
    return {record: r, records: group, batch: b, speedup: eligible && baseline && typeof b.output_tps === "number" ? b.output_tps / baseline : null};
  });
}

function renderPerformanceLeaderboard() {
  renderLeaderboardFilter();
  const selected = new Set(state.leaderboardSelectedModels);
  const records = currentRecords().filter(r => selected.has(r.comparison_model)
    && (!r.run_status || ["completed", "legacy"].includes(r.run_status)));
  const groups = sortRows(performanceGroups(records).map(performanceSortRow), state.leaderboardSort);
  document.getElementById("leaderboard-body").innerHTML = groups.length ? groups.map(({group}) => {
    const r = group[0], m = performanceMetrics(r);
    const stage = r.measurement?.inference_stage || "unknown";
    const label = performanceTestLabel(r);
    return `<tr><td>${renderModelName(r.comparison_model, r)}</td><td>${renderProviderBadge(r.provider)}</td>
      <td>${escapeHtml(label)}</td><td title="推論段階: ${escapeHtml(stage)}; cache: ${escapeHtml(m.cache_state)}">${escapeHtml(r.phase)}</td>
      <td>${group.filter(x => x.status === "success").length}/${group.length}${group.some(x => x.status !== "success") ? `<small class="muted"> 失敗 ${group.filter(x => x.status !== "success").length}</small>` : ""}</td>
      <td>${performanceCell(group, "input_tokens", 0)} / ${performanceCell(group, "output_tokens", 0)}</td>
      <td>${performanceCell(group, "pp_tps")}</td><td>${performanceCell(group, "tg_tps")}</td>
      <td>${performanceCell(group, "ttft_ms")}</td><td>${performanceCell(group, "tpot_ms", 2)}</td>
      <td>${performanceCell(group, "e2e_ms", 3, 1000)}</td><td>${performanceCell(group, "total_tps")}</td><td>${latestInspectButton(group)}</td></tr>`;
  }).join("") : '<tr><td colspan="13" class="muted">選択した条件の推論性能データはありません。</td></tr>';
  document.getElementById("leaderboard-note").textContent = "同じ条件の試行を集計した中央値です。計測元・算出方法・範囲・nは各値のツールチップで確認できます。ログは最新試行を開きます。"
    + (records.some(r => performanceMetrics(r).scope === "inference_requests")
      ? " タスクのPP/TG TPS・TTFT・TPOTは各試行内の推論リクエストの中央値、入出力とE2Eは合計です。E2Eと総合TPSはツール実行時間を除きます。" : "")
    + (records.some(r => r.benchmark_mode === "docker_task" && performanceMetrics(r).version === 0)
      ? " 旧タスク履歴のE2Eと総合TPSはツール実行を含む値です。旧履歴に残っていないPP/TG TPS・TTFT・TPOTは復元できないため「—」で表示します。" : "");
  updateSortHeaders("#leaderboard-table", state.leaderboardSort);
  const cohorts = sortRows(concurrencyRows(records).map(row => {
    const r = row.record, b = row.batch;
    return {...row, model: modelNameText(r.comparison_model, r), provider_sort: providerLabel(r.provider),
      target_input_tokens: r.target_input_tokens, trial: `${r.phase} ${r.iteration}`,
      concurrency_actual: r.concurrency_actual, concurrency_requested: r.concurrency_requested,
      input_tps: b?.input_tps, output_tps: b?.output_tps, elapsed_sec: b?.elapsed_sec,
      ttft_ms: performanceMedian(row.records, "ttft_ms"), e2e_ms: performanceMedian(row.records, "e2e_ms"),
      error_count: row.records.filter(x => x.status !== "success").length};
  }), state.concurrencySort);
  document.getElementById("concurrency-section").hidden = !cohorts.length;
  document.getElementById("concurrency-body").innerHTML = cohorts.length ? cohorts.map(({record: r, records: group, batch: b, speedup}) => `<tr>
    <td>${renderModelName(r.comparison_model, r)}</td><td>${renderProviderBadge(r.provider)}</td><td>≈${escapeHtml(r.target_input_tokens)}</td>
    <td>${escapeHtml(r.phase)} #${escapeHtml(r.iteration)}</td><td title="開始時刻の差 ${formatNumber(b?.dispatch_spread_ms, 1)} ms">${escapeHtml(r.concurrency_actual)} / ${escapeHtml(r.concurrency_requested)}<br><small class="muted">同時実行ピーク ${escapeHtml(b?.peak_inflight ?? "未取得")}</small></td>
    <td>${formatNumber(b?.input_tps, 1)}</td><td>${formatNumber(b?.output_tps, 1)}</td><td>${speedup == null ? "—" : `${speedup.toFixed(2)}×`}</td>
    <td>${performanceCell(group, "ttft_ms")}</td><td>${performanceCell(group, "e2e_ms", 3, 1000)}</td>
    <td>${formatNumber(b?.elapsed_sec, 3)}</td><td>${group.filter(x => x.status !== "success").length}/${group.length}${b?.complete ? "" : " · 未完了"}${r.concurrency_actual !== r.concurrency_requested ? " · 再開時の部分測定" : ""}</td><td>${latestInspectButton(group)}</td></tr>`).join("") : "";
  updateSortHeaders("#concurrency-table", state.concurrencySort);
  const quality = sortRows(selectedLeaderboardRows().filter(r => r.overall_benchmark_total_count > 0)
    .map(r => ({...r, display_model: modelNameText(r.model, r), provider_sort: providerLabel(r.provider)})), state.qualitySort);
  document.getElementById("quality-section").hidden = !quality.length;
  document.getElementById("quality-body").innerHTML = quality.length ? quality.map(r => `<tr><td>${renderModelName(r.model, r)}</td><td>${renderProviderBadge(r.provider)}</td>
    <td>${r.overall_benchmark_correct_count}/${r.overall_benchmark_total_count}</td><td>${formatPercent(r.benchmark_correct_rate)}</td><td>${r.overall_benchmark_error_count ?? "—"} (${formatPercent(r.benchmark_error_rate)})</td><td>${formatSec(r.warm_mean_total_latency_ms)}</td><td>${latestInspectButton(records.filter(record => record.comparison_model === r.model))}</td></tr>`).join("") : "";
  updateSortHeaders("#quality-table", state.qualitySort);
}
