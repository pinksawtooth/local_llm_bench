// The benchmark owns measurements and comparisons; Inspect owns transcripts.
function inspectStorageLabel(storage) {
  if (!storage || storage.status === "unavailable") return "圧縮情報は未取得";
  const bytes = normalizeNumeric(storage.file_bytes);
  const size = bytes == null ? "容量不明" : bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KiB` : `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  const compression = (Array.isArray(storage.compression) ? storage.compression : []).map(x => x === "zstd" ? "Zstandard" : x).join(" / ");
  return `.eval · ${compression || "方式不明"} · ${size}`;
}

function latestInspectButton(records) {
  const ordered = [...records].filter(r => r.inspect || r.evaluation?.name === "inspect")
    .sort((a, b) => String(b.started_at || b.run_started_at || "").localeCompare(String(a.started_at || a.run_started_at || ""))
      || (b.iteration || 0) - (a.iteration || 0));
  return ordered.length ? inspectButton(ordered[0], null, "最新ログ") : '<span class="muted">未記録</span>';
}

function renderDetailPane(record) {
  const pane = document.getElementById("detail-pane");
  if (!record) {
    pane.innerHTML = '<div class="detail-empty">実行を選ぶと成績・速度・測定条件を表示します。会話・思考・ツール履歴は対応するInspectログで確認できます。</div>';
    return;
  }
  const modelInfo = normalizeModelInfo(record.model_info, record.model) || modelInfoFor(record.comparison_model || record.model);
  const model = record.comparison_model || record.model;
  const questions = (Array.isArray(record.question_results) ? record.question_results : []).filter(q => q && typeof q === "object");
  const isTask = record.benchmark_mode === "docker_task" || questions.length > 0;
  const correct = normalizeNumeric(record.benchmark_correct_count);
  const incorrect = normalizeNumeric(record.benchmark_incorrect_count);
  const errors = normalizeNumeric(record.benchmark_error_count);
  const scoreCount = [correct, incorrect, errors].every(x => x != null) ? correct + incorrect + errors : null;
  const rate = scoreCount > 0 && correct != null ? correct / scoreCount : null;
  const latency = isTask ? escapeHtml(formatMs(record.total_latency_ms)) : performanceCell([record], "e2e_ms", 3, 1000) + " s";
  const cards = isTask ? [
    ["正答率", formatPercent(rate)],
    ["正答 / 全問", `${correct ?? "—"} / ${scoreCount ?? "—"}`],
    ["誤答 / エラー", `${incorrect ?? "—"} / ${errors ?? "—"}`],
    ["課題セットの所要時間", latency],
  ] : [
    ["Prefill · pp TPS", performanceCell([record], "pp_tps")],
    ["Decode · tg TPS", performanceCell([record], "tg_tps")],
    ["TTFT (ms)", performanceCell([record], "ttft_ms")],
    ["E2E", latency],
  ];
  const context = renderDetailKvRows([
    ["実行元", renderProviderBadge(record.provider)],
    ["評価基盤", escapeHtml(record.evaluation?.name === "inspect" ? `Inspect AI ${record.evaluation.version || ""}` : "未記録")],
    ["測定条件ID", escapeHtml(record.comparison?.group_id || "未記録")],
    ["Run ID / 状態", escapeHtml(`${record.run_id || "—"} / ${record.run_status || "未記録"}`)],
    ["開始", formatTime(record.run_started_at || record.started_at)],
    ["Benchmark ID", escapeHtml(record.benchmark_id || "—")],
    ["推論段階", escapeHtml(record.measurement?.inference_stage || (isTask ? "問題別に記録" : "不明"))],
    ["キャッシュ状態", escapeHtml(record.metrics?.cache_state || record.measurement?.cache_state || "不明")],
    ["エンジン / 版", escapeHtml(`${record.conditions?.runtime?.engine || "不明"} / ${record.conditions?.runtime?.version || "不明"}`)],
    ["Model SHA-256", escapeHtml(record.conditions?.model?.artifact?.sha256 || "不明")],
    ["形式 / 量子化", escapeHtml(`${modelFormat(modelInfo) || "不明"} / ${modelQuantization(modelInfo) || "不明"}`)],
    ...(record.provider === "lmstudio" ? [["LM Studio Parallelism", escapeHtml(String(record.lmstudio_parallelism ?? "不明"))]] : []),
  ]);
  const timings = renderDetailKvRows([
    ["ロード時間（run 合計）", formatRunSeconds(record.timings?.load_wall_sec)],
    ["ロード直後の初回（run 合計）", formatRunSeconds(record.timings?.first_after_load_wall_sec)],
    ["反復推論（run 合計）", formatRunSeconds(record.timings?.repeat_wall_sec)],
    ["準備推論（統計から除外）", formatRunSeconds(record.timings?.warmup_wall_sec)],
  ]);
  const metrics = renderDetailKvRows(isTask ? [
    ["入力 / 出力 (tok)", `${formatNumber(record.prompt_tokens, 0)} / ${formatNumber(record.completion_tokens, 0)}`],
    ["ツール呼び出し数", formatNumber(record.tool_call_count, 0)],
    ["Prefill · pp TPS", performanceCell([record], "pp_tps")],
    ["Decode · tg TPS", performanceCell([record], "tg_tps")],
    ["TTFT (ms)", performanceCell([record], "ttft_ms")],
    ["TPOT (ms/token)", performanceCell([record], "tpot_ms", 2)],
    ...(record.metrics?.scope === "inference_requests" ? [
      ["推論E2Eの合計 (s)", performanceCell([record], "e2e_ms", 3, 1000)],
      ["総合 TPS (入力+出力)/推論E2E", performanceCell([record], "total_tps")],
    ] : []),
  ] : [
    ["入力 / 出力 (tok)", `${performanceCell([record], "input_tokens", 0)} / ${performanceCell([record], "output_tokens", 0)}`],
    ["TPOT (ms/token)", performanceCell([record], "tpot_ms", 2)],
    ["総合 TPS (入力+出力)/E2E", performanceCell([record], "total_tps")],
    ["キャッシュ済み入力 (tok)", performanceCell([record], "cached_prompt_tokens", 0)],
  ]);
  const questionRows = questions.map(q => {
    const error = q.status !== "success";
    const verdict = error ? "実行エラー" : q.benchmark_score === 1 ? "正答" : q.benchmark_score === 0 ? "誤答" : "未採点";
    return `<tr><td>${escapeHtml(q.question_id || "—")}</td><td>${verdict}${error ? `<small class="muted"> ${escapeHtml(truncateText(q.error_signature || q.error || q.status, 100))}</small>` : ""}</td>
      <td>${formatMs(q.total_latency_ms)}</td><td>${formatNumber(q.prompt_tokens, 0)} / ${formatNumber(q.completion_tokens, 0)}</td>
      <td>${formatNumber(q.tool_call_count, 0)}</td><td>${inspectButton(record, q) || '<span class="muted">未記録</span>'}</td></tr>`;
  }).join("");
  const logUrl = resolveLogUrl(record.log_path);
  const audit = renderDetailDisclosure("測定条件・診断・計測定義", `
    <table class="detail-kv-table"><tbody>${context}</tbody></table>
    <pre class="catalog-prompt">${escapeHtml(JSON.stringify({conditions: record.conditions, comparison: record.comparison,
      preflight: record.preflight, lifecycle: record.lifecycle, metric_sources: record.metrics?.sources,
      metric_source_details: record.metrics?.source_details, metric_aggregation: record.metrics?.aggregation}, null, 2))}</pre>
    ${logUrl ? `<a href="${escapeHtml(logUrl)}" target="_blank" rel="noopener noreferrer">監査JSONを開く</a>` : ""}`);
  const error = record.error || record.error_signature;
  pane.innerHTML = `<div class="detail-pane-shell">
    <div class="detail-pane-header"><div class="detail-pane-title-block">
      <h2 class="detail-pane-title">${escapeHtml(modelNameText(model, record))}</h2>
      <div class="detail-pane-subtitle">${escapeHtml(record.benchmark_title || modelPresentation(model, record).measurement || "選択した試行")}</div>
      <div class="detail-row-tags">${renderProviderBadge(record.provider)}${renderStatusPill(record.status)}${renderPhasePill(record.phase, record.iteration)}</div>
    </div><div class="detail-pane-actions">${inspectButton(record)}</div></div>
    <div class="detail-summary-grid">${cards.map(([label, value]) => `<div class="detail-summary-item"><div class="detail-metric-label">${label}</div><div class="detail-metric-value">${value}</div></div>`).join("")}</div>
    <section class="detail-section"><h3 class="detail-section-heading">計測サマリー</h3><table class="detail-kv-table"><tbody>${metrics}</tbody></table>
      <div class="table-note">計測元・算出方法は各値のツールチップと計測定義で確認できます。未取得の値は補完しません。</div></section>
    ${questions.length ? `<section class="detail-section"><h3 class="detail-section-heading">問題別の成績</h3><div style="overflow-x:auto"><table>
      <thead><tr><th>問題</th><th>判定</th><th>所要時間</th><th>入力 / 出力 (tok)</th><th>ツール数</th><th>Inspect</th></tr></thead><tbody>${questionRows}</tbody></table></div></section>` : ""}
    <section class="detail-section"><h3 class="detail-section-heading">会話・思考・ツール履歴</h3>
      <p class="muted">入力・出力の全文、思考、ツール入出力、採点根拠は対応するInspectログで確認できます。</p>
      <div>${inspectButton(record) || '<span class="muted">Inspectログの参照は未記録です。</span>'}</div>
      ${record.inspect?.log_storage ? `<p class="muted">${escapeHtml(inspectStorageLabel(record.inspect.log_storage))}</p>` : ""}
    </section>
    ${error ? renderDetailDisclosure("実行エラー", `<pre class="catalog-prompt">${escapeHtml(error)}</pre>`, {open: true}) : ""}
    ${renderDetailDisclosure("ロード・初回・反復の時間", `<table class="detail-kv-table"><tbody>${timings}</tbody></table>`)}
    ${audit}
  </div>`;
}
