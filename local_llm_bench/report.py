from __future__ import annotations

import json
import os
from pathlib import Path


def render_report_html(history_url: str) -> str:
    template = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Local LLM Bench Dashboard</title>
  <style>
    :root {
      --bg-primary: #1e1e1e;
      --bg-secondary: #252526;
      --card: #2d2d2d;
      --card-hover: #333333;
      --border: #3c3c3c;
      --border-accent: #007acc;
      --text: #d4d4d4;
      --muted: #808080;
      --accent: #0e639c;
      --accent-2: #007acc;
      --good: #4ec9b0;
      --bad: #f14c4c;
      --gold: #dcdcaa;
      --silver: #9cdcfe;
      --bronze: #ce9178;
      --mono: "JetBrains Mono", "Fira Code", "SF Mono", monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg-primary);
      color: var(--text);
      font-family: "Inter", "Noto Sans JP", system-ui, sans-serif;
      line-height: 1.5;
    }
    header {
      padding: 32px 32px 16px;
      border-bottom: 1px solid var(--border);
    }
    .header-content {
      max-width: 1400px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo-icon {
      width: 40px;
      height: 40px;
      background: var(--accent-2);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 18px;
      color: #ffffff;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 700;
      color: var(--text);
    }
    .header-meta {
      color: var(--muted);
      font-size: 13px;
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
    }
    main {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px 32px 48px;
    }
    .btn {
      background: var(--accent-2);
      color: #ffffff;
      border: none;
      border-radius: 4px;
      padding: 8px 16px;
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    .btn:hover { background: #1177bb; }
    .btn-secondary {
      background: var(--bg-secondary);
      color: var(--text);
      border: 1px solid var(--border);
    }
    .btn-secondary:hover { background: var(--card-hover); }
    .hidden { display: none !important; }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }
    .stat {
      padding: 20px;
      border-radius: 14px;
      background: var(--card);
      border: 1px solid var(--border);
      transition: all 0.2s ease;
    }
    .stat:hover {
      border-color: var(--border-accent);
      transform: translateY(-2px);
    }
    .stat .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }
    .stat .value {
      font-size: 28px;
      font-weight: 800;
      margin-top: 8px;
      color: var(--text);
    }
    .stat .extra {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      margin-top: 16px;
    }
    .source-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }
    .toolbar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }
    .leaderboard-toolbar {
      align-items: center;
      justify-content: space-between;
    }
    .leaderboard-filter-group {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    .filter-dropdown {
      position: relative;
    }
    .filter-dropdown summary {
      list-style: none;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .filter-dropdown summary::-webkit-details-marker {
      display: none;
    }
    .filter-dropdown[open] summary {
      background: var(--card-hover);
      border-color: var(--border-accent);
    }
    .filter-dropdown-menu {
      position: absolute;
      top: calc(100% + 8px);
      left: 0;
      min-width: 320px;
      max-width: min(420px, 82vw);
      padding: 14px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--bg-secondary);
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
      z-index: 6;
    }
    .filter-actions {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
    }
    .filter-option-list {
      display: grid;
      gap: 8px;
      max-height: 280px;
      overflow-y: auto;
    }
    .filter-option {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
      cursor: pointer;
    }
    .filter-option input {
      margin-top: 2px;
      accent-color: var(--accent);
    }
    .filter-option span {
      color: var(--text);
      font-size: 13px;
      line-height: 1.5;
      word-break: break-word;
    }
    .filter-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .tabs {
      display: inline-flex;
      gap: 4px;
      margin-top: 28px;
      padding: 4px;
      background: var(--bg-secondary);
      border-radius: 12px;
      border: 1px solid var(--border);
      flex-wrap: wrap;
    }
    .tab {
      padding: 10px 20px;
      border-radius: 8px;
      border: none;
      background: transparent;
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
      color: var(--muted);
      transition: all 0.2s ease;
    }
    .tab:hover {
      color: var(--text);
      background: rgba(255,255,255,0.05);
    }
    .tab.active {
      background: var(--accent-2);
      color: #ffffff;
    }
    .muted { color: var(--muted); }
    .tag {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 20px;
      background: rgba(255,255,255,0.08);
      color: var(--text);
      font-size: 12px;
      font-weight: 600;
      border: 1px solid var(--border);
    }
    .tag-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .load-state-error {
      color: #ffd6d6;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th {
      text-align: left;
      padding: 14px 12px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      font-weight: 600;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
      user-select: none;
    }
    th:hover { color: var(--text); }
    th.sort-asc::after { content: " ▲"; font-size: 10px; }
    th.sort-desc::after { content: " ▼"; font-size: 10px; }
    td {
      padding: 16px 12px;
      font-size: 14px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    tr:hover td {
      background: rgba(255,255,255,0.02);
    }
    .delta-positive {
      color: var(--bad);
      font-weight: 700;
    }
    .delta-negative {
      color: var(--good);
      font-weight: 700;
    }
    .delta-neutral {
      color: var(--muted);
      font-weight: 700;
    }
    .ok {
      color: var(--good);
      font-weight: 700;
    }
    .ng {
      color: var(--bad);
      font-weight: 700;
    }
    select, input[type="search"] {
      background: var(--bg-secondary);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 13px;
      min-width: 140px;
    }
    select:focus, input[type="search"]:focus {
      outline: none;
      border-color: var(--accent);
    }
    .table-note {
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
    }
    .catalog-layout {
      display: grid;
      grid-template-columns: minmax(340px, 460px) minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }
    .catalog-list-wrap {
      border: 1px solid var(--border);
      border-radius: 12px;
      max-height: min(72vh, 920px);
      overflow: auto;
      background: var(--bg-secondary);
    }
    .catalog-list-wrap thead th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--bg-secondary);
    }
    .catalog-row {
      cursor: pointer;
    }
    .catalog-row.active td {
      background: rgba(0, 122, 204, 0.18);
      border-bottom-color: rgba(0, 122, 204, 0.28);
    }
    .catalog-detail {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--bg-secondary);
      padding: 20px;
      min-height: 420px;
      max-height: min(72vh, 920px);
      overflow-y: auto;
    }
    .catalog-detail h2 {
      margin: 0;
      font-size: 22px;
      line-height: 1.3;
    }
    .catalog-meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }
    .catalog-meta-item {
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
    }
    .catalog-meta-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }
    .catalog-meta-value {
      color: var(--text);
      font-size: 14px;
      font-weight: 600;
      word-break: break-word;
    }
    .catalog-section-title {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .catalog-prompt {
      margin: 0;
      padding: 16px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.18);
      color: var(--text);
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.65;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .details-layout {
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      align-items: stretch;
    }
    .details-panel-block {
      display: grid;
      gap: 12px;
    }
    .details-panel-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 14px;
      padding: 2px 2px 0;
    }
    .details-panel-title-block {
      display: grid;
      gap: 6px;
      min-width: 0;
    }
    .details-panel-eyebrow {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .details-panel-title {
      margin: 0;
      color: var(--text);
      font-size: 18px;
      font-weight: 800;
      line-height: 1.2;
    }
    .details-panel-copy {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .detail-grid-wrap {
      background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)),
        var(--bg-secondary);
    }
    .details-list-panel {
      min-height: 420px;
      max-height: min(50vh, 780px);
      overflow: auto;
    }
    .details-pane-panel {
      min-height: 620px;
      max-height: none;
      overflow: visible;
    }
    .detail-grid-table {
      table-layout: fixed;
    }
    .detail-grid-table th:nth-child(1) { width: 16%; }
    .detail-grid-table th:nth-child(2) { width: 19%; }
    .detail-grid-table th:nth-child(3) { width: 17%; }
    .detail-grid-table th:nth-child(4) { width: 21%; }
    .detail-grid-table th:nth-child(5) { width: 13%; }
    .detail-grid-table th:nth-child(6) { width: 14%; }
    .detail-grid-table td {
      padding: 14px 14px;
    }
    .detail-grid-table th {
      padding: 12px 14px;
      background: rgba(0, 0, 0, 0.14);
    }
    .detail-row-cell {
      display: grid;
      gap: 6px;
      min-width: 0;
    }
    .detail-row-title {
      color: var(--text);
      font-size: 13px;
      font-weight: 700;
      line-height: 1.35;
      word-break: break-word;
    }
    .detail-row-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      word-break: break-word;
    }
    .detail-row-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .status-pill,
    .phase-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      border: 1px solid transparent;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .status-pill-success {
      color: #d7fff5;
      background: rgba(78, 201, 176, 0.14);
      border-color: rgba(78, 201, 176, 0.32);
    }
    .status-pill-error {
      color: #ffd7d7;
      background: rgba(241, 76, 76, 0.14);
      border-color: rgba(241, 76, 76, 0.28);
    }
    .status-pill-timeout {
      color: #ffe7b3;
      background: rgba(220, 220, 170, 0.14);
      border-color: rgba(220, 220, 170, 0.28);
    }
    .status-pill-other,
    .phase-pill {
      color: #d7ebff;
      background: rgba(0, 122, 204, 0.12);
      border-color: rgba(0, 122, 204, 0.28);
    }
    .detail-metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 10px;
    }
    .detail-metric-line {
      display: grid;
      gap: 2px;
    }
    .detail-metric-label {
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .detail-metric-value {
      color: var(--text);
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 600;
      line-height: 1.4;
    }
    .detail-signature-preview {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      word-break: break-word;
    }
    .detail-pane-shell {
      display: grid;
      gap: 18px;
    }
    .detail-pane-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    .detail-pane-title-block {
      display: grid;
      gap: 10px;
      min-width: 0;
    }
    .detail-pane-title {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      font-weight: 800;
      word-break: break-word;
    }
    .detail-pane-subtitle {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      word-break: break-word;
    }
    .detail-pane-actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }
    .detail-summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }
    .detail-summary-item {
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.035);
    }
    .detail-summary-item .detail-metric-label {
      font-size: 10px;
    }
    .detail-summary-item .detail-metric-value {
      font-size: 14px;
      margin-top: 4px;
    }
    .detail-columns {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
      gap: 16px;
    }
    .detail-stack {
      display: grid;
      gap: 16px;
    }
    .detail-section {
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(255,255,255,0.03);
      padding: 16px;
    }
    .detail-section-heading {
      margin: 0 0 12px;
      color: var(--text);
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    .detail-kv-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0 10px;
    }
    .detail-kv-table th,
    .detail-kv-table td {
      padding: 0;
      border: none;
      text-align: left;
      vertical-align: top;
      cursor: default;
    }
    .detail-kv-table th:hover {
      color: var(--muted);
    }
    .detail-kv-table th {
      width: 132px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: none;
    }
    .detail-kv-table td {
      color: var(--text);
      font-size: 13px;
      line-height: 1.55;
      word-break: break-word;
    }
    .detail-tool-breakdown {
      display: grid;
      gap: 8px;
    }
    .detail-tool-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 9px 12px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.07);
      background: rgba(0,0,0,0.16);
      font-family: var(--mono);
      font-size: 12px;
    }
    .detail-tool-row strong {
      color: var(--text);
      font-size: 13px;
    }
    .detail-disclosure {
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(255,255,255,0.028);
      overflow: hidden;
    }
    .detail-disclosure + .detail-disclosure {
      margin-top: 14px;
    }
    .detail-disclosure summary {
      list-style: none;
      cursor: pointer;
      padding: 14px 16px;
      color: var(--text);
      font-size: 13px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .detail-disclosure summary::-webkit-details-marker {
      display: none;
    }
    .detail-disclosure[open] summary {
      border-bottom: 1px solid var(--border);
      background: rgba(0,0,0,0.12);
    }
    .detail-disclosure .catalog-prompt {
      border: none;
      border-radius: 0;
      background: transparent;
    }
    .detail-empty {
      padding: 36px 20px;
      color: var(--muted);
      text-align: center;
      font-size: 13px;
      line-height: 1.7;
    }
    .status-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 24px;
      padding: 8px 16px;
      background: var(--accent-2);
      color: #ffffff;
      font-size: 12px;
      border-radius: 4px;
      flex-wrap: wrap;
      gap: 8px;
    }
    @media (max-width: 768px) {
      header { padding: 20px 16px 12px; }
      main { padding: 16px; }
      .card { padding: 16px; }
      th, td { padding: 12px 8px; font-size: 12px; }
      .stat .value { font-size: 22px; }
      .catalog-layout { grid-template-columns: 1fr; }
      .catalog-list-wrap, .catalog-detail { max-height: none; }
      .catalog-detail { min-height: auto; }
      .source-row { align-items: stretch; }
      .details-layout { grid-template-columns: 1fr; }
      .details-panel-header { align-items: flex-start; }
      .details-list-panel { min-height: 320px; max-height: none; }
      .details-pane-panel { min-height: 0; }
      .detail-grid-table { min-width: 920px; }
      .detail-summary-grid,
      .detail-columns,
      .detail-metric-grid { grid-template-columns: 1fr; }
      .detail-pane-header { flex-direction: column; }
      .detail-pane-actions { justify-content: flex-start; }
      .detail-kv-table { border-spacing: 0 8px; }
      .detail-kv-table th,
      .detail-kv-table td { display: block; width: auto; }
      .detail-kv-table td { margin-top: 2px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-content">
      <div class="logo">
        <div class="logo-icon">L</div>
        <div>
          <h1>Local LLM Bench</h1>
          <div class="header-meta" id="header-meta"></div>
        </div>
      </div>
      <div class="toolbar" style="margin-bottom:0;">
        <button class="btn" id="reload">Reload History</button>
        <button class="btn btn-secondary" id="open-history">Open history.json</button>
        <input id="history-file" type="file" accept=".json,application/json" class="hidden" />
      </div>
    </div>
  </header>

  <main>
    <div class="card">
      <div class="source-row">
        <div class="tag" id="history-source"></div>
        <div class="muted" id="load-state"></div>
      </div>
      <div class="table-note">`index.html` は `history.json` を読み込むビュアーとして動作します。自動読込に失敗した場合は `Open history.json` から手動で選択できます。</div>
    </div>

    <div class="stat-grid" id="stats"></div>

    <div class="card">
      <div class="toolbar" style="align-items:center; margin-bottom:12px;">
        <div class="catalog-section-title" style="margin-bottom:0;">Prompt</div>
        <select id="prompt-filter"></select>
        <div class="filter-meta" id="prompt-filter-meta"></div>
      </div>
      <pre class="catalog-prompt" id="prompt-preview"></pre>
    </div>

      <div class="tabs">
        <button class="tab active" data-tab="leaderboard">Leaderboard</button>
        <button class="tab" data-tab="compare">Compare</button>
        <button class="tab" data-tab="coldwarm">Cold vs Warm</button>
        <button class="tab" data-tab="stability">Stability</button>
        <button class="tab" data-tab="errors">Error Analysis</button>
        <button class="tab" data-tab="details">Run Details</button>
      </div>

    <div id="leaderboard-panel" class="card">
      <div class="toolbar leaderboard-toolbar">
        <div class="leaderboard-filter-group">
          <details class="filter-dropdown" id="leaderboard-model-filter">
            <summary class="btn btn-secondary" id="leaderboard-model-summary">Models: All</summary>
            <div class="filter-dropdown-menu">
              <div class="filter-actions">
                <button class="btn btn-secondary" id="leaderboard-select-all" type="button">All</button>
                <button class="btn btn-secondary" id="leaderboard-clear-all" type="button">Clear</button>
              </div>
              <div class="filter-option-list" id="leaderboard-model-options"></div>
            </div>
          </details>
          <div class="filter-meta" id="leaderboard-filter-meta"></div>
        </div>
      </div>
      <table id="leaderboard-table">
        <thead>
          <tr>
            <th data-sort="model">Model</th>
            <th data-sort="format_sort">Format</th>
            <th data-sort="quantization_sort">Quantization</th>
            <th data-sort="total_samples">Samples</th>
            <th data-sort="success_rate">Success</th>
            <th data-sort="warm_mean_benchmark_score" data-benchmark-column="true">Warm Score</th>
            <th data-sort="benchmark_correct_rate" data-benchmark-column="true">Correct</th>
            <th data-sort="warm_benchmark_error_rate" data-benchmark-column="true">Error</th>
            <th data-sort="warm_mean_ttft_ms">Warm TTFT</th>
            <th data-sort="warm_mean_total_latency_ms">Warm Latency</th>
            <th data-sort="warm_mean_decode_tps">Decode Speed</th>
            <th data-sort="warm_mean_initial_prompt_tps">Init Prompt Speed</th>
            <th data-sort="warm_mean_conversation_prompt_tps">Conv Prompt Speed</th>
            <th data-sort="cold_mean_total_latency_ms">Cold Latency</th>
          </tr>
        </thead>
        <tbody id="leaderboard-body"></tbody>
      </table>
      <div class="table-note" id="leaderboard-note">history.json 全体を集計しています。速度系は Warm 平均を中心に比較し、Correct は cold + warm を通した全体正答率です。Init Prompt は初回投入、Conv Prompt は会話全体の prompt throughput です。usage が返らないモデルでも、同一 prompt の実測 token 数が history 内にあれば代表値で補完します。参照がない場合のみ N/A です。</div>
    </div>

    <div id="compare-panel" class="card" style="display:none;">
      <div class="toolbar">
        <select id="compare-left-model"></select>
        <button class="btn btn-secondary" id="compare-swap">Swap</button>
        <select id="compare-right-model"></select>
      </div>
      <table id="compare-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th id="compare-left-heading">Model A</th>
            <th id="compare-right-heading">Model B</th>
            <th>Delta</th>
          </tr>
        </thead>
        <tbody id="compare-body"></tbody>
      </table>
      <div class="table-note">Delta は `left - right` です。Latency 系は負の値ほど左モデルが高速で、Speed / Success 系は正の値ほど左モデルが優位です。</div>
    </div>

    <div id="coldwarm-panel" class="card" style="display:none;">
      <table id="coldwarm-table">
        <thead>
          <tr>
            <th data-sort="model">Model</th>
            <th data-sort="cold_mean_ttft_ms">Cold TTFT</th>
            <th data-sort="warm_mean_ttft_ms">Warm TTFT</th>
            <th data-sort="delta_ttft_ms">Delta TTFT</th>
            <th data-sort="cold_mean_total_latency_ms">Cold Latency</th>
            <th data-sort="warm_mean_total_latency_ms">Warm Latency</th>
            <th data-sort="delta_total_latency_ms">Delta Latency</th>
            <th data-sort="warm_mean_decode_tps">Warm Decode</th>
          </tr>
        </thead>
        <tbody id="coldwarm-body"></tbody>
      </table>
      <div class="table-note">Delta は `warm - cold` です。負の値ほど warm 化で改善しています。</div>
    </div>

      <div id="stability-panel" class="card" style="display:none;">
        <table id="stability-table">
        <thead>
          <tr>
            <th data-sort="model">Model</th>
            <th data-sort="success_rate">Success</th>
            <th data-sort="warm_mean_benchmark_score" data-benchmark-column="true">Warm Score</th>
            <th data-sort="benchmark_correct_rate" data-benchmark-column="true">Correct</th>
            <th data-sort="warm_benchmark_error_rate" data-benchmark-column="true">Benchmark Error</th>
            <th data-sort="warm_stddev_total_latency_ms">Warm Latency Stddev</th>
            <th data-sort="warm_cv_total_latency_ms">Warm Latency CV</th>
            <th data-sort="warm_p95_ttft_ms">Warm TTFT p95</th>
            <th data-sort="error_count">Errors</th>
            <th data-sort="warm_samples">Warm Samples</th>
            <th data-sort="overall_mean_completion_tokens">Output Tokens</th>
          </tr>
        </thead>
        <tbody id="stability-body"></tbody>
        </table>
        <div class="table-note">CV は標準偏差 / 平均です。小さいほどばらつきが少なく、安定しています。</div>
      </div>

      <div id="errors-panel" class="card" style="display:none;">
        <div class="toolbar">
          <select id="error-model-filter"></select>
          <select id="error-category-filter"></select>
          <input id="error-query" type="search" placeholder="Search signature / error / run / benchmark" />
          <button class="btn btn-secondary" id="error-reset">Clear</button>
        </div>
        <div class="catalog-section-title">Signatures</div>
        <table id="error-signature-table">
          <thead>
            <tr>
              <th>Signature</th>
              <th>Category</th>
              <th>Count</th>
              <th>Latest</th>
              <th>Models</th>
              <th>Benchmark</th>
              <th>Run</th>
            </tr>
          </thead>
          <tbody id="error-signature-body"></tbody>
        </table>
        <div class="catalog-section-title" style="margin-top: 20px;">Model Breakdown</div>
        <table id="error-model-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Error Rate</th>
              <th>Events</th>
              <th>Top Signature</th>
              <th>Timeouts</th>
            </tr>
          </thead>
          <tbody id="error-model-body"></tbody>
        </table>
        <div class="catalog-section-title" style="margin-top: 20px;">Occurrences</div>
        <div class="catalog-layout">
          <div class="catalog-list-wrap">
            <table id="error-event-table">
              <thead>
                <tr>
                  <th>Started</th>
                  <th>Run</th>
                  <th>Model</th>
                  <th>Location</th>
                  <th>Category</th>
                </tr>
              </thead>
              <tbody id="error-event-body"></tbody>
            </table>
          </div>
          <div class="catalog-detail" id="error-detail-pane"></div>
        </div>
      </div>

      <div id="details-panel" class="card" style="display:none;">
      <div class="toolbar">
        <select id="detail-model-filter"></select>
        <select id="detail-phase-filter">
          <option value="">Phase</option>
          <option value="cold">cold</option>
          <option value="warm">warm</option>
        </select>
        <select id="detail-status-filter"></select>
        <input id="detail-query" type="search" placeholder="Search response / error / model / run / tool" />
        <button class="btn btn-secondary" id="detail-reset">Clear</button>
      </div>
      <div class="details-layout">
        <section class="details-panel-block">
          <div class="details-panel-header">
            <div class="details-panel-title-block">
              <div class="details-panel-eyebrow">Run List</div>
              <h3 class="details-panel-title">実行一覧</h3>
              <div class="details-panel-copy">上段で run を選ぶと、下段の Inspector に詳細を大きく表示します。</div>
            </div>
          </div>
          <div class="catalog-list-wrap detail-grid-wrap details-list-panel">
            <table id="detail-table" class="detail-grid-table">
              <thead>
                <tr>
                  <th data-sort="run_started_at">Started</th>
                  <th data-sort="model">Model / Run</th>
                  <th data-sort="phase">Attempt</th>
                  <th data-sort="total_latency_ms">Metrics</th>
                  <th data-sort="tool_call_count">Tool Calls</th>
                  <th data-sort="status">Outcome</th>
                </tr>
              </thead>
              <tbody id="detail-body"></tbody>
            </table>
          </div>
        </section>
        <section class="details-panel-block">
          <div class="details-panel-header">
            <div class="details-panel-title-block">
              <div class="details-panel-eyebrow">Inspector</div>
              <h3 class="details-panel-title">選択中の run 詳細</h3>
              <div class="details-panel-copy">metrics、benchmark、question results、transcript を下段で連続して確認できます。</div>
            </div>
          </div>
          <div class="catalog-detail details-pane-panel" id="detail-pane"></div>
        </section>
      </div>
    </div>

    <div class="status-bar">
      <div id="summary-text"></div>
      <div id="subsummary"></div>
    </div>
  </main>

  <script>
    const DEFAULT_HISTORY_URL = __DEFAULT_HISTORY_URL__;
    const ALL_PROMPTS = "__ALL_PROMPTS__";
    const COMMON_METRICS = [
      "ttft_ms",
      "total_latency_ms",
      "completion_window_ms",
      "prompt_tokens",
      "completion_tokens",
      "decode_tps",
      "end_to_end_tps",
      "approx_prompt_tps",
      "initial_prompt_tps",
      "conversation_prompt_tps",
    ];
    const BENCHMARK_METRICS = ["benchmark_score"];
    const ALL_METRICS = [...COMMON_METRICS, ...BENCHMARK_METRICS];
    const PROMPT_TOKEN_FIELDS = [
      "prompt_tokens",
      "initial_prompt_tokens",
      "conversation_prompt_tokens",
    ];
    const PROMPT_PEER_KEY_SEPARATOR = "::prompt::";
    const PROMPT_TOKEN_FALLBACK_FIELDS = {
      prompt_tokens: ["prompt_tokens", "initial_prompt_tokens", "conversation_prompt_tokens"],
      initial_prompt_tokens: ["initial_prompt_tokens", "prompt_tokens", "conversation_prompt_tokens"],
      conversation_prompt_tokens: ["conversation_prompt_tokens", "prompt_tokens", "initial_prompt_tokens"],
    };

      let sourceLabel = DEFAULT_HISTORY_URL;
      let historyBaseUrl = null;
      let reportData = emptyPayload([]);
      let viewData = emptyPayload([]);

      const state = {
        leaderboardSort: { key: "warm_mean_total_latency_ms", asc: true },
      leaderboardSelectedModels: [],
      leaderboardFilterTouched: false,
        compare: { leftModel: "", rightModel: "" },
        coldwarmSort: { key: "delta_total_latency_ms", asc: true },
        stabilitySort: { key: "success_rate", asc: false },
        detailSort: { key: "run_started_at", asc: false },
        selectedErrorSignature: "",
        selectedErrorEventKey: "",
        selectedRecordKey: "",
        selectedPrompt: "",
        rawLogCache: {},
      };

      const filters = {
        model: "",
        phase: "",
        status: "",
        query: "",
        errorModel: "",
        errorCategory: "",
        errorQuery: "",
      };

      const panels = {
        leaderboard: document.getElementById("leaderboard-panel"),
        compare: document.getElementById("compare-panel"),
        coldwarm: document.getElementById("coldwarm-panel"),
        stability: document.getElementById("stability-panel"),
        errors: document.getElementById("errors-panel"),
        details: document.getElementById("details-panel"),
      };

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function emptyPayload(historyRuns) {
      return {
        generated_at: "-",
        history_runs: Array.isArray(historyRuns) ? historyRuns : [],
        latest_run: {},
        records: [],
        summary: {
          total_runs: 0,
          total_models: 0,
          total_samples: 0,
          successful_samples: 0,
          failed_samples: 0,
          cards: {
            fastest_ttft: null,
            fastest_warm_latency: null,
            fastest_decode_speed: null,
            total_samples: { value: 0 },
          },
          models: [],
          latest_run_id: null,
          latest_started_at: null,
        },
        model_catalog: {},
        prompt_preview: "",
        prompt_count: 0,
        prompts: [],
      };
    }

    function firstText(values) {
      for (const value of values) {
        if (typeof value === "string" && value.trim()) {
          return value.trim();
        }
      }
      return "";
    }

      function truncateText(value, maxLength = 96) {
        const normalized = String(value || "").trim().replace(/\\s+/g, " ");
        if (!normalized) return "";
        return normalized.length > maxLength ? `${normalized.slice(0, maxLength - 1)}…` : normalized;
      }

      function excerptText(value, maxLength = 400) {
        const normalized = String(value || "").trim();
        if (!normalized) return "";
        return normalized.length > maxLength ? `${normalized.slice(0, maxLength - 1)}…` : normalized;
      }

      function normalizeErrorSignature(value) {
        const text = String(value || "").trim();
        if (!text) return "";
        const firstLine = text.split(/\\r?\\n/).find((line) => line.trim()) || "";
        return excerptText(
          firstLine
            .replace(/^\\[[^\\]]+\\]\\s*/, "")
            .replace(/(\\/private\\/tmp\\/[^\\s:]+|\\/tmp\\/[^\\s:]+|\\/var\\/folders\\/[^\\s:]+|\\/Users\\/[^\\s:]+)/g, "<path>")
            .replace(/\\s+/g, " ")
            .trim(),
          240,
        );
      }

      function categorizeError(value, status = "") {
        const lowered = `${status || ""} ${value || ""}`.toLowerCase();
        if (!lowered.trim()) return "";
        if (lowered.includes("timeout") || lowered.includes("timed out")) return "timeout";
        if (["httperror", "urlerror", "api_base is required", "chat completion response"].some((token) => lowered.includes(token))) return "api";
        if (["tool '", 'tool "', "call_tool", "mcp"].some((token) => lowered.includes(token))) return "tool";
        if ([
          "docker exited",
          "docker コマンド",
          "no such container",
          "unable to find image",
          "pull access denied",
          "no matching manifest",
          "manifest for",
          "docker image '",
        ].some((token) => lowered.includes(token))) return "docker";
        if ([
          "worker returned malformed json",
          "final_answer",
          "analyzeheadless",
          "ghidra",
          "unhandled errors in a taskgroup",
          "binary_path",
        ].some((token) => lowered.includes(token))) return "worker";
        return "other";
      }

      function normalizeErrorFields(entry) {
        const normalized = { ...(entry || {}) };
        const signatureSource = firstText([normalized.error, normalized.stderr_excerpt]);
        const computedCategory = categorizeError(signatureSource, normalized.status);
        normalized.error_signature = normalized.error_signature || normalizeErrorSignature(signatureSource);
        normalized.error_category = (!normalized.error_category || (normalized.error_category === "other" && computedCategory && computedCategory !== "other"))
          ? computedCategory
          : normalized.error_category;
        normalized.stderr_excerpt = excerptText(normalized.stderr_excerpt || "");
        if (!normalized.log_path) normalized.log_path = "";
        return normalized;
      }

      function normalizeToolCount(value) {
        if (typeof value === "boolean") return Number(value);
        if (typeof value === "number" && Number.isFinite(value)) return Math.max(0, Math.trunc(value));
        return null;
      }

      function normalizeCountMap(value) {
        if (!value || typeof value !== "object" || Array.isArray(value)) return {};
        const normalized = {};
        Object.entries(value).forEach(([key, rawCount]) => {
          const name = String(key || "").trim();
          const count = normalizeToolCount(rawCount);
          if (!name || count == null || count <= 0) return;
          normalized[name] = (normalized[name] || 0) + count;
        });
        return normalized;
      }

      function mergeCountMaps(values) {
        const merged = {};
        (Array.isArray(values) ? values : []).forEach((value) => {
          Object.entries(normalizeCountMap(value)).forEach(([name, count]) => {
            merged[name] = (merged[name] || 0) + count;
          });
        });
        return merged;
      }

      function summarizeCountMap(value) {
        const normalized = normalizeCountMap(value);
        const entries = Object.entries(normalized).sort((left, right) => {
          if (right[1] !== left[1]) return right[1] - left[1];
          return left[0].localeCompare(right[0]);
        });
        return entries.map(([name, count]) => `${name} x${count}`).join(", ");
      }

      function normalizeToolFields(entry) {
        const normalized = { ...(entry || {}) };
        const benchmarkMode = String(normalized.benchmark_mode || "").trim().toLowerCase();
        let toolNameCounts = normalizeCountMap(normalized.tool_name_counts);
        let toolCallCount = normalizeToolCount(normalized.tool_call_count);
        const questionResults = Array.isArray(normalized.question_results) ? normalized.question_results : null;

        if (questionResults) {
          const derivedCounts = questionResults
            .map((item) => normalizeToolCount(item.tool_call_count))
            .filter((value) => value != null);
          if (toolCallCount == null && derivedCounts.length) {
            toolCallCount = derivedCounts.reduce((sum, value) => sum + value, 0);
          }
          if (!Object.keys(toolNameCounts).length) {
            toolNameCounts = mergeCountMaps(questionResults.map((item) => item.tool_name_counts));
          }
        }

        if (toolCallCount == null && Object.keys(toolNameCounts).length) {
          toolCallCount = Object.values(toolNameCounts).reduce((sum, value) => sum + value, 0);
        }
        if (toolCallCount == null && benchmarkMode && benchmarkMode !== "docker_task") {
          toolCallCount = 0;
        }

        if (toolCallCount != null) {
          normalized.tool_call_count = toolCallCount;
        } else {
          delete normalized.tool_call_count;
        }
        normalized.tool_name_counts = toolNameCounts;
        return normalized;
      }

      function normalizeNumeric(value) {
        if (typeof value === "boolean") return Number(value);
        if (typeof value === "number" && Number.isFinite(value)) return value;
        return null;
      }

      function computeTps(tokenCount, latencyMs) {
        if (typeof tokenCount !== "number" || !Number.isFinite(tokenCount) || tokenCount <= 0) return null;
        if (typeof latencyMs !== "number" || !Number.isFinite(latencyMs) || latencyMs <= 0) return null;
        return tokenCount / (latencyMs / 1000.0);
      }

      function promptTokensFromTotals(entry) {
        const totalTokens = normalizeNumeric(entry?.total_tokens);
        const completionTokens = normalizeNumeric(entry?.completion_tokens);
        if (totalTokens == null || completionTokens == null || totalTokens < completionTokens) return null;
        const promptTokens = Math.trunc(totalTokens - completionTokens);
        return promptTokens > 0 ? promptTokens : null;
      }

      function medianPromptTokens(values) {
        const ordered = (Array.isArray(values) ? values : [])
          .filter((value) => typeof value === "number" && Number.isFinite(value) && value > 0)
          .sort((left, right) => left - right);
        if (!ordered.length) return null;
        const middle = Math.floor(ordered.length / 2);
        if (ordered.length % 2 === 1) return ordered[middle];
        return Math.round((ordered[middle - 1] + ordered[middle]) / 2);
      }

      function backfillPromptMetricsFromPeers(historyRuns) {
        const ensureBucket = (store, key) => {
          const existing = store.get(key);
          if (existing) return existing;
          const created = {
            prompt_tokens: [],
            initial_prompt_tokens: [],
            conversation_prompt_tokens: [],
          };
          store.set(key, created);
          return created;
        };

        const byModelPrompt = new Map();
        const byPrompt = new Map();

        historyRuns.forEach((run) => {
          const runPromptText = firstText([run?.prompt_text]);
          const runModel = firstText([run?.model]) || "(unknown)";
          const runRecords = Array.isArray(run?.records) ? run.records : [];
          runRecords.forEach((record) => {
            const benchmarkMode = String(record?.benchmark_mode || run?.benchmark_mode || "").trim().toLowerCase();
            if (benchmarkMode === "docker_task") return;
            const promptText = firstText([record?.prompt_text, runPromptText]);
            if (!promptText) return;
            const model = firstText([record?.model, runModel]) || "(unknown)";
            const modelBucket = ensureBucket(byModelPrompt, `${model}${PROMPT_PEER_KEY_SEPARATOR}${promptText}`);
            const promptBucket = ensureBucket(byPrompt, promptText);
            PROMPT_TOKEN_FIELDS.forEach((field) => {
              const value = normalizeToolCount(record?.[field]);
              if (value == null || value <= 0) return;
              modelBucket[field].push(value);
              promptBucket[field].push(value);
            });
          });
        });

        historyRuns.forEach((run) => {
          const runPromptText = firstText([run?.prompt_text]);
          const runModel = firstText([run?.model]) || "(unknown)";
          const runRecords = Array.isArray(run?.records) ? run.records : [];
          run.records = runRecords.map((record) => {
            const benchmarkMode = String(record?.benchmark_mode || run?.benchmark_mode || "").trim().toLowerCase();
            if (benchmarkMode === "docker_task") return record;
            const promptText = firstText([record?.prompt_text, runPromptText]);
            if (!promptText) return record;
            const model = firstText([record?.model, runModel]) || "(unknown)";
            const modelBucket = byModelPrompt.get(`${model}${PROMPT_PEER_KEY_SEPARATOR}${promptText}`) || {};
            const promptBucket = byPrompt.get(promptText) || {};
            const nextRecord = { ...record };
            let changed = false;

            PROMPT_TOKEN_FIELDS.forEach((field) => {
              const currentValue = normalizeToolCount(nextRecord[field]);
              if (currentValue != null && currentValue > 0) return;
              const candidateFields = PROMPT_TOKEN_FALLBACK_FIELDS[field] || [field];
              let priorValue = null;
              for (const candidateField of candidateFields) {
                priorValue = medianPromptTokens(modelBucket[candidateField]);
                if (priorValue != null) break;
                priorValue = medianPromptTokens(promptBucket[candidateField]);
                if (priorValue != null) break;
              }
              if (priorValue == null) return;
              nextRecord[field] = priorValue;
              changed = true;
            });

            return changed ? normalizePromptFields(nextRecord) : record;
          });
        });
      }

      function normalizePromptFields(entry) {
        const normalized = { ...(entry || {}) };
        const benchmarkMode = String(normalized.benchmark_mode || "").trim().toLowerCase();
        const ttftMs = normalizeNumeric(normalized.ttft_ms);
        const totalLatencyMs = normalizeNumeric(normalized.total_latency_ms);
        let promptTokens = normalizeToolCount(normalized.prompt_tokens);
        let promptLatencyMs = normalizeNumeric(normalized.prompt_latency_ms);
        let approxPromptTps = normalizeNumeric(normalized.approx_prompt_tps);
        let initialPromptTokens = normalizeToolCount(normalized.initial_prompt_tokens);
        let initialPromptLatencyMs = normalizeNumeric(normalized.initial_prompt_latency_ms);
        let initialPromptTps = normalizeNumeric(normalized.initial_prompt_tps);
        let conversationPromptTokens = normalizeToolCount(normalized.conversation_prompt_tokens);
        let conversationPromptLatencyMs = normalizeNumeric(normalized.conversation_prompt_latency_ms);
        let conversationPromptTps = normalizeNumeric(normalized.conversation_prompt_tps);
        const questionResults = Array.isArray(normalized.question_results) ? normalized.question_results : null;

        if (questionResults) {
          const sumInt = (field) => {
            const values = questionResults
              .map((item) => normalizeToolCount(item?.[field]))
              .filter((value) => value != null);
            return values.length ? values.reduce((sum, value) => sum + value, 0) : null;
          };
          const sumFloat = (field) => {
            const values = questionResults
              .map((item) => normalizeNumeric(item?.[field]))
              .filter((value) => value != null && value > 0);
            return values.length ? values.reduce((sum, value) => sum + value, 0) : null;
          };
          if (promptTokens == null) promptTokens = sumInt("prompt_tokens");
          if (promptLatencyMs == null) promptLatencyMs = sumFloat("prompt_latency_ms");
          if (initialPromptTokens == null) initialPromptTokens = sumInt("initial_prompt_tokens");
          if (initialPromptLatencyMs == null) initialPromptLatencyMs = sumFloat("initial_prompt_latency_ms");
          if (conversationPromptTokens == null) conversationPromptTokens = sumInt("conversation_prompt_tokens");
          if (conversationPromptLatencyMs == null) conversationPromptLatencyMs = sumFloat("conversation_prompt_latency_ms");
        }

        if (promptTokens == null) {
          promptTokens = promptTokensFromTotals(normalized);
        }

        if (promptTokens == null && benchmarkMode !== "docker_task") {
          if (initialPromptTokens != null && initialPromptTokens > 0) {
            promptTokens = initialPromptTokens;
          } else if (conversationPromptTokens != null && conversationPromptTokens > 0) {
            promptTokens = conversationPromptTokens;
          }
        }

        if (promptLatencyMs == null && benchmarkMode !== "docker_task") {
          if (initialPromptLatencyMs != null && initialPromptLatencyMs > 0) {
            promptLatencyMs = initialPromptLatencyMs;
          } else if (conversationPromptLatencyMs != null && conversationPromptLatencyMs > 0) {
            promptLatencyMs = conversationPromptLatencyMs;
          }
        }

        if (initialPromptTokens == null && benchmarkMode !== "docker_task" && promptTokens != null) {
          initialPromptTokens = promptTokens;
        }
        if (initialPromptLatencyMs == null && benchmarkMode !== "docker_task" && ttftMs != null && ttftMs > 0) {
          initialPromptLatencyMs = ttftMs;
        }
        if (initialPromptTps == null) {
          initialPromptTps = computeTps(initialPromptTokens, initialPromptLatencyMs);
        }
        if (initialPromptTps == null && benchmarkMode !== "docker_task" && approxPromptTps != null) {
          initialPromptTps = approxPromptTps;
        }

        if (conversationPromptTokens == null) {
          if (benchmarkMode === "docker_task" && promptTokens != null) {
            conversationPromptTokens = promptTokens;
          } else if (benchmarkMode !== "docker_task") {
            conversationPromptTokens = promptTokens != null ? promptTokens : initialPromptTokens;
          }
        }
        if (conversationPromptLatencyMs == null) {
          if (benchmarkMode === "docker_task" && promptLatencyMs != null && promptLatencyMs > 0) {
            conversationPromptLatencyMs = promptLatencyMs;
          } else if (benchmarkMode !== "docker_task") {
            if (totalLatencyMs != null && totalLatencyMs > 0) {
              conversationPromptLatencyMs = totalLatencyMs;
            } else if (promptLatencyMs != null && promptLatencyMs > 0) {
              conversationPromptLatencyMs = promptLatencyMs;
            }
          }
        }
        if (conversationPromptTps == null) {
          conversationPromptTps = computeTps(conversationPromptTokens, conversationPromptLatencyMs);
        }
        if (conversationPromptTps == null && approxPromptTps != null) {
          conversationPromptTps = approxPromptTps;
        }

        if (initialPromptTokens != null) normalized.initial_prompt_tokens = initialPromptTokens;
        else delete normalized.initial_prompt_tokens;
        if (initialPromptLatencyMs != null && initialPromptLatencyMs > 0) normalized.initial_prompt_latency_ms = initialPromptLatencyMs;
        else delete normalized.initial_prompt_latency_ms;
        if (initialPromptTps != null) normalized.initial_prompt_tps = initialPromptTps;
        else delete normalized.initial_prompt_tps;
        if (conversationPromptTokens != null) normalized.conversation_prompt_tokens = conversationPromptTokens;
        else delete normalized.conversation_prompt_tokens;
        if (conversationPromptLatencyMs != null && conversationPromptLatencyMs > 0) normalized.conversation_prompt_latency_ms = conversationPromptLatencyMs;
        else delete normalized.conversation_prompt_latency_ms;
        if (conversationPromptTps != null) normalized.conversation_prompt_tps = conversationPromptTps;
        else delete normalized.conversation_prompt_tps;
        return normalized;
      }

    function looksLikeFileSystemPath(value) {
      const normalized = String(value || "").trim();
      if (!normalized) return false;
      return normalized.startsWith("/")
        || normalized.startsWith("~/")
        || /^[A-Za-z]:[\\\\/]/.test(normalized);
    }

    function humanizeQuantizationToken(token) {
      const normalized = String(token || "").trim();
      if (!normalized) return "";
      return normalized.replace(/[\\s-]+/g, "_").toUpperCase();
    }

    function quantizationBitsFromToken(token) {
      const normalized = humanizeQuantizationToken(token);
      if (!normalized) return null;
      const directBitMatch = normalized.match(/^(\\d+)BIT$/i);
      if (directBitMatch) {
        const bits = Number(directBitMatch[1]);
        return Number.isFinite(bits) ? bits : null;
      }
      const prefixedMatch = normalized.match(/^(?:IQ|Q|BF|FP|F|MXFP)(\\d+)/i);
      if (!prefixedMatch) return null;
      const bits = Number(prefixedMatch[1]);
      return Number.isFinite(bits) ? bits : null;
    }

    function extractQuantizationToken(value) {
      const normalized = String(value || "").trim();
      if (!normalized) return "";
      const candidates = [];
      const pushCandidate = (candidate) => {
        const text = String(candidate || "").trim();
        if (text && !candidates.includes(text)) {
          candidates.push(text);
        }
      };

      pushCandidate(normalized);
      if (normalized.includes("@")) {
        pushCandidate(normalized.split("@").slice(-1)[0] || "");
      }

      const parts = normalized.split(/[\\\\/]/).filter(Boolean);
      const tail = parts.length ? parts[parts.length - 1] : "";
      pushCandidate(tail);
      if (tail) {
        pushCandidate(tail.replace(/\\.gguf$/i, ""));
      }

      for (const candidate of candidates) {
        const match = candidate.match(/(?:^|[^A-Za-z0-9])((?:IQ|Q)\\d+(?:[_-][A-Za-z0-9]+)*|(?:BF|FP|F)\\d+|MXFP\\d+|\\d+BIT)(?:$|[^A-Za-z0-9])/i);
        if (match) {
          return humanizeQuantizationToken(match[1]);
        }
      }
      return "";
    }

    function inferModelInfoFromModelName(model) {
      const requestedModel = String(model || "").trim();
      if (!requestedModel) return null;

      const lowered = requestedModel.toLowerCase();
      const [baseModel, variantTokenRaw = ""] = requestedModel.split("@", 2);
      const variantToken = variantTokenRaw.trim();
      const quantizationName = extractQuantizationToken(variantToken) || humanizeQuantizationToken(variantToken);
      const quantizationBits = quantizationBitsFromToken(quantizationName || variantToken);
      let format = "";
      if (lowered.includes(".gguf") || lowered.includes("-gguf") || lowered.includes("@q") || lowered.includes("@f")) {
        format = "gguf";
      } else if (lowered.includes("@mlx") || lowered.endsWith("-mlx")) {
        format = "mlx";
      }

      const displayStem = String(baseModel || requestedModel).split("/").pop() || requestedModel;
      const inferred = {
        requested_model: requestedModel,
        identifier: "",
        model_key: String(baseModel || requestedModel).trim(),
        display_name: displayStem,
        format,
        quantization: quantizationName ? (
          quantizationBits != null ? `${quantizationName} (${quantizationBits}-bit)` : quantizationName
        ) : "",
        quantization_name: quantizationName,
        quantization_bits: quantizationBits,
        publisher: requestedModel.includes("/") ? requestedModel.split("/", 1)[0] : "",
        architecture: lowered.includes("gpt-oss") ? "gpt-oss" : "",
        selected_variant: "",
        indexed_model_identifier: "",
        path: "",
      };
      return Object.values(inferred).some((value) => {
        if (typeof value === "string") return value.trim();
        return value != null;
      }) ? inferred : null;
    }

    function normalizeModelInfo(modelInfo, fallbackModel = "") {
      const normalized = modelInfo && typeof modelInfo === "object" ? { ...modelInfo } : {};
      const inferred = inferModelInfoFromModelName(fallbackModel) || {};
      const merged = { ...normalized };
      Object.entries(inferred).forEach(([key, value]) => {
        const current = merged[key];
        const missing = current == null || (typeof current === "string" && !current.trim());
        if (missing && value != null && (!(typeof value === "string") || value.trim())) {
          merged[key] = value;
        }
      });
      if (!merged.requested_model && fallbackModel) {
        merged.requested_model = String(fallbackModel).trim();
      }
      const quantizationName = firstText([
        merged.quantization_name,
        extractQuantizationToken(merged.quantization),
        extractQuantizationToken(merged.selected_variant),
        extractQuantizationToken(merged.display_name),
        extractQuantizationToken(merged.identifier),
        extractQuantizationToken(merged.indexed_model_identifier),
        extractQuantizationToken(merged.path),
        extractQuantizationToken(merged.model_key),
        extractQuantizationToken(merged.requested_model),
        extractQuantizationToken(fallbackModel),
      ]);
      const quantizationBits = quantizationBitsFromToken(quantizationName);
      if ((!merged.quantization || !String(merged.quantization).trim()) && quantizationName) {
        merged.quantization = quantizationBits != null
          ? `${quantizationName} (${quantizationBits}-bit)`
          : quantizationName;
      }
      if ((!merged.quantization_name || !String(merged.quantization_name).trim()) && quantizationName) {
        merged.quantization_name = quantizationName;
      }
      if (merged.quantization_bits == null && quantizationBits != null) {
        merged.quantization_bits = quantizationBits;
      }
      return Object.values(merged).some((value) => {
        if (typeof value === "string") return value.trim();
        return value != null;
      }) ? merged : null;
    }

    function normalizeRunEntry(runData) {
      const normalized = { ...(runData || {}) };
      const rawRecords = Array.isArray(normalized.records) ? normalized.records : [];
      const records = rawRecords.filter((record) => record && typeof record === "object").map((record) => ({ ...record }));

      const modelCandidates = [normalized.model];
      if (Array.isArray(normalized.models)) {
        modelCandidates.push(...normalized.models);
      }
      modelCandidates.push(...records.map((record) => record.model));
      const model = firstText(modelCandidates) || "(unknown)";

      const promptCandidates = [normalized.prompt_text, ...records.map((record) => record.prompt_text)];
      const promptText = firstText(promptCandidates);

      const enrichedRecords = records.map((record) => {
        const benchmarkMode = record.benchmark_mode || normalized.benchmark_mode || "prompt";
        const questionResults = Array.isArray(record.question_results)
          ? record.question_results
            .filter((item) => item && typeof item === "object")
            .map((item) => normalizePromptFields(normalizeToolFields(normalizeErrorFields({
              ...item,
              benchmark_mode: item.benchmark_mode || benchmarkMode,
            }))))
          : [];
        return normalizePromptFields(normalizeToolFields(normalizeErrorFields({
          ...record,
          model: record.model || model,
          prompt_text: record.prompt_text || promptText,
          run_id: record.run_id || normalized.run_id,
          run_started_at: record.run_started_at || normalized.started_at,
          benchmark_id: record.benchmark_id || normalized.benchmark_id || "",
          benchmark_title: record.benchmark_title || normalized.benchmark_title || normalized.benchmark_id || "",
          question_count: record.question_count ?? normalized.question_count ?? questionResults.length ?? null,
          model_info: normalizeModelInfo(record.model_info || normalized.model_info, record.model || model),
          benchmark_mode: benchmarkMode,
          question_results: questionResults,
        })));
      });

      normalized.model = model;
      normalized.prompt_text = promptText;
      normalized.model_info = normalizeModelInfo(normalized.model_info, model);
      normalized.records = enrichedRecords;
      delete normalized.models;
      delete normalized.config;
      delete normalized.summary;
      return normalized;
    }

    function numericValues(records, field) {
      return records
        .map((record) => record[field])
        .filter((value) => typeof value === "number" && Number.isFinite(value))
        .map((value) => Number(value));
    }

    function percentile(values, p) {
      if (!values.length) return null;
      if (values.length === 1) return values[0];
      const ordered = [...values].sort((a, b) => a - b);
      const rank = (ordered.length - 1) * p;
      const lower = Math.floor(rank);
      const upper = Math.ceil(rank);
      if (lower === upper) return ordered[lower];
      const fraction = rank - lower;
      return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction;
    }

    function mean(values) {
      if (!values.length) return null;
      return values.reduce((sum, value) => sum + value, 0) / values.length;
    }

    function median(values) {
      if (!values.length) return null;
      const ordered = [...values].sort((a, b) => a - b);
      const middle = Math.floor(ordered.length / 2);
      if (ordered.length % 2 === 0) {
        return (ordered[middle - 1] + ordered[middle]) / 2;
      }
      return ordered[middle];
    }

    function pstdev(values) {
      if (values.length <= 1) return 0;
      const currentMean = mean(values);
      if (currentMean == null) return 0;
      const variance = values.reduce((sum, value) => sum + ((value - currentMean) ** 2), 0) / values.length;
      return Math.sqrt(variance);
    }

    function metricStats(values) {
      if (!values.length) {
        return {
          count: 0,
          mean: null,
          median: null,
          p95: null,
          min: null,
          max: null,
          stddev: null,
          cv: null,
        };
      }
      const currentMean = mean(values);
      const currentStddev = pstdev(values);
      return {
        count: values.length,
        mean: currentMean,
        median: median(values),
        p95: percentile(values, 0.95),
        min: Math.min(...values),
        max: Math.max(...values),
        stddev: currentStddev,
        cv: currentMean ? (currentStddev / currentMean) : null,
      };
    }

    function countBy(records, keyFn) {
      const counts = {};
      for (const record of records) {
        const key = String(keyFn(record));
        counts[key] = (counts[key] || 0) + 1;
      }
      return counts;
    }

    function phaseSummary(records) {
      const successRecords = records.filter((record) => record.status === "success");
      const metrics = {};
      for (const metric of ALL_METRICS) {
        metrics[metric] = metricStats(numericValues(successRecords, metric));
      }
      const benchmarkCorrectCount = records.reduce((sum, record) => sum + Number(record.benchmark_correct_count || 0), 0);
      const benchmarkIncorrectCount = records.reduce((sum, record) => sum + Number(record.benchmark_incorrect_count || 0), 0);
      const benchmarkErrorCount = records.reduce((sum, record) => sum + Number(record.benchmark_error_count || 0), 0);
      const benchmarkTotalCount = benchmarkCorrectCount + benchmarkIncorrectCount + benchmarkErrorCount;
      return {
        samples: records.length,
        success_count: successRecords.length,
        error_count: Math.max(records.length - successRecords.length, 0),
        success_rate: records.length ? successRecords.length / records.length : 0,
        status_counts: countBy(records, (record) => record.status || "unknown"),
        finish_reasons: countBy(successRecords, (record) => record.finish_reason || "unknown"),
        metrics,
        benchmark: {
          correct_count: benchmarkCorrectCount,
          incorrect_count: benchmarkIncorrectCount,
          error_count: benchmarkErrorCount,
          total_count: benchmarkTotalCount,
          correct_rate: benchmarkTotalCount ? benchmarkCorrectCount / benchmarkTotalCount : null,
          error_rate: benchmarkTotalCount ? benchmarkErrorCount / benchmarkTotalCount : null,
        },
      };
    }

    function metricMean(summary, phase, metric) {
      return summary.phases?.[phase]?.metrics?.[metric]?.mean ?? null;
    }

    function flattenPhaseMetrics(summary, phaseName) {
      const phase = summary.phases[phaseName];
      summary[`${phaseName}_samples`] = phase.samples;
      summary[`${phaseName}_success_rate`] = phase.success_rate;
      summary[`${phaseName}_benchmark_correct_count`] = phase.benchmark?.correct_count ?? 0;
      summary[`${phaseName}_benchmark_incorrect_count`] = phase.benchmark?.incorrect_count ?? 0;
      summary[`${phaseName}_benchmark_error_count`] = phase.benchmark?.error_count ?? 0;
      summary[`${phaseName}_benchmark_total_count`] = phase.benchmark?.total_count ?? 0;
      summary[`${phaseName}_benchmark_correct_rate`] = phase.benchmark?.correct_rate ?? null;
      summary[`${phaseName}_benchmark_error_rate`] = phase.benchmark?.error_rate ?? null;
      for (const metric of ALL_METRICS) {
        const metricSummary = phase.metrics[metric];
        summary[`${phaseName}_mean_${metric}`] = metricSummary.mean;
        summary[`${phaseName}_p95_${metric}`] = metricSummary.p95;
        summary[`${phaseName}_stddev_${metric}`] = metricSummary.stddev;
        summary[`${phaseName}_cv_${metric}`] = metricSummary.cv;
      }
    }

    function delta(coldValue, warmValue) {
      if (typeof coldValue !== "number" || typeof warmValue !== "number") return null;
      return warmValue - coldValue;
    }

    function buildModelSummary(model, records, modelInfo) {
      const grouped = { cold: [], warm: [], overall: records };
      for (const record of records) {
        const phase = String(record.phase || "unknown");
        if (!grouped[phase]) grouped[phase] = [];
        grouped[phase].push(record);
      }

      const summary = {
        model,
        model_info: modelInfo || null,
        total_samples: records.length,
        phases: {
          cold: phaseSummary(grouped.cold || []),
          warm: phaseSummary(grouped.warm || []),
          overall: phaseSummary(records),
        },
      };
      summary.success_count = summary.phases.overall.success_count;
      summary.error_count = summary.phases.overall.error_count;
      summary.success_rate = summary.phases.overall.success_rate;
      summary.finish_reasons = summary.phases.overall.finish_reasons;
      summary.status_counts = summary.phases.overall.status_counts;
      summary.benchmark_correct_rate = summary.phases.overall.benchmark.correct_rate;
      summary.benchmark_error_rate = summary.phases.overall.benchmark.error_rate;
      summary.delta = {
        ttft_ms: delta(metricMean(summary, "cold", "ttft_ms"), metricMean(summary, "warm", "ttft_ms")),
        total_latency_ms: delta(metricMean(summary, "cold", "total_latency_ms"), metricMean(summary, "warm", "total_latency_ms")),
        decode_tps: delta(metricMean(summary, "cold", "decode_tps"), metricMean(summary, "warm", "decode_tps")),
      };
      for (const phaseName of ["cold", "warm", "overall"]) {
        flattenPhaseMetrics(summary, phaseName);
      }
      return summary;
    }

    function bestModel(rows, field, maximize = false) {
      const candidates = rows.filter((row) => typeof row[field] === "number" && Number.isFinite(row[field]));
      if (!candidates.length) return null;
      const ordered = [...candidates].sort((left, right) => maximize ? right[field] - left[field] : left[field] - right[field]);
      return { model: ordered[0].model, value: ordered[0][field] };
    }

    function summaryPayloadFromRecords(records, modelCatalog = {}) {
      const grouped = new Map();
      for (const record of records) {
        const comparisonModel = String(record.comparison_model || record.model || "(unknown)");
        if (!grouped.has(comparisonModel)) grouped.set(comparisonModel, []);
        grouped.get(comparisonModel).push(record);
      }

      const modelSummaries = [...grouped.entries()]
        .sort((left, right) => left[0].localeCompare(right[0]))
        .map(([model, modelRecords]) => buildModelSummary(model, modelRecords, modelCatalog[model] || null));

      const cardsSource = modelSummaries.map((summary) => ({
        model: summary.model,
        warm_mean_ttft_ms: summary.warm_mean_ttft_ms,
        warm_mean_total_latency_ms: summary.warm_mean_total_latency_ms,
        warm_mean_decode_tps: summary.warm_mean_decode_tps,
        warm_mean_benchmark_score: summary.warm_mean_benchmark_score,
        warm_benchmark_error_rate: summary.warm_benchmark_error_rate,
      }));
      const successfulSamples = records.filter((record) => record.status === "success").length;

      return {
        total_models: modelSummaries.length,
        total_samples: records.length,
        successful_samples: successfulSamples,
        failed_samples: Math.max(records.length - successfulSamples, 0),
        cards: {
          fastest_ttft: bestModel(cardsSource, "warm_mean_ttft_ms"),
          fastest_warm_latency: bestModel(cardsSource, "warm_mean_total_latency_ms"),
          fastest_decode_speed: bestModel(cardsSource, "warm_mean_decode_tps", true),
          best_benchmark_score: bestModel(cardsSource, "warm_mean_benchmark_score", true),
          lowest_benchmark_error_rate: bestModel(cardsSource, "warm_benchmark_error_rate"),
          total_samples: { value: records.length },
        },
        models: modelSummaries,
      };
    }

    function buildReportPayload(historyEntries) {
      const historyRuns = (Array.isArray(historyEntries) ? historyEntries : [historyEntries])
        .filter((entry) => entry && typeof entry === "object")
        .map((entry) => normalizeRunEntry(entry));
      backfillPromptMetricsFromPeers(historyRuns);

      const records = [];
      const prompts = [];
      const modelCatalog = {};
      for (const run of historyRuns) {
        if (run.prompt_text) prompts.push(run.prompt_text);
        const runModelInfo = normalizeModelInfo(run.model_info, run.model);
        const runComparisonModel = comparisonModelKey(run.model, runModelInfo);
        if (runComparisonModel && runModelInfo) {
          modelCatalog[runComparisonModel] = runModelInfo;
        }
        for (const record of Array.isArray(run.records) ? run.records : []) {
          const recordModelInfo = normalizeModelInfo(record.model_info)
            || runModelInfo
            || normalizeModelInfo(null, record.model || run.model)
            || null;
          records.push({
            ...record,
            comparison_model: comparisonModelKey(record.model || run.model, recordModelInfo),
            model_info: recordModelInfo,
            benchmark_mode: record.benchmark_mode || run.benchmark_mode || null,
            benchmark_id: record.benchmark_id || run.benchmark_id || null,
            benchmark_title: record.benchmark_title || run.benchmark_title || null,
            question_count: record.question_count || run.question_count || null,
          });
        }
      }

      const latestRun = historyRuns.length ? historyRuns[historyRuns.length - 1] : {};
      const uniquePrompts = [...new Set(prompts)];
      const promptPreview = uniquePrompts.length === 1 ? uniquePrompts[0] : (latestRun.prompt_text || "");
      const summary = summaryPayloadFromRecords(records, modelCatalog);
      summary.total_runs = historyRuns.length;
      summary.latest_run_id = latestRun.run_id || null;
      summary.latest_started_at = latestRun.started_at || null;

      return {
        generated_at: latestRun.ended_at || latestRun.started_at || "-",
        history_runs: historyRuns,
        latest_run: latestRun,
        records,
        summary,
        model_catalog: modelCatalog,
        prompt_preview: promptPreview,
        prompt_count: uniquePrompts.length,
        prompts: uniquePrompts,
      };
    }

    function availablePrompts() {
      return Array.isArray(reportData.prompts) ? reportData.prompts : [];
    }

    function promptSelectValue(prompt) {
      const index = availablePrompts().indexOf(prompt);
      return index >= 0 ? String(index + 1) : "";
    }

    function promptFromSelectValue(value) {
      const index = Number(value) - 1;
      const prompts = availablePrompts();
      return Number.isInteger(index) && index >= 0 && index < prompts.length ? prompts[index] : ALL_PROMPTS;
    }

    function syncPromptSelection() {
      const prompts = availablePrompts();
      if (!prompts.length) {
        state.selectedPrompt = ALL_PROMPTS;
        return prompts;
      }
      if (state.selectedPrompt === ALL_PROMPTS) {
        return prompts;
      }
      if (state.selectedPrompt && prompts.includes(state.selectedPrompt)) {
        return prompts;
      }
      if (prompts.length <= 1) {
        state.selectedPrompt = ALL_PROMPTS;
        return prompts;
      }
      state.selectedPrompt = firstText([reportData.latest_run?.prompt_text, prompts[prompts.length - 1], prompts[0]]);
      return prompts;
    }

    function filteredHistoryRunsByPrompt(promptText = state.selectedPrompt) {
      const historyRuns = Array.isArray(reportData.history_runs) ? reportData.history_runs : [];
      if (!promptText || promptText === ALL_PROMPTS) return historyRuns;
      return historyRuns.filter((run) => String(run.prompt_text || "") === promptText);
    }

    function refreshCurrentView() {
      syncPromptSelection();
      viewData = state.selectedPrompt && state.selectedPrompt !== ALL_PROMPTS
        ? buildReportPayload(filteredHistoryRunsByPrompt())
        : reportData;
    }

    function currentView() {
      return viewData && typeof viewData === "object" ? viewData : emptyPayload([]);
    }

    function currentSummary() {
      return currentView().summary || emptyPayload([]).summary;
    }

    function currentRecords() {
      return Array.isArray(currentView().records) ? currentView().records : [];
    }

    function currentModelRows() {
      return Array.isArray(currentSummary().models) ? currentSummary().models : [];
    }

    function currentHasBenchmarkMetrics() {
      return currentModelRows().some((row) => {
        const benchmarkTotals = [
          row.overall_benchmark_total_count,
          row.warm_benchmark_total_count,
          row.cold_benchmark_total_count,
        ];
        if (benchmarkTotals.some((value) => typeof value === "number" && Number.isFinite(value) && value > 0)) {
          return true;
        }
        return [
          row.warm_mean_benchmark_score,
          row.benchmark_correct_rate,
          row.warm_benchmark_error_rate,
        ].some((value) => typeof value === "number" && Number.isFinite(value));
      });
    }

    function toggleBenchmarkColumns(tableSelector, visible) {
      document.querySelectorAll(`${tableSelector} [data-benchmark-column="true"]`).forEach((cell) => {
        cell.style.display = visible ? "" : "none";
      });
    }

    function availableLeaderboardModels() {
      return currentModelRows()
        .map((row) => row.model)
        .filter(Boolean)
        .sort((left, right) => left.localeCompare(right));
    }

    function syncLeaderboardFilterSelection() {
      const models = availableLeaderboardModels();
      if (!state.leaderboardFilterTouched) {
        state.leaderboardSelectedModels = [...models];
        return models;
      }
      const available = new Set(models);
      state.leaderboardSelectedModels = (Array.isArray(state.leaderboardSelectedModels) ? state.leaderboardSelectedModels : [])
        .filter((model) => available.has(model));
      return models;
    }

    function selectedLeaderboardRows() {
      syncLeaderboardFilterSelection();
      const selected = new Set(state.leaderboardSelectedModels);
      if (!selected.size) return [];
      return currentModelRows().filter((row) => selected.has(row.model));
    }

    function currentLatestRun() {
      return currentView().latest_run || {};
    }

      function currentModelCatalog() {
        return currentView().model_catalog || {};
      }

      function resolveLogUrl(logPath) {
        if (!logPath || !historyBaseUrl) return "";
        try {
          return new URL(logPath, historyBaseUrl).toString();
        } catch (_error) {
          return "";
        }
      }

      function errorEventKey(event) {
        return [
          event.record_key || "-",
          event.question_id || "-",
          event.error_signature || "-",
          event.log_path || "-",
        ].join("::");
      }

      function flattenErrorEvents(records) {
        const events = [];
        for (const record of records) {
          const baseEvent = normalizeErrorFields({
            run_id: record.run_id,
            run_started_at: record.run_started_at || record.started_at,
            started_at: record.started_at,
            model: record.model,
            comparison_model: record.comparison_model || record.model,
            benchmark_id: record.benchmark_id,
            benchmark_title: record.benchmark_title,
            prompt_text: record.prompt_text,
            phase: record.phase,
            iteration: record.iteration,
            status: record.status,
            error: record.error,
            stderr_excerpt: record.stderr_excerpt,
            log_path: record.log_path,
            record_key: recordKey(record),
            source_level: "record",
          });
          const questionResults = Array.isArray(record.question_results)
            ? record.question_results.filter((item) => item && typeof item === "object")
            : [];
          const failingQuestions = questionResults.filter((item) => (item.status && item.status !== "success") || item.error);
          if (failingQuestions.length) {
            for (const questionResult of failingQuestions) {
              const event = normalizeErrorFields({
                ...questionResult,
                run_id: record.run_id,
                run_started_at: record.run_started_at || record.started_at,
                started_at: record.started_at,
                model: record.model,
                comparison_model: record.comparison_model || record.model,
                benchmark_id: record.benchmark_id,
                benchmark_title: record.benchmark_title,
                prompt_text: record.prompt_text,
                phase: record.phase,
                iteration: record.iteration,
                record_key: recordKey(record),
                source_level: "question",
              });
              if (event.error_signature || event.error || event.status !== "success") {
                events.push(event);
              }
            }
            continue;
          }
          if (baseEvent.error_signature || baseEvent.error || baseEvent.status !== "success") {
            events.push(baseEvent);
          }
        }
        return events;
      }

      function availableErrorEvents() {
        return flattenErrorEvents(currentRecords());
      }

      function filteredErrorEvents() {
        const query = filters.errorQuery.trim().toLowerCase();
        return availableErrorEvents().filter((event) => {
          if (filters.errorModel && (event.comparison_model || event.model) !== filters.errorModel) return false;
          if (filters.errorCategory && event.error_category !== filters.errorCategory) return false;
          if (!query) return true;
          const haystack = [
            event.error_signature,
            event.error_category,
            event.error,
            event.stderr_excerpt,
            event.run_id,
            event.model,
            event.comparison_model,
            event.benchmark_id,
            event.benchmark_title,
            event.question_id,
            event.phase,
          ].join("\\n").toLowerCase();
          return haystack.includes(query);
        });
      }

      function errorSignatureRows(events) {
        const grouped = new Map();
        for (const event of events) {
          const signature = event.error_signature || "(unknown)";
          if (!grouped.has(signature)) {
            grouped.set(signature, {
              signature,
              category: event.error_category || "other",
              count: 0,
              latest_started_at: event.run_started_at || event.started_at || "",
              models: new Set(),
              sample_benchmark: firstText([event.benchmark_title, event.benchmark_id]) || "-",
              sample_run_id: event.run_id || "-",
            });
          }
          const group = grouped.get(signature);
          group.count += 1;
          group.models.add(event.comparison_model || event.model || "(unknown)");
          const startedAt = event.run_started_at || event.started_at || "";
          if (startedAt >= group.latest_started_at) {
            group.latest_started_at = startedAt;
            group.sample_benchmark = firstText([event.benchmark_title, event.benchmark_id]) || "-";
            group.sample_run_id = event.run_id || "-";
            group.category = event.error_category || group.category;
          }
        }
        return [...grouped.values()]
          .map((group) => ({
            signature: group.signature,
            category: group.category,
            count: group.count,
            latest_started_at: group.latest_started_at,
            model_count: group.models.size,
            sample_benchmark: group.sample_benchmark,
            sample_run_id: group.sample_run_id,
          }))
          .sort((left, right) => right.count - left.count || String(right.latest_started_at).localeCompare(String(left.latest_started_at)));
      }

      function errorModelRows(events) {
        const summaryByModel = new Map(currentModelRows().map((row) => [row.model, row]));
        const grouped = new Map();
        for (const event of events) {
          const model = event.comparison_model || event.model || "(unknown)";
          if (!grouped.has(model)) grouped.set(model, []);
          grouped.get(model).push(event);
        }
        const allModels = new Set([...summaryByModel.keys(), ...grouped.keys()]);
        return [...allModels].map((model) => {
          const modelEvents = grouped.get(model) || [];
          const summary = summaryByModel.get(model) || null;
          const topSignature = errorSignatureRows(modelEvents)[0]?.signature || "-";
          return {
            model,
            error_rate: summary && typeof summary.total_samples === "number" && summary.total_samples > 0
              ? (Number(summary.error_count || 0) / summary.total_samples)
              : null,
            events: modelEvents.length,
            top_signature: topSignature,
            timeout_count: modelEvents.filter((event) => event.error_category === "timeout").length,
          };
        }).sort((left, right) => right.events - left.events || String(left.model).localeCompare(String(right.model)));
      }

    function comparisonModelKey(model, modelInfo) {
      const requestedModel = modelInfo?.requested_model || model || "(unknown)";
      const quantizationName = firstText([
        modelInfo?.quantization_name,
        extractQuantizationToken(modelInfo?.quantization),
        extractQuantizationToken(modelInfo?.display_name),
        extractQuantizationToken(modelInfo?.path),
      ]);
      const selectedVariant = modelInfo?.selected_variant || "";
      if (selectedVariant.includes("@")) {
        const variantSuffix = selectedVariant.split("@")[1] || "";
        const variantStem = variantSuffix.split("-")[0] || "";
        if (variantStem) {
          return `${requestedModel}-${variantStem}`;
        }
      }
      if (quantizationName) {
        const normalizedVariant = quantizationName.toLowerCase();
        const requestedLower = String(requestedModel).toLowerCase();
        const alreadySpecific = requestedLower.includes(`@${normalizedVariant}`)
          || requestedLower.endsWith(`-${normalizedVariant}`)
          || requestedLower.includes(`.${normalizedVariant}`);
        if (alreadySpecific) {
          return requestedModel;
        }
        return `${requestedModel}-${normalizedVariant}`;
      }
      if (modelInfo?.identifier && modelInfo.identifier !== requestedModel && !looksLikeFileSystemPath(modelInfo.identifier)) {
        return modelInfo.identifier;
      }
      const format = modelFormat(modelInfo).toLowerCase();
      if (format) {
        const requestedLower = String(requestedModel).toLowerCase();
        const alreadySpecific = requestedLower.includes("@")
          || requestedLower.endsWith(`-${format}`)
          || requestedLower.includes(`.${format}`);
        if (alreadySpecific) {
          return requestedModel;
        }
        return `${requestedModel}-${format}`;
      }
      return requestedModel;
    }

    function modelInfoFor(model) {
      const info = currentModelCatalog()[model];
      return normalizeModelInfo(info, model);
    }

    function modelDisplayName(model, modelInfo) {
      return modelInfo?.display_name || modelInfo?.model_key || model || "(unknown)";
    }

    function modelFormat(modelInfo) {
      return firstText([modelInfo?.format, modelInfo?.format_label]).toUpperCase();
    }

    function modelQuantization(modelInfo) {
      return firstText([modelInfo?.quantization, modelInfo?.quantization_name]);
    }

    function modelIdentityText(model, modelInfo) {
      const parts = [];
      if (modelInfo?.display_name) parts.push(modelInfo.display_name);
      if (modelInfo?.requested_model && modelInfo.requested_model !== model) parts.push(`requested:${modelInfo.requested_model}`);
      if (modelInfo?.identifier && modelInfo.identifier !== model && !looksLikeFileSystemPath(modelInfo.identifier)) parts.push(`id:${modelInfo.identifier}`);
      if (modelInfo?.selected_variant) parts.push(modelInfo.selected_variant);
      return parts.join(" · ");
    }

    function modelMetaText(model, modelInfo) {
      return [modelFormat(modelInfo), modelQuantization(modelInfo)].filter(Boolean).join(" · ") || modelIdentityText(model, modelInfo);
    }

    function formatMs(value) {
      return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)} ms` : "N/A";
    }

    function formatSec(value) {
      return typeof value === "number" && Number.isFinite(value) ? `${(value / 1000).toFixed(3)} s` : "N/A";
    }

    function formatTps(value) {
      return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)} tok/s` : "N/A";
    }

    function formatNumber(value, digits = 1) {
      return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "N/A";
    }

    function formatPercent(value) {
      return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "N/A";
    }

    function formatTime(value) {
      return value ? escapeHtml(value) : "-";
    }

    function sortClass(sortState, key) {
      if (sortState.key !== key) return "";
      return sortState.asc ? "sort-asc" : "sort-desc";
    }

    function normalizeSortValue(value) {
      if (typeof value === "number") return value;
      if (typeof value === "string") return value.toLowerCase();
      if (value == null) return Number.POSITIVE_INFINITY;
      return String(value).toLowerCase();
    }

    function sortRows(rows, sortState) {
      return [...rows].sort((left, right) => {
        const a = normalizeSortValue(left[sortState.key]);
        const b = normalizeSortValue(right[sortState.key]);
        if (a < b) return sortState.asc ? -1 : 1;
        if (a > b) return sortState.asc ? 1 : -1;
        return String(left.model || left.run_id || "").localeCompare(String(right.model || right.run_id || ""));
      });
    }

    function deltaClass(value) {
      if (typeof value !== "number" || !Number.isFinite(value)) return "delta-neutral";
      if (value < 0) return "delta-negative";
      if (value > 0) return "delta-positive";
      return "delta-neutral";
    }

    function deltaText(value, unit = "ms", digits = 1) {
      if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
      const sign = value > 0 ? "+" : "";
      return `${sign}${value.toFixed(digits)} ${unit}`;
    }

    function compareDeltaClass(value, preferHigher = false) {
      if (typeof value !== "number" || !Number.isFinite(value) || value === 0) return "delta-neutral";
      const leftBetter = preferHigher ? value > 0 : value < 0;
      return leftBetter ? "delta-negative" : "delta-positive";
    }

    function compareDeltaText(value, formatter, digits = 3, suffix = "") {
      if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
      const sign = value > 0 ? "+" : "";
      if (formatter === "seconds") {
        return `${sign}${(value / 1000).toFixed(digits)} s`;
      }
      if (formatter === "percent-points") {
        return `${sign}${(value * 100).toFixed(1)} pt`;
      }
      if (formatter === "tps") {
        return `${sign}${value.toFixed(2)} tok/s`;
      }
      return `${sign}${value.toFixed(digits)}${suffix}`;
    }

    function tokenBadges(map) {
      const entries = Object.entries(map || {});
      if (!entries.length) return '<span class="muted">-</span>';
      return `<div class="tag-list">${entries.map(([label, count]) => `<span class="tag">${escapeHtml(label)} · ${count}</span>`).join("")}</div>`;
    }

    function statusTone(status) {
      const normalized = String(status || "").trim().toLowerCase();
      if (normalized === "success") return "success";
      if (normalized === "timeout") return "timeout";
      if (normalized === "error") return "error";
      return "other";
    }

    function renderStatusPill(status) {
      const normalized = String(status || "unknown").trim() || "unknown";
      return `<span class="status-pill status-pill-${statusTone(normalized)}">${escapeHtml(normalized)}</span>`;
    }

    function renderPhasePill(phase, iteration) {
      const label = `${phase || "phase"} #${iteration ?? "-"}`;
      return `<span class="phase-pill">${escapeHtml(label)}</span>`;
    }

    function rankedToolEntries(value, limit = Infinity) {
      return Object.entries(normalizeCountMap(value))
        .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
        .slice(0, limit);
    }

    function summarizeToolsCompact(value, limit = 2) {
      const entries = rankedToolEntries(value, limit);
      if (!entries.length) return "-";
      const summary = entries.map(([name, count]) => `${name} x${count}`).join(" · ");
      const total = Object.keys(normalizeCountMap(value)).length;
      return total > entries.length ? `${summary} +${total - entries.length}` : summary;
    }

    function renderMetricLine(label, value) {
      return `
        <div class="detail-metric-line">
          <div class="detail-metric-label">${escapeHtml(label)}</div>
          <div class="detail-metric-value">${escapeHtml(value)}</div>
        </div>
      `;
    }

    function renderDetailKvRows(rows) {
      return rows.map(([label, value]) => `
        <tr>
          <th>${escapeHtml(label)}</th>
          <td>${value}</td>
        </tr>
      `).join("");
    }

    function renderDetailDisclosure(title, body, { open = false, count = "" } = {}) {
      const openAttr = open ? " open" : "";
      const countMarkup = count ? `<span class="muted">${escapeHtml(count)}</span>` : "";
      return `
        <details class="detail-disclosure"${openAttr}>
          <summary>
            <span>${escapeHtml(title)}</span>
            ${countMarkup}
          </summary>
          ${body}
        </details>
      `;
    }

      function recordKey(record) {
        return `${record.run_id || "-"}::${record.phase || "-"}::${record.iteration || "-"}::${record.started_at || "-"}`;
      }

      function showTab(selected) {
        document.querySelectorAll(".tab").forEach((node) => {
          node.classList.toggle("active", node.dataset.tab === selected);
        });
        Object.entries(panels).forEach(([key, panel]) => {
          panel.style.display = key === selected ? "block" : "none";
        });
      }

      function buildErrorFilters() {
        const events = availableErrorEvents();
        const modelSelect = document.getElementById("error-model-filter");
        const categorySelect = document.getElementById("error-category-filter");
        const models = [...new Set(events.map((event) => event.comparison_model || event.model).filter(Boolean))].sort();
        const categories = [...new Set(events.map((event) => event.error_category).filter(Boolean))].sort();
        if (filters.errorModel && !models.includes(filters.errorModel)) filters.errorModel = "";
        if (filters.errorCategory && !categories.includes(filters.errorCategory)) filters.errorCategory = "";
        modelSelect.innerHTML = ['<option value="">Model</option>'].concat(
          models.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
        ).join("");
        categorySelect.innerHTML = ['<option value="">Category</option>'].concat(
          categories.map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`)
        ).join("");
        modelSelect.value = filters.errorModel;
        categorySelect.value = filters.errorCategory;
      }

      async function populateRawLogPreview(event) {
        const node = document.getElementById("error-log-preview");
        if (!node) return;
        if (!event || !event.log_path) {
          node.textContent = "Raw log はありません。";
          return;
        }
        const url = resolveLogUrl(event.log_path);
        if (!url) {
          node.textContent = `Raw log path: ${event.log_path}\nfile input で history.json を開いた場合は自動読込できません。`;
          return;
        }
        if (state.rawLogCache[url]) {
          node.textContent = state.rawLogCache[url];
          return;
        }
        node.textContent = `loading ${url} ...`;
        try {
          const response = await fetch(url, { cache: "no-store" });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const text = await response.text();
          state.rawLogCache[url] = text;
          node.textContent = text;
        } catch (error) {
          const reason = error && error.message ? error.message : String(error);
          node.textContent = `Raw log の読込に失敗しました: ${reason}`;
        }
      }

      function openRecordDetails(recordKeyValue) {
        filters.model = "";
        filters.phase = "";
        filters.status = "";
        filters.query = "";
        document.getElementById("detail-model-filter").value = "";
        document.getElementById("detail-phase-filter").value = "";
        document.getElementById("detail-status-filter").value = "";
        document.getElementById("detail-query").value = "";
        state.selectedRecordKey = recordKeyValue;
        renderDetails();
        showTab("details");
      }

      function renderErrorDetailPane(event) {
        const pane = document.getElementById("error-detail-pane");
        if (!event) {
          pane.innerHTML = '<div class="muted">署名を選ぶと発生箇所と raw log を表示します。</div>';
          return;
        }
        const modelName = event.comparison_model || event.model || "(unknown)";
        const logUrl = resolveLogUrl(event.log_path);
        const locationText = event.question_id
          ? `${escapeHtml(event.phase)} #${event.iteration} · ${escapeHtml(event.question_id)}`
          : `${escapeHtml(event.phase)} #${event.iteration}`;
        const logLink = logUrl
          ? `<a class="btn btn-secondary" href="${escapeHtml(logUrl)}" target="_blank" rel="noopener noreferrer">Open Raw Log</a>`
          : `<span class="tag">${escapeHtml(event.log_path || "-")}</span>`;
        pane.innerHTML = `
          <h2>${escapeHtml(modelName)}</h2>
          <div class="catalog-meta-grid">
            <div class="catalog-meta-item">
              <div class="catalog-meta-label">Run ID</div>
              <div class="catalog-meta-value">${escapeHtml(event.run_id || "-")}</div>
            </div>
            <div class="catalog-meta-item">
              <div class="catalog-meta-label">Location</div>
              <div class="catalog-meta-value">${locationText}</div>
            </div>
            <div class="catalog-meta-item">
              <div class="catalog-meta-label">Category</div>
              <div class="catalog-meta-value">${escapeHtml(event.error_category || "-")}</div>
            </div>
            <div class="catalog-meta-item">
              <div class="catalog-meta-label">Benchmark</div>
              <div class="catalog-meta-value">${escapeHtml(firstText([event.benchmark_title, event.benchmark_id]) || "-")}</div>
            </div>
          </div>
          <div class="catalog-section-title">Signature</div>
          <pre class="catalog-prompt">${escapeHtml(event.error_signature || "-")}</pre>
          <div class="catalog-section-title" style="margin-top: 20px;">Error</div>
          <pre class="catalog-prompt">${escapeHtml(event.error || "-")}</pre>
          <div class="catalog-section-title" style="margin-top: 20px;">stderr excerpt</div>
          <pre class="catalog-prompt">${escapeHtml(event.stderr_excerpt || "-")}</pre>
          <div class="toolbar" style="margin-top: 16px; margin-bottom: 12px;">
            ${logLink}
            <button class="btn btn-secondary" id="open-record-detail" type="button">Open Run Details</button>
          </div>
          <div class="catalog-section-title">Raw Log Preview</div>
          <pre class="catalog-prompt" id="error-log-preview"></pre>
        `;
        document.getElementById("open-record-detail").addEventListener("click", () => {
          openRecordDetails(event.record_key);
        });
        void populateRawLogPreview(event);
      }

      function renderErrorAnalysis() {
        buildErrorFilters();
        const events = filteredErrorEvents();
        const signatureRows = errorSignatureRows(events);
        const modelRows = errorModelRows(events);
        const signatureBody = document.getElementById("error-signature-body");
        const modelBody = document.getElementById("error-model-body");
        const eventBody = document.getElementById("error-event-body");

        if (!signatureRows.length) {
          signatureBody.innerHTML = '<tr><td colspan="7" class="muted">分析対象のエラーがありません。</td></tr>';
          modelBody.innerHTML = '<tr><td colspan="5" class="muted">モデル別エラー集計はありません。</td></tr>';
          eventBody.innerHTML = '<tr><td colspan="5" class="muted">発生ログはありません。</td></tr>';
          renderErrorDetailPane(null);
          return;
        }

        if (!state.selectedErrorSignature || !signatureRows.some((row) => row.signature === state.selectedErrorSignature)) {
          state.selectedErrorSignature = signatureRows[0].signature;
        }
        const selectedEvents = events
          .filter((event) => (event.error_signature || "(unknown)") === state.selectedErrorSignature)
          .sort((left, right) => String(right.run_started_at || right.started_at).localeCompare(String(left.run_started_at || left.started_at)));
        if (!state.selectedErrorEventKey || !selectedEvents.some((event) => errorEventKey(event) === state.selectedErrorEventKey)) {
          state.selectedErrorEventKey = selectedEvents[0] ? errorEventKey(selectedEvents[0]) : "";
        }

        signatureBody.innerHTML = signatureRows.map((row) => {
          const active = row.signature === state.selectedErrorSignature ? "active" : "";
          return `
            <tr class="catalog-row ${active}" data-error-signature="${escapeHtml(row.signature)}">
              <td>${escapeHtml(row.signature)}</td>
              <td>${escapeHtml(row.category || "-")}</td>
              <td>${row.count}</td>
              <td>${formatTime(row.latest_started_at)}</td>
              <td>${row.model_count}</td>
              <td>${escapeHtml(row.sample_benchmark || "-")}</td>
              <td><span class="tag">${escapeHtml(row.sample_run_id || "-")}</span></td>
            </tr>
          `;
        }).join("");
        signatureBody.querySelectorAll("tr[data-error-signature]").forEach((rowEl) => {
          rowEl.addEventListener("click", () => {
            state.selectedErrorSignature = rowEl.dataset.errorSignature;
            state.selectedErrorEventKey = "";
            renderErrorAnalysis();
          });
        });

        modelBody.innerHTML = modelRows.length ? modelRows.map((row) => `
          <tr>
            <td>${escapeHtml(row.model)}</td>
            <td>${formatPercent(row.error_rate)}</td>
            <td>${row.events}</td>
            <td>${escapeHtml(truncateText(row.top_signature || "-", 80) || "-")}</td>
            <td>${row.timeout_count}</td>
          </tr>
        `).join("") : '<tr><td colspan="5" class="muted">モデル別エラー集計はありません。</td></tr>';

        eventBody.innerHTML = selectedEvents.length ? selectedEvents.map((event) => {
          const active = errorEventKey(event) === state.selectedErrorEventKey ? "active" : "";
          const location = event.question_id ? `${event.phase} #${event.iteration} · ${event.question_id}` : `${event.phase} #${event.iteration}`;
          return `
            <tr class="catalog-row ${active}" data-error-event-key="${escapeHtml(errorEventKey(event))}">
              <td>${formatTime(event.run_started_at || event.started_at)}</td>
              <td><span class="tag">${escapeHtml(event.run_id || "-")}</span></td>
              <td>${escapeHtml(event.comparison_model || event.model || "-")}</td>
              <td>${escapeHtml(location)}</td>
              <td>${escapeHtml(event.error_category || "-")}</td>
            </tr>
          `;
        }).join("") : '<tr><td colspan="5" class="muted">選択した signature の発生はありません。</td></tr>';
        eventBody.querySelectorAll("tr[data-error-event-key]").forEach((rowEl) => {
          rowEl.addEventListener("click", () => {
            state.selectedErrorEventKey = rowEl.dataset.errorEventKey;
            renderErrorAnalysis();
          });
        });

        renderErrorDetailPane(selectedEvents.find((event) => errorEventKey(event) === state.selectedErrorEventKey) || selectedEvents[0] || null);
      }

      function setLoadState(message, isError = false) {
        const node = document.getElementById("load-state");
      node.textContent = message;
      node.className = isError ? "muted load-state-error" : "muted";
    }

    function renderPromptSelector() {
      const prompts = syncPromptSelection();
      const selectNode = document.getElementById("prompt-filter");
      const metaNode = document.getElementById("prompt-filter-meta");
      const previewNode = document.getElementById("prompt-preview");
      const current = currentView();
      const selectedPrompt = state.selectedPrompt === ALL_PROMPTS ? "" : state.selectedPrompt;

      if (!prompts.length) {
        selectNode.innerHTML = '<option value="">Prompt</option>';
        selectNode.value = "";
        selectNode.disabled = true;
        metaNode.textContent = "利用可能な prompt がありません。";
        previewNode.textContent = "history.json に prompt がありません。";
        return;
      }

      selectNode.disabled = false;
      const options = ['<option value="">All Prompts</option>'].concat(
        prompts.map((prompt) => {
          const runCount = filteredHistoryRunsByPrompt(prompt).length;
          const label = `${truncateText(prompt, 88)} [${runCount} runs]`;
          return `<option value="${promptSelectValue(prompt)}">${escapeHtml(label)}</option>`;
        })
      );
      selectNode.innerHTML = options.join("");
      selectNode.value = promptSelectValue(selectedPrompt);

      if (selectedPrompt) {
        metaNode.textContent = `選択中: 1 / ${prompts.length} prompts · ${current.summary?.total_runs || 0} runs`;
        previewNode.textContent = selectedPrompt;
        return;
      }

      if (prompts.length === 1) {
        metaNode.textContent = `${current.summary?.total_runs || 0} runs`;
        previewNode.textContent = prompts[0];
        return;
      }

      metaNode.textContent = `全プロンプト集計 · ${prompts.length} prompts · ${current.summary?.total_runs || 0} runs`;
      previewNode.textContent =
        `全プロンプトを横断集計中 (${prompts.length}件)\n\n`
        + `Leaderboard / Compare / Run Details は上のセレクタで1つの prompt に絞り込めます。\n\n`
        + `最新プロンプト:\n${reportData.prompt_preview || prompts[prompts.length - 1]}`;
    }

    function setHeader() {
      const summary = currentSummary();
      const latestRun = currentLatestRun();
      const latestModelInfo = normalizeModelInfo(latestRun.model_info, latestRun.model)
        || modelInfoFor(comparisonModelKey(latestRun.model, normalizeModelInfo(latestRun.model_info, latestRun.model)));
      const latestComparisonModel = comparisonModelKey(latestRun.model, latestModelInfo);
      const totalHistoryRuns = reportData.summary?.total_runs || 0;
      document.getElementById("history-source").textContent = `source: ${sourceLabel}`;
      const meta = [
        `history runs: ${totalHistoryRuns}`,
        `visible runs: ${summary.total_runs || 0}`,
        `models: ${summary.total_models || 0}`,
        `prompts: ${reportData.prompt_count || 0}`,
        `latest run: ${escapeHtml(summary.latest_run_id || "-")}`,
        `loaded: ${escapeHtml(reportData.generated_at || "-")}`,
      ];
      document.getElementById("header-meta").innerHTML = meta.map((item) => `<span>${item}</span>`).join("");

      document.getElementById("summary-text").textContent =
        `runs=${summary.total_runs || 0} / models=${summary.total_models || 0} / samples=${summary.total_samples || 0} / success=${summary.successful_samples || 0}`;
      document.getElementById("subsummary").textContent =
        `latest: ${escapeHtml(latestComparisonModel || "-")} / ${escapeHtml(modelFormat(latestModelInfo) || "-")} / ${escapeHtml(modelQuantization(latestModelInfo) || "-")} / api: ${escapeHtml(latestRun.api_base || "-")}`;
    }

    function renderStats() {
      const summary = currentSummary();
      const showBenchmark = currentHasBenchmarkMetrics();
      const cards = [
        {
          label: "最速TTFT",
          value: summary.cards?.fastest_ttft?.value,
          model: summary.cards?.fastest_ttft?.model,
          formatter: formatMs,
          extra: "history 上の warm 平均で最短",
        },
        {
          label: "最速Warm Latency",
          value: summary.cards?.fastest_warm_latency?.value,
          model: summary.cards?.fastest_warm_latency?.model,
          formatter: formatMs,
          extra: "history 上の warm 平均総レイテンシ",
        },
        {
          label: "最速Decode Speed",
          value: summary.cards?.fastest_decode_speed?.value,
          model: summary.cards?.fastest_decode_speed?.model,
          formatter: formatTps,
          extra: "history 上の warm 平均 decode 速度",
        },
        {
          label: "総サンプル数",
          value: summary.cards?.total_samples?.value,
          model: "",
          formatter: (value) => typeof value === "number" ? String(value) : "0",
          extra: `${summary.successful_samples || 0} success / ${summary.failed_samples || 0} failed`,
        },
      ];
      if (showBenchmark) {
        cards.splice(3, 0, {
          label: "最高Benchmark Score",
          value: summary.cards?.best_benchmark_score?.value,
          model: summary.cards?.best_benchmark_score?.model,
          formatter: (value) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "N/A",
          extra: "history 上の warm 平均 exact-match score",
        });
      }
      document.getElementById("stats").innerHTML = cards.map((card) => `
        <div class="stat">
          <div class="label">${escapeHtml(card.label)}</div>
          <div class="value">${card.formatter(card.value)}</div>
          <div class="extra">${card.model ? `${escapeHtml(card.model)} · ` : ""}${escapeHtml(card.extra)}</div>
        </div>
      `).join("");
    }

    function renderLeaderboardFilter() {
      const models = syncLeaderboardFilterSelection();
      const selected = new Set(state.leaderboardSelectedModels);
      const summaryNode = document.getElementById("leaderboard-model-summary");
      const optionsNode = document.getElementById("leaderboard-model-options");
      const metaNode = document.getElementById("leaderboard-filter-meta");
      const selectAllButton = document.getElementById("leaderboard-select-all");
      const clearAllButton = document.getElementById("leaderboard-clear-all");

      if (!models.length) {
        summaryNode.textContent = "Models: 0";
        metaNode.textContent = "表示可能なモデルがありません。";
        optionsNode.innerHTML = '<div class="muted">history.json にモデル集計がありません。</div>';
        selectAllButton.disabled = true;
        clearAllButton.disabled = true;
        selectAllButton.onclick = null;
        clearAllButton.onclick = null;
        return;
      }

      const selectedCount = state.leaderboardSelectedModels.length;
      const allSelected = selectedCount === models.length;
      summaryNode.textContent = allSelected
        ? `Models: All (${models.length})`
        : `Models: ${selectedCount}/${models.length}`;
      metaNode.textContent = allSelected
        ? `${models.length} モデルを表示中`
        : `${selectedCount} / ${models.length} モデルを表示中`;
      optionsNode.innerHTML = models.map((model) => `
        <label class="filter-option">
          <input
            type="checkbox"
            class="leaderboard-model-checkbox"
            value="${escapeHtml(model)}"
            ${selected.has(model) ? "checked" : ""}
          />
          <span>${escapeHtml(model)}</span>
        </label>
      `).join("");

      optionsNode.querySelectorAll(".leaderboard-model-checkbox").forEach((checkbox) => {
        checkbox.addEventListener("change", (event) => {
          const nextSelected = new Set(state.leaderboardSelectedModels);
          const model = event.target.value;
          if (event.target.checked) {
            nextSelected.add(model);
          } else {
            nextSelected.delete(model);
          }
          state.leaderboardSelectedModels = models.filter((name) => nextSelected.has(name));
          state.leaderboardFilterTouched = true;
          renderLeaderboard();
        });
      });

      selectAllButton.disabled = allSelected;
      clearAllButton.disabled = selectedCount === 0;
      selectAllButton.onclick = () => {
        state.leaderboardSelectedModels = [...models];
        state.leaderboardFilterTouched = true;
        renderLeaderboard();
      };
      clearAllButton.onclick = () => {
        state.leaderboardSelectedModels = [];
        state.leaderboardFilterTouched = true;
        renderLeaderboard();
      };
    }

    function renderLeaderboard() {
      renderLeaderboardFilter();
      const showBenchmark = currentHasBenchmarkMetrics();
      const rows = sortRows(selectedLeaderboardRows().map((row) => ({
        ...row,
        format_sort: modelFormat(row.model_info),
        quantization_sort: modelQuantization(row.model_info),
      })), state.leaderboardSort);
      document.getElementById("leaderboard-body").innerHTML = rows.length ? rows.map((row) => `
        <tr>
          <td>
            <div class="tag">${escapeHtml(row.model)}</div>
          </td>
          <td>${escapeHtml(modelFormat(row.model_info) || "N/A")}</td>
          <td>${escapeHtml(modelQuantization(row.model_info) || "N/A")}</td>
          <td>${row.total_samples}</td>
          <td class="${row.success_rate === 1 ? "ok" : row.success_rate === 0 ? "ng" : ""}">${formatPercent(row.success_rate)}</td>
          ${showBenchmark ? `<td data-benchmark-column="true">${formatNumber(row.warm_mean_benchmark_score, 3)}</td>` : ""}
          ${showBenchmark ? `<td data-benchmark-column="true" class="${row.benchmark_correct_rate === 1 ? "ok" : row.benchmark_correct_rate === 0 ? "ng" : ""}">${formatPercent(row.benchmark_correct_rate)}</td>` : ""}
          ${showBenchmark ? `<td data-benchmark-column="true" class="${row.warm_benchmark_error_rate > 0 ? "ng" : "ok"}">${formatPercent(row.warm_benchmark_error_rate)}</td>` : ""}
          <td>${formatSec(row.warm_mean_ttft_ms)}</td>
          <td>${formatSec(row.warm_mean_total_latency_ms)}</td>
          <td>${formatTps(row.warm_mean_decode_tps)}</td>
          <td>${formatTps(row.warm_mean_initial_prompt_tps)}</td>
          <td>${formatTps(row.warm_mean_conversation_prompt_tps)}</td>
          <td>${formatSec(row.cold_mean_total_latency_ms)}</td>
        </tr>
      `).join("") : `<tr><td colspan="${showBenchmark ? 14 : 11}" class="muted">選択中のモデルに一致するデータがありません。</td></tr>`;
      toggleBenchmarkColumns("#leaderboard-table", showBenchmark);
      document.getElementById("leaderboard-note").textContent = showBenchmark
        ? "history.json 全体を集計しています。速度系は Warm 平均を中心に比較し、Correct は cold + warm を通した全体正答率です。Init Prompt は初回投入、Conv Prompt は会話全体の prompt throughput です。usage が返らないモデルでも、同一 prompt の実測 token 数が history 内にあれば代表値で補完します。参照がない場合のみ N/A です。"
        : "history.json 全体を集計しています。速度系は Warm 平均を中心に比較します。Init Prompt は初回投入、Conv Prompt は会話全体の prompt throughput です。usage が返らないモデルでも、同一 prompt の実測 token 数が history 内にあれば代表値で補完します。参照がない場合のみ N/A です。";
      document.querySelectorAll("#leaderboard-table th[data-sort]").forEach((th) => {
        th.className = sortClass(state.leaderboardSort, th.dataset.sort);
      });
    }

    function syncCompareSelection() {
      const modelIds = currentModelRows().map((row) => row.model);
      if (!modelIds.length) {
        state.compare.leftModel = "";
        state.compare.rightModel = "";
        return;
      }
      if (!modelIds.includes(state.compare.leftModel)) {
        state.compare.leftModel = modelIds[0];
      }
      if (!modelIds.includes(state.compare.rightModel)) {
        state.compare.rightModel = modelIds[1] || modelIds[0];
      }
      if (!state.compare.rightModel) {
        state.compare.rightModel = modelIds[1] || modelIds[0];
      }
    }

    function compareTextRow(label, leftValue, rightValue) {
      return `
        <tr>
          <td>${escapeHtml(label)}</td>
          <td>${escapeHtml(leftValue || "N/A")}</td>
          <td>${escapeHtml(rightValue || "N/A")}</td>
          <td class="delta-neutral">-</td>
        </tr>
      `;
    }

    function compareMetricRow(label, leftValue, rightValue, valueFormatter, deltaFormatter, preferHigher = false) {
      const delta = (typeof leftValue === "number" && Number.isFinite(leftValue) && typeof rightValue === "number" && Number.isFinite(rightValue))
        ? leftValue - rightValue
        : null;
      return `
        <tr>
          <td>${escapeHtml(label)}</td>
          <td>${valueFormatter(leftValue)}</td>
          <td>${valueFormatter(rightValue)}</td>
          <td class="${compareDeltaClass(delta, preferHigher)}">${deltaFormatter(delta)}</td>
        </tr>
      `;
    }

    function renderCompare() {
      syncCompareSelection();
      const showBenchmark = currentHasBenchmarkMetrics();
      const rows = currentModelRows();
      const leftSelect = document.getElementById("compare-left-model");
      const rightSelect = document.getElementById("compare-right-model");
      const compareBody = document.getElementById("compare-body");

      const options = rows.map((row) => `<option value="${escapeHtml(row.model)}">${escapeHtml(row.model)}</option>`).join("");
      leftSelect.innerHTML = options;
      rightSelect.innerHTML = options;
      leftSelect.value = state.compare.leftModel;
      rightSelect.value = state.compare.rightModel;

      if (rows.length < 2) {
        document.getElementById("compare-left-heading").textContent = "Model A";
        document.getElementById("compare-right-heading").textContent = "Model B";
        compareBody.innerHTML = '<tr><td colspan="4" class="muted">比較には少なくとも 2 モデル必要です。</td></tr>';
        return;
      }

      const leftRow = rows.find((row) => row.model === state.compare.leftModel) || rows[0];
      const rightRow = rows.find((row) => row.model === state.compare.rightModel) || rows[1] || rows[0];
      document.getElementById("compare-left-heading").textContent = leftRow.model;
      document.getElementById("compare-right-heading").textContent = rightRow.model;

      const leftInfo = leftRow.model_info;
      const rightInfo = rightRow.model_info;
      compareBody.innerHTML = [
        compareTextRow("Format", modelFormat(leftInfo), modelFormat(rightInfo)),
        compareTextRow("Quantization", modelQuantization(leftInfo), modelQuantization(rightInfo)),
        compareMetricRow("Success Rate", leftRow.success_rate, rightRow.success_rate, formatPercent, (value) => compareDeltaText(value, "percent-points"), true),
        ...(showBenchmark ? [
          compareMetricRow("Warm Benchmark Score", leftRow.warm_mean_benchmark_score, rightRow.warm_mean_benchmark_score, (value) => formatNumber(value, 3), (value) => compareDeltaText(value, "number", 3), true),
          compareMetricRow("Benchmark Correct Rate", leftRow.benchmark_correct_rate, rightRow.benchmark_correct_rate, formatPercent, (value) => compareDeltaText(value, "percent-points"), true),
          compareMetricRow("Benchmark Error Rate", leftRow.warm_benchmark_error_rate, rightRow.warm_benchmark_error_rate, formatPercent, (value) => compareDeltaText(value, "percent-points")),
        ] : []),
        compareMetricRow("Samples", leftRow.total_samples, rightRow.total_samples, (value) => typeof value === "number" ? String(value) : "N/A", (value) => compareDeltaText(value, "number", 0), true),
        compareMetricRow("Warm TTFT", leftRow.warm_mean_ttft_ms, rightRow.warm_mean_ttft_ms, formatSec, (value) => compareDeltaText(value, "seconds")),
        compareMetricRow("Warm Latency", leftRow.warm_mean_total_latency_ms, rightRow.warm_mean_total_latency_ms, formatSec, (value) => compareDeltaText(value, "seconds")),
        compareMetricRow("Cold Latency", leftRow.cold_mean_total_latency_ms, rightRow.cold_mean_total_latency_ms, formatSec, (value) => compareDeltaText(value, "seconds")),
        compareMetricRow("Warm Decode Speed", leftRow.warm_mean_decode_tps, rightRow.warm_mean_decode_tps, formatTps, (value) => compareDeltaText(value, "tps"), true),
        compareMetricRow("Warm Initial Prompt Speed", leftRow.warm_mean_initial_prompt_tps, rightRow.warm_mean_initial_prompt_tps, formatTps, (value) => compareDeltaText(value, "tps"), true),
        compareMetricRow("Warm Conversation Prompt Speed", leftRow.warm_mean_conversation_prompt_tps, rightRow.warm_mean_conversation_prompt_tps, formatTps, (value) => compareDeltaText(value, "tps"), true),
        compareMetricRow("Warm TTFT p95", leftRow.warm_p95_ttft_ms, rightRow.warm_p95_ttft_ms, formatSec, (value) => compareDeltaText(value, "seconds")),
        compareMetricRow("Warm Latency Stddev", leftRow.warm_stddev_total_latency_ms, rightRow.warm_stddev_total_latency_ms, formatSec, (value) => compareDeltaText(value, "seconds")),
        compareMetricRow("Warm Latency CV", leftRow.warm_cv_total_latency_ms, rightRow.warm_cv_total_latency_ms, (value) => formatNumber(value, 3), (value) => compareDeltaText(value, "number", 3)),
        compareMetricRow("Output Tokens", leftRow.overall_mean_completion_tokens, rightRow.overall_mean_completion_tokens, (value) => formatNumber(value, 1), (value) => compareDeltaText(value, "number", 1), true),
        compareMetricRow("Errors", leftRow.error_count, rightRow.error_count, (value) => typeof value === "number" ? String(value) : "N/A", (value) => compareDeltaText(value, "number", 0)),
      ].join("");
    }

    function renderColdWarm() {
      const rows = currentModelRows().map((row) => ({
        ...row,
        delta_ttft_ms: row.delta?.ttft_ms,
        delta_total_latency_ms: row.delta?.total_latency_ms,
      }));
      const sorted = sortRows(rows, state.coldwarmSort);
      document.getElementById("coldwarm-body").innerHTML = sorted.length ? sorted.map((row) => `
        <tr>
          <td><span class="tag">${escapeHtml(row.model)}</span></td>
          <td>${formatMs(row.cold_mean_ttft_ms)}</td>
          <td>${formatMs(row.warm_mean_ttft_ms)}</td>
          <td class="${deltaClass(row.delta_ttft_ms)}">${deltaText(row.delta_ttft_ms)}</td>
          <td>${formatMs(row.cold_mean_total_latency_ms)}</td>
          <td>${formatMs(row.warm_mean_total_latency_ms)}</td>
          <td class="${deltaClass(row.delta_total_latency_ms)}">${deltaText(row.delta_total_latency_ms)}</td>
          <td>${formatTps(row.warm_mean_decode_tps)}</td>
        </tr>
      `).join("") : '<tr><td colspan="8" class="muted">cold / warm データがありません。</td></tr>';
      document.querySelectorAll("#coldwarm-table th[data-sort]").forEach((th) => {
        th.className = sortClass(state.coldwarmSort, th.dataset.sort);
      });
    }

    function renderStability() {
      const showBenchmark = currentHasBenchmarkMetrics();
      const rows = sortRows(currentModelRows(), state.stabilitySort);
      document.getElementById("stability-body").innerHTML = rows.length ? rows.map((row) => `
        <tr>
          <td>
            <div class="tag">${escapeHtml(row.model)}</div>
            <div class="muted" style="margin-top: 8px;">${escapeHtml(modelMetaText(row.model, row.model_info) || "N/A")}</div>
            <div class="table-note">${tokenBadges(row.finish_reasons)}</div>
          </td>
          <td class="${row.success_rate === 1 ? "ok" : row.success_rate === 0 ? "ng" : ""}">${formatPercent(row.success_rate)}</td>
          ${showBenchmark ? `<td data-benchmark-column="true">${formatNumber(row.warm_mean_benchmark_score, 3)}</td>` : ""}
          ${showBenchmark ? `<td data-benchmark-column="true">${formatPercent(row.benchmark_correct_rate)}</td>` : ""}
          ${showBenchmark ? `<td data-benchmark-column="true">${formatPercent(row.warm_benchmark_error_rate)}</td>` : ""}
          <td>${formatMs(row.warm_stddev_total_latency_ms)}</td>
          <td>${formatNumber(row.warm_cv_total_latency_ms, 3)}</td>
          <td>${formatMs(row.warm_p95_ttft_ms)}</td>
          <td>${row.error_count}</td>
          <td>${row.warm_samples}</td>
          <td>${formatNumber(row.overall_mean_completion_tokens, 1)}</td>
        </tr>
      `).join("") : `<tr><td colspan="${showBenchmark ? 11 : 8}" class="muted">stability に使えるデータがありません。</td></tr>`;
      toggleBenchmarkColumns("#stability-table", showBenchmark);
      document.querySelectorAll("#stability-table th[data-sort]").forEach((th) => {
        th.className = sortClass(state.stabilitySort, th.dataset.sort);
      });
    }

    function buildDetailFilters() {
      const records = currentRecords();
      const modelSelect = document.getElementById("detail-model-filter");
      const statusSelect = document.getElementById("detail-status-filter");
      const models = [...new Set(records.map((record) => record.comparison_model || record.model).filter(Boolean))].sort();
      const statuses = [...new Set(records.map((record) => record.status).filter(Boolean))].sort();
      if (filters.model && !models.includes(filters.model)) filters.model = "";
      if (filters.status && !statuses.includes(filters.status)) filters.status = "";
      modelSelect.innerHTML = ['<option value="">Model</option>'].concat(
        models.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
      ).join("");
      statusSelect.innerHTML = ['<option value="">Status</option>'].concat(
        statuses.map((status) => `<option value="${escapeHtml(status)}">${escapeHtml(status)}</option>`)
      ).join("");
      modelSelect.value = filters.model;
      statusSelect.value = filters.status;
    }

      function filteredRecords() {
        const query = filters.query.trim().toLowerCase();
        return currentRecords().filter((record) => {
        if (filters.model && (record.comparison_model || record.model) !== filters.model) return false;
        if (filters.phase && record.phase !== filters.phase) return false;
        if (filters.status && record.status !== filters.status) return false;
        if (!query) return true;
        const recordModelInfo = normalizeModelInfo(record.model_info, record.model) || modelInfoFor(record.comparison_model || record.model);
        const haystack = [
          record.run_id,
          record.model,
          record.comparison_model,
          record.prompt_text,
          modelDisplayName(record.model, recordModelInfo),
          modelFormat(recordModelInfo),
          modelQuantization(recordModelInfo),
            record.phase,
            record.status,
            record.tool_call_count,
            summarizeCountMap(record.tool_name_counts),
            record.error_signature,
            record.error_category,
            record.error,
            ...(Array.isArray(record.question_results) ? record.question_results.map((item) => [
              item.question_id,
              item.tool_call_count,
              summarizeCountMap(item.tool_name_counts),
              item.error_signature,
              item.error,
              item.stderr_excerpt,
            ].join("\\n")) : []),
            record.reasoning_text,
            record.response_text,
            record.predicted_answer,
          record.benchmark_id,
          record.benchmark_title,
        ].join("\\n").toLowerCase();
        return haystack.includes(query);
      });
    }

      function renderDetailPane(record) {
        const pane = document.getElementById("detail-pane");
        if (!record) {
          pane.innerHTML = '<div class="detail-empty">上段の実行一覧から 1 件選ぶと、この領域に概要、指標、ツール利用、本文ログを整理して表示します。</div>';
          return;
        }
        const recordModelInfo = normalizeModelInfo(record.model_info, record.model) || modelInfoFor(record.comparison_model || record.model);
        const comparisonModel = record.comparison_model || record.model;
        const logUrl = resolveLogUrl(record.log_path);
        const questionResults = Array.isArray(record.question_results)
          ? record.question_results.filter((item) => item && typeof item === "object")
          : [];
        const toolEntries = rankedToolEntries(record.tool_name_counts, 10);
        const toolBreakdownMarkup = toolEntries.length
          ? `<div class="detail-tool-breakdown">${toolEntries.map(([name, count]) => `
              <div class="detail-tool-row">
                <strong>${escapeHtml(name)}</strong>
                <span>${escapeHtml(String(count))} call${count === 1 ? "" : "s"}</span>
              </div>
            `).join("")}</div>`
          : '<div class="muted">この試行でツール呼び出しは記録されていません。</div>';
        const questionTableMarkup = questionResults.length
          ? questionResults.map((item) => {
            const itemLogUrl = resolveLogUrl(item.log_path);
            const logMarkup = itemLogUrl
              ? `<a href="${escapeHtml(itemLogUrl)}" target="_blank" rel="noopener noreferrer">Open</a>`
              : escapeHtml(item.log_path || "-");
            return `
              <tr>
                <td>${escapeHtml(item.question_id || "-")}</td>
                <td>${renderStatusPill(item.status || "-")}</td>
                <td>${formatNumber(item.tool_call_count, 0)}</td>
                <td>${formatNumber(item.benchmark_score, 3)}</td>
                <td>${escapeHtml(truncateText(item.error_signature || item.predicted_answer || "-", 88) || "-")}</td>
                <td>${logMarkup}</td>
              </tr>
            `;
          }).join("")
          : '<tr><td colspan="6" class="muted">question 単位の記録はありません。</td></tr>';

        const summaryTags = [
          renderStatusPill(record.status),
          renderPhasePill(record.phase, record.iteration),
          record.benchmark_title ? `<span class="tag">${escapeHtml(record.benchmark_title)}</span>` : "",
          `<span class="tag">Run ${escapeHtml(record.run_id || "-")}</span>`,
          record.error_category ? `<span class="tag">${escapeHtml(record.error_category)}</span>` : "",
        ].filter(Boolean).join("");

        const overviewRows = renderDetailKvRows([
          ["Run ID", `<span class="tag">${escapeHtml(record.run_id || "-")}</span>`],
          ["Requested Model", escapeHtml(record.model || "-")],
          ["LM Studio Model", escapeHtml(modelDisplayName(record.model, recordModelInfo))],
          ["Benchmark ID", escapeHtml(record.benchmark_id || "-")],
          ["Benchmark Title", escapeHtml(record.benchmark_title || "-")],
          ["Started", formatTime(record.run_started_at || record.started_at)],
          ["Format", escapeHtml(modelFormat(recordModelInfo) || "-")],
          ["Quantization", escapeHtml(modelQuantization(recordModelInfo) || "-")],
        ]);

        const benchmarkRows = renderDetailKvRows([
          ["Predicted Answer", escapeHtml(record.predicted_answer || "-")],
          ["Benchmark Score", escapeHtml(formatNumber(record.benchmark_score, 3))],
          ["Correct Count", escapeHtml(String(record.benchmark_correct_count ?? "-"))],
          ["Incorrect Count", escapeHtml(String(record.benchmark_incorrect_count ?? "-"))],
          ["Benchmark Error Count", escapeHtml(String(record.benchmark_error_count ?? "-"))],
          ["Finish Reason", escapeHtml(record.finish_reason || "-")],
          ["Question Count", escapeHtml(record.question_count != null ? String(record.question_count) : "-")],
        ]);

        const performanceRows = renderDetailKvRows([
          ["TTFT", escapeHtml(formatMs(record.ttft_ms))],
          ["Total Latency", escapeHtml(formatMs(record.total_latency_ms))],
          ["Completion Window", escapeHtml(formatMs(record.completion_window_ms))],
          ["Initial Prompt Latency", escapeHtml(formatMs(record.initial_prompt_latency_ms))],
          ["Initial Prompt Speed", escapeHtml(formatTps(record.initial_prompt_tps))],
          ["Conversation Prompt Latency", escapeHtml(formatMs(record.conversation_prompt_latency_ms))],
          ["Conversation Prompt Speed", escapeHtml(formatTps(record.conversation_prompt_tps))],
          ["Decode Speed", escapeHtml(formatTps(record.decode_tps))],
          ["End-to-End Speed", escapeHtml(formatTps(record.end_to_end_tps))],
          ["Initial Prompt Tokens", escapeHtml(formatNumber(record.initial_prompt_tokens, 0))],
          ["Conversation Prompt Tokens", escapeHtml(formatNumber(record.conversation_prompt_tokens, 0))],
          ["Completion Tokens", escapeHtml(formatNumber(record.completion_tokens, 0))],
          ["Total Tokens", escapeHtml(formatNumber(record.total_tokens, 0))],
        ]);

        const promptText = String(record.prompt_text || "");
        const reasoningText = String(record.reasoning_text || "");
        const responseText = String(record.response_text || "");
        const errorText = String(record.error || "");
        const sections = [
          renderDetailDisclosure("Prompt", `<pre class="catalog-prompt">${escapeHtml(promptText || "-")}</pre>`, {
            open: false,
            count: record.conversation_prompt_tokens != null ? `${formatNumber(record.conversation_prompt_tokens, 0)} tok` : "",
          }),
        ];
        if (errorText || record.error_signature) {
          sections.push(
            renderDetailDisclosure("Error", `<pre class="catalog-prompt">${escapeHtml(errorText || record.error_signature || "-")}</pre>`, {
              open: record.status !== "success",
              count: record.error_category || "",
            }),
          );
        }
        if (reasoningText) {
          sections.push(
            renderDetailDisclosure("Reasoning", `<pre class="catalog-prompt">${escapeHtml(reasoningText)}</pre>`, {
              open: false,
            }),
          );
        }
        if (responseText || record.predicted_answer) {
          sections.push(
            renderDetailDisclosure("Response", `<pre class="catalog-prompt">${escapeHtml(responseText || "-")}</pre>`, {
              open: record.status === "success",
            }),
          );
        }

        pane.innerHTML = `
          <div class="detail-pane-shell">
            <div class="detail-pane-header">
              <div class="detail-pane-title-block">
                <h2 class="detail-pane-title">${escapeHtml(comparisonModel)}</h2>
                <div class="detail-pane-subtitle">${escapeHtml(modelIdentityText(record.model, recordModelInfo) || "選択した run の詳細")}</div>
                <div class="detail-row-tags">${summaryTags}</div>
              </div>
              <div class="detail-pane-actions">
                ${logUrl ? `<a class="btn btn-secondary" href="${escapeHtml(logUrl)}" target="_blank" rel="noopener noreferrer">Open Raw Log</a>` : `<span class="tag">${escapeHtml(record.log_path || "-")}</span>`}
              </div>
            </div>

            <div class="detail-summary-grid">
              <div class="detail-summary-item">
                <div class="detail-metric-label">Total Latency</div>
                <div class="detail-metric-value">${escapeHtml(formatMs(record.total_latency_ms))}</div>
              </div>
              <div class="detail-summary-item">
                <div class="detail-metric-label">TTFT</div>
                <div class="detail-metric-value">${escapeHtml(formatMs(record.ttft_ms))}</div>
              </div>
              <div class="detail-summary-item">
                <div class="detail-metric-label">Initial Prompt</div>
                <div class="detail-metric-value">${escapeHtml(formatTps(record.initial_prompt_tps))}</div>
              </div>
              <div class="detail-summary-item">
                <div class="detail-metric-label">Conv Prompt</div>
                <div class="detail-metric-value">${escapeHtml(formatTps(record.conversation_prompt_tps))}</div>
              </div>
              <div class="detail-summary-item">
                <div class="detail-metric-label">Tool Calls</div>
                <div class="detail-metric-value">${escapeHtml(formatNumber(record.tool_call_count, 0))}</div>
              </div>
            </div>

            <div class="detail-columns">
              <div class="detail-stack">
                <section class="detail-section">
                  <h3 class="detail-section-heading">Run Context</h3>
                  <table class="detail-kv-table"><tbody>${overviewRows}</tbody></table>
                </section>
                <section class="detail-section">
                  <h3 class="detail-section-heading">Performance</h3>
                  <table class="detail-kv-table"><tbody>${performanceRows}</tbody></table>
                </section>
              </div>
              <div class="detail-stack">
                <section class="detail-section">
                  <h3 class="detail-section-heading">Benchmark</h3>
                  <table class="detail-kv-table"><tbody>${benchmarkRows}</tbody></table>
                </section>
                <section class="detail-section">
                  <h3 class="detail-section-heading">Tool Activity</h3>
                  <table class="detail-kv-table">
                    <tbody>
                      ${renderDetailKvRows([
                        ["Tool Calls", escapeHtml(formatNumber(record.tool_call_count, 0))],
                        ["Top Tools", escapeHtml(summarizeCountMap(record.tool_name_counts) || "-")],
                        ["Error Signature", escapeHtml(record.error_signature || "-")],
                      ])}
                    </tbody>
                  </table>
                  ${toolBreakdownMarkup}
                </section>
              </div>
            </div>

            <section class="detail-section">
              <h3 class="detail-section-heading">Question Results</h3>
              <table>
                <thead>
                  <tr>
                    <th>Question</th>
                    <th>Status</th>
                    <th>Tool Calls</th>
                    <th>Score</th>
                    <th>Outcome</th>
                    <th>Log</th>
                  </tr>
                </thead>
                <tbody>${questionTableMarkup}</tbody>
              </table>
            </section>

            <section class="detail-section">
              <h3 class="detail-section-heading">Transcript</h3>
              ${sections.join("")}
            </section>
          </div>
        `;
      }

    function renderDetails() {
      const body = document.getElementById("detail-body");
      const rows = sortRows(filteredRecords(), state.detailSort);
      if (!rows.length) {
        body.innerHTML = '<tr><td colspan="6" class="muted">No matching runs.</td></tr>';
        renderDetailPane(null);
        return;
      }
      if (!state.selectedRecordKey || !rows.some((row) => recordKey(row) === state.selectedRecordKey)) {
        state.selectedRecordKey = recordKey(rows[0]);
      }
      body.innerHTML = rows.map((row) => {
        const active = recordKey(row) === state.selectedRecordKey ? "active" : "";
        const modelInfo = normalizeModelInfo(row.model_info, row.model) || modelInfoFor(row.comparison_model || row.model);
        const outcomeDetail = row.status === "success"
          ? `score ${formatNumber(row.benchmark_score, 3)} · ${row.finish_reason || "done"}`
          : (row.error_category || "error");
        return `
          <tr class="catalog-row ${active}" data-record-key="${escapeHtml(recordKey(row))}">
            <td>
              <div class="detail-row-cell">
                <div class="detail-row-title">${formatTime(row.run_started_at || row.started_at)}</div>
                <div class="detail-row-meta">${escapeHtml(row.run_id || "-")}</div>
              </div>
            </td>
            <td>
              <div class="detail-row-cell">
                <div class="detail-row-title">${escapeHtml(row.comparison_model || row.model)}</div>
                <div class="detail-row-meta">${escapeHtml(modelMetaText(row.comparison_model || row.model, modelInfo) || "-")}</div>
                <div class="detail-row-tags">
                  <span class="tag">${escapeHtml(row.run_id || "-")}</span>
                </div>
              </div>
            </td>
            <td>
              <div class="detail-row-cell">
                <div class="detail-row-title">${escapeHtml(row.benchmark_title || row.benchmark_id || "-")}</div>
                <div class="detail-row-tags">
                  ${renderPhasePill(row.phase, row.iteration)}
                  ${row.question_count != null ? `<span class="tag">${escapeHtml(String(row.question_count))}q</span>` : ""}
                </div>
                <div class="detail-row-meta">${escapeHtml(row.benchmark_id || "-")}</div>
              </div>
            </td>
            <td>
              <div class="detail-metric-grid">
                ${renderMetricLine("Latency", formatMs(row.total_latency_ms))}
                ${renderMetricLine("TTFT", formatMs(row.ttft_ms))}
                ${renderMetricLine("Decode", formatTps(row.decode_tps))}
                ${renderMetricLine("Init Prompt", formatTps(row.initial_prompt_tps))}
                ${renderMetricLine("Conv Prompt", formatTps(row.conversation_prompt_tps))}
              </div>
            </td>
            <td>
              <div class="detail-row-cell">
                <div class="detail-row-title">${formatNumber(row.tool_call_count, 0)}</div>
                <div class="detail-row-meta">${escapeHtml(summarizeToolsCompact(row.tool_name_counts, 2))}</div>
              </div>
            </td>
            <td>
              <div class="detail-row-cell">
                <div class="detail-row-tags">${renderStatusPill(row.status)}</div>
                <div class="detail-row-meta">${escapeHtml(outcomeDetail)}</div>
                <div class="detail-signature-preview">${escapeHtml(truncateText(row.error_signature || row.predicted_answer || "", 88) || "-")}</div>
              </div>
            </td>
          </tr>
        `;
      }).join("");
      document.querySelectorAll("#detail-table th[data-sort]").forEach((th) => {
        th.className = sortClass(state.detailSort, th.dataset.sort);
      });
      document.querySelectorAll(".catalog-row").forEach((rowEl) => {
        rowEl.addEventListener("click", () => {
          state.selectedRecordKey = rowEl.dataset.recordKey;
          renderDetails();
        });
      });
      renderDetailPane(rows.find((row) => recordKey(row) === state.selectedRecordKey) || rows[0]);
    }

    function bindSorters(tableSelector, sortState, rerender) {
      document.querySelectorAll(`${tableSelector} th[data-sort]`).forEach((th) => {
        th.addEventListener("click", () => {
          const key = th.dataset.sort;
          if (sortState.key === key) {
            sortState.asc = !sortState.asc;
          } else {
            sortState.key = key;
            sortState.asc = !["success_rate", "warm_mean_decode_tps"].includes(key);
          }
          rerender();
        });
      });
    }

      function bindTabs() {
        document.querySelectorAll(".tab").forEach((tab) => {
          tab.addEventListener("click", () => {
            showTab(tab.dataset.tab);
          });
        });
      }

      function bindPromptControls() {
        document.getElementById("prompt-filter").addEventListener("change", (event) => {
          state.selectedPrompt = promptFromSelectValue(event.target.value);
          state.leaderboardSelectedModels = [];
          state.leaderboardFilterTouched = false;
          state.selectedErrorSignature = "";
          state.selectedErrorEventKey = "";
          state.selectedRecordKey = "";
          refreshCurrentView();
          renderAll();
        });
      }

      function bindDetailFilters() {
        document.getElementById("detail-model-filter").addEventListener("change", (event) => {
          filters.model = event.target.value;
        renderDetails();
      });
      document.getElementById("detail-phase-filter").addEventListener("change", (event) => {
        filters.phase = event.target.value;
        renderDetails();
      });
      document.getElementById("detail-status-filter").addEventListener("change", (event) => {
        filters.status = event.target.value;
        renderDetails();
      });
      document.getElementById("detail-query").addEventListener("input", (event) => {
        filters.query = event.target.value;
        renderDetails();
      });
      document.getElementById("detail-reset").addEventListener("click", () => {
        filters.model = "";
        filters.phase = "";
        filters.status = "";
        filters.query = "";
        document.getElementById("detail-model-filter").value = "";
        document.getElementById("detail-phase-filter").value = "";
        document.getElementById("detail-status-filter").value = "";
          document.getElementById("detail-query").value = "";
          renderDetails();
        });
      }

      function bindErrorFilters() {
        document.getElementById("error-model-filter").addEventListener("change", (event) => {
          filters.errorModel = event.target.value;
          state.selectedErrorSignature = "";
          state.selectedErrorEventKey = "";
          renderErrorAnalysis();
        });
        document.getElementById("error-category-filter").addEventListener("change", (event) => {
          filters.errorCategory = event.target.value;
          state.selectedErrorSignature = "";
          state.selectedErrorEventKey = "";
          renderErrorAnalysis();
        });
        document.getElementById("error-query").addEventListener("input", (event) => {
          filters.errorQuery = event.target.value;
          state.selectedErrorSignature = "";
          state.selectedErrorEventKey = "";
          renderErrorAnalysis();
        });
        document.getElementById("error-reset").addEventListener("click", () => {
          filters.errorModel = "";
          filters.errorCategory = "";
          filters.errorQuery = "";
          document.getElementById("error-model-filter").value = "";
          document.getElementById("error-category-filter").value = "";
          document.getElementById("error-query").value = "";
          state.selectedErrorSignature = "";
          state.selectedErrorEventKey = "";
          renderErrorAnalysis();
        });
      }

    function bindCompareControls() {
      document.getElementById("compare-left-model").addEventListener("change", (event) => {
        state.compare.leftModel = event.target.value;
        renderCompare();
      });
      document.getElementById("compare-right-model").addEventListener("change", (event) => {
        state.compare.rightModel = event.target.value;
        renderCompare();
      });
      document.getElementById("compare-swap").addEventListener("click", () => {
        const currentLeft = state.compare.leftModel;
        state.compare.leftModel = state.compare.rightModel;
        state.compare.rightModel = currentLeft;
        renderCompare();
      });
    }

      function renderAll() {
        renderPromptSelector();
        setHeader();
        renderStats();
        renderCompare();
        buildDetailFilters();
        renderLeaderboard();
        renderColdWarm();
        renderStability();
        renderErrorAnalysis();
        renderDetails();
      }

      function applyHistoryData(rawData, source, baseUrl = null) {
        reportData = buildReportPayload(rawData);
        sourceLabel = source;
        historyBaseUrl = baseUrl;
        state.selectedPrompt = "";
        syncPromptSelection();
        refreshCurrentView();
        state.leaderboardSelectedModels = [];
        state.leaderboardFilterTouched = false;
        state.selectedErrorSignature = "";
        state.selectedErrorEventKey = "";
        state.selectedRecordKey = "";
        state.rawLogCache = {};
        renderAll();
      }

    async function fetchHistory(url) {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    }

      async function loadHistoryFromDefaultSource() {
        const params = new URLSearchParams(window.location.search);
        const url = params.get("history") || DEFAULT_HISTORY_URL;
        sourceLabel = url;
        setLoadState(`loading ${url} ...`);
        try {
          const raw = await fetchHistory(url);
          applyHistoryData(raw, url, new URL(url, window.location.href).toString());
          setLoadState(`loaded ${url}`);
        } catch (error) {
          reportData = emptyPayload([]);
        renderAll();
        const reason = error && error.message ? error.message : String(error);
        setLoadState(`自動読込に失敗しました (${reason})。file:// で開いている場合は Open history.json を使ってください。`, true);
      }
    }

    function bindFileLoader() {
      const input = document.getElementById("history-file");
      document.getElementById("open-history").addEventListener("click", () => input.click());
        input.addEventListener("change", async (event) => {
          const file = event.target.files && event.target.files[0];
          if (!file) return;
          setLoadState(`loading ${file.name} ...`);
          try {
            const raw = JSON.parse(await file.text());
            applyHistoryData(raw, file.name, null);
            setLoadState(`loaded ${file.name}`);
          } catch (error) {
            const reason = error && error.message ? error.message : String(error);
          setLoadState(`history.json の読込に失敗しました (${reason})`, true);
        } finally {
          input.value = "";
        }
      });
    }

    document.getElementById("reload").addEventListener("click", () => {
      loadHistoryFromDefaultSource();
    });

      bindTabs();
      bindPromptControls();
      bindCompareControls();
      bindDetailFilters();
      bindErrorFilters();
      bindSorters("#leaderboard-table", state.leaderboardSort, renderLeaderboard);
      bindSorters("#coldwarm-table", state.coldwarmSort, renderColdWarm);
      bindSorters("#stability-table", state.stabilitySort, renderStability);
      bindSorters("#detail-table", state.detailSort, renderDetails);
      bindFileLoader();
      renderAll();
      showTab("leaderboard");
      loadHistoryFromDefaultSource();
  </script>
</body>
</html>
"""
    return template.replace("__DEFAULT_HISTORY_URL__", json.dumps(history_url, ensure_ascii=False))


def write_report_html(output_path: Path, history_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_url = os.path.relpath(history_path, output_path.parent).replace(os.sep, "/")
    output_path.write_text(render_report_html(history_url), encoding="utf-8")


def ensure_report_html(output_path: Path, history_path: Path, *, force: bool = False) -> bool:
    history_url = os.path.relpath(history_path, output_path.parent).replace(os.sep, "/")
    rendered = render_report_html(history_url)
    if not force and output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
        if existing == rendered:
            return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return True
