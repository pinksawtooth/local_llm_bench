from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_llm_bench.report import ensure_report_html


class ReportTests(unittest.TestCase):
    def test_ensure_report_html_creates_reuses_and_refreshes_stale_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "docs" / "index.html"
            history_path = root / "runs" / "history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text("[]", encoding="utf-8")

            created = ensure_report_html(report_path, history_path)
            original_text = report_path.read_text(encoding="utf-8")
            reused = ensure_report_html(report_path, history_path)
            report_path.write_text("sentinel", encoding="utf-8")
            refreshed = ensure_report_html(report_path, history_path)
            forced = ensure_report_html(report_path, history_path, force=True)
            refreshed_text = report_path.read_text(encoding="utf-8")

        self.assertTrue(created)
        self.assertFalse(reused)
        self.assertTrue(refreshed)
        self.assertTrue(forced)
        self.assertIn("../runs/history.json", original_text)
        self.assertIn("comparisonModelKey", original_text)
        self.assertIn("Benchmark ID", original_text)
        self.assertIn('data-tab="compare"', original_text)
        self.assertIn('id="compare-left-model"', original_text)
        self.assertIn('id="leaderboard-model-filter"', original_text)
        self.assertIn('id="leaderboard-model-options"', original_text)
        self.assertIn('id="prompt-filter"', original_text)
        self.assertIn("ALL_PROMPTS", original_text)
        self.assertIn("renderPromptSelector", original_text)
        self.assertIn("leaderboardSelectedModels", original_text)
        self.assertIn("inferModelInfoFromModelName", original_text)
        self.assertIn("extractQuantizationToken", original_text)
        self.assertIn("looksLikeFileSystemPath", original_text)
        self.assertIn("Warm Score", original_text)
        self.assertIn("Init Prompt Speed", original_text)
        self.assertIn("Conv Prompt Speed", original_text)
        self.assertIn("Initial Prompt Speed", original_text)
        self.assertIn("Conversation Prompt Speed", original_text)
        self.assertIn("currentHasBenchmarkMetrics", original_text)
        self.assertIn('data-benchmark-column="true"', original_text)
        self.assertIn('id="leaderboard-note"', original_text)
        self.assertIn("同一 prompt の実測 token 数", original_text)
        self.assertIn("Benchmark Error", original_text)
        self.assertIn('data-sort="benchmark_correct_rate"', original_text)
        self.assertIn("Correct は cold + warm を通した全体正答率です", original_text)
        self.assertIn("Error Analysis", original_text)
        self.assertIn('id="errors-panel"', original_text)
        self.assertIn('id="error-signature-table"', original_text)
        self.assertIn("renderErrorAnalysis", original_text)
        self.assertIn("resolveLogUrl", original_text)
        self.assertIn("Tool Calls", original_text)
        self.assertIn("summarizeCountMap", original_text)
        self.assertIn('data-sort="tool_call_count"', original_text)
        self.assertIn('class="details-layout"', original_text)
        self.assertIn("選択中の run 詳細", original_text)
        self.assertNotEqual(refreshed_text, "sentinel")
