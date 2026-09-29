"""Native Inspect compression contract; fixtures only, no runtime requests."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from local_llm_bench.inspect_log import completed_log_info, storage_metadata


class InspectLogStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "圧縮 fixture.eval"

    def fixture(self):
        from inspect_ai.log import EvalLog, EvalSpec, EvalDataset, EvalConfig, EvalSample, write_eval_log
        from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ModelOutput, ModelUsage
        from inspect_ai.scorer import Score

        content = "保存する会話と測定結果。Fixture text, not inference.\n" * 500
        log = EvalLog(status="success", eval=EvalSpec(
            created="2026-09-15T00:00:00Z", task="compression_fixture",
            dataset=EvalDataset(samples=1, sample_ids=["q1"]), config=EvalConfig(), model="fixture"),
            samples=[EvalSample(id="q1", epoch=1, input=content, target="",
                messages=[ChatMessageUser(content=content), ChatMessageAssistant(content=content)],
                output=ModelOutput.from_content("fixture", content),
                model_usage={"fixture": ModelUsage(input_tokens=11, input_tokens_cache_read=3,
                    output_tokens=7, total_tokens=21, reasoning_tokens=2)},
                scores={"host_typed": Score(value=1, answer="42")},
                metadata={"metrics": {"prefill_tps": None, "decode_tps": 12.5}})])
        write_eval_log(log, self.path, format="eval")
        return log

    def info(self):
        return completed_log_info(SimpleNamespace(status="success", location=str(self.path)), self.root)

    def test_native_zstd_roundtrip_preserves_content_usage_scores_and_unknowns(self):
        from inspect_ai.log import read_eval_log
        expected = self.fixture()
        before = self.path.read_bytes()
        info = self.info()
        storage = info["log_storage"]
        self.assertEqual(storage["format"], "eval")
        self.assertEqual(storage["compression"], ["zstd"])
        self.assertEqual(storage["file_bytes"], len(before))
        # A compressible fixture proves compression, not a production ratio.
        self.assertLess(storage["file_bytes"], storage["uncompressed_bytes"] / 4)
        self.assertEqual(read_eval_log(self.path, resolve_attachments=True).model_dump(), expected.model_dump())
        self.assertEqual(self.path.read_bytes(), before)  # no second rewrite
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_capacity_inspection_does_not_decompress_members(self):
        self.fixture()
        with patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("Inflated log body")):
            self.assertEqual(storage_metadata(self.path)["compression"], ["zstd"])

    def test_missing_corrupt_or_uncompressed_logs_are_not_claimed_compressed(self):
        for content in (None, b"incomplete log", b"stored"):
            with self.subTest(content=content):
                self.path.unlink(missing_ok=True)
                if content == b"stored":
                    with zipfile.ZipFile(self.path, "w") as archive:
                        archive.writestr("header.json", "{}")
                elif content is not None:
                    self.path.write_bytes(content)
                original = self.path.read_bytes() if self.path.exists() else None
                info = self.info()
                self.assertEqual(info["status"], "success")  # model outcome is independent
                self.assertEqual(info["log_storage"]["status"], "unavailable")
                self.assertNotIn("compression", info["log_storage"])
                self.assertEqual(self.path.read_bytes() if self.path.exists() else None, original)

    def test_unreadable_storage_does_not_replace_or_hide_model_result(self):
        self.fixture()
        before = self.path.read_bytes()
        with patch("local_llm_bench.inspect_log.storage_metadata", side_effect=PermissionError("private text")):
            info = self.info()
        self.assertEqual(info["log_storage"]["status"], "unavailable")
        self.assertNotIn("private text", str(info))
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
