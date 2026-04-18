from __future__ import annotations

import json
import unittest

from local_llm_bench.lmstudio_api import (
    CONTINUATION_PROMPT,
    LMStudioAPIError,
    _non_stream_payload_to_result,
    consume_sse_stream,
    stream_chat_completion,
)


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status: int = 200) -> None:
        self._lines = [line.encode("utf-8") for line in lines]
        self.status = status

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class LMStudioAPITests(unittest.TestCase):
    def test_consume_sse_stream_collects_metrics_with_usage(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            "\n",
            'data: {"choices":[{"delta":{"content":"print("}}]}\n',
            "\n",
            'data: {"choices":[{"delta":{"content":"42)"}}]}\n',
            "\n",
            'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]
        timestamps = iter([0.05, 0.20, 0.50, 0.70, 0.80])
        result = consume_sse_stream(lines, started_at=0.0, now_fn=lambda: next(timestamps))

        self.assertEqual(result.response_text, "print(42)")
        self.assertEqual(result.reasoning_text, "")
        self.assertAlmostEqual(result.ttft_ms or 0.0, 200.0, places=4)
        self.assertAlmostEqual(result.total_latency_ms, 800.0, places=4)
        self.assertAlmostEqual(result.completion_window_ms or 0.0, 600.0, places=4)
        self.assertEqual(result.prompt_tokens, 20)
        self.assertEqual(result.initial_prompt_tokens, 20)
        self.assertAlmostEqual(result.initial_prompt_latency_ms or 0.0, 200.0, places=4)
        self.assertAlmostEqual(result.initial_prompt_tps or 0.0, 100.0, places=3)
        self.assertEqual(result.conversation_prompt_tokens, 20)
        self.assertAlmostEqual(result.conversation_prompt_latency_ms or 0.0, 800.0, places=4)
        self.assertAlmostEqual(result.conversation_prompt_tps or 0.0, 25.0, places=3)
        self.assertEqual(result.completion_tokens, 10)
        self.assertEqual(result.total_tokens, 30)
        self.assertAlmostEqual(result.decode_tps or 0.0, 16.6666666, places=3)
        self.assertAlmostEqual(result.end_to_end_tps or 0.0, 12.5, places=3)
        self.assertAlmostEqual(result.approx_prompt_tps or 0.0, 100.0, places=3)
        self.assertEqual(result.finish_reason, "stop")

    def test_consume_sse_stream_handles_missing_usage(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
            "\n",
            'data: {"choices":[{"finish_reason":"stop"}]}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]
        timestamps = iter([0.10, 0.25, 0.40])
        result = consume_sse_stream(lines, started_at=0.0, now_fn=lambda: next(timestamps))

        self.assertEqual(result.response_text, "hello")
        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)
        self.assertIsNone(result.decode_tps)
        self.assertIsNone(result.end_to_end_tps)
        self.assertIsNone(result.approx_prompt_tps)
        self.assertEqual(result.reasoning_text, "")

    def test_consume_sse_stream_accepts_reasoning_only_output(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"reasoning":"step 1"}}]}\n',
            "\n",
            'data: {"choices":[{"delta":{"reasoning":" -> step 2"},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"completion_tokens":12,"total_tokens":30}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]
        timestamps = iter([0.10, 0.30, 0.50])
        result = consume_sse_stream(lines, started_at=0.0, now_fn=lambda: next(timestamps))

        self.assertEqual(result.response_text, "step 1 -> step 2")
        self.assertEqual(result.reasoning_text, "step 1 -> step 2")
        self.assertAlmostEqual(result.ttft_ms or 0.0, 100.0, places=4)
        self.assertEqual(result.completion_tokens, 12)
        self.assertEqual(result.finish_reason, "stop")

    def test_empty_stream_raises_error(self) -> None:
        lines = ["data: [DONE]\n", "\n"]
        with self.assertRaises(LMStudioAPIError):
            consume_sse_stream(lines, started_at=0.0, now_fn=lambda: 0.10)

    def test_non_stream_payload_accepts_reasoning_content_and_top_level_output(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": [{"text": "step 1"}],
                    },
                    "finish_reason": "stop",
                }
            ],
            "output_text": [{"type": "output_text", "text": "final answer"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        }

        result = _non_stream_payload_to_result(payload, started_at=1.0, ended_at=1.4)

        self.assertEqual(result.reasoning_text, "step 1")
        self.assertEqual(result.response_text, "final answer")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.completion_tokens, 7)

    def test_stream_chat_completion_continues_until_remaining_budget(self) -> None:
        requests: list[dict[str, object]] = []
        responses = [
            _FakeStreamResponse(
                [
                    'data: {"choices":[{"delta":{"content":"alpha"}}]}\n',
                    "\n",
                    'data: {"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":12,"completion_tokens":6,"total_tokens":18}}\n',
                    "\n",
                    "data: [DONE]\n",
                    "\n",
                ]
            ),
            _FakeStreamResponse(
                [
                    'data: {"choices":[{"delta":{"content":" beta"}}]}\n',
                    "\n",
                    'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":4,"total_tokens":20}}\n',
                    "\n",
                    "data: [DONE]\n",
                    "\n",
                ]
            ),
        ]

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 30.0)
            requests.append(json.loads(request.data.decode("utf-8")))
            return responses.pop(0)

        timestamps = iter([0.0, 0.10, 0.30, 0.40, 1.0, 1.05, 1.20, 1.30])
        result = stream_chat_completion(
            api_base="http://localhost:1234/v1",
            model="bench-model",
            prompt_text="RC4を書いて",
            temperature=0.0,
            max_tokens=10,
            timeout_sec=30.0,
            now_fn=lambda: next(timestamps),
            urlopen=fake_urlopen,
        )

        self.assertEqual(result.response_text, "alpha beta")
        self.assertEqual(result.reasoning_text, "")
        self.assertAlmostEqual(result.ttft_ms or 0.0, 100.0, places=4)
        self.assertAlmostEqual(result.total_latency_ms, 700.0, places=4)
        self.assertAlmostEqual(result.completion_window_ms or 0.0, 600.0, places=4)
        self.assertEqual(result.prompt_tokens, 12)
        self.assertEqual(result.initial_prompt_tokens, 12)
        self.assertAlmostEqual(result.initial_prompt_latency_ms or 0.0, 100.0, places=4)
        self.assertAlmostEqual(result.initial_prompt_tps or 0.0, 120.0, places=3)
        self.assertEqual(result.conversation_prompt_tokens, 28)
        self.assertAlmostEqual(result.conversation_prompt_latency_ms or 0.0, 700.0, places=4)
        self.assertAlmostEqual(result.conversation_prompt_tps or 0.0, 40.0, places=3)
        self.assertEqual(result.completion_tokens, 10)
        self.assertEqual(result.total_tokens, 22)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(requests[0]["max_tokens"], 10)
        self.assertEqual(requests[1]["max_tokens"], 4)
        self.assertEqual(
            requests[1]["messages"],
            [
                {"role": "user", "content": "RC4を書いて"},
                {"role": "assistant", "content": "alpha"},
                {"role": "user", "content": CONTINUATION_PROMPT},
            ],
        )
