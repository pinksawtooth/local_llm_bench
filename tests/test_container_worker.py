from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from local_llm_bench.config import AuthSettings
from local_llm_bench.docker_task import container_worker


class _FakeSessionStack:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup (1 sub-exception)",
            [RuntimeError("session close failed")],
        )


class _FakeToolList:
    tools: list[object] = []


class _AsyncNullContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeToolSession:
    async def call_tool(self, tool_name: str, arguments: dict[str, object] | None = None):
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=f"{tool_name} ok")])


class ContainerWorkerTests(unittest.IsolatedAsyncioTestCase):
    def test_drop_none_values_removes_nested_nulls(self) -> None:
        sanitized = container_worker._drop_none_values(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "demo",
                            "arguments": "{}",
                            "description": None,
                        },
                        "metadata": None,
                    }
                ],
                "unused": None,
            }
        )
        self.assertNotIn("unused", sanitized)
        self.assertNotIn("metadata", sanitized["tool_calls"][0])
        self.assertNotIn("description", sanitized["tool_calls"][0]["function"])

    def test_assistant_message_payload_never_emits_null_content(self) -> None:
        payload = container_worker._assistant_message_payload(
            assistant_text="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "demo", "arguments": "{}"},
                }
            ],
        )
        self.assertEqual(payload["role"], "assistant")
        self.assertEqual(payload["content"], "")
        self.assertIn("tool_calls", payload)
        self.assertNotIn(None, payload.values())

    async def test_run_question_preserves_primary_error_when_cleanup_raises(self) -> None:
        async def fake_open_mcp_stdio_session(**_: object):
            return _FakeSessionStack(), object(), _FakeToolList()

        payload = {
            "api_base": "http://127.0.0.1:1/v1",
            "model": "dummy-model",
            "temperature": 0.0,
            "max_tokens": 128,
            "system_prompt": "system",
            "task_prompt": "task",
            "question": {
                "id": "q1",
                "prompt": "task",
                "answer_type": "exact",
                "binary_path": None,
            },
        }

        with (
            patch(
                "local_llm_bench.docker_task.container_worker.resolve_native_binary_target",
                return_value=types.SimpleNamespace(path=None),
            ),
            patch(
                "local_llm_bench.docker_task.container_worker._open_mcp_stdio_session",
                side_effect=fake_open_mcp_stdio_session,
            ),
            patch(
                "local_llm_bench.docker_task.container_worker._chat_completion",
                side_effect=RuntimeError("URLError: connection refused"),
            ),
        ):
            result = await container_worker._run_question(payload)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "URLError: connection refused")
        self.assertIn("trace", result)
        self.assertIn("cleanup_error", result["trace"])
        self.assertIn("session close failed", result["trace"]["cleanup_error"])

    async def test_run_question_keeps_success_when_cleanup_raises(self) -> None:
        async def fake_open_mcp_stdio_session(**_: object):
            return _FakeSessionStack(), object(), _FakeToolList()

        payload = {
            "api_base": "http://127.0.0.1:1/v1",
            "model": "dummy-model",
            "temperature": 0.0,
            "max_tokens": 128,
            "system_prompt": "system",
            "task_prompt": "task",
            "question": {
                "id": "q1",
                "prompt": "task",
                "answer_type": "exact",
                "binary_path": None,
            },
        }
        response_payload = {
            "choices": [
                {
                    "message": {"content": "FINAL_ANSWER: flag{ok}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with (
            patch(
                "local_llm_bench.docker_task.container_worker.resolve_native_binary_target",
                return_value=types.SimpleNamespace(path=None),
            ),
            patch(
                "local_llm_bench.docker_task.container_worker._open_mcp_stdio_session",
                side_effect=fake_open_mcp_stdio_session,
            ),
            patch(
                "local_llm_bench.docker_task.container_worker._chat_completion",
                return_value=(response_payload, 12.5),
            ),
        ):
            result = await container_worker._run_question(payload)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["predicted_answer"], "flag{ok}")
        self.assertIn("trace", result)
        self.assertIn("cleanup_error", result["trace"])
        self.assertIn("session close failed", result["trace"]["cleanup_error"])

    def test_error_payload_formats_exception_group(self) -> None:
        payload = container_worker._error_payload(
            "error",
            ExceptionGroup(
                "unhandled errors in a TaskGroup (1 sub-exception)",
                [RuntimeError("inner boom")],
            ),
        )
        self.assertEqual(payload["error"], "RuntimeError: inner boom")

    async def test_run_question_uses_total_prompt_latency_for_prompt_speed(self) -> None:
        tool_spec = {
            "type": "function",
            "function": {
                "name": "demo_tool",
                "description": "demo",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        async def fake_open_mcp_stdio_session(**_: object):
            return _AsyncNullContext(), _FakeToolSession(), types.SimpleNamespace(tools=[tool_spec])

        payload = {
            "api_base": "http://127.0.0.1:1/v1",
            "model": "dummy-model",
            "temperature": 0.0,
            "max_tokens": 128,
            "system_prompt": "system",
            "task_prompt": "task",
            "question": {
                "id": "q1",
                "prompt": "task",
                "answer_type": "exact",
                "binary_path": None,
            },
        }
        response_payloads = [
            (
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {"name": "demo_tool", "arguments": "{}"},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                },
                10.0,
            ),
            (
                {
                    "choices": [
                        {
                            "message": {"content": "FINAL_ANSWER: flag{ok}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 200,
                        "completion_tokens": 20,
                        "total_tokens": 220,
                    },
                },
                40.0,
            ),
        ]

        with (
            patch(
                "local_llm_bench.docker_task.container_worker.resolve_native_binary_target",
                return_value=types.SimpleNamespace(path=None),
            ),
            patch(
                "local_llm_bench.docker_task.container_worker._open_mcp_stdio_session",
                side_effect=fake_open_mcp_stdio_session,
            ),
            patch(
                "local_llm_bench.docker_task.container_worker._tool_specs_to_openai",
                return_value=([tool_spec], []),
            ),
            patch(
                "local_llm_bench.docker_task.container_worker._chat_completion",
                side_effect=response_payloads,
            ),
        ):
            result = await container_worker._run_question(payload)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["prompt_tokens"], 300)
        self.assertEqual(result["prompt_latency_ms"], 50.0)
        self.assertAlmostEqual(result["approx_prompt_tps"], 6000.0)
        self.assertEqual(result["initial_prompt_tokens"], 100)
        self.assertEqual(result["initial_prompt_latency_ms"], 10.0)
        self.assertAlmostEqual(result["initial_prompt_tps"], 10000.0)
        self.assertEqual(result["conversation_prompt_tokens"], 300)
        self.assertEqual(result["conversation_prompt_latency_ms"], 50.0)
        self.assertAlmostEqual(result["conversation_prompt_tps"], 6000.0)

    async def test_run_question_uses_unsloth_auth_session_for_unsloth_provider(self) -> None:
        async def fake_open_mcp_stdio_session(**_: object):
            return _AsyncNullContext(), object(), _FakeToolList()

        payload = {
            "provider": "unsloth_studio",
            "api_base": "http://host.docker.internal:8888/v1",
            "model": "dummy-model",
            "temperature": 0.0,
            "max_tokens": 128,
            "system_prompt": "system",
            "task_prompt": "task",
            "question": {
                "id": "q1",
                "prompt": "task",
                "answer_type": "exact",
                "binary_path": None,
            },
        }
        auth_session = types.SimpleNamespace(urlopen=object())
        seen: dict[str, object] = {}

        def fake_chat_completion(*, api_base, body, timeout_sec, urlopen):
            seen["api_base"] = api_base
            seen["urlopen"] = urlopen
            return (
                {
                    "choices": [
                        {
                            "message": {"content": "FINAL_ANSWER: flag{ok}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
                12.5,
            )

        with (
            patch(
                "local_llm_bench.docker_task.container_worker.resolve_native_binary_target",
                return_value=types.SimpleNamespace(path=None),
            ),
            patch(
                "local_llm_bench.docker_task.container_worker._open_mcp_stdio_session",
                side_effect=fake_open_mcp_stdio_session,
            ),
            patch(
                "local_llm_bench.docker_task.container_worker.load_unsloth_auth_from_env",
                return_value=AuthSettings(bearer_token="token"),
            ),
            patch(
                "local_llm_bench.docker_task.container_worker.UnslothStudioAuthSession",
                return_value=auth_session,
            ),
            patch(
                "local_llm_bench.docker_task.container_worker._chat_completion",
                side_effect=fake_chat_completion,
            ),
        ):
            result = await container_worker._run_question(payload)

        self.assertEqual(result["status"], "success")
        self.assertEqual(seen["api_base"], "http://host.docker.internal:8888/v1")
        self.assertIs(seen["urlopen"], auth_session.urlopen)
