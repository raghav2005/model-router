from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from model_router.switchyard import (
    SwitchyardClient,
    SwitchyardExecutor,
    UnsafeDelegatedRoute,
)
from model_router.types import RoutingRequest


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeStreamingResponse:
    def __enter__(self) -> FakeStreamingResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):  # type: ignore[no-untyped-def]
        events = [
            {
                "id": "chatcmpl-stream",
                "model": "efficient",
                "choices": [{"delta": {"content": "To"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-stream",
                "model": "efficient",
                "choices": [{"delta": {"content": "kyo"}, "finish_reason": "stop"}],
            },
            {
                "id": "chatcmpl-stream",
                "model": "efficient",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        ]
        for event in events:
            yield f"data: {json.dumps(event)}\n".encode()
        yield b"data: [DONE]\n"


class SwitchyardClientTests(unittest.TestCase):
    @patch("model_router.switchyard.urlopen")
    def test_chat_completion_uses_openai_compatible_endpoint(
        self, mocked_urlopen: object
    ) -> None:
        mocked_urlopen.return_value = FakeHTTPResponse(  # type: ignore[attr-defined]
            {
                "id": "chatcmpl-test",
                "model": "efficient",
                "choices": [{"message": {"role": "assistant", "content": "4"}}],
            }
        )
        client = SwitchyardClient(
            "http://127.0.0.1:4000", api_key="test-token", timeout_seconds=5
        )

        result = client.chat_completions(
            model="efficient",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=20,
        )

        self.assertEqual(result["id"], "chatcmpl-test")
        request = mocked_urlopen.call_args.args[0]  # type: ignore[attr-defined]
        self.assertEqual(request.full_url, "http://127.0.0.1:4000/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "efficient")
        self.assertEqual(payload["max_tokens"], 20)

    @patch("model_router.switchyard.urlopen")
    def test_list_models_calls_switchyard(self, mocked_urlopen: object) -> None:
        mocked_urlopen.return_value = FakeHTTPResponse(  # type: ignore[attr-defined]
            {"object": "list", "data": [{"id": "smart-general"}]}
        )
        result = SwitchyardClient().list_models()
        self.assertEqual(result["data"][0]["id"], "smart-general")
        request = mocked_urlopen.call_args.args[0]  # type: ignore[attr-defined]
        self.assertEqual(request.get_method(), "GET")
        self.assertTrue(request.full_url.endswith("/v1/models"))

    @patch("model_router.switchyard.urlopen")
    def test_streaming_completion_records_ttft_and_usage(
        self, mocked_urlopen: object
    ) -> None:
        mocked_urlopen.return_value = FakeStreamingResponse()  # type: ignore[attr-defined]
        result = SwitchyardClient().chat_completions_stream(
            model="efficient",
            messages=[{"role": "user", "content": "Capital of Japan?"}],
            max_tokens=20,
        )
        self.assertEqual(result["choices"][0]["message"]["content"], "Tokyo")
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertGreaterEqual(result["_router_timing"]["ttft_ms"], 0)
        request = mocked_urlopen.call_args.args[0]  # type: ignore[attr-defined]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertTrue(payload["stream"])
        self.assertTrue(payload["stream_options"]["include_usage"])

    def test_client_rejects_protected_extra_body_fields(self) -> None:
        client = SwitchyardClient()
        with self.assertRaises(ValueError):
            client.chat_completions(
                model="efficient",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=20,
                extra_body={"model": "capable"},
            )


class SwitchyardExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = SwitchyardExecutor(SwitchyardClient())

    def test_policy_mode_maps_decision_to_direct_target(self) -> None:
        plan = self.executor.plan(
            RoutingRequest("What is the capital of Japan?", expected_output_tokens=40)
        )
        self.assertEqual(plan.routing_mode, "policy")
        self.assertEqual(plan.policy_decision.model_id, "efficient")
        self.assertEqual(plan.switchyard_model, "efficient")

    def test_delegated_profile_is_blocked_if_one_target_breaks_budget(self) -> None:
        with self.assertRaises(UnsafeDelegatedRoute):
            self.executor.plan(
                RoutingRequest(
                    "What is the capital of Japan?",
                    expected_output_tokens=100,
                    max_cost_usd=0.001,
                ),
                routing_mode="switchyard-general",
            )

    def test_coding_profile_rejects_general_question(self) -> None:
        with self.assertRaises(UnsafeDelegatedRoute):
            self.executor.plan(
                RoutingRequest("What is the capital of Japan?"),
                routing_mode="switchyard-coding",
            )

    def test_stage_profile_requires_tool_result_history(self) -> None:
        with self.assertRaises(UnsafeDelegatedRoute):
            self.executor.plan(
                RoutingRequest(
                    "Write a Python hello world function.", expected_output_tokens=100
                ),
                routing_mode="switchyard-stage",
            )

    def test_stage_profile_accepts_eligible_coding_turn_with_tool_history(
        self,
    ) -> None:
        messages = [
            {"role": "user", "content": "Write a Python hello world function."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "No existing file.",
            },
        ]
        plan = self.executor.plan(
            RoutingRequest(
                "Write a Python hello world function.", expected_output_tokens=100
            ),
            routing_mode="switchyard-stage",
            messages=messages,
        )
        self.assertEqual(plan.switchyard_model, "smart-agent-stage")


if __name__ == "__main__":
    unittest.main()
