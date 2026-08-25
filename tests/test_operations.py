from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from model_router.audit import DecisionAuditLogger
from model_router.catalog import catalog_sha256
from model_router.observability import RoutingMetrics
from model_router.router import ModelRouter, NoEligibleModel
from model_router.switchyard import (
    CircuitBreaker,
    CircuitOpenError,
    RetryPolicy,
    SwitchyardClient,
    SwitchyardError,
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


def service_unavailable() -> HTTPError:
    return HTTPError(
        "http://127.0.0.1:4000/v1/models",
        503,
        "Service Unavailable",
        {},
        io.BytesIO(b"busy"),
    )


class ResilienceTests(unittest.TestCase):
    @patch("model_router.switchyard.time.sleep")
    @patch("model_router.switchyard.urlopen")
    def test_get_retries_transient_failure(
        self, mocked_urlopen: object, _mocked_sleep: object
    ) -> None:
        mocked_urlopen.side_effect = [  # type: ignore[attr-defined]
            service_unavailable(),
            FakeHTTPResponse({"data": []}),
        ]
        client = SwitchyardClient(
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0)
        )
        self.assertEqual(client.list_models(), {"data": []})
        self.assertEqual(mocked_urlopen.call_count, 2)  # type: ignore[attr-defined]

    @patch("model_router.switchyard.urlopen")
    def test_post_does_not_retry_without_explicit_opt_in(
        self, mocked_urlopen: object
    ) -> None:
        mocked_urlopen.side_effect = service_unavailable()  # type: ignore[attr-defined]
        client = SwitchyardClient(retry_policy=RetryPolicy(max_attempts=3))
        with self.assertRaises(SwitchyardError):
            client.chat_completions(
                model="efficient",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
            )
        self.assertEqual(mocked_urlopen.call_count, 1)  # type: ignore[attr-defined]

    def test_circuit_breaker_recovers_with_probe(self) -> None:
        now = [0.0]
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_seconds=10,
            clock=lambda: now[0],
        )
        breaker.record_failure()
        with self.assertRaises(CircuitOpenError):
            breaker.before_call()
        now[0] = 11.0
        breaker.before_call()
        breaker.record_success()
        self.assertEqual(breaker.snapshot()["state"], "closed")


class OperationsTests(unittest.TestCase):
    def test_audit_log_uses_hmac_and_does_not_store_prompt(self) -> None:
        request = RoutingRequest(
            "private customer prompt", tenant_id="tenant-a", expected_output_tokens=20
        )
        decision = ModelRouter().route(request)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            logger = DecisionAuditLogger(path, hmac_key=b"test-key")
            logger.log_decision(
                request,
                decision,
                request_id="request-1",
                catalog_sha256=catalog_sha256(),
            )
            content = path.read_text(encoding="utf-8")
            event = json.loads(content)
        self.assertNotIn(request.prompt, content)
        self.assertEqual(event["request_id"], "request-1")
        self.assertEqual(len(event["prompt_hmac_sha256"]), 64)

    def test_metrics_export_is_prometheus_compatible_text(self) -> None:
        decision = ModelRouter.from_artifact().route(
            RoutingRequest(
                "User: Explain an API.\n"
                "Assistant: Here is the overview.\n"
                "User: Compare two authentication options and recommend one.",
                expected_output_tokens=20,
            )
        )
        metrics = RoutingMetrics()
        metrics.record_decision(decision)
        metrics.record_execution(
            decision.model_id,
            success=True,
            latency_ms=123.0,
            input_tokens=10,
            output_tokens=5,
        )
        rendered = metrics.prometheus()
        self.assertIn("model_router_decisions_total", rendered)
        self.assertIn("model_router_execution_latency_ms_bucket", rendered)
        self.assertIn("model_router_classifier_entropy_bucket", rendered)
        self.assertIn(
            'model_router_classifier_view_tier_decisions_total{disagrees="true"} 1',
            rendered,
        )

    def test_request_provider_allowlist_is_a_hard_gate(self) -> None:
        with self.assertRaises(NoEligibleModel):
            ModelRouter().route(
                RoutingRequest("hello", allowed_providers=frozenset({"other"}))
            )


if __name__ == "__main__":
    unittest.main()
