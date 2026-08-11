from __future__ import annotations

import io
import json
import unittest
from typing import Any

from model_router.api import RouterApplication, RuntimeConfig
from model_router.router import ModelRouter


class FakeClient:
    def __init__(self) -> None:
        self.called_model: str | None = None

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def chat_completions(self, **kwargs: Any) -> dict[str, Any]:
        self.called_model = kwargs["model"]
        return {
            "id": "response-1",
            "model": kwargs["model"],
            "choices": [{"message": {"role": "assistant", "content": "Tokyo"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }


def invoke(
    app: RouterApplication,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    token: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    encoded = json.dumps(body).encode("utf-8") if body is not None else b""
    environ: dict[str, Any] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(encoded)),
        "wsgi.input": io.BytesIO(encoded),
    }
    if token:
        environ["HTTP_AUTHORIZATION"] = f"Bearer {token}"
    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = dict(headers)

    response = b"".join(app(environ, start_response))
    return int(captured["status"].split()[0]), captured["headers"], response


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeClient()
        self.app = RouterApplication(
            router=ModelRouter(),
            client=self.client,  # type: ignore[arg-type]
            config=RuntimeConfig(mode="shadow", api_token="secret"),
        )

    def test_health_does_not_require_authentication(self) -> None:
        status, _, body = invoke(self.app, "GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_route_requires_authentication_and_returns_provenance(self) -> None:
        request = {
            "messages": [{"role": "user", "content": "What is Japan's capital?"}],
            "max_tokens": 20,
        }
        status, _, _ = invoke(self.app, "POST", "/v1/route", request)
        self.assertEqual(status, 401)
        status, headers, body = invoke(
            self.app, "POST", "/v1/route", request, token="secret"
        )
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["model"], "efficient")
        self.assertEqual(headers["X-Model-Router-Mode"], "shadow")
        self.assertIn("provider_model", payload["candidates"][0])

    def test_shadow_mode_executes_baseline_and_reports_proposal(self) -> None:
        status, headers, body = invoke(
            self.app,
            "POST",
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "user", "content": "What is the capital of Japan?"}
                ],
                "max_tokens": 20,
            },
            token="secret",
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["id"], "response-1")
        self.assertEqual(self.client.called_model, "capable")
        self.assertEqual(headers["X-Model-Router-Proposed-Role"], "efficient")

    def test_metrics_are_exposed(self) -> None:
        status, _, body = invoke(self.app, "GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn(b"model_router_decisions_total", body)


if __name__ == "__main__":
    unittest.main()
