from __future__ import annotations

import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable, Iterable, Mapping
from wsgiref.simple_server import make_server

from .audit import DecisionAuditLogger
from .catalog import catalog_sha256, load_catalog
from .messages import flatten_messages, validate_string_array
from .observability import RoutingMetrics
from .readiness import evaluate_release_gates
from .router import ModelRouter, NoEligibleModel
from .switchyard import SwitchyardClient, SwitchyardError, SwitchyardExecutor
from .types import RoutingRequest

StartResponse = Callable[[str, list[tuple[str, str]]], None]


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = "shadow"
    shadow_target: str = "capable"
    api_token: str | None = None
    metrics_token: str | None = None
    max_body_bytes: int = 1_048_576
    fallback_on_error: bool = False
    release_policy_path: str = "config/release_policy.json"
    training_report_path: str = "reports/complexity_router_v2.json"
    live_summary_path: str = "reports/live_eval_summary.json"

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "enforce"}:
            raise ValueError("mode must be shadow or enforce")
        if self.max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")

    @classmethod
    def from_environment(cls) -> RuntimeConfig:
        return cls(
            mode=os.getenv("MODEL_ROUTER_MODE", "shadow"),
            shadow_target=os.getenv("MODEL_ROUTER_SHADOW_TARGET", "capable"),
            api_token=os.getenv("MODEL_ROUTER_API_TOKEN"),
            metrics_token=os.getenv("MODEL_ROUTER_METRICS_TOKEN"),
            max_body_bytes=int(os.getenv("MODEL_ROUTER_MAX_BODY_BYTES", "1048576")),
            fallback_on_error=os.getenv("MODEL_ROUTER_FALLBACK_ON_ERROR", "0") == "1",
            release_policy_path=os.getenv(
                "MODEL_ROUTER_RELEASE_POLICY", "config/release_policy.json"
            ),
            training_report_path=os.getenv(
                "MODEL_ROUTER_TRAINING_REPORT", "reports/complexity_router_v2.json"
            ),
            live_summary_path=os.getenv(
                "MODEL_ROUTER_LIVE_SUMMARY", "reports/live_eval_summary.json"
            ),
        )


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _usage(response: Mapping[str, Any]) -> tuple[int, int]:
    usage = response.get("usage", {})
    if not isinstance(usage, Mapping):
        return 0, 0
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


class RouterApplication:
    """Minimal OpenAI-compatible WSGI edge around the routing core."""

    def __init__(
        self,
        *,
        router: ModelRouter | None = None,
        client: SwitchyardClient | None = None,
        metrics: RoutingMetrics | None = None,
        audit_logger: DecisionAuditLogger | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.config = config or RuntimeConfig.from_environment()
        self.release_report: dict[str, object] | None = None
        if self.config.mode == "enforce":
            try:
                self.release_report = evaluate_release_gates(
                    policy_path=self.config.release_policy_path,
                    training_report_path=self.config.training_report_path,
                    live_summary_path=self.config.live_summary_path,
                )
            except (OSError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "enforcement mode refused: release evidence could not be evaluated"
                ) from error
            if not self.release_report["ready_for_enforcement"]:
                failures = ", ".join(
                    str(gate["name"])
                    for gate in self.release_report["gates"]
                    if not gate["passed"]
                )
                raise RuntimeError(
                    f"enforcement mode refused: failed release gates: {failures}"
                )
        self.models = load_catalog()
        self.model_by_id = {model.id: model for model in self.models}
        if self.config.shadow_target not in self.model_by_id:
            raise ValueError("shadow_target must be a catalog role")
        self.router = router or ModelRouter.from_artifact(
            classifier_mode=os.getenv("MODEL_ROUTER_CLASSIFIER_MODE", "hybrid"),
            decision_policy=os.getenv("MODEL_ROUTER_COMPLEXITY_POLICY", "adaptive"),
            underroute_tolerance=float(
                os.getenv("MODEL_ROUTER_UNDERROUTE_TOLERANCE", "0.15")
            ),
        )
        self.client = client or SwitchyardClient(
            os.getenv("SWITCHYARD_URL", "http://127.0.0.1:4000"),
            api_key=os.getenv("SWITCHYARD_API_KEY"),
            timeout_seconds=float(os.getenv("SWITCHYARD_TIMEOUT_SECONDS", "120")),
        )
        self.executor = SwitchyardExecutor(
            self.client, router=self.router, models=self.models
        )
        self.metrics = metrics or RoutingMetrics()
        self.audit = audit_logger
        self.catalog_digest = catalog_sha256()

    @classmethod
    def from_environment(cls) -> RouterApplication:
        audit_path = os.getenv("MODEL_ROUTER_AUDIT_PATH")
        audit_key = os.getenv("MODEL_ROUTER_AUDIT_HMAC_KEY")
        logger = (
            DecisionAuditLogger(
                audit_path,
                hmac_key=audit_key.encode("utf-8") if audit_key else None,
            )
            if audit_path
            else None
        )
        return cls(audit_logger=logger)

    @staticmethod
    def _bearer(environ: Mapping[str, Any]) -> str | None:
        value = str(environ.get("HTTP_AUTHORIZATION", ""))
        prefix = "Bearer "
        return value[len(prefix) :] if value.startswith(prefix) else None

    def _authorized(self, environ: Mapping[str, Any], *, metrics: bool = False) -> bool:
        expected = self.config.metrics_token if metrics else self.config.api_token
        if expected is None:
            return True
        presented = self._bearer(environ)
        return presented is not None and hmac.compare_digest(presented, expected)

    @staticmethod
    def _respond(
        start_response: StartResponse,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str = "application/json",
        headers: Iterable[tuple[str, str]] = (),
    ) -> list[bytes]:
        response_headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
            ("X-Content-Type-Options", "nosniff"),
            *headers,
        ]
        start_response(f"{status.value} {status.phrase}", response_headers)
        return [body]

    def _read_json(self, environ: Mapping[str, Any]) -> dict[str, Any]:
        try:
            length = int(environ.get("CONTENT_LENGTH") or "0")
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length <= 0 or length > self.config.max_body_bytes:
            raise ValueError("request body is empty or exceeds the configured limit")
        raw = environ["wsgi.input"].read(length)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _routing_request(
        self, body: Mapping[str, Any], environ: Mapping[str, Any]
    ) -> RoutingRequest:
        routing = body.get("routing", {})
        if not isinstance(routing, Mapping):
            raise ValueError("routing must be an object")
        prompt = (
            str(body["prompt"])
            if isinstance(body.get("prompt"), str)
            else flatten_messages(body.get("messages"))
        )
        max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 500)))
        return RoutingRequest(
            prompt=prompt,
            request_id=str(environ.get("HTTP_X_REQUEST_ID") or uuid.uuid4()),
            tenant_id=(
                str(environ["HTTP_X_TENANT_ID"])
                if environ.get("HTTP_X_TENANT_ID")
                else None
            ),
            input_tokens=(
                int(routing["input_tokens"])
                if routing.get("input_tokens") is not None
                else None
            ),
            cached_input_tokens=int(routing.get("cached_input_tokens", 0)),
            cache_write_tokens=int(routing.get("cache_write_tokens", 0)),
            expected_output_tokens=max_tokens,
            required_capabilities=validate_string_array(
                routing.get("required_capabilities", []), "required_capabilities"
            ),
            priority=str(routing.get("priority", "balanced")),
            max_cost_usd=(
                float(routing["max_cost_usd"])
                if routing.get("max_cost_usd") is not None
                else None
            ),
            max_latency_ms=(
                int(routing["max_latency_ms"])
                if routing.get("max_latency_ms") is not None
                else None
            ),
            use_case=(str(routing["use_case"]) if routing.get("use_case") else None),
            allowed_model_ids=validate_string_array(
                routing.get("allowed_model_ids", []), "allowed_model_ids"
            ),
            allowed_providers=validate_string_array(
                routing.get("allowed_providers", []), "allowed_providers"
            ),
        )

    def _record_decision(self, request: RoutingRequest, decision: Any) -> None:
        self.metrics.record_decision(decision)
        if self.audit is not None:
            self.audit.log_decision(
                request,
                decision,
                request_id=request.request_id,
                catalog_sha256=self.catalog_digest,
                shadow=self.config.mode == "shadow",
            )

    def _route(
        self, body: Mapping[str, Any], environ: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        request = self._routing_request(body, environ)
        decision = self.router.route(request)
        self._record_decision(request, decision)
        return request, decision

    def _chat(
        self, body: dict[str, Any], environ: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        request, decision = self._route(body, environ)
        messages = body.get("messages")
        if not isinstance(messages, list):
            messages = [{"role": "user", "content": request.prompt}]
        temperature = (
            float(body["temperature"]) if body.get("temperature") is not None else None
        )
        started = time.perf_counter()
        executed_role = decision.model_id
        try:
            if self.config.mode == "shadow":
                executed_role = self.config.shadow_target
                response = self.client.chat_completions(
                    model=self.model_by_id[executed_role].switchyard_target,
                    messages=messages,
                    max_tokens=request.expected_output_tokens,
                    temperature=temperature,
                    request_id=request.request_id,
                )
            else:
                result = self.executor.execute(
                    request,
                    messages=messages,
                    temperature=temperature,
                    request_id=request.request_id,
                    fallback_on_error=self.config.fallback_on_error,
                )
                response = result.response
            latency_ms = (time.perf_counter() - started) * 1_000
            input_tokens, output_tokens = _usage(response)
            self.metrics.record_execution(
                executed_role,
                success=True,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if self.audit is not None:
                self.audit.log_execution(
                    request_id=request.request_id or "unknown",
                    model_id=executed_role,
                    success=True,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_model=str(response.get("model", "")),
                )
            return response, decision
        except SwitchyardError as error:
            latency_ms = (time.perf_counter() - started) * 1_000
            self.metrics.record_execution(
                executed_role, success=False, latency_ms=latency_ms
            )
            if self.audit is not None:
                self.audit.log_execution(
                    request_id=request.request_id or "unknown",
                    model_id=executed_role,
                    success=False,
                    latency_ms=latency_ms,
                    error_type=type(error).__name__,
                )
            raise

    def __call__(
        self, environ: dict[str, Any], start_response: StartResponse
    ) -> list[bytes]:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        request_id = str(environ.get("HTTP_X_REQUEST_ID") or uuid.uuid4())
        environ["HTTP_X_REQUEST_ID"] = request_id
        common_headers = (("X-Request-ID", request_id),)
        try:
            if method == "GET" and path == "/healthz":
                return self._respond(
                    start_response,
                    HTTPStatus.OK,
                    _json_bytes({"status": "ok", "mode": self.config.mode}),
                    headers=common_headers,
                )
            if method == "GET" and path == "/readyz":
                gateway = self.client.health()
                return self._respond(
                    start_response,
                    HTTPStatus.OK,
                    _json_bytes(
                        {
                            "status": "ready",
                            "mode": self.config.mode,
                            "release_gate_ready": (
                                self.release_report["ready_for_enforcement"]
                                if self.release_report is not None
                                else None
                            ),
                            "switchyard": gateway,
                        }
                    ),
                    headers=common_headers,
                )
            if method == "GET" and path == "/metrics":
                if not self._authorized(environ, metrics=True):
                    return self._respond(
                        start_response,
                        HTTPStatus.UNAUTHORIZED,
                        _json_bytes({"error": "unauthorized"}),
                        headers=common_headers,
                    )
                return self._respond(
                    start_response,
                    HTTPStatus.OK,
                    self.metrics.prometheus().encode("utf-8"),
                    content_type="text/plain; version=0.0.4; charset=utf-8",
                    headers=common_headers,
                )
            if method == "POST" and path in {"/v1/route", "/v1/chat/completions"}:
                if not self._authorized(environ):
                    return self._respond(
                        start_response,
                        HTTPStatus.UNAUTHORIZED,
                        _json_bytes({"error": "unauthorized"}),
                        headers=common_headers,
                    )
                body = self._read_json(environ)
                if path == "/v1/route":
                    _, decision = self._route(body, environ)
                    payload = decision.as_dict()
                else:
                    payload, decision = self._chat(body, environ)
                return self._respond(
                    start_response,
                    HTTPStatus.OK,
                    _json_bytes(payload),
                    headers=(
                        *common_headers,
                        ("X-Model-Router-Proposed-Role", decision.model_id),
                        ("X-Model-Router-Policy", decision.policy_version),
                        ("X-Model-Router-Mode", self.config.mode),
                    ),
                )
            return self._respond(
                start_response,
                HTTPStatus.NOT_FOUND,
                _json_bytes({"error": "not found"}),
                headers=common_headers,
            )
        except (ValueError, json.JSONDecodeError) as error:
            return self._respond(
                start_response,
                HTTPStatus.BAD_REQUEST,
                _json_bytes({"error": str(error)}),
                headers=common_headers,
            )
        except NoEligibleModel as error:
            return self._respond(
                start_response,
                HTTPStatus.UNPROCESSABLE_ENTITY,
                _json_bytes({"error": str(error)}),
                headers=common_headers,
            )
        except SwitchyardError as error:
            return self._respond(
                start_response,
                HTTPStatus.SERVICE_UNAVAILABLE,
                _json_bytes({"error": str(error), "retryable": error.retryable}),
                headers=common_headers,
            )


def create_app() -> RouterApplication:
    return RouterApplication.from_environment()


def main() -> None:
    host = os.getenv("MODEL_ROUTER_HOST", "127.0.0.1")
    port = int(os.getenv("MODEL_ROUTER_PORT", "8080"))
    with make_server(host, port, create_app()) as server:
        print(f"model router listening on http://{host}:{port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
