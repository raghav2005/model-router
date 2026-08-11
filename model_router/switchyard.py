from __future__ import annotations

import json
import random
import threading
import time
import uuid
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .catalog import load_catalog
from .router import ModelRouter
from .types import ModelProfile, RouteDecision, RoutingRequest


JSON = dict[str, Any]
Message = Mapping[str, Any]


class SwitchyardError(RuntimeError):
    """Raised when the local Switchyard gateway cannot serve a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class CircuitOpenError(SwitchyardError):
    """Raised while the gateway circuit breaker is open."""


class ResponseValidationError(SwitchyardError):
    """Raised when a model response fails an application validator."""


class UnsafeDelegatedRoute(ValueError):
    """Raised when a Switchyard profile could violate a hard routing constraint."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 2.0
    retry_post_requests: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("retry backoff cannot be negative")


class CircuitBreaker:
    """Small thread-safe breaker around the Switchyard service boundary."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1 or recovery_seconds <= 0:
            raise ValueError("invalid circuit-breaker configuration")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False
        self._lock = threading.Lock()

    def before_call(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            if self._clock() - self._opened_at < self.recovery_seconds:
                raise CircuitOpenError(
                    "Switchyard circuit breaker is open", retryable=True
                )
            if self._probe_in_flight:
                raise CircuitOpenError(
                    "Switchyard circuit breaker recovery probe is in flight",
                    retryable=True,
                )
            self._probe_in_flight = True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._probe_in_flight = False
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = self._clock()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            state = "open" if self._opened_at is not None else "closed"
            return {"state": state, "consecutive_failures": self._failures}


@dataclass(frozen=True)
class DelegatedProfile:
    mode: str
    switchyard_model: str
    candidate_model_ids: tuple[str, ...]
    required_use_case: str | None = None
    requires_tool_history: bool = False


DELEGATED_PROFILES: dict[str, DelegatedProfile] = {
    "switchyard-general": DelegatedProfile(
        mode="switchyard-general",
        switchyard_model="smart-general",
        candidate_model_ids=("efficient", "capable"),
    ),
    "switchyard-coding": DelegatedProfile(
        mode="switchyard-coding",
        switchyard_model="smart-coding",
        candidate_model_ids=("efficient", "capable"),
        required_use_case="coding",
    ),
    "switchyard-stage": DelegatedProfile(
        mode="switchyard-stage",
        switchyard_model="smart-agent-stage",
        candidate_model_ids=("efficient", "capable"),
        required_use_case="coding",
        requires_tool_history=True,
    ),
    "switchyard-random": DelegatedProfile(
        mode="switchyard-random",
        switchyard_model="benchmark-random",
        candidate_model_ids=("efficient", "capable"),
    ),
}


@dataclass(frozen=True)
class ExecutionPlan:
    """A reviewable decision before any request is sent to a model."""

    routing_mode: str
    switchyard_model: str
    policy_decision: RouteDecision
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "routing_mode": self.routing_mode,
            "switchyard_model": self.switchyard_model,
            "reasons": list(self.reasons),
            "policy_decision": self.policy_decision.as_dict(),
        }


@dataclass(frozen=True)
class ExecutionResult:
    plan: ExecutionPlan
    response: JSON
    attempts: tuple[dict[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.as_dict(),
            "response": self.response,
            "attempts": list(self.attempts),
        }


class SwitchyardClient:
    """Small OpenAI-compatible HTTP client for a Switchyard proxy."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4000",
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        user_agent: str = "model-router/0.4",
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Switchyard base_url must be an http(s) URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "dummy"
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.user_agent = user_agent

    @staticmethod
    def _retry_after_seconds(error: HTTPError) -> float | None:
        value = error.headers.get("Retry-After") if error.headers else None
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(
                    0.0,
                    parsedate_to_datetime(value).timestamp() - time.time(),
                )
            except (TypeError, ValueError, OverflowError):
                return None

    def _request(
        self,
        method: str,
        path: str,
        payload: JSON | None = None,
        *,
        request_id: str | None = None,
    ) -> JSON:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        actual_request_id = request_id or str(uuid.uuid4())
        can_retry = method in {"GET", "HEAD"} or self.retry_policy.retry_post_requests
        attempts = self.retry_policy.max_attempts if can_retry else 1
        last_error: SwitchyardError | None = None

        for attempt in range(1, attempts + 1):
            self.circuit_breaker.before_call()
            request = Request(
                f"{self.base_url}{path}",
                data=body,
                method=method,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": self.user_agent,
                    "X-Request-ID": actual_request_id,
                    **(
                        {"Content-Type": "application/json"} if body is not None else {}
                    ),
                },
            )
            retry_after: float | None = None
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                self.circuit_breaker.record_success()
                break
            except HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")
                retryable = error.code in {408, 429, 500, 502, 503, 504}
                last_error = SwitchyardError(
                    f"Switchyard returned HTTP {error.code}: "
                    f"{details or error.reason}",
                    status_code=error.code,
                    retryable=retryable,
                )
                retry_after = self._retry_after_seconds(error)
            except URLError as error:
                last_error = SwitchyardError(
                    f"Could not reach Switchyard at {self.base_url}: {error.reason}",
                    retryable=True,
                )

            assert last_error is not None
            if last_error.retryable:
                self.circuit_breaker.record_failure()
            if not last_error.retryable or attempt == attempts:
                raise last_error
            backoff = min(
                self.retry_policy.max_backoff_seconds,
                self.retry_policy.initial_backoff_seconds * (2 ** (attempt - 1)),
            )
            delay = retry_after if retry_after is not None else backoff
            time.sleep(delay + random.uniform(0.0, max(0.001, delay * 0.25)))
        else:  # pragma: no cover - the loop either breaks or raises
            assert last_error is not None
            raise last_error

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SwitchyardError("Switchyard returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise SwitchyardError("Switchyard returned a non-object JSON response")
        return parsed

    def list_models(self) -> JSON:
        return self._request("GET", "/v1/models")

    def health(self) -> JSON:
        return self._request("GET", "/health")

    def stats(self) -> JSON:
        return self._request("GET", "/v1/stats")

    def chat_completions(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> JSON:
        if not model:
            raise ValueError("model is required")
        if not messages:
            raise ValueError("at least one message is required")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        payload: JSON = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if extra_body:
            protected = {"model", "messages", "max_tokens"}
            overlap = protected.intersection(extra_body)
            if overlap:
                names = ", ".join(sorted(overlap))
                raise ValueError(f"extra_body cannot replace protected fields: {names}")
            payload.update(extra_body)
        return self._request(
            "POST", "/v1/chat/completions", payload, request_id=request_id
        )

    def chat_completions_stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> JSON:
        """Collect one SSE stream while retaining TTFT and total latency."""
        if not model or not messages or max_tokens <= 0:
            raise ValueError("model, messages, and positive max_tokens are required")
        payload: JSON = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if extra_body:
            protected = {
                "model",
                "messages",
                "max_tokens",
                "stream",
                "stream_options",
            }
            overlap = protected.intersection(extra_body)
            if overlap:
                names = ", ".join(sorted(overlap))
                raise ValueError(f"extra_body cannot replace protected fields: {names}")
            payload.update(extra_body)

        actual_request_id = request_id or str(uuid.uuid4())
        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
                "X-Request-ID": actual_request_id,
            },
        )
        started = time.perf_counter()
        first_token_at: float | None = None
        text_parts: list[str] = []
        response_model = model
        response_id: str | None = None
        finish_reason: str | None = None
        usage: JSON = {}
        self.circuit_breaker.before_call()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as error:
                        raise SwitchyardError(
                            "Switchyard returned invalid streaming JSON"
                        ) from error
                    if not isinstance(event, dict):
                        continue
                    response_id = str(event.get("id", response_id or "")) or response_id
                    response_model = str(event.get("model", response_model))
                    event_usage = event.get("usage")
                    if isinstance(event_usage, dict):
                        usage = event_usage
                    choices = event.get("choices", [])
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        text_parts.append(content)
            self.circuit_breaker.record_success()
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            retryable = error.code in {408, 429, 500, 502, 503, 504}
            if retryable:
                self.circuit_breaker.record_failure()
            raise SwitchyardError(
                f"Switchyard returned HTTP {error.code}: {details or error.reason}",
                status_code=error.code,
                retryable=retryable,
            ) from error
        except URLError as error:
            self.circuit_breaker.record_failure()
            raise SwitchyardError(
                f"Could not reach Switchyard at {self.base_url}: {error.reason}",
                retryable=True,
            ) from error

        completed = time.perf_counter()
        if first_token_at is None:
            first_token_at = completed
        return {
            "id": response_id or actual_request_id,
            "model": response_model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "".join(text_parts)},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
            "_router_timing": {
                "ttft_ms": (first_token_at - started) * 1_000,
                "total_ms": (completed - started) * 1_000,
            },
        }


class SwitchyardExecutor:
    """
    Combines the proposal's policy router with Switchyard's execution profiles.

    `policy` mode sends directly to the target selected by the custom router.
    Delegated modes let Switchyard choose between efficient and capable targets,
    but only after the custom layer confirms that both targets pass every hard
    request constraint.
    """

    def __init__(
        self,
        client: SwitchyardClient,
        *,
        router: ModelRouter | None = None,
        models: list[ModelProfile] | None = None,
    ) -> None:
        catalog = models or load_catalog()
        self.models = {model.id: model for model in catalog}
        self.router = router or ModelRouter(catalog)
        self.client = client

    @staticmethod
    def _has_tool_result(messages: Sequence[Message]) -> bool:
        return any(message.get("role") == "tool" for message in messages)

    def plan(
        self,
        request: RoutingRequest,
        *,
        routing_mode: str = "policy",
        messages: Sequence[Message] | None = None,
    ) -> ExecutionPlan:
        decision = self.router.route(request)

        if routing_mode == "policy":
            selected = self.models[decision.model_id]
            return ExecutionPlan(
                routing_mode=routing_mode,
                switchyard_model=selected.switchyard_target,
                policy_decision=decision,
                reasons=(
                    "custom policy applied all hard gates and utility scoring",
                    f"Switchyard will execute direct target {selected.switchyard_target}",
                ),
            )

        profile = DELEGATED_PROFILES.get(routing_mode)
        if profile is None:
            supported = ", ".join(["policy", *DELEGATED_PROFILES])
            raise ValueError(
                f"Unknown routing mode {routing_mode!r}; expected one of: {supported}"
            )

        if (
            profile.required_use_case
            and decision.features.use_case != profile.required_use_case
        ):
            raise UnsafeDelegatedRoute(
                f"{routing_mode} is intended for {profile.required_use_case} requests; "
                f"this request was classified as {decision.features.use_case}"
            )

        actual_messages = messages or ({"role": "user", "content": request.prompt},)
        if profile.requires_tool_history and not self._has_tool_result(actual_messages):
            raise UnsafeDelegatedRoute(
                "switchyard-stage requires tool-result history; use switchyard-coding "
                "for a first-turn coding request"
            )

        candidate_scores = {
            candidate.model_id: candidate for candidate in decision.candidates
        }
        unsafe: list[str] = []
        for model_id in profile.candidate_model_ids:
            candidate = candidate_scores.get(model_id)
            if candidate is None:
                unsafe.append(f"{model_id}: missing from policy catalog")
            elif not candidate.eligible:
                unsafe.append(f"{model_id}: {', '.join(candidate.rejection_reasons)}")
        if unsafe:
            raise UnsafeDelegatedRoute(
                "Switchyard profile could select a target rejected by a hard gate: "
                + "; ".join(unsafe)
            )

        return ExecutionPlan(
            routing_mode=routing_mode,
            switchyard_model=profile.switchyard_model,
            policy_decision=decision,
            reasons=(
                "custom policy confirmed every profile target is eligible",
                f"Switchyard profile {profile.switchyard_model} will make the final tier choice",
            ),
        )

    def execute(
        self,
        request: RoutingRequest,
        *,
        routing_mode: str = "policy",
        messages: Sequence[Message] | None = None,
        temperature: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        request_id: str | None = None,
        fallback_on_error: bool = False,
        response_validator: Callable[[JSON], bool] | None = None,
    ) -> ExecutionResult:
        actual_messages = messages or ({"role": "user", "content": request.prompt},)
        plan = self.plan(
            request,
            routing_mode=routing_mode,
            messages=actual_messages,
        )
        target_names = [plan.switchyard_model]
        if fallback_on_error and routing_mode == "policy":
            target_names.extend(
                self.models[model_id].switchyard_target
                for model_id in plan.policy_decision.fallback_models
            )
        attempts: list[dict[str, object]] = []
        last_error: SwitchyardError | None = None
        for target in target_names:
            started = time.perf_counter()
            try:
                response = self.client.chat_completions(
                    model=target,
                    messages=actual_messages,
                    max_tokens=request.expected_output_tokens,
                    temperature=temperature,
                    extra_body=extra_body,
                    request_id=request_id,
                )
                if response_validator is not None and not response_validator(response):
                    raise ResponseValidationError(
                        "model response failed application validation"
                    )
                attempts.append(
                    {
                        "target": target,
                        "success": True,
                        "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
                    }
                )
                return ExecutionResult(
                    plan=plan, response=response, attempts=tuple(attempts)
                )
            except SwitchyardError as error:
                last_error = error
                attempts.append(
                    {
                        "target": target,
                        "success": False,
                        "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
                        "error": type(error).__name__,
                        "retryable": error.retryable,
                    }
                )
                if not fallback_on_error:
                    raise
        assert last_error is not None
        raise SwitchyardError(
            f"all eligible Switchyard targets failed after {len(attempts)} attempts",
            status_code=last_error.status_code,
            retryable=last_error.retryable,
        ) from last_error
