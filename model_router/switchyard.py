from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
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


class UnsafeDelegatedRoute(ValueError):
    """Raised when a Switchyard profile could violate a hard routing constraint."""


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

    def as_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.as_dict(),
            "response": self.response,
        }


class SwitchyardClient:
    """Small OpenAI-compatible HTTP client for a Switchyard proxy."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4000",
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Switchyard base_url must be an http(s) URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "dummy"
        self.timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str, payload: JSON | None = None) -> JSON:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise SwitchyardError(
                f"Switchyard returned HTTP {error.code}: {details or error.reason}"
            ) from error
        except URLError as error:
            raise SwitchyardError(
                f"Could not reach Switchyard at {self.base_url}: {error.reason}"
            ) from error

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SwitchyardError("Switchyard returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise SwitchyardError("Switchyard returned a non-object JSON response")
        return parsed

    def list_models(self) -> JSON:
        return self._request("GET", "/v1/models")

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
        return self._request("POST", "/v1/chat/completions", payload)


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
    ) -> ExecutionResult:
        actual_messages = messages or ({"role": "user", "content": request.prompt},)
        plan = self.plan(
            request,
            routing_mode=routing_mode,
            messages=actual_messages,
        )
        response = self.client.chat_completions(
            model=plan.switchyard_model,
            messages=actual_messages,
            max_tokens=request.expected_output_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        return ExecutionResult(plan=plan, response=response)
