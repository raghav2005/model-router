from __future__ import annotations

import hashlib
import hmac
import json
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .types import RouteDecision, RoutingRequest


class DecisionAuditLogger:
    """Append-only decision audit log that does not retain prompt content."""

    def __init__(self, path: str | Path, *, hmac_key: bytes | None = None) -> None:
        self.path = Path(path)
        self.hmac_key = hmac_key
        self._lock = threading.Lock()

    def _prompt_fingerprint(self, prompt: str) -> str | None:
        if self.hmac_key is None:
            return None
        return hmac.new(
            self.hmac_key,
            prompt.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def log_decision(
        self,
        request: RoutingRequest,
        decision: RouteDecision,
        *,
        request_id: str | None = None,
        catalog_sha256: str | None = None,
        shadow: bool = False,
    ) -> str:
        actual_request_id = request_id or request.request_id or str(uuid.uuid4())
        event = {
            "schema_version": "model-router-audit-v1",
            "event": "route_decision",
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": actual_request_id,
            "tenant_id": request.tenant_id,
            "prompt_hmac_sha256": self._prompt_fingerprint(request.prompt),
            "prompt_stored": False,
            "shadow": shadow,
            "policy_version": decision.policy_version,
            "catalog_sha256": catalog_sha256,
            "selected_model": decision.model_id,
            "fallback_models": list(decision.fallback_models),
            "classifier_source": decision.features.classifier_source,
            "classifier_model_version": decision.features.classifier_model_version,
            "classifier_confidence": decision.features.classifier_confidence,
            "complexity_level": decision.features.complexity_level,
            "use_case": decision.features.use_case,
            "risk": decision.features.risk,
            "estimated_cost_usd": decision.estimated_cost_usd,
        }
        self._append(event)
        return actual_request_id

    def log_execution(
        self,
        *,
        request_id: str,
        model_id: str,
        success: bool,
        latency_ms: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        response_model: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self._append(
            {
                "schema_version": "model-router-audit-v1",
                "event": "execution",
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "selected_model": model_id,
                "response_model": response_model,
                "success": success,
                "latency_ms": round(latency_ms, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error_type": error_type,
            }
        )

    def _append(self, event: dict[str, object]) -> None:
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            if str(self.path) == "-":
                sys.stdout.write(encoded)
                sys.stdout.flush()
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
