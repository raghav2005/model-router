from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict, deque
from typing import Iterable

from .types import RouteDecision


LATENCY_BUCKETS_MS = (10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000)


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class RoutingMetrics:
    """Dependency-free, bounded-cardinality metrics for the router service."""

    def __init__(self, latency_sample_limit: int = 10_000) -> None:
        if latency_sample_limit < 1:
            raise ValueError("latency_sample_limit must be positive")
        self._lock = threading.Lock()
        self._routes: Counter[tuple[str, str, str]] = Counter()
        self._executions: Counter[tuple[str, str]] = Counter()
        self._estimated_cost: Counter[str] = Counter()
        self._actual_tokens: Counter[tuple[str, str]] = Counter()
        self._latencies: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=latency_sample_limit)
        )

    def record_decision(self, decision: RouteDecision) -> None:
        key = (
            decision.model_id,
            decision.features.use_case,
            decision.features.classifier_source,
        )
        with self._lock:
            self._routes[key] += 1
            self._estimated_cost[decision.model_id] += decision.estimated_cost_usd

    def record_execution(
        self,
        model_id: str,
        *,
        success: bool,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        status = "success" if success else "error"
        with self._lock:
            self._executions[(model_id, status)] += 1
            self._latencies[model_id].append(max(0.0, latency_ms))
            self._actual_tokens[(model_id, "input")] += max(0, input_tokens)
            self._actual_tokens[(model_id, "output")] += max(0, output_tokens)

    def prometheus(self) -> str:
        lines = [
            "# HELP model_router_decisions_total Routing decisions by selected role.",
            "# TYPE model_router_decisions_total counter",
        ]
        with self._lock:
            for (model, use_case, source), count in sorted(self._routes.items()):
                lines.append(
                    "model_router_decisions_total"
                    f'{{model="{_label(model)}",use_case="{_label(use_case)}",'
                    f'classifier="{_label(source)}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP model_router_estimated_cost_usd_total Sum of pre-call cost estimates.",
                    "# TYPE model_router_estimated_cost_usd_total counter",
                ]
            )
            for model, cost in sorted(self._estimated_cost.items()):
                lines.append(
                    "model_router_estimated_cost_usd_total"
                    f'{{model="{_label(model)}"}} {cost:.12f}'
                )
            lines.extend(
                [
                    "# HELP model_router_executions_total Gateway executions by outcome.",
                    "# TYPE model_router_executions_total counter",
                ]
            )
            for (model, status), count in sorted(self._executions.items()):
                lines.append(
                    "model_router_executions_total"
                    f'{{model="{_label(model)}",status="{status}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP model_router_tokens_total Provider-reported tokens.",
                    "# TYPE model_router_tokens_total counter",
                ]
            )
            for (model, token_type), count in sorted(self._actual_tokens.items()):
                lines.append(
                    "model_router_tokens_total"
                    f'{{model="{_label(model)}",type="{token_type}"}} {count}'
                )
            lines.extend(self._latency_lines())
        return "\n".join(lines) + "\n"

    def _latency_lines(self) -> Iterable[str]:
        yield "# HELP model_router_execution_latency_ms Gateway completion latency."
        yield "# TYPE model_router_execution_latency_ms histogram"
        for model, samples in sorted(self._latencies.items()):
            values = list(samples)
            for boundary in LATENCY_BUCKETS_MS:
                count = sum(value <= boundary for value in values)
                yield (
                    "model_router_execution_latency_ms_bucket"
                    f'{{model="{_label(model)}",le="{boundary}"}} {count}'
                )
            yield (
                "model_router_execution_latency_ms_bucket"
                f'{{model="{_label(model)}",le="+Inf"}} {len(values)}'
            )
            yield (
                "model_router_execution_latency_ms_sum"
                f'{{model="{_label(model)}"}} {math.fsum(values):.6f}'
            )
            yield (
                "model_router_execution_latency_ms_count"
                f'{{model="{_label(model)}"}} {len(values)}'
            )
