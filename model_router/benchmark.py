from __future__ import annotations

import json
import statistics
import time
from importlib.resources import files
from pathlib import Path

from .catalog import load_catalog
from .router import ModelRouter
from .types import RoutingRequest


def load_cases(path: str | Path | None = None) -> list[dict[str, object]]:
    case_path = (
        Path(path) if path else files("model_router").joinpath("benchmark_cases.json")
    )
    with case_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_benchmark(
    iterations: int = 300, router: ModelRouter | None = None
) -> dict[str, object]:
    """Compare policies with a hand-labeled, offline task-fit proxy benchmark."""
    models = load_catalog()
    model_by_id = {model.id: model for model in models}
    cases = load_cases()
    active_router = router or ModelRouter(models)

    strategies: dict[str, list[str]] = {
        "always_efficient": ["efficient"] * len(cases),
        "always_balanced": ["balanced"] * len(cases),
        "always_capable": ["capable"] * len(cases),
        "router_balanced": [],
    }

    timings_us: list[float] = []
    for case in cases:
        request = RoutingRequest(
            prompt=str(case["prompt"]),
            expected_output_tokens=int(case["output_tokens"]),
        )
        start = time.perf_counter_ns()
        decision = active_router.route(request)
        timings_us.append((time.perf_counter_ns() - start) / 1_000)
        strategies["router_balanced"].append(decision.model_id)

    # Repeat only for stable router-overhead timing; no network/model calls occur.
    for _ in range(max(0, iterations - 1)):
        for case in cases:
            request = RoutingRequest(
                prompt=str(case["prompt"]),
                expected_output_tokens=int(case["output_tokens"]),
            )
            start = time.perf_counter_ns()
            active_router.route(request)
            timings_us.append((time.perf_counter_ns() - start) / 1_000)

    results: dict[str, object] = {
        "methodology": "Offline proxy: pass means selected model tier >= hand-labeled minimum tier. Prices, quality, and latency are illustrative catalog inputs.",
        "case_count": len(cases),
        "strategies": {},
        "router_overhead_us": {
            "median": round(statistics.median(timings_us), 2),
            "p95": round(sorted(timings_us)[int(len(timings_us) * 0.95) - 1], 2),
        },
    }
    strategy_results = results["strategies"]
    assert isinstance(strategy_results, dict)

    for strategy, selections in strategies.items():
        passed = 0
        total_cost = 0.0
        total_latency = 0
        overqualified = 0
        counts: dict[str, int] = {}
        for case, model_id in zip(cases, selections, strict=True):
            model = model_by_id[model_id]
            input_tokens = max(1, (len(str(case["prompt"])) + 3) // 4)
            total_cost += model.estimate_cost(input_tokens, int(case["output_tokens"]))
            total_latency += model.latency_p95_ms
            minimum = int(case["min_tier"])
            passed += model.tier >= minimum
            overqualified += model.tier > minimum
            counts[model_id] = counts.get(model_id, 0) + 1
        strategy_results[strategy] = {
            "proxy_pass_rate": round(passed / len(cases), 4),
            "total_estimated_cost_usd": round(total_cost, 6),
            "average_latency_prior_ms": round(total_latency / len(cases), 1),
            "overqualified_rate": round(overqualified / len(cases), 4),
            "selection_counts": counts,
        }

    capable_cost = strategy_results["always_capable"]["total_estimated_cost_usd"]
    for metrics in strategy_results.values():
        metrics["cost_saving_vs_always_capable"] = round(
            1.0 - metrics["total_estimated_cost_usd"] / capable_cost, 4
        )
    return results
