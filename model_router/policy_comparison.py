from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .catalog import catalog_sha256, load_catalog
from .classifier import DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD
from .learned import DecisionPolicy, default_artifact_path
from .live_eval import EvaluationCase, case_set_sha256, load_cases
from .messages import flatten_messages
from .router import POLICY_VERSION, ModelRouter
from .types import Priority, RoutingRequest
from .workload_evidence import (
    apply_workload_evidence,
    sha256_file,
    validate_workload_evidence,
)

SCHEMA_VERSION = "model-router-live-policy-comparison-v1"


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_results(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON result at line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"Result at line {line_number} must be an object")
            rows.append(value)
    if not rows:
        raise ValueError("live result file is empty")
    return rows


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return round(ordered[index], 3)


def _wilson_95(successes: int, total: int) -> dict[str, float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    observed = successes / total
    denominator = 1 + z * z / total
    centre = (observed + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(observed * (1 - observed) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "lower": round(max(0.0, centre - margin), 6),
        "upper": round(min(1.0, centre + margin), 6),
    }


def _validated_pass(run: Mapping[str, Any]) -> bool:
    return run.get("success") is True and run.get("score") == 1.0


def _arm_summary(
    observations: Sequence[tuple[Mapping[str, Any], bool]],
) -> dict[str, object]:
    successes = sum(run.get("success") is True for run, _ in observations)
    validated = [(run, expected) for run, expected in observations if expected]
    passed = sum(_validated_pass(run) for run, _ in validated)
    costs = [
        float(run["estimated_cost_usd"])
        for run, _ in observations
        if run.get("estimated_cost_usd") is not None
    ]
    latencies = [
        float(run["latency_ms"])
        for run, _ in observations
        if run.get("latency_ms") is not None
    ]
    return {
        "observations": len(observations),
        "successful_calls": successes,
        "call_success_rate": round(successes / len(observations), 6),
        "validated_observations": len(validated),
        "validator_passes": passed,
        "validator_pass_rate": (
            round(passed / len(validated), 6) if validated else None
        ),
        "validator_pass_rate_wilson_95": _wilson_95(passed, len(validated)),
        "cost_observations": len(costs),
        "total_cost_usd": (
            round(sum(costs), 10) if len(costs) == len(observations) else None
        ),
        "latency_ms": {
            "p50": (round(statistics.median(latencies), 3) if latencies else None),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
    }


def _comparison(
    router_arm: Mapping[str, Any], capable_arm: Mapping[str, Any]
) -> dict[str, float | None]:
    router_quality = router_arm.get("validator_pass_rate")
    capable_quality = capable_arm.get("validator_pass_rate")
    router_cost = router_arm.get("total_cost_usd")
    capable_cost = capable_arm.get("total_cost_usd")
    router_latency = router_arm.get("latency_ms", {})
    capable_latency = capable_arm.get("latency_ms", {})
    router_p95 = (
        router_latency.get("p95") if isinstance(router_latency, Mapping) else None
    )
    capable_p95 = (
        capable_latency.get("p95") if isinstance(capable_latency, Mapping) else None
    )
    return {
        "validator_pass_rate_delta": (
            round(float(router_quality) - float(capable_quality), 6)
            if router_quality is not None and capable_quality is not None
            else None
        ),
        "quality_retention": (
            round(float(router_quality) / float(capable_quality), 6)
            if router_quality is not None
            and capable_quality is not None
            and float(capable_quality) > 0
            else None
        ),
        "cost_savings_rate": (
            round(1 - float(router_cost) / float(capable_cost), 6)
            if router_cost is not None
            and capable_cost is not None
            and float(capable_cost) > 0
            else None
        ),
        "p95_latency_reduction_rate": (
            round(1 - float(router_p95) / float(capable_p95), 6)
            if router_p95 is not None
            and capable_p95 is not None
            and float(capable_p95) > 0
            else None
        ),
    }


def _routing_request(case: EvaluationCase, priority: str) -> RoutingRequest:
    use_case = case.metadata.get("use_case")
    supported_use_case = (
        str(use_case) if use_case in {"general_qa", "coding", "reasoning"} else None
    )
    return RoutingRequest(
        prompt=flatten_messages(list(case.messages)),
        expected_output_tokens=case.max_tokens,
        priority=cast(Priority, priority),
        use_case=supported_use_case,
    )


def build_policy_comparison(
    cases_path: str | Path,
    results_path: str | Path,
    live_summary_path: str | Path,
    *,
    workload_evidence_path: str | Path = "reports/workload_evidence.json",
    artifact_path: str | Path = default_artifact_path(),
    catalog_path: str | Path | None = None,
    priority: str = "balanced",
    decision_policy: str = "adaptive",
    underroute_tolerance: float = 0.15,
    adaptive_confidence_threshold: float = DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD,
) -> dict[str, object]:
    if priority not in {"balanced", "cost", "quality", "latency"}:
        raise ValueError(f"unsupported priority: {priority!r}")
    if decision_policy not in {
        "argmax",
        "expected",
        "conservative",
        "tier_risk",
        "adaptive",
    }:
        raise ValueError(f"unsupported decision policy: {decision_policy!r}")
    cases = load_cases(cases_path)
    summary = _read_json(live_summary_path)
    if summary.get("schema_version") != "switchyard-live-eval-summary-v3":
        raise ValueError("unsupported live-evaluation summary schema")
    provenance = summary.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("live-evaluation provenance is missing")
    if provenance.get("case_set_sha256") != case_set_sha256(cases):
        raise ValueError("case set does not match the live summary")
    if provenance.get("catalog_sha256") != catalog_sha256(catalog_path):
        raise ValueError("catalogue does not match the live summary")
    if provenance.get("results_sha256") != sha256_file(results_path):
        raise ValueError("result file does not match the live summary")

    workload_evidence = _read_json(workload_evidence_path)
    workload_valid, workload_detail = validate_workload_evidence(
        workload_evidence,
        live_summary_path=live_summary_path,
        catalog_path=catalog_path,
    )
    if not workload_valid:
        raise ValueError(f"invalid workload evidence: {workload_detail}")
    models = apply_workload_evidence(load_catalog(catalog_path), workload_evidence)
    targets = tuple(str(value) for value in provenance.get("targets", []))
    expected_targets = {model.switchyard_target for model in models}
    if set(targets) != expected_targets or len(targets) != len(expected_targets):
        raise ValueError(
            "live benchmark must cover every catalogue target exactly once"
        )
    repetitions = int(provenance.get("repetitions", 0))
    if repetitions <= 0:
        raise ValueError("live summary has an invalid repetition count")

    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in _read_results(results_path):
        key = (
            str(row.get("case_id", "")),
            str(row.get("target", "")),
            int(row.get("repetition", 0)),
        )
        if key in indexed:
            raise ValueError(f"duplicate live result: {key}")
        indexed[key] = row
    expected_keys = {
        (case.id, target, repetition)
        for case in cases
        for target in targets
        for repetition in range(1, repetitions + 1)
    }
    if set(indexed) != expected_keys:
        missing = len(expected_keys - set(indexed))
        unexpected = len(set(indexed) - expected_keys)
        raise ValueError(
            f"live result matrix is incomplete: missing={missing}, unexpected={unexpected}"
        )

    router = ModelRouter.from_artifact(
        artifact_path,
        models=models,
        classifier_mode="hybrid",
        decision_policy=cast(DecisionPolicy, decision_policy),
        underroute_tolerance=underroute_tolerance,
        adaptive_confidence_threshold=adaptive_confidence_threshold,
    )
    roles_by_target = {model.switchyard_target: model.id for model in models}
    targets_by_role = {model.id: model.switchyard_target for model in models}
    capable_role = max(models, key=lambda model: model.tier).id
    arm_observations: dict[str, list[tuple[Mapping[str, Any], bool]]] = {
        model.id: [] for model in models
    }
    router_observations: list[tuple[Mapping[str, Any], bool]] = []
    routing_distribution: Counter[str] = Counter()
    avoidable_failures = 0
    validated_observations = 0
    cheapest_passing_matches = 0
    observations_with_known_passing_cost = 0
    cost_regret = 0.0

    for case in cases:
        decision = router.route(_routing_request(case, priority))
        selected_target = targets_by_role[decision.model_id]
        routing_distribution[decision.model_id] += 1
        expects_validation = bool(case.validators)
        for repetition in range(1, repetitions + 1):
            runs = {
                target: indexed[(case.id, target, repetition)] for target in targets
            }
            for target, run in runs.items():
                arm_observations[roles_by_target[target]].append(
                    (run, expects_validation)
                )
            selected_run = runs[selected_target]
            router_observations.append((selected_run, expects_validation))
            if not expects_validation:
                continue
            validated_observations += 1
            passing = [run for run in runs.values() if _validated_pass(run)]
            if not _validated_pass(selected_run) and passing:
                avoidable_failures += 1
            priced_passing = [
                run for run in passing if run.get("estimated_cost_usd") is not None
            ]
            if priced_passing and len(priced_passing) == len(passing):
                observations_with_known_passing_cost += 1
                cheapest_cost = min(
                    float(run["estimated_cost_usd"]) for run in priced_passing
                )
                if (
                    _validated_pass(selected_run)
                    and selected_run.get("estimated_cost_usd") is not None
                ):
                    selected_cost = float(selected_run["estimated_cost_usd"])
                    if math.isclose(
                        selected_cost, cheapest_cost, rel_tol=1e-9, abs_tol=1e-12
                    ):
                        cheapest_passing_matches += 1
                    cost_regret += max(0.0, selected_cost - cheapest_cost)

    arms = {role: _arm_summary(values) for role, values in arm_observations.items()}
    router_arm = _arm_summary(router_observations)
    comparison = _comparison(router_arm, arms[capable_role])
    complete = (
        router_arm["validator_pass_rate"] is not None
        and router_arm["total_cost_usd"] is not None
        and comparison["quality_retention"] is not None
        and comparison["cost_savings_rate"] is not None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "complete": complete,
        "source": {
            "case_set_sha256": provenance.get("case_set_sha256"),
            "results_sha256": provenance.get("results_sha256"),
            "live_summary_sha256": sha256_file(live_summary_path),
            "catalog_sha256": catalog_sha256(catalog_path),
            "artifact_sha256": sha256_file(artifact_path),
            "workload_evidence_sha256": sha256_file(workload_evidence_path),
            "benchmark_fingerprint": provenance.get("benchmark_fingerprint"),
        },
        "policy": {
            "version": POLICY_VERSION,
            "classifier_mode": "hybrid",
            "decision_policy": decision_policy,
            "underroute_tolerance": underroute_tolerance,
            "adaptive_confidence_threshold": adaptive_confidence_threshold,
            "priority": priority,
        },
        "coverage": {
            "unique_cases": len(cases),
            "repetitions": repetitions,
            "paired_observations": len(router_observations),
            "validated_observations": validated_observations,
        },
        "routing_distribution": dict(sorted(routing_distribution.items())),
        "arms": {"router_policy": router_arm, **arms},
        "baseline_role": capable_role,
        "comparison_vs_capable": comparison,
        "oracle_diagnostics": {
            "avoidable_validator_failures": avoidable_failures,
            "avoidable_validator_failure_rate": (
                round(avoidable_failures / validated_observations, 6)
                if validated_observations
                else None
            ),
            "observations_with_known_passing_cost": (
                observations_with_known_passing_cost
            ),
            "cheapest_passing_selection_rate": (
                round(
                    cheapest_passing_matches / observations_with_known_passing_cost,
                    6,
                )
                if observations_with_known_passing_cost
                else None
            ),
            "total_cost_regret_usd": round(cost_regret, 10),
        },
        "privacy": {
            "contains_prompt_content": False,
            "contains_response_content": False,
            "contains_aggregate_measurements_only": True,
        },
        "caveat": (
            "This is a paired replay over responses already collected from every role. "
            "It measures this router policy on the approved case mix; it does not prove "
            "that the benchmark represents future production traffic."
        ),
    }


def validate_policy_comparison(
    report: Mapping[str, Any],
    *,
    live_summary_path: str | Path,
    workload_evidence_path: str | Path = "reports/workload_evidence.json",
    artifact_path: str | Path = default_artifact_path(),
    catalog_path: str | Path | None = None,
) -> tuple[bool, str]:
    if report.get("schema_version") != SCHEMA_VERSION:
        return False, "unsupported policy-comparison schema"
    if report.get("complete") is not True:
        return False, "policy comparison is incomplete"
    source = report.get("source", {})
    policy = report.get("policy", {})
    privacy = report.get("privacy", {})
    if not all(isinstance(value, Mapping) for value in (source, policy, privacy)):
        return False, "policy comparison is malformed"
    summary = _read_json(live_summary_path)
    provenance = summary.get("provenance", {})
    if not isinstance(provenance, Mapping):
        return False, "live summary provenance is missing"
    workload_evidence = _read_json(workload_evidence_path)
    workload_valid, workload_detail = validate_workload_evidence(
        workload_evidence,
        live_summary_path=live_summary_path,
        catalog_path=catalog_path,
    )
    if not workload_valid:
        return False, f"invalid workload evidence: {workload_detail}"
    expected = {
        "live_summary_sha256": sha256_file(live_summary_path),
        "catalog_sha256": catalog_sha256(catalog_path),
        "artifact_sha256": sha256_file(artifact_path),
        "workload_evidence_sha256": sha256_file(workload_evidence_path),
        "results_sha256": provenance.get("results_sha256"),
        "case_set_sha256": provenance.get("case_set_sha256"),
        "benchmark_fingerprint": provenance.get("benchmark_fingerprint"),
    }
    if not all(isinstance(value, str) and value for value in expected.values()):
        return False, "live summary is missing immutable comparison provenance"
    if any(source.get(name) != value for name, value in expected.items()):
        return False, "policy comparison does not match its release inputs"
    if policy.get("version") != POLICY_VERSION:
        return False, "policy comparison uses a different routing policy"
    if (
        privacy.get("contains_prompt_content") is not False
        or privacy.get("contains_response_content") is not False
        or privacy.get("contains_aggregate_measurements_only") is not True
    ):
        return False, "policy comparison is not marked aggregate-only"
    return True, "policy comparison matches the live summary, artifact, and catalogue"


def write_policy_comparison(path: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
