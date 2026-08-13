from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .catalog import load_catalog, load_catalog_document


@dataclass(frozen=True)
class ReleaseGate:
    name: str
    passed: bool
    detail: str


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _pricing_age_days(as_of: str, today: date) -> int:
    return (today - date.fromisoformat(as_of)).days


def evaluate_release_gates(
    *,
    policy_path: str | Path = "config/release_policy.json",
    catalog_path: str | Path | None = None,
    training_report_path: str | Path = "reports/complexity_router_v2.json",
    live_summary_path: str | Path = "reports/live_eval_summary.json",
    today: date | None = None,
) -> dict[str, object]:
    policy = _read_json(policy_path)
    catalog_document = load_catalog_document(catalog_path)
    models = load_catalog(catalog_path)
    actual_today = today or datetime.now(UTC).date()
    gates: list[ReleaseGate] = []

    deployment_mode = str(policy.get("deployment_mode", "shadow"))
    gates.append(
        ReleaseGate(
            "enforcement explicitly approved",
            deployment_mode == "enforce",
            f"deployment_mode is {deployment_mode!r}",
        )
    )

    maximum_age = int(policy["maximum_pricing_age_days"])
    ages = [_pricing_age_days(model.pricing_as_of, actual_today) for model in models]
    gates.append(
        ReleaseGate(
            "pricing is current and sourced",
            all(age <= maximum_age for age in ages),
            f"maximum pricing age is {max(ages)} days; limit is {maximum_age}",
        )
    )

    require_quality = bool(policy["require_workload_measured_quality"])
    unmeasured_quality = [
        model.id for model in models if model.quality_evidence != "workload_measured"
    ]
    gates.append(
        ReleaseGate(
            "model quality is workload-measured",
            not require_quality or not unmeasured_quality,
            (
                "unmeasured roles: " + ", ".join(unmeasured_quality)
                if unmeasured_quality
                else "all roles have workload measurements"
            ),
        )
    )

    require_latency = bool(policy["require_workload_measured_latency"])
    unmeasured_latency = [
        model.id for model in models if model.latency_evidence != "workload_measured"
    ]
    gates.append(
        ReleaseGate(
            "model latency is workload-measured",
            not require_latency or not unmeasured_latency,
            (
                "unmeasured roles: " + ", ".join(unmeasured_latency)
                if unmeasured_latency
                else "all roles have workload measurements"
            ),
        )
    )

    training_path = Path(training_report_path)
    if training_path.exists():
        training = _read_json(training_path)
        external = training.get("external_context_slice", {})
        metrics = external.get("metrics", {}) if isinstance(external, dict) else {}
        count = int(metrics.get("count", 0))
        accuracy = float(metrics.get("accuracy", 0.0))
        underroute = float(metrics.get("tier_underroute_rate", 1.0))
        minimum_examples = int(policy["minimum_external_examples"])
        minimum_accuracy = float(policy["minimum_external_accuracy"])
        maximum_underroute = float(policy["maximum_external_tier_underroute_rate"])
        external_passed = (
            count >= minimum_examples
            and accuracy >= minimum_accuracy
            and underroute <= maximum_underroute
        )
        detail = (
            f"count={count}/{minimum_examples}, accuracy={accuracy:.3f}/"
            f"{minimum_accuracy:.3f}, tier_underroute={underroute:.3f}/"
            f"{maximum_underroute:.3f} max"
        )
    else:
        external_passed = False
        detail = f"training report not found: {training_path}"
    gates.append(
        ReleaseGate("router generalises to external data", external_passed, detail)
    )

    live_path = Path(live_summary_path)
    if live_path.exists():
        live = _read_json(live_path)
        targets = live.get("targets", {})
        if not isinstance(targets, dict):
            targets = {}
        minimum_cases = int(policy["minimum_live_cases_per_target"])
        minimum_pass = float(policy["minimum_live_all_validators_pass_rate"])
        missing = []
        for model in models:
            result = targets.get(model.switchyard_target, {})
            runs = int(result.get("runs", 0)) if isinstance(result, dict) else 0
            pass_rate = (
                result.get("all_validators_pass_rate")
                if isinstance(result, dict)
                else None
            )
            if (
                runs < minimum_cases
                or pass_rate is None
                or float(pass_rate) < minimum_pass
            ):
                missing.append(model.id)
        live_passed = not missing
        live_detail = (
            "roles below live threshold: " + ", ".join(missing)
            if missing
            else "all roles meet live case and validator thresholds"
        )
    else:
        live_passed = False
        live_detail = f"live evaluation summary not found: {live_path}"
    gates.append(
        ReleaseGate("live response benchmark passes", live_passed, live_detail)
    )

    switchyard = policy.get("switchyard", {})
    switchyard_pin = (
        switchyard.get("tested_version_or_commit")
        if isinstance(switchyard, dict)
        else None
    )
    trusted_fallback = bool(
        switchyard.get("trusted_direct_provider_fallback_configured", False)
        if isinstance(switchyard, dict)
        else False
    )
    gates.append(
        ReleaseGate(
            "Switchyard build is pinned",
            bool(switchyard_pin),
            (
                f"tested pin: {switchyard_pin}"
                if switchyard_pin
                else "no tested Switchyard version or commit is recorded"
            ),
        )
    )
    gates.append(
        ReleaseGate(
            "trusted gateway bypass exists",
            trusted_fallback,
            (
                "direct-provider fallback is configured"
                if trusted_fallback
                else "direct-provider fallback is not configured"
            ),
        )
    )

    return {
        "schema_version": "model-router-release-report-v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "catalog_schema_version": catalog_document["schema_version"],
        "ready_for_enforcement": all(gate.passed for gate in gates),
        "passed": sum(gate.passed for gate in gates),
        "total": len(gates),
        "gates": [asdict(gate) for gate in gates],
    }
