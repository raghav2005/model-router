from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from .catalog import catalog_sha256, load_catalog, load_catalog_document
from .learned import MODEL_SCHEMA_VERSION, default_artifact_path
from .router import POLICY_VERSION
from .workload_evidence import sha256_file, validate_workload_evidence


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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval_lower(value: object) -> float | None:
    if not isinstance(value, dict) or value.get("lower") is None:
        return None
    return float(value["lower"])


def evaluate_release_gates(
    *,
    policy_path: str | Path = "config/release_policy.json",
    catalog_path: str | Path | None = None,
    training_report_path: str | Path = "reports/complexity_router_v3.json",
    live_summary_path: str | Path = "reports/live_eval_summary.json",
    pricing_report_path: str | Path = "reports/pricing_verification.json",
    external_dataset_evidence_path: str | Path = (
        "reports/external_dataset_evidence.json"
    ),
    metamorphic_report_path: str | Path = "reports/metamorphic_routing_eval.json",
    switchyard_contract_report_path: str | Path = "reports/switchyard_contract.json",
    workload_evidence_path: str | Path = "reports/workload_evidence.json",
    artifact_path: str | Path = default_artifact_path(),
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
            all(0 <= age <= maximum_age for age in ages),
            f"pricing age range is {min(ages)} to {max(ages)} days; "
            f"limit is 0 to {maximum_age}",
        )
    )

    pricing_path = Path(pricing_report_path)
    if pricing_path.exists():
        pricing = _read_json(pricing_path)
        raw_verification_models = pricing.get("models", [])
        verification_models = (
            raw_verification_models if isinstance(raw_verification_models, list) else []
        )
        expected_models = {model.id: model.provider_model for model in models}
        expected_roles = set(expected_models)
        verified_roles = {
            str(result.get("role"))
            for result in verification_models
            if isinstance(result, dict)
            and result.get("passed") is True
            and result.get("provider_model")
            == expected_models.get(str(result.get("role")))
        }
        verified_at = str(pricing.get("verified_at", ""))
        try:
            verification_age = _pricing_age_days(verified_at[:10], actual_today)
        except ValueError:
            verification_age = maximum_age + 1
        verification_passed = (
            pricing.get("schema_version") == "model-router-pricing-verification-v1"
            and pricing.get("passed") is True
            and pricing.get("catalog_sha256") == catalog_sha256(catalog_path)
            and verified_roles == expected_roles
            and len(verification_models) == len(expected_models)
            and 0 <= verification_age <= maximum_age
        )
        verification_detail = (
            f"verified roles={len(verified_roles)}/{len(expected_roles)}, "
            f"age={verification_age} days, catalogue digest "
            f"{'matches' if pricing.get('catalog_sha256') == catalog_sha256(catalog_path) else 'differs'}"
        )
    else:
        verification_passed = False
        verification_detail = f"pricing verification not found: {pricing_path}"
    gates.append(
        ReleaseGate(
            "pricing verification matches catalogue",
            verification_passed,
            verification_detail,
        )
    )

    workload_path = Path(workload_evidence_path)
    live_path = Path(live_summary_path)
    approved_workload_sha = policy.get("approved_workload_evidence_sha256")
    if workload_path.exists() and live_path.exists():
        workload = _read_json(workload_path)
        workload_valid, workload_detail = validate_workload_evidence(
            workload,
            live_summary_path=live_path,
            catalog_path=catalog_path,
        )
        actual_workload_sha = sha256_file(workload_path)
        workload_approved = (
            bool(approved_workload_sha)
            and approved_workload_sha == actual_workload_sha
            and workload_valid
        )
        workload_approval_detail = (
            workload_detail
            if workload_approved
            else (
                "workload evidence is valid but its SHA-256 is not approved"
                if workload_valid
                else workload_detail
            )
        )
    else:
        workload_approved = False
        missing_workload_inputs = [
            str(path) for path in (workload_path, live_path) if not path.exists()
        ]
        workload_approval_detail = "missing workload evidence: " + ", ".join(
            missing_workload_inputs
        )

    require_quality = bool(policy["require_workload_measured_quality"])
    gates.append(
        ReleaseGate(
            "model quality is workload-measured",
            not require_quality or workload_approved,
            workload_approval_detail,
        )
    )

    require_latency = bool(policy["require_workload_measured_latency"])
    gates.append(
        ReleaseGate(
            "model latency is workload-measured",
            not require_latency or workload_approved,
            workload_approval_detail,
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
        external_source_sha = (
            str(external.get("source_sha256", "")) if isinstance(external, dict) else ""
        )
        external_policy = policy.get("external_dataset_evidence", {})
        if not isinstance(external_policy, dict):
            raise ValueError("external_dataset_evidence must be an object")
        require_independent = bool(external_policy.get("require_independent", True))
        approved_external_evidence_sha = external_policy.get("approved_evidence_sha256")
        external_evidence_path = Path(external_dataset_evidence_path)
        if external_evidence_path.exists():
            external_evidence = _read_json(external_evidence_path)
            dataset = external_evidence.get("dataset", {})
            evidence_novel_rows = (
                dataset.get("novel_rows") if isinstance(dataset, dict) else None
            )
            evidence_digest_matches = bool(
                approved_external_evidence_sha
            ) and approved_external_evidence_sha == sha256_file(external_evidence_path)
            evidence_dataset_matches = (
                isinstance(dataset, dict)
                and bool(external_source_sha)
                and dataset.get("sha256") == external_source_sha
                and isinstance(evidence_novel_rows, int)
                and not isinstance(evidence_novel_rows, bool)
                and evidence_novel_rows == count
            )
            evidence_independent = external_evidence.get("independent") is True
            evidence_review_approved = (
                external_evidence.get("approved_for_release") is True
            )
            external_provenance_passed = (
                external_evidence.get("schema_version")
                == "model-router-external-dataset-evidence-v1"
                and external_evidence.get("contains_prompt_content") is False
                and evidence_digest_matches
                and evidence_dataset_matches
                and evidence_review_approved
                and (not require_independent or evidence_independent)
            )
            provenance_detail = (
                f"evidence_digest={'approved' if evidence_digest_matches else 'unapproved'}, "
                f"dataset={'matches' if evidence_dataset_matches else 'differs'}, "
                f"review_approved={evidence_review_approved}, "
                f"independent={evidence_independent}"
            )
        else:
            external_provenance_passed = False
            provenance_detail = (
                f"external dataset evidence not found: {external_evidence_path}"
            )
        external_passed = (
            count >= minimum_examples
            and accuracy >= minimum_accuracy
            and underroute <= maximum_underroute
            and external_provenance_passed
        )
        detail = (
            f"count={count}/{minimum_examples}, accuracy={accuracy:.3f}/"
            f"{minimum_accuracy:.3f}, tier_underroute={underroute:.3f}/"
            f"{maximum_underroute:.3f} max, {provenance_detail}"
        )
    else:
        external_passed = False
        detail = f"training report not found: {training_path}"
    gates.append(
        ReleaseGate("router generalises to external data", external_passed, detail)
    )

    synthetic_policy = policy.get("synthetic_regression")
    if isinstance(synthetic_policy, dict):
        metamorphic_path = Path(metamorphic_report_path)
        selected_policy = str(synthetic_policy.get("policy", "hybrid_adaptive_p15"))
        if metamorphic_path.exists():
            metamorphic = _read_json(metamorphic_path)
            policies = metamorphic.get("policies", {})
            selected = (
                policies.get(selected_policy, {}) if isinstance(policies, dict) else {}
            )
            if not isinstance(selected, dict):
                selected = {}
            count = int(selected.get("count", 0))
            underroute = float(selected.get("tier_underroute_rate", 1.0))
            invariance = float(selected.get("tier_invariance_rate", 0.0))
            minimum_cases = int(synthetic_policy["minimum_cases"])
            maximum_underroute = float(synthetic_policy["maximum_tier_underroute_rate"])
            minimum_invariance = float(synthetic_policy["minimum_tier_invariance_rate"])
            metamorphic_passed = (
                metamorphic.get("schema_version") == "model-router-metamorphic-eval-v1"
                and count >= minimum_cases
                and underroute <= maximum_underroute
                and invariance >= minimum_invariance
            )
            metamorphic_detail = (
                f"policy={selected_policy}, count={count}/{minimum_cases}, "
                f"underroute={underroute:.3f}/{maximum_underroute:.3f} max, "
                f"tier_invariance={invariance:.3f}/{minimum_invariance:.3f} min"
            )
        else:
            metamorphic_passed = False
            metamorphic_detail = (
                f"metamorphic regression report not found: {metamorphic_path}"
            )
        gates.append(
            ReleaseGate(
                "metamorphic routing regression passes",
                metamorphic_passed,
                metamorphic_detail,
            )
        )

    artifact = Path(artifact_path)
    if training_path.exists() and artifact.exists():
        training = _read_json(training_path)
        provenance = training.get("provenance", {})
        artifact_evidence = (
            provenance.get("artifact", {}) if isinstance(provenance, dict) else {}
        )
        catalog_evidence = (
            provenance.get("catalog", {}) if isinstance(provenance, dict) else {}
        )
        artifact_matches = (
            isinstance(artifact_evidence, dict)
            and artifact_evidence.get("sha256") == _sha256_file(artifact)
            and artifact_evidence.get("schema_version") == MODEL_SCHEMA_VERSION
        )
        catalog_matches = (
            isinstance(catalog_evidence, dict)
            and catalog_evidence.get("sha256") == catalog_sha256(catalog_path)
            and catalog_evidence.get("schema_version")
            == catalog_document["schema_version"]
        )
        policy_matches = (
            isinstance(provenance, dict)
            and provenance.get("routing_policy_version") == POLICY_VERSION
        )
        evidence_passed = (
            training.get("schema_version") == "router-training-report-v2"
            and artifact_matches
            and catalog_matches
            and policy_matches
        )
        evidence_detail = (
            f"artifact={'matches' if artifact_matches else 'differs'}, "
            f"catalogue={'matches' if catalog_matches else 'differs'}, "
            f"policy={'matches' if policy_matches else 'differs'}"
        )
    else:
        evidence_passed = False
        missing = []
        if not training_path.exists():
            missing.append(str(training_path))
        if not artifact.exists():
            missing.append(str(artifact))
        evidence_detail = "missing release evidence: " + ", ".join(missing)
    gates.append(
        ReleaseGate(
            "router evidence matches deployed artifact",
            evidence_passed,
            evidence_detail,
        )
    )

    if live_path.exists():
        live = _read_json(live_path)
        targets = live.get("targets", {})
        if not isinstance(targets, dict):
            targets = {}
        minimum_cases = int(policy["minimum_live_cases_per_target"])
        minimum_unique_cases = int(
            policy.get("minimum_live_unique_cases_per_target", minimum_cases)
        )
        required_use_case_slices = policy.get(
            "minimum_live_unique_cases_by_use_case", {}
        )
        if not isinstance(required_use_case_slices, dict):
            raise ValueError("minimum_live_unique_cases_by_use_case must be an object")
        required_use_case_quality = policy.get(
            "minimum_live_all_validators_pass_rate_by_use_case", {}
        )
        if not isinstance(required_use_case_quality, dict):
            raise ValueError(
                "minimum_live_all_validators_pass_rate_by_use_case must be an object"
            )
        minimum_pass = float(policy["minimum_live_all_validators_pass_rate"])
        minimum_call_success = float(policy["minimum_live_call_success_rate"])
        minimum_pass_lower = float(
            policy.get("minimum_live_all_validators_pass_wilson_lower_bound", 0.0)
        )
        minimum_call_success_lower = float(
            policy.get("minimum_live_call_success_wilson_lower_bound", 0.0)
        )
        maximum_model_mismatches = int(
            policy["maximum_live_response_model_mismatch_count"]
        )
        failures: list[str] = []
        for model in models:
            result = targets.get(model.switchyard_target, {})
            runs = int(result.get("runs", 0)) if isinstance(result, dict) else 0
            unique_cases = (
                int(result.get("unique_cases", 0)) if isinstance(result, dict) else 0
            )
            pass_rate = (
                result.get("all_validators_pass_rate")
                if isinstance(result, dict)
                else None
            )
            scored_runs = (
                int(result.get("validator_scored_runs", 0))
                if isinstance(result, dict)
                else 0
            )
            call_success_rate = (
                result.get("call_success_rate") if isinstance(result, dict) else None
            )
            pass_lower = (
                _interval_lower(result.get("all_validators_pass_rate_wilson_95"))
                if isinstance(result, dict)
                else None
            )
            call_success_lower = (
                _interval_lower(result.get("call_success_rate_wilson_95"))
                if isinstance(result, dict)
                else None
            )
            model_mismatches = (
                int(result.get("response_model_mismatch_count", 0))
                if isinstance(result, dict)
                else maximum_model_mismatches + 1
            )
            latency = result.get("latency_ms", {}) if isinstance(result, dict) else {}
            latency_p95 = latency.get("p95") if isinstance(latency, dict) else None
            slices = result.get("slices", {}) if isinstance(result, dict) else {}
            use_case_slices = (
                slices.get("use_case", {}) if isinstance(slices, dict) else {}
            )
            missing_slices: list[str] = []
            for slice_name, raw_minimum in required_use_case_slices.items():
                slice_result = (
                    use_case_slices.get(slice_name, {})
                    if isinstance(use_case_slices, dict)
                    else {}
                )
                slice_unique_cases = (
                    int(slice_result.get("unique_cases", 0))
                    if isinstance(slice_result, dict)
                    else 0
                )
                if slice_unique_cases < int(raw_minimum):
                    missing_slices.append(
                        f"{slice_name}={slice_unique_cases}/{int(raw_minimum)}"
                    )
                slice_pass_rate = (
                    slice_result.get("all_validators_pass_rate")
                    if isinstance(slice_result, dict)
                    else None
                )
                minimum_slice_pass_rate = required_use_case_quality.get(slice_name)
                if minimum_slice_pass_rate is not None and (
                    slice_pass_rate is None
                    or float(slice_pass_rate) < float(minimum_slice_pass_rate)
                ):
                    missing_slices.append(
                        f"{slice_name}_pass={slice_pass_rate}/"
                        f"{float(minimum_slice_pass_rate):.3f}"
                    )
            if (
                runs < minimum_cases
                or unique_cases < minimum_unique_cases
                or scored_runs < minimum_cases
                or pass_rate is None
                or float(pass_rate) < minimum_pass
                or call_success_rate is None
                or float(call_success_rate) < minimum_call_success
                or (minimum_pass_lower > 0 and pass_lower is None)
                or (pass_lower is not None and pass_lower < minimum_pass_lower)
                or (minimum_call_success_lower > 0 and call_success_lower is None)
                or (
                    call_success_lower is not None
                    and call_success_lower < minimum_call_success_lower
                )
                or model_mismatches > maximum_model_mismatches
                or latency_p95 is None
                or missing_slices
            ):
                reasons = [
                    f"runs={runs}/{minimum_cases}",
                    f"unique={unique_cases}/{minimum_unique_cases}",
                    f"scored={scored_runs}/{minimum_cases}",
                ]
                if latency_p95 is None:
                    reasons.append("latency_p95=missing")
                if missing_slices:
                    reasons.append("use_case_slices=" + ",".join(missing_slices))
                failures.append(f"{model.id} ({'; '.join(reasons)})")
        live_passed = not failures
        live_detail = (
            "roles below live threshold: " + "; ".join(failures)
            if failures
            else "all roles meet live unique-case, slice, latency, and validator thresholds"
        )
    else:
        live_passed = False
        live_detail = f"live evaluation summary not found: {live_path}"
    gates.append(
        ReleaseGate("live response benchmark passes", live_passed, live_detail)
    )

    approved_case_set = policy.get("approved_live_case_set_sha256")
    if live_path.exists():
        live = _read_json(live_path)
        live_provenance = live.get("provenance", {})
        switchyard = policy.get("switchyard", {})
        tested_switchyard_revision = (
            switchyard.get("approved_version_or_commit")
            or switchyard.get("tested_version_or_commit")
            if isinstance(switchyard, dict)
            else None
        )
        provenance_targets = (
            live_provenance.get("targets", [])
            if isinstance(live_provenance, dict)
            else []
        )
        live_evidence_passed = (
            live.get("schema_version") == "switchyard-live-eval-summary-v3"
            and isinstance(live_provenance, dict)
            and live_provenance.get("schema_version")
            == "switchyard-live-eval-provenance-v1"
            and live_provenance.get("catalog_sha256") == catalog_sha256(catalog_path)
            and bool(approved_case_set)
            and live_provenance.get("case_set_sha256") == approved_case_set
            and isinstance(provenance_targets, list)
            and all(isinstance(target, str) for target in provenance_targets)
            and set(provenance_targets) == {model.switchyard_target for model in models}
            and bool(tested_switchyard_revision)
            and live_provenance.get("switchyard_revision") == tested_switchyard_revision
        )
        live_evidence_detail = (
            "live summary matches the exact catalogue and approved case set"
            if live_evidence_passed
            else "live summary is unbound, stale, or not from the approved case set"
        )
    else:
        live_evidence_passed = False
        live_evidence_detail = f"live evaluation summary not found: {live_path}"
    gates.append(
        ReleaseGate(
            "live benchmark evidence is approved and immutable",
            live_evidence_passed,
            live_evidence_detail,
        )
    )

    switchyard = policy.get("switchyard", {})
    switchyard_pin = (
        switchyard.get("approved_version_or_commit")
        or switchyard.get("tested_version_or_commit")
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
            "Switchyard release is approved and pinned",
            bool(switchyard_pin),
            (
                f"approved pin: {switchyard_pin}"
                if switchyard_pin
                else "no approved Switchyard version or commit is recorded"
            ),
        )
    )
    if isinstance(switchyard, dict) and switchyard.get("contract_report_required"):
        contract_path = Path(switchyard_contract_report_path)
        config_path = Path("config/switchyard_routes.toml")
        if contract_path.exists() and config_path.exists():
            contract = _read_json(contract_path)
            release_version = str(switchyard.get("release", "")).removeprefix("v")
            contract_passed = (
                contract.get("schema_version") == "model-router-switchyard-contract-v1"
                and contract.get("passed") is True
                and contract.get("switchyard_version") == release_version
                and contract.get("config_sha256") == _sha256_file(config_path)
            )
            contract_detail = (
                f"version={contract.get('switchyard_version')}, config digest "
                f"{'matches' if contract.get('config_sha256') == _sha256_file(config_path) else 'differs'}"
            )
        else:
            contract_passed = False
            contract_detail = f"Switchyard contract report not found: {contract_path}"
        gates.append(
            ReleaseGate(
                "Switchyard runtime contract passes",
                contract_passed,
                contract_detail,
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
        "approved_workload_evidence_sha256": (
            actual_workload_sha if workload_approved else None
        ),
        "ready_for_enforcement": all(gate.passed for gate in gates),
        "passed": sum(gate.passed for gate in gates),
        "total": len(gates),
        "gates": [asdict(gate) for gate in gates],
    }
