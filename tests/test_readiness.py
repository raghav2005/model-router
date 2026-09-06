from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from model_router.catalog import catalog_sha256, load_catalog_document
from model_router.readiness import evaluate_release_gates
from model_router.router import POLICY_VERSION
from model_router.workload_evidence import (
    build_workload_evidence,
    sha256_file,
    write_workload_evidence,
)


class ReleaseGateTests(unittest.TestCase):
    def test_current_project_is_blocked_from_enforcement(self) -> None:
        report = evaluate_release_gates(today=date(2026, 8, 25))
        self.assertFalse(report["ready_for_enforcement"])
        failures = {gate["name"] for gate in report["gates"] if not gate["passed"]}
        self.assertIn("model quality is workload-measured", failures)
        self.assertIn("live response benchmark passes", failures)

    def test_all_release_gates_can_pass_with_measured_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = load_catalog_document()
            for model in catalog["models"]:
                model["quality_evidence"] = "workload_measured"
                model["latency_evidence"] = "workload_measured"
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

            policy = {
                "deployment_mode": "enforce",
                "maximum_pricing_age_days": 30,
                "minimum_external_examples": 100,
                "minimum_external_accuracy": 0.75,
                "maximum_external_tier_underroute_rate": 0.05,
                "external_dataset_evidence": {
                    "require_independent": True,
                    "approved_evidence_sha256": None,
                },
                "minimum_live_cases_per_target": 10,
                "minimum_live_unique_cases_per_target": 10,
                "minimum_live_unique_cases_by_use_case": {
                    "general_qa": 2,
                    "coding": 2,
                    "reasoning": 2,
                },
                "minimum_live_all_validators_pass_rate": 0.9,
                "minimum_live_call_success_rate": 0.99,
                "maximum_live_response_model_mismatch_count": 0,
                "approved_live_case_set_sha256": "approved-cases",
                "require_workload_measured_quality": True,
                "require_workload_measured_latency": True,
                "switchyard": {
                    "tested_version_or_commit": "abc123",
                    "trusted_direct_provider_fallback_configured": True,
                },
            }
            external_evidence_path = root / "external-evidence.json"
            external_evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": "model-router-external-dataset-evidence-v1",
                        "independent": True,
                        "approved_for_release": True,
                        "contains_prompt_content": False,
                        "dataset": {
                            "sha256": "external-source",
                            "novel_rows": 500,
                        },
                    }
                ),
                encoding="utf-8",
            )
            policy["external_dataset_evidence"]["approved_evidence_sha256"] = (
                sha256_file(external_evidence_path)
            )
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8")

            training_path = root / "training.json"
            artifact_path = root / "complexity_router_v2.npz"
            artifact_path.write_bytes(b"exact deployed router artifact")
            artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            training_path.write_text(
                json.dumps(
                    {
                        "schema_version": "router-training-report-v2",
                        "external_context_slice": {
                            "source_sha256": "external-source",
                            "metrics": {
                                "count": 500,
                                "accuracy": 0.9,
                                "tier_underroute_rate": 0.02,
                            },
                        },
                        "provenance": {
                            "artifact": {
                                "sha256": artifact_digest,
                                "schema_version": "complexity-router-nb-v2",
                            },
                            "catalog": {
                                "sha256": catalog_sha256(catalog_path),
                                "schema_version": catalog["schema_version"],
                            },
                            "routing_policy_version": POLICY_VERSION,
                        },
                    }
                ),
                encoding="utf-8",
            )

            pricing_path = root / "pricing.json"
            pricing_path.write_text(
                json.dumps(
                    {
                        "schema_version": "model-router-pricing-verification-v1",
                        "verified_at": "2026-08-25T12:00:00+00:00",
                        "catalog_sha256": catalog_sha256(catalog_path),
                        "passed": True,
                        "models": [
                            {
                                "role": role,
                                "provider_model": provider_model,
                                "passed": True,
                            }
                            for role, provider_model in (
                                ("efficient", "gpt-5.6-luna"),
                                ("balanced", "gpt-5.6-terra"),
                                ("capable", "gpt-5.6-sol"),
                            )
                        ],
                    }
                ),
                encoding="utf-8",
            )

            live_path = root / "live.json"
            live_path.write_text(
                json.dumps(
                    {
                        "schema_version": "switchyard-live-eval-summary-v3",
                        "targets": {
                            role: {
                                "runs": 20,
                                "unique_cases": 20,
                                "validator_scored_runs": 20,
                                "call_success_rate": 1.0,
                                "call_success_rate_wilson_95": {
                                    "lower": 0.9,
                                    "upper": 1.0,
                                },
                                "all_validators_pass_rate": 0.95,
                                "all_validators_pass_rate_wilson_95": {
                                    "lower": 0.8,
                                    "upper": 1.0,
                                },
                                "response_model_mismatch_count": 0,
                                "latency_ms": {"p50": 100, "p95": 200, "p99": 250},
                                "slices": {
                                    "use_case": {
                                        use_case: {"runs": 5, "unique_cases": 5}
                                        for use_case in (
                                            "general_qa",
                                            "coding",
                                            "reasoning",
                                        )
                                    }
                                },
                            }
                            for role in ("efficient", "balanced", "capable")
                        },
                        "provenance": {
                            "schema_version": "switchyard-live-eval-provenance-v1",
                            "catalog_sha256": catalog_sha256(catalog_path),
                            "case_set_sha256": "approved-cases",
                            "targets": ["efficient", "balanced", "capable"],
                            "switchyard_revision": "abc123",
                        },
                    }
                ),
                encoding="utf-8",
            )
            live_document = json.loads(live_path.read_text(encoding="utf-8"))
            for target in live_document["targets"].values():
                for use_case in target["slices"]["use_case"].values():
                    use_case.update(
                        {
                            "validator_scored_runs": 5,
                            "all_validators_pass_rate": 1.0,
                            "all_validators_pass_rate_wilson_95": {"lower": 0.5},
                        }
                    )
            live_path.write_text(json.dumps(live_document), encoding="utf-8")
            workload_path = root / "workload.json"
            write_workload_evidence(
                workload_path,
                build_workload_evidence(live_path, catalog_path=catalog_path),
            )
            policy["approved_workload_evidence_sha256"] = sha256_file(workload_path)
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            report = evaluate_release_gates(
                policy_path=policy_path,
                catalog_path=catalog_path,
                training_report_path=training_path,
                live_summary_path=live_path,
                pricing_report_path=pricing_path,
                external_dataset_evidence_path=external_evidence_path,
                workload_evidence_path=workload_path,
                artifact_path=artifact_path,
                today=date(2026, 8, 25),
            )
        self.assertTrue(report["ready_for_enforcement"], report)

    def test_router_evidence_rejects_a_different_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "router.npz"
            artifact_path.write_bytes(b"new artifact")
            training_path = root / "training.json"
            training_path.write_text(
                json.dumps(
                    {
                        "schema_version": "router-training-report-v2",
                        "external_context_slice": {"metrics": {}},
                        "provenance": {
                            "artifact": {
                                "sha256": hashlib.sha256(b"old artifact").hexdigest(),
                                "schema_version": "complexity-router-nb-v2",
                            },
                            "catalog": {
                                "sha256": catalog_sha256(),
                                "schema_version": load_catalog_document()[
                                    "schema_version"
                                ],
                            },
                            "routing_policy_version": POLICY_VERSION,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_release_gates(
                training_report_path=training_path,
                artifact_path=artifact_path,
                today=date(2026, 8, 17),
            )
        gate = next(
            gate
            for gate in report["gates"]
            if gate["name"] == "router evidence matches deployed artifact"
        )
        self.assertFalse(gate["passed"])

    def test_pricing_verification_is_bound_to_exact_catalogue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pricing_path = Path(directory) / "pricing.json"
            pricing_path.write_text(
                json.dumps(
                    {
                        "schema_version": "model-router-pricing-verification-v1",
                        "verified_at": "2026-08-11T12:00:00+00:00",
                        "catalog_sha256": "stale-digest",
                        "passed": True,
                        "models": [
                            {"role": role, "passed": True}
                            for role in ("efficient", "balanced", "capable")
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_release_gates(
                pricing_report_path=pricing_path,
                today=date(2026, 8, 11),
            )
        gate = next(
            gate
            for gate in report["gates"]
            if gate["name"] == "pricing verification matches catalogue"
        )
        self.assertFalse(gate["passed"])

    def test_repetitions_cannot_substitute_for_unique_live_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            live_path = Path(directory) / "live.json"
            live_path.write_text(
                json.dumps(
                    {
                        "schema_version": "switchyard-live-eval-summary-v3",
                        "targets": {
                            role: {
                                "runs": 100,
                                "unique_cases": 1,
                                "validator_scored_runs": 100,
                                "call_success_rate": 1.0,
                                "call_success_rate_wilson_95": {"lower": 0.99},
                                "all_validators_pass_rate": 1.0,
                                "all_validators_pass_rate_wilson_95": {"lower": 0.99},
                                "response_model_mismatch_count": 0,
                                "latency_ms": {"p50": 100, "p95": 200, "p99": 250},
                                "slices": {
                                    "use_case": {
                                        use_case: {
                                            "runs": 100,
                                            "unique_cases": 1,
                                        }
                                        for use_case in (
                                            "general_qa",
                                            "coding",
                                            "reasoning",
                                        )
                                    }
                                },
                            }
                            for role in ("efficient", "balanced", "capable")
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_release_gates(
                live_summary_path=live_path,
                today=date(2026, 8, 25),
            )
        gate = next(
            gate
            for gate in report["gates"]
            if gate["name"] == "live response benchmark passes"
        )
        self.assertFalse(gate["passed"])
        self.assertIn("unique=1/100", gate["detail"])

    def test_related_synthetic_external_data_cannot_pass_as_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "external.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": "model-router-external-dataset-evidence-v1",
                        "independent": False,
                        "approved_for_release": False,
                        "contains_prompt_content": False,
                        "dataset": {
                            "sha256": (
                                "71c9b9c521a12fbf7675430f938b9559eb6a1a40e96762a16f99b72b6c08fddd"
                            ),
                            "novel_rows": 455,
                        },
                    }
                ),
                encoding="utf-8",
            )
            policy = json.loads(Path("config/release_policy.json").read_text())
            policy["minimum_external_examples"] = 1
            policy["minimum_external_accuracy"] = 0.0
            policy["maximum_external_tier_underroute_rate"] = 1.0
            policy["external_dataset_evidence"]["approved_evidence_sha256"] = (
                sha256_file(evidence_path)
            )
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            report = evaluate_release_gates(
                policy_path=policy_path,
                external_dataset_evidence_path=evidence_path,
                today=date(2026, 8, 25),
            )
        gate = next(
            gate
            for gate in report["gates"]
            if gate["name"] == "router generalises to external data"
        )
        self.assertFalse(gate["passed"])
        self.assertIn("independent=False", gate["detail"])


if __name__ == "__main__":
    unittest.main()
