from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from model_router.catalog import catalog_sha256, load_catalog_document
from model_router.readiness import evaluate_release_gates


class ReleaseGateTests(unittest.TestCase):
    def test_current_project_is_blocked_from_enforcement(self) -> None:
        report = evaluate_release_gates(today=date(2026, 8, 11))
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
                "minimum_live_cases_per_target": 10,
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
                            "metrics": {
                                "count": 500,
                                "accuracy": 0.9,
                                "tier_underroute_rate": 0.02,
                            }
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
                            "routing_policy_version": "hybrid-utility-policy-v4",
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
                        "verified_at": "2026-08-17T12:00:00+00:00",
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
                                "validator_scored_runs": 20,
                                "call_success_rate": 1.0,
                                "all_validators_pass_rate": 0.95,
                                "response_model_mismatch_count": 0,
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
            report = evaluate_release_gates(
                policy_path=policy_path,
                catalog_path=catalog_path,
                training_report_path=training_path,
                live_summary_path=live_path,
                pricing_report_path=pricing_path,
                artifact_path=artifact_path,
                today=date(2026, 8, 17),
            )
        self.assertTrue(report["ready_for_enforcement"])

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
                            "routing_policy_version": "hybrid-utility-policy-v4",
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


if __name__ == "__main__":
    unittest.main()
