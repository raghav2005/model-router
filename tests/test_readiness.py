from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from model_router.catalog import load_catalog_document
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
            training_path.write_text(
                json.dumps(
                    {
                        "external_context_slice": {
                            "metrics": {
                                "count": 500,
                                "accuracy": 0.9,
                                "tier_underroute_rate": 0.02,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            live_path = root / "live.json"
            live_path.write_text(
                json.dumps(
                    {
                        "targets": {
                            role: {"runs": 20, "all_validators_pass_rate": 0.95}
                            for role in ("efficient", "balanced", "capable")
                        }
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_release_gates(
                policy_path=policy_path,
                catalog_path=catalog_path,
                training_report_path=training_path,
                live_summary_path=live_path,
                today=date(2026, 8, 11),
            )
        self.assertTrue(report["ready_for_enforcement"])


if __name__ == "__main__":
    unittest.main()
