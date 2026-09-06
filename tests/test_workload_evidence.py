from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_router.catalog import catalog_sha256, load_catalog
from model_router.workload_evidence import (
    apply_workload_evidence,
    build_workload_evidence,
    validate_workload_evidence,
)


def live_summary() -> dict[str, object]:
    return {
        "schema_version": "switchyard-live-eval-summary-v3",
        "provenance": {
            "catalog_sha256": catalog_sha256(),
            "case_set_sha256": "case-digest",
            "switchyard_revision": "switchyard-commit",
            "benchmark_fingerprint": "benchmark-digest",
        },
        "targets": {
            model.switchyard_target: {
                "unique_cases": 120,
                "validator_scored_runs": 120,
                "all_validators_pass_rate": 0.95,
                "all_validators_pass_rate_wilson_95": {
                    "lower": 0.90,
                    "upper": 0.98,
                },
                "call_success_rate": 1.0,
                "call_success_rate_wilson_95": {"lower": 0.97, "upper": 1.0},
                "latency_ms": {"p50": 100.1, "p95": 250.2, "p99": 400.3},
                "ttft_ms": {"p50": 30.1, "p95": 60.2, "p99": 90.3},
                "estimated_total_cost_usd": 1.25,
                "response_model_mismatch_count": 0,
                "slices": {
                    "use_case": {
                        use_case: {
                            "unique_cases": 40,
                            "validator_scored_runs": 40,
                            "all_validators_pass_rate": score,
                            "all_validators_pass_rate_wilson_95": {
                                "lower": score - 0.05,
                                "upper": min(1.0, score + 0.05),
                            },
                        }
                        for use_case, score in (
                            ("general_qa", 0.91),
                            ("coding", 0.92),
                            ("reasoning", 0.93),
                        )
                    }
                },
            }
            for model in load_catalog()
        },
    }


class WorkloadEvidenceTests(unittest.TestCase):
    def test_builds_prompt_free_evidence_and_applies_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.json"
            path.write_text(json.dumps(live_summary()), encoding="utf-8")
            evidence = build_workload_evidence(path)
            valid, _ = validate_workload_evidence(
                evidence,
                live_summary_path=path,
            )

        self.assertTrue(valid)
        self.assertTrue(evidence["complete"])
        self.assertFalse(evidence["privacy"]["contains_prompt_content"])
        updated = apply_workload_evidence(load_catalog(), evidence)
        efficient = next(model for model in updated if model.id == "efficient")
        self.assertEqual(efficient.latency_p95_ms, 251)
        self.assertEqual(efficient.skills["coding"], 0.92)
        self.assertEqual(efficient.quality_evidence, "workload_measured")
        self.assertEqual(efficient.latency_evidence, "workload_measured")

    def test_validation_rejects_a_changed_live_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.json"
            summary = live_summary()
            path.write_text(json.dumps(summary), encoding="utf-8")
            evidence = build_workload_evidence(path)
            summary["total_runs"] = 999
            path.write_text(json.dumps(summary), encoding="utf-8")
            valid, detail = validate_workload_evidence(
                evidence,
                live_summary_path=path,
            )

        self.assertFalse(valid)
        self.assertIn("does not match the live summary", detail)


if __name__ == "__main__":
    unittest.main()
