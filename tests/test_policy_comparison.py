from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_router.catalog import catalog_sha256, load_catalog
from model_router.learned import default_artifact_path
from model_router.live_eval import case_set_sha256, load_cases
from model_router.policy_comparison import (
    build_policy_comparison,
    validate_policy_comparison,
)
from model_router.workload_evidence import (
    build_workload_evidence,
    sha256_file,
    write_workload_evidence,
)


class PolicyComparisonTests(unittest.TestCase):
    def _fixtures(self, root: Path) -> tuple[Path, Path, Path, Path]:
        cases_path = root / "cases.jsonl"
        prompts = (
            ("fact", "What is 2+2?", "general_qa"),
            ("code", "Write a function that adds two integers.", "coding"),
            ("reason", "Which number follows 2, 4, 6?", "reasoning"),
        )
        cases_path.write_text(
            "".join(
                json.dumps(
                    {
                        "id": case_id,
                        "prompt": prompt,
                        "max_tokens": 50,
                        "validators": [{"type": "contains_all", "values": ["ok"]}],
                        "metadata": {"use_case": use_case},
                    }
                )
                + "\n"
                for case_id, prompt, use_case in prompts
            ),
            encoding="utf-8",
        )
        models = load_catalog()
        results_path = root / "results.jsonl"
        results_path.write_text(
            "".join(
                json.dumps(
                    {
                        "case_id": case_id,
                        "target": model.switchyard_target,
                        "repetition": 1,
                        "success": True,
                        "score": 1.0,
                        "latency_ms": 100 * model.tier,
                        "estimated_cost_usd": 0.001 * model.tier,
                    }
                )
                + "\n"
                for case_id, _, _ in prompts
                for model in models
            ),
            encoding="utf-8",
        )
        cases = load_cases(cases_path)
        summary_path = root / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "switchyard-live-eval-summary-v3",
                    "targets": {
                        model.switchyard_target: {
                            "unique_cases": 3,
                            "validator_scored_runs": 3,
                            "all_validators_pass_rate": 1.0,
                            "all_validators_pass_rate_wilson_95": {
                                "lower": 0.44,
                                "upper": 1.0,
                            },
                            "call_success_rate": 1.0,
                            "call_success_rate_wilson_95": {
                                "lower": 0.44,
                                "upper": 1.0,
                            },
                            "latency_ms": {"p50": 100, "p95": 200, "p99": 300},
                            "ttft_ms": {"p50": 50, "p95": 80, "p99": 90},
                            "response_model_mismatch_count": 0,
                            "slices": {
                                "use_case": {
                                    use_case: {
                                        "unique_cases": 1,
                                        "validator_scored_runs": 1,
                                        "all_validators_pass_rate": 1.0,
                                        "all_validators_pass_rate_wilson_95": {
                                            "lower": 0.2,
                                            "upper": 1.0,
                                        },
                                    }
                                    for use_case in (
                                        "general_qa",
                                        "coding",
                                        "reasoning",
                                    )
                                }
                            },
                        }
                        for model in models
                    },
                    "provenance": {
                        "case_set_sha256": case_set_sha256(cases),
                        "catalog_sha256": catalog_sha256(),
                        "results_sha256": sha256_file(results_path),
                        "targets": [model.switchyard_target for model in models],
                        "repetitions": 1,
                        "benchmark_fingerprint": "test-fingerprint",
                    },
                }
            ),
            encoding="utf-8",
        )
        workload_path = root / "workload.json"
        write_workload_evidence(workload_path, build_workload_evidence(summary_path))
        return cases_path, results_path, summary_path, workload_path

    def test_builds_paired_prompt_free_policy_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_path, results_path, summary_path, workload_path = self._fixtures(
                Path(directory)
            )
            report = build_policy_comparison(
                cases_path,
                results_path,
                summary_path,
                workload_evidence_path=workload_path,
            )
            valid, detail = validate_policy_comparison(
                report,
                live_summary_path=summary_path,
                workload_evidence_path=workload_path,
            )
        self.assertTrue(valid, detail)
        self.assertTrue(report["complete"])
        self.assertEqual(report["coverage"]["paired_observations"], 3)
        self.assertEqual(report["arms"]["capable"]["observations"], 3)
        rendered = json.dumps(report)
        self.assertNotIn("What is 2+2?", rendered)
        self.assertFalse(report["privacy"]["contains_prompt_content"])

    def test_rejects_an_incomplete_paired_result_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path, results_path, summary_path, workload_path = self._fixtures(root)
            lines = results_path.read_text(encoding="utf-8").splitlines()
            results_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["provenance"]["results_sha256"] = sha256_file(results_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            write_workload_evidence(
                workload_path, build_workload_evidence(summary_path)
            )
            with self.assertRaisesRegex(ValueError, "matrix is incomplete"):
                build_policy_comparison(
                    cases_path,
                    results_path,
                    summary_path,
                    workload_evidence_path=workload_path,
                    artifact_path=default_artifact_path(),
                )


if __name__ == "__main__":
    unittest.main()
