from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from model_router.eval_cases import build_baseline_cases, write_baseline_cases
from model_router.live_eval import (
    EvaluationCase,
    _validate,
    evaluate_one,
    load_cases,
    run_benchmark,
    summarize,
)


class FakeClient:
    def chat_completions(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        max_tokens: int,
        **_: Any,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "choices": [{"message": {"content": "4"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }


class LiveEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = EvaluationCase(
            id="case-1",
            messages=({"role": "user", "content": "What is 2+2?"},),
            max_tokens=10,
            validators=({"type": "exact_match", "value": "4"},),
            metadata={"category": "math"},
        )

    def test_evaluate_one_records_usage_and_validator(self) -> None:
        result = evaluate_one(FakeClient(), self.case, "efficient")  # type: ignore[arg-type]
        self.assertTrue(result.success)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 1)
        self.assertIsNone(result.content)

    def test_summary_groups_targets(self) -> None:
        runs = [
            evaluate_one(FakeClient(), self.case, target)  # type: ignore[arg-type]
            for target in ("efficient", "capable")
        ]
        summary = summarize(runs)
        self.assertEqual(summary["total_runs"], 2)
        self.assertEqual(
            summary["targets"]["efficient"]["all_validators_pass_rate"], 1.0
        )
        target = summary["targets"]["efficient"]
        self.assertEqual(target["unique_cases"], 1)
        self.assertLess(target["all_validators_pass_rate_wilson_95"]["lower"], 1.0)
        self.assertIsNotNone(target["latency_ms"]["p99"])
        self.assertEqual(target["slices"]["category"]["math"]["runs"], 1)

    def test_load_cases_and_resumable_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            cases_path.write_text(
                json.dumps(
                    {
                        "id": "case-1",
                        "prompt": "What is 2+2?",
                        "max_tokens": 10,
                        "validators": [{"type": "exact_match", "value": "4"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cases = load_cases(cases_path)
            result_path = root / "results.jsonl"
            summary_path = root / "summary.json"
            first = run_benchmark(
                FakeClient(),  # type: ignore[arg-type]
                cases,
                ["efficient", "capable"],
                output_path=result_path,
                summary_path=summary_path,
                concurrency=2,
                repetitions=2,
            )
            second = run_benchmark(
                FakeClient(),  # type: ignore[arg-type]
                cases,
                ["efficient", "capable"],
                output_path=result_path,
                summary_path=summary_path,
                concurrency=2,
                resume=True,
                repetitions=2,
            )
            self.assertEqual(first, second)
            self.assertEqual(len(result_path.read_text().splitlines()), 4)
            self.assertIsNotNone(
                first["targets"]["efficient"]["estimated_total_cost_usd"]
            )
            self.assertEqual(
                first["provenance"]["benchmark_fingerprint"],
                second["provenance"]["benchmark_fingerprint"],
            )
            self.assertEqual(first["targets"]["efficient"]["unique_cases"], 1)
            self.assertEqual(first["provenance"]["case_set_profile"]["unique_cases"], 1)
            self.assertFalse(
                first["provenance"]["case_set_profile"]["contains_prompt_content"]
            )
            self.assertEqual(first["schema_version"], "switchyard-live-eval-summary-v3")

    def test_resume_rejects_changed_case_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "results.jsonl"
            summary_path = root / "summary.json"
            run_benchmark(
                FakeClient(),  # type: ignore[arg-type]
                [self.case],
                ["efficient"],
                output_path=result_path,
                summary_path=summary_path,
            )
            changed = EvaluationCase(
                id=self.case.id,
                messages=({"role": "user", "content": "What is 3+3?"},),
                max_tokens=self.case.max_tokens,
                validators=self.case.validators,
                metadata=self.case.metadata,
            )
            with self.assertRaisesRegex(ValueError, "cannot safely resume"):
                run_benchmark(
                    FakeClient(),  # type: ignore[arg-type]
                    [changed],
                    ["efficient"],
                    output_path=result_path,
                    summary_path=summary_path,
                )

    def test_extended_machine_validators(self) -> None:
        self.assertTrue(
            _validate(
                '{"name":"Ada","city":"London"}',
                {
                    "type": "json_equals",
                    "value": {"name": "Ada", "city": "London"},
                },
            ).passed
        )
        self.assertTrue(
            _validate(
                "3.1416",
                {"type": "numeric_tolerance", "value": 3.14, "tolerance": 0.01},
            ).passed
        )

    def test_generated_baseline_has_release_scale_and_slices(self) -> None:
        self.assertEqual(len(build_baseline_cases()), 120)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            write_baseline_cases(path)
            cases = load_cases(path)
        self.assertEqual(len(cases), 120)
        self.assertEqual(
            {case.metadata["use_case"] for case in cases},
            {"general_qa", "reasoning", "coding"},
        )


if __name__ == "__main__":
    unittest.main()
