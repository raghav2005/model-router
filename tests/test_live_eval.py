from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from model_router.live_eval import (
    EvaluationCase,
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


if __name__ == "__main__":
    unittest.main()
