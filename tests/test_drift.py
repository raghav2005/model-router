from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_router.drift import (
    build_baseline,
    compare_to_baseline,
    jensen_shannon_distance,
    load_route_events,
)


def event(
    index: int,
    *,
    model: str = "balanced",
    use_case: str = "general_qa",
    level: int = 3,
    confidence: float = 0.7,
) -> dict[str, object]:
    return {
        "event": "route_decision",
        "request_id": f"request-{index}",
        "policy_version": "policy-v1",
        "classifier_model_version": "model-v1",
        "selected_model": model,
        "use_case": use_case,
        "risk": "normal",
        "complexity_level": level,
        "classifier_source": "hybrid",
        "classifier_confidence": confidence,
        "classifier_entropy": 0.4,
        "tier_underroute_probability": 0.1,
        "estimated_cost_usd": 0.01,
    }


class DriftTests(unittest.TestCase):
    def test_identical_distribution_passes(self) -> None:
        events = [event(index) for index in range(200)]
        report = compare_to_baseline(build_baseline(events), events)
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "ok")

    def test_material_distribution_change_is_detected(self) -> None:
        baseline_events = [event(index) for index in range(200)]
        current_events = [
            event(index, model="capable", use_case="reasoning", level=5, confidence=0.2)
            for index in range(200)
        ]
        report = compare_to_baseline(build_baseline(baseline_events), current_events)
        self.assertFalse(report["passed"])
        self.assertIn("selected_model", report["drifted_fields"])
        self.assertIn("classifier_confidence", report["drifted_fields"])

    def test_small_window_is_not_treated_as_healthy(self) -> None:
        baseline = build_baseline([event(index) for index in range(100)])
        report = compare_to_baseline(baseline, [event(1)], minimum_events=10)
        self.assertEqual(report["status"], "insufficient_data")

    def test_loader_ignores_execution_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            path.write_text(
                json.dumps(event(1)) + "\n" + json.dumps({"event": "execution"}) + "\n",
                encoding="utf-8",
            )
            loaded = load_route_events(path)
        self.assertEqual(len(loaded), 1)

    def test_jensen_shannon_distance_is_bounded_and_symmetric(self) -> None:
        left = {"a": 1.0}
        right = {"b": 1.0}
        self.assertEqual(jensen_shannon_distance(left, right), 1.0)
        self.assertEqual(
            jensen_shannon_distance(left, right),
            jensen_shannon_distance(right, left),
        )


if __name__ == "__main__":
    unittest.main()
