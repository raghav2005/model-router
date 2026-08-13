from __future__ import annotations

import unittest

from model_router.adversarial import (
    build_adversarial_cases,
    evaluate_adversarial_suite,
)


class AdversarialEvaluationTests(unittest.TestCase):
    def test_cases_are_unique_and_cover_expected_failure_modes(self) -> None:
        cases = build_adversarial_cases()
        self.assertGreaterEqual(len(cases), 150)
        self.assertEqual(len({case.id for case in cases}), len(cases))
        categories = {case.category for case in cases}
        self.assertIn("lexical_trap_simple", categories)
        self.assertIn("concise_hard", categories)
        self.assertIn("multi_turn_shift", categories)
        self.assertIn("routing_injection", categories)

    def test_report_keeps_synthetic_evidence_separate(self) -> None:
        report = evaluate_adversarial_suite()
        self.assertEqual(report["case_count"], len(build_adversarial_cases()))
        self.assertIn("hybrid_adaptive_p15", report["policies"])
        self.assertTrue(
            any("never be counted" in item for item in report["limitations"])
        )


if __name__ == "__main__":
    unittest.main()
