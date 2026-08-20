from __future__ import annotations

import unittest

from model_router.metamorphic import (
    TRANSFORMATIONS,
    build_metamorphic_cases,
    evaluate_metamorphic_suite,
)
from model_router.normalization import (
    SUPPORTED_ENVELOPES,
    transform_prompt,
    unwrap_supported_envelope,
)


class MetamorphicEvaluationTests(unittest.TestCase):
    def test_cases_are_unique_and_large_enough_for_regression(self) -> None:
        cases = build_metamorphic_cases()
        self.assertGreaterEqual(len(cases), 1_000)
        self.assertEqual(len({case.id for case in cases}), len(cases))
        self.assertEqual({case.transformation for case in cases}, set(TRANSFORMATIONS))

    def test_report_separates_synthetic_from_workload_evidence(self) -> None:
        report = evaluate_metamorphic_suite()
        selected = report["policies"]["hybrid_adaptive_p15"]
        self.assertEqual(selected["count"], len(build_metamorphic_cases()))
        self.assertIsNotNone(selected["tier_invariance_rate"])
        self.assertTrue(
            any("never be counted" in item for item in report["limitations"])
        )

    def test_only_exact_supported_envelopes_are_unwrapped(self) -> None:
        task = "Prove the invariant under network partitions."
        for envelope in SUPPORTED_ENVELOPES:
            wrapped = transform_prompt(task, envelope)
            self.assertEqual(unwrap_supported_envelope(wrapped), (task, envelope))
        modified = transform_prompt(task, "xml_envelope") + " extra instruction"
        self.assertEqual(unwrap_supported_envelope(modified), (modified, None))


if __name__ == "__main__":
    unittest.main()
