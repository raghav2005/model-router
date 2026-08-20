from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_router.evaluation import classification_metrics
from model_router.learned import (
    HashedTextVectorizer,
    NaiveBayesComplexityModel,
    default_artifact_path,
    normalize_prompt,
)
from model_router.router import ModelRouter
from model_router.training import (
    DatasetRow,
    TrainingConfig,
    _label_weights,
    split_for_prompt,
)
from model_router.types import RoutingRequest


class LearnedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = NaiveBayesComplexityModel.load()

    def test_default_artifact_is_present_and_loadable(self) -> None:
        self.assertTrue(default_artifact_path().exists())
        self.assertEqual(self.model.feature_dimension, 32_768)
        self.assertIn("training_dataset_sha256", self.model.metadata)
        self.assertIn("training_recipe_sha256", self.model.metadata)
        self.assertIn("training_implementation_sha256", self.model.metadata)
        self.assertGreater(self.model.final_turn_weight, 0.0)

    def test_probabilities_are_normalized(self) -> None:
        prediction = self.model.predict("What is the capital of Japan?")
        self.assertAlmostEqual(sum(prediction.probabilities), 1.0)
        self.assertGreaterEqual(prediction.level, 1)
        self.assertLessEqual(prediction.level, 5)

    def test_complex_task_scores_higher_than_simple_task(self) -> None:
        simple = self.model.predict("What is the capital of Japan?")
        complex_request = self.model.predict(
            "Debug a distributed race condition, prove the root cause, compare "
            "recovery strategies, and design adversarial verification tests."
        )
        self.assertGreater(complex_request.expected_level, simple.expected_level)

    def test_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.npz"
            self.model.save(path)
            loaded = NaiveBayesComplexityModel.load(path)
            before = self.model.predict("Explain this algorithm step by step.")
            after = loaded.predict("Explain this algorithm step by step.")
            self.assertEqual(before.level, after.level)
            np.testing.assert_allclose(before.probabilities, after.probabilities)

    def test_hashed_features_are_stable(self) -> None:
        vectorizer = HashedTextVectorizer(2_048)
        first = vectorizer.transform_one("User: explain recursion")
        second = vectorizer.transform_one("User: explain recursion")
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class LearnedRouterTests(unittest.TestCase):
    def test_hybrid_router_uses_learned_complexity(self) -> None:
        router = ModelRouter.from_artifact(classifier_mode="hybrid")
        decision = router.route(
            RoutingRequest(
                "Debug a distributed race condition, identify the root cause, "
                "compare recovery strategies, and design verification tests.",
                expected_output_tokens=1_200,
            )
        )
        self.assertEqual(decision.model_id, "capable")
        self.assertEqual(decision.features.classifier_source, "hybrid")
        self.assertGreaterEqual(decision.features.minimum_model_tier, 2)
        self.assertEqual(len(decision.features.level_probabilities), 5)

    def test_simple_question_can_use_efficient_target(self) -> None:
        router = ModelRouter.from_artifact(classifier_mode="hybrid")
        decision = router.route(
            RoutingRequest("What is the capital of Japan?", expected_output_tokens=50)
        )
        self.assertEqual(decision.model_id, "efficient")

    def test_conservative_policy_is_not_lower_than_argmax(self) -> None:
        prompt = "Compare two database indexes and explain the trade-offs."
        model = NaiveBayesComplexityModel.load()
        argmax = model.predict(prompt, decision_policy="argmax")
        conservative = model.predict(
            prompt,
            decision_policy="conservative",
            underroute_tolerance=0.20,
        )
        self.assertGreaterEqual(conservative.level, argmax.level)

    def test_tier_risk_policy_bounds_posterior_underroute_probability(self) -> None:
        model = NaiveBayesComplexityModel.load()
        prediction = model.predict(
            "Compare database indexes and explain the operational trade-offs.",
            decision_policy="tier_risk",
            underroute_tolerance=0.15,
        )
        self.assertLessEqual(prediction.tier_underroute_probability, 0.15)
        self.assertIn(prediction.minimum_tier, {1, 2, 3})

    def test_router_exposes_classifier_uncertainty(self) -> None:
        decision = ModelRouter.from_artifact().route(
            RoutingRequest("Compare database indexes and their trade-offs.")
        )
        self.assertIsNotNone(decision.features.classifier_entropy)
        self.assertIsNotNone(decision.features.tier_underroute_probability)


class EvaluationTests(unittest.TestCase):
    def test_borderline_audit_suggestion_can_supply_soft_label_evidence(self) -> None:
        row = DatasetRow(
            prompt="Compare both options.",
            level=2,
            category="reasoning",
            status="borderline",
            confidence=1.0,
            suggested_level=3,
            row_id=1,
        )
        weights = _label_weights(
            row,
            TrainingConfig(borderline_suggested_weight=0.25),
        )
        self.assertAlmostEqual(weights[2], 0.4875)
        self.assertAlmostEqual(weights[3], 0.1625)
        self.assertAlmostEqual(sum(weights.values()), 0.65)

    def test_split_is_whitespace_and_case_stable(self) -> None:
        first = split_for_prompt("User: Explain recursion")
        second = split_for_prompt("  user:   explain RECURSION  ")
        self.assertEqual(first, second)
        self.assertEqual(normalize_prompt(" A  test\nvalue "), "a test value")

    def test_metrics_distinguish_under_and_over_routing(self) -> None:
        metrics = classification_metrics([1, 3, 5], [2, 2, 5])
        self.assertAlmostEqual(metrics["under_level_rate"], 1 / 3, places=6)
        self.assertAlmostEqual(metrics["over_level_rate"], 1 / 3, places=6)
        self.assertAlmostEqual(metrics["within_one_level_rate"], 1.0)

    def test_metrics_accept_numpy_label_arrays(self) -> None:
        metrics = classification_metrics(np.asarray([1, 2]), [1, 2])
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
