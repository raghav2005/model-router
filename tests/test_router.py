from __future__ import annotations

import unittest

from model_router.catalog import load_catalog
from model_router.classifier import classify_request, estimate_tokens
from model_router.router import ModelRouter, NoEligibleModel
from model_router.types import ModelProfile, RoutingRequest


class ClassifierTests(unittest.TestCase):
    def test_token_estimate_is_never_zero(self) -> None:
        self.assertEqual(estimate_tokens(""), 1)

    def test_infers_coding_and_tools(self) -> None:
        features = classify_request(
            RoutingRequest("Debug this repository and run the unit tests")
        )
        self.assertEqual(features.use_case, "coding")
        self.assertIn("tools", features.inferred_capabilities)

    def test_infers_web_for_current_information(self) -> None:
        features = classify_request(RoutingRequest("What is the current price?"))
        self.assertIn("web", features.inferred_capabilities)


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = ModelRouter()

    def test_simple_question_routes_to_efficient_tier(self) -> None:
        decision = self.router.route(
            RoutingRequest("What is the capital of Japan?", expected_output_tokens=40)
        )
        self.assertEqual(decision.model_id, "efficient")

    def test_complex_coding_routes_to_flagship(self) -> None:
        decision = self.router.route(
            RoutingRequest(
                "Debug a distributed race condition, analyze the root cause, compare retry trade-offs, and propose verification tests.",
                expected_output_tokens=1800,
            )
        )
        self.assertEqual(decision.model_id, "capable")

    def test_budget_is_hard_constraint(self) -> None:
        decision = self.router.route(
            RoutingRequest(
                "Explain recursion", expected_output_tokens=100, max_cost_usd=0.001
            )
        )
        self.assertEqual(decision.model_id, "efficient")

    def test_impossible_capability_raises(self) -> None:
        with self.assertRaises(NoEligibleModel):
            self.router.route(
                RoutingRequest(
                    "Create audio", required_capabilities=frozenset({"audio"})
                )
            )

    def test_context_limit_is_enforced(self) -> None:
        with self.assertRaises(NoEligibleModel):
            self.router.route(RoutingRequest("x", input_tokens=1_100_000))

    def test_unhealthy_model_is_excluded(self) -> None:
        models = load_catalog()
        unhealthy = [
            ModelProfile(**{**model.__dict__, "health": 0.5})
            if model.id == "efficient"
            else model
            for model in models
        ]
        decision = ModelRouter(unhealthy).route(
            RoutingRequest("Hello", expected_output_tokens=20)
        )
        self.assertNotEqual(decision.model_id, "efficient")

    def test_cost_formula(self) -> None:
        efficient = next(model for model in load_catalog() if model.id == "efficient")
        self.assertAlmostEqual(efficient.estimate_cost(1_000_000, 1_000_000), 7.0)


if __name__ == "__main__":
    unittest.main()
