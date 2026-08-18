from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .adversarial import AdversarialCase, build_adversarial_cases
from .catalog import load_catalog
from .normalization import TRANSFORMATIONS, transform_prompt
from .router import ModelRouter, NoEligibleModel
from .types import Priority, RoutingRequest


@dataclass(frozen=True)
class MetamorphicCase:
    id: str
    source_case_id: str
    source_category: str
    transformation: str
    prompt: str
    expected_tier: int
    expected_use_case: str
    priority: Priority
    required_capabilities: frozenset[str]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["required_capabilities"] = sorted(self.required_capabilities)
        return value


def build_metamorphic_cases() -> list[MetamorphicCase]:
    cases = [
        MetamorphicCase(
            id=f"{source.id}--{transformation}",
            source_case_id=source.id,
            source_category=source.category,
            transformation=transformation,
            prompt=transform_prompt(source.prompt, transformation),
            expected_tier=source.expected_tier,
            expected_use_case=source.expected_use_case,
            priority=source.priority,
            required_capabilities=source.required_capabilities,
        )
        for source in build_adversarial_cases()
        for transformation in TRANSFORMATIONS
    ]
    identifiers = [case.id for case in cases]
    prompts = [case.prompt.strip().lower() for case in cases]
    if len(identifiers) != len(set(identifiers)) or len(prompts) != len(set(prompts)):
        raise AssertionError("metamorphic cases must be unique")
    return cases


def _route(router: ModelRouter, case: AdversarialCase | MetamorphicCase):
    return router.route(
        RoutingRequest(
            case.prompt,
            expected_output_tokens=500,
            priority=case.priority,
            required_capabilities=case.required_capabilities,
        )
    )


def evaluate_metamorphic_suite(
    *, artifact_path: str | Path | None = None
) -> dict[str, object]:
    source_cases = build_adversarial_cases()
    cases = build_metamorphic_cases()
    catalog = load_catalog()
    tier_by_model = {model.id: model.tier for model in catalog}
    routers = {
        "hybrid_argmax": ModelRouter.from_artifact(
            artifact_path, models=catalog, decision_policy="argmax"
        ),
        "hybrid_adaptive_p15": ModelRouter.from_artifact(
            artifact_path,
            models=catalog,
            decision_policy="adaptive",
            underroute_tolerance=0.15,
        ),
        "hybrid_tier_risk_p15": ModelRouter.from_artifact(
            artifact_path,
            models=catalog,
            decision_policy="tier_risk",
            underroute_tolerance=0.15,
        ),
    }
    policies: dict[str, object] = {}
    for policy_name, router in routers.items():
        baselines: dict[str, tuple[str, int]] = {}
        for source in source_cases:
            try:
                decision = _route(router, source)
                baselines[source.id] = (
                    decision.model_id,
                    tier_by_model[decision.model_id],
                )
            except NoEligibleModel:
                continue

        under = 0
        over = 0
        route_errors = 0
        use_case_correct = 0
        invariant_tier = 0
        invariant_model = 0
        comparable = 0
        total_cost = 0.0
        by_transform: dict[str, Counter[str]] = defaultdict(Counter)
        failure_examples: list[dict[str, object]] = []
        for case in cases:
            try:
                decision = _route(router, case)
            except NoEligibleModel:
                route_errors += 1
                under += 1
                by_transform[case.transformation]["route_error"] += 1
                continue
            selected_tier = tier_by_model[decision.model_id]
            total_cost += decision.estimated_cost_usd
            use_case_correct += decision.features.use_case == case.expected_use_case
            outcome = (
                "exact"
                if selected_tier == case.expected_tier
                else "under"
                if selected_tier < case.expected_tier
                else "over"
            )
            under += outcome == "under"
            over += outcome == "over"
            by_transform[case.transformation][outcome] += 1
            baseline = baselines.get(case.source_case_id)
            if baseline is not None:
                comparable += 1
                invariant_model += decision.model_id == baseline[0]
                invariant_tier += selected_tier == baseline[1]
            if (
                outcome != "exact"
                or baseline is None
                or selected_tier != baseline[1]
                or decision.features.use_case != case.expected_use_case
            ) and len(failure_examples) < 30:
                failure_examples.append(
                    {
                        "id": case.id,
                        "source_case_id": case.source_case_id,
                        "transformation": case.transformation,
                        "expected_tier": case.expected_tier,
                        "selected_tier": selected_tier,
                        "baseline_tier": baseline[1] if baseline is not None else None,
                        "expected_use_case": case.expected_use_case,
                        "selected_use_case": decision.features.use_case,
                        "classifier_confidence": decision.features.classifier_confidence,
                        "signals": list(decision.features.signals),
                    }
                )
        count = len(cases)
        policies[policy_name] = {
            "count": count,
            "route_errors": route_errors,
            "tier_accuracy": round((count - under - over) / count, 6),
            "tier_underroute_rate": round(under / count, 6),
            "tier_overroute_rate": round(over / count, 6),
            "use_case_accuracy": round(use_case_correct / count, 6),
            "tier_invariance_rate": (
                round(invariant_tier / comparable, 6) if comparable else None
            ),
            "model_invariance_rate": (
                round(invariant_model / comparable, 6) if comparable else None
            ),
            "comparable_to_baseline": comparable,
            "average_estimated_cost_usd": round(total_cost / count, 8),
            "transformations": {
                name: dict(sorted(values.items()))
                for name, values in sorted(by_transform.items())
            },
            "failure_examples": failure_examples,
        }
    return {
        "schema_version": "model-router-metamorphic-eval-v1",
        "case_count": len(cases),
        "source_case_count": len(source_cases),
        "transformations": list(TRANSFORMATIONS),
        "policies": policies,
        "limitations": [
            "Every case is a deterministic transformation of a synthetic design test.",
            "Invariance catches sensitivity to irrelevant presentation changes; it does not prove answer quality.",
            "This suite is a regression gate and must never be counted as independent workload evidence.",
        ],
    }


def write_cases(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(case.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for case in build_metamorphic_cases()
        ),
        encoding="utf-8",
    )


def write_report(path: str | Path, report: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
