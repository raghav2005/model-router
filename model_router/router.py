from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from .catalog import load_catalog
from .classifier import (
    DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD,
    classify_request,
    classify_request_with_model,
)
from .learned import DecisionPolicy, NaiveBayesComplexityModel
from .types import CandidateScore, ModelProfile, RouteDecision, RoutingRequest

WEIGHTS = {
    "balanced": (0.55, 0.25, 0.20),
    "cost": (0.35, 0.50, 0.15),
    "quality": (0.90, 0.05, 0.05),
    "latency": (0.35, 0.10, 0.55),
}
POLICY_VERSION = "hybrid-utility-policy-v4"


class NoEligibleModel(RuntimeError):
    pass


class ModelRouter:
    """Hard constraints followed by a transparent multi-objective score."""

    def __init__(
        self,
        models: list[ModelProfile] | None = None,
        *,
        complexity_model: NaiveBayesComplexityModel | None = None,
        classifier_mode: Literal["heuristic", "learned", "hybrid"] = "heuristic",
        decision_policy: DecisionPolicy = "adaptive",
        underroute_tolerance: float = 0.20,
        adaptive_confidence_threshold: float = DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD,
        require_measured_quality: bool = False,
        require_measured_latency: bool = False,
    ) -> None:
        self.models = models or load_catalog()
        if classifier_mode != "heuristic" and complexity_model is None:
            raise ValueError(
                f"classifier_mode={classifier_mode!r} requires a learned artifact"
            )
        self.complexity_model = complexity_model
        self.classifier_mode = classifier_mode
        self.decision_policy = decision_policy
        if not 0 <= underroute_tolerance < 1:
            raise ValueError("underroute_tolerance must be in [0, 1)")
        if not 0 <= adaptive_confidence_threshold <= 1:
            raise ValueError("adaptive_confidence_threshold must be in [0, 1]")
        self.underroute_tolerance = underroute_tolerance
        self.adaptive_confidence_threshold = adaptive_confidence_threshold
        self.require_measured_quality = require_measured_quality
        self.require_measured_latency = require_measured_latency

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path | None = None,
        *,
        models: list[ModelProfile] | None = None,
        classifier_mode: Literal["learned", "hybrid"] = "hybrid",
        decision_policy: DecisionPolicy = "adaptive",
        underroute_tolerance: float = 0.20,
        adaptive_confidence_threshold: float = DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD,
        require_measured_quality: bool = False,
        require_measured_latency: bool = False,
    ) -> ModelRouter:
        return cls(
            models,
            complexity_model=NaiveBayesComplexityModel.load(artifact_path),
            classifier_mode=classifier_mode,
            decision_policy=decision_policy,
            underroute_tolerance=underroute_tolerance,
            adaptive_confidence_threshold=adaptive_confidence_threshold,
            require_measured_quality=require_measured_quality,
            require_measured_latency=require_measured_latency,
        )

    @staticmethod
    def _predicted_quality(
        model: ModelProfile, use_case: str, complexity: float
    ) -> float:
        domain_skill = model.skills.get(use_case, model.skills["general_qa"])
        reasoning_skill = model.skills.get("reasoning", domain_skill)
        difficulty_penalty = complexity * (1.0 - reasoning_skill) * 0.75
        return max(0.0, min(1.0, domain_skill - difficulty_penalty))

    def route(self, request: RoutingRequest) -> RouteDecision:
        if request.priority not in WEIGHTS:
            raise ValueError(f"Unknown priority: {request.priority}")
        if self.classifier_mode == "heuristic":
            features = classify_request(request)
        else:
            assert self.complexity_model is not None
            features = classify_request_with_model(
                request,
                self.complexity_model,
                mode=self.classifier_mode,
                decision_policy=self.decision_policy,
                underroute_tolerance=self.underroute_tolerance,
                adaptive_confidence_threshold=self.adaptive_confidence_threshold,
            )
        evaluated: list[tuple[ModelProfile, CandidateScore]] = []

        for model in self.models:
            rejection_reasons: list[str] = []
            cost = model.estimate_cost(
                features.input_tokens,
                features.expected_output_tokens,
                cached_input_tokens=request.cached_input_tokens,
                cache_write_tokens=request.cache_write_tokens,
            )
            if not model.enabled or model.health < 0.8:
                rejection_reasons.append("model disabled or unhealthy")
            if request.allowed_model_ids and model.id not in request.allowed_model_ids:
                rejection_reasons.append("model is not permitted by request policy")
            if (
                request.allowed_providers
                and model.provider not in request.allowed_providers
            ):
                rejection_reasons.append("provider is not permitted by request policy")
            if model.tier < features.minimum_model_tier:
                rejection_reasons.append(
                    f"model tier below required tier {features.minimum_model_tier}"
                )
            if (
                features.input_tokens + features.expected_output_tokens
                > model.context_window
            ):
                rejection_reasons.append("context window exceeded")
            if features.expected_output_tokens > model.max_output_tokens:
                rejection_reasons.append("requested output exceeds model maximum")
            missing = features.inferred_capabilities - model.capabilities
            if missing:
                rejection_reasons.append(
                    f"missing capabilities: {', '.join(sorted(missing))}"
                )
            if request.max_cost_usd is not None and cost > request.max_cost_usd:
                rejection_reasons.append("estimated cost exceeds request budget")
            if (
                request.max_latency_ms is not None
                and model.latency_evidence != "workload_measured"
            ):
                rejection_reasons.append(
                    "latency SLA requested but model latency is not workload-measured"
                )
            elif (
                request.max_latency_ms is not None
                and model.latency_p95_ms > request.max_latency_ms
            ):
                rejection_reasons.append("p95 latency exceeds request SLA")
            if (
                self.require_measured_latency
                and model.latency_evidence != "workload_measured"
            ):
                rejection_reasons.append("production policy requires measured latency")
            if (
                self.require_measured_quality
                and model.quality_evidence != "workload_measured"
            ):
                rejection_reasons.append("production policy requires measured quality")

            quality = self._predicted_quality(
                model, features.use_case, features.complexity
            )
            if quality < features.quality_floor:
                rejection_reasons.append("predicted quality below request floor")
            evaluated.append(
                (
                    model,
                    CandidateScore(
                        model_id=model.id,
                        provider=model.provider,
                        provider_model=model.provider_model,
                        utility=float("-inf"),
                        predicted_quality=quality,
                        estimated_cost_usd=cost,
                        latency_p95_ms=model.latency_p95_ms,
                        pricing_source=model.pricing_source,
                        quality_evidence=model.quality_evidence,
                        latency_evidence=model.latency_evidence,
                        eligible=not rejection_reasons,
                        rejection_reasons=tuple(rejection_reasons),
                    ),
                )
            )

        eligible = [(model, score) for model, score in evaluated if score.eligible]
        if not eligible:
            details = "; ".join(
                f"{score.model_id}: {', '.join(score.rejection_reasons)}"
                for _, score in evaluated
            )
            raise NoEligibleModel(f"No eligible model. {details}")

        max_cost = max(score.estimated_cost_usd for _, score in eligible) or 1.0
        max_latency = max(score.latency_p95_ms for _, score in eligible) or 1
        quality_weight, cost_weight, latency_weight = WEIGHTS[request.priority]
        scored: list[tuple[ModelProfile, CandidateScore]] = []

        for model, score in evaluated:
            if not score.eligible:
                scored.append((model, score))
                continue
            satisfaction = min(1.0, score.predicted_quality / features.quality_floor)
            # Adequacy matters most, while quality-first callers still benefit from headroom.
            quality_component = (
                score.predicted_quality
                if request.priority == "quality"
                else 0.80 * satisfaction + 0.20 * score.predicted_quality
            )
            cost_component = 1.0 - score.estimated_cost_usd / max_cost
            latency_component = 1.0 - score.latency_p95_ms / max_latency
            health_bonus = max(0.0, model.health - 0.8) * 0.05
            utility = (
                quality_weight * quality_component
                + cost_weight * cost_component
                + latency_weight * latency_component
                + health_bonus
            )
            scored.append((model, replace(score, utility=utility)))

        ranked = sorted(
            (item for item in scored if item[1].eligible),
            key=lambda item: (item[1].utility, -item[1].estimated_cost_usd),
            reverse=True,
        )
        selected_model, selected = ranked[0]
        next_best_utility = ranked[1][1].utility if len(ranked) > 1 else 0.0
        utility_confidence = min(
            1.0, 0.5 + max(0.0, selected.utility - next_best_utility)
        )
        confidence = (
            min(utility_confidence, features.classifier_confidence)
            if features.classifier_confidence is not None
            else utility_confidence
        )
        reasons = (
            f"classified by {features.classifier_source} as {features.use_case}, "
            f"level {features.complexity_level}, complexity {features.complexity:.2f}",
            f"{request.priority} policy utility {selected.utility:.3f}",
            f"predicted quality {selected.predicted_quality:.2f} vs floor {features.quality_floor:.2f}",
            f"estimated request cost ${selected.estimated_cost_usd:.6f}",
            f"quality evidence: {selected.quality_evidence}",
            f"latency evidence: {selected.latency_evidence}",
            (
                "posterior tier under-route probability "
                f"{features.tier_underroute_probability:.3f}"
                if features.tier_underroute_probability is not None
                else "posterior tier risk unavailable for heuristic classification"
            ),
        )
        all_scores = tuple(
            score
            for _, score in sorted(
                scored,
                key=lambda item: (item[1].eligible, item[1].utility),
                reverse=True,
            )
        )
        return RouteDecision(
            policy_version=POLICY_VERSION,
            model_id=selected_model.id,
            features=features,
            estimated_cost_usd=selected.estimated_cost_usd,
            latency_p95_ms=selected.latency_p95_ms,
            predicted_quality=selected.predicted_quality,
            confidence=confidence,
            fallback_models=tuple(model.id for model, _ in ranked[1:]),
            reasons=reasons,
            candidates=all_scores,
        )
