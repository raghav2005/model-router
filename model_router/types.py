from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Priority = Literal["balanced", "cost", "quality", "latency"]


@dataclass(frozen=True)
class ModelProfile:
    id: str
    switchyard_target: str
    provider: str
    provider_model: str
    tier: int
    input_price_per_million: float
    cached_input_price_per_million: float
    cache_write_price_per_million: float
    output_price_per_million: float
    latency_p95_ms: int
    context_window: int
    max_output_tokens: int
    capabilities: frozenset[str]
    skills: dict[str, float]
    pricing_source: str
    pricing_as_of: str
    quality_evidence: str
    latency_evidence: str
    long_context_threshold_tokens: int | None = None
    long_context_input_multiplier: float = 1.0
    long_context_output_multiplier: float = 1.0
    enabled: bool = True
    health: float = 1.0

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        if (
            min(input_tokens, output_tokens, cached_input_tokens, cache_write_tokens)
            < 0
        ):
            raise ValueError("token counts cannot be negative")
        if cached_input_tokens + cache_write_tokens > input_tokens:
            raise ValueError("cached and cache-write tokens cannot exceed input tokens")
        standard_input_tokens = input_tokens - cached_input_tokens - cache_write_tokens
        long_context = (
            self.long_context_threshold_tokens is not None
            and input_tokens > self.long_context_threshold_tokens
        )
        input_multiplier = self.long_context_input_multiplier if long_context else 1.0
        output_multiplier = self.long_context_output_multiplier if long_context else 1.0
        return (
            input_multiplier
            * (
                standard_input_tokens * self.input_price_per_million
                + cached_input_tokens * self.cached_input_price_per_million
                + cache_write_tokens * self.cache_write_price_per_million
            )
            + output_multiplier * output_tokens * self.output_price_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class RoutingRequest:
    prompt: str
    input_tokens: int | None = None
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    expected_output_tokens: int = 500
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    priority: Priority = "balanced"
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None
    use_case: str | None = None


@dataclass(frozen=True)
class RequestFeatures:
    use_case: str
    complexity: float
    complexity_level: int
    quality_floor: float
    minimum_model_tier: int
    risk: str
    input_tokens: int
    expected_output_tokens: int
    inferred_capabilities: frozenset[str]
    signals: tuple[str, ...]
    classifier_source: str = "heuristic"
    classifier_model_version: str | None = None
    classifier_confidence: float | None = None
    level_probabilities: tuple[float, ...] = ()


@dataclass(frozen=True)
class CandidateScore:
    model_id: str
    provider: str
    provider_model: str
    utility: float
    predicted_quality: float
    estimated_cost_usd: float
    latency_p95_ms: int
    pricing_source: str
    quality_evidence: str
    latency_evidence: str
    eligible: bool
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteDecision:
    policy_version: str
    model_id: str
    features: RequestFeatures
    estimated_cost_usd: float
    latency_p95_ms: int
    predicted_quality: float
    confidence: float
    fallback_models: tuple[str, ...]
    reasons: tuple[str, ...]
    candidates: tuple[CandidateScore, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "model": self.model_id,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "latency_p95_ms": self.latency_p95_ms,
            "predicted_quality": round(self.predicted_quality, 4),
            "confidence": round(self.confidence, 4),
            "fallback_models": list(self.fallback_models),
            "reasons": list(self.reasons),
            "features": {
                "use_case": self.features.use_case,
                "complexity": round(self.features.complexity, 4),
                "complexity_level": self.features.complexity_level,
                "quality_floor": round(self.features.quality_floor, 4),
                "minimum_model_tier": self.features.minimum_model_tier,
                "risk": self.features.risk,
                "classifier_source": self.features.classifier_source,
                "classifier_model_version": self.features.classifier_model_version,
                "classifier_confidence": (
                    round(self.features.classifier_confidence, 4)
                    if self.features.classifier_confidence is not None
                    else None
                ),
                "level_probabilities": {
                    str(index + 1): round(probability, 5)
                    for index, probability in enumerate(
                        self.features.level_probabilities
                    )
                },
                "input_tokens": self.features.input_tokens,
                "expected_output_tokens": self.features.expected_output_tokens,
                "inferred_capabilities": sorted(self.features.inferred_capabilities),
                "signals": list(self.features.signals),
            },
            "candidates": [
                {
                    "model": item.model_id,
                    "provider": item.provider,
                    "provider_model": item.provider_model,
                    "eligible": item.eligible,
                    "utility": round(item.utility, 4) if item.eligible else None,
                    "predicted_quality": round(item.predicted_quality, 4),
                    "estimated_cost_usd": round(item.estimated_cost_usd, 8),
                    "latency_p95_ms": item.latency_p95_ms,
                    "pricing_source": item.pricing_source,
                    "quality_evidence": item.quality_evidence,
                    "latency_evidence": item.latency_evidence,
                    "rejection_reasons": list(item.rejection_reasons),
                }
                for item in self.candidates
            ],
        }
