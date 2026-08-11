from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Literal

from .learned import DecisionPolicy, NaiveBayesComplexityModel
from .types import RequestFeatures, RoutingRequest


CODE_TERMS = re.compile(
    r"\b(code|coding|function|class|api|sql|regex|debug|bug|refactor|repository|"
    r"unit tests?|typescript|javascript|python|java|rust|golang|compiler|database|"
    r"deploy|architecture|distributed system|migration)\b",
    re.IGNORECASE,
)
REASONING_TERMS = re.compile(
    r"\b(prove|proof|derive|reason|step[- ]by[- ]step|trade-?offs?|optimi[sz]e|"
    r"evaluate|compare|analy[sz]e|root cause|strategy|probability|theorem)\b",
    re.IGNORECASE,
)
HIGH_STAKES_TERMS = re.compile(
    r"\b(medical|diagnosis|patient|legal advice|lawsuit|compliance|investment advice|"
    r"portfolio|vulnerability|exploit|security audit|production outage)\b",
    re.IGNORECASE,
)
CURRENT_INFO_TERMS = re.compile(
    r"\b(latest|current|today|this week|news|price|pricing|schedule|weather)\b",
    re.IGNORECASE,
)
STRUCTURED_TERMS = re.compile(
    r"\b(valid json|json schema|structured output|extract fields?|return json)\b",
    re.IGNORECASE,
)
MODERATE_CODE_TERMS = re.compile(
    r"\b(unit tests?|refactor|testability|error handling|indexes?|query plan)\b",
    re.IGNORECASE,
)
ADVANCED_SYSTEM_TERMS = re.compile(
    r"\b(distributed|race condition|multi-region|migration plan|security audit|"
    r"production|concurrency|authentication design|failure recovery)\b",
    re.IGNORECASE,
)
DEEP_REASONING_TERMS = re.compile(
    r"\b(prove|proof|derive|derivation|theorem|hidden assumptions?|alternative derivations?)\b",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    """A conservative tokenizer-free approximation suitable for routing."""
    return max(1, math.ceil(len(text) / 4))


def complexity_to_level(complexity: float) -> int:
    if complexity < 0.20:
        return 1
    if complexity < 0.40:
        return 2
    if complexity < 0.60:
        return 3
    if complexity < 0.80:
        return 4
    return 5


def level_to_complexity(level: float) -> float:
    return max(0.12, min(0.95, 0.12 + (level - 1.0) * (0.83 / 4.0)))


def quality_floor_for(complexity: float, risk: str) -> float:
    floor = 0.60 + 0.30 * complexity + (0.03 if risk == "high" else 0.0)
    # The catalogue's capable role must remain eligible at maximum complexity.
    return min(0.90, floor)


def classify_request(request: RoutingRequest) -> RequestFeatures:
    text = request.prompt
    signals: list[str] = []
    inferred_capabilities = set(request.required_capabilities)

    has_code = bool(CODE_TERMS.search(text) or "```" in text)
    has_reasoning = bool(REASONING_TERMS.search(text))
    high_stakes = bool(HIGH_STAKES_TERMS.search(text))

    if request.use_case:
        use_case = request.use_case
        signals.append("caller supplied use case")
    elif has_code:
        use_case = "coding"
        signals.append("coding vocabulary or code block")
    elif has_reasoning:
        use_case = "reasoning"
        signals.append("multi-step reasoning vocabulary")
    else:
        use_case = "general_qa"
        signals.append("general question/answer pattern")

    input_tokens = request.input_tokens or estimate_tokens(text)
    complexity = 0.12
    if input_tokens > 250:
        complexity += 0.12
        signals.append("long prompt")
    if input_tokens > 1_000:
        complexity += 0.13
        signals.append("very long prompt")
    if has_code:
        complexity += 0.13
    if has_reasoning:
        complexity += 0.22
    if MODERATE_CODE_TERMS.search(text):
        complexity += 0.12
        signals.append("non-trivial implementation or verification")
    if ADVANCED_SYSTEM_TERMS.search(text):
        complexity += 0.20
        signals.append("advanced system-level task")
    if DEEP_REASONING_TERMS.search(text):
        complexity += 0.20
        signals.append("deep formal reasoning")
    if "```" in text:
        complexity += 0.08
        signals.append("code block")
    if (
        len(
            re.findall(
                r"\b(and|also|then|finally|versus|vs\.?|trade-?off)\b", text, re.I
            )
        )
        >= 2
    ):
        complexity += 0.12
        signals.append("multiple constraints or subtasks")
    if request.expected_output_tokens > 1_000:
        complexity += 0.10
        signals.append("large requested output")
    if high_stakes:
        complexity += 0.18
        signals.append("high-stakes domain")
    complexity = min(1.0, complexity)

    if CURRENT_INFO_TERMS.search(text):
        inferred_capabilities.add("web")
        signals.append("fresh information requested")
    if STRUCTURED_TERMS.search(text):
        inferred_capabilities.add("structured_output")
        signals.append("structured output requested")
    if has_code and re.search(r"\b(run|execute|test|repository|files?)\b", text, re.I):
        inferred_capabilities.add("tools")
        signals.append("tool-assisted coding request")

    risk = "high" if high_stakes else "normal"
    complexity_level = complexity_to_level(complexity)
    quality_floor = quality_floor_for(complexity, risk)

    return RequestFeatures(
        use_case=use_case,
        complexity=complexity,
        complexity_level=complexity_level,
        quality_floor=quality_floor,
        minimum_model_tier=1,
        risk=risk,
        input_tokens=input_tokens,
        expected_output_tokens=request.expected_output_tokens,
        inferred_capabilities=frozenset(inferred_capabilities),
        signals=tuple(signals),
    )


def classify_request_with_model(
    request: RoutingRequest,
    model: NaiveBayesComplexityModel,
    *,
    mode: Literal["learned", "hybrid"] = "hybrid",
    decision_policy: DecisionPolicy = "argmax",
    underroute_tolerance: float = 0.20,
) -> RequestFeatures:
    heuristic = classify_request(request)
    prediction = model.predict(
        request.prompt,
        decision_policy=decision_policy,
        underroute_tolerance=underroute_tolerance,
    )
    learned_complexity = level_to_complexity(prediction.expected_level)
    if mode == "learned":
        complexity = learned_complexity
    elif mode == "hybrid":
        complexity = 0.75 * learned_complexity + 0.25 * heuristic.complexity
        if heuristic.risk == "high":
            complexity = max(complexity, heuristic.complexity)
    else:
        raise ValueError(f"Unknown learned classifier mode: {mode}")
    complexity = max(0.0, min(1.0, complexity))
    minimum_tier = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}[prediction.level]
    dataset_hash = str(model.metadata.get("training_dataset_sha256", "unknown"))
    model_version = f"complexity-router-nb-v1:{dataset_hash[:12]}"
    return replace(
        heuristic,
        complexity=complexity,
        complexity_level=prediction.level,
        quality_floor=quality_floor_for(complexity, heuristic.risk),
        minimum_model_tier=minimum_tier,
        signals=heuristic.signals
        + (
            f"learned complexity level {prediction.level} using {decision_policy}",
            f"learned classifier confidence {prediction.confidence:.2f}",
        ),
        classifier_source=mode,
        classifier_model_version=model_version,
        classifier_confidence=prediction.confidence,
        level_probabilities=prediction.probabilities,
    )
