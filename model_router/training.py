from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from .catalog import catalog_sha256, load_catalog, load_catalog_document
from .classifier import classify_request
from .evaluation import (
    LEVEL_TO_TIER,
    choose_view_ensemble,
    classification_metrics,
    relative_cost_saving,
    slice_metrics,
)
from .learned import (
    DEFAULT_FEATURE_DIMENSION,
    MODEL_SCHEMA_VERSION,
    HashedTextVectorizer,
    NaiveBayesComplexityModel,
    final_user_turn,
    normalize_prompt,
)
from .router import POLICY_VERSION
from .types import RoutingRequest


@dataclass(frozen=True)
class DatasetRow:
    prompt: str
    level: int
    category: str
    status: str
    confidence: float
    row_id: int | None


@dataclass(frozen=True)
class TrainingConfig:
    feature_dimension: int = DEFAULT_FEATURE_DIMENSION
    smoothing: float = 1.0
    borderline_weight: float = 0.65
    appropriate_weight: float = 1.0
    train_percentage: int = 80
    validation_percentage: int = 10
    class_prior: str = "uniform"
    augmentation_copies_per_train_row: int = 1
    augmentation_weight: float = 0.5

    def validate(self) -> None:
        if self.feature_dimension < 1_024:
            raise ValueError("feature_dimension must be at least 1024")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be greater than zero")
        if not 0 < self.borderline_weight <= 1:
            raise ValueError("borderline_weight must be in (0, 1]")
        if not 0 < self.appropriate_weight <= 1:
            raise ValueError("appropriate_weight must be in (0, 1]")
        if self.train_percentage + self.validation_percentage >= 100:
            raise ValueError("the split must leave a non-empty test percentage")
        if self.class_prior not in {"uniform", "empirical"}:
            raise ValueError("class_prior must be uniform or empirical")
        if self.augmentation_copies_per_train_row < 0:
            raise ValueError("augmentation_copies_per_train_row cannot be negative")
        if not 0 < self.augmentation_weight <= 1:
            raise ValueError("augmentation_weight must be in (0, 1]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_prompt(
    prompt: str,
    *,
    train_percentage: int = 80,
    validation_percentage: int = 10,
) -> str:
    normalized = normalize_prompt(prompt)
    bucket = int(hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < train_percentage:
        return "train"
    if bucket < train_percentage + validation_percentage:
        return "validation"
    return "test"


def read_dataset(path: str | Path) -> Iterator[DatasetRow]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {dataset_path.name}:{line_number}"
                ) from error
            try:
                level = int(raw["complexity_level"])
                prompt = str(raw["prompt"]).strip()
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid routing row at {dataset_path.name}:{line_number}"
                ) from error
            if not prompt or level not in range(1, 6):
                raise ValueError(
                    f"Invalid prompt or level at {dataset_path.name}:{line_number}"
                )
            if raw.get("complexity_valid_for_training") is False:
                continue
            yield DatasetRow(
                prompt=prompt,
                level=level,
                category=str(raw.get("category", "unknown")),
                status=str(raw.get("complexity_audit_status", "unreviewed")),
                confidence=float(raw.get("complexity_audit_confidence", 1.0)),
                row_id=int(raw["row_id"]) if "row_id" in raw else None,
            )


def heuristic_level(prompt: str) -> int:
    features = classify_request(RoutingRequest(prompt=prompt))
    if features.complexity < 0.20:
        return 1
    if features.complexity < 0.40:
        return 2
    if features.complexity < 0.60:
        return 3
    if features.complexity < 0.80:
        return 4
    return 5


def _row_weight(row: DatasetRow, config: TrainingConfig) -> float:
    status_weight = (
        config.borderline_weight
        if row.status == "borderline"
        else config.appropriate_weight
    )
    return max(0.05, min(1.0, row.confidence)) * status_weight


def _augmented_prompts(prompt: str, copies: int) -> list[str]:
    if copies == 0:
        return []
    from .normalization import TRANSFORMATIONS, transform_prompt

    offset = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    return [
        transform_prompt(
            prompt, TRANSFORMATIONS[(offset + index) % len(TRANSFORMATIONS)]
        )
        for index in range(copies)
    ]


def _predict_rows(
    model: NaiveBayesComplexityModel,
    rows: list[DatasetRow],
    *,
    decision_policy: str = "argmax",
) -> tuple[list[int], np.ndarray, np.ndarray]:
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for row in rows:
        raw_scores = model.raw_scores(row.prompt)
        scores.append(raw_scores)
        probability = model.probabilities(row.prompt)
        probabilities.append(probability)
        if decision_policy == "conservative":
            prediction = int(np.searchsorted(np.cumsum(probability), 0.80)) + 1
        elif decision_policy == "expected":
            expected = float(np.dot(probability, np.arange(1, 6)))
            prediction = max(1, min(5, int(np.floor(expected + 0.5))))
        else:
            prediction = int(np.argmax(probability)) + 1
        predictions.append(min(5, prediction))
    return predictions, np.vstack(probabilities), np.vstack(scores)


def _evaluate_strategy(
    labels: list[int], predictions: list[int], probabilities: np.ndarray | None = None
) -> dict[str, object]:
    return classification_metrics(labels, predictions, probabilities)


def _audit_dataset(rows: list[DatasetRow]) -> dict[str, object]:
    normalized_counts: Counter[str] = Counter()
    labels_by_prompt: dict[str, set[int]] = defaultdict(set)
    levels: Counter[int] = Counter()
    statuses: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for row in rows:
        normalized = normalize_prompt(row.prompt)
        normalized_counts[normalized] += 1
        labels_by_prompt[normalized].add(row.level)
        levels[row.level] += 1
        statuses[row.status] += 1
        categories[row.category] += 1
    return {
        "rows": len(rows),
        "unique_normalized_prompts": len(normalized_counts),
        "duplicate_rows": sum(count - 1 for count in normalized_counts.values()),
        "conflicting_duplicate_labels": sum(
            len(labels) > 1 for labels in labels_by_prompt.values()
        ),
        "levels": {str(level): levels[level] for level in range(1, 6)},
        "statuses": dict(sorted(statuses.items())),
        "categories": len(categories),
    }


def _external_context_evaluation(
    model: NaiveBayesComplexityModel,
    external_path: Path,
    training_prompts: set[str],
    adaptive_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    rows = list(read_dataset(external_path))
    novel = [
        row for row in rows if normalize_prompt(row.prompt) not in training_prompts
    ]
    overlap = len(rows) - len(novel)
    result: dict[str, object] = {
        "source_sha256": sha256_file(external_path),
        "source_rows": len(rows),
        "overlap_with_primary_dataset": overlap,
        "novel_rows": len(novel),
        "warning": (
            "This slice is generated from the same synthetic process and is not "
            "a production-distribution benchmark. Overlapping rows were excluded."
        ),
    }
    if novel:
        predictions, probabilities, _ = _predict_rows(model, novel)
        result["metrics"] = classification_metrics(
            [row.level for row in novel], predictions, probabilities
        )
        result["routing_policies"] = _routing_policy_evaluation(
            model, novel, adaptive_policy=adaptive_policy
        )
    return result


def _routing_policy_evaluation(
    model: NaiveBayesComplexityModel,
    rows: list[DatasetRow],
    *,
    adaptive_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    from .router import ModelRouter, NoEligibleModel

    catalog = load_catalog()
    tier_by_model = {profile.id: profile.tier for profile in catalog}
    routers = {
        "hybrid_argmax": ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="hybrid",
            decision_policy="argmax",
        ),
        "hybrid_adaptive_p15": ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="hybrid",
            decision_policy="adaptive",
            underroute_tolerance=0.15,
        ),
        "hybrid_tier_risk_p15": ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="hybrid",
            decision_policy="tier_risk",
            underroute_tolerance=0.15,
        ),
    }
    if adaptive_policy:
        routers["hybrid_adaptive_validation_selected"] = ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="hybrid",
            decision_policy="adaptive",
            underroute_tolerance=float(adaptive_policy["underroute_tolerance"]),
            adaptive_confidence_threshold=float(
                adaptive_policy["confidence_threshold"]
            ),
        )
    results: dict[str, object] = {}
    for name, router in routers.items():
        selected_tiers: list[int] = []
        required_tiers: list[int] = []
        total_cost = 0.0
        errors = 0
        for row in rows:
            try:
                decision = router.route(RoutingRequest(prompt=row.prompt))
            except NoEligibleModel:
                errors += 1
                continue
            selected_tiers.append(tier_by_model[decision.model_id])
            required_tiers.append(LEVEL_TO_TIER[row.level])
            total_cost += decision.estimated_cost_usd
        count = len(rows)
        tier_under = (
            sum(
                selected < required
                for selected, required in zip(
                    selected_tiers, required_tiers, strict=True
                )
            )
            + errors
        )
        tier_over = sum(
            selected > required
            for selected, required in zip(selected_tiers, required_tiers, strict=True)
        )
        tier_exact = count - tier_under - tier_over
        results[name] = {
            "count": count,
            "route_errors": errors,
            "tier_accuracy": round(tier_exact / count, 6),
            "tier_underroute_rate": round(tier_under / count, 6),
            "tier_overroute_rate": round(tier_over / count, 6),
            "average_estimated_cost_usd": round(total_cost / count, 8),
        }
    return results


def _end_to_end_router_evaluation(
    model: NaiveBayesComplexityModel,
    rows: list[DatasetRow],
    *,
    adaptive_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    # Imported here to keep the reusable dataset/evaluation helpers lightweight.
    from .router import ModelRouter, NoEligibleModel

    catalog = load_catalog()
    model_by_id = {profile.id: profile for profile in catalog}
    routers = {
        "heuristic": ModelRouter(catalog),
        "learned_argmax": ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="learned",
            decision_policy="argmax",
        ),
        "hybrid_argmax": ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="hybrid",
            decision_policy="argmax",
        ),
        "hybrid_conservative_p80": ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="hybrid",
            decision_policy="conservative",
            underroute_tolerance=0.20,
        ),
        "hybrid_adaptive_p15": ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="hybrid",
            decision_policy="adaptive",
            underroute_tolerance=0.15,
        ),
        "hybrid_tier_risk_p15": ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="hybrid",
            decision_policy="tier_risk",
            underroute_tolerance=0.15,
        ),
    }
    if adaptive_policy:
        routers["hybrid_adaptive_validation_selected"] = ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="hybrid",
            decision_policy="adaptive",
            underroute_tolerance=float(adaptive_policy["underroute_tolerance"]),
            adaptive_confidence_threshold=float(
                adaptive_policy["confidence_threshold"]
            ),
        )
    results: dict[str, object] = {}
    for name, router in routers.items():
        selected_tiers: list[int] = []
        required_tiers: list[int] = []
        route_counts: Counter[str] = Counter()
        total_cost = 0.0
        total_latency = 0
        errors = 0
        for row in rows:
            request = RoutingRequest(prompt=row.prompt, expected_output_tokens=500)
            try:
                decision = router.route(request)
            except NoEligibleModel:
                errors += 1
                continue
            profile = model_by_id[decision.model_id]
            selected_tiers.append(profile.tier)
            required_tiers.append(LEVEL_TO_TIER[row.level])
            route_counts[decision.model_id] += 1
            total_cost += decision.estimated_cost_usd
            total_latency += decision.latency_p95_ms

        # A no-eligible-model result is a failed route, not silently removed.
        successful = sum(
            selected >= required
            for selected, required in zip(selected_tiers, required_tiers, strict=True)
        )
        under = (
            sum(
                selected < required
                for selected, required in zip(
                    selected_tiers, required_tiers, strict=True
                )
            )
            + errors
        )
        over = sum(
            selected > required
            for selected, required in zip(selected_tiers, required_tiers, strict=True)
        )
        count = len(rows)
        results[name] = {
            "count": count,
            "route_errors": errors,
            "tier_success_proxy": round(successful / count, 6),
            "tier_underroute_rate": round(under / count, 6),
            "tier_overroute_rate": round(over / count, 6),
            "route_counts": dict(sorted(route_counts.items())),
            "average_estimated_cost_usd": round(total_cost / count, 8),
            "average_latency_prior_ms": round(total_latency / count, 3),
        }

    capable = model_by_id["capable"]
    always_capable_cost = sum(
        capable.estimate_cost(max(1, (len(row.prompt) + 3) // 4), 500) for row in rows
    ) / len(rows)
    for metrics in results.values():
        assert isinstance(metrics, dict)
        metrics["estimated_cost_saving_vs_always_capable"] = round(
            1.0 - float(metrics["average_estimated_cost_usd"]) / always_capable_cost,
            6,
        )
    results["always_capable_reference"] = {
        "count": len(rows),
        "route_errors": 0,
        "tier_success_proxy": 1.0,
        "tier_underroute_rate": 0.0,
        "tier_overroute_rate": round(
            sum(3 > LEVEL_TO_TIER[row.level] for row in rows) / len(rows), 6
        ),
        "route_counts": {"capable": len(rows)},
        "average_estimated_cost_usd": round(always_capable_cost, 8),
        "average_latency_prior_ms": capable.latency_p95_ms,
        "estimated_cost_saving_vs_always_capable": 0.0,
    }
    return results


def _tune_adaptive_policy(
    model: NaiveBayesComplexityModel,
    validation_rows: list[DatasetRow],
    *,
    maximum_tier_underroute_rate: float = 0.03,
) -> dict[str, object]:
    """Select a validation policy while reusing the expensive route decisions."""
    from .router import ModelRouter, NoEligibleModel

    if not validation_rows:
        raise ValueError("adaptive policy tuning requires validation rows")
    confidence_thresholds = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    underroute_tolerances = (0.05, 0.10, 0.15, 0.20)
    catalog = load_catalog()
    tier_by_model = {profile.id: profile.tier for profile in catalog}
    argmax_router = ModelRouter(
        catalog,
        complexity_model=model,
        classifier_mode="hybrid",
        decision_policy="argmax",
    )
    risk_routers = {
        tolerance: ModelRouter(
            catalog,
            complexity_model=model,
            classifier_mode="hybrid",
            decision_policy="tier_risk",
            underroute_tolerance=tolerance,
        )
        for tolerance in underroute_tolerances
    }
    accumulators: dict[tuple[float, float], dict[str, object]] = {
        (threshold, tolerance): {
            "under": 0,
            "over": 0,
            "errors": 0,
            "total_cost": 0.0,
            "route_counts": Counter(),
        }
        for threshold in confidence_thresholds
        for tolerance in underroute_tolerances
    }
    for row in validation_rows:
        request = RoutingRequest(prompt=row.prompt, expected_output_tokens=500)
        try:
            argmax = argmax_router.route(request)
            risk_decisions = {
                tolerance: router.route(request)
                for tolerance, router in risk_routers.items()
            }
        except NoEligibleModel:
            for accumulator in accumulators.values():
                accumulator["under"] = int(accumulator["under"]) + 1
                accumulator["errors"] = int(accumulator["errors"]) + 1
            continue
        required_tier = LEVEL_TO_TIER[row.level]
        for threshold in confidence_thresholds:
            use_risk = (
                argmax.features.risk == "high"
                or float(argmax.features.classifier_confidence or 0.0) < threshold
            )
            for tolerance in underroute_tolerances:
                decision = risk_decisions[tolerance] if use_risk else argmax
                selected_tier = tier_by_model[decision.model_id]
                accumulator = accumulators[(threshold, tolerance)]
                accumulator["under"] = int(accumulator["under"]) + int(
                    selected_tier < required_tier
                )
                accumulator["over"] = int(accumulator["over"]) + int(
                    selected_tier > required_tier
                )
                accumulator["total_cost"] = (
                    float(accumulator["total_cost"]) + decision.estimated_cost_usd
                )
                route_counts = accumulator["route_counts"]
                assert isinstance(route_counts, Counter)
                route_counts[decision.model_id] += 1

    count = len(validation_rows)
    candidates: list[dict[str, object]] = []
    for threshold in confidence_thresholds:
        for tolerance in underroute_tolerances:
            accumulator = accumulators[(threshold, tolerance)]
            route_counts = accumulator["route_counts"]
            assert isinstance(route_counts, Counter)
            candidates.append(
                {
                    "confidence_threshold": threshold,
                    "underroute_tolerance": tolerance,
                    "count": count,
                    "route_errors": int(accumulator["errors"]),
                    "tier_underroute_rate": round(int(accumulator["under"]) / count, 6),
                    "tier_overroute_rate": round(int(accumulator["over"]) / count, 6),
                    "average_estimated_cost_usd": round(
                        float(accumulator["total_cost"]) / count, 8
                    ),
                    "route_counts": dict(sorted(route_counts.items())),
                }
            )
    feasible = [
        candidate
        for candidate in candidates
        if float(candidate["tier_underroute_rate"]) <= maximum_tier_underroute_rate
    ]
    pool = feasible or candidates
    selected = min(
        pool,
        key=lambda candidate: (
            float(candidate["average_estimated_cost_usd"]),
            float(candidate["tier_underroute_rate"]),
            float(candidate["tier_overroute_rate"]),
            float(candidate["confidence_threshold"]),
            float(candidate["underroute_tolerance"]),
        ),
    )
    return {
        "selection_split": "validation",
        "objective": "minimise estimated cost subject to tier under-route risk cap",
        "maximum_tier_underroute_rate": maximum_tier_underroute_rate,
        "feasible_candidates": len(feasible),
        "selected": selected,
        "candidates": candidates,
        "note": (
            "The selected settings must first pass held-out and external "
            "prompt-distribution checks, then response-level evaluation before "
            "enforcement."
        ),
    }


def _classifier_latency_benchmark(
    model: NaiveBayesComplexityModel,
    rows: list[DatasetRow],
    *,
    maximum_samples: int = 1_000,
) -> dict[str, object]:
    selected = rows[:maximum_samples]
    timings: list[float] = []
    for row in selected:
        started = time.perf_counter_ns()
        model.predict(row.prompt)
        timings.append((time.perf_counter_ns() - started) / 1_000)
    ordered = sorted(timings)
    return {
        "samples": len(selected),
        "median_us": round(statistics.median(ordered), 3),
        "p95_us": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 3),
        "p99_us": round(ordered[max(0, int(len(ordered) * 0.99) - 1)], 3),
        "note": "Single-process local measurement; rerun in the deployment environment.",
    }


def _markdown_report(report: dict[str, object]) -> str:
    dataset = report["dataset"]
    assert isinstance(dataset, dict)
    test = report["evaluation"]
    assert isinstance(test, dict)
    strategies = test["strategies"]
    assert isinstance(strategies, dict)

    lines = [
        "# Learned Complexity Router — Training and Evaluation Report",
        "",
        f"**Generated:** {report['generated_at']}",
        "",
        "## Result summary",
        "",
        "The trained model predicts the audited Level 1–5 complexity label from "
        "the flattened conversation text. It does **not** prove which deployed LLM "
        "will produce the best answer; that requires response-level model evaluations.",
        "",
        "| Strategy | Accuracy | Macro F1 | MAE | Tier success proxy | Tier under-route | Relative cost index |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in strategies.items():
        assert isinstance(metrics, dict)
        lines.append(
            f"| {name} | {float(metrics['accuracy']):.2%} | "
            f"{float(metrics['macro_f1']):.3f} | "
            f"{float(metrics['mean_absolute_error']):.3f} | "
            f"{float(metrics['tier_success_proxy']):.2%} | "
            f"{float(metrics['tier_underroute_rate']):.2%} | "
            f"{float(metrics['average_relative_cost_index']):.3f} |"
        )

    learned = strategies["learned_argmax"]
    assert isinstance(learned, dict)
    end_to_end = test["end_to_end_router"]
    assert isinstance(end_to_end, dict)
    latency = test["classifier_latency_us"]
    assert isinstance(latency, dict)
    training = report["training"]
    assert isinstance(training, dict)
    augmentation = training.get("augmentation", {})
    assert isinstance(augmentation, dict)
    lines.extend(
        [
            "",
            "## Dataset and split",
            "",
            f"- Source file: `{dataset['file']}`",
            f"- SHA-256: `{dataset['sha256']}`",
            f"- Valid rows: {int(dataset['rows']):,}",
            f"- Train: {int(dataset['split_counts']['train']):,}",
            f"- Validation: {int(dataset['split_counts']['validation']):,}",
            f"- Test: {int(dataset['split_counts']['test']):,}",
            "- Split method: normalized-prompt SHA-256 hash; duplicates would remain in one split.",
            f"- Duplicate prompts: {dataset['duplicate_rows']}",
            f"- Borderline rows: {dataset['statuses'].get('borderline', 0):,}; down-weighted during training.",
            f"- Training-only augmented views: {int(augmentation.get('augmented_training_views', 0)):,}; original validation, test, and external prompts were unchanged.",
            "",
            "## Learned model test metrics",
            "",
            f"- Exact five-level accuracy: {float(learned['accuracy']):.2%}",
            f"- Macro F1: {float(learned['macro_f1']):.3f}",
            f"- Mean absolute level error: {float(learned['mean_absolute_error']):.3f}",
            f"- Within one level: {float(learned['within_one_level_rate']):.2%}",
            f"- Severe error rate: {float(learned['severe_error_rate']):.2%}",
            f"- Collapsed tier accuracy: {float(learned['tier_accuracy']):.2%}",
            f"- Collapsed tier under-route rate: {float(learned['tier_underroute_rate']):.2%}",
            f"- Expected calibration error: {float(learned['expected_calibration_error']):.3f}",
            f"- Local inference latency: median {float(latency['median_us']):.1f} µs; p95 {float(latency['p95_us']):.1f} µs over {int(latency['samples']):,} prompts",
            "",
            "## End-to-end router simulation",
            "",
            "This section runs the complete policy router, including hard gates and catalogue scoring, over the held-out prompts. Cost uses the illustrative catalogue and a fixed 500-token output allowance.",
            "",
            "| Router mode | Tier success proxy | Tier under-route | Tier over-route | Est. cost/request | Saving vs capable | Route mix |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, metrics in end_to_end.items():
        assert isinstance(metrics, dict)
        mix = ", ".join(
            f"{model_name}: {count}"
            for model_name, count in metrics["route_counts"].items()
        )
        lines.append(
            f"| {name} | {float(metrics['tier_success_proxy']):.2%} | "
            f"{float(metrics['tier_underroute_rate']):.2%} | "
            f"{float(metrics['tier_overroute_rate']):.2%} | "
            f"${float(metrics['average_estimated_cost_usd']):.5f} | "
            f"{float(metrics['estimated_cost_saving_vs_always_capable']):.2%} | {mix} |"
        )
    tuning = report.get("adaptive_policy_tuning")
    if isinstance(tuning, dict) and isinstance(tuning.get("selected"), dict):
        selected = tuning["selected"]
        confirmation = tuning.get("external_confirmation", {})
        assert isinstance(confirmation, dict)
        lines.extend(
            [
                "",
                "## Validation-selected adaptive policy",
                "",
                "The confidence threshold and posterior risk tolerance were selected "
                "using only the validation split. The objective minimises estimated "
                "cost while respecting the declared validation under-routing cap.",
                "",
                f"- Confidence threshold: {float(selected['confidence_threshold']):.2f}",
                f"- Posterior under-route tolerance: {float(selected['underroute_tolerance']):.2f}",
                f"- Validation tier under-route: {float(selected['tier_underroute_rate']):.2%}",
                f"- Validation tier over-route: {float(selected['tier_overroute_rate']):.2%}",
                f"- Validation estimated cost/request: ${float(selected['average_estimated_cost_usd']):.5f}",
                f"- Feasible candidates: {int(tuning['feasible_candidates'])}/{len(tuning['candidates'])}",
                f"- External confirmation: {'passed' if confirmation.get('passed') else 'not passed'}",
                f"- Decision: {confirmation.get('decision', 'external confirmation pending')}",
                "- This is a validation candidate, not a deployment recommendation.",
            ]
        )
    lines.extend(
        [
            "",
            "## Important interpretation",
            "",
            "- The prompts were synthetically generated to match assigned complexity levels, so lexical cues may make the held-out task easier than real traffic.",
            "- The labels were audited by another LLM, not verified through candidate-model responses.",
            "- `borderline` rows are retained but receive lower training weight.",
            "- The relative cost index is a three-tier planning proxy, not provider billing.",
            "- The conservative policy intentionally trades higher cost for a lower under-routing rate.",
            "- A production release still requires a frozen set of real requests executed against every candidate model.",
            "",
            "## Confusion matrix",
            "",
            "Rows are true levels and columns are predicted levels.",
            "",
            "| True \\ Predicted | 1 | 2 | 3 | 4 | 5 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(learned["confusion_matrix"], start=1):
        lines.append(f"| {index} | " + " | ".join(str(value) for value in row) + " |")
    lines.extend(
        [
            "",
            "## Release recommendation",
            "",
            "Use this artifact in shadow mode as a complexity signal behind the existing hard gates. Do not promote it as the sole production model selector until response-level candidate benchmarks and real-traffic drift tests pass.",
            "",
        ]
    )
    external = report.get("external_context_slice")
    if isinstance(external, dict) and isinstance(external.get("metrics"), dict):
        metrics = external["metrics"]
        lines.extend(
            [
                "",
                "## Novel multi-turn context slice",
                "",
                f"- Source rows: {int(external['source_rows']):,}",
                f"- Rows overlapping the primary dataset and excluded: {int(external['overlap_with_primary_dataset']):,}",
                f"- Novel rows evaluated: {int(external['novel_rows']):,}",
                f"- Exact accuracy: {float(metrics['accuracy']):.2%}",
                f"- Macro F1: {float(metrics['macro_f1']):.3f}",
                f"- Tier under-route rate: {float(metrics['tier_underroute_rate']):.2%}",
                "",
                f"**Warning:** {external['warning']} The large drop from the hash-held-out test is evidence that the main test result is not sufficient for a production release.",
            ]
        )
    return "\n".join(lines)


def train_and_evaluate(
    dataset_path: str | Path,
    artifact_path: str | Path,
    report_json_path: str | Path,
    report_markdown_path: str | Path,
    *,
    external_context_path: str | Path | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, object]:
    training_config = config or TrainingConfig()
    training_config.validate()
    source_path = Path(dataset_path)
    rows = list(read_dataset(source_path))
    audit = _audit_dataset(rows)
    if audit["conflicting_duplicate_labels"]:
        raise ValueError("dataset contains duplicate prompts with conflicting labels")

    split_rows: dict[str, list[DatasetRow]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for row in rows:
        split_rows[
            split_for_prompt(
                row.prompt,
                train_percentage=training_config.train_percentage,
                validation_percentage=training_config.validation_percentage,
            )
        ].append(row)

    vectorizer = HashedTextVectorizer(training_config.feature_dimension)
    feature_counts = np.full(
        (5, training_config.feature_dimension),
        training_config.smoothing,
        dtype=np.float64,
    )
    class_weights = np.full(5, training_config.smoothing, dtype=np.float64)
    for row in split_rows["train"]:
        indices, values = vectorizer.transform_one(row.prompt)
        weight = _row_weight(row, training_config)
        np.add.at(feature_counts[row.level - 1], indices, values * weight)
        class_weights[row.level - 1] += weight
        for augmented_prompt in _augmented_prompts(
            row.prompt, training_config.augmentation_copies_per_train_row
        ):
            augmented_indices, augmented_values = vectorizer.transform_one(
                augmented_prompt
            )
            augmented_weight = weight * training_config.augmentation_weight
            np.add.at(
                feature_counts[row.level - 1],
                augmented_indices,
                augmented_values * augmented_weight,
            )
            class_weights[row.level - 1] += augmented_weight

    log_feature_probability = np.log(
        feature_counts / feature_counts.sum(axis=1, keepdims=True)
    )
    if training_config.class_prior == "uniform":
        log_class_prior = np.log(np.full(5, 0.2, dtype=np.float64))
    else:
        log_class_prior = np.log(class_weights / class_weights.sum())

    uncalibrated = NaiveBayesComplexityModel(
        log_feature_probability,
        log_class_prior,
        temperature=1.0,
    )
    validation_labels = [row.level for row in split_rows["validation"]]
    validation_scores = np.vstack(
        [uncalibrated.raw_scores(row.prompt) for row in split_rows["validation"]]
    )
    final_turn_scores = np.vstack(
        [
            uncalibrated.raw_scores(final_user_turn(row.prompt))
            for row in split_rows["validation"]
        ]
    )
    final_turn_weight, temperature = choose_view_ensemble(
        validation_scores, final_turn_scores, validation_labels
    )

    generated_at = datetime.now(UTC).isoformat()
    metadata: dict[str, object] = {
        "generated_at": generated_at,
        "training_dataset": source_path.name,
        "training_dataset_sha256": sha256_file(source_path),
        "training_rows": len(split_rows["train"]),
        "validation_rows": len(split_rows["validation"]),
        "feature_dimension": training_config.feature_dimension,
        "temperature": temperature,
        "final_turn_weight": final_turn_weight,
        "class_prior": training_config.class_prior,
        "borderline_weight": training_config.borderline_weight,
        "augmentation_copies_per_train_row": (
            training_config.augmentation_copies_per_train_row
        ),
        "augmentation_weight": training_config.augmentation_weight,
        "label_semantics": "synthetic, LLM-audited prompt complexity level 1-5",
    }
    model = NaiveBayesComplexityModel(
        log_feature_probability,
        log_class_prior,
        temperature=temperature,
        final_turn_weight=final_turn_weight,
        metadata=metadata,
    )
    validation_predictions, validation_probabilities, _ = _predict_rows(
        model, split_rows["validation"]
    )
    adaptive_policy_tuning = _tune_adaptive_policy(model, split_rows["validation"])
    selected_adaptive_policy = adaptive_policy_tuning["selected"]
    assert isinstance(selected_adaptive_policy, dict)

    test_rows = split_rows["test"]
    labels = [row.level for row in test_rows]
    learned_predictions, probabilities, _ = _predict_rows(model, test_rows)
    conservative_predictions, _, _ = _predict_rows(
        model, test_rows, decision_policy="conservative"
    )
    expected_predictions, _, _ = _predict_rows(
        model, test_rows, decision_policy="expected"
    )
    heuristic_predictions = [heuristic_level(row.prompt) for row in test_rows]
    majority_level = Counter(row.level for row in split_rows["train"]).most_common(1)[
        0
    ][0]

    strategies: dict[str, dict[str, object]] = {
        "learned_argmax": _evaluate_strategy(
            labels, learned_predictions, probabilities
        ),
        "learned_expected": _evaluate_strategy(labels, expected_predictions),
        "learned_conservative_p80": _evaluate_strategy(
            labels, conservative_predictions
        ),
        "heuristic": _evaluate_strategy(labels, heuristic_predictions),
        "majority_level": _evaluate_strategy(labels, [majority_level] * len(labels)),
        "always_efficient_tier": _evaluate_strategy(labels, [2] * len(labels)),
        "always_balanced_tier": _evaluate_strategy(labels, [3] * len(labels)),
        "always_capable_tier": _evaluate_strategy(labels, [5] * len(labels)),
    }
    capable_baseline = strategies["always_capable_tier"]
    for metrics in strategies.values():
        metrics["relative_cost_saving_vs_always_capable"] = relative_cost_saving(
            metrics, capable_baseline
        )

    dataset_report = {
        "file": source_path.name,
        "sha256": metadata["training_dataset_sha256"],
        **audit,
        "split_counts": {
            split: len(split_rows[split]) for split in ("train", "validation", "test")
        },
        "split_level_counts": {
            split: dict(sorted(Counter(row.level for row in split_rows[split]).items()))
            for split in ("train", "validation", "test")
        },
    }
    report: dict[str, object] = {
        "schema_version": "router-training-report-v2",
        "generated_at": generated_at,
        "dataset": dataset_report,
        "training": {
            "algorithm": "weighted multinomial naive Bayes",
            "vectorizer": "stable hashed word unigrams, bigrams, and structural features",
            "feature_dimension": training_config.feature_dimension,
            "smoothing": training_config.smoothing,
            "borderline_weight": training_config.borderline_weight,
            "appropriate_weight": training_config.appropriate_weight,
            "class_prior": training_config.class_prior,
            "augmentation": {
                "method": "deterministic meaning-preserving prompt envelopes",
                "copies_per_train_row": (
                    training_config.augmentation_copies_per_train_row
                ),
                "weight": training_config.augmentation_weight,
                "augmented_training_views": (
                    len(split_rows["train"])
                    * training_config.augmentation_copies_per_train_row
                ),
                "split_safety": "only train-split rows were augmented",
            },
            "temperature": temperature,
            "final_turn_weight": final_turn_weight,
        },
        "validation": classification_metrics(
            validation_labels,
            validation_predictions,
            validation_probabilities,
        ),
        "adaptive_policy_tuning": adaptive_policy_tuning,
        "evaluation": {
            "split": "test",
            "strategies": strategies,
            "end_to_end_router": _end_to_end_router_evaluation(
                model,
                test_rows,
                adaptive_policy=selected_adaptive_policy,
            ),
            "slices": {
                "audit_status": slice_metrics(
                    labels,
                    learned_predictions,
                    [row.status for row in test_rows],
                ),
                "category": slice_metrics(
                    labels,
                    learned_predictions,
                    [row.category for row in test_rows],
                    minimum_size=50,
                ),
                "user_turn_count": slice_metrics(
                    labels,
                    learned_predictions,
                    [str(row.prompt.count("User:")) for row in test_rows],
                    minimum_size=25,
                ),
            },
            "classifier_latency_us": _classifier_latency_benchmark(model, test_rows),
        },
        "limitations": [
            "Labels represent synthetic prompt complexity, not measured candidate-model success.",
            "Prompts were generated to fit the label rubric and may contain learnable stylistic cues.",
            "An LLM audit is not equivalent to expert or task-outcome ground truth.",
            "The cost index is illustrative and is not provider billing.",
        ],
    }
    if external_context_path:
        report["external_context_slice"] = _external_context_evaluation(
            model,
            Path(external_context_path),
            {normalize_prompt(row.prompt) for row in rows},
            adaptive_policy=selected_adaptive_policy,
        )
        external = report["external_context_slice"]
        assert isinstance(external, dict)
        external_policies = external.get("routing_policies", {})
        assert isinstance(external_policies, dict)
        selected_external = external_policies.get(
            "hybrid_adaptive_validation_selected", {}
        )
        default_external = external_policies.get("hybrid_adaptive_p15", {})
        assert isinstance(selected_external, dict)
        assert isinstance(default_external, dict)
        selected_external_underroute = float(
            selected_external.get("tier_underroute_rate", 1.0)
        )
        default_external_underroute = float(
            default_external.get("tier_underroute_rate", 1.0)
        )
        external_confirmation_passed = (
            int(external.get("novel_rows", 0)) >= 1_000
            and selected_external_underroute <= 0.05
            and selected_external_underroute <= default_external_underroute
        )
        adaptive_policy_tuning["external_confirmation"] = {
            "passed": external_confirmation_passed,
            "minimum_examples": 1_000,
            "maximum_tier_underroute_rate": 0.05,
            "examples": int(external.get("novel_rows", 0)),
            "selected_tier_underroute_rate": selected_external_underroute,
            "default_tier_underroute_rate": default_external_underroute,
            "decision": (
                "eligible for further response-level validation"
                if external_confirmation_passed
                else "retain current production default and continue shadow evaluation"
            ),
        }
    else:
        external_confirmation_passed = False
        adaptive_policy_tuning["external_confirmation"] = {
            "passed": False,
            "decision": "external confirmation dataset was not supplied",
        }

    model.metadata.update(
        {
            "test_accuracy": strategies["learned_argmax"]["accuracy"],
            "test_macro_f1": strategies["learned_argmax"]["macro_f1"],
            "test_tier_underroute_rate": strategies["learned_argmax"][
                "tier_underroute_rate"
            ],
            "adaptive_policy_validation_selected": {
                "confidence_threshold": selected_adaptive_policy[
                    "confidence_threshold"
                ],
                "underroute_tolerance": selected_adaptive_policy[
                    "underroute_tolerance"
                ],
                "maximum_tier_underroute_rate": adaptive_policy_tuning[
                    "maximum_tier_underroute_rate"
                ],
                "external_confirmation_passed": external_confirmation_passed,
            },
        }
    )
    model.save(artifact_path)
    artifact = Path(artifact_path)
    catalog_document = load_catalog_document()
    report["provenance"] = {
        "artifact": {
            "file": artifact.name,
            "sha256": sha256_file(artifact),
            "schema_version": MODEL_SCHEMA_VERSION,
        },
        "catalog": {
            "sha256": catalog_sha256(),
            "schema_version": catalog_document["schema_version"],
        },
        "routing_policy_version": POLICY_VERSION,
    }

    json_path = Path(report_json_path)
    markdown_path = Path(report_markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the learned complexity router"
    )
    parser.add_argument("dataset", help="Audited JSONL dataset")
    parser.add_argument(
        "--artifact", default="model_router/artifacts/complexity_router_v2.npz"
    )
    parser.add_argument("--report-json", default="reports/complexity_router_v2.json")
    parser.add_argument("--report-markdown", default="reports/complexity_router_v2.md")
    parser.add_argument("--external-context-dataset")
    parser.add_argument("--feature-dimension", type=int, default=32_768)
    parser.add_argument("--borderline-weight", type=float, default=0.65)
    parser.add_argument(
        "--class-prior", choices=["uniform", "empirical"], default="uniform"
    )
    parser.add_argument("--augmentation-copies", type=int, default=1)
    parser.add_argument("--augmentation-weight", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = train_and_evaluate(
        args.dataset,
        args.artifact,
        args.report_json,
        args.report_markdown,
        external_context_path=args.external_context_dataset,
        config=TrainingConfig(
            feature_dimension=args.feature_dimension,
            borderline_weight=args.borderline_weight,
            class_prior=args.class_prior,
            augmentation_copies_per_train_row=args.augmentation_copies,
            augmentation_weight=args.augmentation_weight,
        ),
    )
    learned = report["evaluation"]["strategies"]["learned_argmax"]
    print(
        json.dumps(
            {
                "artifact": args.artifact,
                "report": args.report_markdown,
                "test_accuracy": learned["accuracy"],
                "test_macro_f1": learned["macro_f1"],
                "tier_underroute_rate": learned["tier_underroute_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
