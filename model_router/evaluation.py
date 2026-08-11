from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np


LEVEL_TO_TIER = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
TIER_COST_INDEX = {1: 1.0, 2: 2.5, 3: 5.0}


def confusion_matrix(
    labels: Sequence[int], predictions: Sequence[int], classes: int = 5
) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions, strict=True):
        if not 1 <= label <= classes or not 1 <= prediction <= classes:
            raise ValueError("labels and predictions must be in the configured range")
        matrix[label - 1, prediction - 1] += 1
    return matrix


def expected_calibration_error(
    labels: Sequence[int], probabilities: np.ndarray, bins: int = 10
) -> float:
    if not labels:
        return 0.0
    predictions = probabilities.argmax(axis=1) + 1
    confidence = probabilities.max(axis=1)
    correct = predictions == np.asarray(labels)
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        selected = (confidence >= lower) & (
            confidence <= upper if upper >= 1.0 else confidence < upper
        )
        count = int(selected.sum())
        if count:
            error += (
                count
                / len(labels)
                * abs(
                    float(correct[selected].mean()) - float(confidence[selected].mean())
                )
            )
    return error


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: np.ndarray | None = None,
) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions)
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("cannot evaluate an empty set")

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    weighted_f1 = 0.0
    for index in range(5):
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        f1_values.append(f1)
        weighted_f1 += f1 * support / total
        per_class[str(index + 1)] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    absolute_errors = [
        abs(label - prediction)
        for label, prediction in zip(labels, predictions, strict=True)
    ]
    under = sum(
        prediction < label
        for label, prediction in zip(labels, predictions, strict=True)
    )
    over = sum(
        prediction > label
        for label, prediction in zip(labels, predictions, strict=True)
    )

    true_tiers = [LEVEL_TO_TIER[label] for label in labels]
    predicted_tiers = [LEVEL_TO_TIER[prediction] for prediction in predictions]
    tier_under = sum(
        prediction < label
        for label, prediction in zip(true_tiers, predicted_tiers, strict=True)
    )
    tier_over = sum(
        prediction > label
        for label, prediction in zip(true_tiers, predicted_tiers, strict=True)
    )
    tier_exact = sum(
        prediction == label
        for label, prediction in zip(true_tiers, predicted_tiers, strict=True)
    )
    relative_cost = sum(TIER_COST_INDEX[tier] for tier in predicted_tiers) / total

    metrics: dict[str, object] = {
        "count": total,
        "accuracy": round(float(np.trace(matrix) / total), 6),
        "macro_f1": round(sum(f1_values) / 5, 6),
        "weighted_f1": round(weighted_f1, 6),
        "mean_absolute_error": round(sum(absolute_errors) / total, 6),
        "within_one_level_rate": round(
            sum(error <= 1 for error in absolute_errors) / total, 6
        ),
        "severe_error_rate": round(
            sum(error >= 2 for error in absolute_errors) / total, 6
        ),
        "under_level_rate": round(under / total, 6),
        "over_level_rate": round(over / total, 6),
        "tier_accuracy": round(tier_exact / total, 6),
        "tier_underroute_rate": round(tier_under / total, 6),
        "tier_overroute_rate": round(tier_over / total, 6),
        "tier_success_proxy": round(1.0 - tier_under / total, 6),
        "average_relative_cost_index": round(relative_cost, 6),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }
    if probabilities is not None:
        clipped = np.clip(
            probabilities[np.arange(total), np.asarray(labels) - 1],
            1e-12,
            1.0,
        )
        metrics["negative_log_likelihood"] = round(-float(np.log(clipped).mean()), 6)
        metrics["expected_calibration_error"] = round(
            expected_calibration_error(labels, probabilities), 6
        )
    return metrics


def slice_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    slice_values: Sequence[str],
    *,
    minimum_size: int = 25,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(slice_values):
        grouped[value].append(index)
    result: dict[str, dict[str, object]] = {}
    for value, indices in sorted(grouped.items()):
        if len(indices) < minimum_size:
            continue
        result[value] = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
    return result


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = scores / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def choose_temperature(scores: np.ndarray, labels: Sequence[int]) -> float:
    if len(scores) != len(labels):
        raise ValueError("score and label lengths differ")
    label_indices = np.asarray(labels, dtype=np.int64) - 1
    candidates = np.geomspace(0.25, 64.0, 65)
    best_temperature = 1.0
    best_loss = math.inf
    for candidate in candidates:
        probabilities = softmax(scores.copy(), float(candidate))
        selected = np.clip(
            probabilities[np.arange(len(labels)), label_indices], 1e-12, 1.0
        )
        loss = -float(np.log(selected).mean())
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(candidate)
    return best_temperature


def relative_cost_saving(
    candidate: dict[str, object], baseline: dict[str, object]
) -> float:
    candidate_cost = float(candidate["average_relative_cost_index"])
    baseline_cost = float(baseline["average_relative_cost_index"])
    return round(1.0 - candidate_cost / baseline_cost, 6)
