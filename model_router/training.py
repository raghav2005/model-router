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

from .catalog import load_catalog
from .classifier import classify_request
from .evaluation import (
    LEVEL_TO_TIER,
    choose_temperature,
    classification_metrics,
    relative_cost_saving,
    slice_metrics,
    softmax,
)
from .learned import (
    DEFAULT_FEATURE_DIMENSION,
    HashedTextVectorizer,
    NaiveBayesComplexityModel,
    normalize_prompt,
)
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
) -> dict[str, object]:
    rows = list(read_dataset(external_path))
    novel = [
        row for row in rows if normalize_prompt(row.prompt) not in training_prompts
    ]
    overlap = len(rows) - len(novel)
    result: dict[str, object] = {
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
    return result


def _end_to_end_router_evaluation(
    model: NaiveBayesComplexityModel, rows: list[DatasetRow]
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
    }
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
    validation_predictions, _, validation_scores = _predict_rows(
        uncalibrated, split_rows["validation"]
    )
    validation_labels = [row.level for row in split_rows["validation"]]
    temperature = choose_temperature(validation_scores, validation_labels)

    generated_at = datetime.now(UTC).isoformat()
    metadata: dict[str, object] = {
        "generated_at": generated_at,
        "training_dataset": source_path.name,
        "training_dataset_sha256": sha256_file(source_path),
        "training_rows": len(split_rows["train"]),
        "validation_rows": len(split_rows["validation"]),
        "feature_dimension": training_config.feature_dimension,
        "temperature": temperature,
        "class_prior": training_config.class_prior,
        "borderline_weight": training_config.borderline_weight,
        "label_semantics": "synthetic, LLM-audited prompt complexity level 1-5",
    }
    model = NaiveBayesComplexityModel(
        log_feature_probability,
        log_class_prior,
        temperature=temperature,
        metadata=metadata,
    )

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
        "schema_version": "router-training-report-v1",
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
            "temperature": temperature,
        },
        "validation": classification_metrics(
            validation_labels,
            validation_predictions,
            softmax(validation_scores.copy(), temperature),
        ),
        "evaluation": {
            "split": "test",
            "strategies": strategies,
            "end_to_end_router": _end_to_end_router_evaluation(model, test_rows),
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
        )

    model.metadata.update(
        {
            "test_accuracy": strategies["learned_argmax"]["accuracy"],
            "test_macro_f1": strategies["learned_argmax"]["macro_f1"],
            "test_tier_underroute_rate": strategies["learned_argmax"][
                "tier_underroute_rate"
            ],
        }
    )
    model.save(artifact_path)

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
        "--artifact", default="model_router/artifacts/complexity_router_v1.npz"
    )
    parser.add_argument("--report-json", default="reports/complexity_router_v1.json")
    parser.add_argument("--report-markdown", default="reports/complexity_router_v1.md")
    parser.add_argument("--external-context-dataset")
    parser.add_argument("--feature-dimension", type=int, default=32_768)
    parser.add_argument("--borderline-weight", type=float, default=0.65)
    parser.add_argument(
        "--class-prior", choices=["uniform", "empirical"], default="uniform"
    )
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
