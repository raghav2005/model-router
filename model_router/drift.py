from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

CATEGORICAL_FIELDS = (
    "selected_model",
    "use_case",
    "risk",
    "complexity_level",
    "classifier_source",
)
NUMERIC_FIELDS = (
    "classifier_confidence",
    "classifier_entropy",
    "tier_underroute_probability",
    "estimated_cost_usd",
)


def load_route_events(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on audit line {line_number}") from error
            if not isinstance(event, dict):
                raise ValueError(f"audit line {line_number} must be a JSON object")
            if event.get("event") == "route_decision":
                events.append(event)
    return events


def _distribution(events: Iterable[dict[str, Any]], field: str) -> dict[str, float]:
    counts = Counter(str(event.get(field, "unknown")) for event in events)
    total = sum(counts.values())
    return {
        value: count / total for value, count in sorted(counts.items()) if total > 0
    }


def _numeric_summary(
    events: Iterable[dict[str, Any]], field: str
) -> dict[str, float | int] | None:
    values = [
        float(event[field])
        for event in events
        if event.get(field) is not None
        and isinstance(event[field], (int, float))
        and math.isfinite(float(event[field]))
    ]
    if not values:
        return None
    return {
        "count": len(values),
        "mean": fmean(values),
        "standard_deviation": pstdev(values),
    }


def build_baseline(events: list[dict[str, Any]]) -> dict[str, object]:
    if not events:
        raise ValueError("baseline requires at least one route_decision event")
    return {
        "schema_version": "model-router-drift-baseline-v1",
        "event_count": len(events),
        "policy_versions": sorted(
            {str(event.get("policy_version", "unknown")) for event in events}
        ),
        "classifier_model_versions": sorted(
            {str(event.get("classifier_model_version", "unknown")) for event in events}
        ),
        "categorical": {
            field: _distribution(events, field) for field in CATEGORICAL_FIELDS
        },
        "numeric": {
            field: summary
            for field in NUMERIC_FIELDS
            if (summary := _numeric_summary(events, field)) is not None
        },
    }


def jensen_shannon_distance(
    baseline: dict[str, float], current: dict[str, float]
) -> float:
    """Return a symmetric distribution distance in the range [0, 1]."""
    keys = set(baseline) | set(current)
    if not keys:
        return 0.0
    divergence = 0.0
    for key in keys:
        left = baseline.get(key, 0.0)
        right = current.get(key, 0.0)
        middle = (left + right) / 2
        if left > 0:
            divergence += 0.5 * left * math.log2(left / middle)
        if right > 0:
            divergence += 0.5 * right * math.log2(right / middle)
    return math.sqrt(max(0.0, min(1.0, divergence)))


def _scale_floor(field: str, baseline_mean: float) -> float:
    if field == "estimated_cost_usd":
        return max(abs(baseline_mean) * 0.10, 1e-6)
    return 0.05


def compare_to_baseline(
    baseline: dict[str, Any],
    current_events: list[dict[str, Any]],
    *,
    minimum_events: int = 100,
    maximum_js_distance: float = 0.10,
    maximum_standardized_mean_shift: float = 2.0,
) -> dict[str, object]:
    if baseline.get("schema_version") != "model-router-drift-baseline-v1":
        raise ValueError("unsupported drift baseline schema")
    if minimum_events < 1:
        raise ValueError("minimum_events must be positive")
    if not 0 <= maximum_js_distance <= 1:
        raise ValueError("maximum_js_distance must be in [0, 1]")
    if maximum_standardized_mean_shift <= 0:
        raise ValueError("maximum_standardized_mean_shift must be positive")

    categorical: dict[str, object] = {}
    drifted_fields: list[str] = []
    baseline_categorical = baseline.get("categorical", {})
    for field in CATEGORICAL_FIELDS:
        expected = baseline_categorical.get(field, {})
        observed = _distribution(current_events, field) if current_events else {}
        distance = jensen_shannon_distance(expected, observed)
        drifted = distance > maximum_js_distance
        if drifted:
            drifted_fields.append(field)
        categorical[field] = {
            "jensen_shannon_distance": round(distance, 6),
            "drifted": drifted,
            "baseline": expected,
            "current": observed,
        }

    numeric: dict[str, object] = {}
    baseline_numeric = baseline.get("numeric", {})
    for field in NUMERIC_FIELDS:
        expected = baseline_numeric.get(field)
        observed = _numeric_summary(current_events, field)
        if not isinstance(expected, dict) or observed is None:
            numeric[field] = {
                "drifted": True,
                "reason": "metric is missing from baseline or current window",
            }
            drifted_fields.append(field)
            continue
        baseline_mean = float(expected["mean"])
        baseline_deviation = float(expected["standard_deviation"])
        current_mean = float(observed["mean"])
        scale = max(baseline_deviation, _scale_floor(field, baseline_mean))
        standardized_shift = abs(current_mean - baseline_mean) / scale
        drifted = standardized_shift > maximum_standardized_mean_shift
        if drifted:
            drifted_fields.append(field)
        numeric[field] = {
            "standardized_mean_shift": round(standardized_shift, 6),
            "drifted": drifted,
            "baseline": expected,
            "current": observed,
        }

    enough_data = len(current_events) >= minimum_events
    unique_drifted = sorted(set(drifted_fields))
    status = (
        "insufficient_data"
        if not enough_data
        else "drift_detected"
        if unique_drifted
        else "ok"
    )
    return {
        "schema_version": "model-router-drift-report-v1",
        "status": status,
        "passed": status == "ok",
        "baseline_event_count": int(baseline.get("event_count", 0)),
        "current_event_count": len(current_events),
        "minimum_current_events": minimum_events,
        "thresholds": {
            "maximum_js_distance": maximum_js_distance,
            "maximum_standardized_mean_shift": maximum_standardized_mean_shift,
        },
        "drifted_fields": unique_drifted,
        "categorical": categorical,
        "numeric": numeric,
    }


def read_baseline(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    if not isinstance(baseline, dict):
        raise ValueError("drift baseline must be a JSON object")
    return baseline


def write_json(path: str | Path, value: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
