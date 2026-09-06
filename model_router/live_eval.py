from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import catalog_sha256, load_catalog
from .switchyard import SwitchyardClient, SwitchyardError
from .types import ModelProfile


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    messages: tuple[dict[str, Any], ...]
    max_tokens: int
    validators: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ValidatorResult:
    type: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class CandidateRun:
    case_id: str
    target: str
    repetition: int
    success: bool
    score: float | None
    latency_ms: float
    ttft_ms: float | None
    response_model: str | None
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    finish_reason: str | None
    validators: tuple[ValidatorResult, ...]
    error: str | None
    content: str | None
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["latency_ms"] = round(self.latency_ms, 3)
        value["ttft_ms"] = round(self.ttft_ms, 3) if self.ttft_ms is not None else None
        value["estimated_cost_usd"] = (
            round(self.estimated_cost_usd, 10)
            if self.estimated_cost_usd is not None
            else None
        )
        return value


def case_set_sha256(cases: Sequence[EvaluationCase]) -> str:
    canonical = json.dumps(
        [asdict(case) for case in cases],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _case_set_profile(cases: Sequence[EvaluationCase]) -> dict[str, object]:
    """Return prompt-free coverage metadata suitable for release evidence."""
    dimensions: dict[str, dict[str, int]] = {}
    for dimension in ("category", "use_case", "risk", "complexity", "source"):
        counts = Counter(
            str(case.metadata[dimension])
            for case in cases
            if case.metadata.get(dimension) is not None
            and not isinstance(case.metadata.get(dimension), (dict, list))
        )
        if counts:
            dimensions[dimension] = dict(sorted(counts.items()))
    validator_types = Counter(
        str(validator.get("type", "unknown"))
        for case in cases
        for validator in case.validators
    )
    return {
        "unique_cases": len({case.id for case in cases}),
        "cases_with_validators": sum(bool(case.validators) for case in cases),
        "cases_without_validators": sum(not case.validators for case in cases),
        "validator_types": dict(sorted(validator_types.items())),
        "metadata_dimensions": dimensions,
        "contains_prompt_content": False,
    }


def _benchmark_provenance(
    cases: Sequence[EvaluationCase],
    targets: Sequence[str],
    *,
    repetitions: int,
    stream: bool,
    store_content: bool,
    switchyard_revision: str | None,
) -> dict[str, object]:
    inputs = {
        "case_set_sha256": case_set_sha256(cases),
        "catalog_sha256": catalog_sha256(),
        "targets": list(targets),
        "repetitions": repetitions,
        "stream": stream,
        "store_content": store_content,
        "switchyard_revision": switchyard_revision,
        "case_set_profile": _case_set_profile(cases),
    }
    fingerprint = sha256(
        json.dumps(inputs, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "switchyard-live-eval-provenance-v1",
        **inputs,
        "benchmark_fingerprint": fingerprint,
    }


def load_cases(path: str | Path) -> list[EvaluationCase]:
    case_path = Path(path)
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    with case_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {case_path.name}:{line_number}"
                ) from error
            case_id = str(raw.get("id", "")).strip()
            if not case_id or case_id in seen:
                raise ValueError(
                    f"Case IDs must be non-empty and unique: {case_path.name}:{line_number}"
                )
            seen.add(case_id)
            messages = raw.get("messages")
            if messages is None and isinstance(raw.get("prompt"), str):
                messages = [{"role": "user", "content": raw["prompt"]}]
            if (
                not isinstance(messages, list)
                or not messages
                or not all(isinstance(message, dict) for message in messages)
            ):
                raise ValueError(f"Case {case_id!r} must contain prompt or messages")
            max_tokens = int(raw.get("max_tokens", 500))
            if max_tokens <= 0:
                raise ValueError(f"Case {case_id!r} has invalid max_tokens")
            validators = raw.get("validators", [])
            if not isinstance(validators, list) or not all(
                isinstance(validator, dict) for validator in validators
            ):
                raise ValueError(f"Case {case_id!r} validators must be an array")
            cases.append(
                EvaluationCase(
                    id=case_id,
                    messages=tuple(dict(message) for message in messages),
                    max_tokens=max_tokens,
                    validators=tuple(dict(validator) for validator in validators),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
    if not cases:
        raise ValueError("evaluation case file is empty")
    return cases


def _extract_content(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("response does not contain OpenAI chat content") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError("response content is not text")


def _validate(content: str, specification: Mapping[str, Any]) -> ValidatorResult:
    validator_type = str(specification.get("type", ""))
    case_sensitive = bool(specification.get("case_sensitive", False))
    comparable = content.strip() if case_sensitive else content.strip().lower()

    if validator_type == "exact_match":
        expected = str(specification.get("value", ""))
        expected_value = (
            expected.strip() if case_sensitive else expected.strip().lower()
        )
        passed = comparable == expected_value
        return ValidatorResult(
            validator_type,
            passed,
            "matched expected text" if passed else "text differed",
        )

    if validator_type == "contains_all":
        values = specification.get("values", [])
        if not isinstance(values, list) or not values:
            raise ValueError("contains_all requires a non-empty values array")
        expected = [str(value) for value in values]
        if not case_sensitive:
            expected = [value.lower() for value in expected]
        missing = [value for value in expected if value not in comparable]
        return ValidatorResult(
            validator_type,
            not missing,
            "all values present" if not missing else f"missing values: {missing}",
        )

    if validator_type == "regex":
        pattern = str(specification.get("pattern", ""))
        if not pattern:
            raise ValueError("regex requires pattern")
        flags = 0 if case_sensitive else re.IGNORECASE
        passed = re.search(pattern, content, flags) is not None
        return ValidatorResult(
            validator_type, passed, "pattern matched" if passed else "pattern not found"
        )

    if validator_type in {"valid_json", "json_keys"}:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return ValidatorResult(validator_type, False, "response was not valid JSON")
        if validator_type == "valid_json":
            return ValidatorResult(validator_type, True, "valid JSON")
        keys = specification.get("keys", [])
        if not isinstance(keys, list) or not keys:
            raise ValueError("json_keys requires a non-empty keys array")
        if not isinstance(parsed, dict):
            return ValidatorResult(
                validator_type, False, "JSON value was not an object"
            )
        missing = [str(key) for key in keys if str(key) not in parsed]
        return ValidatorResult(
            validator_type,
            not missing,
            "all keys present" if not missing else f"missing keys: {missing}",
        )

    if validator_type == "json_equals":
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return ValidatorResult(validator_type, False, "response was not valid JSON")
        if "value" not in specification:
            raise ValueError("json_equals requires value")
        passed = parsed == specification["value"]
        return ValidatorResult(
            validator_type,
            passed,
            "JSON value matched" if passed else "JSON value differed",
        )

    if validator_type == "numeric_tolerance":
        if specification.get("value") is None:
            raise ValueError("numeric_tolerance requires value")
        tolerance = float(specification.get("tolerance", 0.0))
        if tolerance < 0:
            raise ValueError("numeric_tolerance tolerance cannot be negative")
        candidate = content.strip().replace(",", "")
        try:
            actual = float(candidate)
            expected = float(specification["value"])
        except ValueError:
            return ValidatorResult(
                validator_type, False, "response was not a single number"
            )
        passed = abs(actual - expected) <= tolerance
        return ValidatorResult(
            validator_type,
            passed,
            "number within tolerance" if passed else "number outside tolerance",
        )

    raise ValueError(f"Unsupported validator type: {validator_type!r}")


def evaluate_one(
    client: SwitchyardClient,
    case: EvaluationCase,
    target: str,
    *,
    store_content: bool = False,
    repetition: int = 1,
    stream: bool = False,
    model_profile: ModelProfile | None = None,
) -> CandidateRun:
    started = time.perf_counter()
    try:
        method = client.chat_completions_stream if stream else client.chat_completions
        response = method(
            model=target, messages=case.messages, max_tokens=case.max_tokens
        )
        measured_latency_ms = (time.perf_counter() - started) * 1_000
        timing = response.get("_router_timing", {})
        if not isinstance(timing, dict):
            timing = {}
        latency_ms = float(timing.get("total_ms", measured_latency_ms))
        ttft_ms = (
            float(timing["ttft_ms"]) if timing.get("ttft_ms") is not None else None
        )
        content = _extract_content(response)
        validator_results = tuple(
            _validate(content, validator) for validator in case.validators
        )
        score = (
            sum(result.passed for result in validator_results) / len(validator_results)
            if validator_results
            else None
        )
        usage = response.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        input_tokens = (
            int(usage["prompt_tokens"])
            if usage.get("prompt_tokens") is not None
            else None
        )
        output_tokens = (
            int(usage["completion_tokens"])
            if usage.get("completion_tokens") is not None
            else None
        )
        prompt_details = usage.get("prompt_tokens_details", {})
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        cached_input_tokens = (
            int(prompt_details["cached_tokens"])
            if prompt_details.get("cached_tokens") is not None
            else None
        )
        estimated_cost = (
            model_profile.estimate_cost(
                input_tokens,
                output_tokens,
                cached_input_tokens=cached_input_tokens or 0,
            )
            if model_profile is not None
            and input_tokens is not None
            and output_tokens is not None
            else None
        )
        choices = response.get("choices", [])
        finish_reason = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = (
                str(choices[0]["finish_reason"])
                if choices[0].get("finish_reason") is not None
                else None
            )
        return CandidateRun(
            case_id=case.id,
            target=target,
            repetition=repetition,
            success=True,
            score=score,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            response_model=str(response.get("model", target)),
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
            finish_reason=finish_reason,
            validators=validator_results,
            error=None,
            content=content if store_content else None,
            metadata=case.metadata,
        )
    except (SwitchyardError, ValueError, TypeError, KeyError) as error:
        return CandidateRun(
            case_id=case.id,
            target=target,
            repetition=repetition,
            success=False,
            score=None,
            latency_ms=(time.perf_counter() - started) * 1_000,
            ttft_ms=None,
            response_model=None,
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
            estimated_cost_usd=None,
            finish_reason=None,
            validators=(),
            error=f"{type(error).__name__}: {error}",
            content=None,
            metadata=case.metadata,
        )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
    return ordered[index]


def _wilson_95(successes: int, total: int) -> dict[str, float] | None:
    """Return a 95% Wilson interval for a binomial outcome."""

    if total <= 0:
        return None
    z = 1.959963984540054
    observed = successes / total
    denominator = 1 + (z * z / total)
    centre = (observed + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(observed * (1 - observed) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "lower": round(max(0.0, centre - margin), 6),
        "upper": round(min(1.0, centre + margin), 6),
    }


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": round(statistics.median(values), 3) if values else None,
        "p95": (
            round(value, 3)
            if (value := _percentile(values, 0.95)) is not None
            else None
        ),
        "p99": (
            round(value, 3)
            if (value := _percentile(values, 0.99)) is not None
            else None
        ),
    }


def _slice_summary(runs: list[CandidateRun]) -> dict[str, object]:
    successful = [run for run in runs if run.success]
    scored = [run for run in successful if run.score is not None]
    perfect = sum(run.score == 1.0 for run in scored)
    return {
        "runs": len(runs),
        "unique_cases": len({run.case_id for run in runs}),
        "successful_calls": len(successful),
        "call_success_rate": round(len(successful) / len(runs), 6),
        "call_success_rate_wilson_95": _wilson_95(len(successful), len(runs)),
        "validator_scored_runs": len(scored),
        "all_validators_pass_rate": (
            round(perfect / len(scored), 6) if scored else None
        ),
        "all_validators_pass_rate_wilson_95": _wilson_95(perfect, len(scored)),
        "latency_ms": _distribution([round(run.latency_ms, 3) for run in successful]),
    }


def _metadata_slices(runs: list[CandidateRun]) -> dict[str, object]:
    dimensions: dict[str, dict[str, list[CandidateRun]]] = {}
    for dimension in ("category", "use_case", "risk", "complexity"):
        values: dict[str, list[CandidateRun]] = {}
        for run in runs:
            raw = run.metadata.get(dimension)
            if raw is None or isinstance(raw, (dict, list)):
                continue
            values.setdefault(str(raw), []).append(run)
        if values:
            dimensions[dimension] = {
                value: _slice_summary(selected)
                for value, selected in sorted(values.items())
            }
    return dimensions


def summarize(runs: Sequence[CandidateRun]) -> dict[str, object]:
    targets = sorted({run.target for run in runs})
    expected_models = {
        model.switchyard_target: model.provider_model for model in load_catalog()
    }
    by_target: dict[str, object] = {}
    for target in targets:
        selected = [run for run in runs if run.target == target]
        successful = [run for run in selected if run.success]
        scored = [run for run in successful if run.score is not None]
        # Summaries use the same canonical precision as the JSONL checkpoint so a
        # resumed benchmark is byte-for-byte comparable with its original run.
        latencies = [round(run.latency_ms, 3) for run in successful]
        ttfts = [round(run.ttft_ms, 3) for run in successful if run.ttft_ms is not None]
        input_tokens = [
            run.input_tokens for run in successful if run.input_tokens is not None
        ]
        output_tokens = [
            run.output_tokens for run in successful if run.output_tokens is not None
        ]
        costs = [
            round(run.estimated_cost_usd, 10)
            for run in successful
            if run.estimated_cost_usd is not None
        ]
        end_to_end_rates = [
            run.output_tokens / (round(run.latency_ms, 3) / 1_000)
            for run in successful
            if run.output_tokens is not None and round(run.latency_ms, 3) > 0
        ]
        generation_rates = [
            run.output_tokens
            / ((round(run.latency_ms, 3) - round(run.ttft_ms, 3)) / 1_000)
            for run in successful
            if run.output_tokens is not None
            and run.ttft_ms is not None
            and round(run.latency_ms, 3) > round(run.ttft_ms, 3)
        ]
        mismatched_models = sum(
            run.response_model
            not in {None, run.target, expected_models.get(run.target)}
            for run in successful
        )
        perfect = sum(run.score == 1.0 for run in scored)
        by_target[target] = {
            "runs": len(selected),
            "unique_cases": len({run.case_id for run in selected}),
            "successful_calls": len(successful),
            "call_success_rate": round(len(successful) / len(selected), 6),
            "call_success_rate_wilson_95": _wilson_95(len(successful), len(selected)),
            "validator_scored_runs": len(scored),
            "average_validator_score": (
                round(statistics.mean(run.score for run in scored), 6)
                if scored
                else None
            ),
            "all_validators_pass_rate": (
                round(perfect / len(scored), 6) if scored else None
            ),
            "all_validators_pass_rate_wilson_95": _wilson_95(perfect, len(scored)),
            "latency_ms": _distribution(latencies),
            "ttft_ms": _distribution(ttfts),
            "end_to_end_output_tokens_per_second": _distribution(end_to_end_rates),
            "generation_output_tokens_per_second": _distribution(generation_rates),
            "total_input_tokens": sum(input_tokens),
            "total_output_tokens": sum(output_tokens),
            "cached_input_token_rate": (
                round(
                    sum(
                        run.cached_input_tokens or 0
                        for run in successful
                        if run.input_tokens is not None
                    )
                    / sum(input_tokens),
                    6,
                )
                if input_tokens and sum(input_tokens) > 0
                else None
            ),
            "estimated_total_cost_usd": (round(sum(costs), 8) if costs else None),
            "response_model_mismatch_count": mismatched_models,
            "finish_reasons": dict(
                sorted(
                    Counter(
                        run.finish_reason or "unknown" for run in successful
                    ).items()
                )
            ),
            "slices": _metadata_slices(selected),
            "errors": Counter(
                run.error or "unknown" for run in selected if not run.success
            ),
        }
    return {
        "schema_version": "switchyard-live-eval-summary-v3",
        "total_runs": len(runs),
        "targets": by_target,
        "caveat": (
            "Deterministic validators cover only cases with machine-checkable outcomes. "
            "Open-ended tasks require an approved judge or human evaluation rubric. "
            "Wilson intervals quantify sampling uncertainty but do not correct for "
            "benchmark representativeness or correlated repetitions. "
            "Estimated cost uses provider-reported token counts and the versioned "
            "catalog; the provider invoice remains authoritative."
        ),
    }


def run_benchmark(
    client: SwitchyardClient,
    cases: Sequence[EvaluationCase],
    targets: Sequence[str],
    *,
    output_path: str | Path,
    summary_path: str | Path,
    concurrency: int = 4,
    store_content: bool = False,
    resume: bool = True,
    repetitions: int = 1,
    stream: bool = False,
    switchyard_revision: str | None = None,
) -> dict[str, object]:
    if not targets:
        raise ValueError("at least one target is required")
    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero")
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must be unique")
    result_path = Path(output_path)
    summary_file = Path(summary_path)
    completed: dict[tuple[str, str, int], CandidateRun] = {}
    models_by_target = {model.switchyard_target: model for model in load_catalog()}
    provenance = _benchmark_provenance(
        cases,
        targets,
        repetitions=repetitions,
        stream=stream,
        store_content=store_content,
        switchyard_revision=switchyard_revision,
    )
    if resume and result_path.exists():
        if not summary_file.exists():
            raise ValueError(
                "cannot safely resume: checkpoint summary/provenance is missing"
            )
        previous_summary = json.loads(summary_file.read_text(encoding="utf-8"))
        previous_provenance = (
            previous_summary.get("provenance", {})
            if isinstance(previous_summary, dict)
            else {}
        )
        if (
            not isinstance(previous_provenance, dict)
            or previous_provenance.get("benchmark_fingerprint")
            != provenance["benchmark_fingerprint"]
        ):
            raise ValueError(
                "cannot safely resume: cases, catalogue, targets, or run settings changed"
            )
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                raw.setdefault("repetition", 1)
                raw.setdefault("ttft_ms", None)
                raw.setdefault("cached_input_tokens", None)
                raw.setdefault("estimated_cost_usd", None)
                raw.setdefault("finish_reason", None)
                validators = tuple(
                    ValidatorResult(**item) for item in raw.get("validators", [])
                )
                raw["validators"] = validators
                completed[
                    (
                        str(raw["case_id"]),
                        str(raw["target"]),
                        int(raw["repetition"]),
                    )
                ] = CandidateRun(**raw)

    pending = [
        (case, target, repetition)
        for case in cases
        for target in targets
        for repetition in range(1, repetitions + 1)
        if (case.id, target, repetition) not in completed
    ]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                client,
                case,
                target,
                store_content=store_content,
                repetition=repetition,
                stream=stream,
                model_profile=models_by_target.get(target),
            ): (case.id, target, repetition)
            for case, target, repetition in pending
        }
        for future in as_completed(futures):
            run = future.result()
            completed[(run.case_id, run.target, run.repetition)] = run

    ordered = [
        completed[(case.id, target, repetition)]
        for case in cases
        for target in targets
        for repetition in range(1, repetitions + 1)
    ]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        "".join(
            json.dumps(run.as_dict(), ensure_ascii=False) + "\n" for run in ordered
        ),
        encoding="utf-8",
    )
    summary = summarize(ordered)
    summary["provenance"] = provenance
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
