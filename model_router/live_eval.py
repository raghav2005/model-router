from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .switchyard import SwitchyardClient, SwitchyardError


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
    success: bool
    score: float | None
    latency_ms: float
    response_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    validators: tuple[ValidatorResult, ...]
    error: str | None
    content: str | None
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["latency_ms"] = round(self.latency_ms, 3)
        return value


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

    raise ValueError(f"Unsupported validator type: {validator_type!r}")


def evaluate_one(
    client: SwitchyardClient,
    case: EvaluationCase,
    target: str,
    *,
    store_content: bool = False,
) -> CandidateRun:
    started = time.perf_counter()
    try:
        response = client.chat_completions(
            model=target,
            messages=case.messages,
            max_tokens=case.max_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1_000
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
        return CandidateRun(
            case_id=case.id,
            target=target,
            success=True,
            score=score,
            latency_ms=latency_ms,
            response_model=str(response.get("model", target)),
            input_tokens=(
                int(usage["prompt_tokens"])
                if usage.get("prompt_tokens") is not None
                else None
            ),
            output_tokens=(
                int(usage["completion_tokens"])
                if usage.get("completion_tokens") is not None
                else None
            ),
            validators=validator_results,
            error=None,
            content=content if store_content else None,
            metadata=case.metadata,
        )
    except (SwitchyardError, ValueError, TypeError, KeyError) as error:
        return CandidateRun(
            case_id=case.id,
            target=target,
            success=False,
            score=None,
            latency_ms=(time.perf_counter() - started) * 1_000,
            response_model=None,
            input_tokens=None,
            output_tokens=None,
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


def summarize(runs: Sequence[CandidateRun]) -> dict[str, object]:
    targets = sorted({run.target for run in runs})
    by_target: dict[str, object] = {}
    for target in targets:
        selected = [run for run in runs if run.target == target]
        successful = [run for run in selected if run.success]
        scored = [run for run in successful if run.score is not None]
        latencies = [run.latency_ms for run in successful]
        input_tokens = [
            run.input_tokens for run in successful if run.input_tokens is not None
        ]
        output_tokens = [
            run.output_tokens for run in successful if run.output_tokens is not None
        ]
        by_target[target] = {
            "runs": len(selected),
            "successful_calls": len(successful),
            "call_success_rate": round(len(successful) / len(selected), 6),
            "validator_scored_runs": len(scored),
            "average_validator_score": (
                round(statistics.mean(run.score for run in scored), 6)
                if scored
                else None
            ),
            "all_validators_pass_rate": (
                round(sum(run.score == 1.0 for run in scored) / len(scored), 6)
                if scored
                else None
            ),
            "latency_ms": {
                "p50": round(statistics.median(latencies), 3) if latencies else None,
                "p95": (
                    round(value, 3)
                    if (value := _percentile(latencies, 0.95)) is not None
                    else None
                ),
            },
            "total_input_tokens": sum(input_tokens),
            "total_output_tokens": sum(output_tokens),
            "errors": Counter(
                run.error or "unknown" for run in selected if not run.success
            ),
        }
    return {
        "schema_version": "switchyard-live-eval-summary-v1",
        "total_runs": len(runs),
        "targets": by_target,
        "caveat": (
            "Deterministic validators cover only cases with machine-checkable outcomes. "
            "Open-ended tasks require an approved judge or human evaluation rubric."
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
) -> dict[str, object]:
    if not targets:
        raise ValueError("at least one target is required")
    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero")
    result_path = Path(output_path)
    completed: dict[tuple[str, str], CandidateRun] = {}
    if resume and result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                validators = tuple(
                    ValidatorResult(**item) for item in raw.get("validators", [])
                )
                raw["validators"] = validators
                completed[(str(raw["case_id"]), str(raw["target"]))] = CandidateRun(
                    **raw
                )

    pending = [
        (case, target)
        for case in cases
        for target in targets
        if (case.id, target) not in completed
    ]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                client,
                case,
                target,
                store_content=store_content,
            ): (case.id, target)
            for case, target in pending
        }
        for future in as_completed(futures):
            run = future.result()
            completed[(run.case_id, run.target)] = run

    ordered = [completed[(case.id, target)] for case in cases for target in targets]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        "".join(
            json.dumps(run.as_dict(), ensure_ascii=False) + "\n" for run in ordered
        ),
        encoding="utf-8",
    )
    summary = summarize(ordered)
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
