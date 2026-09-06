from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import catalog_sha256, load_catalog
from .types import ModelProfile

SCHEMA_VERSION = "model-router-workload-evidence-v1"
REQUIRED_USE_CASES = ("general_qa", "coding", "reasoning")


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _distribution(value: object) -> dict[str, float | None]:
    raw = value if isinstance(value, Mapping) else {}
    return {
        percentile: (
            float(raw[percentile]) if raw.get(percentile) is not None else None
        )
        for percentile in ("p50", "p95", "p99")
    }


def _quality(value: Mapping[str, Any]) -> dict[str, object]:
    slices = value.get("slices", {})
    use_cases = slices.get("use_case", {}) if isinstance(slices, Mapping) else {}
    by_use_case: dict[str, object] = {}
    for use_case in REQUIRED_USE_CASES:
        raw = use_cases.get(use_case, {}) if isinstance(use_cases, Mapping) else {}
        by_use_case[use_case] = {
            "unique_cases": int(raw.get("unique_cases", 0)),
            "scored_runs": int(raw.get("validator_scored_runs", 0)),
            "all_validators_pass_rate": raw.get("all_validators_pass_rate"),
            "all_validators_pass_rate_wilson_95": raw.get(
                "all_validators_pass_rate_wilson_95"
            ),
        }
    return {
        "scored_runs": int(value.get("validator_scored_runs", 0)),
        "all_validators_pass_rate": value.get("all_validators_pass_rate"),
        "all_validators_pass_rate_wilson_95": value.get(
            "all_validators_pass_rate_wilson_95"
        ),
        "by_use_case": by_use_case,
    }


def _role_complete(role: Mapping[str, Any]) -> bool:
    quality = role.get("quality", {})
    latency = role.get("latency_ms", {})
    by_use_case = quality.get("by_use_case", {}) if isinstance(quality, Mapping) else {}
    return (
        int(role.get("unique_cases", 0)) > 0
        and isinstance(quality, Mapping)
        and int(quality.get("scored_runs", 0)) > 0
        and quality.get("all_validators_pass_rate") is not None
        and isinstance(quality.get("all_validators_pass_rate_wilson_95"), Mapping)
        and isinstance(latency, Mapping)
        and latency.get("p95") is not None
        and all(
            isinstance(by_use_case.get(use_case), Mapping)
            and int(by_use_case[use_case].get("unique_cases", 0)) > 0
            and by_use_case[use_case].get("all_validators_pass_rate") is not None
            for use_case in REQUIRED_USE_CASES
        )
    )


def build_workload_evidence(
    live_summary_path: str | Path,
    *,
    catalog_path: str | Path | None = None,
) -> dict[str, object]:
    live = _read_json(live_summary_path)
    if live.get("schema_version") != "switchyard-live-eval-summary-v3":
        raise ValueError("unsupported live-evaluation summary schema")
    provenance = live.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("live-evaluation provenance is missing")
    expected_catalog_sha = catalog_sha256(catalog_path)
    if provenance.get("catalog_sha256") != expected_catalog_sha:
        raise ValueError("live summary does not match the selected catalogue")
    targets = live.get("targets", {})
    if not isinstance(targets, Mapping):
        raise ValueError("live summary targets must be an object")

    roles: dict[str, object] = {}
    models = load_catalog(catalog_path)
    for model in models:
        raw = targets.get(model.switchyard_target)
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"live summary is missing target {model.switchyard_target!r}"
            )
        role = {
            "target": model.switchyard_target,
            "provider": model.provider,
            "provider_model": model.provider_model,
            "unique_cases": int(raw.get("unique_cases", 0)),
            "quality": _quality(raw),
            "latency_ms": _distribution(raw.get("latency_ms")),
            "ttft_ms": _distribution(raw.get("ttft_ms")),
            "call_success_rate": raw.get("call_success_rate"),
            "call_success_rate_wilson_95": raw.get("call_success_rate_wilson_95"),
            "estimated_total_cost_usd": raw.get("estimated_total_cost_usd"),
            "response_model_mismatch_count": int(
                raw.get("response_model_mismatch_count", 0)
            ),
        }
        role["complete"] = _role_complete(role)
        roles[model.id] = role

    complete = all(
        isinstance(role, Mapping) and role.get("complete") is True
        for role in roles.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "complete": complete,
        "source": {
            "live_summary_sha256": sha256_file(live_summary_path),
            "live_summary_schema_version": live.get("schema_version"),
            "catalog_sha256": expected_catalog_sha,
            "case_set_sha256": provenance.get("case_set_sha256"),
            "switchyard_revision": provenance.get("switchyard_revision"),
            "benchmark_fingerprint": provenance.get("benchmark_fingerprint"),
        },
        "roles": roles,
        "privacy": {
            "contains_prompt_content": False,
            "contains_response_content": False,
            "contains_aggregate_measurements_only": True,
        },
        "approval": {
            "status": "candidate",
            "instruction": (
                "Review this aggregate artifact, then record its exact SHA-256 in "
                "approved_workload_evidence_sha256 in the release policy."
            ),
        },
        "caveats": [
            "Completeness proves measurements exist; release thresholds are evaluated separately.",
            "The case set must be representative and privacy-approved before its digest is approved.",
            "Open-ended work requires calibrated human or approved judge evaluation in addition to deterministic validators.",
        ],
    }


def validate_workload_evidence(
    evidence: Mapping[str, Any],
    *,
    live_summary_path: str | Path,
    catalog_path: str | Path | None = None,
) -> tuple[bool, str]:
    source = evidence.get("source", {})
    roles = evidence.get("roles", {})
    models = load_catalog(catalog_path)
    expected_roles = {model.id: model for model in models}
    if evidence.get("schema_version") != SCHEMA_VERSION:
        return False, "unsupported workload-evidence schema"
    if evidence.get("complete") is not True:
        return False, "workload evidence is incomplete"
    if not isinstance(source, Mapping) or not isinstance(roles, Mapping):
        return False, "workload evidence is malformed"
    if source.get("live_summary_sha256") != sha256_file(live_summary_path):
        return False, "workload evidence does not match the live summary"
    if source.get("catalog_sha256") != catalog_sha256(catalog_path):
        return False, "workload evidence does not match the catalogue"
    if set(roles) != set(expected_roles):
        return False, "workload evidence does not cover every catalogue role"
    for role_name, model in expected_roles.items():
        role = roles.get(role_name)
        if not isinstance(role, Mapping) or not _role_complete(role):
            return False, f"workload evidence for {role_name} is incomplete"
        if (
            role.get("target") != model.switchyard_target
            or role.get("provider") != model.provider
            or role.get("provider_model") != model.provider_model
        ):
            return False, f"workload evidence identity differs for {role_name}"
    return True, "workload evidence matches the exact live summary and catalogue"


def apply_workload_evidence(
    models: Sequence[ModelProfile], evidence: Mapping[str, Any]
) -> list[ModelProfile]:
    """Overlay approved empirical quality and latency on stable catalogue entries."""
    roles = evidence.get("roles", {})
    if not isinstance(roles, Mapping):
        raise ValueError("workload evidence roles must be an object")
    updated: list[ModelProfile] = []
    for model in models:
        role = roles.get(model.id)
        if not isinstance(role, Mapping) or not _role_complete(role):
            raise ValueError(f"workload evidence for {model.id} is incomplete")
        quality = role["quality"]
        latency = role["latency_ms"]
        assert isinstance(quality, Mapping) and isinstance(latency, Mapping)
        by_use_case = quality["by_use_case"]
        assert isinstance(by_use_case, Mapping)
        skills = dict(model.skills)
        for use_case in REQUIRED_USE_CASES:
            measurement = by_use_case[use_case]
            assert isinstance(measurement, Mapping)
            skills[use_case] = float(measurement["all_validators_pass_rate"])
        updated.append(
            replace(
                model,
                skills=skills,
                latency_p95_ms=math.ceil(float(latency["p95"])),
                quality_evidence="workload_measured",
                latency_evidence="workload_measured",
            )
        )
    return updated


def write_workload_evidence(path: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
