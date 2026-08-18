from __future__ import annotations

import json
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .types import ModelProfile


def _catalog_path(path: str | Path | None = None) -> Any:
    return Path(path) if path else files("model_router").joinpath("catalog.json")


def load_catalog_document(path: str | Path | None = None) -> dict[str, Any]:
    catalog_path = _catalog_path(path)
    with catalog_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise ValueError("catalog must contain a models array")
    if not isinstance(raw.get("schema_version"), str):
        raise ValueError("catalog must declare schema_version")
    return raw


def catalog_sha256(path: str | Path | None = None) -> str:
    catalog_path = _catalog_path(path)
    with catalog_path.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def _validated_model(item: dict[str, Any]) -> ModelProfile:
    required = {
        "id",
        "provider",
        "provider_model",
        "tier",
        "input_price_per_million",
        "cached_input_price_per_million",
        "cache_write_price_per_million",
        "output_price_per_million",
        "latency_p95_ms",
        "context_window",
        "max_output_tokens",
        "capabilities",
        "skills",
        "pricing_source",
        "pricing_as_of",
        "quality_evidence",
        "latency_evidence",
    }
    missing = sorted(required - item.keys())
    if missing:
        raise ValueError(f"catalog model is missing fields: {', '.join(missing)}")
    if item["tier"] not in {1, 2, 3}:
        raise ValueError(f"model {item['id']!r} has invalid tier")
    for field_name in (
        "input_price_per_million",
        "cached_input_price_per_million",
        "cache_write_price_per_million",
        "output_price_per_million",
        "latency_p95_ms",
        "context_window",
        "max_output_tokens",
    ):
        if float(item[field_name]) < 0:
            raise ValueError(f"model {item['id']!r} has negative {field_name}")
    source = urlparse(str(item["pricing_source"]))
    if source.scheme != "https" or not source.netloc:
        raise ValueError(f"model {item['id']!r} needs an HTTPS pricing source")
    if int(item["max_output_tokens"]) > int(item["context_window"]):
        raise ValueError(f"model {item['id']!r} max output exceeds context window")
    if float(item["cached_input_price_per_million"]) > float(
        item["input_price_per_million"]
    ):
        raise ValueError(f"model {item['id']!r} cached input exceeds input price")
    if float(item["cache_write_price_per_million"]) < float(
        item["input_price_per_million"]
    ):
        raise ValueError(f"model {item['id']!r} cache write is below input price")
    if any(not 0 <= float(score) <= 1 for score in dict(item["skills"]).values()):
        raise ValueError(f"model {item['id']!r} has a skill outside [0, 1]")
    for multiplier in (
        item.get("long_context_input_multiplier", 1.0),
        item.get("long_context_output_multiplier", 1.0),
    ):
        if float(multiplier) < 1:
            raise ValueError(f"model {item['id']!r} has an invalid price multiplier")

    return ModelProfile(
        id=item["id"],
        switchyard_target=item.get("switchyard_target", item["id"]),
        provider=item["provider"],
        provider_model=item["provider_model"],
        tier=item["tier"],
        input_price_per_million=item["input_price_per_million"],
        cached_input_price_per_million=item["cached_input_price_per_million"],
        cache_write_price_per_million=item["cache_write_price_per_million"],
        output_price_per_million=item["output_price_per_million"],
        latency_p95_ms=item["latency_p95_ms"],
        context_window=item["context_window"],
        max_output_tokens=item["max_output_tokens"],
        capabilities=frozenset(item["capabilities"]),
        skills=item["skills"],
        pricing_source=item["pricing_source"],
        pricing_as_of=item["pricing_as_of"],
        quality_evidence=item["quality_evidence"],
        latency_evidence=item["latency_evidence"],
        long_context_threshold_tokens=item.get("long_context_threshold_tokens"),
        long_context_input_multiplier=item.get("long_context_input_multiplier", 1.0),
        long_context_output_multiplier=item.get("long_context_output_multiplier", 1.0),
        enabled=item.get("enabled", True),
        health=item.get("health", 1.0),
    )


def load_catalog(path: str | Path | None = None) -> list[ModelProfile]:
    raw = load_catalog_document(path)
    models = [_validated_model(item) for item in raw["models"]]
    ids = [model.id for model in models]
    targets = [model.switchyard_target for model in models]
    if len(ids) != len(set(ids)):
        raise ValueError("catalog model IDs must be unique")
    if len(targets) != len(set(targets)):
        raise ValueError("catalog Switchyard targets must be unique")
    if sorted(model.tier for model in models) != [1, 2, 3]:
        raise ValueError("catalog must contain exactly one model in each tier")
    return models
