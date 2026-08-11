from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from .types import ModelProfile


def load_catalog(path: str | Path | None = None) -> list[ModelProfile]:
    catalog_path = (
        Path(path) if path else files("model_router").joinpath("catalog.json")
    )
    with catalog_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [
        ModelProfile(
            id=item["id"],
            switchyard_target=item.get("switchyard_target", item["id"]),
            tier=item["tier"],
            input_price_per_million=item["input_price_per_million"],
            output_price_per_million=item["output_price_per_million"],
            latency_p95_ms=item["latency_p95_ms"],
            context_window=item["context_window"],
            capabilities=frozenset(item["capabilities"]),
            skills=item["skills"],
            enabled=item.get("enabled", True),
            health=item.get("health", 1.0),
        )
        for item in raw["models"]
    ]
