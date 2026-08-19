from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tomllib
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, build_opener

from .catalog import load_catalog
from .switchyard import DELEGATED_PROFILES

APPROVED_SWITCHYARD_VERSION = "0.2.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(base_url: str, path: str) -> dict[str, Any]:
    opener = build_opener(ProxyHandler({}))
    with opener.open(f"{base_url}{path}", timeout=2.0) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"Switchyard {path} returned a non-object response")
    return value


def validate_switchyard_contract(
    config_path: str | Path = "config/switchyard_routes.toml",
) -> dict[str, object]:
    config = Path(config_path)
    try:
        installed_version = version("nemo-switchyard")
    except PackageNotFoundError as error:
        raise RuntimeError("install the project switchyard extra first") from error
    if installed_version != APPROVED_SWITCHYARD_VERSION:
        raise RuntimeError(
            f"expected nemo-switchyard {APPROVED_SWITCHYARD_VERSION}, "
            f"found {installed_version}"
        )

    with config.open("rb") as handle:
        parsed = tomllib.load(handle)
    targets = parsed.get("targets", {})
    if not isinstance(targets, dict) or not targets:
        raise ValueError("Switchyard config must define targets")
    upstream_keys: list[tuple[str, str]] = []
    for name, target in targets.items():
        if not isinstance(target, dict):
            raise ValueError(f"target {name!r} must be a table")
        upstream_keys.append((str(target.get("llm_client")), str(target.get("id"))))
    if len(upstream_keys) != len(set(upstream_keys)):
        raise ValueError(
            "Switchyard targets must not duplicate an upstream client/model pair"
        )

    from switchyard_rust.server import Server

    server = Server(config, port=0)
    try:
        health = _read_json(server.base_url, "/health")
        models = _read_json(server.base_url, "/v1/models")
        stats = _read_json(server.base_url, "/v1/stats")
    finally:
        server.close()
    if health.get("status") != "ok":
        raise RuntimeError(f"Switchyard health failed: {health}")
    raw_models = models.get("data", [])
    if not isinstance(raw_models, list):
        raise ValueError("Switchyard model list has an invalid data field")
    route_ids = {str(item.get("id")) for item in raw_models if isinstance(item, dict)}
    expected_routes = {model.switchyard_target for model in load_catalog()} | {
        profile.switchyard_model for profile in DELEGATED_PROFILES.values()
    }
    missing_routes = sorted(expected_routes - route_ids)
    if missing_routes:
        raise RuntimeError(f"Switchyard did not expose routes: {missing_routes}")
    if int(stats.get("total_requests", -1)) != 0:
        raise RuntimeError(
            "fresh Switchyard contract server had non-zero request stats"
        )

    return {
        "schema_version": "model-router-switchyard-contract-v1",
        "validated_at": datetime.now(UTC).isoformat(),
        "passed": True,
        "switchyard_version": installed_version,
        "config": str(config),
        "config_sha256": _sha256_file(config),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "expected_routes": sorted(expected_routes),
        "exposed_routes": sorted(route_ids),
        "health": health,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the pinned Switchyard package and route contract"
    )
    parser.add_argument("--config", default="config/switchyard_routes.toml")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = validate_switchyard_contract(args.config)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
