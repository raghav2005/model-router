from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .catalog import catalog_sha256, load_catalog

OFFICIAL_OPENAI_HOST = "developers.openai.com"
MAX_SOURCE_BYTES = 1_000_000


@dataclass(frozen=True)
class PublishedPricing:
    model_id: str
    input_price_per_million: float
    cached_input_price_per_million: float
    output_price_per_million: float
    context_window: int
    max_output_tokens: int
    long_context_threshold_tokens: int
    long_context_input_multiplier: float
    long_context_output_multiplier: float
    cache_write_multiplier: float


def _number(pattern: str, markdown: str, field: str) -> str:
    match = re.search(pattern, markdown, re.IGNORECASE | re.MULTILINE)
    if match is None:
        raise ValueError(f"official model page is missing {field}")
    return match.group(1)


def parse_model_markdown(markdown: str) -> PublishedPricing:
    """Extract only stable, explicitly published model economics and limits."""
    model_id = _number(r"^Model ID:\s*`([^`]+)`", markdown, "model ID")
    context_window = int(
        _number(r"-\s*([\d,]+)\s+context window", markdown, "context window").replace(
            ",", ""
        )
    )
    max_output_tokens = int(
        _number(
            r"-\s*([\d,]+)\s+max output tokens", markdown, "maximum output"
        ).replace(",", "")
    )

    def price(label: str) -> float:
        return float(
            _number(
                rf"^\|\s*{re.escape(label)}\s*\|\s*\$([\d.]+)\s*\|\s*1M tokens\s*\|",
                markdown,
                f"{label} price",
            )
        )

    long_match = re.search(
        r"Prompts with\s*>\s*([\d,]+)K input tokens are priced at\s*"
        r"([\d.]+)x input and\s*([\d.]+)x output",
        markdown,
        re.IGNORECASE,
    )
    if long_match is None:
        raise ValueError("official model page is missing long-context pricing")
    cache_write_multiplier = float(
        _number(
            r"Cache writes are billed at\s*([\d.]+)x the uncached input",
            markdown,
            "cache-write multiplier",
        )
    )
    return PublishedPricing(
        model_id=model_id,
        input_price_per_million=price("Input"),
        cached_input_price_per_million=price("Cached input"),
        output_price_per_million=price("Output"),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        long_context_threshold_tokens=int(long_match.group(1).replace(",", "")) * 1_000,
        long_context_input_multiplier=float(long_match.group(2)),
        long_context_output_multiplier=float(long_match.group(3)),
        cache_write_multiplier=cache_write_multiplier,
    )


def fetch_model_markdown(source_url: str, timeout_seconds: float = 15.0) -> str:
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or parsed.netloc != OFFICIAL_OPENAI_HOST:
        raise ValueError(
            "OpenAI pricing source must use official developer documentation"
        )
    markdown_url = source_url.rstrip("/") + ".md"
    request = Request(
        markdown_url,
        headers={
            "Accept": "text/markdown",
            "User-Agent": "model-router-pricing-verifier/0.6",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        final_url = urlparse(response.geturl())
        if final_url.scheme != "https" or final_url.netloc != OFFICIAL_OPENAI_HOST:
            raise ValueError("official pricing request redirected to an untrusted host")
        payload = response.read(MAX_SOURCE_BYTES + 1)
    if len(payload) > MAX_SOURCE_BYTES:
        raise ValueError("official pricing response exceeded the safety limit")
    return payload.decode("utf-8")


def _same(left: float | int | None, right: float | int) -> bool:
    return left is not None and math.isclose(float(left), float(right), abs_tol=1e-12)


def verify_catalog_pricing(
    *,
    catalog_path: str | Path | None = None,
    timeout_seconds: float = 15.0,
    fetcher: Callable[[str, float], str] = fetch_model_markdown,
    now: datetime | None = None,
) -> dict[str, object]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    results: list[dict[str, object]] = []
    for model in load_catalog(catalog_path):
        mismatches: list[str] = []
        facts: PublishedPricing | None = None
        try:
            facts = parse_model_markdown(fetcher(model.pricing_source, timeout_seconds))
            expected = {
                "model_id": model.provider_model,
                "input_price_per_million": model.input_price_per_million,
                "cached_input_price_per_million": (
                    model.cached_input_price_per_million
                ),
                "output_price_per_million": model.output_price_per_million,
                "context_window": model.context_window,
                "max_output_tokens": model.max_output_tokens,
                "long_context_threshold_tokens": (model.long_context_threshold_tokens),
                "long_context_input_multiplier": (model.long_context_input_multiplier),
                "long_context_output_multiplier": (
                    model.long_context_output_multiplier
                ),
                "cache_write_price_per_million": (model.cache_write_price_per_million),
            }
            actual: dict[str, str | float | int] = {
                **asdict(facts),
                "cache_write_price_per_million": (
                    facts.input_price_per_million * facts.cache_write_multiplier
                ),
            }
            for field, expected_value in expected.items():
                actual_value = actual[field]
                matches = (
                    expected_value == actual_value
                    if isinstance(expected_value, str)
                    else _same(expected_value, actual_value)
                )
                if not matches:
                    mismatches.append(
                        f"{field}: catalog={expected_value!r}, source={actual_value!r}"
                    )
        except (OSError, UnicodeError, ValueError) as error:
            mismatches.append(f"source verification failed: {error}")
        results.append(
            {
                "role": model.id,
                "provider_model": model.provider_model,
                "source": model.pricing_source,
                "passed": not mismatches,
                "mismatches": mismatches,
                "published": asdict(facts) if facts is not None else None,
            }
        )

    actual_now = now or datetime.now(UTC)
    return {
        "schema_version": "model-router-pricing-verification-v1",
        "verified_at": actual_now.isoformat(),
        "catalog_sha256": catalog_sha256(catalog_path),
        "passed": all(bool(result["passed"]) for result in results),
        "models": results,
    }


def write_report(path: str | Path, report: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
