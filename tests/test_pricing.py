from __future__ import annotations

import unittest
from datetime import UTC, datetime

from model_router.catalog import catalog_sha256
from model_router.pricing import parse_model_markdown, verify_catalog_pricing

MODEL_PAGE = """# GPT-5.6 Terra

Model ID: `gpt-5.6-terra`

## Model details

- 1,050,000 context window
- 128,000 max output tokens

## Pricing

| Metric | Price | Unit |
| --- | ---: | --- |
| Input | $2 | 1M tokens |
| Cached input | $0.2 | 1M tokens |
| Output | $12 | 1M tokens |

- Prompts with >272K input tokens are priced at 2x input and 1.5x output for the full request.
- Cache writes are billed at 1.25x the uncached input token rate.
"""


class PricingTests(unittest.TestCase):
    def test_parses_official_model_markdown(self) -> None:
        pricing = parse_model_markdown(MODEL_PAGE)
        self.assertEqual(pricing.model_id, "gpt-5.6-terra")
        self.assertEqual(pricing.input_price_per_million, 2.0)
        self.assertEqual(pricing.cached_input_price_per_million, 0.2)
        self.assertEqual(pricing.output_price_per_million, 12.0)
        self.assertEqual(pricing.long_context_threshold_tokens, 272_000)
        self.assertEqual(pricing.cache_write_multiplier, 1.25)

    def test_verifier_detects_source_mismatch(self) -> None:
        pages = {
            "gpt-5.6-sol": MODEL_PAGE.replace("terra", "sol")
            .replace("| Input | $2 |", "| Input | $5 |")
            .replace("| Cached input | $0.2 |", "| Cached input | $0.5 |")
            .replace("| Output | $12 |", "| Output | $30 |"),
            "gpt-5.6-terra": MODEL_PAGE,
            "gpt-5.6-luna": MODEL_PAGE.replace("terra", "luna")
            .replace("| Input | $2 |", "| Input | $0.2 |")
            .replace("| Cached input | $0.2 |", "| Cached input | $0.02 |")
            .replace("| Output | $12 |", "| Output | $1.2 |"),
        }

        def fetcher(url: str, _: float) -> str:
            return pages[url.rsplit("/", 1)[-1]]

        report = verify_catalog_pricing(
            fetcher=fetcher, now=datetime(2026, 8, 17, tzinfo=UTC)
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["catalog_sha256"], catalog_sha256())

        stale_luna = pages["gpt-5.6-luna"].replace(
            "| Output | $1.2 |", "| Output | $6 |"
        )
        pages["gpt-5.6-luna"] = stale_luna
        failed = verify_catalog_pricing(fetcher=fetcher)
        self.assertFalse(failed["passed"])
        luna = next(item for item in failed["models"] if item["role"] == "efficient")
        self.assertTrue(any("output_price" in item for item in luna["mismatches"]))


if __name__ == "__main__":
    unittest.main()
