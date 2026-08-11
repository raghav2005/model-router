# Learned Complexity Router — Training and Evaluation Report

**Generated:** 2026-08-11T08:27:23.654416+00:00

## Result summary

The trained model predicts the audited Level 1–5 complexity label from the flattened conversation text. It does **not** prove which deployed LLM will produce the best answer; that requires response-level model evaluations.

| Strategy | Accuracy | Macro F1 | MAE | Tier success proxy | Tier under-route | Relative cost index |
|---|---:|---:|---:|---:|---:|---:|
| learned_argmax | 90.93% | 0.909 | 0.132 | 96.36% | 3.64% | 3.101 |
| learned_expected | 87.03% | 0.872 | 0.158 | 95.35% | 4.65% | 3.090 |
| learned_conservative_p80 | 87.48% | 0.872 | 0.178 | 98.60% | 1.40% | 3.262 |
| heuristic | 25.57% | 0.225 | 1.227 | 46.09% | 53.91% | 1.700 |
| majority_level | 23.29% | 0.076 | 1.252 | 100.00% | 0.00% | 5.000 |
| always_efficient_tier | 18.19% | 0.062 | 1.473 | 33.34% | 66.66% | 1.000 |
| always_balanced_tier | 22.26% | 0.073 | 1.140 | 55.60% | 44.40% | 2.500 |
| always_capable_tier | 21.11% | 0.070 | 1.830 | 100.00% | 0.00% | 5.000 |

## Dataset and split

- Source file: `routing_dataset_100k_valid_only.jsonl`
- SHA-256: `bca11574006fac48d17f1efb67cd471822911e366b9a580755b68964499aa30e`
- Valid rows: 86,967
- Train: 69,841
- Validation: 8,604
- Test: 8,522
- Split method: normalized-prompt SHA-256 hash; duplicates would remain in one split.
- Duplicate prompts: 0
- Borderline rows: 42,759; down-weighted during training.

## Learned model test metrics

- Exact five-level accuracy: 90.93%
- Macro F1: 0.909
- Mean absolute level error: 0.132
- Within one level: 96.84%
- Severe error rate: 3.16%
- Collapsed tier accuracy: 93.43%
- Collapsed tier under-route rate: 3.64%
- Expected calibration error: 0.018
- Local inference latency: median 72.9 µs; p95 108.4 µs over 1,000 prompts

## End-to-end router simulation

This section runs the complete policy router, including hard gates and catalogue scoring, over the held-out prompts. Cost uses the illustrative catalogue and a fixed 500-token output allowance.

| Router mode | Tier success proxy | Tier under-route | Tier over-route | Est. cost/request | Saving vs capable | Route mix |
|---|---:|---:|---:|---:|---:|---|
| heuristic | 44.90% | 55.10% | 9.14% | $0.00533 | 65.78% | balanced: 3655, capable: 140, efficient: 4727 |
| learned_argmax | 97.29% | 2.71% | 15.51% | $0.01035 | 33.50% | balanced: 3097, capable: 3770, efficient: 1655 |
| hybrid_argmax | 97.11% | 2.89% | 10.02% | $0.01009 | 35.19% | balanced: 2613, capable: 3771, efficient: 2138 |
| hybrid_conservative_p80 | 98.71% | 1.29% | 12.39% | $0.01045 | 32.87% | balanced: 2345, capable: 4119, efficient: 2058 |
| always_capable_reference | 100.00% | 0.00% | 55.60% | $0.01556 | 0.00% | capable: 8522 |

## Important interpretation

- The prompts were synthetically generated to match assigned complexity levels, so lexical cues may make the held-out task easier than real traffic.
- The labels were audited by another LLM, not verified through candidate-model responses.
- `borderline` rows are retained but receive lower training weight.
- The relative cost index is a three-tier planning proxy, not provider billing.
- The conservative policy intentionally trades higher cost for a lower under-routing rate.
- A production release still requires a frozen set of real requests executed against every candidate model.

## Confusion matrix

Rows are true levels and columns are predicted levels.

| True \ Predicted | 1 | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1182 | 61 | 16 | 25 | 7 |
| 2 | 67 | 1374 | 53 | 45 | 11 |
| 3 | 22 | 84 | 1698 | 64 | 29 |
| 4 | 8 | 36 | 90 | 1812 | 39 |
| 5 | 7 | 15 | 48 | 46 | 1683 |

## Release recommendation

Use this artifact in shadow mode as a complexity signal behind the existing hard gates. Do not promote it as the sole production model selector until response-level candidate benchmarks and real-traffic drift tests pass.


## Novel multi-turn context slice

- Source rows: 8,284
- Rows overlapping the primary dataset and excluded: 7,829
- Novel rows evaluated: 455
- Exact accuracy: 57.14%
- Macro F1: 0.549
- Tier under-route rate: 18.68%

**Warning:** This slice is generated from the same synthetic process and is not a production-distribution benchmark. Overlapping rows were excluded. The large drop from the hash-held-out test is evidence that the main test result is not sufficient for a production release.