# Learned Complexity Router — Training and Evaluation Report

**Generated:** 2026-08-11T08:28:14.117069+00:00

## Result summary

The trained model predicts the audited Level 1–5 complexity label from the flattened conversation text. It does **not** prove which deployed LLM will produce the best answer; that requires response-level model evaluations.

| Strategy | Accuracy | Macro F1 | MAE | Tier success proxy | Tier under-route | Relative cost index |
|---|---:|---:|---:|---:|---:|---:|
| learned_argmax | 89.64% | 0.897 | 0.153 | 95.84% | 4.16% | 2.894 |
| learned_expected | 85.66% | 0.859 | 0.175 | 95.06% | 4.94% | 2.891 |
| learned_conservative_p80 | 85.77% | 0.857 | 0.205 | 98.52% | 1.48% | 3.079 |
| heuristic | 27.17% | 0.232 | 1.173 | 51.40% | 48.60% | 1.667 |
| majority_level | 19.66% | 0.066 | 1.999 | 100.00% | 0.00% | 5.000 |
| always_efficient_tier | 19.74% | 0.066 | 1.399 | 39.65% | 60.35% | 1.000 |
| always_balanced_tier | 20.37% | 0.068 | 1.192 | 60.02% | 39.98% | 2.500 |
| always_capable_tier | 19.66% | 0.066 | 1.999 | 100.00% | 0.00% | 5.000 |

## Dataset and split

- Source file: `routing_dataset_100k_valid_balanced_equal_levels.jsonl`
- SHA-256: `78ec17032a7995f9fa12cd2ebed6e34c3e3ef277f615673b35b5ec93bed05cc1`
- Valid rows: 66,560
- Train: 53,526
- Validation: 6,550
- Test: 6,484
- Split method: normalized-prompt SHA-256 hash; duplicates would remain in one split.
- Duplicate prompts: 0
- Borderline rows: 33,309; down-weighted during training.

## Learned model test metrics

- Exact five-level accuracy: 89.64%
- Macro F1: 0.897
- Mean absolute level error: 0.153
- Within one level: 96.36%
- Severe error rate: 3.64%
- Collapsed tier accuracy: 92.50%
- Collapsed tier under-route rate: 4.16%
- Expected calibration error: 0.012
- Local inference latency: median 74.2 µs; p95 115.5 µs over 1,000 prompts

## End-to-end router simulation

This section runs the complete policy router, including hard gates and catalogue scoring, over the held-out prompts. Cost uses the illustrative catalogue and a fixed 500-token output allowance.

| Router mode | Tier success proxy | Tier under-route | Tier over-route | Est. cost/request | Saving vs capable | Route mix |
|---|---:|---:|---:|---:|---:|---|
| heuristic | 50.31% | 49.69% | 10.58% | $0.00524 | 66.32% | balanced: 2681, capable: 100, efficient: 3703 |
| learned_argmax | 96.88% | 3.12% | 16.95% | $0.00976 | 37.26% | balanced: 2340, capable: 2569, efficient: 1575 |
| hybrid_argmax | 96.70% | 3.30% | 11.23% | $0.00948 | 39.03% | balanced: 1953, capable: 2570, efficient: 1961 |
| hybrid_conservative_p80 | 98.75% | 1.25% | 13.79% | $0.00990 | 36.34% | balanced: 1737, capable: 2869, efficient: 1878 |
| always_capable_reference | 100.00% | 0.00% | 60.02% | $0.01555 | 0.00% | capable: 6484 |

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
| 1 | 1175 | 67 | 15 | 29 | 5 |
| 2 | 55 | 1129 | 51 | 30 | 15 |
| 3 | 21 | 73 | 1156 | 43 | 28 |
| 4 | 7 | 27 | 83 | 1177 | 23 |
| 5 | 4 | 14 | 41 | 41 | 1175 |

## Release recommendation

Use this artifact in shadow mode as a complexity signal behind the existing hard gates. Do not promote it as the sole production model selector until response-level candidate benchmarks and real-traffic drift tests pass.


## Novel multi-turn context slice

- Source rows: 8,284
- Rows overlapping the primary dataset and excluded: 6,129
- Novel rows evaluated: 2,155
- Exact accuracy: 81.72%
- Macro F1: 0.729
- Tier under-route rate: 9.00%

**Warning:** This slice is generated from the same synthetic process and is not a production-distribution benchmark. Overlapping rows were excluded. The large drop from the hash-held-out test is evidence that the main test result is not sufficient for a production release.