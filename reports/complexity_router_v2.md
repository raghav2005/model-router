# Learned Complexity Router — Training and Evaluation Report

**Generated:** 2026-08-13T08:04:01.369340+00:00

## Result summary

The trained model predicts the audited Level 1–5 complexity label from the flattened conversation text. It does **not** prove which deployed LLM will produce the best answer; that requires response-level model evaluations.

| Strategy | Accuracy | Macro F1 | MAE | Tier success proxy | Tier under-route | Relative cost index |
|---|---:|---:|---:|---:|---:|---:|
| learned_argmax | 91.14% | 0.911 | 0.130 | 96.44% | 3.56% | 3.102 |
| learned_expected | 88.59% | 0.887 | 0.143 | 96.00% | 4.00% | 3.099 |
| learned_conservative_p80 | 89.32% | 0.891 | 0.152 | 98.29% | 1.71% | 3.212 |
| heuristic | 25.57% | 0.225 | 1.227 | 46.10% | 53.90% | 1.700 |
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

- Exact five-level accuracy: 91.14%
- Macro F1: 0.911
- Mean absolute level error: 0.130
- Within one level: 96.84%
- Severe error rate: 3.16%
- Collapsed tier accuracy: 93.53%
- Collapsed tier under-route rate: 3.56%
- Expected calibration error: 0.023
- Local inference latency: median 115.6 µs; p95 168.5 µs over 1,000 prompts

## End-to-end router simulation

This section runs the complete policy router, including hard gates and catalogue scoring, over the held-out prompts. Cost uses the illustrative catalogue and a fixed 500-token output allowance.

| Router mode | Tier success proxy | Tier under-route | Tier over-route | Est. cost/request | Saving vs capable | Route mix |
|---|---:|---:|---:|---:|---:|---|
| heuristic | 44.91% | 55.09% | 9.14% | $0.00533 | 65.77% | balanced: 3656, capable: 140, efficient: 4726 |
| learned_argmax | 97.32% | 2.68% | 15.45% | $0.01035 | 33.51% | balanced: 3100, capable: 3767, efficient: 1655 |
| hybrid_argmax | 97.13% | 2.87% | 9.63% | $0.01007 | 35.29% | balanced: 2584, capable: 3769, efficient: 2169 |
| hybrid_conservative_p80 | 98.47% | 1.53% | 11.24% | $0.01033 | 33.64% | balanced: 2386, capable: 4019, efficient: 2117 |
| hybrid_adaptive_p15 | 97.43% | 2.57% | 9.83% | $0.01012 | 35.00% | balanced: 2551, capable: 3812, efficient: 2159 |
| hybrid_tier_risk_p15 | 98.73% | 1.27% | 11.84% | $0.01041 | 33.11% | balanced: 2334, capable: 4096, efficient: 2092 |
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
| 1 | 1183 | 60 | 15 | 26 | 7 |
| 2 | 65 | 1374 | 55 | 45 | 11 |
| 3 | 22 | 79 | 1707 | 59 | 30 |
| 4 | 8 | 36 | 89 | 1814 | 38 |
| 5 | 7 | 12 | 50 | 41 | 1689 |

## Release recommendation

Use this artifact in shadow mode as a complexity signal behind the existing hard gates. Do not promote it as the sole production model selector until response-level candidate benchmarks and real-traffic drift tests pass.


## Novel multi-turn context slice

- Source rows: 8,284
- Rows overlapping the primary dataset and excluded: 7,829
- Novel rows evaluated: 455
- Exact accuracy: 57.14%
- Macro F1: 0.548
- Tier under-route rate: 18.68%

**Warning:** This slice is generated from the same synthetic process and is not a production-distribution benchmark. Overlapping rows were excluded. The large drop from the hash-held-out test is evidence that the main test result is not sufficient for a production release.