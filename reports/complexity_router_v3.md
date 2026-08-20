# Learned Complexity Router — Training and Evaluation Report

**Generated:** 2026-08-20T10:05:37.294562+00:00

## Result summary

The trained model predicts the audited Level 1–5 complexity label from the flattened conversation text. It does **not** prove which deployed LLM will produce the best answer; that requires response-level model evaluations.

| Strategy | Accuracy | Macro F1 | MAE | Tier success proxy | Tier under-route | Relative cost index |
|---|---:|---:|---:|---:|---:|---:|
| learned_argmax | 91.95% | 0.919 | 0.119 | 97.21% | 2.79% | 3.126 |
| learned_expected | 89.42% | 0.894 | 0.132 | 96.81% | 3.19% | 3.125 |
| learned_conservative_p80 | 89.39% | 0.892 | 0.155 | 98.67% | 1.33% | 3.239 |
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
- Training-only augmented views: 139,682; original validation, test, and external prompts were unchanged.

## Learned model test metrics

- Exact five-level accuracy: 91.95%
- Macro F1: 0.919
- Mean absolute level error: 0.119
- Within one level: 97.09%
- Severe error rate: 2.91%
- Collapsed tier accuracy: 94.22%
- Collapsed tier under-route rate: 2.79%
- Expected calibration error: 0.014
- Local inference latency: median 109.2 µs; p95 156.9 µs over 1,000 prompts

## End-to-end router simulation

This section runs the complete policy router, including hard gates and catalogue scoring, over the held-out prompts. Cost uses the illustrative catalogue and a fixed 500-token output allowance.

| Router mode | Tier success proxy | Tier under-route | Tier over-route | Est. cost/request | Saving vs capable | Route mix |
|---|---:|---:|---:|---:|---:|---|
| heuristic | 44.91% | 55.09% | 9.14% | $0.00328 | 78.92% | balanced: 3656, capable: 140, efficient: 4726 |
| learned_argmax | 97.96% | 2.04% | 16.09% | $0.00941 | 39.56% | balanced: 3111, capable: 3821, efficient: 1590 |
| hybrid_argmax | 97.77% | 2.23% | 9.72% | $0.00905 | 41.89% | balanced: 2550, capable: 3823, efficient: 2149 |
| hybrid_conservative_p80 | 98.79% | 1.21% | 11.50% | $0.00936 | 39.87% | balanced: 2355, capable: 4076, efficient: 2091 |
| hybrid_adaptive_p15 | 97.97% | 2.03% | 9.88% | $0.00909 | 41.62% | balanced: 2524, capable: 3857, efficient: 2141 |
| hybrid_tier_risk_p15 | 99.12% | 0.88% | 12.19% | $0.00946 | 39.20% | balanced: 2298, capable: 4157, efficient: 2067 |
| hybrid_adaptive_validation_selected | 97.81% | 2.19% | 9.81% | $0.00906 | 41.80% | balanced: 2538, capable: 3835, efficient: 2149 |
| always_capable_reference | 100.00% | 0.00% | 55.60% | $0.01556 | 0.00% | capable: 8522 |

## Validation-selected adaptive policy

The confidence threshold and posterior risk tolerance were selected using only the validation split. The objective minimises estimated cost while respecting the declared validation under-routing cap.

- Confidence threshold: 0.30
- Posterior under-route tolerance: 0.10
- Validation tier under-route: 2.45%
- Validation tier over-route: 9.84%
- Validation estimated cost/request: $0.00901
- Feasible candidates: 28/28
- External confirmation: not passed
- Decision: retain current production default and continue shadow evaluation
- This is a validation candidate, not a deployment recommendation.

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
| 1 | 1181 | 57 | 16 | 30 | 7 |
| 2 | 60 | 1379 | 52 | 44 | 15 |
| 3 | 19 | 62 | 1725 | 56 | 35 |
| 4 | 4 | 25 | 75 | 1842 | 39 |
| 5 | 3 | 8 | 42 | 37 | 1709 |

## Release recommendation

Use this artifact in shadow mode as a complexity signal behind the existing hard gates. Do not promote it as the sole production model selector until response-level candidate benchmarks and real-traffic drift tests pass.


## Novel multi-turn context slice

- Source rows: 8,284
- Rows overlapping the primary dataset and excluded: 7,829
- Novel rows evaluated: 455
- Exact accuracy: 59.12%
- Macro F1: 0.572
- Tier under-route rate: 17.36%

**Warning:** This slice is generated from the same synthetic process and is not a production-distribution benchmark. Overlapping rows were excluded. The large drop from the hash-held-out test is evidence that the main test result is not sufficient for a production release.