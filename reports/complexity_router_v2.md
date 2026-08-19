# Learned Complexity Router — Training and Evaluation Report

**Generated:** 2026-08-18T09:34:53.948121+00:00

## Result summary

The trained model predicts the audited Level 1–5 complexity label from the flattened conversation text. It does **not** prove which deployed LLM will produce the best answer; that requires response-level model evaluations.

| Strategy | Accuracy | Macro F1 | MAE | Tier success proxy | Tier under-route | Relative cost index |
|---|---:|---:|---:|---:|---:|---:|
| learned_argmax | 91.57% | 0.915 | 0.124 | 96.87% | 3.13% | 3.118 |
| learned_expected | 88.90% | 0.889 | 0.138 | 96.44% | 3.56% | 3.113 |
| learned_conservative_p80 | 89.17% | 0.890 | 0.157 | 98.57% | 1.43% | 3.237 |
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
- Training-only augmented views: 69,841; original validation, test, and external prompts were unchanged.

## Learned model test metrics

- Exact five-level accuracy: 91.57%
- Macro F1: 0.915
- Mean absolute level error: 0.124
- Within one level: 96.93%
- Severe error rate: 3.07%
- Collapsed tier accuracy: 93.86%
- Collapsed tier under-route rate: 3.13%
- Expected calibration error: 0.015
- Local inference latency: median 168.8 µs; p95 240.6 µs over 1,000 prompts

## End-to-end router simulation

This section runs the complete policy router, including hard gates and catalogue scoring, over the held-out prompts. Cost uses the illustrative catalogue and a fixed 500-token output allowance.

| Router mode | Tier success proxy | Tier under-route | Tier over-route | Est. cost/request | Saving vs capable | Route mix |
|---|---:|---:|---:|---:|---:|---|
| heuristic | 44.91% | 55.09% | 9.14% | $0.00328 | 78.92% | balanced: 3656, capable: 140, efficient: 4726 |
| learned_argmax | 97.70% | 2.30% | 15.96% | $0.00938 | 39.76% | balanced: 3108, capable: 3804, efficient: 1610 |
| hybrid_argmax | 97.51% | 2.49% | 9.69% | $0.00902 | 42.04% | balanced: 2556, capable: 3807, efficient: 2159 |
| hybrid_conservative_p80 | 98.73% | 1.27% | 11.52% | $0.00935 | 39.91% | balanced: 2352, capable: 4073, efficient: 2097 |
| hybrid_adaptive_p15 | 97.77% | 2.23% | 9.87% | $0.00907 | 41.73% | balanced: 2526, capable: 3846, efficient: 2150 |
| hybrid_tier_risk_p15 | 98.97% | 1.03% | 12.13% | $0.00944 | 39.33% | balanced: 2307, capable: 4142, efficient: 2073 |
| hybrid_adaptive_validation_selected | 97.54% | 2.46% | 9.79% | $0.00903 | 41.96% | balanced: 2545, capable: 3818, efficient: 2159 |
| always_capable_reference | 100.00% | 0.00% | 55.60% | $0.01556 | 0.00% | capable: 8522 |

## Validation-selected adaptive policy

The confidence threshold and posterior risk tolerance were selected using only the validation split. The objective minimises estimated cost while respecting the declared validation under-routing cap.

- Confidence threshold: 0.30
- Posterior under-route tolerance: 0.10
- Validation tier under-route: 2.71%
- Validation tier over-route: 9.77%
- Validation estimated cost/request: $0.00898
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
| 1 | 1183 | 56 | 16 | 29 | 7 |
| 2 | 62 | 1377 | 53 | 44 | 14 |
| 3 | 24 | 70 | 1710 | 59 | 34 |
| 4 | 5 | 31 | 79 | 1830 | 40 |
| 5 | 3 | 10 | 45 | 37 | 1704 |

## Release recommendation

Use this artifact in shadow mode as a complexity signal behind the existing hard gates. Do not promote it as the sole production model selector until response-level candidate benchmarks and real-traffic drift tests pass.


## Novel multi-turn context slice

- Source rows: 8,284
- Rows overlapping the primary dataset and excluded: 7,829
- Novel rows evaluated: 455
- Exact accuracy: 57.80%
- Macro F1: 0.555
- Tier under-route rate: 18.02%

**Warning:** This slice is generated from the same synthetic process and is not a production-distribution benchmark. Overlapping rows were excluded. The large drop from the hash-held-out test is evidence that the main test result is not sufficient for a production release.