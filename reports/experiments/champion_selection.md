# Complexity Router Champion Selection

Two training datasets were evaluated with the same model family, vectorizer, weighting, calibration, and deterministic split method.

| Candidate | Training source | Rows |
|---|---|---:|
| Full audited | `routing_dataset_100k_valid_only.jsonl` | 86,967 |
| Equal-level balanced | `routing_dataset_100k_valid_balanced_equal_levels.jsonl` | 66,560 |

## Fair common-set comparison

Both artifacts were evaluated on exactly the same examples. The common primary test uses the normalized-prompt hash test bucket from the full audited dataset. The external slice contains only the 455 multi-turn prompts that do not occur anywhere in the full audited dataset.

| Candidate | Slice | Rows | Accuracy | Macro F1 | Tier under-route | NLL |
|---|---|---:|---:|---:|---:|---:|
| Full audited | Common primary test | 8,522 | 90.93% | 0.909 | 3.64% | 0.268 |
| Balanced | Common primary test | 8,522 | 89.62% | 0.897 | 4.54% | 0.299 |
| Full audited | Truly novel multi-turn | 455 | 57.14% | 0.549 | 18.68% | 1.619 |
| Balanced | Truly novel multi-turn | 455 | 54.29% | 0.524 | 20.44% | 1.691 |

## Decision

The full audited dataset is the champion because it is better on accuracy, macro F1, tier under-routing, and negative log-likelihood on both fair comparison slices.

The balanced dataset remains useful as an experimental challenger, but uniform class priors already prevent the full model from simply reproducing the synthetic dataset's class distribution. Downsampling removes more than 20,000 valid examples without improving the common-set results.

Neither artifact is approved as a sole production selector. The distribution-shift result remains well below the internal hash-held-out result and requires representative real-traffic and response-level evaluation.
