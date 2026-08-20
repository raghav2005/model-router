# Complexity router artifacts

`complexity_router_v3.npz` is the current default. It is a five-class weighted
multinomial Naive Bayes model trained from the audited
`routing_dataset_100k_valid_only.jsonl` snapshot.

Artifact SHA-256: `61723945b1469461e6dc88484c5e2c0d88895501a4c240fc22e42e1e74849d9d`

Training recipe SHA-256: `a73d4d4742623dd8bbb948174e2ed542fad890954a79f1fa550b20f9cd5be7ce`

Training implementation SHA-256: `8b297237157a6f215b4c00296337aa475d372c2d90d5e93e31ae055b120384ef`

V1 and v2 are retained as frozen baselines. Git history and the matching
`reports/complexity_router_v1.*` and `reports/complexity_router_v2.*` files
preserve their original evidence.

## Provenance

- Dataset rows: 86,967
- Dataset SHA-256: `bca11574006fac48d17f1efb67cd471822911e366b9a580755b68964499aa30e`
- Train/validation/test split: deterministic normalized-prompt hash, 80/10/10
- Train rows: 69,841
- Validation rows: 8,604
- Test rows: 8,522
- Features: hashed word unigrams, word bigrams, and conversation structure
- Feature dimension: 32,768
- Audit weighting: `appropriate=1.0`; `borderline=0.65`; multiplied by audit confidence
- Class prior: uniform
- Training-only augmentation: two deterministic meaning-preserving views per row, each at 0.5 weight
- Multi-view inference: 95% complete conversation and 5% latest user turn
- Probability calibration: ensemble weight and temperature selected by validation log loss
- Adaptive policy search: 28 confidence/risk settings evaluated on validation only
- The validation-selected candidate was rejected for deployment because external confirmation failed

## Champion evidence

| Measure | v2 | v3 |
|---|---:|---:|
| Internal test accuracy | 91.57% | 91.95% |
| Internal test macro F1 | 0.915 | 0.919 |
| Internal tier under-route | 3.13% | 2.79% |
| Internal negative log loss | 0.244 | 0.236 |
| Novel multi-turn accuracy | 57.80% | 59.12% |
| Novel multi-turn tier under-route | 18.02% | 17.36% |
| Novel multi-turn negative log loss | 1.447 | 1.426 |

V3 was selected by validation accuracy, macro F1, under-routing, log loss, and
calibration error. The held-out internal and novel multi-turn slices then
confirmed the improvement. The external calibration error is still poor and the
novel slice remains too small; the artifact therefore remains shadow-only.

## Intended use

Use the artifact as one prompt-complexity signal behind capability, budget,
latency, context, health, and governance gates. `hybrid` mode combines the
learned estimate with deterministic risk and capability features. The default
adaptive policy promotes uncertain or high-risk work while retaining the cheap
tier for confident straightforward requests.

## Limitations

- Labels represent synthetic, LLM-audited complexity rather than measured model outcomes.
- Prompts were generated to fit the label rubric and can contain stylistic cues.
- The model has not been evaluated on representative production traffic.
- It does not know current price, health, capability, tenancy, or compliance policy.
