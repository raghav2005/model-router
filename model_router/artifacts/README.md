# Complexity router artifacts

`complexity_router_v3.npz` is the current default. It is a five-class weighted
multinomial Naive Bayes model trained from the audited
`routing_dataset_100k_valid_only.jsonl` snapshot.

Artifact SHA-256: `39b8a46f2b88891587ec114e7f58c7cb4a42e8cfcb8c597404a9ea90c4f0191b`

Training recipe SHA-256: `b81af54e1cfb0bcce94619217fb36d6dcd1fef50e5a7624d134c6a47cd5bcb77`

Training implementation SHA-256: `644ccd43e4ffcfd024f7c92ce9c4de06b4b5431767254526776c1e0e05b00c63`

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
adaptive policy promotes uncertain, high-risk, or cross-view-disagreement work
while retaining the cheap tier for confident straightforward requests.

## Limitations

- Labels represent synthetic, LLM-audited complexity rather than measured model outcomes.
- Prompts were generated to fit the label rubric and can contain stylistic cues.
- The model has not been evaluated on representative production traffic.
- It does not know current price, health, capability, tenancy, or compliance policy.
