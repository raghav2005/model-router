# Complexity router artifacts

`complexity_router_v2.npz` is the current default. It is a five-class weighted
multinomial Naive Bayes model trained from the audited
`routing_dataset_100k_valid_only.jsonl` snapshot.

Artifact SHA-256: `b893eaa86e744e86eb3efa195a9b8fa56cc702286bc1cc78bc294c773375f650`

`complexity_router_v1.npz` is retained as the frozen baseline. Git history and
`reports/complexity_router_v1.*` preserve its original evidence.

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
- Multi-view inference: 90% complete conversation and 10% latest user turn
- Probability calibration: ensemble weight and temperature selected by validation log loss
- Adaptive policy search: 28 confidence/risk settings evaluated on validation only
- The validation-selected candidate was rejected for deployment because external confirmation failed

## Champion evidence

| Measure | v1 | v2 |
|---|---:|---:|
| Internal test accuracy | 90.93% | 91.14% |
| Internal test macro F1 | 0.909 | 0.911 |
| Internal tier under-route | 3.64% | 3.56% |
| Internal negative log loss | 0.268 | 0.257 |
| Novel multi-turn accuracy | 57.14% | 57.14% |
| Novel multi-turn negative log loss | 1.619 | 1.468 |

V2 was promoted because it improves untouched internal accuracy, tier safety,
and probabilistic log loss without reducing novel multi-turn accuracy. The
external calibration error is still poor and the novel slice remains too small;
the artifact therefore remains shadow-only.

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
