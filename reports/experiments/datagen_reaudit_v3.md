# Datagen re-audit and v3 training decision

**Decision date:** 20 August 2026

## Outcome

Promote `complexity_router_v3.npz` as the default shadow-mode classifier. Keep
v2 as a frozen baseline. Do not approve enforced production routing.

No source training prompts are included in this repository. The public
`dataset_audit` reports contain only aggregate counts, field names, and hashes.

## What the private-folder re-audit found

- The full valid dataset contains 86,967 unique normalized prompts and no
  conflicting duplicate labels.
- The equal-level dataset is an exact 66,560-row subset of the full dataset. It
  adds no new prompts and previously underperformed the full-data model.
- The four smaller multi-turn files contain 8,282 unique prompts in total. They
  are schema/metadata variants of one prompt family, not four independent data
  sources. Only 455 prompts are novel relative to the full dataset.
- Three prompts are absent from the backoff copy, and two normalized duplicates
  occur throughout the multi-turn family. No cross-file label conflicts exist.
- All 42,759 `borderline` rows include an adjacent judge-suggested level. This
  field was previously unused by the trainer.

## Controlled candidates

Every candidate used the same private source snapshot, normalized-prompt hash
split, model family, feature dimension, class prior, audit weights, and unchanged
validation/test/external prompts. Only train-split rows were augmented.

| Candidate | Validation accuracy | Validation NLL | Internal test accuracy | Internal tier under-route | External accuracy | External tier under-route |
|---|---:|---:|---:|---:|---:|---:|
| v2 baseline: 1 view × 0.50 | 91.10% | 0.256 | 91.57% | 3.13% | 57.80% | 18.02% |
| 2 views × 0.25 | 91.07% | 0.256 | 91.57% | 3.13% | 57.58% | 18.02% |
| **2 views × 0.50** | **91.45%** | **0.248** | **91.95%** | **2.79%** | **59.12%** | **17.36%** |
| 3 views × 0.25 | 91.33% | 0.252 | 91.77% | 2.95% | 58.02% | 17.80% |
| 15% audit-suggested soft label | 88.70% | 0.320 | 89.04% | 3.81% | 49.67% | 20.66% |
| 25% audit-suggested soft label | 86.61% | 0.368 | 87.32% | 4.18% | 45.05% | 21.54% |
| 35% audit-suggested soft label | 84.41% | 0.425 | 85.11% | 4.56% | 38.46% | 23.52% |

The two-view, half-weight configuration won on every declared validation metric:
accuracy, macro F1, tier under-routing, negative log likelihood, and calibration
error. The held-out and external diagnostics then confirmed the direction of the
change. The default adaptive router's internal under-routing fell from 2.23% to
2.03%; its estimated saving versus always-capable moved from 41.73% to 41.62%.

The soft-label experiment was rejected. The adjacent suggestions are useful
evidence of label ambiguity, but treating one LLM judge's suggestion as a second
target materially worsened every generalisation measure. The capability remains
an explicit experimental option with a zero default so the result is
reproducible; it is not active in v3.

## Robustness result

The router now normalizes all eight exact, versioned application envelopes before
semantic classification while retaining the raw token estimate for cost. On the
1,336-case metamorphic suite, default-policy tier and model invariance are now
100%, up from 88.92%. Default adversarial under-routing remains 2.99%.

## Research interpretation

This result is an incremental prompt-complexity improvement, not the final
routing objective. [RouteLLM](https://arxiv.org/abs/2406.18665) learns from model
preference outcomes rather than complexity labels. The accepted
[LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) design records each
model's output, ground truth, per-query score, tokens, and cost, and reports that
several sophisticated routers do not reliably beat a strong simple baseline.
Recent calibration research such as
[UCCI](https://arxiv.org/abs/2605.18796) likewise evaluates actual model outputs
and measured serving cost. These sources reinforce the existing production plan:
the next decisive dataset must contain per-prompt candidate-model outcomes and
measured latency/cost on representative workloads.

The custom harness remains vendor-neutral. OpenAI's current
[evaluation guidance](https://developers.openai.com/api/docs/guides/evals)
describes an iterative task/input/result workflow and now directs new users
toward Datasets as the legacy Evals platform approaches retirement.

## Evaluation caveat

The internal test and 455-row external slice were observed during iterative
research. They confirm this promotion but should no longer be treated as pristine
one-shot model-selection sets for future augmentation tuning. The next training
round must freeze a new customer- or time-separated holdout before candidate
experiments begin.
