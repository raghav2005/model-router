# Multi-view disagreement safety experiment

**Decision date:** 25 August 2026

## Decision

Adopt full-conversation/final-turn tier disagreement as an additional trigger for
the existing posterior-risk escalation in the default adaptive policy. Keep the
router in shadow mode.

## Rationale

The v3 classifier already evaluates the complete conversation and the final user
turn, but the final-turn view has only 5% ensemble weight. A new difficult request
can therefore be obscured by easier vocabulary earlier in a conversation. The new
guard does not train on, relabel, or publish the external prompts. When the two
views' argmax levels collapse to different model tiers, the adaptive policy treats
that disagreement as uncertainty and selects the cheapest tier whose posterior
under-routing probability is within the configured tolerance.

## Untouched and diagnostic results

| Evaluation | Previous adaptive under-route | New adaptive under-route | Change |
|---|---:|---:|---:|
| Internal hash-held-out test (8,522) | 2.03% | 1.26% | -0.77 pp |
| Novel multi-turn diagnostic (455) | 14.95% | 10.99% | -3.96 pp |

On the internal test, over-routing rises from 9.88% to 10.76%. With the corrected
catalogue, the new adaptive policy costs an estimated $0.00685 per fixed benchmark
request versus $0.00674 for hybrid argmax, a 1.63% premium. It still costs 34.45%
less than the always-capable reference under this offline token allowance.

On the original frozen 167-case adversarial suite, adaptive under-routing falls
from 2.99% to 1.80%, over-routing rises from 21.56% to 22.16%, and exact tier
accuracy rises from 75.45% to 76.05%. The expanded suite contains 179 cases,
including 18 multi-turn task switches; adaptive under-routing is 1.68%. Its 1,432
meaning-preserving variants retain 100% tier and model invariance.

## Interpretation

This is a safety/efficiency trade-off, not an independent quality result. The
internal test is untouched, but the small multi-turn slice and deterministic
regression suite have both been observed during development. They cannot be used
as fresh production evidence. The structured disagreement flag is therefore
exported to route responses, prompt-free audit events, and metrics so its real
frequency and downstream outcomes can be measured during shadow operation.

The production release gate remains unchanged: representative time- or
customer-separated requests, response-level success measurements for every model,
and live latency/cost evidence are still required.
