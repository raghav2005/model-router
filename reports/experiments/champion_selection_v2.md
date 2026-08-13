# Complexity router v2 champion decision

**Decision date:** 13 August 2026

## Decision

Promote `complexity_router_v2.npz` as the default shadow-mode complexity
classifier. Retain v1 as a frozen baseline. Do not approve enforcement.

## Change evaluated

V2 classifies two views of a conversation: the complete flattened conversation
and the latest user turn. A validation-only grid search selected a 10% weight
for the latest-turn view and a temperature of 3.668. Selection minimized
negative log loss; the test and novel multi-turn slices were not used to choose
these values.

## Untouched comparison

| Measure | v1 | v2 | Change |
|---|---:|---:|---:|
| Internal exact accuracy | 90.93% | 91.14% | +0.21 pp |
| Internal macro F1 | 0.909 | 0.911 | +0.002 |
| Internal tier under-route | 3.64% | 3.56% | -0.08 pp |
| Internal negative log loss | 0.268 | 0.257 | -4.1% |
| Novel multi-turn exact accuracy | 57.14% | 57.14% | unchanged |
| Novel multi-turn negative log loss | 1.619 | 1.468 | -9.3% |

The v2 adaptive routing policy reduces tier under-routing on the 455-row novel
slice from 18.46% for hybrid argmax to 17.14%, with estimated mean cost rising
from $0.01105 to $0.01125 for the benchmark token allowance. The explicit 15%
tier-risk policy lowers it further to 11.65% at $0.01209, but is not the default
because the cost and over-routing penalty is materially higher.

## Why this is not a production approval

The external exact score remains 57.14%, external expected calibration error is
21.76%, and only 455 genuinely non-overlapping examples exist. All data came
from related synthetic generation processes. V2 is a better calibrated and
slightly more accurate shadow candidate, not proof of production quality.
