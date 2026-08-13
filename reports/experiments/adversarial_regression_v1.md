# Adversarial routing regression v1

Date: 2026-08-13

## Purpose

This deterministic suite tests routing failure modes that ordinary random splits
do not cover well: misleading keywords in trivial tasks, concise difficult tasks,
short non-English questions, changing multi-turn intent, high-stakes requests,
and attempts to influence the router through prompt text.

The 167 generated cases are design and regression tests. They are not independent
production traffic, are not used to train the learned classifier, and must not be
included in the independent-data release gate.

## Failure found

The first run showed that simple operations over complex-looking phrases were
over-routed. For example, a request to spell the phrase `security audit` inherited
the complexity and risk associated with the quoted words. All eight short
multilingual factual questions were also over-routed.

## Change

The classifier now recognizes a deliberately narrow set of bounded meta-tasks and
short-answer factual requests. The short-answer rule is disabled when the prompt
contains code, a high-stakes domain, an advanced action, or a code block. Therefore,
`prove the theorem; answer in one word` remains a reasoning task. Learned posterior
values are still emitted in telemetry even when the bounded-task guard applies.

## Results

| Policy | Before accuracy | After accuracy | Before under-route | After under-route | Before over-route | After over-route | Before avg cost | After avg cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid argmax | 52.69% | 82.63% | 8.38% | 8.38% | 38.92% | 8.98% | $0.011054 | $0.008841 |
| Hybrid adaptive (p=0.15) | 47.31% | 77.84% | 7.19% | 7.19% | 45.51% | 14.97% | $0.012854 | $0.009349 |
| Hybrid tier-risk (p=0.15) | 40.72% | 74.25% | 1.80% | 1.80% | 57.49% | 23.95% | $0.014266 | $0.010310 |

The lexical-trap slice improved from 5/48 exact to 48/48 exact for the default
adaptive policy. The multilingual-simple slice improved from 0/8 to 8/8 exact.
All 48 concise-hard and all six high-stakes cases remain on the capable tier.

## Interpretation and next action

The change removes a large and expensive false-positive pattern without weakening
the difficult or high-stakes slices. Medium-complexity composition and multi-turn
shifts remain the main synthetic failure modes. Those slices should guide additional
real-data collection, but should not be optimized repeatedly in isolation because
that would overfit the deterministic benchmark.
