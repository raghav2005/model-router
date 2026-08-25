# Model router: production-readiness update

**Date:** 25 August 2026

**Status:** Deployable for integration and shadow evaluation; blocked from enforced production routing

## What has been delivered

The prototype is now an operationally structured service rather than only a classifier and command-line demonstration. It includes the trained multi-view complexity model, explainable adaptive routing policy, current Switchyard configuration, automated official-source pricing verification, an authenticated HTTP interface, privacy-safe audit events and drift detection, metrics, resilience controls, immutable live-benchmark provenance, fail-closed release enforcement, container/Kubernetes assets, CI, and a deployment/rollback runbook.

The first model set is OpenAI GPT-5.6 Luna, Terra, and Sol in efficient, balanced, and capable roles. Official prices, context limits, cached-input rates, cache-write rates, and long-context multipliers are versioned in the catalogue. Published benchmark results are documented as directional research; they are not misrepresented as application quality.

## Evidence available now

- 86,967 audited training examples.
- 91.95% exact accuracy and 2.79% tier under-routing on the internal 8,522-row test set.
- 59.12% exact accuracy and 17.36% tier under-routing on the 455 genuinely non-overlapping multi-turn examples.
- The new adaptive task-switch guard reduces complete-policy tier under-routing
  from 2.03% to 1.26% internally and from 14.95% to 10.99% on the multi-turn
  diagnostic. It does not change the classifier-label metrics above.
- Local classifier-only latency over 1,000 prompts: 116 µs median, 166 µs p95,
  and 191 µs p99. This excludes Switchyard and model-generation latency.
- 83 passing credential-free tests, plus a native Switchyard runtime/configuration contract in CI.
- A live harness capable of measuring validator outcomes, p50/p95/p99 TTFT and completion latency, token throughput, Wilson confidence intervals, category/use-case/risk/complexity slices, cached tokens, estimated cost, errors, finish reason, and actual response model across repeated trials.
- A frozen 120-case machine-checkable synthetic live baseline, sufficient to run
  the harness at release-scale sample counts but intentionally not approved as a
  substitute for representative workload evidence.
- A 179-case adversarial regression suite that covers keyword traps and 18
  multi-turn task switches; it remains separate from release evidence.
- A 1,432-case metamorphic suite. Exact normalization of all eight versioned
  application envelopes raises tier and model invariance to 100%, with 1.68%
  under-routing.
- A reproducible prompt-free datagen audit proving that the four multi-turn files
  are one metadata family, plus a controlled training grid that selected two
  half-weight train-only augmentations and rejected audit-suggested soft labels.
- A startup interlock that prevents enforcement from bypassing the release gates.
- Cryptographic binding between release reports, the exact catalogue, trained artifact, routing policy, and live case set.
- A validation-only 28-point confidence/risk policy search. Its current candidate
  improves the diagnostic policy under-routing to 9.45%, but is still rejected
  because only 455 related synthetic examples exist and the 5% release threshold
  is not met.
- A corrected, official-source-verified Sol price of $4 input and $20 output per
  million tokens; cost comparisons and release provenance were regenerated.
- A prompt-free drift detector for workload, route, complexity, uncertainty, tier-risk, and cost changes.

## Why production enforcement remains blocked

The existing dataset measures synthetic request complexity. It does not show whether each candidate model produces an acceptable response on our intended workloads. The external multi-turn score is also below a reasonable release boundary.

The system currently passes six of thirteen release gates: current catalogue dates,
official-source pricing verification, metamorphic robustness, exact
artifact/catalogue/policy provenance, the approved Switchyard pin, and the native
Switchyard runtime/configuration contract. It fails closed on the remaining seven:

1. production enforcement has not been approved;
2. workload-specific model quality is unmeasured;
3. workload-specific model latency is unmeasured;
4. the independent generalisation dataset is too small and below threshold;
5. the paid live response benchmark has not been run;
6. no live case-set digest has been approved and bound to benchmark evidence;
7. a direct-provider gateway bypass has not been configured.

## Inputs required from the organisation

1. Approved provider/model deployments, regions, retention terms, credentials, quotas, and target SLOs.
2. A privacy-reviewed frozen sample of real intended workloads with task-specific quality criteria and enough multi-turn coverage.
3. Budget and approval to run every candidate model on the frozen benchmark, plus owners for human review, security approval, and production operations.

Once those inputs exist, the included harness and release policy provide a direct path through live evaluation, recalibration, shadow traffic, guarded enforcement, and rollback testing.
