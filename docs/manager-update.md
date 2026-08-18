# Model router: production-readiness update

**Date:** 18 August 2026

**Status:** Deployable for integration and shadow evaluation; blocked from enforced production routing

## What has been delivered

The prototype is now an operationally structured service rather than only a classifier and command-line demonstration. It includes the trained multi-view complexity model, explainable adaptive routing policy, current Switchyard configuration, automated official-source pricing verification, an authenticated HTTP interface, privacy-safe audit events and drift detection, metrics, resilience controls, immutable live-benchmark provenance, fail-closed release enforcement, container/Kubernetes assets, CI, and a deployment/rollback runbook.

The first model set is OpenAI GPT-5.6 Luna, Terra, and Sol in efficient, balanced, and capable roles. Official prices, context limits, cached-input rates, cache-write rates, and long-context multipliers are versioned in the catalogue. Published benchmark results are documented as directional research; they are not misrepresented as application quality.

## Evidence available now

- 86,967 audited training examples.
- 91.57% exact accuracy and 3.13% tier under-routing on the internal 8,522-row test set.
- 57.80% exact accuracy and 18.02% tier under-routing on the 455 genuinely non-overlapping multi-turn examples.
- Local classifier-only latency over 1,000 prompts: 169 µs median, 241 µs p95,
  and 274 µs p99. This excludes Switchyard and model-generation latency.
- 79 passing credential-free tests, plus a native Switchyard runtime/configuration contract in CI.
- A live harness capable of measuring validator outcomes, p50/p95/p99 TTFT and completion latency, token throughput, Wilson confidence intervals, category/use-case/risk/complexity slices, cached tokens, estimated cost, errors, finish reason, and actual response model across repeated trials.
- A frozen 120-case machine-checkable synthetic live baseline, sufficient to run
  the harness at release-scale sample counts but intentionally not approved as a
  substitute for representative workload evidence.
- A 167-case adversarial regression suite that found and fixed a major keyword-driven over-routing pattern; it remains separate from release evidence.
- A 1,336-case metamorphic suite. Default-policy tier invariance improved from
  65.87% to 88.92%, with 5.16% under-routing after train-only augmentation and
  exact supported-envelope normalization.
- A startup interlock that prevents enforcement from bypassing the release gates.
- Cryptographic binding between release reports, the exact catalogue, trained artifact, routing policy, and live case set.
- A validation-only 28-point confidence/risk policy search. Its cheaper candidate was correctly rejected because external multi-turn under-routing worsened from 17.14% to 18.46%.
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
