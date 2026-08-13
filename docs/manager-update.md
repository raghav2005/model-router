# Model router: production-readiness update

**Date:** 13 August 2026

**Status:** Deployable for integration and shadow evaluation; blocked from enforced production routing

## What has been delivered

The prototype is now an operationally structured service rather than only a classifier and command-line demonstration. It includes the trained multi-view complexity model, explainable adaptive routing policy, current Switchyard configuration, verified pricing semantics, an authenticated HTTP interface, privacy-safe audit events and drift detection, metrics, resilience controls, expanded live benchmarking, fail-closed release enforcement, container/Kubernetes assets, CI, and a deployment/rollback runbook.

The first model set is OpenAI GPT-5.6 Luna, Terra, and Sol in efficient, balanced, and capable roles. Official prices, context limits, cached-input rates, cache-write rates, and long-context multipliers are versioned in the catalogue. Published benchmark results are documented as directional research; they are not misrepresented as application quality.

## Evidence available now

- 86,967 audited training examples.
- 91.14% exact accuracy and 3.56% tier under-routing on the internal 8,522-row test set.
- 57.14% exact accuracy and 18.68% tier under-routing on the 455 genuinely non-overlapping multi-turn examples.
- 68 passing credential-free tests.
- A live harness capable of measuring validator outcomes, time to first token, completion latency, tokens, cached tokens, estimated cost, errors, finish reason, and actual response model across repeated trials.
- A 167-case adversarial regression suite that found and fixed a major keyword-driven over-routing pattern; it remains separate from release evidence.
- A startup interlock that prevents enforcement from bypassing the release gates.
- A prompt-free drift detector for workload, route, complexity, uncertainty, tier-risk, and cost changes.

## Why production enforcement remains blocked

The existing dataset measures synthetic request complexity. It does not show whether each candidate model produces an acceptable response on our intended workloads. The external multi-turn score is also below a reasonable release boundary.

The system therefore fails closed on seven current release gates:

1. production enforcement has not been approved;
2. workload-specific model quality is unmeasured;
3. workload-specific model latency is unmeasured;
4. the independent generalisation dataset is too small and below threshold;
5. the paid live response benchmark has not been run;
6. a precise Switchyard build has not been pinned and qualified; and
7. a direct-provider gateway bypass has not been configured.

Pricing provenance is the one current gate that passes.

## Inputs required from the organisation

1. Approved provider/model deployments, regions, retention terms, credentials, quotas, and target SLOs.
2. A privacy-reviewed frozen sample of real intended workloads with task-specific quality criteria and enough multi-turn coverage.
3. Budget and approval to run every candidate model on the frozen benchmark, plus owners for human review, security approval, and production operations.

Once those inputs exist, the included harness and release policy provide a direct path through live evaluation, recalibration, shadow traffic, guarded enforcement, and rollback testing.
