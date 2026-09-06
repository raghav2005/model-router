# Production-readiness design

## Current position

The repository is deployable for integration and shadow-mode evaluation. It is intentionally blocked from enforced production routing.

Run:

```bash
model-router release-gates
```

The command exits successfully only when every release condition passes. The current report is expected to fail because live model outcomes and production-distribution evidence have not been supplied.

## Runtime architecture

```mermaid
flowchart LR
    C[Client] --> I[Authenticated ingress]
    I --> R[Model-router API]
    R --> A[Learned complexity model]
    R --> P[Hard policy gates]
    R --> U[Quality/cost/latency scorer]
    U --> S[Switchyard service]
    S --> M1[Efficient model]
    S --> M2[Balanced model]
    S --> M3[Capable model]
    R --> O[Metrics and prompt-free audit]
    S --> O
    S -. trusted bypass .-> M3
```

The custom service owns decisions and business controls. Switchyard owns the
provider-facing protocol boundary. Switchyard v0.2.0 is an approved dependency,
pinned to immutable commit `1fc9ab887d1c663b0048ae24d5f473d15ed8daaa`.
The adapter boundary still prevents a gateway failure or upgrade from requiring a
rewrite of the learned router.

## Implemented safeguards

### Safe rollout modes

- `shadow`: records the proposed route but executes a fixed capable baseline.
- `enforce`: executes the selected target and refuses to start unless every release gate passes.

The container and Kubernetes examples default to `shadow`.

### Hard eligibility controls

Models can be rejected for tier, capability, context, maximum output, health, request budget, provider allowlist, model allowlist, latency SLA, or predicted-quality floor. Explicit latency SLAs are rejected while latency evidence is marked unmeasured.

### Evidence-aware catalogue

The catalogue records provider and upstream model, verified price source and date,
cache pricing, long-context multipliers, context/output limits, and evidence status
for quality and latency. An automated verifier compares every price and limit with
the official provider pages. Enforcement requires a current verification report
whose catalogue digest matches the exact deployed catalogue. Published vendor
benchmarks are stored separately from workload-measured scores.

The training report similarly records hashes for the exact learned artifact,
catalogue, external dataset, and routing-policy version. A stale report cannot
approve a different artifact or policy. Live benchmark summaries carry a fingerprint
of their cases, catalogue, targets, repetitions, and execution settings; resume and
release approval both fail on a mismatch.

The pinned Switchyard package is started in CI against the exact TOML. Its health,
statistics payload, route table, version, and configuration digest are recorded in
`reports/switchyard_contract.json`. Duplicate upstream target definitions fail the
contract check.

### Dependency resilience

- connect/read timeout at the HTTP boundary;
- bounded exponential backoff with jitter for safe/idempotent requests;
- `Retry-After` handling;
- circuit breaker with a recovery probe;
- no automatic generation retry by default, avoiding silent duplicate billing;
- optional cross-target fallback with an explicit audit trail; and
- Switchyard liveness/readiness checks.

### Data minimisation

The normal audit trail stores request ID, tenant ID, prompt HMAC, policy/catalog/model versions, classification, decision, estimated cost, tokens, latency, and outcome. It never stores prompt text. Response content is also omitted. A secret HMAC key prevents practical dictionary matching of prompt fingerprints.

### Observability

The API exports bounded-cardinality Prometheus metrics for:

- route counts by model, use case, and classifier;
- estimated spend by selected role;
- execution success/error counts;
- provider-reported input/output tokens; and
- completion-latency histograms;
- classifier confidence and normalized entropy; and
- posterior tier-underroute probability.

An offline drift command builds an approved baseline from prompt-free shadow audit
events and compares later windows using Jensen-Shannon distance and standardized
mean shifts. It fails for insufficient sample size rather than reporting a healthy
window without evidence.

Switchyard's own `/v1/stats`, selected-model headers, and provider telemetry should be collected alongside these metrics.

## Release gates

The default release policy requires:

1. an explicit change from shadow to enforce mode;
2. current, source-linked pricing;
3. a current official-source verification report matching the exact catalogue;
4. workload-measured response quality for every role;
5. workload-measured latency for every role;
6. at least 1,000 independent external examples, at least 75% exact complexity accuracy, and no more than 5% tier under-routing;
7. a passing 1,000+ case metamorphic regression with bounded under-routing and
   tier-invariance thresholds;
8. a training report matching the exact router artifact, catalogue, and policy version;
9. at least 100 distinct scored live cases per role, at least 20 distinct cases for
   each supported use case, at least an 85% validator pass rate within each use-case
   slice, a measured p95 completion latency, at least a 99% call-success rate, at
   least a 90% all-validator pass rate, required Wilson-interval lower bounds, and
   no unexpected response-model substitutions;
10. live evidence matching the exact catalogue, approved Switchyard revision, and
   an explicitly approved case-set digest;
11. an approved, pinned Switchyard release or commit;
12. a passing Switchyard native runtime/configuration contract; and
13. a trusted direct-provider bypass.

Quality and latency approval is represented by a separate prompt-free workload-
evidence artifact, not by editing the base model catalogue. The release policy pins
the artifact's exact SHA-256, and enforcement rechecks that it matches the unchanged
live summary and catalogue before loading its measured per-use-case quality and p95
latency values. This removes a prior circularity where marking the catalogue as
measured invalidated the benchmark's catalogue digest.

The external-generalisation gate similarly requires a prompt-free provenance report
whose exact SHA-256 is approved in policy. The report's dataset digest and novel-row
count must match the training report, and the reviewer must classify the dataset as
independent and set `approved_for_release` to true. The current multi-turn data is
explicitly marked `diagnostic_only` and cannot pass even if its numerical thresholds
are relaxed.

Thresholds are initial release criteria and must be approved against business risk. Open-ended work also needs blinded human review or an approved judge model; deterministic validators alone are insufficient.

## Remaining production work

### Evidence and calibration

- Collect a privacy-reviewed sample of real intended traffic.
- Create frozen train, calibration, test, and out-of-distribution sets by customer or time boundary—not random row split alone.
- Run all three models on every eligible response-level case. Repetitions measure
  nondeterminism but never count as additional workload coverage.
- Measure quality, TTFT, completion latency, output rate, tokens, price, errors, and refusal rate by slice.
- Generate and approve the aggregate workload-evidence artifact only after the
  underlying report is reviewed and versioned; keep the base catalogue's bootstrap
  markers as an honest record of its standalone state.
- Continue retraining or recalibration because the promoted augmentation model
  improves the novel multi-turn slice but remains below the release threshold.

### Platform and security

- Put OAuth/JWT or service identity, tenant quotas, and rate limiting at the ingress layer.
- Use a managed secret store and automated rotation.
- Confirm provider retention, training, residency, and zero-data-retention terms.
- Add prompt-injection and content-safety controls appropriate to the application.
- Send audit records to immutable central storage with retention and access controls.
- Perform threat modelling, dependency scanning, SBOM generation, image signing, and penetration testing.

### Reliability

- Fault-test the pinned Switchyard image under the expected load and failure modes.
- Implement and exercise a direct-provider bypass independently of Switchyard.
- Load-test the entire path at expected peak concurrency.
- Define SLOs and alerts for success, p95/p99 latency, under-routing, fallback, cost, token use, and distribution drift.
- Approve a shadow-traffic drift baseline, automate scheduled comparisons, and connect failed reports to alerting and rollback.
- Add multi-region or multi-provider failover if required by the availability target.
- Test graceful shutdown, rolling upgrades, rollback, quota exhaustion, 429s, slow responses, invalid JSON, and partial streams.

### Governance

- Assign owners for the routing policy, catalogue, training data, model artifact, Switchyard build, and release approval.
- Version every decision input and retain reproducible reports.
- Establish model/provider deprecation procedures and emergency kill switches.
- Review high-risk use cases separately; the generic complexity model is not a domain safety system.

## Acceptance statement

Passing unit tests or the synthetic complexity benchmark is not production approval. Enforced routing is approved only when the executable release-gate report, live response evidence, security review, load test, operational runbook, and rollback exercise all pass for the exact immutable artifact and deployment configuration.
