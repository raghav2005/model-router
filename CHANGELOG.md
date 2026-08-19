# Changelog

## 0.7.0 — 2026-08-18

- Promoted Switchyard v0.2.0 as an approved, exact dependency and pinned its
  immutable source commit for the native server image and live evidence.
- Added a native Switchyard contract that verifies package version, TOML digest,
  target uniqueness, health, statistics, and all expected routes; fixed the
  duplicate Luna target found by the first native startup test.
- Added a pinned Switchyard container, Kubernetes Deployment/Service/NetworkPolicy,
  and CI jobs for the runtime contract and gateway container.
- Added deterministic train-only prompt-envelope augmentation and promoted the
  resulting artifact after it improved the untouched internal and external slices
  and reduced adversarial under-routing.
- Added exact normalization for five supported application envelopes plus a 1,336-
  case metamorphic regression gate. Default tier invariance improved from 65.87%
  to 88.92% and transformed-case under-routing fell to 5.16%.
- Expanded live evaluation with p99 latency and TTFT, token throughput, cache rate,
  finish-reason counts, Wilson 95% intervals, and workload-slice summaries.

## 0.6.0 — 2026-08-17

- Corrected Luna and Terra token prices from the official model pages and added
  an automated verifier for model IDs, all token rates, cache-write pricing,
  context/output limits, and long-context multipliers.
- Bound pricing evidence to the exact catalogue digest and training evidence to
  the exact router artifact, catalogue schema/digest, external data hash, and
  routing-policy version.
- Added validation-only tuning across 28 adaptive confidence/risk settings. The
  cheaper candidate was retained only as research evidence because it failed
  external confirmation; the runtime default was not changed.
- Reduced adaptive-policy tuning work by reusing route decisions across the
  search grid.
- Made live benchmark resume fail closed when the cases, catalogue, targets,
  repetitions, streaming mode, retention setting, or Switchyard revision changes.
- Required live release evidence to match an explicitly approved case-set hash,
  the qualified Switchyard revision, call-success and validator thresholds, and
  the expected upstream response-model identities.
- Expanded the credential-free suite to 74 tests and regenerated the public-safe
  artifact, adversarial report, and aggregate training reports without publishing
  source training data.

## 0.5.0 — 2026-08-13

- Added full-conversation classification, calibrated classifier confidence,
  normalized entropy, posterior tier-underroute risk, and adaptive routing.
- Promoted a validation-selected multi-view classifier with better internal
  accuracy, macro F1, calibration loss, and external negative log likelihood.
- Added a deterministic 167-case adversarial regression suite covering lexical
  traps, concise hard tasks, multilingual requests, prompt injection, high-risk
  requests, and multi-turn intent shifts.
- Fixed keyword-driven over-routing for tightly bounded meta-tasks and guarded
  short-answer facts without weakening hard or high-stakes regression slices.
- Made enforcement fail closed at service startup unless every production
  release gate passes, and exposed the release state from readiness.
- Added a privacy-safe offline drift monitor for route, workload, complexity,
  uncertainty, tier-risk, and cost distributions.
- Expanded the credential-free test suite to 68 tests.

## 0.4.2 — 2026-08-11

- Pinned Ruff and declared the CI lint rules explicitly so local and hosted
  verification use the same toolchain and policy.

## 0.4.1 — 2026-08-11

- Made live-evaluation summaries deterministic across checkpoint and resume by
  canonicalising timing and cost precision before aggregation.

## 0.4.0 — 2026-08-11

- Replaced anonymous catalogue assumptions with sourced GPT-5.6 model IDs, prices, cache pricing, long-context multipliers, context/output limits, and evidence markers.
- Added the current Switchyard Rust-server TOML deployment candidate while retaining the legacy 0.1 configuration for reproducibility.
- Added request/provider/model policy constraints and fail-closed handling for unmeasured latency SLAs.
- Added bounded retries, `Retry-After` support, circuit breaking, request IDs, explicit fallback, and response validation hooks.
- Added privacy-safe decision/execution audit events and dependency-free Prometheus metrics.
- Added an authenticated shadow/enforce HTTP API with health, readiness, route, chat, and metrics endpoints.
- Added executable release gates that intentionally block enforcement until live quality, latency, generalisation, Switchyard pinning, and gateway-bypass requirements pass.
- Expanded live evaluation with repetitions, streaming TTFT, completion latency, cache/token usage, cost estimates, finish reason, and response-model verification.
- Added container, Gunicorn, Kubernetes, CI, security, deployment, rollback, and production-readiness assets.
- Expanded the credential-free test suite to 47 tests.

## 0.3.0 — 2026-08-11

- Added the trained five-level complexity classifier and packaged artifact.
- Added hybrid and conservative routing policies.
- Added deterministic dataset splits, audit-aware weighting, calibration, challenger comparison, offline policy evaluation, and the initial response-level Switchyard harness.
- Added the first manager-facing training and evaluation reports.
