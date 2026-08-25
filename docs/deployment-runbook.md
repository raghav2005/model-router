# Deployment and rollback runbook

## 1. Build and verify

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
model-router verify-pricing
python -m pip install -e '.[switchyard]'
OPENAI_API_KEY=contract-validation-only \
  python -m model_router.switchyard_contract \
  --output reports/switchyard_contract.json
model-router release-gates
python -m build
docker build -t model-router:<immutable-version> .
docker build -f Dockerfile.switchyard \
  -t model-router-switchyard:<immutable-version> .
```

Record the source commit, image digest, model-artifact SHA-256, catalogue SHA-256, Switchyard commit/version, release-gate report, and live-evaluation report in the release ticket.

## 2. Prepare Switchyard

1. Build `Dockerfile.switchyard`, which fetches and verifies approved commit
   `1fc9ab887d1c663b0048ae24d5f473d15ed8daaa`, or install the equivalent pinned
   `switchyard-server` binary.
2. Export `OPENAI_API_KEY` through the secret manager.
3. Validate `config/switchyard_routes.toml` using both
   `python -m model_router.switchyard_contract` and `switchyard-server --dry-run`.
4. Start Switchyard on the private service network.
5. Confirm `/health`, `/v1/models`, and `/v1/stats`.
6. Verify that `efficient`, `balanced`, and `capable` resolve to the exact catalogue model IDs.

Do not expose Switchyard directly to untrusted clients. The router is the policy boundary.

## 3. Deploy in shadow mode

Required configuration:

```text
MODEL_ROUTER_MODE=shadow
MODEL_ROUTER_SHADOW_TARGET=capable
MODEL_ROUTER_API_TOKEN=<secret>
MODEL_ROUTER_METRICS_TOKEN=<secret>
MODEL_ROUTER_AUDIT_HMAC_KEY=<secret>
MODEL_ROUTER_AUDIT_PATH=-
MODEL_ROUTER_CLASSIFIER_MODE=hybrid
MODEL_ROUTER_COMPLEXITY_POLICY=adaptive
MODEL_ROUTER_UNDERROUTE_TOLERANCE=0.15
MODEL_ROUTER_ADAPTIVE_CONFIDENCE_THRESHOLD=0.45
MODEL_ROUTER_ARTIFACT=model_router/artifacts/complexity_router_v3.npz
MODEL_ROUTER_PRICING_REPORT=reports/pricing_verification.json
SWITCHYARD_URL=http://switchyard:4000
SWITCHYARD_REVISION=1fc9ab887d1c663b0048ae24d5f473d15ed8daaa
```

In shadow mode, `X-Model-Router-Proposed-Role` reports the proposed target while the capable baseline is executed. Compare proposed decisions with observed outcomes without changing customer behaviour.

## 4. Shadow acceptance

Monitor for at least one complete business cycle. Review by use case, tenant, complexity, risk, prompt length, conversation length, and required capability.

Required evidence includes:

- quality delta against capable baseline;
- under-route and fallback rate;
- p50/p95/p99 TTFT and completion latency;
- actual tokens and invoiced cost;
- provider errors, timeouts, and refusals;
- classifier confidence and distribution drift; and
- any security or data-policy violations.

Freeze the live case set before the paid benchmark, run
`model-router case-set-hash <cases.jsonl>`, and record the generated
`case_set_sha256` as `approved_live_case_set_sha256` in the reviewed release policy.
Do not reuse results after cases, targets, catalogue, repetition count, or streaming
settings change; the harness enforces this by refusing an incompatible resume.

Update the catalogue evidence markers and release report only after approval of the underlying data.

Create the drift baseline only after the shadow window and its traffic mix are
approved. Store it with the release evidence:

```bash
model-router drift-baseline audit-shadow.jsonl --output config/drift_baseline.json
model-router drift-report audit-next-window.jsonl --baseline config/drift_baseline.json
```

Do not overwrite an approved baseline silently. A new baseline is a reviewed policy
change, not a way to clear an alert.

## 5. Guarded enforcement

1. Confirm `model-router release-gates` passes for the release inputs. The service
   independently repeats this check and refuses to start in `enforce` mode if it fails.
2. Route a small, explicitly allowlisted traffic percentage through `enforce` mode.
3. Keep high-risk and unvalidated workloads pinned to capable.
4. Compare against a concurrent control group.
5. Increase traffic only after the agreed observation window passes.

Generation retries and cross-target fallback are disabled by default because a failed response may still have incurred provider cost. Enable them only after idempotency and duplicate-charge behaviour are understood.

## 6. Rollback

Immediate rollback is changing `MODEL_ROUTER_MODE` to `shadow`, which returns execution to the capable baseline while preserving proposed-decision telemetry.

If Switchyard is unhealthy, move traffic to the separately tested direct-provider bypass. Do not wait for an unproven gateway recovery path.

Rollback triggers should include:

- quality or under-routing outside the release boundary;
- p95/p99 latency or error-budget breach;
- unexplained cost increase;
- catalogue/configuration mismatch;
- provider model alias change;
- audit or metrics loss;
- security incident; or
- material classifier drift.

After rollback, preserve logs and immutable deployment metadata, open an incident review, and require the release gates to pass again before re-enforcement.
