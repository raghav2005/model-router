# Model research and catalogue rationale

**Research date:** 25 August 2026

**Scope:** Initial OpenAI deployment candidate behind NVIDIA NeMo Switchyard

## Decision

The first controlled deployment uses one model family across three stable routing roles:

| Router role | Upstream model | Intended workload | Standard input / output per 1M tokens |
|---|---|---|---:|
| `efficient` | `gpt-5.6-luna` | High-volume and straightforward work | $0.20 / $1.20 |
| `balanced` | `gpt-5.6-terra` | Everyday professional work | $2 / $12 |
| `capable` | `gpt-5.6-sol` | Complex reasoning, coding, and high-risk work | $4 / $20 |

Using one provider family simplifies the first evaluation: formats, tool behaviour, context limits, and billing semantics are comparable. A second provider should be added later for resilience, but only after its models have been run through the same response-level benchmark.

The prices, model IDs, context windows, and feature claims above come from the official [OpenAI model catalogue](https://developers.openai.com/api/docs/models) and individual model pages. The production catalogue records an exact source URL and verification date for every role. The automated verifier re-downloads those pages and compares prices, context/output limits, cache-write pricing, and long-context multipliers before a release; the verification report is bound to the exact catalogue SHA-256.

OpenAI's current [model-selection guidance](https://developers.openai.com/tracks/building-agents#how-to-choose)
recommends starting with the flagship model, moving simple or latency-sensitive
work to smaller models, and using a faster conversational model that delegates
demanding tasks to the flagship. It also recommends varying prompts during
evaluation. This supports the efficient/balanced/capable role design and the
adaptive escalation policy, but it does not establish our routing thresholds;
those still require workload-specific evaluation.

## Pricing details implemented

All three models have a 1,050,000-token context window and a 128,000-token maximum output. The router also accounts for:

- the 90% cached-input discount;
- cache writes billed at 1.25 times ordinary input;
- prompts over 272,000 input tokens billed at twice the input rate; and
- output on those long-context requests billed at 1.5 times the ordinary output rate.

These details come from the official model pages for [Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol), [Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra), and [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna). Tool-specific charges are not yet estimated because they depend on which hosted tools the application enables. Actual provider usage and invoices remain the source of truth.

## Published quality evidence

The [GPT-5.6 launch report](https://openai.com/index/gpt-5-6/) provides useful directional evidence:

| Benchmark | Sol | Terra | Luna |
|---|---:|---:|---:|
| Agents' Last Exam | 52.7% | 50.4% | 50.3% |
| GPQA Diamond | 94.6% | 92.9% | 92.3% |
| FrontierMath Tier 1–3 | 89.0% | 84.9% | 78.6% |
| Toolathlon | 58.0% | 53.1% | 53.4% |

These are vendor-published benchmark results, not probabilities that a model will satisfy our requests. They should not be inserted directly into the utility formula. The catalogue's `skills` values remain explicitly marked as heuristic priors until our response-level harness produces workload-specific success rates.

The production quality table should ultimately be estimated per model, use case, complexity band, tool configuration, prompt version, and reasoning setting. Confidence intervals and sample counts should be stored alongside every estimate.

## Latency evidence

OpenAI describes Luna as the fastest tier and Terra as a balance of capability, speed, and cost, but does not publish a p95 end-to-end completion latency that is valid for our prompts, region, account tier, concurrency, output length, or reasoning configuration.

The existing latency numbers are therefore retained only as bootstrap priors and marked `unmeasured_bootstrap_prior`. The router refuses to claim that it can meet an explicit latency SLA while this marker is present.

Before enforcement, the live harness must measure at least:

- time to first token and total completion latency;
- p50, p95, and p99 by target and workload slice;
- output tokens per second;
- cold and warm behaviour;
- concurrent-load behaviour at expected and peak traffic;
- 429, timeout, and 5xx rates; and
- the effect of reasoning effort and long context.

Measurements must be repeated in the intended deployment region and provider service tier. Published relative-speed descriptions are not sufficient for an SLA.

## Evaluation-harness approach

The local, versioned harness remains the release system of record. This follows
OpenAI's [evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices):
define task-specific success criteria, use production-like inputs, combine automated
checks with calibrated human judgement, and evaluate continuously. The harness uses
deterministic validators where an objectively correct outcome exists and reserves
open-ended work for an approved [grader](https://developers.openai.com/api/docs/guides/graders),
executable task outcome, or blinded human rubric.

The initial 120-case synthetic live baseline is large enough to exercise repeated
quality, latency, throughput, and cost measurement, but it is deliberately not the
approved production case set. Representative coding, reasoning, tool-use, refusal,
and long-context cases still need to come from the intended workload. The local
harness also avoids coupling release evidence to the hosted OpenAI Evals platform,
which the official documentation says will become read-only on 31 October 2026 and
shut down on 30 November 2026.

Latency work follows OpenAI's
[latency-optimization guidance](https://developers.openai.com/api/docs/guides/latency-optimization)
while retaining end-to-end measurements at our own boundary. Provider-side speed
techniques are useful, but only client-observed TTFT, completion latency, output
rate, error rate, and concurrency behaviour can establish this service's SLO.

## Recent routing research and design implications

Recent preprints reinforce the direction of the prototype while also showing why
the current evidence is insufficient for enforcement:

- [UCCI](https://arxiv.org/abs/2605.18796) calibrates an uncertainty score to a
  per-query failure probability and chooses an escalation threshold through
  constrained cost minimisation. Importantly, its reported cascade evidence uses
  actual model outputs and measured hardware latency. Version 0.6 adopts the
  validation-only constrained-search structure, but its uncertainty still predicts
  synthetic prompt complexity rather than candidate-model failure, so the selected
  threshold remains shadow evidence.
- [LLMRouterBench](https://arxiv.org/abs/2601.07206) reports that several complex
  routers do not reliably beat simple baselines under unified evaluation. The
  harness therefore retains argmax, always-capable, tier-risk, heuristic, and random
  comparisons and requires material held-out benefit before promotion.
- [When Routing Collapses](https://arxiv.org/abs/2602.03478) describes routers
  converging on the most expensive model as budgets rise because score prediction
  and discrete model comparison are misaligned. Route mix and the full quality-cost
  frontier should therefore be release evidence, not just aggregate accuracy.
- [R2-Router](https://arxiv.org/abs/2602.02823) treats output length as part of the
  routing decision. Expected output is already included in this router's cost and
  hard-limit checks; jointly optimising model and approved output budget is a useful
  later experiment once response-level quality data exists.

These are research results, not production guarantees. The practical conclusion is
to keep the selector simple, calibrated, and benchmarked against strong baselines
until application outcomes show that a more complex router earns its operational
cost.

## Switchyard deployment position

[NVIDIA NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) supplies
protocol translation, routing algorithms, operational metrics, and fallback
behaviour. For this design, Switchyard is an approved dependency. The repository
pins v0.2.0 and its immutable source commit rather than treating gateway maturity as
a release blocker.

The architecture keeps Switchyard behind a narrow OpenAI-compatible adapter. A deployment should:

1. use the approved v0.2.0 package and source commit;
2. validate its TOML, health, statistics, and exposed routes during CI and startup;
3. run it as an isolated service with resource limits;
4. monitor `/health`, `/v1/stats`, errors, latency, and selected-model headers;
5. keep a trusted direct-provider fail-open path outside Switchyard; and
6. load-test and fault-inject the pinned build before production approval.

The current `config/switchyard_routes.toml` passes the v0.2.0 native Rust-server
contract. A duplicate Luna target found during native startup testing was removed;
the classifier and efficient route now intentionally share one target definition.
The legacy YAML remains only to reproduce the earlier `nemo-switchyard==0.1.0`
prototype.

## Remaining research and measurement

The OpenAI family is an initial controlled candidate, not a permanent provider decision. Before multi-provider routing, benchmark approved Anthropic, Google, NVIDIA NIM, or self-hosted candidates under identical prompts, validators, reasoning budgets, and concurrency. Include contractual data retention, regional availability, quotas, support, deprecation policy, and incident history in the selection—not only token price.

The v3 augmentation-trained multi-view classifier improves internal accuracy to
91.95% and internal tier under-routing to 2.79%, but the non-overlapping multi-turn
slice remains at 59.12% exact accuracy and 17.36% tier under-routing. Better
calibration loss on that external slice does not remove this generalisation
failure. The next research
priority is therefore real, time- or customer-separated traffic with response-level
outcomes, not further optimization against the existing synthetic generator.
