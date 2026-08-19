from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .adversarial import evaluate_adversarial_suite, write_cases, write_report
from .benchmark import run_benchmark
from .classifier import DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD
from .drift import (
    build_baseline,
    compare_to_baseline,
    load_route_events,
    read_baseline,
)
from .drift import write_json as write_drift_json
from .learned import default_artifact_path
from .live_eval import case_set_sha256
from .live_eval import load_cases as load_live_eval_cases
from .live_eval import run_benchmark as run_live_benchmark
from .messages import copy_messages, flatten_messages
from .metamorphic import evaluate_metamorphic_suite
from .metamorphic import write_cases as write_metamorphic_cases
from .metamorphic import write_report as write_metamorphic_report
from .pricing import verify_catalog_pricing
from .pricing import write_report as write_pricing_report
from .readiness import evaluate_release_gates
from .router import ModelRouter, NoEligibleModel
from .switchyard import (
    DELEGATED_PROFILES,
    SwitchyardClient,
    SwitchyardError,
    SwitchyardExecutor,
    UnsafeDelegatedRoute,
)
from .training import TrainingConfig, train_and_evaluate
from .types import RoutingRequest

ROUTING_MODES = ("policy", *DELEGATED_PROFILES)


def _add_request_options(
    parser: argparse.ArgumentParser, *, optional_prompt: bool = False
) -> None:
    parser.add_argument("prompt", nargs="?" if optional_prompt else None)
    parser.add_argument(
        "--priority",
        choices=["balanced", "cost", "quality", "latency"],
        default="balanced",
    )
    parser.add_argument("--output-tokens", type=int, default=500)
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--request-id")
    parser.add_argument("--tenant-id")
    parser.add_argument("--cached-input-tokens", type=int, default=0)
    parser.add_argument("--cache-write-tokens", type=int, default=0)
    parser.add_argument("--max-cost", type=float)
    parser.add_argument("--max-latency", type=int)
    parser.add_argument("--capability", action="append", default=[])
    parser.add_argument("--use-case", choices=["general_qa", "coding", "reasoning"])
    parser.add_argument("--allow-model", action="append", default=[])
    parser.add_argument("--allow-provider", action="append", default=[])
    parser.add_argument(
        "--classifier-mode",
        choices=["heuristic", "learned", "hybrid"],
        default=os.getenv("MODEL_ROUTER_CLASSIFIER_MODE", "hybrid"),
    )
    parser.add_argument(
        "--model-artifact",
        default=os.getenv("MODEL_ROUTER_ARTIFACT", str(default_artifact_path())),
    )
    parser.add_argument(
        "--complexity-policy",
        choices=["argmax", "expected", "conservative", "tier_risk", "adaptive"],
        default=os.getenv("MODEL_ROUTER_COMPLEXITY_POLICY", "adaptive"),
    )
    parser.add_argument("--underroute-tolerance", type=float, default=0.20)
    parser.add_argument(
        "--adaptive-confidence-threshold",
        type=float,
        default=float(
            os.getenv(
                "MODEL_ROUTER_ADAPTIVE_CONFIDENCE_THRESHOLD",
                str(DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD),
            )
        ),
    )
    parser.add_argument("--require-measured-quality", action="store_true")
    parser.add_argument("--require-measured-latency", action="store_true")


def _load_messages(path: str | None, prompt: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not all(
            isinstance(message, dict) for message in raw
        ):
            raise ValueError("messages file must contain a JSON array of objects")
        messages.extend(copy_messages(raw))
    if prompt:
        messages.append({"role": "user", "content": prompt})
    if not messages:
        raise ValueError("provide a prompt, a messages file, or both")
    return messages


def _routing_prompt(prompt: str | None, messages: list[dict[str, Any]]) -> str:
    return prompt if prompt and len(messages) == 1 else flatten_messages(messages)


def _routing_request(args: argparse.Namespace, prompt: str) -> RoutingRequest:
    return RoutingRequest(
        prompt=prompt,
        request_id=args.request_id,
        tenant_id=args.tenant_id,
        input_tokens=args.input_tokens,
        cached_input_tokens=args.cached_input_tokens,
        cache_write_tokens=args.cache_write_tokens,
        expected_output_tokens=args.output_tokens,
        required_capabilities=frozenset(args.capability),
        priority=args.priority,
        max_cost_usd=args.max_cost,
        max_latency_ms=args.max_latency,
        use_case=args.use_case,
        allowed_model_ids=frozenset(args.allow_model),
        allowed_providers=frozenset(args.allow_provider),
    )


def _model_router(args: argparse.Namespace) -> ModelRouter:
    if args.classifier_mode == "heuristic":
        return ModelRouter(
            require_measured_quality=args.require_measured_quality,
            require_measured_latency=args.require_measured_latency,
        )
    return ModelRouter.from_artifact(
        args.model_artifact,
        classifier_mode=args.classifier_mode,
        decision_policy=args.complexity_policy,
        underroute_tolerance=args.underroute_tolerance,
        adaptive_confidence_threshold=args.adaptive_confidence_threshold,
        require_measured_quality=args.require_measured_quality,
        require_measured_latency=args.require_measured_latency,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explainable LLM model router")
    subparsers = parser.add_subparsers(dest="command", required=True)
    route = subparsers.add_parser("route", help="Route one prompt")
    _add_request_options(route)

    plan = subparsers.add_parser(
        "plan", help="Show the custom/Switchyard execution plan without a model call"
    )
    _add_request_options(plan, optional_prompt=True)
    plan.add_argument("--mode", choices=ROUTING_MODES, default="policy")
    plan.add_argument("--messages-file")

    execute = subparsers.add_parser(
        "execute", help="Route and send a chat completion through Switchyard"
    )
    _add_request_options(execute, optional_prompt=True)
    execute.add_argument("--mode", choices=ROUTING_MODES, default="policy")
    execute.add_argument("--messages-file")
    execute.add_argument(
        "--switchyard-url",
        default=os.getenv("SWITCHYARD_URL", "http://127.0.0.1:4000"),
    )
    execute.add_argument("--timeout", type=float, default=60.0)
    execute.add_argument("--temperature", type=float)

    status = subparsers.add_parser(
        "switchyard-status", help="Read models or runtime statistics from Switchyard"
    )
    status.add_argument(
        "--switchyard-url",
        default=os.getenv("SWITCHYARD_URL", "http://127.0.0.1:4000"),
    )
    status.add_argument("--timeout", type=float, default=10.0)
    status.add_argument("--stats", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="Run the 12-case policy smoke test"
    )
    benchmark.add_argument("--iterations", type=int, default=300)
    benchmark.add_argument(
        "--classifier-mode",
        choices=["heuristic", "learned", "hybrid"],
        default="hybrid",
    )
    benchmark.add_argument("--model-artifact", default=str(default_artifact_path()))
    benchmark.add_argument(
        "--complexity-policy",
        choices=["argmax", "expected", "conservative", "tier_risk", "adaptive"],
        default="adaptive",
    )
    benchmark.add_argument("--underroute-tolerance", type=float, default=0.20)
    benchmark.add_argument(
        "--adaptive-confidence-threshold",
        type=float,
        default=DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD,
    )
    benchmark.add_argument("--require-measured-quality", action="store_true")
    benchmark.add_argument("--require-measured-latency", action="store_true")

    train = subparsers.add_parser(
        "train", help="Train and evaluate a learned complexity router"
    )
    train.add_argument("dataset")
    train.add_argument(
        "--artifact", default="model_router/artifacts/complexity_router_v2.npz"
    )
    train.add_argument("--report-json", default="reports/complexity_router_v2.json")
    train.add_argument("--report-markdown", default="reports/complexity_router_v2.md")
    train.add_argument("--external-context-dataset")
    train.add_argument("--feature-dimension", type=int, default=32_768)
    train.add_argument("--borderline-weight", type=float, default=0.65)
    train.add_argument(
        "--class-prior", choices=["uniform", "empirical"], default="uniform"
    )
    train.add_argument("--augmentation-copies", type=int, default=1)
    train.add_argument("--augmentation-weight", type=float, default=0.5)

    live_eval = subparsers.add_parser(
        "live-eval",
        help="Run frozen cases against direct Switchyard targets",
    )
    live_eval.add_argument("cases")
    live_eval.add_argument("--target", action="append", required=True, dest="targets")
    live_eval.add_argument("--output", default="reports/live_eval_results.jsonl")
    live_eval.add_argument("--summary", default="reports/live_eval_summary.json")
    live_eval.add_argument(
        "--switchyard-url",
        default=os.getenv("SWITCHYARD_URL", "http://127.0.0.1:4000"),
    )
    live_eval.add_argument("--timeout", type=float, default=120.0)
    live_eval.add_argument("--concurrency", type=int, default=4)
    live_eval.add_argument("--store-content", action="store_true")
    live_eval.add_argument("--no-resume", action="store_true")
    live_eval.add_argument("--repetitions", type=int, default=1)
    live_eval.add_argument(
        "--stream", action="store_true", help="Measure streaming time to first token"
    )
    live_eval.add_argument(
        "--switchyard-revision",
        default=os.getenv("SWITCHYARD_REVISION"),
        help="Exact qualified Switchyard version or commit used by this run",
    )

    case_set_hash = subparsers.add_parser(
        "case-set-hash",
        help="Compute the canonical digest for an evaluation case set",
    )
    case_set_hash.add_argument("cases")

    release_gates = subparsers.add_parser(
        "release-gates", help="Evaluate production-enforcement release gates"
    )
    release_gates.add_argument("--policy", default="config/release_policy.json")
    release_gates.add_argument(
        "--training-report", default="reports/complexity_router_v2.json"
    )
    release_gates.add_argument(
        "--live-summary", default="reports/live_eval_summary.json"
    )
    release_gates.add_argument(
        "--pricing-report", default="reports/pricing_verification.json"
    )
    release_gates.add_argument(
        "--metamorphic-report", default="reports/metamorphic_routing_eval.json"
    )
    release_gates.add_argument(
        "--switchyard-contract-report", default="reports/switchyard_contract.json"
    )
    release_gates.add_argument("--model-artifact", default=str(default_artifact_path()))

    verify_pricing = subparsers.add_parser(
        "verify-pricing",
        help="Compare catalogue economics with official provider documentation",
    )
    verify_pricing.add_argument("--catalog")
    verify_pricing.add_argument("--output", default="reports/pricing_verification.json")
    verify_pricing.add_argument("--timeout", type=float, default=15.0)

    adversarial = subparsers.add_parser(
        "adversarial-eval",
        help="Run deterministic synthetic routing regression cases",
    )
    adversarial.add_argument("--model-artifact", default=str(default_artifact_path()))
    adversarial.add_argument(
        "--cases-output", default="data/adversarial_routing_cases.jsonl"
    )
    adversarial.add_argument(
        "--report-output", default="reports/adversarial_routing_eval.json"
    )

    metamorphic = subparsers.add_parser(
        "metamorphic-eval",
        help="Test routing invariance under meaning-preserving prompt transformations",
    )
    metamorphic.add_argument("--model-artifact", default=str(default_artifact_path()))
    metamorphic.add_argument(
        "--cases-output", default="data/metamorphic_routing_cases.jsonl"
    )
    metamorphic.add_argument(
        "--report-output", default="reports/metamorphic_routing_eval.json"
    )

    drift_baseline = subparsers.add_parser(
        "drift-baseline",
        help="Create an approved routing-distribution baseline from audit events",
    )
    drift_baseline.add_argument("audit_log")
    drift_baseline.add_argument("--output", default="config/drift_baseline.json")

    drift_report = subparsers.add_parser(
        "drift-report", help="Compare recent prompt-free audit events to a baseline"
    )
    drift_report.add_argument("audit_log")
    drift_report.add_argument("--baseline", default="config/drift_baseline.json")
    drift_report.add_argument("--output", default="reports/drift_report.json")
    drift_report.add_argument("--minimum-events", type=int, default=100)
    drift_report.add_argument("--maximum-js-distance", type=float, default=0.10)
    drift_report.add_argument(
        "--maximum-standardized-mean-shift", type=float, default=2.0
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "case-set-hash":
            cases = load_live_eval_cases(args.cases)
            print(
                json.dumps(
                    {
                        "case_count": len(cases),
                        "case_set_sha256": case_set_sha256(cases),
                    },
                    indent=2,
                )
            )
            return

        if args.command == "verify-pricing":
            report = verify_catalog_pricing(
                catalog_path=args.catalog,
                timeout_seconds=args.timeout,
            )
            write_pricing_report(args.output, report)
            print(json.dumps(report, indent=2))
            if not report["passed"]:
                raise SystemExit(4)
            return

        if args.command == "drift-baseline":
            baseline = build_baseline(load_route_events(args.audit_log))
            write_drift_json(args.output, baseline)
            print(json.dumps(baseline, indent=2))
            return

        if args.command == "drift-report":
            report = compare_to_baseline(
                read_baseline(args.baseline),
                load_route_events(args.audit_log),
                minimum_events=args.minimum_events,
                maximum_js_distance=args.maximum_js_distance,
                maximum_standardized_mean_shift=(args.maximum_standardized_mean_shift),
            )
            write_drift_json(args.output, report)
            print(json.dumps(report, indent=2))
            if not report["passed"]:
                raise SystemExit(2)
            return

        if args.command == "adversarial-eval":
            report = evaluate_adversarial_suite(artifact_path=args.model_artifact)
            write_cases(args.cases_output)
            write_report(args.report_output, report)
            print(json.dumps(report, indent=2))
            return

        if args.command == "metamorphic-eval":
            report = evaluate_metamorphic_suite(artifact_path=args.model_artifact)
            write_metamorphic_cases(args.cases_output)
            write_metamorphic_report(args.report_output, report)
            print(json.dumps(report, indent=2))
            return

        if args.command == "release-gates":
            report = evaluate_release_gates(
                policy_path=args.policy,
                training_report_path=args.training_report,
                live_summary_path=args.live_summary,
                pricing_report_path=args.pricing_report,
                metamorphic_report_path=args.metamorphic_report,
                switchyard_contract_report_path=args.switchyard_contract_report,
                artifact_path=args.model_artifact,
            )
            print(json.dumps(report, indent=2))
            if not report["ready_for_enforcement"]:
                parser.exit(3)
            return

        if args.command == "train":
            report = train_and_evaluate(
                args.dataset,
                args.artifact,
                args.report_json,
                args.report_markdown,
                external_context_path=args.external_context_dataset,
                config=TrainingConfig(
                    feature_dimension=args.feature_dimension,
                    borderline_weight=args.borderline_weight,
                    class_prior=args.class_prior,
                    augmentation_copies_per_train_row=args.augmentation_copies,
                    augmentation_weight=args.augmentation_weight,
                ),
            )
            print(
                json.dumps(
                    {
                        "artifact": args.artifact,
                        "report": args.report_markdown,
                        "test_metrics": report["evaluation"]["strategies"][
                            "learned_argmax"
                        ],
                    },
                    indent=2,
                )
            )
            return

        if args.command == "live-eval":
            client = SwitchyardClient(
                args.switchyard_url,
                api_key=os.getenv("SWITCHYARD_API_KEY"),
                timeout_seconds=args.timeout,
            )
            result = run_live_benchmark(
                client,
                load_live_eval_cases(args.cases),
                args.targets,
                output_path=args.output,
                summary_path=args.summary,
                concurrency=args.concurrency,
                store_content=args.store_content,
                resume=not args.no_resume,
                repetitions=args.repetitions,
                stream=args.stream,
                switchyard_revision=args.switchyard_revision,
            )
            print(json.dumps(result, indent=2))
            return

        if args.command == "benchmark":
            print(
                json.dumps(
                    run_benchmark(args.iterations, router=_model_router(args)),
                    indent=2,
                )
            )
            return

        if args.command == "switchyard-status":
            client = SwitchyardClient(
                args.switchyard_url,
                api_key=os.getenv("SWITCHYARD_API_KEY"),
                timeout_seconds=args.timeout,
            )
            result = client.stats() if args.stats else client.list_models()
            print(json.dumps(result, indent=2))
            return

        if args.command == "route":
            decision = _model_router(args).route(_routing_request(args, args.prompt))
            print(json.dumps(decision.as_dict(), indent=2))
            return

        messages = _load_messages(args.messages_file, args.prompt)
        request = _routing_request(args, _routing_prompt(args.prompt, messages))
        client = SwitchyardClient(
            getattr(args, "switchyard_url", "http://127.0.0.1:4000"),
            api_key=os.getenv("SWITCHYARD_API_KEY"),
            timeout_seconds=getattr(args, "timeout", 60.0),
        )
        executor = SwitchyardExecutor(client, router=_model_router(args))
        if args.command == "plan":
            result = executor.plan(
                request, routing_mode=args.mode, messages=messages
            ).as_dict()
        else:
            result = executor.execute(
                request,
                routing_mode=args.mode,
                messages=messages,
                temperature=args.temperature,
            ).as_dict()
        print(json.dumps(result, indent=2))
    except (
        NoEligibleModel,
        SwitchyardError,
        UnsafeDelegatedRoute,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
