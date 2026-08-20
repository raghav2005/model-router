from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .learned import normalize_prompt

PRIMARY_DATASET = "routing_dataset_100k_valid_only.jsonl"
BALANCED_DATASET = "routing_dataset_100k_valid_balanced_equal_levels.jsonl"
MULTITURN_FAMILY = (
    "routing_dataset_multiturn.jsonl",
    "routing_dataset_multiturn_100k.jsonl",
    "routing_dataset_multiturn_with_models.jsonl",
    "routing_dataset_openai_backoff_test.jsonl",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * fraction) - 1)]


def _read_dataset(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path.name}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"row at {path.name}:{line_number} is not an object")
            prompt = value.get("prompt")
            try:
                level = int(value["complexity_level"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid complexity level at {path.name}:{line_number}"
                ) from error
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or level not in range(1, 6)
            ):
                raise ValueError(f"invalid routing row at {path.name}:{line_number}")
            rows.append(value)
    return rows


def _prompt_map(rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        normalize_prompt(str(row["prompt"])): int(row["complexity_level"])
        for row in rows
    }


def _file_summary(
    path: Path,
    rows: list[dict[str, object]],
    primary_prompts: set[str],
) -> dict[str, object]:
    prompts = [normalize_prompt(str(row["prompt"])) for row in rows]
    labels: dict[str, set[int]] = defaultdict(set)
    for prompt, row in zip(prompts, rows, strict=True):
        labels[prompt].add(int(row["complexity_level"]))
    unique_prompts = set(prompts)
    prompt_lengths = [len(str(row["prompt"])) for row in rows]
    confidence = [
        float(row["complexity_audit_confidence"])
        for row in rows
        if "complexity_audit_confidence" in row
    ]
    return {
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "unique_normalized_prompts": len(unique_prompts),
        "duplicate_rows": len(prompts) - len(unique_prompts),
        "conflicting_duplicate_labels": sum(
            len(prompt_labels) > 1 for prompt_labels in labels.values()
        ),
        "overlap_with_primary": len(unique_prompts & primary_prompts),
        "novel_vs_primary": len(unique_prompts - primary_prompts),
        "levels": dict(
            sorted(Counter(int(row["complexity_level"]) for row in rows).items())
        ),
        "categories": len({str(row.get("category", "unknown")) for row in rows}),
        "statuses": dict(
            sorted(
                Counter(
                    str(row.get("complexity_audit_status", "missing")) for row in rows
                ).items()
            )
        ),
        "mean_audit_confidence": (
            round(statistics.mean(confidence), 6) if confidence else None
        ),
        "fields": dict(sorted(Counter(key for row in rows for key in row).items())),
        "prompt_characters": {
            "min": min(prompt_lengths),
            "median": statistics.median(prompt_lengths),
            "p95": _percentile(prompt_lengths, 0.95),
            "max": max(prompt_lengths),
        },
    }


def audit_datagen_directory(path: str | Path) -> dict[str, object]:
    """Return prompt-free aggregate diagnostics for the private datagen folder."""

    root = Path(path)
    required = (PRIMARY_DATASET, BALANCED_DATASET, *MULTITURN_FAMILY)
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required dataset files: {', '.join(missing)}")

    datasets = {name: _read_dataset(root / name) for name in required}
    prompt_maps = {name: _prompt_map(rows) for name, rows in datasets.items()}
    primary_prompts = set(prompt_maps[PRIMARY_DATASET])
    files = {
        name: _file_summary(root / name, rows, primary_prompts)
        for name, rows in datasets.items()
    }

    balanced_prompts = set(prompt_maps[BALANCED_DATASET])
    primary_audit_matrix: Counter[tuple[str, int, int]] = Counter()
    for row in datasets[PRIMARY_DATASET]:
        status = str(row.get("complexity_audit_status", "missing"))
        assigned = int(row["complexity_level"])
        suggested = int(row.get("complexity_audit_suggested_level", assigned))
        primary_audit_matrix[(status, assigned, suggested)] += 1

    family_maps = {name: prompt_maps[name] for name in MULTITURN_FAMILY}
    family_sets = [set(values) for values in family_maps.values()]
    family_union = set().union(*family_sets)
    family_intersection = set.intersection(*family_sets)
    family_label_conflicts = 0
    for prompt in family_union:
        labels = {values[prompt] for values in family_maps.values() if prompt in values}
        family_label_conflicts += len(labels) > 1

    cross_file_overlap: dict[str, dict[str, int]] = {}
    names = list(required)
    for left_index, left_name in enumerate(names):
        left = prompt_maps[left_name]
        for right_name in names[left_index + 1 :]:
            right = prompt_maps[right_name]
            shared = set(left) & set(right)
            cross_file_overlap[f"{left_name}|{right_name}"] = {
                "shared_prompts": len(shared),
                "label_conflicts": sum(left[key] != right[key] for key in shared),
            }

    return {
        "schema_version": "model-router-datagen-audit-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "privacy": "Aggregate counts and hashes only; no prompt text is emitted.",
        "files": files,
        "balanced_subset": {
            "all_rows_in_primary": balanced_prompts <= primary_prompts,
            "coverage_of_primary": round(
                len(balanced_prompts) / len(primary_prompts), 6
            ),
        },
        "primary_audit_annotations": {
            "status_assigned_suggested": {
                f"{status}:{assigned}->{suggested}": count
                for (status, assigned, suggested), count in sorted(
                    primary_audit_matrix.items()
                )
            },
            "borderline_with_adjacent_suggestion": sum(
                count
                for (status, assigned, suggested), count in primary_audit_matrix.items()
                if status == "borderline" and assigned != suggested
            ),
        },
        "multiturn_family": {
            "union_unique_prompts": len(family_union),
            "intersection_unique_prompts": len(family_intersection),
            "novel_union_vs_primary": len(family_union - primary_prompts),
            "label_conflicts": family_label_conflicts,
            "independent_sources": 1,
            "interpretation": (
                "The files are metadata/schema variants of one prompt family and "
                "must not be counted as independent datasets."
            ),
        },
        "cross_file_overlap": cross_file_overlap,
        "limitations": [
            "This audit measures data structure, overlap, and labels, not response quality.",
            "The smaller multi-turn family shares its synthetic generation process with the primary dataset.",
            "Audit suggestions indicate ambiguity; they are not independent human labels.",
        ],
    }


def markdown_report(report: dict[str, object]) -> str:
    files = report["files"]
    assert isinstance(files, dict)
    primary = files[PRIMARY_DATASET]
    assert isinstance(primary, dict)
    family = report["multiturn_family"]
    assert isinstance(family, dict)
    annotations = report["primary_audit_annotations"]
    assert isinstance(annotations, dict)
    return "\n".join(
        [
            "# Private Datagen Audit",
            "",
            f"**Generated:** {report['generated_at']}",
            "",
            "This report contains aggregate counts and hashes only. It does not contain training prompts.",
            "",
            "## Findings",
            "",
            f"- The primary dataset contains {int(primary['rows']):,} unique valid prompts across {int(primary['categories'])} categories.",
            f"- {int(annotations['borderline_with_adjacent_suggestion']):,} borderline rows include an unused adjacent audit-suggested label.",
            f"- The four multi-turn files contain {int(family['union_unique_prompts']):,} unique prompts in total, of which only {int(family['novel_union_vs_primary']):,} are novel relative to the primary dataset.",
            "- Those four files are one metadata/schema family, not four independent datasets.",
            f"- Cross-file multi-turn label conflicts: {int(family['label_conflicts']):,}.",
            "",
            "## Decision",
            "",
            f"Keep the {int(family['novel_union_vs_primary']):,} novel multi-turn prompts isolated as an external diagnostic slice. Do not merge the metadata variants or count them repeatedly. Treat audit-suggested labels as experimental ambiguity evidence and promote them only if untouched evaluation improves.",
            "",
            "## Limitation",
            "",
            "All available prompts are synthetic and largely generator-related. Production approval still requires response-level outcomes on representative, time- or customer-separated workload data.",
            "",
        ]
    )


def write_report(
    report: dict[str, object],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(markdown_report(report), encoding="utf-8")
