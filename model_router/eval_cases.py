from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_baseline_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for index in range(24):
        left = 7 + index
        right = 3 + (index % 9)
        offset = index * 2 - 5
        answer = left * right + offset
        cases.append(
            {
                "id": f"arithmetic-{index:03d}",
                "prompt": (
                    f"Calculate ({left} × {right}) + ({offset}). "
                    "Reply with only the number."
                ),
                "max_tokens": 16,
                "validators": [
                    {"type": "numeric_tolerance", "value": answer, "tolerance": 0}
                ],
                "metadata": {
                    "category": "arithmetic_reasoning",
                    "use_case": "reasoning",
                    "complexity": "low",
                    "risk": "low",
                    "source": "synthetic_baseline_v1",
                },
            }
        )

    names = (
        "Ada",
        "Grace",
        "Linus",
        "Margaret",
        "Edsger",
        "Barbara",
        "Donald",
        "Frances",
    )
    cities = ("London", "Paris", "Tokyo", "Oslo", "Lima", "Accra")
    priorities = ("low", "medium", "high")
    for index in range(24):
        value = {
            "name": names[index % len(names)],
            "city": cities[(index * 2) % len(cities)],
            "priority": priorities[index % len(priorities)],
            "ticket": f"T-{1200 + index}",
        }
        cases.append(
            {
                "id": f"structured-{index:03d}",
                "prompt": (
                    "Return JSON only with string keys name, city, priority, and ticket. "
                    f"Record: {value['name']} in {value['city']}; priority "
                    f"{value['priority']}; ticket {value['ticket']}."
                ),
                "max_tokens": 80,
                "validators": [{"type": "json_equals", "value": value}],
                "metadata": {
                    "category": "structured_extraction",
                    "use_case": "general_qa",
                    "complexity": "low",
                    "risk": "low",
                    "source": "synthetic_baseline_v1",
                },
            }
        )

    word_sets = (
        ("router", "model", "latency", "quality"),
        ("cache", "token", "budget", "policy"),
        ("audit", "trace", "metric", "alert"),
        ("client", "server", "request", "response"),
        ("alpha", "bravo", "charlie", "delta"),
        ("north", "south", "east", "west"),
    )
    for index in range(24):
        words = word_sets[index % len(word_sets)]
        rotation = index % len(words)
        supplied = words[rotation:] + words[:rotation]
        expected = " | ".join(sorted(supplied))
        cases.append(
            {
                "id": f"strings-{index:03d}",
                "prompt": (
                    f"Sort these words alphabetically: {', '.join(supplied)}. "
                    "Return only the sorted words separated by ' | '."
                ),
                "max_tokens": 40,
                "validators": [{"type": "exact_match", "value": expected}],
                "metadata": {
                    "category": "string_manipulation",
                    "use_case": "general_qa",
                    "complexity": "low",
                    "risk": "low",
                    "source": "synthetic_baseline_v1",
                },
            }
        )

    people = ("Amir", "Bea", "Chen", "Dara", "Elena", "Farah")
    for index in range(24):
        ordered = tuple(people[(index + step) % len(people)] for step in range(4))
        cases.append(
            {
                "id": f"logic-{index:03d}",
                "prompt": (
                    f"{ordered[0]} finished before {ordered[1]}. {ordered[1]} finished "
                    f"before {ordered[2]}. {ordered[2]} finished before {ordered[3]}. "
                    "Who finished first? Reply with only the name."
                ),
                "max_tokens": 16,
                "validators": [{"type": "exact_match", "value": ordered[0]}],
                "metadata": {
                    "category": "logical_reasoning",
                    "use_case": "reasoning",
                    "complexity": "medium",
                    "risk": "low",
                    "source": "synthetic_baseline_v1",
                },
            }
        )

    for index in range(24):
        start = index % 5
        stop = 6 + (index % 8)
        expression = f"sum(x * x for x in range({start}, {stop}) if x % 2 == 0)"
        answer = sum(x * x for x in range(start, stop) if x % 2 == 0)
        cases.append(
            {
                "id": f"python-{index:03d}",
                "prompt": (
                    "What integer does this Python expression return? "
                    f"`{expression}` Reply with only the integer."
                ),
                "max_tokens": 16,
                "validators": [
                    {"type": "numeric_tolerance", "value": answer, "tolerance": 0}
                ],
                "metadata": {
                    "category": "code_comprehension",
                    "use_case": "coding",
                    "complexity": "medium",
                    "risk": "low",
                    "source": "synthetic_baseline_v1",
                },
            }
        )

    if len(cases) != 120 or len({str(case["id"]) for case in cases}) != 120:
        raise AssertionError("baseline evaluation set must contain 120 unique cases")
    return cases


def write_baseline_cases(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n"
            for case in build_baseline_cases()
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen synthetic live cases")
    parser.add_argument("--output", default="examples/live_eval_cases.jsonl")
    args = parser.parse_args()
    write_baseline_cases(args.output)
    print(json.dumps({"cases": 120, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
