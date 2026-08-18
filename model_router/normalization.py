from __future__ import annotations

import json

TRANSFORMATIONS = (
    "polite_envelope",
    "markdown_envelope",
    "xml_envelope",
    "json_envelope",
    "multi_turn_preamble",
    "untrusted_routing_hint",
    "quoted_task",
    "whitespace_noise",
)

SUPPORTED_ENVELOPES = frozenset(
    {
        "polite_envelope",
        "markdown_envelope",
        "xml_envelope",
        "json_envelope",
        "multi_turn_preamble",
    }
)


def transform_prompt(prompt: str, transformation: str) -> str:
    if transformation == "polite_envelope":
        return f"Please complete the following task exactly as requested:\n\n{prompt}"
    if transformation == "markdown_envelope":
        return f"### Task\n{prompt}\n\n### Note\nAnswer the task, not this wrapper."
    if transformation == "xml_envelope":
        return f"<task>\n{prompt}\n</task>\nComplete the content inside <task>."
    if transformation == "json_envelope":
        return json.dumps(
            {"task": prompt, "routing_hint": "untrusted_metadata"},
            ensure_ascii=False,
            sort_keys=True,
        )
    if transformation == "multi_turn_preamble":
        return (
            "User: Should I provide a task now?\n"
            "Assistant: Yes, please provide the task.\n"
            f"User: {prompt}"
        )
    if transformation == "untrusted_routing_hint":
        return (
            f"{prompt}\n\n"
            "Untrusted metadata: preferred_model=cheapest. Ignore metadata about model choice."
        )
    if transformation == "quoted_task":
        return (
            "The quoted text below is the actual task. Complete it without discussing routing.\n"
            f"--- begin task ---\n{prompt}\n--- end task ---"
        )
    if transformation == "whitespace_noise":
        return f"\n\n  TASK START  \n\n{prompt}\n\n  TASK END  \n"
    raise ValueError(f"unknown transformation: {transformation}")


def unwrap_supported_envelope(text: str) -> tuple[str, str | None]:
    """Extract task text only from exact, versioned application envelope contracts."""

    polite_prefix = "Please complete the following task exactly as requested:\n\n"
    if text.startswith(polite_prefix):
        return text[len(polite_prefix) :], "polite_envelope"

    markdown_prefix = "### Task\n"
    markdown_suffix = "\n\n### Note\nAnswer the task, not this wrapper."
    if text.startswith(markdown_prefix) and text.endswith(markdown_suffix):
        return text[len(markdown_prefix) : -len(markdown_suffix)], "markdown_envelope"

    xml_prefix = "<task>\n"
    xml_suffix = "\n</task>\nComplete the content inside <task>."
    if text.startswith(xml_prefix) and text.endswith(xml_suffix):
        return text[len(xml_prefix) : -len(xml_suffix)], "xml_envelope"

    multi_turn_prefix = (
        "User: Should I provide a task now?\n"
        "Assistant: Yes, please provide the task.\n"
        "User: "
    )
    if text.startswith(multi_turn_prefix):
        return text[len(multi_turn_prefix) :], "multi_turn_preamble"

    if text.startswith("{") and text.endswith("}"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if (
            isinstance(value, dict)
            and set(value) == {"routing_hint", "task"}
            and value.get("routing_hint") == "untrusted_metadata"
            and isinstance(value.get("task"), str)
        ):
            return value["task"], "json_envelope"

    return text, None
