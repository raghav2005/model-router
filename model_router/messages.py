from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ALLOWED_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part_type = str(part.get("type", ""))
        text = part.get("text")
        if part_type in TEXT_PART_TYPES and isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def flatten_messages(messages: object) -> str:
    """Serialize conversation context into the format used by the classifier.

    The previous API path routed only on the final user message. That discarded
    escalation, constraints, code, and tool context from earlier turns. This
    representation is deterministic, rejects malformed roles, and never mutates
    the provider-facing messages.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    flattened: list[str] = []
    has_user_text = False
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"message {index} must be an object")
        role = str(message.get("role", ""))
        if role not in ALLOWED_ROLES:
            raise ValueError(f"message {index} has an unsupported role")
        content = _content_text(message.get("content"))
        if not content.strip():
            # Tool calls and assistant messages can legitimately omit text.
            if role == "user":
                raise ValueError(f"user message {index} must include text")
            continue
        has_user_text = has_user_text or role == "user"
        flattened.append(f"{role.title()}: {content.strip()}")
    if not has_user_text:
        raise ValueError("messages must include a text user message")
    return "\n".join(flattened)


def validate_string_array(value: object, name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array of strings")
    result: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must contain non-empty strings")
        result.add(item.strip())
    return frozenset(result)


def copy_messages(messages: object) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not all(
        isinstance(message, Mapping) for message in messages
    ):
        raise ValueError("messages must be a JSON array of objects")
    return [dict(message) for message in messages]
