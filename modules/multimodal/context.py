from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .media import MediaItem, find_media_items


@dataclass(frozen=True)
class CurrentRequestContext:
    """The active user request and only the media added by the current event."""

    user_request_text: str
    event_kind: str
    media_items: list[MediaItem]
    tool_call_ids: tuple[str, ...] = ()


def extract_current_request_context(
    body: dict[str, Any], endpoint: str
) -> CurrentRequestContext:
    endpoint = endpoint.rstrip("/")

    if endpoint in {"/v1/chat/completions", "/v1/messages"}:
        messages = body.get("messages")
        if isinstance(messages, list):
            return _messages_context(messages)
        return CurrentRequestContext("", "text_only", [])

    if endpoint == "/v1/responses":
        return _responses_context(body.get("input"))

    media_items = find_media_items(body)
    return CurrentRequestContext("", "unknown", media_items)


def _messages_context(messages: list[Any]) -> CurrentRequestContext:
    last_index = _last_dict_index(messages)
    if last_index is None:
        return CurrentRequestContext("", "text_only", [])

    last = messages[last_index]
    role = last.get("role")

    if role == "tool":
        start = last_index
        while start > 0:
            previous = messages[start - 1]
            if not isinstance(previous, dict) or previous.get("role") != "tool":
                break
            start -= 1
        event_messages = [
            message
            for message in messages[start : last_index + 1]
            if isinstance(message, dict)
        ]
        return CurrentRequestContext(
            user_request_text=_latest_actual_user_text(messages, start - 1),
            event_kind="tool_result",
            media_items=find_media_items(event_messages),
            tool_call_ids=_tool_result_ids(event_messages),
        )

    if role == "user":
        content = last.get("content")
        is_tool_result = _contains_tool_result(content)
        is_synthetic = is_tool_result or (
            bool(find_media_items(content))
            and _is_immediately_after_tool_activity(messages, last_index)
        )
        if is_synthetic:
            return CurrentRequestContext(
                user_request_text=_latest_actual_user_text(messages, last_index - 1),
                event_kind="synthetic_tool_result",
                media_items=find_media_items(content),
                tool_call_ids=_tool_ids_from_value(content),
            )
        media_items = find_media_items(content)
        return CurrentRequestContext(
            user_request_text=_text_from_content(content),
            event_kind="direct_user" if media_items else "text_only",
            media_items=media_items,
        )

    return CurrentRequestContext(
        user_request_text=_latest_actual_user_text(messages, last_index),
        event_kind="text_only",
        media_items=[],
    )


def _responses_context(input_value: Any) -> CurrentRequestContext:
    if isinstance(input_value, str):
        return CurrentRequestContext(input_value, "text_only", [])
    if not isinstance(input_value, list):
        media_items = find_media_items(input_value)
        return CurrentRequestContext(
            _text_from_content(input_value),
            "direct_user" if media_items else "text_only",
            media_items,
        )
    if not input_value:
        return CurrentRequestContext("", "text_only", [])

    last_index = _last_dict_index(input_value)
    if last_index is None:
        media_items = find_media_items(input_value)
        return CurrentRequestContext(
            _text_from_content(input_value),
            "direct_user" if media_items else "text_only",
            media_items,
        )

    last = input_value[last_index]
    if _is_responses_tool_output(last):
        start = last_index
        while start > 0:
            previous = input_value[start - 1]
            if not isinstance(previous, dict) or not _is_responses_tool_output(previous):
                break
            start -= 1
        event_items = [
            item
            for item in input_value[start : last_index + 1]
            if isinstance(item, dict)
        ]
        return CurrentRequestContext(
            user_request_text=_latest_actual_user_text(input_value, start - 1),
            event_kind="tool_result",
            media_items=find_media_items(event_items),
            tool_call_ids=_tool_result_ids(event_items),
        )

    if last.get("role") == "user":
        content = last.get("content")
        is_tool_result = _contains_tool_result(content)
        is_synthetic = is_tool_result or (
            bool(find_media_items(content))
            and _is_immediately_after_tool_activity(input_value, last_index)
        )
        if is_synthetic:
            return CurrentRequestContext(
                user_request_text=_latest_actual_user_text(input_value, last_index - 1),
                event_kind="synthetic_tool_result",
                media_items=find_media_items(content),
                tool_call_ids=_tool_ids_from_value(content),
            )
        media_items = find_media_items(content)
        return CurrentRequestContext(
            user_request_text=_text_from_content(content),
            event_kind="direct_user" if media_items else "text_only",
            media_items=media_items,
        )

    # Some Responses clients send the current content parts directly in input.
    media_items = find_media_items(input_value)
    return CurrentRequestContext(
        _text_from_content(input_value),
        "direct_user" if media_items else "text_only",
        media_items,
    )


def _last_dict_index(values: list[Any]) -> int | None:
    for index in range(len(values) - 1, -1, -1):
        if isinstance(values[index], dict):
            return index
    return None


def _latest_actual_user_text(messages: list[Any], before_index: int) -> str:
    for index in range(min(before_index, len(messages) - 1), -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        if _contains_tool_result(message.get("content")):
            continue
        if _is_immediately_after_tool_activity(messages, index):
            continue
        text = _text_from_content(message.get("content"))
        if text:
            return text
    return ""


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    return ""


def _contains_tool_result(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("type") in {
            "tool_result",
            "function_call_output",
            "computer_call_output",
        }:
            return True
        return any(_contains_tool_result(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_tool_result(child) for child in value)
    return False


def _is_responses_tool_output(item: dict[str, Any]) -> bool:
    return item.get("type") in {
        "tool_result",
        "function_call_output",
        "computer_call_output",
    }


def _is_immediately_after_tool_activity(messages: list[Any], index: int) -> bool:
    for previous_index in range(index - 1, -1, -1):
        previous = messages[previous_index]
        if not isinstance(previous, dict):
            continue
        role = previous.get("role")
        if role == "tool" or _is_responses_tool_output(previous):
            return True
        if role == "assistant" or previous.get("type") in {
            "function_call",
            "computer_call",
        }:
            return _has_tool_call(previous)
        return False
    return False


def _has_tool_call(message: dict[str, Any]) -> bool:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and bool(tool_calls):
        return True
    content = message.get("content")
    if isinstance(content, list):
        return any(
            isinstance(item, dict)
            and item.get("type") in {"tool_use", "function_call", "computer_call"}
            for item in content
        )
    return message.get("type") in {"function_call", "computer_call"}


def _tool_result_ids(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    ids: list[str] = []
    for message in messages:
        for key in ("tool_call_id", "tool_use_id", "call_id"):
            value = message.get(key)
            if isinstance(value, str) and value and value not in ids:
                ids.append(value)
        for value in _tool_ids_from_value(message):
            if value not in ids:
                ids.append(value)
    return tuple(ids)


def _tool_ids_from_value(value: Any) -> tuple[str, ...]:
    ids: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("tool_call_id", "tool_use_id", "call_id"):
                candidate = node.get(key)
                if isinstance(candidate, str) and candidate and candidate not in ids:
                    ids.append(candidate)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return tuple(ids)
