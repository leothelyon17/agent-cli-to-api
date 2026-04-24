from __future__ import annotations

import json
from typing import Any

from .openai_compat import ChatCompletionRequest, ChatMessage, RequestInputError


_FILE_KEYS = ("file_id", "file_data", "filename", "file_url")


def normalize_cursor_chat_request(req: ChatCompletionRequest) -> ChatCompletionRequest:
    """Normalize Cursor and legacy Chat Completions variants for backend conversion."""
    messages = [_normalize_message(message) for message in req.messages]
    extra = dict(getattr(req, "model_extra", None) or {})

    tools = extra.get("tools")
    if tools is None and "functions" in extra:
        tools = _functions_to_tools(extra.get("functions"))
    elif tools is not None:
        tools = _normalize_tools(tools)
    if tools is not None:
        extra["tools"] = tools

    if "tool_choice" in extra:
        extra["tool_choice"] = _normalize_tool_choice(extra["tool_choice"])
    elif "function_call" in extra:
        extra["tool_choice"] = _normalize_function_call(extra["function_call"])

    if "parallel_tool_calls" in extra:
        extra["parallel_tool_calls"] = bool(extra["parallel_tool_calls"])

    for key in ("model", "messages", "stream", "max_tokens"):
        extra.pop(key, None)

    return ChatCompletionRequest(
        model=req.model,
        messages=messages,
        stream=req.stream,
        max_tokens=req.max_tokens,
        **extra,
    )


def format_streaming_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format complete tool calls as OpenAI-compatible streaming deltas."""
    formatted: list[dict[str, Any]] = []
    for fallback_index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue

        function = call.get("function")
        if not isinstance(function, dict):
            function = {}

        name = function.get("name")
        if not isinstance(name, str) or not name:
            name = call.get("name") if isinstance(call.get("name"), str) else "tool"

        arguments = function.get("arguments")
        if arguments is None:
            arguments = call.get("arguments")
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments or {}, ensure_ascii=False)
            except Exception:
                arguments = "{}"

        call_id = call.get("id") or call.get("call_id") or call.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"call_{fallback_index + 1}"

        index = call.get("index")
        if not isinstance(index, int):
            index = fallback_index

        formatted.append(
            {
                "index": index,
                "id": call_id.strip(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return formatted


def _normalize_message(message: ChatMessage) -> ChatMessage:
    extra = dict(getattr(message, "model_extra", None) or {})
    content = _normalize_content(message.content)

    if "tool_calls" in extra:
        extra["tool_calls"] = _normalize_tool_calls(extra["tool_calls"])
    elif "function_call" in extra:
        call = _normalize_function_call_object(extra["function_call"])
        extra["function_call"] = call
        extra["tool_calls"] = [{"type": "function", "function": call}]

    return ChatMessage(role=message.role, content=content, **extra)


def _normalize_content(content: Any) -> Any:
    if isinstance(content, list):
        return [_normalize_content_part(part) for part in content if isinstance(part, dict)]
    if isinstance(content, dict):
        normalized = _normalize_content_part(content)
        if normalized.get("type") == "text":
            return normalized
        return [normalized]
    return content


def _normalize_content_part(part: dict[str, Any]) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type in {"text", "input_text", "output_text"}:
        text = part.get("text")
        if isinstance(text, str):
            return {"type": "text", "text": text}
        return dict(part)

    if part_type in {"image_url", "input_image"}:
        url = _extract_image_url(part)
        if isinstance(url, str) and url.strip():
            return {"type": "image_url", "image_url": {"url": url.strip()}}
        return dict(part)

    if part_type in {"file", "input_file"}:
        file_part = _extract_file_part(part)
        if file_part is not None:
            return {"type": "file", "file": file_part}
        return dict(part)

    return dict(part)


def _extract_image_url(part: dict[str, Any]) -> str | None:
    image = part.get("image_url")
    if isinstance(image, dict) and isinstance(image.get("url"), str):
        return image["url"]
    if isinstance(image, str):
        return image
    url = part.get("url")
    if isinstance(url, str):
        return url
    return None


def _extract_file_part(part: dict[str, Any]) -> dict[str, Any] | None:
    source = part.get("file") if isinstance(part.get("file"), dict) else part
    out: dict[str, Any] = {}
    for key in _FILE_KEYS:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    return out or None


def _functions_to_tools(functions: Any) -> list[dict[str, Any]]:
    if not isinstance(functions, list):
        raise RequestInputError("Invalid functions: expected an array of function definitions")
    return [_normalize_function_tool(item) for item in functions]


def _normalize_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise RequestInputError("Invalid tools: expected an array of tool definitions")
    return [_normalize_tool(tool) for tool in tools]


def _normalize_tool(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise RequestInputError("Invalid tool definition: each tool must be an object")
    if tool.get("type") == "function":
        function = tool.get("function")
        if function is None and _looks_like_flat_function_tool(tool):
            return _normalize_function_tool(tool)
        if not isinstance(function, dict):
            raise RequestInputError("Invalid tool definition: function tool requires a function object")
        return _normalize_function_tool(function, outer=tool)
    if "function" in tool:
        function = tool.get("function")
        if isinstance(function, dict):
            return _normalize_function_tool(function, outer=tool)
    return _normalize_function_tool(tool)


def _looks_like_flat_function_tool(tool: dict[str, Any]) -> bool:
    return any(key in tool for key in ("name", "description", "parameters", "input_schema"))


def _normalize_function_tool(function: Any, *, outer: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(function, dict):
        raise RequestInputError("Invalid function definition: expected an object")

    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RequestInputError("Invalid function definition: missing required string field 'name'")

    normalized_function: dict[str, Any] = {"name": name.strip()}
    description = function.get("description")
    if isinstance(description, str):
        normalized_function["description"] = description

    parameters = function.get("parameters")
    if parameters is None:
        parameters = function.get("input_schema")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise RequestInputError("Invalid function definition: 'parameters' must be an object")
    normalized_function["parameters"] = parameters

    strict = function.get("strict")
    if strict is None and outer is not None:
        strict = outer.get("strict")
    if isinstance(strict, bool):
        normalized_function["strict"] = strict

    return {"type": "function", "function": normalized_function}


def _normalize_tool_choice(choice: Any) -> Any:
    if choice is None:
        return None
    if isinstance(choice, str):
        stripped = choice.strip()
        if stripped in {"auto", "none", "required"}:
            return stripped
        return {"type": "function", "function": {"name": stripped}}
    if not isinstance(choice, dict):
        raise RequestInputError("Invalid tool_choice: expected a string or object")

    choice_type = choice.get("type")
    if choice_type in {"auto", "none", "required"}:
        return choice_type
    if choice_type == "function":
        function = choice.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str) and function["name"].strip():
            return {"type": "function", "function": {"name": function["name"].strip()}}
        name = choice.get("name")
        if isinstance(name, str) and name.strip():
            return {"type": "function", "function": {"name": name.strip()}}
    raise RequestInputError("Invalid tool_choice: function choice requires a function name")


def _normalize_function_call(function_call: Any) -> Any:
    if function_call is None:
        return None
    if isinstance(function_call, str):
        stripped = function_call.strip()
        if stripped in {"auto", "none"}:
            return stripped
        return {"type": "function", "function": {"name": stripped}}
    call = _normalize_function_call_object(function_call)
    return {"type": "function", "function": {"name": call["name"]}}


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        raise RequestInputError("Invalid tool_calls: expected an array")
    return [_normalize_tool_call(call) for call in tool_calls]


def _normalize_tool_call(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise RequestInputError("Invalid tool_call: each tool call must be an object")
    function = call.get("function")
    if function is None:
        function = {"name": call.get("name"), "arguments": call.get("arguments")}
    function_call = _normalize_function_call_object(function)

    out: dict[str, Any] = {"type": "function", "function": function_call}
    call_id = call.get("id") or call.get("call_id") or call.get("tool_call_id")
    if isinstance(call_id, str) and call_id.strip():
        out["id"] = call_id.strip()
    index = call.get("index")
    if isinstance(index, int):
        out["index"] = index
    return out


def _normalize_function_call_object(function_call: Any) -> dict[str, Any]:
    if not isinstance(function_call, dict):
        raise RequestInputError("Invalid function_call: expected an object")
    name = function_call.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RequestInputError("Invalid function_call: missing required string field 'name'")
    arguments = function_call.get("arguments")
    if arguments is None:
        arguments = "{}"
    elif not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, ensure_ascii=False)
        except Exception as exc:
            raise RequestInputError("Invalid function_call: 'arguments' must be JSON serializable") from exc
    return {"name": name.strip(), "arguments": arguments}
