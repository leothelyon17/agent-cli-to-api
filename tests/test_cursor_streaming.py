import asyncio
import json
import unittest
from pathlib import Path
from typing import Any

from starlette.requests import Request

from codex_gateway import server
from codex_gateway.codex_responses import CodexAuth, collect_codex_responses_text_and_usage
from codex_gateway.cursor_compat import format_streaming_tool_calls
from codex_gateway.openai_compat import ChatCompletionRequestCompat


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cursor"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _request(path: str = "/v1/chat/completions") -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request({"type": "http", "method": "POST", "path": path, "headers": []}, receive)


async def _collect_sse_json_events(response) -> list[dict[str, Any] | str]:
    events: list[dict[str, Any] | str] = []
    async for chunk in response.body_iterator:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        for frame in text.split("\n\n"):
            frame = frame.strip()
            if not frame or frame.startswith(":"):
                continue
            if not frame.startswith("data:"):
                continue
            data = frame.removeprefix("data:").strip()
            if data == "[DONE]":
                events.append(data)
            else:
                events.append(json.loads(data))
    return events


class CursorStreamingTests(unittest.TestCase):
    def test_format_streaming_tool_calls_adds_indexes_and_stable_ids(self) -> None:
        """Test streaming tool-call formatting adds Cursor-required indexes."""
        # Arrange
        calls = [
            {
                "id": "call_cursor_001",
                "type": "function",
                "function": {"name": "run_terminal_cmd", "arguments": {"command": "pwd"}},
            },
            {"name": "read_file", "arguments": "{}"},
        ]

        # Act
        result = format_streaming_tool_calls(calls)

        # Assert
        self.assertEqual(result[0]["index"], 0)
        self.assertEqual(result[0]["id"], "call_cursor_001")
        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[0]["function"]["name"], "run_terminal_cmd")
        self.assertEqual(result[0]["function"]["arguments"], '{"command": "pwd"}')
        self.assertEqual(result[1]["index"], 1)
        self.assertEqual(result[1]["id"], "call_2")

    def test_collect_codex_responses_text_and_usage_extracts_output_item_tool_call(self) -> None:
        """Test Codex output item done events can provide streamed tool calls."""
        # Arrange
        async def events():
            yield {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_cursor_001",
                    "name": "run_terminal_cmd",
                    "arguments": {"command": "pwd"},
                },
            }
            yield {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 3, "output_tokens": 4}},
            }

        # Act
        text, usage, tool_calls = asyncio.run(collect_codex_responses_text_and_usage(events()))

        # Assert
        self.assertEqual(text, "")
        self.assertEqual(usage["total_tokens"], 7)
        self.assertEqual(tool_calls[0]["id"], "call_cursor_001")
        self.assertEqual(tool_calls[0]["function"]["arguments"], '{"command": "pwd"}')

    def test_chat_route_streaming_tool_call_order_matches_cursor_shape(self) -> None:
        """Test route-level SSE order for tool-call-only Codex responses."""
        # Arrange
        expected_fixture = _fixture("streaming_tool_call_response.sse.json")
        expected_events = expected_fixture["events"]
        original_load_auth = server.load_codex_auth
        original_iter_events = server.iter_codex_responses_events
        original_provider = server.settings.provider
        original_use_codex_responses_api = server.settings.use_codex_responses_api
        original_allow_client_model_override = server.settings.allow_client_model_override

        async def fake_iter_codex_responses_events(**_kwargs):
            yield {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_cursor_001",
                    "name": "run_terminal_cmd",
                    "arguments": '{"command":"pwd"}',
                },
            }
            yield {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 1, "output_tokens": 2}, "output": []},
            }

        server.load_codex_auth = lambda *, codex_cli_home: CodexAuth(
            api_key="test-token",
            access_token=None,
            refresh_token=None,
            account_id=None,
            last_refresh=None,
        )
        server.iter_codex_responses_events = fake_iter_codex_responses_events
        object.__setattr__(server.settings, "provider", "codex")
        object.__setattr__(server.settings, "use_codex_responses_api", True)
        object.__setattr__(server.settings, "allow_client_model_override", True)

        try:
            req = ChatCompletionRequestCompat(**_fixture("tool_prompt_request.json")["request"])
            authorization = f"Bearer {server.settings.bearer_token}" if server.settings.bearer_token else None

            # Act
            response = asyncio.run(server.chat_completions(req, _request(), authorization))
            events = asyncio.run(_collect_sse_json_events(response))

        finally:
            server.load_codex_auth = original_load_auth
            server.iter_codex_responses_events = original_iter_events
            object.__setattr__(server.settings, "provider", original_provider)
            object.__setattr__(server.settings, "use_codex_responses_api", original_use_codex_responses_api)
            object.__setattr__(
                server.settings,
                "allow_client_model_override",
                original_allow_client_model_override,
            )

        # Assert
        self.assertEqual(events[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(events[1]["choices"][0]["finish_reason"], None)
        tool_calls = events[1]["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(tool_calls[0]["index"], 0)
        self.assertEqual(tool_calls[0]["id"], "call_cursor_001")
        self.assertEqual(tool_calls[0]["type"], "function")
        self.assertEqual(tool_calls[0]["function"]["name"], "run_terminal_cmd")
        self.assertEqual(events[2]["choices"][0]["delta"], {})
        self.assertEqual(events[2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(events[3], "[DONE]")
        self.assertEqual([event["choices"][0]["finish_reason"] for event in events[:3]], [None, None, "tool_calls"])
        self.assertEqual(events[1]["choices"][0]["delta"], expected_events[1]["choices"][0]["delta"])
        self.assertEqual(events[2]["choices"][0]["finish_reason"], expected_events[2]["choices"][0]["finish_reason"])


if __name__ == "__main__":
    unittest.main()
