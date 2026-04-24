import asyncio
import json
import unittest
from pathlib import Path
from typing import Any

from fastapi.responses import JSONResponse
from starlette.requests import Request

from codex_gateway import server
from codex_gateway.codex_responses import CodexAuth, convert_chat_completions_to_codex_responses
from codex_gateway.cursor_compat import normalize_cursor_chat_request
from codex_gateway.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionRequestCompat,
    RequestInputError,
    compat_chat_request_to_chat_request,
    extract_image_urls,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cursor"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _request(path: str = "/v1/chat/completions") -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request({"type": "http", "method": "POST", "path": path, "headers": []}, receive)


async def _collect_sse_json_events(response: Any) -> list[dict[str, Any] | str]:
    events: list[dict[str, Any] | str] = []
    async for chunk in response.body_iterator:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        for frame in text.split("\n\n"):
            frame = frame.strip()
            if not frame or frame.startswith(":") or not frame.startswith("data:"):
                continue
            data = frame.removeprefix("data:").strip()
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


class _FakeCodexResponses:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.payloads: list[dict[str, Any]] = []
        self._original_load_auth: Any = None
        self._original_iter_events: Any = None
        self._original_provider: Any = None
        self._original_use_codex_responses_api: Any = None
        self._original_allow_client_model_override: Any = None

    def __enter__(self) -> "_FakeCodexResponses":
        self._original_load_auth = server.load_codex_auth
        self._original_iter_events = server.iter_codex_responses_events
        self._original_provider = server.settings.provider
        self._original_use_codex_responses_api = server.settings.use_codex_responses_api
        self._original_allow_client_model_override = server.settings.allow_client_model_override

        async def fake_iter_codex_responses_events(**kwargs: Any):
            self.payloads.append(kwargs["payload"])
            for event in self.events:
                yield event

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
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        server.load_codex_auth = self._original_load_auth
        server.iter_codex_responses_events = self._original_iter_events
        object.__setattr__(server.settings, "provider", self._original_provider)
        object.__setattr__(server.settings, "use_codex_responses_api", self._original_use_codex_responses_api)
        object.__setattr__(server.settings, "allow_client_model_override", self._original_allow_client_model_override)


def _text_delta_events(text: str = "fixture ok") -> list[dict[str, Any]]:
    return [
        {"type": "response.output_text.delta", "delta": text},
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 2, "output_tokens": 3}, "output": []},
        },
    ]


def _normalize_request_payload(payload: dict[str, Any]) -> ChatCompletionRequest:
    compat = ChatCompletionRequestCompat(**payload)
    return normalize_cursor_chat_request(compat_chat_request_to_chat_request(compat))


class CursorCompatTests(unittest.TestCase):
    def test_text_fixture_passes_through_chat_route_streaming(self) -> None:
        """Test Cursor text fixture passes through the chat route as SSE."""
        # Arrange
        req = ChatCompletionRequestCompat(**_fixture("text_chat_request.json")["request"])
        authorization = f"Bearer {server.settings.bearer_token}" if server.settings.bearer_token else None

        with _FakeCodexResponses(_text_delta_events("health ok")) as fake:
            # Act
            response = asyncio.run(server.chat_completions(req, _request(), authorization))
            events = asyncio.run(_collect_sse_json_events(response))

        # Assert
        self.assertEqual(events[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(events[1]["choices"][0]["delta"], {"content": "health ok"})
        self.assertEqual(events[2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(events[3], "[DONE]")
        self.assertEqual(fake.payloads[0]["input"][0]["content"][0]["text"], "Reply with a short health check sentence.")

    def test_tool_result_followup_fixture_passes_through_chat_route(self) -> None:
        """Test Cursor tool-result follow-up fixture reaches Codex as function output."""
        # Arrange
        req = ChatCompletionRequestCompat(**_fixture("tool_result_followup_request.json")["request"])
        authorization = f"Bearer {server.settings.bearer_token}" if server.settings.bearer_token else None

        with _FakeCodexResponses(_text_delta_events("tool result accepted")) as fake:
            # Act
            response = asyncio.run(server.chat_completions(req, _request(), authorization))
            events = asyncio.run(_collect_sse_json_events(response))

        # Assert
        self.assertEqual(events[-2]["choices"][0]["finish_reason"], "stop")
        self.assertIn(
            {
                "type": "function_call_output",
                "call_id": "call_cursor_001",
                "output": "{\"stdout\":\"/workspace\",\"stderr\":\"\",\"exit_code\":0}",
            },
            fake.payloads[0]["input"],
        )

    def test_image_fixture_passes_through_chat_route_non_streaming(self) -> None:
        """Test Cursor image fixture reaches Codex through the chat route."""
        # Arrange
        req = ChatCompletionRequestCompat(**_fixture("image_upload_request.json")["request"])
        authorization = f"Bearer {server.settings.bearer_token}" if server.settings.bearer_token else None

        with _FakeCodexResponses(_text_delta_events("one pixel image")) as fake:
            # Act
            response = asyncio.run(server.chat_completions(req, _request(), authorization))

        # Assert
        self.assertIsInstance(response, dict)
        self.assertEqual(response["choices"][0]["message"]["content"], "one pixel image")
        content = fake.payloads[0]["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))

    def test_cursor_request_fixtures_normalize_without_validation_errors(self) -> None:
        """Test every Cursor request fixture normalizes into a chat request."""
        # Arrange
        fixture_names = [
            "image_upload_request.json",
            "subagent_prompt_request.json",
            "text_chat_request.json",
            "tool_prompt_request.json",
            "tool_result_followup_request.json",
        ]

        for name in fixture_names:
            with self.subTest(name=name):
                payload = _fixture(name)["request"]

                # Act
                req = _normalize_request_payload(payload)

                # Assert
                self.assertGreaterEqual(len(req.messages), 1)
                self.assertIsNotNone(req.model)

    def test_cursor_tool_fixture_normalizes_to_function_tools(self) -> None:
        """Test Cursor tool definitions normalize to OpenAI function tools."""
        # Arrange
        payload = _fixture("tool_prompt_request.json")["request"]

        # Act
        req = _normalize_request_payload(payload)

        # Assert
        tools = req.model_extra["tools"]
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "run_terminal_cmd")
        self.assertEqual(tools[0]["function"]["parameters"]["type"], "object")
        self.assertEqual(req.model_extra["tool_choice"], "auto")
        self.assertFalse(req.model_extra["parallel_tool_calls"])

    def test_legacy_functions_and_function_call_normalize_to_tool_fields(self) -> None:
        """Test legacy function-calling fields normalize to current tool fields."""
        # Arrange
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "run pwd"}],
            "functions": [
                {
                    "name": "run_terminal_cmd",
                    "description": "Run a terminal command.",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                }
            ],
            "function_call": {"name": "run_terminal_cmd"},
        }

        # Act
        req = _normalize_request_payload(payload)

        # Assert
        self.assertEqual(req.model_extra["tools"][0]["function"]["name"], "run_terminal_cmd")
        self.assertEqual(
            req.model_extra["tool_choice"],
            {"type": "function", "function": {"name": "run_terminal_cmd"}},
        )

    def test_flat_cursor_tool_definition_normalizes_to_function_tool(self) -> None:
        """Test flat function objects are accepted as tool definitions."""
        # Arrange
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "run pwd"}],
            "tools": [{"name": "run_terminal_cmd", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "run_terminal_cmd"},
        }

        # Act
        req = _normalize_request_payload(payload)

        # Assert
        self.assertEqual(
            req.model_extra["tools"],
            [
                {
                    "type": "function",
                    "function": {"name": "run_terminal_cmd", "parameters": {"type": "object"}},
                }
            ],
        )
        self.assertEqual(req.model_extra["tool_choice"]["function"]["name"], "run_terminal_cmd")

    def test_flat_cursor_tool_with_function_type_normalizes_to_function_tool(self) -> None:
        """Test Cursor function tools without nested function objects are accepted."""
        # Arrange
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "run pwd"}],
            "tools": [
                {
                    "type": "function",
                    "name": "run_terminal_cmd",
                    "description": "Run a terminal command.",
                    "parameters": {"type": "object"},
                }
            ],
        }

        # Act
        req = _normalize_request_payload(payload)

        # Assert
        self.assertEqual(
            req.model_extra["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "run_terminal_cmd",
                        "description": "Run a terminal command.",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

    def test_cursor_content_part_variants_normalize_to_canonical_parts(self) -> None:
        """Test Cursor image and file content variants normalize to canonical OpenAI parts."""
        # Arrange
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "inspect"},
                        {"type": "input_image", "url": "data:image/png;base64,AAA"},
                        {"type": "input_file", "filename": "label.pdf", "file_data": "JVBERi0="},
                    ],
                }
            ],
        }

        # Act
        req = _normalize_request_payload(payload)

        # Assert
        self.assertEqual(
            req.messages[0].content,
            [
                {"type": "text", "text": "inspect"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "file", "file": {"file_data": "JVBERi0=", "filename": "label.pdf"}},
            ],
        )

    def test_tool_result_followup_fixture_converts_to_codex_function_output(self) -> None:
        """Test Cursor tool-result follow-ups convert to Codex function call output."""
        # Arrange
        req = _normalize_request_payload(_fixture("tool_result_followup_request.json")["request"])

        # Act
        payload = convert_chat_completions_to_codex_responses(
            req,
            model_name="gpt-5.4",
            force_stream=True,
            allow_tools=True,
        )

        # Assert
        self.assertIn(
            {
                "type": "function_call_output",
                "call_id": "call_cursor_001",
                "output": "{\"stdout\":\"/workspace\",\"stderr\":\"\",\"exit_code\":0}",
            },
            payload["input"],
        )

    def test_image_upload_fixture_converts_to_codex_input_image(self) -> None:
        """Test Cursor image fixtures convert to Codex input image parts."""
        # Arrange
        req = _normalize_request_payload(_fixture("image_upload_request.json")["request"])

        # Act
        payload = convert_chat_completions_to_codex_responses(
            req,
            model_name="gpt-5.4",
            force_stream=False,
        )

        # Assert
        content = payload["input"][0]["content"]
        self.assertEqual(
            content[0],
            {"type": "input_text", "text": "Describe this synthetic test image in one sentence."},
        )
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))

    def test_input_image_url_variant_converts_to_codex_input_image(self) -> None:
        """Test input_image parts with top-level url are accepted."""
        # Arrange
        req = ChatCompletionRequest(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        {"type": "input_image", "url": "data:image/png;base64,QUJD"},
                    ],
                }
            ],
        )

        # Act
        image_urls = extract_image_urls(req.messages)
        payload = convert_chat_completions_to_codex_responses(
            req,
            model_name="gpt-5.4",
            force_stream=False,
        )

        # Assert
        self.assertEqual(image_urls, ["data:image/png;base64,QUJD"])
        self.assertIn(
            {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
            payload["input"][0]["content"],
        )

    def test_chat_route_rejects_oversized_image_data_url(self) -> None:
        """Test oversized image data URLs return OpenAI-style errors."""
        # Arrange
        req = ChatCompletionRequestCompat(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                    ],
                }
            ],
        )
        authorization = f"Bearer {server.settings.bearer_token}" if server.settings.bearer_token else None
        original_max_image_bytes = server.settings.max_image_bytes

        try:
            object.__setattr__(server.settings, "max_image_bytes", 1)

            # Act
            response = asyncio.run(server.chat_completions(req, _request(), authorization))
        finally:
            object.__setattr__(server.settings, "max_image_bytes", original_max_image_bytes)

        # Assert
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 413)
        body = json.loads(response.body.decode("utf-8"))
        self.assertIn("error", body)
        self.assertIn("Image too large", body["error"]["message"])

    def test_curl_logging_redacts_inline_image_data(self) -> None:
        """Test diagnostic curl payloads include image metadata without raw base64."""
        # Arrange
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                    ],
                }
            ],
        }

        # Act
        curl = server._build_curl_command(
            url="http://testserver/v1/chat/completions",
            authorization=None,
            payload=payload,
            stream=False,
        )

        # Assert
        self.assertIn("<redacted:3 bytes>", curl)
        self.assertNotIn("QUJD", curl)

    def test_curl_logging_redacts_authorization_header(self) -> None:
        """Test diagnostic curl output never includes bearer token values."""
        # Arrange
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        # Act
        curl = server._build_curl_command(
            url="http://testserver/v1/chat/completions",
            authorization="Bearer secret-cursor-token",
            payload=payload,
            stream=False,
        )

        # Assert
        self.assertIn("Authorization: Bearer <redacted>", curl)
        self.assertNotIn("secret-cursor-token", curl)

    def test_log_payload_redaction_masks_sensitive_tool_fields(self) -> None:
        """Test tool argument/result logs redact common credential fields."""
        # Arrange
        payload = {
            "command": "curl -H 'Authorization: Bearer tool-token' https://example.invalid",
            "api_key": "sk-test",
            "max_tokens": 128,
            "nested": {"refresh_token": "refresh-secret", "normal": "visible"},
        }

        # Act
        dumped = server._json_for_log(payload)

        # Assert
        self.assertIn('"api_key": "<redacted>"', dumped)
        self.assertIn('"refresh_token": "<redacted>"', dumped)
        self.assertIn("Bearer <redacted>", dumped)
        self.assertIn('"max_tokens": 128', dumped)
        self.assertIn('"normal": "visible"', dumped)
        self.assertNotIn("sk-test", dumped)
        self.assertNotIn("refresh-secret", dumped)
        self.assertNotIn("tool-token", dumped)

    def test_invalid_tool_schema_raises_request_input_error(self) -> None:
        """Test malformed tools fail closed before backend conversion."""
        # Arrange
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "bad", "parameters": []}}],
        }

        # Act & Assert
        with self.assertRaisesRegex(RequestInputError, "parameters"):
            _normalize_request_payload(payload)

    def test_chat_route_returns_openai_style_error_for_invalid_tool_schema(self) -> None:
        """Test malformed Cursor tools return the gateway OpenAI-style error response."""
        # Arrange
        req = ChatCompletionRequestCompat(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "bad", "parameters": []}}],
        )
        authorization = f"Bearer {server.settings.bearer_token}" if server.settings.bearer_token else None

        # Act
        response = asyncio.run(server.chat_completions(req, _request(), authorization))

        # Assert
        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 422)
        body = json.loads(response.body.decode("utf-8"))
        self.assertIn("error", body)
        self.assertIn("parameters", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
