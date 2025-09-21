from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

from mcp.quantconnect_client import QuantConnectMCPClient


class _FakeProcess:
    """Minimal async-compatible subprocess stub for client tests."""

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:  # pragma: no cover - trivial
        await asyncio.sleep(0)


class _StubStream:
    """Async stream stub for `_call_mcp_tool` round-trip validation."""

    def __init__(self) -> None:
        self.buffer: bytes = b""

    def write(self, data: bytes) -> None:
        self.buffer += data

    async def drain(self) -> None:  # pragma: no cover - trivial
        await asyncio.sleep(0)


class _StubStdout:
    def __init__(self, response: Dict[str, Any]) -> None:
        payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": response,
                    }
                ]
            },
        }
        self._line = (json.dumps(payload) + "\n").encode("utf-8")
        self._served = False

    async def readline(self) -> bytes:
        if self._served:
            return b""
        self._served = True
        await asyncio.sleep(0)
        return self._line


class QuantConnectClientFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.original_user = os.environ.pop("QUANTCONNECT_USER_ID", None)
        self.original_token = os.environ.pop("QUANTCONNECT_API_TOKEN", None)

    async def asyncTearDown(self) -> None:
        # Restore environment variables for other tests
        if self.original_user is not None:
            os.environ["QUANTCONNECT_USER_ID"] = self.original_user
        if self.original_token is not None:
            os.environ["QUANTCONNECT_API_TOKEN"] = self.original_token

    async def test_create_project_uses_fallback_without_credentials(self) -> None:
        client = QuantConnectMCPClient(user_id=None, api_token=None)
        result = await client.create_project("demo")
        self.assertEqual(result["method"], "lean_cli_fallback")
        self.assertTrue(result["success"])


class QuantConnectClientIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Ensure credentials exist so non-fallback path is selected
        os.environ["QUANTCONNECT_USER_ID"] = "user"
        os.environ["QUANTCONNECT_API_TOKEN"] = "token"

    async def asyncTearDown(self) -> None:
        os.environ.pop("QUANTCONNECT_USER_ID", None)
        os.environ.pop("QUANTCONNECT_API_TOKEN", None)

    async def test_create_project_happy_path_uses_mcp_and_cleans_up(self) -> None:
        client = QuantConnectMCPClient()
        fake_process = _FakeProcess()

        mock_tool_result = {"projects": [{"projectId": "abc123", "name": "demo"}]}

        with patch.object(
            client,
            "_start_mcp_server",
            new=AsyncMock(return_value=fake_process),
        ) as start_mock, patch.object(
            client,
            "_call_mcp_tool",
            new=AsyncMock(return_value=mock_tool_result),
        ) as call_mock:
            result = await client.create_project("demo")

        start_mock.assert_awaited()
        call_mock.assert_awaited_once()
        call_args = call_mock.call_args.args
        self.assertEqual(call_args[1], "create_project")
        self.assertEqual(result["project_id"], "abc123")
        self.assertTrue(fake_process.terminated)

    async def test_call_mcp_tool_parses_json_rpc_content(self) -> None:
        client = QuantConnectMCPClient()

        stdin_stub = _StubStream()
        stdout_stub = _StubStdout({"ok": True})

        class _Process:
            stdin = stdin_stub
            stdout = stdout_stub

        response = await client._call_mcp_tool(_Process(), "test", {"a": 1})
        # Ensure request serialized correctly and response parsed back
        self.assertIn(b"\n", stdin_stub.buffer)
        self.assertEqual(response, {"ok": True})


class QuantConnectFallbackPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = QuantConnectMCPClient(user_id=None, api_token=None)

    async def test_update_file_fallback_writes_content(self) -> None:
        project_id = "proj"
        file_name = "main.py"
        content = "print('hello')"

        result = await self.client._fallback_update_file(project_id, file_name, content)
        self.assertTrue(result["success"])
        path = Path(result["file_path"])  # type: ignore[index]
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), content)

        # Clean up temp file
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass
        try:
            path.parent.parent.rmdir()
        except OSError:
            pass

