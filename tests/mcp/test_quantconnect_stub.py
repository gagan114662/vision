from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import mcp.servers.quantconnect_stub as qc_stub


class DummyHTTPStatusError(Exception):
    def __init__(self, status_code: int):
        self.response = types.SimpleNamespace(status_code=status_code)


class QuantConnectIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Skip environment validation.
        self.ensure_patcher = patch.object(qc_stub, "_ensure_available", return_value=None)
        self.ensure_patcher.start()

        # Replace model constructors with simple dictionaries so we don't depend on pydantic.
        def _factory(**kwargs):
            return kwargs

        self.model_patchers = []
        for name in [
            "CreateProjectFileRequest",
            "UpdateFileContentsRequest",
            "CreateCompileRequest",
            "ReadCompileRequest",
            "CreateBacktestRequest",
            "ReadBacktestRequest",
        ]:
            if hasattr(qc_stub, name):
                patcher = patch.object(qc_stub, name, side_effect=lambda **kw: kw)
                self.model_patchers.append(patcher)
                patcher.start()

        # Provide a dummy httpx module with our custom error type.
        self.httpx_patcher = patch.object(
            qc_stub,
            "httpx",
            types.SimpleNamespace(HTTPStatusError=DummyHTTPStatusError)
        )
        self.httpx_patcher.start()

    def tearDown(self) -> None:
        self.ensure_patcher.stop()
        for patcher in self.model_patchers:
            patcher.stop()
        self.httpx_patcher.stop()

    def test_project_sync_create_then_update(self) -> None:
        calls = []

        def fake_run_post(endpoint, model, timeout=30.0):
            calls.append(endpoint)
            if endpoint == '/files/create':
                raise DummyHTTPStatusError(400)
            return {}

        with patch.object(qc_stub, "_run_post", side_effect=fake_run_post):
            result = qc_stub.project_sync({
                "project_id": "25185664",
                "files": [
                    {"path": "main.py", "content": "print('hi')"}
                ],
            })

        self.assertEqual(result["project_id"], "25185664")
        self.assertEqual(result["synced_files"], ["main.py"])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(calls, ['/files/create', '/files/update'])

    def test_project_sync_records_warning(self) -> None:
        def fake_run_post(endpoint, model, timeout=30.0):
            raise Exception("boom")

        with patch.object(qc_stub, "_run_post", side_effect=fake_run_post):
            result = qc_stub.project_sync({
                "project_id": "123",
                "files": [{"path": "algo.py", "content": ""}],
            })
        self.assertEqual(result["synced_files"], [])
        self.assertEqual(len(result["warnings"]), 1)

    def test_backtest_run_success(self) -> None:
        def fake_run_post(endpoint, model, timeout=30.0):
            if endpoint == '/compile/create':
                return {"compileId": "comp-1"}
            if endpoint == '/compile/read':
                return {"state": "BuildSuccess"}
            if endpoint == '/backtests/create':
                return {"backtest": {"backtestId": "bt-1", "status": "In Queue..."}}
            raise AssertionError(endpoint)

        with patch.object(qc_stub, "_run_post", side_effect=fake_run_post):
            result = qc_stub.backtest_run({
                "project_id": "25185664",
                "name": "ci-dry-run",
            })

        self.assertEqual(result["project_id"], "25185664")
        self.assertEqual(result["backtest_id"], "bt-1")
        self.assertEqual(result["status"], "In Queue...")

    def test_backtest_run_compile_error(self) -> None:
        def fake_run_post(endpoint, model, timeout=30.0):
            if endpoint == '/compile/create':
                return {"compileId": "comp-err"}
            if endpoint == '/compile/read':
                return {"state": "BuildError", "errors": ["syntax"]}
            raise AssertionError(endpoint)

        with patch.object(qc_stub, "_run_post", side_effect=fake_run_post):
            with self.assertRaises(RuntimeError):
                qc_stub.backtest_run({"project_id": "1", "name": "fail"})

    def test_backtest_status(self) -> None:
        def fake_run_post(endpoint, model, timeout=30.0):
            if endpoint == '/backtests/read':
                return {
                    "backtest": {
                        "backtestId": "bt-123",
                        "status": "Completed.",
                        "statistics": {"Sharpe": 1.25, "Nested": {"MaxDrawdown": -0.1}},
                    }
                }
            raise AssertionError(endpoint)

        with patch.object(qc_stub, "_run_post", side_effect=fake_run_post):
            result = qc_stub.backtest_status({
                "project_id": "25185664",
                "backtest_id": "bt-123",
            })

        self.assertEqual(result["status"], "Completed.")
        self.assertIn("Sharpe", result["statistics"])
        self.assertIn("Nested.MaxDrawdown", result["statistics"])


if __name__ == "__main__":
    unittest.main()
