from __future__ import annotations

import unittest

import mcp.servers.quantconnect_stub as qc_stub


class QuantConnectStubTests(unittest.TestCase):
    def test_project_sync_stub(self) -> None:
        with self.assertRaises(RuntimeError) as exc:
            qc_stub.project_sync({"project_id": "123"})
        self.assertIn("QuantConnect MCP integration", str(exc.exception))

    def test_backtest_run_stub(self) -> None:
        with self.assertRaises(RuntimeError):
            qc_stub.backtest_run({"project_id": "123", "name": "test"})

    def test_backtest_status_stub(self) -> None:
        with self.assertRaises(RuntimeError):
            qc_stub.backtest_status({"project_id": "123", "backtest_id": "abc"})


if __name__ == "__main__":
    unittest.main()
