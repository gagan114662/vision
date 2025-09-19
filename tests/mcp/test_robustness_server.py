from __future__ import annotations

import math
import unittest

import mcp.servers.robustness_server as robustness_server


class RobustnessServerTests(unittest.TestCase):
    def test_run_robustness_pass(self) -> None:
        # synthetic mean reversion returns
        returns = [0.01 if i % 2 == 0 else -0.005 for i in range(240)]
        benchmark = [0.003 for _ in range(240)]
        response = robustness_server.run_robustness({
            "returns": returns,
            "benchmark_returns": benchmark,
            "folds": 4,
            "walk_forward_window": 40,
            "bootstrap_iterations": 20,
        })
        self.assertIn("sharpe", response)
        self.assertIn("cross_validation", response)
        self.assertEqual(len(response["cross_validation"]), 4)
        self.assertIn("walk_forward", response)
        self.assertGreaterEqual(len(response["walk_forward"]), 1)
        self.assertEqual(response["verdict"], "pass")

    def test_run_robustness_flags_buy_and_hold(self) -> None:
        returns = [0.001] * 240
        benchmark = [0.001] * 240
        response = robustness_server.run_robustness({
            "returns": returns,
            "benchmark_returns": benchmark,
        })
        self.assertEqual(response["verdict"], "fail")

    def test_run_robustness_short_series_error(self) -> None:
        with self.assertRaises(ValueError):
            robustness_server.run_robustness({"returns": [0.01] * 30})


if __name__ == "__main__":
    unittest.main()
