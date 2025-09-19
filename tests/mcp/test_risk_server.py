from __future__ import annotations

import unittest

import mcp.servers.risk_server as risk_server


class RiskServerTests(unittest.TestCase):
    def test_evaluate_portfolio_no_breaches(self) -> None:
        params = {
            "positions": [
                {"symbol": "AAPL", "quantity": 10, "price": 150, "volatility": 0.2},
                {"symbol": "MSFT", "quantity": 8, "price": 300, "volatility": 0.18},
            ],
            "limits": {"max_drawdown": 0.6, "var_limit": 2000, "exposure_limit": 4000},
            "confidence_level": 0.95,
        }
        result = risk_server.evaluate_portfolio(params)
        self.assertIn("var", result)
        self.assertLessEqual(result["var"], 1000)
        self.assertFalse(result["breaches"])

    def test_evaluate_portfolio_with_breaches(self) -> None:
        params = {
            "positions": [
                {"symbol": "TSLA", "quantity": 100, "price": 250, "volatility": 0.4},
            ],
            "limits": {"max_drawdown": 0.1, "var_limit": 1000, "exposure_limit": 15000},
            "confidence_level": 0.99,
        }
        result = risk_server.evaluate_portfolio(params)
        self.assertTrue(result["breaches"])
        self.assertTrue(any("VaR" in breach for breach in result["breaches"]))
        self.assertTrue(result["recommendations"])


if __name__ == "__main__":
    unittest.main()
