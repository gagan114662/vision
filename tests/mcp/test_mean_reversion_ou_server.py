from __future__ import annotations

import unittest

import numpy as np

import mcp.servers.mean_reversion_ou_server as ou_server


class MeanReversionOUTests(unittest.TestCase):
    def test_invalid_length(self) -> None:
        with self.assertRaises(ValueError):
            ou_server.estimate_parameters({"prices": [1, 2, 3]})

    def test_estimate_parameters(self) -> None:
        prices = list(100 + np.sin(np.linspace(0, 4 * np.pi, 60)))
        result = ou_server.estimate_parameters({"prices": prices})
        self.assertIn("kappa", result)
        self.assertIn("theta", result)
        self.assertIn("sigma", result)
        self.assertTrue(np.isfinite(result["profit_score"]))

    def test_negative_sigma_error(self) -> None:
        prices = [1.0] * 40
        with self.assertRaises(ValueError):
            ou_server.estimate_parameters({"prices": prices})


if __name__ == "__main__":
    unittest.main()
