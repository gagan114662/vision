from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import numpy as np

import mcp.servers.regime_hmm_server as regime_server


class DummyHMM:
    def __init__(self, n_components, covariance_type, n_iter):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.transmat_ = np.eye(n_components)
        self.means_ = np.zeros((n_components, 3))

    def fit(self, features):
        self._features = features

    def predict(self, features):
        return np.zeros(len(features), dtype=int)

    def score(self, features):
        return -123.45


class RegimeHMMSuite(unittest.TestCase):
    def test_missing_dependency_raises(self) -> None:
        with patch.object(regime_server, "_HMM_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                regime_server.detect_regimes({"prices": list(range(20))})

    def test_detect_regimes(self) -> None:
        prices = [100 + i * 0.1 for i in range(30)]
        volumes = [1000 + 10 * i for i in range(30)]

        with patch.object(regime_server, "_HMM_AVAILABLE", True):
            with patch.object(regime_server, "GaussianHMM", DummyHMM):
                result = regime_server.detect_regimes({
                    "prices": prices,
                    "volumes": volumes,
                    "n_regimes": 2,
                })

        self.assertEqual(len(result["states"]), len(prices) - 1)  # returns length
        self.assertEqual(len(result["transition_matrix"]), 2)
        self.assertAlmostEqual(result["log_likelihood"], -123.45)


if __name__ == "__main__":
    unittest.main()
