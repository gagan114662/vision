from __future__ import annotations

import math
import unittest

import numpy as np

import mcp.servers.signal_wavelet_server as wavelet_server


class SignalWaveletTests(unittest.TestCase):
    def test_requires_minimum_length(self) -> None:
        with self.assertRaises(ValueError):
            wavelet_server.multiscale_decomposition({"series": [1.0, 2.0]})

    def test_default_levels_and_energy(self) -> None:
        n = 128
        x = np.linspace(0, 4 * math.pi, n)
        high_freq = np.sin(6 * x)
        low_freq = 0.5 * np.sin(0.5 * x)
        series = (high_freq + low_freq).tolist()
        result = wavelet_server.multiscale_decomposition({"series": series})

        levels = result["levels"]
        self.assertGreaterEqual(len(levels), 3)
        self.assertAlmostEqual(result["reconstruction_rmse"], 0.0, places=6)

        # Highest frequency detail (level 1) should capture more energy
        energy_values = [level["energy_ratio"] for level in levels]
        for energy in energy_values:
            self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(sum(energy_values), 1.0 + 1e-6)

    def test_level_override(self) -> None:
        series = list(range(64))
        result = wavelet_server.multiscale_decomposition({
            "series": series,
            "levels": 2
        })
        self.assertEqual(len(result["levels"]), 2)


if __name__ == "__main__":
    unittest.main()
