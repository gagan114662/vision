from __future__ import annotations

import math
import unittest

import numpy as np

import mcp.servers.signal_filter_server as filter_server


class AdaptiveNoiseReductionTests(unittest.TestCase):
    def test_requires_minimum_length(self) -> None:
        with self.assertRaises(ValueError):
            filter_server.adaptive_noise_reduction({"series": [1.0, 2.0, 3.0]})

    def test_noise_reduction_improves_signal(self) -> None:
        rng = np.random.default_rng(seed=42)
        x = np.linspace(0, 4 * math.pi, 256)
        clean = np.sin(x)
        noise = rng.normal(scale=0.3, size=x.size)
        series = (clean + noise).tolist()

        result = filter_server.adaptive_noise_reduction({
            "series": series,
            "window": 12,
            "min_alpha": 0.05,
            "max_alpha": 0.5,
        })

        smoothed = np.asarray(result["smoothed"])
        effective_alpha = np.asarray(result["effective_alpha"])

        self.assertEqual(smoothed.shape, (256,))
        self.assertEqual(effective_alpha.shape, (256,))
        self.assertTrue(np.all(effective_alpha >= 0.05 - 1e-9))
        self.assertTrue(np.all(effective_alpha <= 0.5 + 1e-9))

        original_noise = np.std(np.diff(series))
        residual_noise = np.std(np.diff(smoothed))
        self.assertLess(residual_noise, original_noise)
        self.assertIsNotNone(result["noise_reduction"])
        self.assertGreater(result["noise_reduction"], 0.0)

    def test_custom_parameters_validation(self) -> None:
        series = list(range(32))
        with self.assertRaises(ValueError):
            filter_server.adaptive_noise_reduction({
                "series": series,
                "window": 1,
            })

        with self.assertRaises(ValueError):
            filter_server.adaptive_noise_reduction({
                "series": series,
                "min_alpha": 0.8,
                "max_alpha": 0.7,
            })


if __name__ == "__main__":
    unittest.main()
