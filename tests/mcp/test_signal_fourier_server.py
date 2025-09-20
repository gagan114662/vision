from __future__ import annotations

import math
import unittest

import numpy as np

import mcp.servers.signal_fourier_server as fourier_server


class SignalFourierTests(unittest.TestCase):
    def test_invalid_length(self) -> None:
        with self.assertRaises(ValueError):
            fourier_server.detect_cycles({"series": [1, 2, 3]})

    def test_detect_cycles_sine_wave(self) -> None:
        # Generate sine wave with known frequency 0.05 cycles/sample
        n = np.arange(256)
        freq = 0.05
        series = np.sin(2 * np.pi * freq * n)
        result = fourier_server.detect_cycles({
            "series": series.tolist(),
            "sample_rate": 1.0,
            "top_n": 2
        })
        self.assertTrue(result["cycles"])
        dominant = result["cycles"][0]
        self.assertTrue(abs(dominant["frequency"] - freq) < 0.01)

    def test_invalid_sample_rate(self) -> None:
        with self.assertRaises(ValueError):
            fourier_server.detect_cycles({
                "series": list(range(32)),
                "sample_rate": 0
            })


if __name__ == "__main__":
    unittest.main()
