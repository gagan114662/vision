"""Fourier-based cycle detection MCP tool."""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List

from mcp.server import register_tool


def _validate(series: List[float]) -> np.ndarray:
    if len(series) < 16:
        raise ValueError("At least 16 samples required for Fourier analysis")
    arr = np.asarray(series, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Series contains non-finite values")
    return arr


def _detect_cycles(series: np.ndarray, sample_rate: float, top_n: int) -> List[Dict[str, float]]:
    series = series - np.mean(series)
    fft = np.fft.rfft(series)
    freqs = np.fft.rfftfreq(len(series), d=1.0 / sample_rate)

    power = np.abs(fft) ** 2
    valid = freqs > 0
    freqs = freqs[valid]
    power = power[valid]

    amplitudes = (2.0 / len(series)) * np.abs(fft[valid])
    indices = np.argsort(power)[::-1][:top_n]

    cycles = []
    for idx in indices:
        freq = float(freqs[idx])
        if freq == 0:
            continue
        period = float(1.0 / freq)
        cycles.append({
            "frequency": freq,
            "period": period,
            "amplitude": float(amplitudes[idx]),
            "power": float(power[idx]),
        })
    return cycles


@register_tool(
    name="signal.fourier.detect_cycles",
    schema="./schemas/tool.signal.fourier.detect_cycles.schema.json",
)
def detect_cycles(params: Dict[str, Any]) -> Dict[str, Any]:
    series = _validate(params["series"])
    sample_rate = float(params.get("sample_rate", 1.0))
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    top_n = int(params.get("top_n", 3))
    cycles = _detect_cycles(series, sample_rate, top_n)
    return {"cycles": cycles}


__all__ = ["detect_cycles"]
