"""Adaptive noise reduction via exponential smoothing with volatility-based gain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from mcp.server import register_tool


@dataclass
class AdaptiveFilterParams:
    series: List[float]
    window: int = 10
    min_alpha: float = 0.1
    max_alpha: float = 0.6

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "AdaptiveFilterParams":
        return cls(
            series=payload["series"],
            window=int(payload.get("window", 10)),
            min_alpha=float(payload.get("min_alpha", 0.1)),
            max_alpha=float(payload.get("max_alpha", 0.6)),
        )


def _validate(params: AdaptiveFilterParams) -> AdaptiveFilterParams:
    if len(params.series) < 16:
        raise ValueError("At least 16 samples required for adaptive filtering")
    if params.window < 2:
        raise ValueError("window must be at least 2")
    if params.max_alpha <= params.min_alpha:
        raise ValueError("max_alpha must be greater than min_alpha")
    if not (0 < params.min_alpha < 1) or not (0 < params.max_alpha < 1):
        raise ValueError("alpha bounds must be between 0 and 1")
    return params


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    ma = np.convolve(padded, kernel, mode="valid")
    return ma[: values.size]


def _compute_alpha(volatility: np.ndarray, min_alpha: float, max_alpha: float) -> np.ndarray:
    vol_min = float(volatility.min())
    vol_max = float(volatility.max())
    if vol_max - vol_min < 1e-9:
        return np.full_like(volatility, max_alpha)
    normalized = (volatility - vol_min) / (vol_max - vol_min)
    gain = max_alpha - normalized * (max_alpha - min_alpha)
    return np.clip(gain, min_alpha, max_alpha)


def _adaptive_smooth(series: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    smoothed = np.empty_like(series)
    smoothed[0] = series[0]
    for idx in range(1, series.size):
        smoothed[idx] = alpha[idx] * series[idx] + (1.0 - alpha[idx]) * smoothed[idx - 1]
    return smoothed


@register_tool(
    name="signal.filter.adaptive_noise_reduction",
    schema="./schemas/tool.signal.filter.adaptive_noise_reduction.schema.json",
)
def adaptive_noise_reduction(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = _validate(AdaptiveFilterParams.from_payload(payload))

    series = np.asarray(params.series, dtype=float)
    deltas = np.abs(np.diff(series, prepend=series[0]))
    volatility = _moving_average(deltas, params.window)
    alpha = _compute_alpha(volatility, params.min_alpha, params.max_alpha)
    smoothed = _adaptive_smooth(series, alpha)

    residual = series - smoothed
    original_noise = float(np.std(deltas))
    residual_noise = float(np.std(np.diff(smoothed, prepend=smoothed[0])))
    noise_reduction = None
    if original_noise > 1e-9:
        noise_reduction = float(1.0 - (residual_noise / original_noise))

    return {
        "smoothed": smoothed.tolist(),
        "effective_alpha": alpha.tolist(),
        "residual": residual.tolist(),
        "noise_reduction": noise_reduction,
    }


__all__ = ["adaptive_noise_reduction"]
