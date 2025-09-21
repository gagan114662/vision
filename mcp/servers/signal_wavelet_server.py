"""Wavelet-based multi-scale decomposition MCP tool."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from mcp.server import register_tool

try:
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
except ImportError:
    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator


_SQRT2 = math.sqrt(2.0)


def _validate_series(series: List[float]) -> np.ndarray:
    if len(series) < 32:
        raise ValueError("At least 32 samples required for wavelet analysis")
    arr = np.asarray(series, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Series contains non-finite values")
    return arr


def _normalize_for_level(data: np.ndarray) -> np.ndarray:
    if data.size % 2 == 0:
        return data
    return np.pad(data, (0, 1), mode="edge")


def _haar_step(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data = _normalize_for_level(data)
    even = data[0::2]
    odd = data[1::2]
    approx = (even + odd) / _SQRT2
    detail = (even - odd) / _SQRT2
    return approx, detail


def _max_levels(length: int) -> int:
    if length < 2:
        return 0
    return int(math.floor(math.log2(length)))


def _haar_inverse(approx: np.ndarray, details: List[np.ndarray]) -> np.ndarray:
    current = approx
    for detail in reversed(details):
        upsampled = np.empty(detail.size * 2, dtype=float)
        upsampled[0::2] = (current + detail) / _SQRT2
        upsampled[1::2] = (current - detail) / _SQRT2
        current = upsampled
    return current


def _decompose(series: np.ndarray, levels: int) -> Tuple[List[Dict[str, Any]], float]:
    total_energy = float(np.sum(series ** 2)) or 1.0
    current = series
    detail_stack: List[np.ndarray] = []
    components: List[Dict[str, Any]] = []

    for level in range(1, levels + 1):
        if current.size < 2:
            break
        approx, detail = _haar_step(current)
        energy_ratio = float(np.sum(detail ** 2) / total_energy)
        components.append({
            "level": level,
            "approximation": approx.tolist(),
            "detail": detail.tolist(),
            "energy_ratio": energy_ratio,
        })
        detail_stack.append(detail)
        current = approx

    reconstructed = _haar_inverse(current, detail_stack)
    reconstructed = reconstructed[: series.size]
    rmse = float(np.sqrt(np.mean((reconstructed - series) ** 2)))
    return components, rmse


@register_tool(
    name="signal.wavelet.multiscale_decomposition",
    schema="./schemas/tool.signal.wavelet.multiscale_decomposition.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def multiscale_decomposition(params: Dict[str, Any]) -> Dict[str, Any]:
    series = _validate_series(params["series"])
    wavelet = params.get("wavelet", "haar").lower()
    if wavelet != "haar":
        raise ValueError("Only the 'haar' wavelet is supported in this release")

    max_levels = _max_levels(series.size)
    requested_levels = params.get("levels")
    if requested_levels is None:
        levels = min(5, max_levels)
    else:
        levels = int(requested_levels)
        if levels < 1:
            raise ValueError("levels must be positive")
        levels = min(levels, max_levels)
    components, rmse = _decompose(series, levels)
    return {
        "levels": components,
        "reconstruction_rmse": rmse,
        "wavelet": wavelet,
    }


__all__ = ["multiscale_decomposition"]
