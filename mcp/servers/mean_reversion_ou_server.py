"""Ornstein-Uhlenbeck mean reversion parameter estimation."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from mcp.server import register_tool

try:
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
except ImportError:
    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator


def _validate_prices(prices: List[float]) -> np.ndarray:
    if len(prices) < 20:
        raise ValueError("At least 20 price observations required for OU estimation")
    arr = np.asarray(prices, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Prices contain non-finite values")
    return arr


def _estimate_ou_parameters(prices: np.ndarray, dt: float) -> Dict[str, float]:
    # Discrete formulation: y = alpha + beta x + epsilon
    x = prices[:-1]
    y = prices[1:]
    n = len(x)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)

    denominator = n * sum_xx - sum_x ** 2
    if denominator == 0:
        raise ValueError("Degenerate price series, cannot estimate OU parameters")

    beta = (n * sum_xy - sum_x * sum_y) / denominator
    alpha = (sum_y - beta * sum_x) / n

    if beta <= 0 or beta >= 1:
        raise ValueError("Estimated beta outside (0,1); OU model unsuitable")

    kappa = -np.log(beta) / dt
    theta = alpha / (1 - beta)

    residuals = y - alpha - beta * x
    sigma_eps_sq = np.sum(residuals ** 2) / (n - 1)
    sigma_sq = sigma_eps_sq * 2 * kappa / (1 - beta ** 2)
    if sigma_sq <= 0:
        raise ValueError("Computed sigma is non-positive; check input series")
    sigma = float(np.sqrt(sigma_sq))

    profit_score = float(kappa * np.sqrt(252) / sigma) if sigma > 0 else float("nan")
    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": sigma,
        "profit_score": profit_score,
    }


@register_tool(
    name="strategy.meanreversion.estimate_parameters",
    schema="./schemas/tool.strategy.meanreversion.estimate_parameters.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        expected_exception=Exception
    )
)
def estimate_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    prices = _validate_prices(params["prices"])
    dt = float(params.get("dt", 1.0))
    if dt <= 0:
        raise ValueError("dt must be positive")
    return _estimate_ou_parameters(prices, dt)


__all__ = ["estimate_parameters"]
