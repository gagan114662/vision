"""Feature engineering MCP server implementing basic factor computations."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List

from mcp.server import register_tool

try:
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
except ImportError:
    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator


@dataclass
class PricePoint:
    timestamp: datetime
    price: float

    @classmethod
    def from_dict(cls, entry: Dict[str, Any]) -> "PricePoint":
        return cls(
            timestamp=datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")),
            price=float(entry["price"]),
        )


class FactorCalculator:
    def __init__(self, data: Iterable[Dict[str, Any]]):
        self.points: List[PricePoint] = [PricePoint.from_dict(item) for item in data]
        if len(self.points) < 2:
            raise ValueError("At least two data points required for factor computation")

    def momentum(self, window: int) -> List[Dict[str, Any]]:
        window = max(window, 1)
        results: List[Dict[str, Any]] = []
        prices = [p.price for p in self.points]
        timestamps = [p.timestamp for p in self.points]
        for idx in range(window, len(prices)):
            past_price = prices[idx - window]
            current_price = prices[idx]
            if past_price == 0:
                continue
            momentum_value = (current_price / past_price) - 1.0
            results.append(
                {
                    "timestamp": timestamps[idx].isoformat(),
                    "value": momentum_value,
                }
            )
        return results

    def volatility(self, window: int) -> List[Dict[str, Any]]:
        if window < 2:
            window = 2
        results: List[Dict[str, Any]] = []
        price_queue: deque[float] = deque(maxlen=window)
        timestamps = [p.timestamp for p in self.points]
        for idx, point in enumerate(self.points):
            price_queue.append(point.price)
            if len(price_queue) < window:
                continue
            mean_price = sum(price_queue) / window
            variance = sum((price - mean_price) ** 2 for price in price_queue) / (window - 1)
            volatility_value = variance ** 0.5
            results.append(
                {
                    "timestamp": timestamps[idx].isoformat(),
                    "value": volatility_value,
                }
            )
        return results

    def mean_reversion(self, window: int) -> List[Dict[str, Any]]:
        if window < 2:
            window = 2
        results: List[Dict[str, Any]] = []
        price_queue: deque[float] = deque(maxlen=window)
        timestamps = [p.timestamp for p in self.points]
        for idx, point in enumerate(self.points):
            price_queue.append(point.price)
            if len(price_queue) < window:
                continue
            mean_price = sum(price_queue) / window
            if mean_price == 0:
                continue
            deviation = (point.price - mean_price) / mean_price
            results.append(
                {
                    "timestamp": timestamps[idx].isoformat(),
                    "value": -deviation,
                }
            )
        return results


def _default_window(factor_name: str) -> int:
    if factor_name == "momentum":
        return 20
    if factor_name == "volatility":
        return 20
    if factor_name == "mean_reversion":
        return 10
    raise ValueError(f"Unsupported factor {factor_name}")


@register_tool(
    name="feature-engineering.compute_factor",
    schema="./schemas/tool.feature-engineering.compute_factor.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def compute_factor(params: Dict[str, Any]) -> Dict[str, Any]:
    factor_name = params["factor_name"]
    data = params["data"]
    window = int(params.get("window") or _default_window(factor_name))

    calculator = FactorCalculator(data)
    if factor_name == "momentum":
        results = calculator.momentum(window)
    elif factor_name == "volatility":
        results = calculator.volatility(window)
    elif factor_name == "mean_reversion":
        results = calculator.mean_reversion(window)
    else:  # pragma: no cover - schema restricts factor_name
        raise ValueError(f"Unsupported factor {factor_name}")

    return {
        "factor_name": factor_name,
        "window": window,
        "results": results,
        "provenance_ids": [entry.get("provenance_id", "") for entry in data if entry.get("provenance_id")],
    }


__all__ = ["compute_factor", "FactorCalculator", "PricePoint"]
