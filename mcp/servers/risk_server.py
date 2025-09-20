"""Risk management MCP server providing portfolio limit evaluation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from mcp.server import register_tool


@dataclass
class Position:
    symbol: str
    quantity: float
    price: float
    volatility: float

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Position":
        return cls(
            symbol=payload["symbol"],
            quantity=float(payload["quantity"]),
            price=float(payload["price"]),
            volatility=float(payload["volatility"]),
        )

    @property
    def market_value(self) -> float:
        return self.quantity * self.price


@dataclass
class Limits:
    max_drawdown: float
    var_limit: float
    exposure_limit: float

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Limits":
        return cls(
            max_drawdown=float(payload["max_drawdown"]),
            var_limit=float(payload["var_limit"]),
            exposure_limit=float(payload["exposure_limit"]),
        )


def _calculate_var(positions: List[Position], confidence_level: float) -> float:
    # Simplified variance-covariance VaR assuming independence.
    portfolio_value = sum(pos.market_value for pos in positions)
    variance = sum((pos.market_value * pos.volatility) ** 2 for pos in positions)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    z_score = 1.65 if confidence_level >= 0.95 else 1.28
    return abs(z_score * std_dev), portfolio_value


@register_tool(
    name="risk.limits.evaluate_portfolio",
    schema="./schemas/tool.risk.limits.evaluate_portfolio.schema.json",
)
def evaluate_portfolio(params: Dict[str, Any]) -> Dict[str, Any]:
    positions = [Position.from_dict(p) for p in params["positions"]]
    limits = Limits.from_dict(params["limits"])
    confidence = float(params.get("confidence_level", 0.95))

    var_value, portfolio_value = _calculate_var(positions, confidence)
    max_position_exposure = max((abs(p.market_value) for p in positions), default=0.0)
    drawdown_estimate = var_value / portfolio_value if portfolio_value else 0.0

    breaches: List[str] = []
    if var_value > limits.var_limit:
        breaches.append("VaR limit exceeded")
    if drawdown_estimate > limits.max_drawdown:
        breaches.append("Max drawdown limit exceeded")
    if max_position_exposure > limits.exposure_limit:
        breaches.append("Single-position exposure limit exceeded")

    recommendations: List[str] = []
    if "VaR limit exceeded" in breaches:
        recommendations.append("Reduce leverage or hedge with offsetting positions")
    if "Max drawdown limit exceeded" in breaches:
        recommendations.append("Tighten stop-loss levels or decrease size")
    if "Single-position exposure limit exceeded" in breaches:
        recommendations.append("Trim large positions or diversify holdings")

    metrics = {
        "confidence_level": confidence,
        "portfolio_value": portfolio_value,
        "max_position_exposure": max_position_exposure,
    }

    return {
        "portfolio_value": portfolio_value,
        "var": var_value,
        "max_drawdown": drawdown_estimate,
        "breaches": breaches,
        "recommendations": recommendations,
        "metrics": metrics,
    }


__all__ = ["evaluate_portfolio"]
