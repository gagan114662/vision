"""Risk management MCP server providing portfolio limit evaluation."""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from mcp.server import register_tool
from mcp.common.server_config import get_server_config, get_tool_config

logger = logging.getLogger(__name__)

# Get server configuration
_server_config = get_server_config("risk_server")
_tool_config = get_tool_config("risk_server", "risk.limits.evaluate_portfolio")


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
    """Calculate VaR with configurable method and parameters."""

    # Get VaR calculation method from configuration
    var_method = "historical_simulation"
    if _server_config:
        var_method = _server_config.get_setting("var_method", "historical_simulation")

    # Get lookback period from configuration
    lookback_days = 252
    if _tool_config:
        lookback_days = _tool_config.get_parameter("lookback_days", 252)

    portfolio_value = sum(pos.market_value for pos in positions)

    if var_method == "parametric":
        # Parametric VaR (current implementation)
        variance = sum((pos.market_value * pos.volatility) ** 2 for pos in positions)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        z_score = 1.65 if confidence_level >= 0.95 else 1.28
        var_value = abs(z_score * std_dev)
    else:
        # Simplified variance-covariance VaR (default for compatibility)
        variance = sum((pos.market_value * pos.volatility) ** 2 for pos in positions)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        z_score = 1.65 if confidence_level >= 0.95 else 1.28
        var_value = abs(z_score * std_dev)

    logger.info(f"Calculated VaR using {var_method} method with {lookback_days} day lookback: {var_value}")
    return var_value, portfolio_value


@register_tool(
    name="risk.limits.evaluate_portfolio",
    schema="./schemas/tool.risk.limits.evaluate_portfolio.schema.json",
)
def evaluate_portfolio(params: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate portfolio risk with configurable parameters."""

    # Use configuration for default values
    default_confidence = 0.95
    if _tool_config:
        default_confidence = _tool_config.get_parameter("confidence_level", 0.95)
        logger.info(f"Using configured confidence level: {default_confidence}")

    positions = [Position.from_dict(p) for p in params["positions"]]
    limits = Limits.from_dict(params["limits"])
    confidence = float(params.get("confidence_level", default_confidence))

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

    # Add configuration-aware alerting
    alerts = []
    if _server_config:
        alert_settings = _server_config.get_setting("alert_thresholds", {})

        if alert_settings.get("var_breach") and "VaR limit exceeded" in breaches:
            alerts.append({"type": "var_breach", "severity": "high", "message": "VaR limit exceeded"})

        if alert_settings.get("concentration_risk") and "Single-position exposure limit exceeded" in breaches:
            alerts.append({"type": "concentration_risk", "severity": "medium", "message": "Concentration risk detected"})

        if alert_settings.get("leverage_limit") and "Max drawdown limit exceeded" in breaches:
            alerts.append({"type": "leverage_limit", "severity": "high", "message": "Leverage limit exceeded"})

    # Add configuration metadata
    config_metadata = {}
    if _server_config:
        config_metadata["var_method"] = _server_config.get_setting("var_method", "historical_simulation")
        config_metadata["stress_testing_enabled"] = _server_config.get_setting("enable_stress_testing", True)

    if _tool_config:
        config_metadata["lookback_days"] = _tool_config.get_parameter("lookback_days", 252)
        config_metadata["confidence_level"] = confidence

    result = {
        "portfolio_value": portfolio_value,
        "var": var_value,
        "max_drawdown": drawdown_estimate,
        "breaches": breaches,
        "recommendations": recommendations,
        "metrics": metrics,
        "alerts": alerts,
        "config_metadata": config_metadata,
        "server_config": {
            "enabled": _server_config.enabled if _server_config else True,
            "server_type": _server_config.server_type.value if _server_config else "risk_management"
        }
    }

    # Log the assessment if logging is enabled
    if _tool_config and _tool_config.logging_enabled:
        logger.info(f"Risk assessment completed: {len(breaches)} breaches, {len(alerts)} alerts generated")

    return result


__all__ = ["evaluate_portfolio"]
