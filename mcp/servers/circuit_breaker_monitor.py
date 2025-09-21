"""Circuit breaker monitoring server for MCP infrastructure."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from mcp.server import register_tool
from mcp.common.resilience import get_all_circuit_breaker_stats, reset_circuit_breaker

logger = logging.getLogger(__name__)


@register_tool(
    name="monitoring.circuit_breakers.get_status",
    schema="./schemas/tool.monitoring.circuit_breakers.get_status.schema.json",
)
def get_circuit_breaker_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get status of all circuit breakers in the system."""

    # Extract request parameters
    include_detailed_stats = params.get("include_detailed_stats", True)
    filter_state = params.get("filter_state")  # None, "open", "closed", "half_open"

    # Get all circuit breaker statistics
    all_stats = get_all_circuit_breaker_stats()

    # Apply state filter if specified
    if filter_state:
        stats = [s for s in all_stats if s["state"] == filter_state]
        logger.info(f"Filtered circuit breakers by state '{filter_state}': {len(stats)}/{len(all_stats)}")
    else:
        stats = all_stats

    # Calculate summary metrics
    total_breakers = len(stats)
    open_breakers = len([s for s in stats if s["state"] == "open"])
    half_open_breakers = len([s for s in stats if s["state"] == "half_open"])
    closed_breakers = len([s for s in stats if s["state"] == "closed"])

    # Calculate overall system health
    if total_breakers == 0:
        health_status = "no_breakers"
        health_score = 100
    else:
        health_score = (closed_breakers / total_breakers) * 100
        if open_breakers == 0 and half_open_breakers == 0:
            health_status = "healthy"
        elif open_breakers > 0:
            health_status = "degraded" if open_breakers < total_breakers / 2 else "critical"
        else:
            health_status = "recovering"

    # Find most problematic breakers
    problematic_breakers = [
        s for s in stats
        if s["state"] in ["open", "half_open"] or s["failure_rate_percent"] > 20
    ]

    # Generate current timestamp
    current_timestamp = datetime.now(timezone.utc).isoformat()

    result = {
        "timestamp": current_timestamp,
        "summary": {
            "total_circuit_breakers": total_breakers,
            "health_status": health_status,
            "health_score": round(health_score, 2),
            "states": {
                "closed": closed_breakers,
                "open": open_breakers,
                "half_open": half_open_breakers
            }
        },
        "circuit_breakers": stats if include_detailed_stats else [],
        "alerts": [
            {
                "type": "circuit_open",
                "severity": "high",
                "message": f"Circuit breaker '{cb['name']}' is OPEN",
                "breaker_name": cb["name"],
                "failure_rate": cb["failure_rate_percent"],
                "last_failure": cb["last_exception"]
            }
            for cb in stats if cb["state"] == "open"
        ] + [
            {
                "type": "high_failure_rate",
                "severity": "medium",
                "message": f"Circuit breaker '{cb['name']}' has high failure rate",
                "breaker_name": cb["name"],
                "failure_rate": cb["failure_rate_percent"],
                "last_failure": cb["last_exception"]
            }
            for cb in stats if cb["failure_rate_percent"] > 20 and cb["state"] != "open"
        ],
        "recommendations": []
    }

    # Add recommendations based on current state
    if open_breakers > 0:
        result["recommendations"].append(
            "Investigate and fix underlying issues with open circuit breakers"
        )

    if half_open_breakers > 0:
        result["recommendations"].append(
            "Monitor half-open circuit breakers for recovery"
        )

    total_calls = sum(cb["total_calls"] for cb in stats)
    total_failures = sum(cb["total_failures"] for cb in stats)

    if total_calls > 0:
        overall_failure_rate = (total_failures / total_calls) * 100
        if overall_failure_rate > 10:
            result["recommendations"].append(
                f"Overall system failure rate is {overall_failure_rate:.1f}% - consider system-wide improvements"
            )

    logger.info(f"Circuit breaker status: {health_status} ({health_score:.1f}% healthy)")

    return result


@register_tool(
    name="monitoring.circuit_breakers.reset",
    schema="./schemas/tool.monitoring.circuit_breakers.reset.schema.json",
)
def reset_circuit_breaker_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Reset a specific circuit breaker to CLOSED state."""

    breaker_name = params["breaker_name"]

    success = reset_circuit_breaker(breaker_name)

    if success:
        logger.info(f"Circuit breaker '{breaker_name}' has been manually reset to CLOSED")
        return {
            "success": True,
            "message": f"Circuit breaker '{breaker_name}' reset successfully",
            "breaker_name": breaker_name,
            "new_state": "closed"
        }
    else:
        logger.warning(f"Failed to reset circuit breaker '{breaker_name}' - not found")
        return {
            "success": False,
            "message": f"Circuit breaker '{breaker_name}' not found",
            "breaker_name": breaker_name,
            "error": "Circuit breaker not found in registry"
        }


__all__ = ["get_circuit_breaker_status", "reset_circuit_breaker_tool"]