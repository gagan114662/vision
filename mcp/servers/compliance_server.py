"""Compliance MCP server that synthesizes control summaries."""
from __future__ import annotations

from typing import Any, Dict, List

from mcp.server import register_tool
from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig


_STATUS_SEVERITY = {"PASS": 0, "WAIVED": 1, "CONCERNS": 2, "FAIL": 3}


def _determine_status(evidence: List[Dict[str, Any]], outstanding: List[str]) -> str:
    severity = 0
    for item in evidence:
        status = item.get("status", "PASS")
        severity = max(severity, _STATUS_SEVERITY.get(status, 3))
    if outstanding:
        severity = max(severity, _STATUS_SEVERITY["CONCERNS"])
    for label, threshold in _STATUS_SEVERITY.items():
        if severity == threshold:
            return label
    return "CONCERNS"


@register_tool(
    name="compliance.generate_summary",
    schema="./schemas/tool.compliance.generate_summary.schema.json",
)
@circuit_breaker(
    name="compliance_server.generate_summary",
    config=CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=30.0,
        expected_exception=Exception
    )
)
def generate_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    strategy_id = params["strategy_id"]
    controls = params["controls"]
    evidence = params["evidence"]
    outstanding = params.get("outstanding_risks", [])

    status = _determine_status(evidence, outstanding)
    next_actions: List[str] = []
    if status in {"CONCERNS", "FAIL"}:
        for item in evidence:
            if item["status"] in {"FAIL", "WAIVED"}:
                next_actions.append(f"Review control {item['control']} and document remediation")
        for risk in outstanding:
            next_actions.append(f"Mitigate outstanding risk: {risk}")
    else:
        next_actions.append("Maintain monitoring cadence and archive evidence")

    report = {
        "controls": controls,
        "evidence": evidence,
        "outstanding_risks": outstanding,
        "next_actions": next_actions,
    }

    return {
        "strategy_id": strategy_id,
        "status": status,
        "report": report,
    }


__all__ = ["generate_summary"]
