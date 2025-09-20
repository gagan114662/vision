"""MCP server exposing robustness evaluation for strategy returns."""
from __future__ import annotations

from typing import Any, Dict

from mcp.server import register_tool

from analysis import robustness


@register_tool(
    name="strategy.validation.run_robustness",
    schema="./schemas/tool.strategy.validation.run_robustness.schema.json",
)
def run_robustness(params: Dict[str, Any]) -> Dict[str, Any]:
    returns = params["returns"]
    folds = params.get("folds", 3)
    walk_forward_window = params.get("walk_forward_window", 60)
    bootstrap_iterations = params.get("bootstrap_iterations", 100)
    benchmark_returns = params.get("benchmark_returns")

    result = robustness.evaluate_returns(
        returns,
        folds=folds,
        walk_forward_window=walk_forward_window,
        bootstrap_iterations=bootstrap_iterations,
        benchmark_returns=benchmark_returns,
    )
    return robustness.to_dict(result)


__all__ = ["run_robustness"]
