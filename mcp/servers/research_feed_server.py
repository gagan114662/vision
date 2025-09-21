"""Research feed MCP server that reads curated innovation insights."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from mcp.server import register_tool

try:
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
except ImportError:
    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator

FEED_PATH = Path("data/processed/research_feed.json")


def _load_feed() -> Dict[str, Any]:
    if not FEED_PATH.exists():
        return {"generated_at": None, "insights": []}
    return json.loads(FEED_PATH.read_text(encoding="utf-8"))


def _filter_insights(insights: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    status_filter = params.get("status")
    category = params.get("category")
    filtered = insights
    if status_filter:
        filtered = [ins for ins in filtered if ins.get("status") in status_filter]
    if category:
        filtered = [ins for ins in filtered if ins.get("category") == category]
    return filtered


@register_tool(
    name="research.feed.list_insights",
    schema="./schemas/tool.research.feed.list.schema.json",
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def list_insights(params: Dict[str, Any]) -> Dict[str, Any]:
    feed = _load_feed()
    insights = feed.get("insights", [])
    filtered = _filter_insights(insights, params)
    return {"insights": filtered}


__all__ = ["list_insights"]
