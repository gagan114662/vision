"""Placeholder QuantConnect MCP client.

Once the official QuantConnect MCP server is vendored, these stubs can be
replaced with real HTTP/WebSocket calls.
"""
from __future__ import annotations

from typing import Any, Dict

try:
    from mcp.server import register_tool
except ImportError:  # pragma: no cover
    def register_tool(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func

        return decorator


def _raise_stub(tool: str) -> None:
    raise RuntimeError(
        f"QuantConnect MCP integration not yet available. "
        f"Attempted to call {tool}. Vendor the official QC MCP server to enable this tool."
    )


@register_tool(
    name="quantconnect.project.sync",
    schema="./schemas/tool.quantconnect.project.sync.schema.json",
)
def project_sync(params: Dict[str, Any]) -> Dict[str, Any]:
    _raise_stub("quantconnect.project.sync")


@register_tool(
    name="quantconnect.backtest.run",
    schema="./schemas/tool.quantconnect.backtest.run.schema.json",
)
def backtest_run(params: Dict[str, Any]) -> Dict[str, Any]:
    _raise_stub("quantconnect.backtest.run")


@register_tool(
    name="quantconnect.backtest.status",
    schema="./schemas/tool.quantconnect.backtest.status.schema.json",
)
def backtest_status(params: Dict[str, Any]) -> Dict[str, Any]:
    _raise_stub("quantconnect.backtest.status")


__all__ = ["project_sync", "backtest_run", "backtest_status"]
