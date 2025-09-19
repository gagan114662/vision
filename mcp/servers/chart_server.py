"""Visualization MCP server for rendering price series charts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from mcp.server import register_tool
except ImportError:  # pragma: no cover
    def register_tool(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func

        return decorator

OUTPUT_DIR = Path("visualizations")


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _render_chart(prices: List[Dict[str, Any]], signals: List[Dict[str, Any]], title: str, output_path: Path) -> List[str]:
    warnings: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("matplotlib required for chart rendering") from exc

    timestamps = [datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")) for p in prices]
    values = [p["price"] for p in prices]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timestamps, values, label="Price", linewidth=1.2)
    for signal in signals:
        ts = datetime.fromisoformat(signal["timestamp"].replace("Z", "+00:00"))
        ax.axvline(ts, color="orange", linestyle="--", linewidth=0.8)
        label = signal.get("label", "signal")
        value = signal.get("value")
        if value is not None:
            ax.text(ts, value, label, fontsize=8, color="orange")
        else:
            ax.text(ts, ax.get_ylim()[1], label, fontsize=8, color="orange")
    ax.set_title(title or "Price Series")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return warnings


@register_tool(
    name="visualization.render_price_series",
    schema="./schemas/tool.visualization.render_price_series.schema.json",
)
def render_price_series(params: Dict[str, Any]) -> Dict[str, Any]:
    prices = params["prices"]
    signals = params.get("signals", [])
    title = params.get("title", "Price Series")

    if len(prices) < 2:
        raise ValueError("At least two price points required to render chart")

    _ensure_output_dir()
    output_path = OUTPUT_DIR / f"chart_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.png"
    warnings = _render_chart(prices, signals, title, output_path)
    return {
        "image_path": str(output_path),
        "warnings": warnings,
    }


__all__ = ["render_price_series"]
