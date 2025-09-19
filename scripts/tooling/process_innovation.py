#!/usr/bin/env python3
"""Process pending innovation insight: fetch returns from sample run and evaluate robustness + chart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp.servers import research_feed_server, robustness_server, chart_server
from mcp.wrappers import lean_backtest


def load_returns(statistics_json: str) -> list[float]:
    data = json.loads(Path(statistics_json).read_text(encoding="utf-8"))
    equity = data.get("equityPoints", [])
    if not equity:
        raise ValueError("statistics json lacks equityPoints")
    returns = []
    for idx in range(1, len(equity)):
        prev = equity[idx - 1]["value"]
        curr = equity[idx]["value"]
        if prev != 0:
            returns.append((curr - prev) / prev)
    return returns


def main() -> None:
    parser = argparse.ArgumentParser(description="Process first pending innovation insight")
    parser.add_argument("--project", default=None)
    parser.add_argument("--algorithm", default="lean/algorithms/monthly_universe_alpha.py")
    parser.add_argument("--config", default="lean/config/monthly_universe_alpha.json")
    args = parser.parse_args()

    insights = research_feed_server.list_insights({"status": ["pending-validation", "queued"]})["insights"]
    if not insights:
        print("No pending insights")
        return
    insight = insights[0]
    print(f"Processing insight {insight['id']} from {insight['source_id']}")

    backtest_result = lean_backtest.run_backtest(
        project=args.project,
        algorithm_path=args.algorithm,
        config_path=args.config,
    )
    stats_path = backtest_result["statistics_file"]
    returns = load_returns(stats_path)

    robustness = robustness_server.run_robustness({
        "returns": returns,
        "folds": 4,
        "walk_forward_window": 60,
        "bootstrap_iterations": 50,
    })
    print("Robustness verdict:", robustness["verdict"])

    prices = []
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    for point in stats.get("equityPoints", [])[-120:]:
        prices.append({
            "timestamp": point["date"],
            "price": point["value"],
        })
    chart = chart_server.render_price_series({
        "prices": prices,
        "title": f"{insight['id']} equity",
    })
    print("Chart saved to:", chart["image_path"])


if __name__ == "__main__":
    main()
