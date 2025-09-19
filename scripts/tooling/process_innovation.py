#!/usr/bin/env python3
"""Process pending innovation insight: fetch returns from sample run and evaluate robustness + chart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datetime import datetime, timezone

from mcp.servers import research_feed_server, robustness_server, chart_server

FEED_PATH = Path("data/processed/research_feed.json")
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

    _update_insight(insight["id"], status="in-validation")

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

    status = "adopted" if robustness["verdict"] == "pass" else "queued"
    if robustness["verdict"] == "fail":
        status = "rejected"
    _update_insight(
        insight["id"],
        status=status,
        robustness=robustness,
        chart_path=chart["image_path"],
        backtest_stats=stats_path,
    )


def _update_insight(
    insight_id: str,
    *,
    status: str | None = None,
    robustness: dict | None = None,
    chart_path: str | None = None,
    backtest_stats: str | None = None,
) -> None:
    if not FEED_PATH.exists():
        return
    feed = json.loads(FEED_PATH.read_text(encoding="utf-8"))
    for insight in feed.get("insights", []):
        if insight.get("id") == insight_id:
            if status is not None:
                insight["status"] = status
            if robustness is not None:
                insight["robustness"] = robustness
            if chart_path is not None:
                insight["chart_path"] = chart_path
            if backtest_stats is not None:
                insight["backtest_statistics"] = backtest_stats
            break
    feed["generated_at"] = (
        datetime.now(timezone.utc).isoformat()
        if backtest_stats
        else feed.get("generated_at")
    )
    FEED_PATH.write_text(json.dumps(feed, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
