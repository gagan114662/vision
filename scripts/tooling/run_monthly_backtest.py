#!/usr/bin/env python3
"""Utility script to run the default monthly universe Lean backtest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp.wrappers.lean_backtest import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the monthly universe Lean backtest")
    parser.add_argument("--project", help="Override Lean project name", default=None)
    parser.add_argument(
        "--algorithm-path",
        help="Path to algorithm file",
        default=str(Path("lean/algorithms/monthly_universe_alpha.py")),
    )
    parser.add_argument(
        "--config-path",
        help="Path to Lean configuration JSON",
        default=str(Path("lean/config/monthly_universe_alpha.json")),
    )
    parser.add_argument(
        "--docker-image",
        help="Docker image to use",
        default="quantconnect/lean:latest",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Additional key=value parameters passed to Lean",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = {}
    for item in args.param:
        if "=" not in item:
            raise ValueError(f"Invalid parameter format: {item}. Use key=value")
        key, value = item.split("=", 1)
        parameters[key] = value

    result = run_backtest(
        project=args.project,
        algorithm_path=args.algorithm_path,
        config_path=args.config_path,
        parameters=parameters,
        docker_image=args.docker_image,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
