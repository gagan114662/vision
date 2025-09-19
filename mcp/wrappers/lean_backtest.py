"""Lean backtest MCP wrapper.

Provides structured access to QuantConnect Lean CLI backtests with secure credential handling.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

REQUIRED_ENV_VARS = ["QUANTCONNECT_USER_ID", "QUANTCONNECT_API_TOKEN"]


@dataclass
class BacktestRequest:
    project: str
    algorithm: str
    config_path: str
    output_dir: Path
    parameters: Dict[str, str]


def _check_env() -> None:
    missing = [env for env in REQUIRED_ENV_VARS if not os.environ.get(env)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


def _build_command(request: BacktestRequest) -> List[str]:
    base_cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        f"QUANTCONNECT_USER_ID={os.environ['QUANTCONNECT_USER_ID']}",
        "-e",
        f"QUANTCONNECT_API_TOKEN={os.environ['QUANTCONNECT_API_TOKEN']}",
        "-v",
        f"{Path(request.config_path).parent}:/Lean/config",
        "-v",
        f"{Path(request.algorithm).parent}:/Lean/Algorithm.Python",
        "-v",
        f"{request.output_dir}:/Lean/results",
        "quantconnect/lean:latest",
        "lean",
        "backtest",
        request.project,
        "--data-folder",
        "/Lean/data",
        "--output",
        "/Lean/results",
    ]
    for key, value in request.parameters.items():
        base_cmd.extend(["--parameter", f"{key}={value}"])
    return base_cmd


def run_backtest(project: str, algorithm_path: str, config_path: str, parameters: Dict[str, str] | None = None) -> Dict[str, str]:
    _check_env()
    parameters = parameters or {}

    output_dir = Path("lean/results") / project
    output_dir.mkdir(parents=True, exist_ok=True)

    request = BacktestRequest(
        project=project,
        algorithm=algorithm_path,
        config_path=config_path,
        output_dir=output_dir,
        parameters=parameters,
    )

    cmd = _build_command(request)

    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".log", delete=False) as log_file:
        try:
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Lean backtest failed: {exc}")

    stats_path = next(output_dir.glob("**/*-statistics.json"), None)
    if not stats_path:
        raise FileNotFoundError("Backtest statistics file not found in results directory")

    with stats_path.open() as f:
        stats = json.load(f)

    return {
        "project": project,
        "statistics_file": str(stats_path),
        "log_file": log_file.name,
        "statistics": json.dumps(stats),
    }
