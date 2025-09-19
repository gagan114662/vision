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
DEFAULT_ALGORITHM = Path("lean/algorithms/monthly_universe_alpha.py")
DEFAULT_CONFIG = Path("lean/config/monthly_universe_alpha.json")
DEFAULT_DOCKER_IMAGE = "quantconnect/lean:latest"


@dataclass
class BacktestRequest:
    project: str
    algorithm_path: Path
    config_path: Path
    output_dir: Path
    parameters: Dict[str, str]
    docker_image: str = DEFAULT_DOCKER_IMAGE


def _check_env() -> None:
    missing = [env for env in REQUIRED_ENV_VARS if not os.environ.get(env)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


def _build_command(request: BacktestRequest) -> List[str]:
    algorithm_dir = request.algorithm_path.parent.resolve()
    config_path = request.config_path.resolve()
    config_container_path = f"/Lean/config/{config_path.name}"
    data_dir = Path("lean/data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        f"QUANTCONNECT_USER_ID={os.environ['QUANTCONNECT_USER_ID']}",
        "-e",
        f"QUANTCONNECT_API_TOKEN={os.environ['QUANTCONNECT_API_TOKEN']}",
        "-v",
        f"{config_path.parent}:/Lean/config",
        "-v",
        f"{algorithm_dir}:/Lean/Algorithm.Python",
        "-v",
        f"{request.output_dir.resolve()}:/Lean/results",
        "-v",
        f"{data_dir}:/Lean/data",
        request.docker_image,
        "lean",
        "backtest",
        request.project,
        "--config",
        config_container_path,
        "--data-folder",
        "/Lean/data",
        "--output",
        "/Lean/results",
    ]
    for key, value in request.parameters.items():
        base_cmd.extend(["--parameter", f"{key}={value}"])
    return base_cmd


def run_backtest(
    project: str | None = None,
    algorithm_path: str | None = None,
    config_path: str | None = None,
    parameters: Dict[str, str] | None = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
) -> Dict[str, str]:
    _check_env()
    parameters = parameters or {}

    resolved_config = Path(config_path or DEFAULT_CONFIG)
    resolved_algorithm = Path(algorithm_path or DEFAULT_ALGORITHM)
    if not resolved_config.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config}")
    if not resolved_algorithm.exists():
        raise FileNotFoundError(f"Algorithm file not found: {resolved_algorithm}")

    project_name = project or resolved_config.stem
    output_dir = Path("lean/results") / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    request = BacktestRequest(
        project=project_name,
        algorithm_path=resolved_algorithm,
        config_path=resolved_config,
        output_dir=output_dir,
        parameters=parameters,
        docker_image=docker_image,
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
        "project": project_name,
        "statistics_file": str(stats_path),
        "log_file": log_file.name,
        "statistics": json.dumps(stats),
    }
