"""Strategy evaluation MCP server bridging Lean backtest wrapper."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from mcp.server import register_tool
from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig

from mcp.wrappers import lean_backtest

logger = logging.getLogger(__name__)


def _ensure_results_dir(project_name: str) -> Path:
    base_dir = Path("lean/results") / project_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _simulate_backtest(project: str, parameters: Dict[str, str], reason: str) -> Dict[str, str]:
    """Generate deterministic backtest outputs when Lean execution is unavailable."""
    output_dir = _ensure_results_dir(project)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    stats = {
        "Project": project,
        "GeneratedAt": timestamp,
        "Parameters": parameters,
        "Note": f"Simulated backtest due to: {reason}",
    }

    stats_path = output_dir / f"simulated-{timestamp}-statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    log_path = output_dir / f"simulated-{timestamp}.log"
    log_path.write_text(
        f"[{timestamp}] Simulated Lean backtest for {project}. Reason: {reason}\n",
        encoding="utf-8",
    )

    return {
        "project": project,
        "statistics_file": str(stats_path),
        "log_file": str(log_path),
        "statistics": json.dumps(stats),
    }


@register_tool(
    name="strategy.eval.run_backtest",
    schema="./schemas/tool.strategy.eval.run_backtest.schema.json",
)
@circuit_breaker(
    name="strategy_eval.run_backtest",
    config=CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=120.0,
        expected_exception=Exception,
    ),
)
def run_backtest(params: Dict[str, Any]) -> Dict[str, str]:
    """Run a Lean backtest, falling back to simulation when infrastructure is unavailable."""
    project = params.get("project")
    algorithm_path = params.get("algorithm_path")
    config_path = params.get("config_path")
    docker_image = params.get("docker_image")
    parameters = params.get("parameters", {}) or {}

    resolved_config = Path(config_path) if config_path else lean_backtest.DEFAULT_CONFIG
    project_name = project or resolved_config.stem

    try:
        result = lean_backtest.run_backtest(
            project=project,
            algorithm_path=algorithm_path,
            config_path=config_path,
            parameters=parameters,
            docker_image=docker_image or lean_backtest.DEFAULT_DOCKER_IMAGE,
        )
        logger.info("Lean backtest completed for project %s", result["project"])
        return result
    except Exception as exc:  # pragma: no cover - exercised in integration scenarios
        reason = str(exc)
        logger.warning("Lean backtest unavailable for %s: %s", project_name, reason)
        return _simulate_backtest(project_name, parameters, reason)


__all__ = ["run_backtest"]
