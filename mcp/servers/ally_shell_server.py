"""MCP server exposing guarded terminal access via Ally-compatible executor."""
from __future__ import annotations

import os
import subprocess
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import register_tool
from mcp.common.server_config import get_server_config, get_tool_config

logger = logging.getLogger(__name__)

# Lazy configuration loading to avoid import-time SECRET_KEY validation
_server_config = None
_tool_config = None

def _get_server_config():
    """Get server config lazily."""
    global _server_config
    if _server_config is None:
        _server_config = get_server_config("ally_shell_server")
    return _server_config

def _get_tool_config():
    """Get tool config lazily."""
    global _tool_config
    if _tool_config is None:
        _tool_config = get_tool_config("ally_shell_server", "ops.shell.run_command")
    return _tool_config

BASE_DIR = Path.cwd().resolve()


def _require_within_workspace(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(BASE_DIR)
    except ValueError as exc:
        raise ValueError(f"workdir '{resolved}' is outside the repository root") from exc
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"workdir '{resolved}' does not exist or is not a directory")
    return resolved


def _validate_command(cmd: List[Any]) -> List[str]:
    """Validate command with configuration-based security checks."""
    if not isinstance(cmd, list) or not cmd:
        raise ValueError("command must be a non-empty list")

    # Get allowed commands from configuration
    allowed_commands = ["git", "lean", "python", "python3", "pip", "ls", "pwd", "find"]
    if _tool_config:
        allowed_commands = _tool_config.get_parameter("allowed_commands", allowed_commands)
        logger.info(f"Using configured allowed commands: {allowed_commands}")

    validated: List[str] = []
    for item in cmd:
        if not isinstance(item, str):
            raise ValueError("command entries must be strings")
        if not item:
            raise ValueError("command entries must not be empty strings")
        validated.append(item)

    # Check if command is in allowed list
    if validated and validated[0] not in allowed_commands:
        raise ValueError(f"Command '{validated[0]}' not in allowed commands: {allowed_commands}")

    return validated


@dataclass
class AllyExecutionRequest:
    command: List[str]
    workdir: Optional[Path] = None
    env: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    dry_run: bool = False
    use_ally: bool = True

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "AllyExecutionRequest":
        command = _validate_command(payload["command"])

        workdir_value = payload.get("workdir")
        workdir = None
        if workdir_value:
            workdir = _require_within_workspace(Path(workdir_value))

        env = {}
        for key, value in (payload.get("env") or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("env keys and values must be strings")
            env[key] = value

        timeout = float(payload.get("timeout_seconds", 30.0))
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")

        dry_run = bool(payload.get("dry_run", False))
        use_ally = bool(payload.get("use_ally", True))

        return cls(
            command=command,
            workdir=workdir,
            env=env,
            timeout=timeout,
            dry_run=dry_run,
            use_ally=use_ally,
        )


def _build_executor_command(params: AllyExecutionRequest) -> List[str]:
    ally_binary = os.getenv("ALLY_BINARY")
    if not params.use_ally or not ally_binary:
        return params.command

    # Ally's CLI accepts a shell command; we join safely for transparency.
    return [ally_binary, "run", "--", *params.command]


def _execute(params: AllyExecutionRequest) -> Dict[str, Any]:
    if params.dry_run:
        return {
            "command": params.command,
            "executor": "dry_run",
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timed_out": False,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0.0,
            "workdir": str(params.workdir or BASE_DIR),
        }

    executor_cmd = _build_executor_command(params)
    cwd = params.workdir or BASE_DIR

    captured_env = os.environ.copy()
    captured_env.update(params.env)

    start = time.perf_counter()
    start_time = datetime.now(timezone.utc).isoformat()
    try:
        completed = subprocess.run(
            executor_cmd,
            cwd=str(cwd),
            env=captured_env,
            capture_output=True,
            text=True,
            timeout=params.timeout,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = exc  # type: ignore[assignment]
        timed_out = True

    end = time.perf_counter()
    end_time = datetime.now(timezone.utc).isoformat()

    stdout = completed.stdout if hasattr(completed, "stdout") else ""
    stderr = completed.stderr if hasattr(completed, "stderr") else ""
    exit_code = completed.returncode if hasattr(completed, "returncode") else None

    return {
        "command": params.command,
        "executor": "ally" if os.getenv("ALLY_BINARY") and params.use_ally else "subprocess",
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": round(end - start, 4),
        "workdir": str(cwd),
    }


@register_tool(
    name="ops.shell.run_command",
    schema="./schemas/tool.ops.shell.run_command.schema.json",
)
def run_command(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute shell command with configuration-based security and timeout settings."""

    # Apply configuration defaults
    if _tool_config:
        # Apply default timeout from configuration
        if "timeout_seconds" not in payload and _tool_config.timeout_seconds:
            payload["timeout_seconds"] = _tool_config.timeout_seconds
            logger.info(f"Using configured timeout: {_tool_config.timeout_seconds}s")

        # Apply workspace root from configuration
        workspace_root = _tool_config.get_parameter("workspace_root", ".")
        if workspace_root != ".":
            logger.info(f"Using configured workspace root: {workspace_root}")

        # Apply dry run default from configuration
        if "dry_run" not in payload:
            enable_dry_run = _tool_config.get_parameter("enable_dry_run", True)
            if enable_dry_run and _server_config and _server_config.get_setting("security_level") == "high":
                payload["dry_run"] = True
                logger.info("Enabling dry run for high security level")

    request = AllyExecutionRequest.from_payload(payload)

    # Add configuration metadata to response
    result = _execute(request)

    # Add server configuration info to response
    if _server_config:
        result["server_config"] = {
            "executor_type": _server_config.get_setting("executor_type", "subprocess"),
            "security_level": _server_config.get_setting("security_level", "medium"),
            "provenance_logging": _server_config.get_setting("enable_provenance_logging", True)
        }

    return result


__all__ = ["run_command"]
