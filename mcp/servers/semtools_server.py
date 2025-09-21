"""Semtools MCP server wrappers.

These tools invoke the semtools CLI (`semtools parse` / `semtools search`) with
hardened security, validation, and circuit breaker protection.
Outputs are returned as structured data for downstream agent processing.
"""
from __future__ import annotations

import logging
import subprocess
import shlex
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from mcp.server import register_tool
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
    from mcp.common.server_config import get_server_config, get_tool_config
except ImportError:  # pragma: no cover
    def register_tool(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator

    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator

logger = logging.getLogger(__name__)

# Lazy configuration loading
_server_config = None
_tool_config = None

def _get_server_config():
    """Get server config lazily."""
    global _server_config
    if _server_config is None:
        try:
            _server_config = get_server_config("semtools_server")
        except Exception:
            _server_config = None
    return _server_config

def _get_tool_config():
    """Get tool config lazily."""
    global _tool_config
    if _tool_config is None:
        try:
            _tool_config = get_tool_config("semtools_server", "semtools.parse")
        except Exception:
            _tool_config = None
    return _tool_config


def _validate_paths(paths: List[str]) -> List[str]:
    """Validate and sanitize file paths with security checks."""
    if not paths:
        raise ValueError("No paths provided")

    if len(paths) > 50:  # Reasonable limit
        raise ValueError(f"Too many paths: {len(paths)} (max 50)")

    validated = []
    workspace_root = Path.cwd().resolve()

    for item in paths:
        # Sanitize path input
        if not isinstance(item, str):
            raise ValueError(f"Path must be string, got {type(item)}")

        if len(item) > 1000:  # Reasonable path length limit
            raise ValueError(f"Path too long: {len(item)} chars (max 1000)")

        # Resolve path and check it exists
        try:
            p = Path(item).resolve()
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid path '{item}': {e}")

        # Security: ensure path is within workspace (no directory traversal)
        try:
            p.relative_to(workspace_root)
        except ValueError:
            raise ValueError(f"Path '{item}' is outside workspace root")

        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {item}")

        # Additional security checks
        if p.is_symlink():
            logger.warning(f"Processing symbolic link: {item}")

        validated.append(str(p))

    logger.info(f"Validated {len(validated)} paths for semtools processing")
    return validated


def _check_semtools_available() -> bool:
    """Check if semtools CLI is available and accessible."""
    try:
        result = subprocess.run(
            ["semtools", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _run_semtools(args: List[str]) -> str:
    """Execute semtools command with enhanced security and error handling."""

    # Validate semtools is available
    if not _check_semtools_available():
        raise RuntimeError("semtools CLI is not available or not in PATH")

    # Get configuration for timeouts and limits
    tool_config = _get_tool_config()
    timeout_seconds = 60  # Default timeout
    if tool_config:
        timeout_seconds = tool_config.get_parameter("timeout_seconds", 60)

    # Security: validate command arguments
    if not args or args[0] != "semtools":
        raise ValueError("Invalid semtools command structure")

    # Limit command length for security
    cmd_str = " ".join(shlex.quote(arg) for arg in args)
    if len(cmd_str) > 4000:  # Reasonable command length limit
        raise ValueError(f"Command too long: {len(cmd_str)} chars (max 4000)")

    logger.info(f"Executing semtools command: {' '.join(args[:3])}...")

    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=dict(os.environ, LANG="C", LC_ALL="C")  # Stable locale
        )

        # Validate output size
        if len(result.stdout) > 10_000_000:  # 10MB limit
            logger.warning(f"semtools output is large: {len(result.stdout)} bytes")

        logger.info(f"semtools command completed successfully ({len(result.stdout)} bytes output)")
        return result.stdout

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"semtools command timed out after {timeout_seconds}s")
    except subprocess.CalledProcessError as exc:
        error_msg = exc.stderr.strip() if exc.stderr else "Unknown error"
        raise RuntimeError(f"semtools command failed with exit code {exc.returncode}: {error_msg}")
    except Exception as exc:
        raise RuntimeError(f"Unexpected error running semtools: {exc}")


def _structure_markdown(markdown: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current = None
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            level = len(stripped) - len(stripped.lstrip('#'))
            heading = stripped[level:].strip()
            current = {"heading": heading, "level": level, "content": ""}
            sections.append(current)
        else:
            if current is None:
                current = {"heading": "", "level": 0, "content": ""}
                sections.append(current)
            current["content"] += (line + "\n")
    return sections


@register_tool(
    name="semtools.parse",
    schema="./schemas/tool.semtools.parse.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def semtools_parse(params: Dict[str, Any]) -> Dict[str, Any]:
    paths = _validate_paths(params["paths"])
    output_format = params.get("output_format", "markdown")
    extra_args = params.get("extra_args", [])
    structured_flag = params.get("structured_output", False)

    cmd = ["semtools", "parse", "--output", output_format, *extra_args, *paths]
    output = _run_semtools(cmd)
    response: Dict[str, Any] = {"output": output}
    if structured_flag:
        response["structured"] = _structure_markdown(output)
    return response


@register_tool(
    name="semtools.search",
    schema="./schemas/tool.semtools.search.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def semtools_search(params: Dict[str, Any]) -> Dict[str, Any]:
    paths = _validate_paths(params["paths"])
    query = params["query"]
    max_distance = params.get("max_distance")
    extra_args = params.get("extra_args", [])

    cmd = ["semtools", "search", query]
    if max_distance is not None:
        cmd.extend(["--max-distance", str(max_distance)])
    cmd.extend(extra_args)
    cmd.extend(paths)
    output = _run_semtools(cmd)
    return {"results": output}


__all__ = ["semtools_parse", "semtools_search"]
