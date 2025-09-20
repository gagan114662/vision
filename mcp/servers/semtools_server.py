"""Semtools MCP server wrappers.

These tools invoke the semtools CLI (`semtools parse` / `semtools search`).
Outputs are returned as raw stdout for agents to process downstream.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List

try:
    from mcp.server import register_tool
except ImportError:  # pragma: no cover
    def register_tool(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func

        return decorator


def _validate_paths(paths: List[str]) -> List[str]:
    validated = []
    for item in paths:
        p = Path(item)
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {item}")
        validated.append(str(p))
    return validated


def _run_semtools(args: List[str]) -> str:
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise RuntimeError(
            f"semtools command failed with exit code {exc.returncode}: {exc.stderr.strip()}"
        ) from exc
    return result.stdout


@register_tool(
    name="semtools.parse",
    schema="./schemas/tool.semtools.parse.schema.json",
)
def semtools_parse(params: Dict[str, Any]) -> Dict[str, Any]:
    paths = _validate_paths(params["paths"])
    output_format = params.get("output_format", "markdown")
    extra_args = params.get("extra_args", [])

    cmd = ["semtools", "parse", "--output", output_format, *extra_args, *paths]
    output = _run_semtools(cmd)
    return {"output": output}


@register_tool(
    name="semtools.search",
    schema="./schemas/tool.semtools.search.schema.json",
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
