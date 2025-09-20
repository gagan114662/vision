from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from mcp.server import register_tool
except ImportError:  # pragma: no cover
    def register_tool(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func

        return decorator


# Attempt to import the vendored QuantConnect MCP server modules.
_VENDOR_SRC = (
    Path(__file__).resolve().parents[2] / "integrations" / "quantconnect_mcp" / "vendor" / "src"
)
if _VENDOR_SRC.exists() and str(_VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDOR_SRC))

try:  # pragma: no cover - exercised via integration tests when dependencies installed
    from api_connection import post as _qc_post
    from models import (
        CreateProjectFileRequest,
        UpdateFileContentsRequest,
        CreateCompileRequest,
        ReadCompileRequest,
        CreateBacktestRequest,
        ReadBacktestRequest,
        ListBacktestRequest,
        DeleteBacktestRequest,
    )
    import httpx

    _QC_AVAILABLE = True
except Exception:  # pragma: no cover - missing dependencies or vendor folder
    _QC_AVAILABLE = False
    httpx = None  # type: ignore

    def _model_factory(**kwargs):
        return kwargs

    CreateProjectFileRequest = _model_factory  # type: ignore
    UpdateFileContentsRequest = _model_factory  # type: ignore
    CreateCompileRequest = _model_factory  # type: ignore
    ReadCompileRequest = _model_factory  # type: ignore
    CreateBacktestRequest = _model_factory  # type: ignore
    ReadBacktestRequest = _model_factory  # type: ignore


def _ensure_available() -> None:
    if not _QC_AVAILABLE:
        raise RuntimeError(
            "QuantConnect MCP vendor server not available. Clone https://github.com/QuantConnect/mcp-server into "
            "integrations/quantconnect_mcp/vendor and install its requirements."
        )
    if not os.getenv("QUANTCONNECT_USER_ID") or not os.getenv("QUANTCONNECT_API_TOKEN"):
        raise RuntimeError(
            "QuantConnect credentials missing. Set QUANTCONNECT_USER_ID and QUANTCONNECT_API_TOKEN env variables."
        )


def _run_post(endpoint: str, model: object = None, timeout: float = 30.0) -> Dict[str, Any]:
    async def _call() -> Dict[str, Any]:
        return await _qc_post(endpoint, model, timeout)

    return asyncio.run(_call())


def _coerce_project_id(project_id: str) -> int:
    try:
        return int(project_id)
    except ValueError as exc:
        raise RuntimeError("QuantConnect project_id must be an integer ID") from exc


@register_tool(
    name="quantconnect.project.sync",
    schema="./schemas/tool.quantconnect.project.sync.schema.json",
)
def project_sync(params: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_available()
    project_id = _coerce_project_id(params["project_id"])
    files: List[Dict[str, str]] = params.get("files", [])
    synced: List[str] = []
    warnings: List[str] = []

    for file_info in files:
        name = file_info["path"]
        content = file_info["content"]
        create_request = CreateProjectFileRequest(
            projectId=project_id,
            name=name,
            content=content,
            codeSourceId="quantconnect-mcp"
        )
        try:
            _run_post('/files/create', create_request)
            synced.append(name)
            continue
        except Exception as exc:
            if httpx and isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {400, 409}:
                update_request = UpdateFileContentsRequest(
                    projectId=project_id,
                    name=name,
                    content=content,
                    codeSourceId="quantconnect-mcp"
                )
                _run_post('/files/update', update_request)
                synced.append(name)
                continue
            warnings.append(f"{name}: {exc}")

    return {
        "project_id": str(project_id),
        "synced_files": synced,
        "warnings": warnings,
    }


def _wait_for_compile(project_id: int, compile_id: str, timeout_seconds: float = 120.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = _run_post(
            '/compile/read',
            ReadCompileRequest(projectId=project_id, compileId=compile_id)
        )
        state = response.get('state')
        if state == 'BuildSuccess':
            return
        if state == 'BuildError':
            errors = response.get('errors') or []
            raise RuntimeError(f"QuantConnect compile failed: {'; '.join(errors)}")
        time.sleep(2.0)
    raise TimeoutError("Timed out waiting for QuantConnect compile to finish")


@register_tool(
    name="quantconnect.backtest.run",
    schema="./schemas/tool.quantconnect.backtest.run.schema.json",
)
def backtest_run(params: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_available()
    project_id = _coerce_project_id(params["project_id"])
    name = params["name"].strip() or f"Backtest-{int(time.time())}"
    parameters = params.get("parameters") or {}

    compile_response = _run_post('/compile/create', CreateCompileRequest(projectId=project_id))
    compile_id = compile_response.get('compileId')
    if not compile_id:
        raise RuntimeError("QuantConnect compile did not return a compileId")

    _wait_for_compile(project_id, compile_id)

    backtest_response = _run_post(
        '/backtests/create',
        CreateBacktestRequest(
            projectId=project_id,
            compileId=compile_id,
            backtestName=name,
            parameters=parameters or None,
        ),
        timeout=120.0,
    )

    backtest = backtest_response.get('backtest', {})
    backtest_id = backtest.get('backtestId') or backtest_response.get('backtestId')
    if not backtest_id:
        raise RuntimeError("QuantConnect backtest response missing backtestId")

    status = backtest.get('status') or ('Completed' if backtest.get('completed') else 'InProgress')
    message = ''
    errors = backtest_response.get('errors') or []
    if errors:
        message = '; '.join(errors)

    return {
        "project_id": str(project_id),
        "backtest_id": str(backtest_id),
        "status": status,
        "message": message,
    }


def _flatten_stats(prefix: str, value: Any, collector: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, sub_value in value.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_stats(new_prefix, sub_value, collector)
    else:
        if isinstance(value, (int, float, str)):
            collector[prefix] = value
        else:
            collector[prefix] = str(value)


@register_tool(
    name="quantconnect.backtest.status",
    schema="./schemas/tool.quantconnect.backtest.status.schema.json",
)
def backtest_status(params: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_available()
    project_id = _coerce_project_id(params["project_id"])
    backtest_id = params["backtest_id"]

    response = _run_post(
        '/backtests/read',
        ReadBacktestRequest(projectId=project_id, backtestId=backtest_id),
        timeout=60.0,
    )

    backtest = response.get('backtest', {})
    status = backtest.get('status') or ('Completed' if backtest.get('completed') else 'InProgress')
    statistics_raw = backtest.get('statistics') or {}
    flattened: Dict[str, Any] = {}
    _flatten_stats('', statistics_raw, flattened)

    return {
        "project_id": str(project_id),
        "backtest_id": str(backtest_id),
        "status": status,
        "statistics": flattened,
    }


@register_tool(
    name="quantconnect.backtest.list",
    schema="./schemas/tool.quantconnect.backtest.list.schema.json",
)
def backtest_list(params: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_available()
    project_id = _coerce_project_id(params["project_id"])

    response = _run_post(
        '/backtests/list',
        ListBacktestRequest(projectId=project_id),
        timeout=60.0,
    )

    backtests = response.get('backtests', [])
    return {
        "project_id": str(project_id),
        "backtests": backtests
    }


@register_tool(
    name="quantconnect.backtest.delete",
    schema="./schemas/tool.quantconnect.backtest.delete.schema.json",
)
def backtest_delete(params: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_available()
    project_id = _coerce_project_id(params["project_id"])
    backtest_id = params["backtest_id"]

    response = _run_post(
        '/backtests/delete',
        DeleteBacktestRequest(projectId=project_id, backtestId=backtest_id),
        timeout=60.0,
    )

    return {
        "project_id": str(project_id),
        "backtest_id": str(backtest_id),
        "success": response.get('success', False)
    }


__all__ = ["project_sync", "backtest_run", "backtest_status", "backtest_list", "backtest_delete"]
