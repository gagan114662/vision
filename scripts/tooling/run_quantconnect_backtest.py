#!/usr/bin/env python3
"""Helper script to sync local files to QuantConnect and launch a cloud backtest.

Prerequisites:
  * QuantConnect MCP server vendored under integrations/quantconnect_mcp/vendor
  * Environment variables QUANTCONNECT_USER_ID / QUANTCONNECT_API_TOKEN
  * Optional: MCP server running locally (only needed for end-to-end validation)

This script uses the MCP wrapper functions in mcp.servers.quantconnect_stub,
which talk directly to QuantConnect's REST API. It performs the following steps:
  1. Reads the specified local algorithm/config files
  2. Pushes them to a QuantConnect project
  3. Triggers a compile + backtest
  4. Polls the backtest status until completion or failure
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from mcp.servers import quantconnect_stub as qc
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Unable to import mcp.servers.quantconnect_stub. Make sure the repository "
        "is on PYTHONPATH and the QuantConnect MCP vendor directory exists."
    ) from exc


def read_file_contents(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def push_files(project_id: str, file_map: dict[str, Path]) -> None:
    payload = [
        {"path": name, "content": read_file_contents(path)}
        for name, path in file_map.items()
    ]
    result = qc.project_sync({
        "project_id": project_id,
        "files": payload,
    })
    warnings = result.get("warnings") or []
    if warnings:
        print("[WARN] Project sync completed with warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print(f"[INFO] Synced files to project {project_id}: {result['synced_files']}")


def run_backtest(project_id: str, backtest_name: str, parameters: dict[str, str] | None = None) -> dict:
    print(f"[INFO] Launching backtest '{backtest_name}' on project {project_id}...")
    response = qc.backtest_run({
        "project_id": project_id,
        "name": backtest_name,
        "parameters": parameters or {},
    })
    print(f"[INFO] Backtest submitted: {response['backtest_id']} (status={response['status']})")
    return response


def wait_for_completion(project_id: str, backtest_id: str, poll_interval: float = 10.0, timeout: float = 3600.0) -> dict:
    print(f"[INFO] Polling backtest status for {backtest_id}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        status_response = qc.backtest_status({
            "project_id": project_id,
            "backtest_id": backtest_id,
        })
        status = status_response["status"]
        print(f"  - Status: {status}")
        if status.lower().startswith("completed") or "runtime" in status.lower():
            return status_response
        time.sleep(poll_interval)
    raise TimeoutError(f"Backtest {backtest_id} did not complete within {timeout} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local QC project and run a cloud backtest")
    parser.add_argument("project_id", help="QuantConnect project ID (integer)")
    parser.add_argument(
        "--name",
        default=f"ci-backtest-{int(time.time())}",
        help="Backtest name to use (default: ci-backtest-<timestamp>)",
    )
    parser.add_argument(
        "--algorithm",
        default="lean/algorithms/monthly_universe_alpha.py",
        help="Path to local algorithm file to upload",
    )
    parser.add_argument(
        "--config",
        default="lean/config/monthly_universe_alpha.json",
        help="Path to Lean config file to upload",
    )
    args = parser.parse_args()

    try:
        push_files(
            project_id=args.project_id,
            file_map={
                "main.py": Path(args.algorithm),
                "config.json": Path(args.config),
            },
        )
        backtest_info = run_backtest(args.project_id, args.name)
        status_response = wait_for_completion(args.project_id, backtest_info["backtest_id"])
        print("[INFO] Backtest finished with status:", status_response["status"])
        stats = status_response.get("statistics", {})
        if stats:
            # show a few interesting metrics
            interesting = {k: stats[k] for k in sorted(stats) if any(metric in k.lower() for metric in ("sharpe", "drawdown", "return"))}
            if interesting:
                print("[INFO] Key statistics:")
                for key, value in interesting.items():
                    print(f"  - {key}: {value}")
    except Exception as exc:  # pragma: no cover - integration path
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
