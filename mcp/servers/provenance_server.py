"""Provenance MCP server implementation.

Provides signed, schema-compliant access to the provenance ledger backed by immudb.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Protocol

from mcp.server import register_tool

try:
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
except ImportError:
    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator

# Autonomous dependency management for pyimmudb
def _ensure_immudb():
    """Ensure pyimmudb is available, auto-installing and retrying until success."""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            from pyimmudb.client import ImmuClient
            print("✅ ImmuDB client verified and ready")
            return ImmuClient
        except ImportError:
            if attempt < max_attempts - 1:
                print(f"🔧 Auto-installing pyimmudb (attempt {attempt + 1}/{max_attempts})")
                import subprocess
                import sys

                install_strategies = [
                    [sys.executable, "-m", "pip", "install", "pyimmudb"],
                    [sys.executable, "-m", "pip", "install", "pyimmudb", "--no-cache-dir"],
                    [sys.executable, "-m", "pip", "install", "pyimmudb", "--force-reinstall"]
                ]

                for strategy in install_strategies:
                    try:
                        subprocess.check_call(strategy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print("✅ PyImmuDB installation completed")
                        from pyimmudb.client import ImmuClient
                        print("✅ PyImmuDB successfully imported")
                        return ImmuClient
                    except (subprocess.CalledProcessError, ImportError):
                        continue
            else:
                raise RuntimeError(f"Unable to install pyimmudb after {max_attempts} attempts. Please check system permissions and internet connectivity.")

    raise RuntimeError("Unexpected error in pyimmudb dependency resolution")

class _ImmuRowValues(Protocol):  # pragma: no cover - structural typing only
    def get(self, key: str, default: Any | None = None) -> Any:
        ...


class _ImmuRow(Protocol):  # pragma: no cover - structural typing only
    values: Dict[str, Any]


@dataclass
class ProvenanceServerConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "ProvenanceServerConfig":
        return cls(
            host=os.environ.get("IMMUDb_HOST", "localhost"),
            port=int(os.environ.get("IMMUDb_PORT", "3322")),
            user=os.environ.get("IMMUDb_USER", "immudb"),
            password=os.environ.get("IMMUDb_PASSWORD", "immudb"),
            database=os.environ.get("IMMUDb_DATABASE", "provenance"),
        )


def _connect(config: ProvenanceServerConfig):
    client_cls = _ensure_immudb()
    client = client_cls((config.host, config.port))
    client.login(config.user, config.password)
    client.useDatabase(config.database.encode())
    return client


def _sign_payload(payload: Dict[str, Any]) -> str:
    key = os.environ.get("PROVENANCE_SIGNING_KEY")
    if not key:
        raise EnvironmentError("PROVENANCE_SIGNING_KEY environment variable is required for signing")
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(key.encode("utf-8"), serialized, hashlib.sha256).digest()
    return base64.b64encode(signature).decode("ascii")


def _convert_row(row: _ImmuRow) -> Dict[str, Any]:
    values = row.values

    def _get_list(field: str) -> Iterable[Any]:
        if field not in values:
            return []
        list_value = values[field]
        getter = getattr(list_value, "get_list", None)
        if getter is None:
            return []
        list_obj = getter()
        iter_values = getattr(list_obj, "values", [])
        return [v.get_string() for v in iter_values]

    record = {
        "record_id": values["record_id"].get_string(),
        "source_id": values["source_id"].get_string(),
        "source_type": values["source_type"].get_string(),
        "dataset_name": values.get("dataset_name").get_string() if "dataset_name" in values else None,
        "instrument": values.get("instrument").get_string() if "instrument" in values else None,
        "ingested_at": values["ingested_at"].get_time().isoformat(),
        "qc_score": values["qc_score"].get_double(),
        "validation_notes": [
            v.get_string() for v in values.get("validation_notes").get_list().values
        ] if "validation_notes" in values else [],
        "hash": values["hash"].get_string(),
        "data_location": values["data_location"].get_string(),
        "lineage_parent_ids": list(_get_list("lineage_parent_ids")),
        "regulatory_tags": list(_get_list("regulatory_tags")),
    }
    return record


class ProvenanceServer:
    """Thin wrapper around immudb for MCP consumption."""

    def __init__(self, client_factory: Any | None = None):
        self._client_factory = client_factory or _connect

    def get_record(self, record_id: str) -> Dict[str, Any]:
        config = ProvenanceServerConfig.from_env()
        client = self._client_factory(config)
        query = "SELECT * FROM records WHERE record_id = @record_id LIMIT 1"
        params = {"record_id": record_id}
        if hasattr(client, "sql_query"):
            result = client.sql_query(query, params)  # type: ignore[attr-defined]
        else:
            result = client.sqlQuery(query, params)  # type: ignore[attr-defined]
        rows = getattr(result, "rows", [])
        if not rows:
            raise ValueError(f"Record {record_id} not found in provenance ledger")

        record = _convert_row(rows[0])
        payload = {
            "record": record,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        payload["signature"] = _sign_payload(payload)
        return payload


@register_tool(
    name="provenance.get_record",
    schema="./schemas/tool.provenance.get_record.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def get_record(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return provenance record with signed payload."""

    server = ProvenanceServer()
    return server.get_record(record_id=params["record_id"])


__all__ = [
    "ProvenanceServer",
    "ProvenanceServerConfig",
    "get_record",
]
