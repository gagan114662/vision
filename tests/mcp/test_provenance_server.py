"""Unit tests for provenance MCP server."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import types
import unittest
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import patch

import os

import mcp.servers.provenance_server as provenance_server


class _Value:
    def __init__(self, value: Any):
        self._value = value

    def get_string(self) -> str:
        return str(self._value)

    def get_double(self) -> float:
        return float(self._value)

    def get_time(self) -> datetime:
        if isinstance(self._value, datetime):
            return self._value
        raise TypeError("Value is not datetime")

    def get_list(self):
        return types.SimpleNamespace(values=self._value)


class ProvenanceServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, {"PROVENANCE_SIGNING_KEY": "super-secret"})
        self.env_patch.start()
        self.config_patch = patch.object(
            provenance_server.ProvenanceServerConfig,
            "from_env",
            return_value=provenance_server.ProvenanceServerConfig(
                host="localhost",
                port=3322,
                user="immudb",
                password="immudb",
                database="provenance",
            ),
        )
        self.config_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.config_patch.stop()

    def test_get_record_success(self) -> None:
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        row_values: Dict[str, Any] = {
            "record_id": _Value("abc"),
            "source_id": _Value("exchange_nyse"),
            "source_type": _Value("exchange"),
            "dataset_name": _Value("nyse_daily"),
            "instrument": _Value("AAPL"),
            "ingested_at": _Value(timestamp),
            "qc_score": _Value(0.98),
            "validation_notes": _Value([_Value("check1 ok"), _Value("check2 ok")]),
            "hash": _Value("123"),
            "data_location": _Value("s3://bucket/path"),
            "lineage_parent_ids": _Value([_Value("parent1"), _Value("parent2")]),
            "regulatory_tags": _Value([_Value("MiFID"), _Value("SEC")]),
        }
        result = types.SimpleNamespace(rows=[types.SimpleNamespace(values=row_values)])

        class DummyClient:
            def sql_query(self, query: str, params: Dict[str, Any]):
                self.query = query
                self.params = params
                return result

        with patch.object(provenance_server, "_connect", return_value=DummyClient()):
            response = provenance_server.get_record({"record_id": "abc"})

        self.assertIn("record", response)
        record = response["record"]
        self.assertEqual(record["record_id"], "abc")
        self.assertEqual(record["source_id"], "exchange_nyse")
        self.assertEqual(record["dataset_name"], "nyse_daily")
        self.assertEqual(record["lineage_parent_ids"], ["parent1", "parent2"])
        self.assertEqual(record["regulatory_tags"], ["MiFID", "SEC"])
        self.assertIn("retrieved_at", response)
        self.assertIn("signature", response)

        payload = {
            "record": record,
            "retrieved_at": response["retrieved_at"],
        }
        expected_signature = base64.b64encode(
            hmac.new(
                b"super-secret",
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("ascii")
        self.assertEqual(response["signature"], expected_signature)

    def test_get_record_not_found(self) -> None:
        result = types.SimpleNamespace(rows=[])
        class DummyClient:
            def sql_query(self, query: str, params: Dict[str, Any]):
                return result

        with patch.object(provenance_server, "_connect", return_value=DummyClient()):
            with self.assertRaises(ValueError):
                provenance_server.get_record({"record_id": "missing"})


if __name__ == "__main__":
    unittest.main()
