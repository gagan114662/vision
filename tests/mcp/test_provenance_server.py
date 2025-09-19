"""Unit tests for provenance MCP server stub."""
from __future__ import annotations

import types
import unittest
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import patch

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

        with patch.object(provenance_server, "_connect", return_value=types.SimpleNamespace(sqlQuery=lambda _query: result)):
            response = provenance_server.get_record({"record_id": "abc"})

        self.assertIn("record", response)
        record = response["record"]
        self.assertEqual(record["record_id"], "abc")
        self.assertEqual(record["source_id"], "exchange_nyse")
        self.assertEqual(record["dataset_name"], "nyse_daily")
        self.assertEqual(record["lineage_parent_ids"], ["parent1", "parent2"])
        self.assertEqual(record["regulatory_tags"], ["MiFID", "SEC"])
        self.assertIn("retrieved_at", response)

    def test_get_record_not_found(self) -> None:
        result = types.SimpleNamespace(rows=[])
        with patch.object(provenance_server, "_connect", return_value=types.SimpleNamespace(sqlQuery=lambda _query: result)):
            with self.assertRaises(ValueError):
                provenance_server.get_record({"record_id": "missing"})


if __name__ == "__main__":
    unittest.main()
