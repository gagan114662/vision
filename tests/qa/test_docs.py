"""QA tests validating presence and integrity of planning artifacts."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
DOCS = BASE_DIR / "docs"
BMAD_CORE = BASE_DIR / ".bmad-core"
REQUIRED_DOCS = [
    DOCS / "prd.md",
    DOCS / "architecture.md",
    DOCS / "architecture" / "coding-standards.md",
    DOCS / "architecture" / "tech-stack.md",
    DOCS / "architecture" / "project-structure.md",
    BMAD_CORE / "core-config.yaml",
    BMAD_CORE / "data" / "technical-preferences.md",
    DOCS / "qa" / "assessments" / "platform.foundation-risk-20250919.md",
    DOCS / "qa" / "assessments" / "platform.foundation-design-20250919.md",
    DOCS / "qa" / "gates" / "platform.foundation-qa-gate.yml",
    BASE_DIR / "mcp" / "registry.yaml",
    BASE_DIR / "lean" / "algorithms" / "monthly_universe_alpha.py",
    BASE_DIR / "lean" / "config" / "monthly_universe_alpha.json",
]


class TestDocumentationIntegrity(unittest.TestCase):
    def test_required_docs_present(self) -> None:
        for path in REQUIRED_DOCS:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"Missing required document: {path}")
                content = path.read_text(encoding="utf-8").strip()
                self.assertTrue(content, f"Document is empty: {path}")

    def test_core_config_references_existing_files(self) -> None:
        config_path = BMAD_CORE / "core-config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        always_files = config.get("devLoadAlwaysFiles", [])
        self.assertTrue(isinstance(always_files, list) and always_files, "devLoadAlwaysFiles must be a non-empty list")
        for rel_path in always_files:
            resolved = BASE_DIR / rel_path
            with self.subTest(file=rel_path):
                self.assertTrue(resolved.exists(), f"Configured always-load file missing: {rel_path}")

    def test_semtools_metadata_schema(self) -> None:
        metadata_path = BASE_DIR / "agents" / "configs" / "semtools-metadata.yaml"
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))

        self.assertIn("embeddings", metadata)
        self.assertIsInstance(metadata["embeddings"], dict)
        self.assertTrue({"primary", "secondary"}.issubset(metadata["embeddings"].keys()))

        routing = metadata.get("routing")
        self.assertIsInstance(routing, dict, "routing section missing")
        threshold = routing.get("rejection_threshold")
        self.assertIsNotNone(threshold)
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 1)

        initial_tools = metadata.get("initial_tools", [])
        self.assertTrue(initial_tools, "initial_tools section must list at least one tool")
        required_fields = {"name", "description", "provenance_required"}
        for tool in initial_tools:
            with self.subTest(tool=tool.get("name")):
                self.assertTrue(required_fields.issubset(tool.keys()), f"Tool missing required fields: {tool}")
        tool_names = {tool["name"] for tool in initial_tools}
        expected_tools = {
            "market-data.pricing.get_ohlcv",
            "feature-engineering.compute_factor",
            "strategy.eval.run_backtest",
            "risk.limits.evaluate_portfolio",
            "compliance.generate_summary",
        }
        self.assertTrue(expected_tools.issubset(tool_names))

    def test_roadmap_covers_goals(self) -> None:
        roadmap = (BASE_DIR / "ROADMAP.md").read_text(encoding="utf-8")
        for goal in ("Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6"):
            with self.subTest(goal=goal):
                self.assertIn(goal, roadmap, f"Roadmap missing {goal} milestone")
        self.assertIn("Backtest Gate", roadmap, "Roadmap missing backtest gate checkpoint")

    def test_provenance_schema_required_fields(self) -> None:
        schema_path = BASE_DIR / "mcp" / "schemas" / "provenance_record.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema.get("title"), "ProvenanceRecord")
        required_fields = set(schema.get("required", []))
        expected = {"record_id", "source_id", "source_type", "ingested_at", "qc_score", "validation_notes", "hash", "data_location"}
        self.assertTrue(expected.issubset(required_fields), "Provenance schema missing required fields")

        response_path = BASE_DIR / "mcp" / "schemas" / "tool.provenance.get_record.response.schema.json"
        response_schema = json.loads(response_path.read_text(encoding="utf-8"))
        self.assertEqual(response_schema.get("title"), "ProvenanceGetRecordResponse")
        response_required = set(response_schema.get("required", []))
        self.assertTrue({"record", "retrieved_at", "signature"}.issubset(response_required))

    def test_feature_engineering_schema_required_fields(self) -> None:
        schema_path = BASE_DIR / "mcp" / "schemas" / "tool.feature-engineering.compute_factor.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema.get("title"), "ComputeFactorRequest")
        required_fields = set(schema.get("required", []))
        self.assertTrue({"factor_name", "data"}.issubset(required_fields))

        response_path = BASE_DIR / "mcp" / "schemas" / "tool.feature-engineering.compute_factor.response.schema.json"
        response_schema = json.loads(response_path.read_text(encoding="utf-8"))
        self.assertEqual(response_schema.get("title"), "ComputeFactorResponse")
        response_required = set(response_schema.get("required", []))
        self.assertTrue({"factor_name", "window", "results"}.issubset(response_required))

    def test_risk_schema_required_fields(self) -> None:
        schema_path = BASE_DIR / "mcp" / "schemas" / "tool.risk.limits.evaluate_portfolio.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema.get("title"), "EvaluatePortfolioRequest")
        required_fields = set(schema.get("required", []))
        self.assertTrue({"positions", "limits"}.issubset(required_fields))

        response_path = BASE_DIR / "mcp" / "schemas" / "tool.risk.limits.evaluate_portfolio.response.schema.json"
        response_schema = json.loads(response_path.read_text(encoding="utf-8"))
        self.assertEqual(response_schema.get("title"), "EvaluatePortfolioResponse")
        self.assertTrue({"portfolio_value", "var", "max_drawdown", "breaches", "recommendations"}.issubset(response_schema.get("required", [])))

    def test_compliance_schema_required_fields(self) -> None:
        schema_path = BASE_DIR / "mcp" / "schemas" / "tool.compliance.generate_summary.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema.get("title"), "GenerateComplianceSummaryRequest")
        required_fields = set(schema.get("required", []))
        self.assertTrue({"strategy_id", "controls", "evidence"}.issubset(required_fields))

        response_path = BASE_DIR / "mcp" / "schemas" / "tool.compliance.generate_summary.response.schema.json"
        response_schema = json.loads(response_path.read_text(encoding="utf-8"))
        self.assertEqual(response_schema.get("title"), "GenerateComplianceSummaryResponse")
        self.assertTrue({"strategy_id", "status", "report"}.issubset(response_schema.get("required", [])))

    def test_registry_contains_new_tools(self) -> None:
        registry = yaml.safe_load((BASE_DIR / "mcp" / "registry.yaml").read_text(encoding="utf-8"))
        namespaces = registry.get("namespaces", {})
        self.assertIn("feature-engineering", namespaces)
        self.assertIn("risk", namespaces)
        self.assertIn("compliance", namespaces)
        feature_tools = [tool["id"] for tool in namespaces["feature-engineering"].get("tools", [])]
        self.assertIn("feature-engineering.compute_factor", feature_tools)
        risk_tools = [tool["id"] for tool in namespaces["risk"].get("tools", [])]
        self.assertIn("risk.limits.evaluate_portfolio", risk_tools)
        compliance_tools = [tool["id"] for tool in namespaces["compliance"].get("tools", [])]
        self.assertIn("compliance.generate_summary", compliance_tools)

    def test_run_backtest_schema_required_fields(self) -> None:
        schema_path = BASE_DIR / "mcp" / "schemas" / "tool.strategy.eval.run_backtest.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema.get("title"), "RunBacktestRequest")
        properties = schema.get("properties", {})
        self.assertTrue({"project", "algorithm_path", "config_path", "parameters", "docker_image"}.issubset(properties.keys()))

        response_path = BASE_DIR / "mcp" / "schemas" / "tool.strategy.eval.run_backtest.response.schema.json"
        response_schema = json.loads(response_path.read_text(encoding="utf-8"))
        self.assertEqual(response_schema.get("title"), "RunBacktestResponse")
        response_required = set(response_schema.get("required", []))
        self.assertTrue({"project", "statistics_file", "log_file", "statistics"}.issubset(response_required))

    def test_market_data_schema_required_fields(self) -> None:
        schema_path = BASE_DIR / "mcp" / "schemas" / "tool.market-data.pricing.get_ohlcv.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema.get("title"), "GetOHLCVRequest")
        required_fields = set(schema.get("required", []))
        self.assertTrue({"symbol", "start", "end", "interval"}.issubset(required_fields))

        response_path = BASE_DIR / "mcp" / "schemas" / "tool.market-data.pricing.get_ohlcv.response.schema.json"
        response_schema = json.loads(response_path.read_text(encoding="utf-8"))
        self.assertEqual(response_schema.get("title"), "GetOHLCVResponse")
        response_required = set(response_schema.get("required", []))
        self.assertTrue({"symbol", "interval", "rows", "provenance_ids"}.issubset(response_required))


if __name__ == "__main__":
    unittest.main()
