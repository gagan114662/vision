from __future__ import annotations

import unittest

import mcp.servers.compliance_server as compliance_server


class ComplianceServerTests(unittest.TestCase):
    def test_generate_summary_pass(self) -> None:
        params = {
            "strategy_id": "strat-001",
            "controls": ["Reg SCI", "MiFID"] ,
            "evidence": [
                {"control": "Reg SCI", "status": "PASS", "provenance_id": "prov-1"},
                {"control": "MiFID", "status": "PASS", "provenance_id": "prov-2"}
            ],
            "outstanding_risks": [],
        }
        result = compliance_server.generate_summary(params)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["report"]["next_actions"][0].startswith("Maintain"))

    def test_generate_summary_concerns(self) -> None:
        params = {
            "strategy_id": "strat-002",
            "controls": ["GDPR"],
            "evidence": [
                {"control": "GDPR", "status": "WAIVED", "provenance_id": "prov-3", "notes": "Awaiting DPO sign-off"}
            ],
            "outstanding_risks": ["Pending privacy impact assessment"],
        }
        result = compliance_server.generate_summary(params)
        self.assertEqual(result["status"], "CONCERNS")
        self.assertTrue(result["report"]["next_actions"])


if __name__ == "__main__":
    unittest.main()
