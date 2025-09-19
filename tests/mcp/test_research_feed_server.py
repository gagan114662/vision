from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import mcp.servers.research_feed_server as research_server

FEED_PATH = Path("data/processed/research_feed.json")


class ResearchFeedServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_feed = {
            "generated_at": "2025-09-19T00:00:00Z",
            "insights": [
                {
                    "id": "insight-1",
                    "source_id": "mql5-alphaevolve",
                    "title": "Alpha Agents",
                    "summary": "Test",
                    "provenance_url": "http://example.com",
                    "category": "strategy",
                    "impact_estimate": "high",
                    "effort_estimate": "medium",
                    "status": "pending-validation",
                    "created_at": "2025-09-19T00:00:00Z",
                },
                {
                    "id": "insight-2",
                    "source_id": "github-options",
                    "title": "Options Tool",
                    "summary": "Test",
                    "provenance_url": "http://example2.com",
                    "category": "risk",
                    "impact_estimate": "medium",
                    "effort_estimate": "medium",
                    "status": "queued",
                    "created_at": "2025-09-19T00:00:00Z",
                },
            ],
        }
        self.patch = patch.object(research_server, "FEED_PATH", FEED_PATH)
        self.patch.start()
        FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEED_PATH.write_text(json.dumps(self.sample_feed), encoding="utf-8")

    def tearDown(self) -> None:
        if FEED_PATH.exists():
            FEED_PATH.unlink()
        self.patch.stop()

    def test_list_insights_default(self) -> None:
        response = research_server.list_insights({})
        self.assertEqual(len(response["insights"]), 2)

    def test_list_insights_with_filters(self) -> None:
        response = research_server.list_insights({"status": ["pending-validation"], "category": "strategy"})
        self.assertEqual(len(response["insights"]), 1)
        self.assertEqual(response["insights"][0]["id"], "insight-1")


if __name__ == "__main__":
    unittest.main()
