from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import mcp.servers.chart_server as chart_server


class ChartServerTests(unittest.TestCase):
    def test_render_price_series_writes_file(self) -> None:
        prices = [
            {"timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(), "price": 100.0},
            {"timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc).isoformat(), "price": 101.5},
        ]
        signals = [
            {"timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc).isoformat(), "label": "Buy", "value": 101.5}
        ]

        def fake_render(prices, signals, title, output_path):
            output_path.write_text("fake-image")
            return []

        with patch.object(chart_server, "_render_chart", side_effect=fake_render):
            response = chart_server.render_price_series({
                "prices": prices,
                "signals": signals,
                "title": "Test Chart",
            })
        self.assertTrue(response["image_path"].endswith(".png"))

    def test_render_price_series_requires_points(self) -> None:
        with self.assertRaises(ValueError):
            chart_server.render_price_series({"prices": [{"timestamp": "2024-01-01T00:00:00Z", "price": 1.0}]})


if __name__ == "__main__":
    unittest.main()
