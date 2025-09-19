from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import mcp.servers.feature_engineering_server as fe_server


def _make_prices(count: int, start_price: float = 100.0, step: float = 1.0):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data = []
    price = start_price
    for i in range(count):
        data.append(
            {
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "price": price,
            }
        )
        price += step
    return data


class FeatureEngineeringTests(unittest.TestCase):
    def test_momentum_computation(self) -> None:
        data = _make_prices(30)
        result = fe_server.compute_factor({
            "factor_name": "momentum",
            "data": data,
            "window": 5,
        })
        self.assertEqual(result["factor_name"], "momentum")
        self.assertEqual(result["window"], 5)
        self.assertTrue(result["results"])
        last_value = result["results"][-1]["value"]
        self.assertAlmostEqual(last_value, (data[-1]["price"] / data[-6]["price"]) - 1.0)

    def test_volatility_requires_min_window(self) -> None:
        data = _make_prices(10)
        result = fe_server.compute_factor({
            "factor_name": "volatility",
            "data": data,
            "window": 3,
        })
        self.assertTrue(all(item["value"] >= 0 for item in result["results"]))

    def test_mean_reversion_sign(self) -> None:
        data = _make_prices(15)
        result = fe_server.compute_factor({
            "factor_name": "mean_reversion",
            "data": data,
            "window": 4,
        })
        # Uptrend should yield negative mean reversion (since price above mean -> negative deviation)
        self.assertTrue(all(value["value"] <= 0 for value in result["results"]))

    def test_insufficient_data_raises(self) -> None:
        with self.assertRaises(ValueError):
            fe_server.compute_factor({
                "factor_name": "momentum",
                "data": _make_prices(1),
            })


if __name__ == "__main__":
    unittest.main()
