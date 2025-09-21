"""
Real QuantConnect Integration with MCP Client.

This replaces the stubbed implementation with genuine calls to QuantConnect's
REST endpoints via the vendor MCP tools.
"""
import asyncio
import logging
import os
from typing import Any, Dict, Optional

from mcp.quantconnect_client import QuantConnectMCPClient

logger = logging.getLogger(__name__)


class RealQuantConnectIntegration:
    """Real QuantConnect integration using MCP client."""

    def __init__(self, user_id: Optional[str] = None, api_token: Optional[str] = None):
        """Initialize with QuantConnect credentials."""
        self.user_id = user_id or os.getenv("QUANTCONNECT_USER_ID")
        self.api_token = api_token or os.getenv("QUANTCONNECT_API_TOKEN")
        self.client = QuantConnectMCPClient(self.user_id, self.api_token)

        if not self.user_id or not self.api_token:
            logger.warning("QuantConnect credentials not available - will use fallback methods")

    async def create_and_backtest_strategy(
        self,
        strategy_name: str,
        algorithm_code: str
    ) -> Dict[str, Any]:
        """Create project and run backtest using real QuantConnect integration."""

        try:
            print(f"📡 Starting QuantConnect integration for: {strategy_name}")

            # Step 1: Create project
            print("🔨 Creating QuantConnect project...")
            project_result = await self.client.create_project(strategy_name, "Python")

            if not project_result["success"]:
                raise Exception("Failed to create project")

            project_id = project_result["project_id"]
            method_used = project_result.get("method", "unknown")

            print(f"✅ Project created: {project_id}")
            print(f"   Method: {method_used}")

            # Step 2: Upload algorithm
            print("📁 Uploading algorithm code...")
            file_result = await self.client.update_file(project_id, "main.py", algorithm_code)

            if not file_result["success"]:
                raise Exception("Failed to upload algorithm file")

            print(f"✅ Algorithm uploaded")
            print(f"   Method: {file_result.get('method', 'unknown')}")

            # Step 3: Create backtest
            print("🚀 Creating backtest...")
            backtest_result = await self.client.create_backtest(
                project_id,
                "latest",
                f"{strategy_name}_backtest"
            )

            if not backtest_result["success"]:
                raise Exception("Failed to create backtest")

            backtest_id = backtest_result["backtest_id"]
            print(f"✅ Backtest created: {backtest_id}")
            print(f"   Method: {backtest_result.get('method', 'unknown')}")

            # Step 4: Wait and get results
            print("⏳ Waiting for backtest completion...")

            # Wait a bit for backtest to process
            await asyncio.sleep(5)

            print("📊 Retrieving backtest results...")
            results = await self.client.read_backtest(project_id, backtest_id)

            if not results["success"]:
                raise Exception("Failed to retrieve backtest results")

            statistics = results["statistics"]
            print(f"✅ Results retrieved")
            print(f"   Method: {results.get('method', 'unknown')}")

            # Step 5: Format results
            final_result = {
                "success": True,
                "integration_type": "real_quantconnect_mcp",
                "project_id": project_id,
                "backtest_id": backtest_id,
                "statistics": statistics,
                "performance_metrics": {
                    "total_return": statistics.get("TotalReturn", 0.0),
                    "sharpe_ratio": statistics.get("SharpeRatio", 0.0),
                    "max_drawdown": statistics.get("MaxDrawdown", 0.0),
                    "total_trades": statistics.get("TotalTrades", 0),
                    "win_rate": statistics.get("WinRate", 0.0),
                    "starting_capital": statistics.get("StartingCapital", 100000),
                    "ending_capital": statistics.get("EndingCapital", 100000)
                },
                "execution_details": {
                    "project_creation_method": project_result.get("method"),
                    "file_upload_method": file_result.get("method"),
                    "backtest_creation_method": backtest_result.get("method"),
                    "results_retrieval_method": results.get("method"),
                    "credentials_available": bool(self.user_id and self.api_token),
                    "mcp_client_used": True
                }
            }

            # Display results
            print(f"\n📈 Backtest Results Summary:")
            print(f"   Total Return: {final_result['performance_metrics']['total_return']:.2%}")
            print(f"   Sharpe Ratio: {final_result['performance_metrics']['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {final_result['performance_metrics']['max_drawdown']:.2%}")
            print(f"   Total Trades: {final_result['performance_metrics']['total_trades']}")
            print(f"   Win Rate: {final_result['performance_metrics']['win_rate']:.2%}")

            print(f"\n🔧 Execution Methods:")
            for step, method in final_result['execution_details'].items():
                if 'method' in step:
                    print(f"   {step}: {method}")

            return final_result

        except Exception as e:
            logger.error(f"QuantConnect integration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "integration_type": "failed",
                "fallback_needed": True
            }

    async def test_integration(self) -> Dict[str, Any]:
        """Test the QuantConnect integration with a simple strategy."""

        test_strategy = '''
import pandas as pd

class TestAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)

        self.spy = self.AddEquity("SPY", Resolution.Daily)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Adjusted)

        # Simple moving average strategy
        self.sma_fast = self.SMA("SPY", 10, Resolution.Daily)
        self.sma_slow = self.SMA("SPY", 30, Resolution.Daily)

    def OnData(self, data):
        if not self.sma_fast.IsReady or not self.sma_slow.IsReady:
            return

        if self.sma_fast.Current.Value > self.sma_slow.Current.Value:
            if not self.Portfolio["SPY"].Invested:
                self.SetHoldings("SPY", 1.0)
        else:
            if self.Portfolio["SPY"].Invested:
                self.Liquidate("SPY")
'''

        return await self.create_and_backtest_strategy(
            "Test_SMA_Strategy",
            test_strategy
        )


async def main():
    """Test the real QuantConnect integration."""
    integration = RealQuantConnectIntegration()

    print("🧪 Testing Real QuantConnect Integration")
    print("=" * 60)

    result = await integration.test_integration()

    print("\n" + "=" * 60)
    if result["success"]:
        print("✅ Integration test SUCCESSFUL!")
        print(f"Integration type: {result['integration_type']}")

        if result.get('execution_details', {}).get('credentials_available'):
            print("🔑 Real QuantConnect credentials detected")
        else:
            print("⚠️ Using fallback methods (no credentials)")

    else:
        print("❌ Integration test FAILED!")
        print(f"Error: {result.get('error', 'Unknown error')}")

        if result.get('fallback_needed'):
            print("💡 Consider setting QUANTCONNECT_USER_ID and QUANTCONNECT_API_TOKEN")


if __name__ == "__main__":
    asyncio.run(main())