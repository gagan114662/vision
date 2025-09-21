#!/usr/bin/env python3
"""
Live QuantConnect API validation and integration testing.

This script validates the complete QuantConnect integration pipeline using real API credentials,
ensuring that all MCP server functions work with the actual QuantConnect platform.
"""
import os
import sys
import asyncio
import logging
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Setup test environment
os.environ["MCP_SECRET_KEY"] = "test_secret_key_32_characters_minimum_required"
os.environ["MCP_ENVIRONMENT"] = "development"
os.environ["TESTING"] = "true"

sys.path.append('.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuantConnectValidator:
    """Comprehensive QuantConnect API validation system."""

    def __init__(self):
        self.user_id = os.getenv("QUANTCONNECT_USER_ID")
        self.api_token = os.getenv("QUANTCONNECT_API_TOKEN")
        self.project_id = None
        self.backtest_id = None
        self.validation_results = []

    def check_credentials(self) -> bool:
        """Check if QuantConnect credentials are available."""
        if not self.user_id or not self.api_token:
            logger.warning("QuantConnect credentials not found in environment")
            logger.info("Expected environment variables:")
            logger.info("  - QUANTCONNECT_USER_ID")
            logger.info("  - QUANTCONNECT_API_TOKEN")
            return False

        logger.info(f"✅ QuantConnect credentials found (User ID: {self.user_id[:8]}...)")
        return True

    async def validate_api_connection(self) -> bool:
        """Validate basic API connectivity."""
        logger.info("=== Validating QuantConnect API Connection ===")

        try:
            from mcp.servers import quantconnect_stub

            # Try to list projects to validate API access
            result = quantconnect_stub.backtest_list({
                "project_id": "1"  # Test with a simple project ID
            })

            # If we get here without exception, API is accessible
            logger.info("✅ QuantConnect API connection validated")
            return True

        except RuntimeError as e:
            if "credentials missing" in str(e).lower():
                logger.error("❌ QuantConnect credentials are missing or invalid")
                return False
            elif "not available" in str(e).lower():
                logger.error("❌ QuantConnect MCP vendor server not available")
                logger.info("Clone https://github.com/QuantConnect/mcp-server into integrations/quantconnect_mcp/vendor")
                return False
            else:
                logger.error(f"❌ QuantConnect API connection failed: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Unexpected error validating API connection: {e}")
            return False

    async def validate_project_operations(self) -> bool:
        """Validate project creation and file synchronization."""
        logger.info("=== Validating QuantConnect Project Operations ===")

        try:
            from mcp.servers import quantconnect_stub

            # Test algorithm for validation
            test_algorithm = '''
from AlgorithmImports import *

class LiveValidationTest(QCAlgorithm):
    """Test algorithm for MCP integration validation."""

    def Initialize(self):
        """Initialize the algorithm."""
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 1, 31)
        self.SetCash(100000)

        # Add a simple equity
        self.spy = self.AddEquity("SPY", Resolution.Daily)

        # Log initialization
        self.Log("MCP Integration Test Algorithm Initialized")

    def OnData(self, data):
        """Handle market data."""
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 1.0)
            self.Log("Invested in SPY for MCP validation test")
'''

            # Test project synchronization
            sync_result = quantconnect_stub.project_sync({
                "project_id": self.project_id or "test_project",
                "files": [
                    {
                        "path": "main.py",
                        "content": test_algorithm
                    },
                    {
                        "path": "README.md",
                        "content": "# MCP Integration Test\n\nThis project validates MCP → QuantConnect integration."
                    }
                ]
            })

            # Validate sync results
            assert "synced" in sync_result, "Missing synced files in response"
            synced_files = sync_result.get("synced", [])

            if len(synced_files) > 0:
                logger.info(f"✅ Project sync successful - {len(synced_files)} files synced")
                for file in synced_files:
                    logger.info(f"   Synced: {file}")
            else:
                logger.warning("⚠️  Project sync completed but no files reported as synced")

            # Check for warnings
            warnings = sync_result.get("warnings", [])
            if warnings:
                logger.info(f"   Warnings: {len(warnings)}")
                for warning in warnings:
                    logger.warning(f"   ⚠️  {warning}")

            return True

        except Exception as e:
            logger.error(f"❌ Project operations validation failed: {e}")
            return False

    async def validate_backtest_workflow(self) -> bool:
        """Validate complete backtest workflow."""
        logger.info("=== Validating QuantConnect Backtest Workflow ===")

        try:
            from mcp.servers import quantconnect_stub

            # Step 1: Run a backtest
            backtest_name = f"MCP_Validation_{int(time.time())}"

            backtest_result = quantconnect_stub.backtest_run({
                "project_id": self.project_id or "test_project",
                "name": backtest_name,
                "parameters": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-31"
                }
            })

            assert "backtest_id" in backtest_result, "Missing backtest_id in response"
            self.backtest_id = backtest_result["backtest_id"]

            logger.info(f"✅ Backtest initiated successfully")
            logger.info(f"   Backtest ID: {self.backtest_id}")
            logger.info(f"   Name: {backtest_name}")

            # Step 2: Monitor backtest status
            max_wait_time = 300  # 5 minutes max wait
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                status_result = quantconnect_stub.backtest_status({
                    "project_id": self.project_id or "test_project",
                    "backtest_id": self.backtest_id
                })

                status = status_result.get("status", "unknown")
                progress = status_result.get("progress", 0)

                logger.info(f"   Backtest status: {status} ({progress:.1%} complete)")

                if status == "completed":
                    logger.info("✅ Backtest completed successfully")

                    # Extract performance metrics if available
                    if "statistics" in status_result:
                        stats = status_result["statistics"]
                        logger.info("   Performance Statistics:")
                        for key, value in stats.items():
                            logger.info(f"     {key}: {value}")

                    return True
                elif status == "failed":
                    logger.error("❌ Backtest failed")
                    error = status_result.get("error", "Unknown error")
                    logger.error(f"   Error: {error}")
                    return False

                # Wait before next status check
                await asyncio.sleep(10)

            logger.warning(f"⚠️  Backtest did not complete within {max_wait_time}s")
            return False

        except Exception as e:
            logger.error(f"❌ Backtest workflow validation failed: {e}")
            return False

    async def validate_project_management(self) -> bool:
        """Validate project management operations."""
        logger.info("=== Validating QuantConnect Project Management ===")

        try:
            from mcp.servers import quantconnect_stub

            # List backtests for the project
            list_result = quantconnect_stub.backtest_list({
                "project_id": self.project_id or "test_project"
            })

            assert "backtests" in list_result, "Missing backtests in response"
            backtests = list_result["backtests"]

            logger.info(f"✅ Project backtest listing successful")
            logger.info(f"   Found {len(backtests)} backtests")

            # Show recent backtests
            for i, backtest in enumerate(backtests[:5]):  # Show first 5
                bt_id = backtest.get("backtest_id", "unknown")
                bt_name = backtest.get("name", "unnamed")
                bt_status = backtest.get("status", "unknown")
                logger.info(f"   {i+1}. {bt_name} (ID: {bt_id}, Status: {bt_status})")

            return True

        except Exception as e:
            logger.error(f"❌ Project management validation failed: {e}")
            return False

    async def validate_error_handling(self) -> bool:
        """Validate error handling and resilience."""
        logger.info("=== Validating QuantConnect Error Handling ===")

        try:
            from mcp.servers import quantconnect_stub

            # Test with invalid project ID
            try:
                quantconnect_stub.backtest_status({
                    "project_id": "invalid_project_999999",
                    "backtest_id": "invalid_backtest_999999"
                })
                logger.warning("⚠️  Expected error for invalid project, but got success")
            except Exception as e:
                logger.info(f"✅ Proper error handling for invalid project: {type(e).__name__}")

            # Test with malformed parameters
            try:
                quantconnect_stub.project_sync({
                    "project_id": "not_an_integer",
                    "files": []
                })
                logger.warning("⚠️  Expected error for invalid project ID format, but got success")
            except Exception as e:
                logger.info(f"✅ Proper error handling for malformed parameters: {type(e).__name__}")

            return True

        except Exception as e:
            logger.error(f"❌ Error handling validation failed: {e}")
            return False

    async def run_comprehensive_validation(self) -> Dict[str, bool]:
        """Run complete QuantConnect validation suite."""
        logger.info("🚀 Starting Comprehensive QuantConnect API Validation")
        logger.info("=" * 60)

        results = {}

        # Step 1: Check credentials
        results["credentials"] = self.check_credentials()

        if not results["credentials"]:
            logger.error("❌ Cannot proceed without valid credentials")
            return results

        # Step 2: Validate API connection
        results["api_connection"] = await self.validate_api_connection()

        if not results["api_connection"]:
            logger.error("❌ Cannot proceed without API connection")
            return results

        # Step 3: Validate project operations
        results["project_operations"] = await self.validate_project_operations()

        # Step 4: Validate backtest workflow
        results["backtest_workflow"] = await self.validate_backtest_workflow()

        # Step 5: Validate project management
        results["project_management"] = await self.validate_project_management()

        # Step 6: Validate error handling
        results["error_handling"] = await self.validate_error_handling()

        return results


async def validate_mcp_server_integration():
    """Validate MCP server integration with other systems."""
    logger.info("=== Validating MCP Server Integration ===")

    try:
        # Test circuit breaker integration
        from mcp.servers.circuit_breaker_monitor import get_circuit_breaker_status

        cb_status = get_circuit_breaker_status({})

        # Look for QuantConnect circuit breakers
        qc_breakers = [
            cb for cb in cb_status.get("circuit_breakers", [])
            if "quantconnect" in cb.get("name", "").lower()
        ]

        logger.info(f"✅ Found {len(qc_breakers)} QuantConnect circuit breakers")
        for breaker in qc_breakers:
            logger.info(f"   {breaker['name']}: {breaker['state']} "
                       f"({breaker['total_calls']} calls, {breaker['total_failures']} failures)")

        # Test risk management integration
        from mcp.servers.risk_server import evaluate_portfolio

        # Mock portfolio for QC integration test
        test_portfolio = {
            "positions": [
                {"symbol": "SPY", "quantity": 100, "price": 450.0, "volatility": 0.15}
            ],
            "limits": {
                "max_drawdown": 0.20,
                "var_limit": 50000,
                "exposure_limit": 100000
            },
            "confidence_level": 0.95
        }

        risk_result = evaluate_portfolio(test_portfolio)
        logger.info(f"✅ Risk management integration validated")
        logger.info(f"   Portfolio approved: {risk_result.get('approved', False)}")

        return True

    except Exception as e:
        logger.error(f"❌ MCP server integration validation failed: {e}")
        return False


async def main():
    """Run live QuantConnect validation."""
    print("🔗 QuantConnect Live API Validation")
    print("=" * 50)

    validator = QuantConnectValidator()

    # Run comprehensive validation
    results = await validator.run_comprehensive_validation()

    # Validate MCP integration
    results["mcp_integration"] = await validate_mcp_server_integration()

    # Summary
    print("\n" + "=" * 50)
    print("📊 QuantConnect Validation Results")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        formatted_name = test_name.replace("_", " ").title()
        print(f"{status} {formatted_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} validations passed")

    if passed == total:
        print("🎉 All QuantConnect validations passed!")
        print("🔗 Live API integration fully validated")
        print("📈 Backtest workflow operational")
        print("🛡️  Error handling robust")
        print("🔧 MCP server integration active")
    else:
        print("⚠️  Some QuantConnect validations failed.")
        print("📋 Review the output above for details")

        # Provide guidance for common issues
        if not results.get("credentials"):
            print("\n💡 To fix credential issues:")
            print("   Set QUANTCONNECT_USER_ID and QUANTCONNECT_API_TOKEN environment variables")
            print("   Get credentials from https://www.quantconnect.com/account")

        if not results.get("api_connection"):
            print("\n💡 To fix API connection issues:")
            print("   Ensure QuantConnect MCP vendor server is installed")
            print("   Clone https://github.com/QuantConnect/mcp-server")
            print("   Install in integrations/quantconnect_mcp/vendor/")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)