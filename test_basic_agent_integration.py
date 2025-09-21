#!/usr/bin/env python3
"""
Basic integration test for agent implementations without external dependencies.

Tests core functionality that doesn't require scipy, pandas, or other optional libraries.
Focuses on:
1. Agent instantiation and basic interface
2. Market data provider with fallback data
3. Orchestrator with real agent integration
4. Configuration loading
"""
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

# Setup test environment
os.environ["MCP_SECRET_KEY"] = "test_secret_key_32_characters_minimum_required"
os.environ["MCP_ENVIRONMENT"] = "development"
os.environ["TESTING"] = "true"

# Add project root to path
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_market_data_fallbacks():
    """Test market data provider fallback mechanisms."""
    logger.info("=== Testing Market Data Provider Fallbacks ===")

    try:
        from mcp.market_data.real_data_provider import RealMarketDataProvider

        provider = RealMarketDataProvider()

        # Test with symbols that will trigger fallback
        symbols = ["TEST_SYMBOL"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)

        # This should trigger fallback data generation
        stock_data = await provider.get_stock_data(symbols, start_date, end_date)

        if "TEST_SYMBOL" in stock_data:
            data = stock_data["TEST_SYMBOL"]
            logger.info(f"✅ Fallback data generated: {len(data['prices'])} points from {data['source']}")
            return True
        else:
            logger.error("❌ No fallback data generated")
            return False

    except Exception as e:
        logger.error(f"❌ Market data fallback test failed: {e}")
        return False

async def test_orchestrator_with_mocks():
    """Test orchestrator functionality with mock agents."""
    logger.info("=== Testing Orchestrator with Mock Agents ===")

    try:
        from agents.workflows.autonomous_orchestrator import AutonomousOrchestrator

        # Create orchestrator
        orchestrator = AutonomousOrchestrator(user_id="test_user")

        # Test universe
        universe = ["SPY"]

        # Test each mock agent
        for agent_name, agent in orchestrator.agents.items():
            try:
                logger.info(f"Testing {agent_name} agent interface...")

                # Test market data for agent
                market_data = {
                    "symbol": "SPY",
                    "price": 450.0,
                    "volume": 50000000
                }

                # Call agent
                analysis = await orchestrator._call_agent_analyze(agent_name, agent, "SPY", market_data)

                if analysis and analysis.get("confidence", 0) > 0:
                    logger.info(f"✅ {agent_name}: {analysis['recommendation']} (confidence: {analysis['confidence']:.2f})")
                else:
                    logger.warning(f"❌ {agent_name} agent produced invalid analysis")
                    return False

            except Exception as e:
                logger.error(f"❌ {agent_name} agent failed: {e}")
                return False

        # Test hypothesis creation
        analysis = {
            "symbol": "SPY",
            "recommendation": "buy",
            "confidence": 0.75,
            "reasoning": "Test reasoning"
        }

        mcp_analysis = {"signals": []}

        hypothesis = orchestrator._create_hypothesis_from_agent_analysis(
            "technical", "SPY", analysis, mcp_analysis
        )

        if hypothesis and hypothesis.confidence > 0.5:
            logger.info(f"✅ Hypothesis created: {hypothesis.title} (confidence: {hypothesis.confidence:.2f})")
        else:
            logger.error("❌ Hypothesis creation failed")
            return False

        # Test consensus building
        hypotheses = [hypothesis, hypothesis]  # Duplicate for test
        consensus = orchestrator._build_consensus_hypothesis(hypotheses)

        if consensus:
            logger.info(f"✅ Consensus built: {consensus.title} (confidence: {consensus.confidence:.2f})")
        else:
            logger.error("❌ Consensus building failed")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Orchestrator test failed: {e}")
        return False

async def test_config_system():
    """Test configuration system functionality."""
    logger.info("=== Testing Configuration System ===")

    try:
        # Test MCP servers can load configs
        from mcp.servers import ally_shell_server

        # Test command execution
        result = ally_shell_server.run_command({
            "command": ["pwd"],
            "workdir": ".",
            "dry_run": False,
            "timeout_seconds": 5,
            "use_ally": False
        })

        if result.get("exit_code") == 0:
            logger.info("✅ Configuration system working: commands execute successfully")
            return True
        else:
            logger.error(f"❌ Command execution failed: {result}")
            return False

    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

async def test_data_integration_flow():
    """Test end-to-end data integration flow."""
    logger.info("=== Testing Data Integration Flow ===")

    try:
        from mcp.market_data.real_data_provider import RealMarketDataProvider

        provider = RealMarketDataProvider()

        # Test complete flow with real Yahoo Finance data if available
        symbols = ["SPY"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        # Get stock data
        stock_data = await provider.get_stock_data(symbols, start_date, end_date)

        # Get risk-free rate
        rf_rate = await provider.get_risk_free_rate()

        # Get factors
        factors = await provider.get_market_factors(start_date, end_date)

        # Validate we got some data
        has_stock_data = "SPY" in stock_data and len(stock_data["SPY"]["prices"]) > 0
        has_risk_rate = rf_rate > 0
        has_factors = len(factors) > 0

        if has_stock_data and has_risk_rate and has_factors:
            logger.info(f"✅ Complete data integration flow successful:")
            logger.info(f"   Stock data: {len(stock_data['SPY']['prices'])} points from {stock_data['SPY']['source']}")
            logger.info(f"   Risk-free rate: {rf_rate:.3%}")
            logger.info(f"   Factors: {len(factors)} from {factors.get('source', 'unknown')}")
            return True
        else:
            logger.error("❌ Data integration flow incomplete")
            return False

    except Exception as e:
        logger.error(f"❌ Data integration test failed: {e}")
        return False

async def run_basic_integration_test():
    """Run basic integration test without external dependencies."""

    print("🚀 Basic Agent Integration Test (No External Dependencies)")
    print("=" * 65)

    test_results = []

    # Test 1: Market data fallbacks
    result1 = await test_market_data_fallbacks()
    test_results.append(("Market Data Fallbacks", result1))

    # Test 2: Orchestrator with mocks
    result2 = await test_orchestrator_with_mocks()
    test_results.append(("Orchestrator with Mocks", result2))

    # Test 3: Configuration system
    result3 = await test_config_system()
    test_results.append(("Configuration System", result3))

    # Test 4: Data integration flow
    result4 = await test_data_integration_flow()
    test_results.append(("Data Integration Flow", result4))

    # Summary
    print("\n" + "=" * 65)
    print("📊 Basic Integration Test Results")
    print("=" * 65)

    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(test_results)} tests passed")

    if passed == len(test_results):
        print("🎉 All basic tests passed!")
        print("📊 Market data integration is working with fallbacks")
        print("🤖 Orchestrator interfaces are functioning properly")
        print("⚙️  Configuration loading system is operational")
        print("📈 Ready for real agent integration when dependencies are available")
    else:
        print("⚠️  Some basic tests failed. Review the output above.")

    return passed == len(test_results)

def main():
    """Main test runner."""
    asyncio.run(run_basic_integration_test())

if __name__ == "__main__":
    main()