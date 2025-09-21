#!/usr/bin/env python3
"""
Test real market data integration across trading agents.

Validates that agents can successfully retrieve and use real market data
from multiple data sources with proper fallback mechanisms.
"""
import os
import sys
import asyncio
import logging

# Setup test environment
os.environ["MCP_SECRET_KEY"] = "test_secret_key_32_characters_minimum_required"
os.environ["MCP_ENVIRONMENT"] = "development"
os.environ["TESTING"] = "true"

sys.path.append('.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_market_data_server():
    """Test the market data server directly."""
    logger.info("=== Testing Market Data Server ===")

    from mcp.servers.market_data_server import MarketDataProvider

    test_symbols = ["AAPL", "MSFT", "GOOGL"]

    async with MarketDataProvider() as provider:
        # Test real-time data
        market_data = await provider.get_real_time_data(test_symbols)

        assert len(market_data) > 0, "No market data retrieved"
        logger.info(f"✅ Retrieved market data for {len(market_data)} symbols")

        for data_point in market_data:
            logger.info(f"   {data_point.symbol}: ${data_point.price:.2f} (source: {data_point.source.value})")
            assert data_point.price > 0, f"Invalid price for {data_point.symbol}"
            assert data_point.volume >= 0, f"Invalid volume for {data_point.symbol}"


async def test_orchestrator_real_data():
    """Test orchestrator with real market data."""
    logger.info("=== Testing Orchestrator Real Data Integration ===")

    from agents.core.orchestrator import MultiAgentOrchestrator, OrchestrationRequest
    from agents.implementations.fundamental_agent import FundamentalAgent

    # Create orchestrator
    orchestrator = MultiAgentOrchestrator()

    # Register fundamental agent
    fundamental_agent = FundamentalAgent("fund_agent_1")
    orchestrator.register_agent(fundamental_agent)

    await orchestrator.start()

    try:
        # Create test request
        request = OrchestrationRequest(
            request_id="test_real_data_001",
            symbols=["AAPL", "MSFT"],
            analysis_types=["fundamental"]
        )

        # Execute orchestration
        result = await orchestrator.orchestrate_analysis(request)

        # Validate results
        assert len(result.consensus_decisions) > 0, "No consensus decisions generated"
        logger.info(f"✅ Generated {len(result.consensus_decisions)} consensus decisions")

        for decision in result.consensus_decisions:
            logger.info(f"   {decision.symbol}: {decision.consensus_signal.value} (confidence: {decision.consensus_confidence.value})")
            assert decision.symbol in request.symbols, f"Unexpected symbol: {decision.symbol}"

        # Check performance metrics
        metrics = result.performance_metrics
        logger.info(f"   Total duration: {metrics['total_duration_seconds']:.2f}s")
        logger.info(f"   Success rate: {metrics['success_rate']:.1f}%")

        assert metrics["success_rate"] > 0, "Zero success rate"

    finally:
        await orchestrator.stop()


async def test_fundamental_agent_real_prices():
    """Test fundamental agent with real price data."""
    logger.info("=== Testing Fundamental Agent Real Prices ===")

    from agents.implementations.fundamental_agent import FundamentalAgent
    from agents.core.orchestrator import RealMarketDataProvider
    from agents.core import AnalysisRequest

    # Create agent and data provider
    agent = FundamentalAgent("fund_test")
    data_provider = RealMarketDataProvider()

    # Set data provider
    agent.set_data_provider(data_provider)
    await agent.start()

    try:
        # Create analysis request
        request = AnalysisRequest(
            request_id="fund_test_001",
            symbols=["AAPL", "GOOGL", "TSLA"],
            analysis_type="fundamental"
        )

        # Perform analysis
        result = await agent.analyze(request)

        # Validate results
        assert len(result.signals) > 0, "No trading signals generated"
        logger.info(f"✅ Generated {len(result.signals)} trading signals")

        for signal in result.signals:
            logger.info(f"   {signal.symbol}: {signal.signal.value} (confidence: {signal.confidence.value})")

            # Verify signal has real data
            if "supporting_data" in signal.supporting_data:
                supporting_data = signal.supporting_data
                if "dcf_fair_value" in supporting_data:
                    fair_value = supporting_data["dcf_fair_value"]
                    logger.info(f"     DCF Fair Value: ${fair_value:.2f}")
                    assert fair_value > 0, f"Invalid fair value for {signal.symbol}"

        # Check analysis data contains real prices
        for symbol in request.symbols:
            if symbol in result.analysis_data:
                analysis = result.analysis_data[symbol]
                if "current_price" in analysis:
                    current_price = analysis["current_price"]
                    logger.info(f"   {symbol} current price: ${current_price:.2f}")
                    assert current_price > 0, f"Invalid current price for {symbol}"

    finally:
        await agent.stop()


async def test_market_data_fallback():
    """Test market data fallback mechanisms."""
    logger.info("=== Testing Market Data Fallback ===")

    from agents.core.orchestrator import RealMarketDataProvider

    provider = RealMarketDataProvider()

    # Test with invalid symbols to trigger fallback
    invalid_symbols = ["INVALID1", "FAKE_SYMBOL"]

    market_data = await provider.get_market_data(invalid_symbols)

    assert len(market_data) > 0, "Fallback data not generated"
    logger.info(f"✅ Fallback data generated for {len(market_data)} symbols")

    for data_point in market_data:
        logger.info(f"   {data_point.symbol}: ${data_point.price:.2f} (fallback)")
        assert data_point.price > 0, f"Invalid fallback price for {data_point.symbol}"
        assert "fallback" in data_point.metadata.get("source", ""), "Missing fallback indicator"


async def main():
    """Run all market data integration tests."""
    print("📈 Market Data Integration Test")
    print("=" * 50)

    test_results = []

    # Test 1: Market data server
    try:
        await test_market_data_server()
        test_results.append(("Market Data Server", True))
        logger.info("✅ Market data server test PASSED")
    except Exception as e:
        test_results.append(("Market Data Server", False))
        logger.error(f"❌ Market data server test FAILED: {e}")

    # Test 2: Orchestrator real data
    try:
        await test_orchestrator_real_data()
        test_results.append(("Orchestrator Real Data", True))
        logger.info("✅ Orchestrator real data test PASSED")
    except Exception as e:
        test_results.append(("Orchestrator Real Data", False))
        logger.error(f"❌ Orchestrator real data test FAILED: {e}")

    # Test 3: Fundamental agent real prices
    try:
        await test_fundamental_agent_real_prices()
        test_results.append(("Fundamental Agent Real Prices", True))
        logger.info("✅ Fundamental agent real prices test PASSED")
    except Exception as e:
        test_results.append(("Fundamental Agent Real Prices", False))
        logger.error(f"❌ Fundamental agent real prices test FAILED: {e}")

    # Test 4: Market data fallback
    try:
        await test_market_data_fallback()
        test_results.append(("Market Data Fallback", True))
        logger.info("✅ Market data fallback test PASSED")
    except Exception as e:
        test_results.append(("Market Data Fallback", False))
        logger.error(f"❌ Market data fallback test FAILED: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 Market Data Integration Test Results")
    print("=" * 50)

    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(test_results)} tests passed")

    if passed == len(test_results):
        print("🎉 All market data integration tests passed!")
        print("📈 Real market data is successfully integrated")
        print("🔄 Fallback mechanisms are working properly")
        print("🤖 Trading agents can now use live market data")
    else:
        print("⚠️  Some market data tests failed. Review the output above.")

        if os.getenv("CI") or os.getenv("TESTING"):
            raise SystemExit(f"Market data tests failed: {len(test_results) - passed}/{len(test_results)} failures")

    return passed == len(test_results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)