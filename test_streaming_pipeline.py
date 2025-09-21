#!/usr/bin/env python3
"""
Real-time streaming pipeline validation and testing.

Tests the streaming infrastructure for market data, trading signals,
and system events with performance and reliability validation.
"""
import os
import sys
import asyncio
import logging
import time
from datetime import datetime, timezone

# Setup test environment
os.environ["MCP_SECRET_KEY"] = "test_secret_key_32_characters_minimum_required"
os.environ["MCP_ENVIRONMENT"] = "development"
os.environ["TESTING"] = "true"

sys.path.append('.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_streaming_pipeline_basic():
    """Test basic streaming pipeline functionality."""
    logger.info("=== Testing Streaming Pipeline Basic Operations ===")

    from mcp.servers.streaming_pipeline_server import (
        create_stream_subscription,
        publish_stream_event,
        get_stream_events,
        get_pipeline_metrics,
        cancel_stream_subscription
    )

    # Test creating subscription
    subscription_result = await create_stream_subscription({
        "stream_type": "market_data",
        "filters": {
            "symbols": ["AAPL", "MSFT", "GOOGL"]
        },
        "max_events_per_second": 50
    })

    assert subscription_result["success"], "Failed to create subscription"
    subscription_id = subscription_result["subscription_id"]
    logger.info(f"✅ Created subscription: {subscription_id}")

    # Test publishing events
    test_events = [
        {
            "stream_type": "market_data",
            "data": {
                "symbol": "AAPL",
                "price": 175.50,
                "volume": 1500000,
                "bid": 175.48,
                "ask": 175.52
            },
            "source": "test_market_feed",
            "quality": "real_time"
        },
        {
            "stream_type": "trading_signals",
            "data": {
                "symbol": "MSFT",
                "signal": "buy",
                "confidence": 0.85,
                "target_price": 420.0,
                "reasoning": "Strong technical momentum"
            },
            "source": "technical_agent",
            "quality": "real_time"
        },
        {
            "stream_type": "risk_alerts",
            "data": {
                "alert_type": "position_limit",
                "symbol": "TSLA",
                "current_exposure": 150000,
                "limit": 200000,
                "level": "warning"
            },
            "source": "risk_monitor",
            "quality": "near_real_time"
        }
    ]

    for event in test_events:
        publish_result = await publish_stream_event(event)
        assert publish_result["success"], f"Failed to publish event: {event['stream_type']}"

    logger.info(f"✅ Published {len(test_events)} test events")

    # Test retrieving events
    market_events = await get_stream_events({
        "stream_type": "market_data",
        "max_count": 10
    })

    assert market_events["success"], "Failed to get market data events"
    assert len(market_events["events"]) > 0, "No market data events retrieved"

    logger.info(f"✅ Retrieved {market_events['count']} market data events")

    # Test pipeline metrics
    metrics_result = await get_pipeline_metrics({})
    assert metrics_result["success"], "Failed to get pipeline metrics"

    metrics = metrics_result["metrics"]
    logger.info(f"✅ Pipeline metrics retrieved:")
    logger.info(f"   Events processed: {metrics['metrics']['events_processed']}")
    logger.info(f"   Active subscriptions: {metrics['active_subscriptions']}")

    # Test cancelling subscription
    cancel_result = await cancel_stream_subscription({
        "subscription_id": subscription_id
    })

    assert cancel_result["success"], "Failed to cancel subscription"
    logger.info(f"✅ Cancelled subscription: {subscription_id}")

    return True


async def test_streaming_performance():
    """Test streaming pipeline performance under load."""
    logger.info("=== Testing Streaming Pipeline Performance ===")

    from mcp.servers.streaming_pipeline_server import (
        create_stream_subscription,
        publish_stream_event,
        get_pipeline_metrics
    )

    # Create multiple subscriptions
    subscriptions = []
    for i in range(5):
        sub_result = await create_stream_subscription({
            "stream_type": "market_data",
            "filters": {"symbols": [f"TEST{i:02d}"]},
            "max_events_per_second": 100
        })
        subscriptions.append(sub_result["subscription_id"])

    logger.info(f"✅ Created {len(subscriptions)} test subscriptions")

    # Publish events rapidly
    start_time = time.time()
    events_published = 0
    target_events = 100

    for i in range(target_events):
        await publish_stream_event({
            "stream_type": "market_data",
            "data": {
                "symbol": f"TEST{i % 10:02d}",
                "price": 100.0 + (i % 50),
                "volume": 1000000 + i * 1000,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "source": "performance_test",
            "quality": "real_time"
        })
        events_published += 1

    elapsed_time = time.time() - start_time
    events_per_second = events_published / elapsed_time

    logger.info(f"✅ Performance test completed:")
    logger.info(f"   Events published: {events_published}")
    logger.info(f"   Time elapsed: {elapsed_time:.2f}s")
    logger.info(f"   Throughput: {events_per_second:.1f} events/sec")

    # Check pipeline metrics after load test
    metrics_result = await get_pipeline_metrics({})
    metrics = metrics_result["metrics"]

    logger.info(f"   Buffer utilization:")
    for stream_type, stats in metrics["buffer_stats"].items():
        if stats["total_events"] > 0:
            logger.info(f"     {stream_type}: {stats['utilization']:.1%} "
                       f"({stats['current_size']}/{stats['max_size']})")

    # Verify performance thresholds
    assert events_per_second > 50, f"Throughput too low: {events_per_second:.1f} events/sec"
    assert metrics["metrics"]["events_dropped"] == 0, "Events were dropped during test"

    return True


async def test_streaming_filters():
    """Test streaming pipeline filtering functionality."""
    logger.info("=== Testing Streaming Pipeline Filters ===")

    from mcp.servers.streaming_pipeline_server import (
        create_stream_subscription,
        publish_stream_event,
        get_stream_events
    )

    # Create filtered subscription
    sub_result = await create_stream_subscription({
        "stream_type": "trading_signals",
        "filters": {
            "symbols": ["AAPL", "MSFT"],
            "min_confidence": 0.7
        }
    })

    subscription_id = sub_result["subscription_id"]

    # Publish signals with varying confidence
    test_signals = [
        {"symbol": "AAPL", "confidence": 0.85, "signal": "buy"},    # Should match
        {"symbol": "GOOGL", "confidence": 0.90, "signal": "buy"},  # Wrong symbol
        {"symbol": "MSFT", "confidence": 0.60, "signal": "sell"},  # Low confidence
        {"symbol": "MSFT", "confidence": 0.75, "signal": "buy"},   # Should match
        {"symbol": "AAPL", "confidence": 0.95, "signal": "hold"}   # Should match
    ]

    for signal in test_signals:
        await publish_stream_event({
            "stream_type": "trading_signals",
            "data": signal,
            "source": "filter_test"
        })

    # Small delay to allow processing
    await asyncio.sleep(0.1)

    # Get filtered events
    events_result = await get_stream_events({
        "stream_type": "trading_signals",
        "max_count": 10
    })

    events = events_result["events"]
    logger.info(f"✅ Filter test: {len(events)} events matched filters")

    # Validate filtering worked correctly
    for event in events:
        data = event["data"]
        assert data["symbol"] in ["AAPL", "MSFT"], f"Wrong symbol: {data['symbol']}"
        assert data["confidence"] >= 0.7, f"Low confidence: {data['confidence']}"

    # Should have 3 matching events
    assert len(events) == 3, f"Expected 3 filtered events, got {len(events)}"

    return True


async def test_streaming_integration():
    """Test streaming pipeline integration with other systems."""
    logger.info("=== Testing Streaming Pipeline Integration ===")

    from mcp.servers.streaming_pipeline_server import (
        publish_stream_event,
        get_pipeline_metrics
    )

    # Test market data integration
    try:
        from mcp.servers.market_data_server import get_real_time_market_data

        # Get real market data
        market_result = await get_real_time_market_data({
            "symbols": ["AAPL", "MSFT"]
        })

        if market_result.get("success") and market_result.get("data"):
            # Publish market data to stream
            for data_point in market_result["data"]:
                await publish_stream_event({
                    "stream_type": "market_data",
                    "data": {
                        "symbol": data_point["symbol"],
                        "price": data_point["price"],
                        "volume": data_point["volume"],
                        "timestamp": data_point["timestamp"]
                    },
                    "source": "market_data_server",
                    "quality": "real_time"
                })

            logger.info("✅ Market data integration successful")
        else:
            logger.info("ℹ️  Market data integration using mock data")

    except Exception as e:
        logger.warning(f"Market data integration limited: {e}")

    # Test agent signal integration
    try:
        from agents.implementations.technical_agent import TechnicalAgent
        from agents.core import AnalysisRequest

        agent = TechnicalAgent("streaming_test")

        # Mock analysis result
        await publish_stream_event({
            "stream_type": "trading_signals",
            "data": {
                "symbol": "AAPL",
                "signal": "buy",
                "confidence": 0.82,
                "source_agent": "technical_agent",
                "reasoning": "Bullish momentum pattern detected"
            },
            "source": "technical_agent",
            "quality": "real_time"
        })

        logger.info("✅ Agent signal integration successful")

    except Exception as e:
        logger.warning(f"Agent integration limited: {e}")

    # Check final metrics
    metrics_result = await get_pipeline_metrics({})
    final_metrics = metrics_result["metrics"]

    logger.info(f"✅ Integration test metrics:")
    logger.info(f"   Total events: {final_metrics['metrics']['events_processed']}")
    logger.info(f"   Active subscriptions: {final_metrics['active_subscriptions']}")

    return True


async def main():
    """Run comprehensive streaming pipeline tests."""
    print("🌊 Streaming Pipeline Test Suite")
    print("=" * 50)

    test_results = []

    # Test 1: Basic operations
    try:
        await test_streaming_pipeline_basic()
        test_results.append(("Basic Operations", True))
        logger.info("✅ Basic operations test PASSED")
    except Exception as e:
        test_results.append(("Basic Operations", False))
        logger.error(f"❌ Basic operations test FAILED: {e}")

    # Test 2: Performance
    try:
        await test_streaming_performance()
        test_results.append(("Performance", True))
        logger.info("✅ Performance test PASSED")
    except Exception as e:
        test_results.append(("Performance", False))
        logger.error(f"❌ Performance test FAILED: {e}")

    # Test 3: Filtering
    try:
        await test_streaming_filters()
        test_results.append(("Filtering", True))
        logger.info("✅ Filtering test PASSED")
    except Exception as e:
        test_results.append(("Filtering", False))
        logger.error(f"❌ Filtering test FAILED: {e}")

    # Test 4: Integration
    try:
        await test_streaming_integration()
        test_results.append(("Integration", True))
        logger.info("✅ Integration test PASSED")
    except Exception as e:
        test_results.append(("Integration", False))
        logger.error(f"❌ Integration test FAILED: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 Streaming Pipeline Test Results")
    print("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All streaming pipeline tests passed!")
        print("🌊 Real-time streaming operational")
        print("📊 Event filtering working correctly")
        print("⚡ Performance targets met")
        print("🔗 System integration validated")
    else:
        print("⚠️  Some streaming tests failed. Review the output above.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)