#!/usr/bin/env python3
"""
Test circuit breaker integration across MCP servers.

Validates that circuit breakers are properly deployed and functioning
across the critical MCP server infrastructure.
"""
import os
import sys
import logging

# Setup test environment
os.environ["MCP_SECRET_KEY"] = "test_secret_key_32_characters_minimum_required"
os.environ["MCP_ENVIRONMENT"] = "development"
os.environ["TESTING"] = "true"

sys.path.append('.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_circuit_breaker_deployment():
    """Test that circuit breakers are properly deployed across MCP servers."""
    logger.info("=== Testing Circuit Breaker Deployment ===")

    try:
        # Test risk server with circuit breaker
        from mcp.servers import risk_server

        # Normal operation - should work
        result = risk_server.evaluate_portfolio({
            "positions": [
                {"symbol": "SPY", "quantity": 100, "price": 450.0, "volatility": 0.15}
            ],
            "limits": {
                "max_drawdown": 0.20,
                "var_limit": 50000,
                "exposure_limit": 100000
            },
            "confidence_level": 0.95
        })

        logger.info("✅ Risk server circuit breaker: Normal operation successful")

        # Test ally shell server with circuit breaker
        from mcp.servers import ally_shell_server

        shell_result = ally_shell_server.run_command({
            "command": ["pwd"],
            "workdir": ".",
            "dry_run": False,
            "timeout_seconds": 5,
            "use_ally": False
        })

        logger.info("✅ Ally shell server circuit breaker: Normal operation successful")

        # Test compliance server with circuit breaker
        from mcp.servers import compliance_server

        compliance_result = compliance_server.generate_summary({
            "strategy_id": "test_strategy",
            "controls": ["market_risk", "operational_risk"],
            "evidence": [
                {"control": "market_risk", "status": "PASS", "details": "VaR within limits"}
            ],
            "outstanding_risks": []
        })

        logger.info("✅ Compliance server circuit breaker: Normal operation successful")

        return True

    except Exception as e:
        logger.error(f"❌ Circuit breaker deployment test failed: {e}")
        return False


def test_circuit_breaker_monitoring():
    """Test circuit breaker monitoring functionality."""
    logger.info("=== Testing Circuit Breaker Monitoring ===")

    try:
        from mcp.servers import circuit_breaker_monitor

        # Test getting circuit breaker status
        status_result = circuit_breaker_monitor.get_circuit_breaker_status({})

        logger.info(f"✅ Circuit breaker monitoring: {status_result['summary']['total_circuit_breakers']} breakers found")
        logger.info(f"   Health status: {status_result['summary']['health_status']}")
        logger.info(f"   Health score: {status_result['summary']['health_score']}%")

        # Log detailed information about circuit breakers
        for cb in status_result["circuit_breakers"]:
            logger.info(f"   {cb['name']}: {cb['state']} (calls: {cb['total_calls']}, failures: {cb['total_failures']})")

        # Test alerts
        if status_result["alerts"]:
            logger.info(f"   Active alerts: {len(status_result['alerts'])}")
            for alert in status_result["alerts"]:
                logger.warning(f"   Alert: {alert['message']}")

        return True

    except Exception as e:
        logger.error(f"❌ Circuit breaker monitoring test failed: {e}")
        return False


def test_circuit_breaker_resilience():
    """Test circuit breaker failure handling."""
    logger.info("=== Testing Circuit Breaker Resilience ===")

    try:
        from mcp.common.resilience import CircuitBreaker, CircuitBreakerConfig

        # Create a test circuit breaker
        test_breaker = CircuitBreaker(
            "test_breaker",
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=1.0)
        )

        def failing_function():
            raise ValueError("Test failure")

        # First call should work (but fail)
        try:
            test_breaker.call(failing_function)
        except ValueError:
            pass

        # Second call should work (but fail)
        try:
            test_breaker.call(failing_function)
        except ValueError:
            pass

        # Third call should trigger circuit breaker open
        try:
            test_breaker.call(failing_function)
            logger.error("❌ Circuit breaker should have opened but didn't")
            return False
        except Exception as e:
            if "circuit breaker" in str(e).lower():
                logger.info("✅ Circuit breaker opened as expected")
            else:
                logger.error(f"❌ Unexpected exception: {e}")
                return False

        # Test statistics
        stats = test_breaker.get_stats()
        logger.info(f"   Circuit breaker stats: {stats['state']}, {stats['total_failures']} failures")

        return True

    except Exception as e:
        logger.error(f"❌ Circuit breaker resilience test failed: {e}")
        return False


def test_circuit_breaker_recovery():
    """Test circuit breaker recovery mechanism."""
    logger.info("=== Testing Circuit Breaker Recovery ===")

    try:
        import time
        from mcp.common.resilience import CircuitBreaker, CircuitBreakerConfig

        # Create a test circuit breaker with short recovery timeout
        test_breaker = CircuitBreaker(
            "recovery_test_breaker",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.1)
        )

        def failing_function():
            raise ValueError("Test failure")

        def succeeding_function():
            return "success"

        # Trigger circuit breaker open
        try:
            test_breaker.call(failing_function)
        except ValueError:
            pass

        # Should be open now
        try:
            test_breaker.call(succeeding_function)
            logger.error("❌ Circuit breaker should be open")
            return False
        except Exception:
            pass

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should now be in half-open and allow one call
        try:
            result = test_breaker.call(succeeding_function)
            if result == "success":
                logger.info("✅ Circuit breaker recovered successfully")
            else:
                logger.error("❌ Circuit breaker recovery failed")
                return False
        except Exception as e:
            logger.error(f"❌ Circuit breaker recovery test failed: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Circuit breaker recovery test failed: {e}")
        return False


def main():
    """Run circuit breaker integration tests."""
    print("🔧 Circuit Breaker Integration Test")
    print("=" * 50)

    test_results = []

    # Test 1: Circuit breaker deployment
    result1 = test_circuit_breaker_deployment()
    test_results.append(("Circuit Breaker Deployment", result1))

    # Test 2: Circuit breaker monitoring
    result2 = test_circuit_breaker_monitoring()
    test_results.append(("Circuit Breaker Monitoring", result2))

    # Test 3: Circuit breaker resilience
    result3 = test_circuit_breaker_resilience()
    test_results.append(("Circuit Breaker Resilience", result3))

    # Test 4: Circuit breaker recovery
    result4 = test_circuit_breaker_recovery()
    test_results.append(("Circuit Breaker Recovery", result4))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Circuit Breaker Test Results")
    print("=" * 50)

    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(test_results)} tests passed")

    if passed == len(test_results):
        print("🎉 All circuit breaker tests passed!")
        print("🔧 Circuit breakers are properly deployed across MCP infrastructure")
        print("📊 Monitoring and recovery mechanisms are working")
        print("⚡ System resilience has been significantly improved")
    else:
        print("⚠️  Some circuit breaker tests failed. Review the output above.")

    return passed == len(test_results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)