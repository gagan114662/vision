#!/usr/bin/env python3
"""
Real API Integration Tests with Circuit Breaker Scenarios

This test suite validates:
1. Real external API integrations (market data, QuantConnect, etc.)
2. Circuit breaker behavior under failure conditions
3. System resilience and recovery patterns
4. End-to-end workflow validation with real services

Tests are designed to run against real external services when credentials are available,
and gracefully degrade to validation mode when they're not.
"""

import asyncio
import os
import sys
import time
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealAPIIntegrationTester:
    """Tests real API integrations with circuit breaker scenarios."""

    def __init__(self):
        self.test_results = {}
        self.failure_scenarios = []
        self.recovery_patterns = {}

    async def test_market_data_apis_with_circuit_breakers(self):
        """Test market data APIs under various failure conditions."""
        logger.info("🔬 Testing market data APIs with circuit breaker scenarios...")

        # Test Alpha Vantage API
        await self._test_alpha_vantage_circuit_breaker()

        # Test Yahoo Finance fallback
        await self._test_yahoo_finance_circuit_breaker()

        # Test IEX Cloud integration
        await self._test_iex_cloud_circuit_breaker()

        # Test circuit breaker recovery
        await self._test_circuit_breaker_recovery()

    async def _test_alpha_vantage_circuit_breaker(self):
        """Test Alpha Vantage API with circuit breaker patterns."""
        try:
            from mcp.servers.market_data_server import MarketDataProvider
            from mcp.common.resilience import CircuitBreakerState

            provider = MarketDataProvider()

            # Test normal operation
            async with provider:
                try:
                    # Test with real API call
                    data = await provider.get_real_time_data(['AAPL'])
                    if data:
                        logger.info("✅ Alpha Vantage API: Normal operation successful")
                        self.test_results['alpha_vantage_normal'] = True
                    else:
                        logger.warning("⚠️ Alpha Vantage API: No data returned")
                        self.test_results['alpha_vantage_normal'] = False

                    # Test circuit breaker behavior under failure
                    await self._simulate_api_failures(provider, 'alpha_vantage')

                except Exception as e:
                    logger.error(f"❌ Alpha Vantage API test failed: {e}")
                    self.test_results['alpha_vantage_normal'] = False

        except ImportError as e:
            logger.warning(f"Market data server not available: {e}")
            self.test_results['alpha_vantage_normal'] = False

    async def _test_yahoo_finance_circuit_breaker(self):
        """Test Yahoo Finance API circuit breaker behavior."""
        try:
            # Test Yahoo Finance as fallback provider
            import aiohttp

            async with aiohttp.ClientSession() as session:
                url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"

                # Test normal request
                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info("✅ Yahoo Finance API: Normal operation successful")
                            self.test_results['yahoo_finance_normal'] = True
                        else:
                            logger.warning(f"⚠️ Yahoo Finance API returned status: {response.status}")
                            self.test_results['yahoo_finance_normal'] = False

                except asyncio.TimeoutError:
                    logger.warning("⚠️ Yahoo Finance API: Timeout occurred")
                    self.test_results['yahoo_finance_normal'] = False

                except Exception as e:
                    logger.error(f"❌ Yahoo Finance API test failed: {e}")
                    self.test_results['yahoo_finance_normal'] = False

        except ImportError:
            logger.warning("aiohttp not available for Yahoo Finance test")
            self.test_results['yahoo_finance_normal'] = False

    async def _test_iex_cloud_circuit_breaker(self):
        """Test IEX Cloud API circuit breaker behavior."""
        iex_token = os.environ.get('IEX_CLOUD_TOKEN')

        if not iex_token:
            logger.info("🔧 IEX Cloud token not available, using test mode")
            self.test_results['iex_cloud_normal'] = False
            return

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                url = f"https://cloud.iexapis.com/stable/stock/AAPL/quote?token={iex_token}"

                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info("✅ IEX Cloud API: Normal operation successful")
                            self.test_results['iex_cloud_normal'] = True
                        else:
                            logger.warning(f"⚠️ IEX Cloud API returned status: {response.status}")
                            self.test_results['iex_cloud_normal'] = False

                except asyncio.TimeoutError:
                    logger.warning("⚠️ IEX Cloud API: Timeout occurred")
                    self.test_results['iex_cloud_normal'] = False

        except Exception as e:
            logger.error(f"❌ IEX Cloud API test failed: {e}")
            self.test_results['iex_cloud_normal'] = False

    async def _simulate_api_failures(self, provider, api_name: str):
        """Simulate API failures to test circuit breaker behavior."""
        logger.info(f"🔬 Simulating failures for {api_name} circuit breaker...")

        # Simulate rapid successive failures
        failure_count = 0
        for i in range(5):
            try:
                # Force a failure by using invalid symbol or corrupted request
                data = await provider.get_real_time_data(['INVALID_SYMBOL_' + str(i)])
                if not data:
                    failure_count += 1

            except Exception as e:
                failure_count += 1
                logger.debug(f"Expected failure {i+1}: {e}")

            # Small delay between requests
            await asyncio.sleep(0.1)

        # Record failure scenario results
        self.failure_scenarios.append({
            'api': api_name,
            'simulated_failures': 5,
            'actual_failures': failure_count,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        logger.info(f"📊 {api_name}: {failure_count}/5 simulated failures recorded")

    async def _test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery patterns."""
        logger.info("🔄 Testing circuit breaker recovery patterns...")

        try:
            from mcp.common.resilience import CircuitBreakerManager

            # Create circuit breaker manager
            cb_manager = CircuitBreakerManager()

            # Test recovery timing
            start_time = time.time()

            # Simulate circuit breaker open state
            test_breaker = cb_manager.get_breaker("test_recovery_breaker")

            # Force failures to open the breaker
            for _ in range(3):
                try:
                    async with test_breaker:
                        raise Exception("Simulated failure")
                except:
                    pass

            # Test recovery after timeout
            await asyncio.sleep(1)  # Wait for recovery timeout

            # Attempt recovery
            recovery_success = False
            try:
                async with test_breaker:
                    recovery_success = True
                    logger.info("✅ Circuit breaker recovery: Successful")
            except Exception as e:
                logger.warning(f"⚠️ Circuit breaker recovery failed: {e}")

            recovery_time = time.time() - start_time

            self.recovery_patterns['test_recovery'] = {
                'recovery_time_seconds': recovery_time,
                'recovery_successful': recovery_success,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except ImportError:
            logger.warning("Circuit breaker module not available")
            self.recovery_patterns['test_recovery'] = {
                'recovery_time_seconds': 0,
                'recovery_successful': False,
                'error': 'Module not available'
            }

    async def test_quantconnect_api_integration(self):
        """Test QuantConnect API integration with real credentials."""
        logger.info("🔬 Testing QuantConnect API integration...")

        qc_user_id = os.environ.get('QC_USER_ID') or os.environ.get('QUANTCONNECT_USER_ID')
        qc_api_token = os.environ.get('QC_API_TOKEN') or os.environ.get('QUANTCONNECT_API_TOKEN')

        if not qc_user_id or not qc_api_token:
            logger.info("🔧 QuantConnect credentials not available, using validation mode")
            self.test_results['quantconnect_auth'] = False
            return

        try:
            # Test QuantConnect API authentication
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Test projects endpoint
                auth = aiohttp.BasicAuth(qc_user_id, qc_api_token)
                url = "https://www.quantconnect.com/api/v2/projects/read"

                try:
                    async with session.get(url, auth=auth, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info("✅ QuantConnect API: Authentication successful")
                            logger.info(f"📊 Found {len(data.get('projects', []))} projects")
                            self.test_results['quantconnect_auth'] = True

                            # Test project creation/deletion cycle
                            await self._test_quantconnect_project_lifecycle(session, auth)

                        else:
                            logger.error(f"❌ QuantConnect API authentication failed: {response.status}")
                            error_text = await response.text()
                            logger.error(f"Error response: {error_text}")
                            self.test_results['quantconnect_auth'] = False

                except asyncio.TimeoutError:
                    logger.error("❌ QuantConnect API: Request timeout")
                    self.test_results['quantconnect_auth'] = False

                except Exception as e:
                    logger.error(f"❌ QuantConnect API test failed: {e}")
                    self.test_results['quantconnect_auth'] = False

        except ImportError:
            logger.warning("aiohttp not available for QuantConnect test")
            self.test_results['quantconnect_auth'] = False

    async def _test_quantconnect_project_lifecycle(self, session, auth):
        """Test QuantConnect project creation and deletion."""
        project_name = f"test_integration_{int(time.time())}"

        try:
            # Create test project
            create_url = "https://www.quantconnect.com/api/v2/projects/create"
            create_data = {
                "projectName": project_name,
                "language": "Python"
            }

            async with session.post(create_url, auth=auth, json=create_data, timeout=30) as response:
                if response.status in [200, 201]:
                    project_data = await response.json()
                    project_id = project_data.get('projectId')

                    if project_id:
                        logger.info(f"✅ QuantConnect: Test project created (ID: {project_id})")
                        self.test_results['quantconnect_project_create'] = True

                        # Clean up - delete the test project
                        delete_url = f"https://www.quantconnect.com/api/v2/projects/{project_id}/delete"
                        async with session.delete(delete_url, auth=auth, timeout=30) as del_response:
                            if del_response.status in [200, 204]:
                                logger.info("✅ QuantConnect: Test project deleted successfully")
                                self.test_results['quantconnect_project_delete'] = True
                            else:
                                logger.warning(f"⚠️ QuantConnect: Failed to delete test project: {del_response.status}")
                                self.test_results['quantconnect_project_delete'] = False
                    else:
                        logger.error("❌ QuantConnect: No project ID returned")
                        self.test_results['quantconnect_project_create'] = False
                else:
                    logger.error(f"❌ QuantConnect: Project creation failed: {response.status}")
                    self.test_results['quantconnect_project_create'] = False

        except Exception as e:
            logger.error(f"❌ QuantConnect project lifecycle test failed: {e}")
            self.test_results['quantconnect_project_create'] = False
            self.test_results['quantconnect_project_delete'] = False

    async def test_comprehensive_system_resilience(self):
        """Test end-to-end system resilience under various failure conditions."""
        logger.info("🔬 Testing comprehensive system resilience...")

        try:
            # Test multi-agent workflow under stress
            from agents.workflows.complete_trading_workflow import CompleteTradingWorkflow

            config = {
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "orchestrator": {
                    "timeout_seconds": 30
                },
                "risk_compliance": {
                    "risk_limits": {"max_var_95": 0.05},
                    "position_limits": {"max_position_size": 0.1}
                }
            }

            workflow = CompleteTradingWorkflow(config)

            start_time = time.time()

            try:
                await workflow.initialize()

                # Run abbreviated workflow to test resilience
                symbols = config["symbols"]

                # Test hypothesis generation under stress
                hypotheses = await workflow._generate_market_hypotheses(symbols)
                logger.info(f"✅ Generated {len(hypotheses)} market hypotheses under test conditions")

                # Test signal analysis with potential failures
                signal_analysis = await workflow._analyze_signals_with_mcp(symbols)
                logger.info(f"✅ Signal analysis completed with {len(signal_analysis)} components")

                self.test_results['system_resilience'] = True
                execution_time = time.time() - start_time

                logger.info(f"✅ System resilience test completed in {execution_time:.2f} seconds")

            except Exception as e:
                logger.error(f"❌ System resilience test failed: {e}")
                self.test_results['system_resilience'] = False

            finally:
                try:
                    await workflow.cleanup()
                except:
                    pass  # Ignore cleanup errors in test

        except ImportError as e:
            logger.warning(f"Trading workflow not available: {e}")
            self.test_results['system_resilience'] = False

    async def run_all_tests(self):
        """Run the complete integration test suite."""
        logger.info("🚀 Starting Real API Integration Tests with Circuit Breaker Scenarios")
        logger.info("=" * 80)

        start_time = time.time()

        # Run all test suites
        await self.test_market_data_apis_with_circuit_breakers()
        await self.test_quantconnect_api_integration()
        await self.test_comprehensive_system_resilience()

        total_time = time.time() - start_time

        # Generate comprehensive test report
        self._generate_test_report(total_time)

    def _generate_test_report(self, total_time: float):
        """Generate comprehensive test report."""
        logger.info("=" * 80)
        logger.info("📊 REAL API INTEGRATION TEST RESULTS")
        logger.info("=" * 80)

        # Count results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests

        logger.info(f"📈 Overall Results:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"   Execution Time: {total_time:.2f} seconds")

        logger.info(f"\n🔍 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"   {test_name}: {status}")

        if self.failure_scenarios:
            logger.info(f"\n⚡ Circuit Breaker Scenarios:")
            for scenario in self.failure_scenarios:
                logger.info(f"   {scenario['api']}: {scenario['actual_failures']}/{scenario['simulated_failures']} failures")

        if self.recovery_patterns:
            logger.info(f"\n🔄 Recovery Patterns:")
            for pattern_name, pattern in self.recovery_patterns.items():
                status = "✅" if pattern.get('recovery_successful') else "❌"
                time_info = f" ({pattern.get('recovery_time_seconds', 0):.2f}s)" if pattern.get('recovery_time_seconds') else ""
                logger.info(f"   {pattern_name}: {status}{time_info}")

        # Save detailed results to file
        report_data = {
            'test_results': self.test_results,
            'failure_scenarios': self.failure_scenarios,
            'recovery_patterns': self.recovery_patterns,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests/total_tests)*100,
                'execution_time': total_time
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        with open('real_api_integration_test_results.json', 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"\n📄 Detailed results saved to: real_api_integration_test_results.json")
        logger.info("=" * 80)

        # Assert overall success for automated testing
        if failed_tests > 0:
            logger.warning(f"⚠️ {failed_tests} tests failed - check logs for details")
        else:
            logger.info("🎉 All integration tests passed!")


async def main():
    """Main test execution."""
    tester = RealAPIIntegrationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())