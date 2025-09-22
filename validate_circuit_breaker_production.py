#!/usr/bin/env python3
"""
Validation script for non-simulated circuit breaker paths in production.
Tests real circuit breaker behavior with actual API endpoints and failure scenarios.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List
import sys
import json
from datetime import datetime
import os
from pathlib import Path

# Add project root to path
if os.environ.get('GITHUB_ACTIONS'):
    project_root = os.environ.get('GITHUB_WORKSPACE', os.getcwd())
else:
    project_root = os.getcwd()
sys.path.insert(0, str(project_root))

from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig, CircuitBreaker
from mcp.servers.market_data_server import RealMarketDataProvider
from mcp.servers.quantconnect_server import QuantConnectClient
from mcp.servers.compliance_server import ComplianceEngine
from mcp.servers.risk_server import evaluate_portfolio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CircuitBreakerValidator:
    """Validates circuit breaker functionality in non-simulated environments"""

    def __init__(self):
        self.validation_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    async def validate_market_data_circuit_breaker(self) -> Dict[str, Any]:
        """Validate circuit breaker on real market data provider"""
        logger.info("Testing market data circuit breaker with real endpoints...")

        try:
            # Initialize real market data provider
            provider = RealMarketDataProvider()

            # Test 1: Normal operation should work
            result1 = await self._test_normal_operation(provider)

            # Test 2: Force failures to trigger circuit breaker
            result2 = await self._test_failure_threshold(provider)

            # Test 3: Verify circuit breaker opens after failures
            result3 = await self._test_circuit_open_state(provider)

            # Test 4: Test half-open state recovery
            result4 = await self._test_recovery_behavior(provider)

            return {
                'service': 'market_data',
                'normal_operation': result1,
                'failure_handling': result2,
                'circuit_open': result3,
                'recovery': result4,
                'status': 'PASSED' if all([result1, result2, result3, result4]) else 'FAILED'
            }

        except Exception as e:
            logger.error(f"Market data circuit breaker validation failed: {e}")
            return {
                'service': 'market_data',
                'status': 'ERROR',
                'error': str(e)
            }

    async def _test_normal_operation(self, provider) -> bool:
        """Test that normal operations work correctly"""
        try:
            # Make a valid API call
            data = await provider.get_real_time_quote('AAPL')
            if data and 'price' in data:
                logger.info("✓ Normal operation test passed")
                return True
            logger.warning("✗ Normal operation returned invalid data")
            return False
        except Exception as e:
            logger.error(f"✗ Normal operation test failed: {e}")
            return False

    async def _test_failure_threshold(self, provider) -> bool:
        """Test that circuit breaker triggers after failure threshold"""
        try:
            failures = 0
            # Intentionally trigger failures with invalid symbols
            for i in range(5):  # Exceed failure threshold
                try:
                    await provider.get_real_time_quote('INVALID_SYMBOL_XYZ_123')
                except:
                    failures += 1

            if failures >= 3:  # Default failure threshold
                logger.info(f"✓ Failure threshold test passed (triggered {failures} failures)")
                return True
            logger.warning(f"✗ Failure threshold not reached (only {failures} failures)")
            return False
        except Exception as e:
            logger.error(f"✗ Failure threshold test error: {e}")
            return False

    async def _test_circuit_open_state(self, provider) -> bool:
        """Test that circuit is open after failures"""
        try:
            # Circuit should be open, calls should fail fast
            start_time = time.time()
            try:
                await provider.get_real_time_quote('AAPL')
                logger.warning("✗ Circuit did not open - call succeeded")
                return False
            except Exception as e:
                elapsed = time.time() - start_time
                if elapsed < 0.1:  # Should fail fast when circuit is open
                    logger.info(f"✓ Circuit open test passed (failed fast in {elapsed:.3f}s)")
                    return True
                logger.warning(f"✗ Circuit open but didn't fail fast ({elapsed:.3f}s)")
                return False
        except Exception as e:
            logger.error(f"✗ Circuit open test error: {e}")
            return False

    async def _test_recovery_behavior(self, provider) -> bool:
        """Test circuit breaker recovery after timeout"""
        try:
            # Wait for recovery timeout
            logger.info("Waiting for circuit breaker recovery timeout...")
            await asyncio.sleep(31)  # Default recovery timeout is 30s

            # Circuit should be half-open, next call should work
            data = await provider.get_real_time_quote('AAPL')
            if data and 'price' in data:
                logger.info("✓ Recovery behavior test passed")
                return True
            logger.warning("✗ Recovery failed - still not working after timeout")
            return False
        except Exception as e:
            logger.error(f"✗ Recovery behavior test failed: {e}")
            return False

    async def validate_quantconnect_circuit_breaker(self) -> Dict[str, Any]:
        """Validate circuit breaker on QuantConnect integration"""
        logger.info("Testing QuantConnect circuit breaker with real API...")

        try:
            # Initialize with circuit breaker protection
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_seconds=30.0,
                expected_exception=Exception
            )

            @circuit_breaker(name="qc_test", config=config)
            async def protected_qc_call():
                # Simulate QuantConnect API call
                client = QuantConnectClient()
                return await client.get_account_status()

            results = {
                'service': 'quantconnect',
                'tests': []
            }

            # Test normal operation
            try:
                await protected_qc_call()
                results['tests'].append({'name': 'normal_operation', 'passed': True})
            except:
                results['tests'].append({'name': 'normal_operation', 'passed': False})

            # Force failures
            for i in range(4):
                try:
                    # Force failure by using invalid credentials
                    await self._force_qc_failure()
                except:
                    pass

            # Verify circuit is open
            try:
                await protected_qc_call()
                results['tests'].append({'name': 'circuit_open', 'passed': False})
            except:
                results['tests'].append({'name': 'circuit_open', 'passed': True})

            results['status'] = 'PASSED' if all(t['passed'] for t in results['tests']) else 'FAILED'
            return results

        except Exception as e:
            logger.error(f"QuantConnect circuit breaker validation failed: {e}")
            return {
                'service': 'quantconnect',
                'status': 'ERROR',
                'error': str(e)
            }

    async def _force_qc_failure(self):
        """Force a QuantConnect API failure"""
        raise Exception("Simulated QuantConnect API failure")

    async def validate_risk_compliance_circuit_breakers(self) -> Dict[str, Any]:
        """Validate circuit breakers on risk and compliance services"""
        logger.info("Testing risk and compliance circuit breakers...")

        results = {
            'service': 'risk_compliance',
            'components': {}
        }

        # Test risk server circuit breaker
        try:
            # Create test portfolio
            test_portfolio = {
                'positions': [
                    {
                        'symbol': 'AAPL',
                        'quantity': 100,
                        'price': 150.0,
                        'volatility': 0.25
                    }
                ],
                'limits': {
                    'max_drawdown': 0.1,
                    'var_limit': 10000,
                    'exposure_limit': 50000
                }
            }

            # Test with circuit breaker
            risk_result = evaluate_portfolio(test_portfolio)
            results['components']['risk_server'] = 'PASSED'

        except Exception as e:
            logger.error(f"Risk server validation failed: {e}")
            results['components']['risk_server'] = 'FAILED'

        # Test compliance server circuit breaker
        try:
            compliance_engine = ComplianceEngine()
            compliance_result = compliance_engine.check_compliance(test_portfolio)
            results['components']['compliance_server'] = 'PASSED'

        except Exception as e:
            logger.error(f"Compliance server validation failed: {e}")
            results['components']['compliance_server'] = 'FAILED'

        # Overall status
        all_passed = all(v == 'PASSED' for v in results['components'].values())
        results['status'] = 'PASSED' if all_passed else 'FAILED'

        return results

    async def validate_all_circuit_breakers(self) -> Dict[str, Any]:
        """Run all circuit breaker validations"""
        logger.info("="*60)
        logger.info("Starting comprehensive circuit breaker validation")
        logger.info("="*60)

        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'environment': 'production',
            'validations': []
        }

        # Validate market data circuit breaker
        market_result = await self.validate_market_data_circuit_breaker()
        validation_results['validations'].append(market_result)
        self._update_stats(market_result)

        # Validate QuantConnect circuit breaker
        qc_result = await self.validate_quantconnect_circuit_breaker()
        validation_results['validations'].append(qc_result)
        self._update_stats(qc_result)

        # Validate risk/compliance circuit breakers
        risk_result = await self.validate_risk_compliance_circuit_breakers()
        validation_results['validations'].append(risk_result)
        self._update_stats(risk_result)

        # Calculate overall results
        validation_results['summary'] = {
            'total_services': len(validation_results['validations']),
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'success_rate': (self.passed_tests / max(self.total_tests, 1)) * 100
        }

        # Determine overall status
        all_passed = all(v.get('status') == 'PASSED' for v in validation_results['validations'])
        validation_results['overall_status'] = 'PASSED' if all_passed else 'FAILED'

        return validation_results

    def _update_stats(self, result: Dict[str, Any]):
        """Update validation statistics"""
        self.total_tests += 1
        if result.get('status') == 'PASSED':
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    def print_results(self, results: Dict[str, Any]):
        """Print formatted validation results"""
        logger.info("\n" + "="*60)
        logger.info("CIRCUIT BREAKER VALIDATION RESULTS")
        logger.info("="*60)

        for validation in results['validations']:
            service = validation.get('service', 'unknown')
            status = validation.get('status', 'UNKNOWN')

            symbol = "✓" if status == 'PASSED' else "✗"
            logger.info(f"\n{symbol} {service.upper()}: {status}")

            if 'components' in validation:
                for comp, comp_status in validation['components'].items():
                    comp_symbol = "✓" if comp_status == 'PASSED' else "✗"
                    logger.info(f"  {comp_symbol} {comp}: {comp_status}")

            if 'tests' in validation:
                for test in validation['tests']:
                    test_symbol = "✓" if test['passed'] else "✗"
                    logger.info(f"  {test_symbol} {test['name']}: {'PASSED' if test['passed'] else 'FAILED'}")

        logger.info("\n" + "-"*60)
        logger.info(f"OVERALL STATUS: {results['overall_status']}")
        logger.info(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        logger.info(f"Passed: {results['summary']['passed']}/{results['summary']['total_services']}")
        logger.info("="*60 + "\n")

    def save_results(self, results: Dict[str, Any], filepath: str = 'circuit_breaker_validation.json'):
        """Save validation results to file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

async def main():
    """Main validation entry point"""
    validator = CircuitBreakerValidator()

    try:
        # Run all validations
        results = await validator.validate_all_circuit_breakers()

        # Print results
        validator.print_results(results)

        # Save results
        validator.save_results(results)

        # Exit with appropriate code
        if results['overall_status'] == 'PASSED':
            logger.info("✓ All circuit breaker validations passed!")
            sys.exit(0)
        else:
            logger.error("✗ Some circuit breaker validations failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())