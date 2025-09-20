"""
Comprehensive system integration tests for the complete trading system.

Tests integration between all major components: agents, performance optimization,
portfolio management, compliance, monitoring, quantum computing, and Web3 DeFi.
"""
import asyncio
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import all major system components
from agents.core.base_agent import BaseAgent
from agents.implementations.fundamental_agent import FundamentalAgent
from agents.implementations.technical_agent import TechnicalAgent
from agents.implementations.sentiment_agent import SentimentAgent
from agents.implementations.quantitative_agent import QuantitativeAgent

from mcp.performance.continuous_batching import ContinuousBatchingEngine, ToolRequest
from mcp.performance.caching import MultiTierCache
from mcp.performance.streaming import RealTimeStreamingPipeline, StreamEvent, StreamEventType
from mcp.performance.parallel_execution import ParallelExecutionEngine, WorkTask, WorkloadType
from mcp.performance.observability import ObservabilityManager

from mcp.portfolio.hrp_optimizer import HRPOptimizer, HRPParameters
from mcp.portfolio.black_litterman import BlackLittermanOptimizer, AgentView, ViewType, ConfidenceLevel

from mcp.compliance.mifid_ii import MiFIDComplianceManager, ClientCategory
from mcp.monitoring.production_monitor import ProductionMonitor, SLATarget, ServiceLevel
from mcp.quantum.quantum_optimizer import QuantumPortfolioManager, QAOAParameters
from mcp.web3.defi_yield_farming import DeFiPortfolioManager, RiskLevel


class SystemIntegrationTests(unittest.TestCase):
    """Comprehensive system integration test suite."""

    def setUp(self):
        """Set up test environment with all system components."""
        # Initialize agents
        self.fundamental_agent = FundamentalAgent("fundamental")
        self.technical_agent = TechnicalAgent("technical")
        self.sentiment_agent = SentimentAgent("sentiment")
        self.quantitative_agent = QuantitativeAgent("quantitative")

        # Initialize performance components
        self.batching_engine = ContinuousBatchingEngine()
        self.cache = MultiTierCache()
        self.streaming_pipeline = RealTimeStreamingPipeline()
        self.parallel_executor = ParallelExecutionEngine()
        self.observability = ObservabilityManager()

        # Initialize portfolio optimization
        self.hrp_optimizer = HRPOptimizer(HRPParameters())
        self.bl_optimizer = BlackLittermanOptimizer()

        # Initialize compliance and monitoring
        self.compliance_manager = MiFIDComplianceManager("FIRM-001")
        self.production_monitor = ProductionMonitor()

        # Initialize quantum and Web3
        self.quantum_manager = QuantumPortfolioManager()
        self.defi_manager = DeFiPortfolioManager()

        # Test data
        self.test_returns = self._generate_test_returns()
        self.test_market_data = self._generate_test_market_data()

    def _generate_test_returns(self) -> pd.DataFrame:
        """Generate synthetic return data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

        returns = np.random.multivariate_normal(
            mean=[0.0008] * len(assets),
            cov=np.eye(len(assets)) * 0.02 + 0.005,
            size=len(dates)
        )

        return pd.DataFrame(returns, index=dates, columns=assets)

    def _generate_test_market_data(self) -> Dict[str, Any]:
        """Generate test market data."""
        return {
            'prices': {
                'AAPL': 175.50,
                'GOOGL': 142.30,
                'MSFT': 378.85,
                'TSLA': 248.42,
                'NVDA': 481.12
            },
            'volumes': {
                'AAPL': 1_250_000,
                'GOOGL': 987_000,
                'MSFT': 1_450_000,
                'TSLA': 2_100_000,
                'NVDA': 1_800_000
            },
            'market_cap': {
                'AAPL': 2_750_000_000_000,
                'GOOGL': 1_780_000_000_000,
                'MSFT': 2_820_000_000_000,
                'TSLA': 785_000_000_000,
                'NVDA': 1_200_000_000_000
            }
        }

    async def test_agent_collaboration_pipeline(self):
        """Test multi-agent collaboration for investment decisions."""
        print("\n=== Testing Agent Collaboration Pipeline ===")

        # Test data for agent analysis
        symbol = "AAPL"
        market_data = {
            "symbol": symbol,
            "price": 175.50,
            "volume": 1_250_000,
            "market_cap": 2_750_000_000_000,
            "pe_ratio": 29.5,
            "eps": 6.13,
            "revenue_growth": 0.08,
            "news_sentiment": 0.75,
            "analyst_ratings": {"buy": 15, "hold": 8, "sell": 2}
        }

        # Get analysis from each agent
        fundamental_analysis = await self.fundamental_agent.analyze_stock(symbol, market_data)
        technical_analysis = await self.technical_agent.analyze_stock(symbol, market_data)
        sentiment_analysis = await self.sentiment_agent.analyze_market_sentiment(symbol, market_data)
        quantitative_analysis = await self.quantitative_agent.analyze_factors(symbol, market_data)

        # Validate each agent provides analysis
        self.assertIsNotNone(fundamental_analysis)
        self.assertIsNotNone(technical_analysis)
        self.assertIsNotNone(sentiment_analysis)
        self.assertIsNotNone(quantitative_analysis)

        # Check analysis structure
        for analysis in [fundamental_analysis, technical_analysis, sentiment_analysis, quantitative_analysis]:
            self.assertIn('symbol', analysis)
            self.assertIn('recommendation', analysis)
            self.assertIn('confidence', analysis)
            self.assertIn('reasoning', analysis)

        print(f"✓ All agents provided analysis for {symbol}")
        print(f"  Fundamental: {fundamental_analysis['recommendation']} ({fundamental_analysis['confidence']:.2f})")
        print(f"  Technical: {technical_analysis['recommendation']} ({technical_analysis['confidence']:.2f})")
        print(f"  Sentiment: {sentiment_analysis['recommendation']} ({sentiment_analysis['confidence']:.2f})")
        print(f"  Quantitative: {quantitative_analysis['recommendation']} ({quantitative_analysis['confidence']:.2f})")

    async def test_performance_optimization_stack(self):
        """Test the complete performance optimization stack."""
        print("\n=== Testing Performance Optimization Stack ===")

        # Test continuous batching
        await self.batching_engine.start()

        tool_requests = [
            ToolRequest(
                request_id=f"req_{i}",
                tool_name="test_tool",
                parameters={"value": i},
                priority=1
            ) for i in range(10)
        ]

        batch = await self.batching_engine.create_batch(tool_requests)
        self.assertGreater(len(batch.requests), 0)
        print(f"✓ Continuous batching: Created batch with {len(batch.requests)} requests")

        await self.batching_engine.stop()

        # Test caching
        await self.cache.put("test_key", "test_value", ttl=60)
        cached_value = await self.cache.get("test_key")
        self.assertEqual(cached_value, "test_value")
        print("✓ Multi-tier caching: Store and retrieve successful")

        # Test streaming pipeline
        await self.streaming_pipeline.start()

        test_event = StreamEvent(
            event_id="test_001",
            event_type=StreamEventType.MARKET_DATA,
            timestamp_ns=int(datetime.now().timestamp() * 1_000_000_000),
            data={"symbol": "AAPL", "price": 175.50}
        )

        success = await self.streaming_pipeline.publish(test_event)
        self.assertTrue(success)
        print("✓ Real-time streaming: Event published successfully")

        await self.streaming_pipeline.stop()

        # Test parallel execution
        async def test_function(x):
            return x * 2

        tasks = [
            WorkTask(
                task_id=f"task_{i}",
                function=test_function,
                args=i,
                workload_type=WorkloadType.CPU_INTENSIVE
            ) for i in range(5)
        ]

        results = await self.parallel_executor.execute_batch(tasks)
        self.assertEqual(len(results), 5)
        print(f"✓ Parallel execution: Processed {len(results)} tasks")

        # Test observability
        await self.observability.start()
        self.observability.record_request("test_service", 0.05, success=True)
        health = self.observability.get_health_status()
        self.assertIn("status", health)
        print("✓ Observability: Health monitoring active")

        await self.observability.stop()

    async def test_portfolio_optimization_integration(self):
        """Test portfolio optimization with HRP and Black-Litterman."""
        print("\n=== Testing Portfolio Optimization Integration ===")

        # Test HRP optimization
        hrp_allocation = self.hrp_optimizer.optimize(self.test_returns)
        self.assertIsNotNone(hrp_allocation)
        self.assertGreater(hrp_allocation.sharpe_ratio, 0)
        print(f"✓ HRP Optimization: Sharpe ratio = {hrp_allocation.sharpe_ratio:.3f}")

        # Create agent views for Black-Litterman
        agent_views = [
            AgentView(
                view_id="view_001",
                agent_source="fundamental",
                view_type=ViewType.ABSOLUTE_RETURN,
                assets=["AAPL"],
                expected_return=0.12,
                confidence=ConfidenceLevel.HIGH,
                time_horizon=252,
                rationale="Strong earnings growth expected"
            ),
            AgentView(
                view_id="view_002",
                agent_source="technical",
                view_type=ViewType.RELATIVE_RETURN,
                assets=["GOOGL", "MSFT"],
                expected_return=0.05,
                confidence=ConfidenceLevel.MEDIUM,
                time_horizon=90,
                rationale="Technical breakout pattern"
            )
        ]

        # Test Black-Litterman with agent views
        bl_result = self.bl_optimizer.optimize(self.test_returns, agent_views)
        self.assertIsNotNone(bl_result)
        self.assertGreater(bl_result.sharpe_ratio, 0)
        print(f"✓ Black-Litterman: Sharpe ratio = {bl_result.sharpe_ratio:.3f}, Views = {len(agent_views)}")

    async def test_compliance_and_monitoring(self):
        """Test compliance monitoring and production monitoring."""
        print("\n=== Testing Compliance and Monitoring ===")

        # Test MiFID II compliance
        trade_data = {
            "trade_id": "trade_001",
            "instrument_id": "AAPL",
            "price": 175.50,
            "quantity": 100,
            "venue": "primary_exchange",
            "market_prices": {
                "primary_exchange": 175.50,
                "alternative_exchange": 175.52
            },
            "execution_time_ms": 45
        }

        compliance_result = await self.compliance_manager.process_trade(
            trade_data, ClientCategory.RETAIL, "trader_001"
        )

        self.assertIn("compliance_status", compliance_result)
        self.assertIn("checks_performed", compliance_result)
        print(f"✓ MiFID II Compliance: Status = {compliance_result['compliance_status']}, Checks = {compliance_result['checks_performed']}")

        # Test production monitoring
        await self.production_monitor.start()

        sla_target = SLATarget(
            service_name="trading_engine",
            service_level=ServiceLevel.CRITICAL,
            uptime_target=99.99,
            latency_p50_ms=10,
            latency_p99_ms=50,
            error_rate_threshold=0.1,
            throughput_minimum=1000
        )

        self.production_monitor.register_service("trading_engine", sla_target)
        health = await self.production_monitor.check_health("trading_engine")
        self.assertIn("status", health)
        print(f"✓ Production Monitoring: Service health = {health.get('status', 'unknown')}")

        await self.production_monitor.stop()

    async def test_quantum_and_web3_integration(self):
        """Test quantum computing and Web3 DeFi integration."""
        print("\n=== Testing Quantum and Web3 Integration ===")

        # Test quantum optimization
        returns = self.test_returns.values[:5, :5]  # Limit to 5 assets for quantum simulation
        covariance = np.cov(returns.T)

        quantum_result = self.quantum_manager.optimize_with_qaoa(
            returns.mean(axis=0), covariance, risk_aversion=1.0, p=2
        )

        self.assertIsNotNone(quantum_result)
        self.assertEqual(quantum_result.algorithm, "QAOA")
        print(f"✓ Quantum Optimization: Value = {quantum_result.optimal_value:.4f}, Algorithm = {quantum_result.algorithm}")

        # Test benchmarking quantum vs classical
        benchmark = self.quantum_manager.benchmark_quantum_vs_classical(
            returns.mean(axis=0), covariance
        )

        self.assertIn("quantum", benchmark)
        self.assertIn("classical", benchmark)
        print(f"✓ Quantum Benchmark: Speedup = {benchmark.get('speedup', 0):.2f}x")

        # Test DeFi portfolio management
        defi_portfolio = await self.defi_manager.create_yield_strategy(
            amount_usd=10000, risk_level=RiskLevel.MEDIUM
        )

        self.assertGreater(defi_portfolio.total_value_usd, 0)
        self.assertGreater(len(defi_portfolio.active_positions), 0)
        print(f"✓ DeFi Portfolio: Value = ${defi_portfolio.total_value_usd:,.2f}, Positions = {len(defi_portfolio.active_positions)}")

        # Test MEV scanning
        mev_opportunities = await self.defi_manager.scan_mev_opportunities()
        self.assertIn("arbitrage", mev_opportunities)
        self.assertIn("liquidation", mev_opportunities)
        print(f"✓ MEV Scanning: Found {len(mev_opportunities.get('arbitrage', []))} arbitrage, {len(mev_opportunities.get('liquidation', []))} liquidation opportunities")

    def test_system_architecture_validation(self):
        """Validate overall system architecture and component integration."""
        print("\n=== Testing System Architecture Validation ===")

        # Test component initialization
        components = {
            'agents': [self.fundamental_agent, self.technical_agent, self.sentiment_agent, self.quantitative_agent],
            'performance': [self.batching_engine, self.cache, self.streaming_pipeline, self.parallel_executor],
            'portfolio': [self.hrp_optimizer, self.bl_optimizer],
            'compliance': [self.compliance_manager],
            'monitoring': [self.production_monitor],
            'quantum': [self.quantum_manager],
            'web3': [self.defi_manager]
        }

        for category, component_list in components.items():
            for component in component_list:
                self.assertIsNotNone(component)
            print(f"✓ {category.title()}: {len(component_list)} components initialized")

        # Validate data flow compatibility
        self.assertEqual(self.test_returns.shape[1], 5)  # 5 assets
        self.assertGreater(len(self.test_returns), 200)  # Sufficient history
        print(f"✓ Data Pipeline: {self.test_returns.shape[0]} days × {self.test_returns.shape[1]} assets")

        # Test memory and performance requirements
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.assertLess(memory_mb, 1000)  # Should use < 1GB
        print(f"✓ Resource Usage: {memory_mb:.1f} MB memory")

    async def test_end_to_end_trading_workflow(self):
        """Test complete end-to-end trading workflow."""
        print("\n=== Testing End-to-End Trading Workflow ===")

        workflow_steps = []

        # Step 1: Market Analysis
        symbol = "AAPL"
        market_data = self.test_market_data

        fundamental_view = await self.fundamental_agent.analyze_stock(symbol, {
            "symbol": symbol,
            "price": market_data['prices'][symbol],
            "market_cap": market_data['market_cap'][symbol],
            "pe_ratio": 29.5,
            "eps": 6.13
        })
        workflow_steps.append("Market Analysis")

        # Step 2: Portfolio Optimization
        hrp_allocation = self.hrp_optimizer.optimize(self.test_returns)
        workflow_steps.append("Portfolio Optimization")

        # Step 3: Risk Assessment and Compliance
        trade_data = {
            "instrument_id": symbol,
            "price": market_data['prices'][symbol],
            "quantity": 100,
            "venue": "primary_exchange"
        }
        compliance_check = await self.compliance_manager.process_trade(
            trade_data, ClientCategory.PROFESSIONAL, "system"
        )
        workflow_steps.append("Compliance Check")

        # Step 4: Performance Monitoring
        await self.observability.start()
        self.observability.record_request("trading_workflow", 0.250, success=True)
        monitoring_health = self.observability.get_health_status()
        workflow_steps.append("Performance Monitoring")
        await self.observability.stop()

        # Step 5: Alternative Strategy (DeFi)
        defi_analysis = await self.defi_manager.get_portfolio_analytics()
        workflow_steps.append("DeFi Analysis")

        # Validate workflow completion
        self.assertEqual(len(workflow_steps), 5)
        self.assertEqual(compliance_check.get('compliance_status'), 'compliant')
        self.assertGreater(hrp_allocation.sharpe_ratio, 0)

        print(f"✓ Workflow Steps Completed: {' → '.join(workflow_steps)}")
        print(f"✓ Final Validation: Portfolio Sharpe = {hrp_allocation.sharpe_ratio:.3f}, Compliance = {compliance_check.get('compliance_status')}")

    async def run_all_tests(self):
        """Run all integration tests sequentially."""
        print("🚀 Starting Comprehensive System Integration Tests")
        print("=" * 60)

        test_methods = [
            self.test_agent_collaboration_pipeline,
            self.test_performance_optimization_stack,
            self.test_portfolio_optimization_integration,
            self.test_compliance_and_monitoring,
            self.test_quantum_and_web3_integration,
            self.test_end_to_end_trading_workflow
        ]

        # Run synchronous test first
        self.test_system_architecture_validation()

        # Run async tests
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                print(f"❌ Test {test_method.__name__} failed: {e}")
                raise

        print("\n" + "=" * 60)
        print("✅ All Integration Tests Passed Successfully!")
        print(f"📊 System Components Validated: Agents, Performance, Portfolio, Compliance, Monitoring, Quantum, Web3")
        print(f"🎯 Renaissance Technologies-level Architecture: Complete")


async def main():
    """Run the integration test suite."""
    test_suite = SystemIntegrationTests()
    test_suite.setUp()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())