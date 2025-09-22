
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import numpy as np

class TestIntegrationFlows:
    """Comprehensive integration tests for critical flows"""

    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self):
        """Test end-to-end trading workflow with real components"""

        # Setup
        from agents.workflows.complete_trading_workflow import CompleteTradingWorkflow
        from mcp.servers.market_data_server import RealMarketDataProvider

        workflow = CompleteTradingWorkflow()

        # Test data
        symbols = ["AAPL", "GOOGL", "MSFT"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        # Execute workflow
        result = await workflow.execute({
            "symbols": symbols,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "strategy": "momentum",
            "risk_limit": 0.02
        })

        # Assertions
        assert result["status"] == "success"
        assert "portfolio" in result
        assert "signals" in result
        assert "risk_metrics" in result
        assert result["risk_metrics"]["var"] < 0.05  # VaR under 5%

    @pytest.mark.asyncio
    async def test_streaming_pipeline(self):
        """Test real-time streaming data pipeline"""

        from mcp.servers.streaming_pipeline_server import StreamingPipeline

        pipeline = StreamingPipeline()

        # Subscribe to market data stream
        subscription_id = await pipeline.subscribe({
            "stream_type": "market_data",
            "symbols": ["SPY"],
            "callback": self._handle_stream_data
        })

        # Let it run for a bit
        await asyncio.sleep(5)

        # Check we received data
        assert self.stream_data_received > 0

        # Unsubscribe
        await pipeline.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_compliance_workflow(self):
        """Test compliance checking workflow"""

        from compliance.regulatory_automation import ComplianceAutomationSystem

        compliance = ComplianceAutomationSystem()

        # Test portfolio
        portfolio = {
            "total_value": 1000000,
            "positions": {
                "AAPL": {"market_value": 250000},
                "GOOGL": {"market_value": 200000},
                "MSFT": {"market_value": 150000}
            },
            "cash": 400000,
            "risk_metrics": {"var_95": 45000}
        }

        # Run compliance check
        report = await compliance.monitor_compliance(
            portfolio,
            market_data={},
            transaction_history=[]
        )

        # Verify compliance
        assert report.total_violations == 0
        assert report.portfolio_value == 1000000

    @pytest.mark.asyncio
    async def test_portfolio_optimization(self):
        """Test portfolio optimization with real data"""

        from portfolio.hierarchical_risk_parity import HierarchicalRiskParity
        from mcp.portfolio.black_litterman import BlackLittermanOptimizer

        # Get historical returns
        returns = await self._get_historical_returns(["AAPL", "GOOGL", "MSFT"])

        # HRP optimization
        hrp = HierarchicalRiskParity()
        hrp_weights = hrp.optimize(returns)

        # Verify HRP results
        assert abs(sum(hrp_weights.values()) - 1.0) < 0.001  # Weights sum to 1
        assert all(0 <= w <= 1 for w in hrp_weights.values())

        # Black-Litterman optimization
        bl = BlackLittermanOptimizer()
        bl_results = bl.optimize(returns, agent_views=[])

        # Verify BL results
        assert bl_results.sharpe_ratio > 0
        assert abs(sum(bl_results.optimal_weights.values()) - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self):
        """Test circuit breaker under failure conditions"""

        from mcp.common.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        # Force failures
        for _ in range(3):
            with pytest.raises(Exception):
                await cb.call(self._failing_function)

        # Circuit should be open
        assert cb.state == "open"

        # Calls should fail fast
        with pytest.raises(Exception):
            await cb.call(self._failing_function)

        # Wait for recovery
        await asyncio.sleep(1.5)

        # Circuit should be half-open
        assert cb.state == "half_open"

    async def _failing_function(self):
        raise Exception("Simulated failure")

    async def _handle_stream_data(self, data):
        self.stream_data_received += 1

    async def _get_historical_returns(self, symbols):
        # Mock historical returns for testing
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        returns = pd.DataFrame(
            np.random.randn(252, len(symbols)) * 0.02,
            index=dates,
            columns=symbols
        )
        return returns
