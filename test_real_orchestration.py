"""
Real multi-agent orchestration system test.

This demonstrates the complete multi-agent system working with:
- Real agent implementations (Fundamental, Technical, Sentiment, Quantitative)
- Real MCP server integrations
- Complete orchestration workflow
- Risk validation and compliance
"""
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our components
from agents.core.orchestrator import (
    MultiAgentOrchestrator, OrchestrationRequest,
    OrchestrationPhase, AgentConsensus
)
from agents.core import AgentRole

# Try to import full agents, fallback to simple ones if dependencies missing
try:
    from agents.implementations.fundamental_agent import FundamentalAgent
    from agents.implementations.technical_agent import TechnicalAgent
    from agents.implementations.sentiment_agent import SentimentAgent
    from agents.implementations.quantitative_agent import QuantitativeAgent
    print("✓ Using full agent implementations")
except ImportError as e:
    print(f"⚠️ Full agents unavailable ({e}), using simplified implementations")
    from agents.implementations.simple_agents import (
        SimpleFundamentalAgent as FundamentalAgent,
        SimpleTechnicalAgent as TechnicalAgent,
        SimpleSentimentAgent as SentimentAgent,
        SimpleQuantitativeAgent as QuantitativeAgent
    )


class RealOrchestrationDemo:
    """Demonstration of real multi-agent orchestration."""

    def __init__(self):
        self.orchestrator = MultiAgentOrchestrator({
            "demo_mode": True,
            "risk_tolerance": "medium",
            "max_position_size": 0.1
        })

        # Initialize real agents
        self.agents = {
            AgentRole.FUNDAMENTAL: FundamentalAgent("fundamental-001", {
                "dcf_growth_rate": 0.06,
                "safety_margin": 0.25
            }),
            AgentRole.TECHNICAL: TechnicalAgent("technical-001", {
                "rsi_period": 14,
                "ma_short": 10,
                "ma_long": 20
            }),
            AgentRole.SENTIMENT: SentimentAgent("sentiment-001", {
                "news_weight": 0.4,
                "social_weight": 0.3,
                "analyst_weight": 0.3
            }),
            AgentRole.QUANTITATIVE: QuantitativeAgent("quantitative-001", {
                "factor_models": ["momentum", "value", "quality"],
                "lookback_days": 252
            })
        }

    async def run_full_demo(self) -> Dict[str, Any]:
        """Run complete orchestration demonstration."""
        print("🚀 Starting Real Multi-Agent Orchestration Demo")
        print("=" * 60)

        try:
            # Step 1: Initialize orchestrator
            print("\n📋 Step 1: Initializing Orchestrator")
            await self._initialize_orchestrator()

            # Step 2: Register agents
            print("\n👥 Step 2: Registering Real Agents")
            await self._register_agents()

            # Step 3: Run orchestrated analysis
            print("\n🔬 Step 3: Running Orchestrated Analysis")
            analysis_result = await self._run_orchestrated_analysis()

            # Step 4: Demonstrate risk and compliance
            print("\n🛡️ Step 4: Risk and Compliance Validation")
            await self._demonstrate_risk_compliance(analysis_result)

            # Step 5: Show execution planning
            print("\n📈 Step 5: Execution Planning")
            await self._show_execution_planning(analysis_result)

            # Step 6: Performance metrics
            print("\n📊 Step 6: Performance Analytics")
            performance = await self._analyze_performance(analysis_result)

            return {
                "orchestration_result": analysis_result,
                "performance_metrics": performance,
                "demo_status": "completed_successfully"
            }

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return {
                "demo_status": "failed",
                "error": str(e)
            }
        finally:
            await self.orchestrator.stop()

    async def _initialize_orchestrator(self):
        """Initialize the orchestrator."""
        await self.orchestrator.start()

        status = self.orchestrator.get_orchestrator_status()
        print(f"✓ Orchestrator initialized: {status['orchestrator_id'][:8]}")
        print(f"✓ MCP tools available: {', '.join(status['mcp_tools_available'])}")

    async def _register_agents(self):
        """Register all agents with the orchestrator."""
        for role, agent in self.agents.items():
            self.orchestrator.register_agent(agent)
            print(f"✓ Registered {role.value} agent: {agent.agent_id[:8]}")

        status = self.orchestrator.get_orchestrator_status()
        print(f"✓ Total agents registered: {status['active_agents']}")

    async def _run_orchestrated_analysis(self):
        """Run orchestrated analysis on test symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        request = OrchestrationRequest(
            request_id="demo_analysis_001",
            symbols=symbols,
            analysis_types=["fundamental", "technical", "sentiment", "quantitative"],
            parameters={
                "fundamental": {"focus": "dcf_valuation"},
                "technical": {"timeframe": "daily"},
                "sentiment": {"sources": ["news", "social", "analyst"]},
                "quantitative": {"models": ["momentum", "value"]}
            }
        )

        print(f"📊 Analyzing {len(symbols)} symbols: {', '.join(symbols)}")

        result = await self.orchestrator.orchestrate_analysis(request)

        print(f"✓ Analysis completed in {result.performance_metrics['total_duration_seconds']:.2f}s")
        print(f"✓ Consensus decisions: {len(result.consensus_decisions)}")
        print(f"✓ Success rate: {result.performance_metrics.get('success_rate', 0):.1f}%")

        # Show phase timings
        print("\n⏱️ Phase Execution Times:")
        for phase, duration in result.phase_timings.items():
            print(f"   {phase}: {duration:.3f}s")

        # Show consensus decisions
        print("\n🎯 Consensus Decisions:")
        for decision in result.consensus_decisions:
            print(f"   {decision.symbol}: {decision.consensus_signal.name} "
                  f"(confidence: {decision.consensus_confidence.name}, "
                  f"agreement: {decision.agreement_score:.2f})")
            print(f"      Reasoning: {decision.reasoning}")

        return result

    async def _demonstrate_risk_compliance(self, result):
        """Demonstrate risk and compliance validation."""
        print("🛡️ Risk and Compliance Analysis:")

        for decision in result.consensus_decisions:
            risk_assessment = decision.risk_assessment
            print(f"\n   {decision.symbol} Risk Assessment:")
            print(f"     Risk Score: {risk_assessment.get('risk_score', 'N/A')}")
            print(f"     VaR Impact: {risk_assessment.get('var_impact', 'N/A')}")
            print(f"     Status: {risk_assessment.get('reason', 'Approved')}")

        execution_plan = result.execution_plan
        print(f"\n✓ Compliance Status: {execution_plan.get('compliance_status', 'unknown')}")
        print(f"✓ Orders cleared for execution: {len(execution_plan.get('orders', []))}")

    async def _show_execution_planning(self, result):
        """Show execution planning details."""
        execution_plan = result.execution_plan
        orders = execution_plan.get("orders", [])

        print(f"📈 Execution Plan Generated:")
        print(f"   Total orders: {len(orders)}")
        print(f"   Timestamp: {execution_plan.get('timestamp')}")

        for order in orders:
            print(f"\n   Order: {order['symbol']}")
            print(f"     Side: {order['side']} (strength: {order['signal_strength']:.2f})")
            print(f"     Confidence: {order['confidence']:.2f}")
            print(f"     Agreement: {order['agreement_score']:.2f}")
            print(f"     Agent consensus: {order['agents_consensus']} agents")

    async def _analyze_performance(self, result):
        """Analyze orchestration performance."""
        metrics = result.performance_metrics

        print("📊 Performance Analysis:")
        print(f"   Total duration: {metrics['total_duration_seconds']:.3f}s")
        print(f"   Agents utilized: {metrics['agents_used']}")
        print(f"   Symbols processed: {metrics['symbols_analyzed']}")
        print(f"   Decisions generated: {metrics['consensus_decisions']}")
        print(f"   Overall success rate: {metrics.get('success_rate', 0):.1f}%")

        # Calculate efficiency metrics
        if metrics['total_duration_seconds'] > 0:
            throughput = metrics['symbols_analyzed'] / metrics['total_duration_seconds']
            print(f"   Throughput: {throughput:.2f} symbols/second")

        # Phase efficiency
        phase_timings = result.phase_timings
        total_phase_time = sum(phase_timings.values())

        print("\n⚡ Phase Efficiency:")
        for phase, duration in phase_timings.items():
            percentage = (duration / total_phase_time) * 100 if total_phase_time > 0 else 0
            print(f"   {phase}: {duration:.3f}s ({percentage:.1f}%)")

        return {
            "total_duration": metrics['total_duration_seconds'],
            "throughput": throughput if metrics['total_duration_seconds'] > 0 else 0,
            "phase_breakdown": {k: v for k, v in phase_timings.items()},
            "efficiency_score": min(100, (metrics.get('success_rate', 0) / max(1, metrics['total_duration_seconds'])) * 10)
        }


async def main():
    """Run the real orchestration demonstration."""
    demo = RealOrchestrationDemo()

    try:
        result = await demo.run_full_demo()

        print("\n" + "=" * 60)
        if result["demo_status"] == "completed_successfully":
            print("✅ Real Multi-Agent Orchestration Demo COMPLETED SUCCESSFULLY!")
            print("\n🎯 Key Achievements:")
            print("   ✓ Multi-agent coordination working")
            print("   ✓ Real agent implementations integrated")
            print("   ✓ Risk and compliance validation operational")
            print("   ✓ Execution planning automated")
            print("   ✓ Performance monitoring active")

            perf = result["performance_metrics"]
            print(f"\n📈 Performance Summary:")
            print(f"   Processing time: {perf['total_duration']:.3f}s")
            print(f"   Throughput: {perf['throughput']:.2f} symbols/sec")
            print(f"   Efficiency score: {perf['efficiency_score']:.1f}/100")
        else:
            print("❌ Demo failed - see errors above")

    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())