"""
Multi-agent orchestration system for coordinated trading decisions.

This orchestrator manages the interaction between different agent types,
MCP servers, and creates a comprehensive trading decision pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from agents.core import (
    BaseAgent, AgentRole, AnalysisRequest, AnalysisResult,
    TradingSignal, SignalDirection, ConfidenceLevel, MarketData,
    AgentMessage, MessageBus, DataProvider
)

logger = logging.getLogger(__name__)


class OrchestrationPhase(Enum):
    """Phases of the orchestration workflow."""
    INITIALIZATION = "initialization"
    DATA_COLLECTION = "data_collection"
    AGENT_ANALYSIS = "agent_analysis"
    SIGNAL_AGGREGATION = "signal_aggregation"
    RISK_VALIDATION = "risk_validation"
    DECISION_MAKING = "decision_making"
    EXECUTION_PLANNING = "execution_planning"
    COMPLETION = "completion"


@dataclass
class OrchestrationRequest:
    """Request for orchestrated analysis."""
    request_id: str
    symbols: List[str]
    analysis_types: List[str] = field(default_factory=lambda: ["fundamental", "technical", "sentiment", "quantitative"])
    priority: int = 0
    deadline: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConsensus:
    """Consensus result from multiple agents."""
    symbol: str
    consensus_signal: SignalDirection
    consensus_confidence: ConfidenceLevel
    agent_signals: List[TradingSignal]
    agreement_score: float  # 0-1, higher = more agreement
    reasoning: str
    risk_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result from orchestrated analysis."""
    request_id: str
    symbols: List[str]
    consensus_decisions: List[AgentConsensus]
    execution_plan: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    phase_timings: Dict[str, float]
    errors: List[str] = field(default_factory=list)


class InMemoryMessageBus:
    """Simple in-memory message bus for agent communication."""

    def __init__(self):
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._broadcast_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()

    async def publish(self, message: AgentMessage) -> None:
        """Publish message to recipients or broadcast."""
        if message.recipient_id:
            # Direct message
            if message.recipient_id in self._subscribers:
                await self._subscribers[message.recipient_id].put(message)
        else:
            # Broadcast message
            await self._broadcast_queue.put(message)

            # Also send to all subscribers
            for queue in self._subscribers.values():
                try:
                    await queue.put(message)
                except asyncio.QueueFull:
                    logger.warning("Message queue full, dropping message")

    async def subscribe(self, agent_id: str):
        """Subscribe to messages for an agent."""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = asyncio.Queue(maxsize=1000)

        queue = self._subscribers[agent_id]

        while True:
            try:
                message = await queue.get()
                yield message
            except asyncio.CancelledError:
                break


class RealMarketDataProvider:
    """Real market data provider using MCP market data server."""

    def __init__(self):
        self._use_fallback = False

    async def get_market_data(self, symbols: List[str]) -> List[MarketData]:
        """Get real market data from MCP server."""
        try:
            # Import market data server
            from mcp.servers.market_data_server import get_real_time_market_data

            # Call MCP tool
            result = await get_real_time_market_data({"symbols": symbols})

            if result.get("success") and result.get("data"):
                market_data = []
                for data_point in result["data"]:
                    # Convert back to MarketData object
                    data = MarketData(
                        symbol=data_point["symbol"],
                        timestamp=datetime.fromisoformat(data_point["timestamp"].replace('Z', '+00:00')),
                        price=data_point["price"],
                        volume=data_point["volume"],
                        bid=data_point.get("bid"),
                        ask=data_point.get("ask"),
                        high_24h=data_point.get("high_24h"),
                        low_24h=data_point.get("low_24h"),
                        metadata=data_point.get("metadata", {})
                    )
                    market_data.append(data)

                logger.info(f"Retrieved real market data for {len(market_data)} symbols")
                return market_data
            else:
                logger.warning(f"Market data request failed: {result.get('error', 'Unknown error')}")
                return await self._get_fallback_data(symbols)

        except Exception as e:
            logger.error(f"Real market data error: {e}")
            return await self._get_fallback_data(symbols)

    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketData]:
        """Get historical market data."""
        try:
            from mcp.servers.market_data_server import get_historical_market_data

            days = (end_time - start_time).days
            result = await get_historical_market_data({
                "symbol": symbol,
                "days": min(days, 365)
            })

            if result.get("success") and result.get("data"):
                market_data = []
                for data_point in result["data"]:
                    data = MarketData(
                        symbol=data_point["symbol"],
                        timestamp=datetime.fromisoformat(data_point["timestamp"].replace('Z', '+00:00')),
                        price=data_point["price"],
                        volume=data_point["volume"],
                        metadata=data_point.get("metadata", {})
                    )
                    market_data.append(data)

                logger.info(f"Retrieved {len(market_data)} historical data points for {symbol}")
                return market_data
            else:
                logger.error(f"Historical data request failed: {result.get('error', 'Unknown error')}")
                from mcp.common.exceptions import DataUnavailableError
                raise DataUnavailableError(f"Historical data service returned error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Historical data error: {e}")
            # DO NOT fall back to synthetic data - raise the error
            from mcp.common.exceptions import DataUnavailableError
            raise DataUnavailableError(f"Unable to get historical data for {symbol}: {str(e)}")

    async def _get_fallback_data(self, symbols: List[str]) -> List[MarketData]:
        """Fallback to mock data when real data is unavailable."""
        if not self._use_fallback:
            logger.warning("Falling back to mock market data")
            self._use_fallback = True

        market_data = []
        base_timestamp = datetime.now(timezone.utc)

        for i, symbol in enumerate(symbols):
            # Generate deterministic mock data
            import hashlib
            seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
            base_price = 50 + (seed % 200)

            data = MarketData(
                symbol=symbol,
                timestamp=base_timestamp,
                price=base_price + (i % 10),
                volume=1000000 + (i * 100000),
                bid=base_price - 0.5,
                ask=base_price + 0.5,
                high_24h=base_price * 1.02,
                low_24h=base_price * 0.98,
                metadata={
                    "source": "fallback_mock",
                    "warning": "Real market data unavailable"
                }
            )
            market_data.append(data)

        return market_data

    async def _get_fallback_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketData]:
        """Fallback historical data generation."""
        data_points = []
        current_time = start_time
        base_price = 100.0

        while current_time <= end_time:
            # Simple random walk
            price_change = (hash(current_time.isoformat()) % 100) / 500 - 0.1
            base_price = max(base_price * (1 + price_change), 1.0)

            data = MarketData(
                symbol=symbol,
                timestamp=current_time,
                price=base_price,
                volume=1000000,
                metadata={
                    "source": "fallback_mock",
                    "historical": True
                }
            )
            data_points.append(data)
            current_time += timedelta(hours=1)

        return data_points


class MultiAgentOrchestrator:
    """Orchestrates multiple agents for coordinated trading decisions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.orchestrator_id = str(uuid.uuid4())

        # Components
        self._message_bus = InMemoryMessageBus()
        self._data_provider = RealMarketDataProvider()
        self._agents: Dict[AgentRole, BaseAgent] = {}

        # MCP Server connections (would be real MCP clients)
        self._mcp_tools = {
            "risk": self._mock_risk_tool,
            "compliance": self._mock_compliance_tool,
            "provenance": self._mock_provenance_tool,
            "strategy_eval": self._mock_strategy_eval_tool,
            "quantconnect": self._mock_quantconnect_tool
        }

        # State tracking
        self._active_requests: Dict[str, OrchestrationRequest] = {}
        self._request_results: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Multi-agent orchestrator initialized: {self.orchestrator_id}")

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        agent.set_message_bus(self._message_bus)
        agent.set_data_provider(self._data_provider)
        self._agents[agent.role] = agent
        logger.info(f"Registered agent: {agent.role.value} ({agent.agent_id})")

    async def start(self) -> None:
        """Start the orchestrator and all registered agents."""
        logger.info("Starting multi-agent orchestrator")

        # Start all agents
        for agent in self._agents.values():
            await agent.start()

        logger.info(f"Started {len(self._agents)} agents")

    async def stop(self) -> None:
        """Stop the orchestrator and all agents."""
        logger.info("Stopping multi-agent orchestrator")

        # Stop all agents
        for agent in self._agents.values():
            await agent.stop()

    async def orchestrate_analysis(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Orchestrate a multi-agent analysis workflow."""
        logger.info(f"Starting orchestration for request {request.request_id}")

        start_time = datetime.now(timezone.utc)
        phase_timings = {}
        errors = []

        try:
            # Phase 1: Initialize
            phase_start = datetime.now(timezone.utc)
            await self._phase_initialization(request)
            phase_timings[OrchestrationPhase.INITIALIZATION.value] = (
                datetime.now(timezone.utc) - phase_start
            ).total_seconds()

            # Phase 2: Data Collection
            phase_start = datetime.now(timezone.utc)
            market_data = await self._phase_data_collection(request)
            phase_timings[OrchestrationPhase.DATA_COLLECTION.value] = (
                datetime.now(timezone.utc) - phase_start
            ).total_seconds()

            # Phase 3: Agent Analysis
            phase_start = datetime.now(timezone.utc)
            agent_results = await self._phase_agent_analysis(request)
            phase_timings[OrchestrationPhase.AGENT_ANALYSIS.value] = (
                datetime.now(timezone.utc) - phase_start
            ).total_seconds()

            # Phase 4: Signal Aggregation
            phase_start = datetime.now(timezone.utc)
            consensus_decisions = await self._phase_signal_aggregation(request, agent_results)
            phase_timings[OrchestrationPhase.SIGNAL_AGGREGATION.value] = (
                datetime.now(timezone.utc) - phase_start
            ).total_seconds()

            # Phase 5: Risk Validation
            phase_start = datetime.now(timezone.utc)
            risk_validated_decisions = await self._phase_risk_validation(consensus_decisions)
            phase_timings[OrchestrationPhase.RISK_VALIDATION.value] = (
                datetime.now(timezone.utc) - phase_start
            ).total_seconds()

            # Phase 6: Execution Planning
            phase_start = datetime.now(timezone.utc)
            execution_plan = await self._phase_execution_planning(risk_validated_decisions)
            phase_timings[OrchestrationPhase.EXECUTION_PLANNING.value] = (
                datetime.now(timezone.utc) - phase_start
            ).total_seconds()

            # Calculate performance metrics
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            performance_metrics = {
                "total_duration_seconds": total_time,
                "agents_used": len(agent_results),
                "symbols_analyzed": len(request.symbols),
                "consensus_decisions": len(consensus_decisions),
                "success_rate": (len(consensus_decisions) / len(request.symbols)) * 100 if request.symbols else 0
            }

            return OrchestrationResult(
                request_id=request.request_id,
                symbols=request.symbols,
                consensus_decisions=risk_validated_decisions,
                execution_plan=execution_plan,
                performance_metrics=performance_metrics,
                phase_timings=phase_timings,
                errors=errors
            )

        except Exception as e:
            logger.error(f"Orchestration failed for request {request.request_id}: {e}")
            errors.append(str(e))

            return OrchestrationResult(
                request_id=request.request_id,
                symbols=request.symbols,
                consensus_decisions=[],
                execution_plan={},
                performance_metrics={"error": True},
                phase_timings=phase_timings,
                errors=errors
            )

    async def _phase_initialization(self, request: OrchestrationRequest) -> None:
        """Initialize orchestration request."""
        self._active_requests[request.request_id] = request
        self._request_results[request.request_id] = {}

        # Log provenance
        await self._mcp_tools["provenance"]({
            "operation": "orchestration_start",
            "request_id": request.request_id,
            "symbols": request.symbols,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    async def _phase_data_collection(self, request: OrchestrationRequest) -> List[MarketData]:
        """Collect market data for analysis."""
        logger.info(f"Collecting data for {len(request.symbols)} symbols")
        return await self._data_provider.get_market_data(request.symbols)

    async def _phase_agent_analysis(self, request: OrchestrationRequest) -> Dict[AgentRole, AnalysisResult]:
        """Coordinate agent analysis."""
        logger.info(f"Running analysis with {len(self._agents)} agents")

        # Create analysis requests for each agent
        agent_requests = {}
        for role, agent in self._agents.items():
            agent_request = AnalysisRequest(
                request_id=f"{request.request_id}_{role.value}",
                symbols=request.symbols,
                analysis_type=role.value,
                parameters=request.parameters.get(role.value, {})
            )
            agent_requests[role] = agent_request

        # Run agent analyses in parallel
        tasks = []
        for role, agent_request in agent_requests.items():
            if role in self._agents:
                task = asyncio.create_task(
                    self._agents[role].analyze(agent_request)
                )
                tasks.append((role, task))

        # Collect results
        agent_results = {}
        for role, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=30.0)  # 30 second timeout
                agent_results[role] = result
                logger.info(f"Agent {role.value} completed analysis with {len(result.signals)} signals")
            except asyncio.TimeoutError:
                logger.warning(f"Agent {role.value} analysis timed out")
            except Exception as e:
                logger.error(f"Agent {role.value} analysis failed: {e}")

        return agent_results

    async def _phase_signal_aggregation(
        self,
        request: OrchestrationRequest,
        agent_results: Dict[AgentRole, AnalysisResult]
    ) -> List[AgentConsensus]:
        """Aggregate signals from multiple agents into consensus decisions."""
        logger.info("Aggregating agent signals into consensus")

        consensus_decisions = []

        for symbol in request.symbols:
            # Collect all signals for this symbol
            symbol_signals = []
            for role, result in agent_results.items():
                for signal in result.signals:
                    if signal.symbol == symbol:
                        symbol_signals.append(signal)

            if not symbol_signals:
                logger.warning(f"No signals generated for symbol {symbol}")
                continue

            # Calculate consensus
            consensus = self._calculate_consensus(symbol, symbol_signals)
            if consensus:
                consensus_decisions.append(consensus)

        return consensus_decisions

    def _calculate_consensus(self, symbol: str, signals: List[TradingSignal]) -> Optional[AgentConsensus]:
        """Calculate consensus from multiple agent signals."""
        if not signals:
            return None

        # Weight signals by confidence
        weighted_signals = []
        total_weight = 0

        for signal in signals:
            weight = signal.confidence.value
            weighted_signals.append((signal, weight))
            total_weight += weight

        if total_weight == 0:
            return None

        # Calculate weighted average signal direction
        weighted_direction = 0
        confidence_sum = 0

        for signal, weight in weighted_signals:
            normalized_weight = weight / total_weight
            weighted_direction += signal.signal.value * normalized_weight
            confidence_sum += weight

        # Determine consensus signal
        if weighted_direction > 0.3:
            consensus_signal = SignalDirection.BUY if weighted_direction < 0.7 else SignalDirection.STRONG_BUY
        elif weighted_direction < -0.3:
            consensus_signal = SignalDirection.SELL if weighted_direction > -0.7 else SignalDirection.STRONG_SELL
        else:
            consensus_signal = SignalDirection.HOLD

        # Calculate consensus confidence
        avg_confidence = confidence_sum / len(signals)
        if avg_confidence > 0.8:
            consensus_confidence = ConfidenceLevel.VERY_HIGH
        elif avg_confidence > 0.6:
            consensus_confidence = ConfidenceLevel.HIGH
        elif avg_confidence > 0.4:
            consensus_confidence = ConfidenceLevel.MEDIUM
        elif avg_confidence > 0.2:
            consensus_confidence = ConfidenceLevel.LOW
        else:
            consensus_confidence = ConfidenceLevel.VERY_LOW

        # Calculate agreement score (how much agents agree)
        signal_values = [s.signal.value for s in signals]
        if len(signal_values) > 1:
            agreement_score = 1.0 - (np.std(signal_values) / 1.0)  # Normalize by max possible std
            agreement_score = max(0.0, min(1.0, agreement_score))
        else:
            agreement_score = 1.0

        # Generate reasoning
        agent_names = [s.agent_role.value for s in signals]
        reasoning = f"Consensus from {len(signals)} agents ({', '.join(agent_names)}): "
        reasoning += f"weighted direction {weighted_direction:.2f}, agreement {agreement_score:.2f}"

        return AgentConsensus(
            symbol=symbol,
            consensus_signal=consensus_signal,
            consensus_confidence=consensus_confidence,
            agent_signals=signals,
            agreement_score=agreement_score,
            reasoning=reasoning
        )

    async def _phase_risk_validation(self, decisions: List[AgentConsensus]) -> List[AgentConsensus]:
        """Validate decisions against risk limits."""
        logger.info(f"Validating {len(decisions)} decisions against risk limits")

        validated_decisions = []

        for decision in decisions:
            # Use risk MCP tool
            risk_result = await self._mcp_tools["risk"]({
                "symbol": decision.symbol,
                "signal": decision.consensus_signal.value,
                "confidence": decision.consensus_confidence.value,
                "position_size": 0.1  # Example position size
            })

            if risk_result.get("approved", True):
                decision.risk_assessment = risk_result
                validated_decisions.append(decision)
                logger.info(f"Risk validation passed for {decision.symbol}")
            else:
                logger.warning(f"Risk validation failed for {decision.symbol}: {risk_result.get('reason')}")

        return validated_decisions

    async def _phase_execution_planning(self, decisions: List[AgentConsensus]) -> Dict[str, Any]:
        """Create execution plan for validated decisions."""
        logger.info(f"Creating execution plan for {len(decisions)} decisions")

        execution_plan = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_decisions": len(decisions),
            "orders": [],
            "risk_summary": {},
            "compliance_status": "pending"
        }

        for decision in decisions:
            if decision.consensus_signal != SignalDirection.HOLD:
                order = {
                    "symbol": decision.symbol,
                    "side": "buy" if decision.consensus_signal.value > 0 else "sell",
                    "signal_strength": abs(decision.consensus_signal.value),
                    "confidence": decision.consensus_confidence.value,
                    "agreement_score": decision.agreement_score,
                    "reasoning": decision.reasoning,
                    "agents_consensus": len(decision.agent_signals)
                }
                execution_plan["orders"].append(order)

        # Compliance check
        compliance_result = await self._mcp_tools["compliance"]({
            "orders": execution_plan["orders"],
            "check_type": "pre_trade"
        })
        execution_plan["compliance_status"] = compliance_result.get("status", "unknown")

        return execution_plan

    # Mock MCP tool implementations (would be real MCP calls)
    async def _mock_risk_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock risk assessment tool."""
        # Simple risk check based on position size and volatility
        return {
            "approved": True,
            "risk_score": 0.3,
            "var_impact": 0.02,
            "reason": "Within risk limits"
        }

    async def _mock_compliance_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock compliance check tool."""
        return {
            "status": "approved",
            "checks_passed": ["position_limits", "concentration", "regulatory"],
            "warnings": []
        }

    async def _mock_provenance_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock provenance logging tool."""
        return {
            "logged": True,
            "record_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _mock_strategy_eval_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock strategy evaluation tool."""
        return {
            "backtest_id": str(uuid.uuid4()),
            "status": "queued",
            "estimated_completion": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        }

    async def _mock_quantconnect_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock QuantConnect integration tool."""
        return {
            "project_id": str(uuid.uuid4()),
            "backtest_id": str(uuid.uuid4()),
            "status": "running",
            "progress": 0.15
        }

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            "orchestrator_id": self.orchestrator_id,
            "active_agents": len(self._agents),
            "agent_roles": [role.value for role in self._agents.keys()],
            "active_requests": len(self._active_requests),
            "mcp_tools_available": list(self._mcp_tools.keys())
        }


# Import numpy for calculations
try:
    import numpy as np
except ImportError:
    # Fallback implementation
    class np:
        @staticmethod
        def std(values):
            if not values:
                return 0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5


__all__ = [
    "MultiAgentOrchestrator",
    "OrchestrationRequest",
    "AgentConsensus",
    "OrchestrationResult",
    "OrchestrationPhase",
    "InMemoryMessageBus",
    "RealMarketDataProvider"
]