"""
Base agent framework for multi-agent trading system.

Provides the foundation for specialized trading agents with standardized
interfaces, communication protocols, and decision-making capabilities.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar, Union

from mcp.common.resilience import CircuitBreaker, retry, RetryConfig
from mcp.common.config import get_config

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AgentRole(Enum):
    """Standardized agent roles in the trading system."""
    FUNDAMENTAL = "fundamental_analyst"
    TECHNICAL = "technical_analyst"
    SENTIMENT = "sentiment_analyst"
    QUANTITATIVE = "quantitative_researcher"
    RISK_MANAGER = "risk_manager"
    EXECUTION = "execution_trader"
    COORDINATOR = "portfolio_coordinator"


class ConfidenceLevel(Enum):
    """Confidence levels for agent decisions."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


class SignalDirection(Enum):
    """Trading signal directions."""
    STRONG_BUY = 1.0
    BUY = 0.5
    HOLD = 0.0
    SELL = -0.5
    STRONG_SELL = -1.0


@dataclass
class MarketData:
    """Standardized market data structure."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingSignal:
    """Trading signal with confidence and reasoning."""
    agent_id: str
    agent_role: AgentRole
    symbol: str
    signal: SignalDirection
    confidence: ConfidenceLevel
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    expiry_time: Optional[datetime] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class AgentMessage:
    """Inter-agent communication message."""
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None


@dataclass
class AnalysisRequest:
    """Request for agent analysis."""
    request_id: str
    symbols: List[str]
    analysis_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    priority: int = 0  # Higher numbers = higher priority


@dataclass
class AnalysisResult:
    """Result from agent analysis."""
    request_id: str
    agent_id: str
    symbols: List[str]
    signals: List[TradingSignal]
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


class MessageBus(Protocol):
    """Protocol for agent message communication."""

    async def publish(self, message: AgentMessage) -> None:
        """Publish message to the bus."""
        ...

    async def subscribe(self, agent_id: str) -> AsyncIterator[AgentMessage]:
        """Subscribe to messages for an agent."""
        ...


class DataProvider(Protocol):
    """Protocol for market data provision."""

    async def get_market_data(self, symbols: List[str]) -> List[MarketData]:
        """Get current market data for symbols."""
        ...

    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketData]:
        """Get historical market data."""
        ...


class BaseAgent(ABC):
    """Base class for all trading agents."""

    def __init__(
        self,
        agent_id: Optional[str] = None,
        role: Optional[AgentRole] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.role = role or AgentRole.FUNDAMENTAL
        self.config = config or {}
        self.is_active = False
        self._circuit_breaker = CircuitBreaker(f"agent_{self.agent_id}")
        self._message_bus: Optional[MessageBus] = None
        self._data_provider: Optional[DataProvider] = None
        self._processing_queue: asyncio.Queue[AnalysisRequest] = asyncio.Queue()

        # Performance metrics
        self.total_analyses = 0
        self.successful_analyses = 0
        self.average_processing_time = 0.0
        self.last_analysis_time: Optional[datetime] = None

        logger.info(f"Initialized {self.role.value} agent: {self.agent_id}")

    def set_message_bus(self, message_bus: MessageBus) -> None:
        """Set the message bus for inter-agent communication."""
        self._message_bus = message_bus

    def set_data_provider(self, data_provider: DataProvider) -> None:
        """Set the data provider for market data access."""
        self._data_provider = data_provider

    async def start(self) -> None:
        """Start the agent and begin processing."""
        self.is_active = True
        logger.info(f"Starting agent {self.agent_id} ({self.role.value})")

        # Start message processing task
        asyncio.create_task(self._process_messages())
        asyncio.create_task(self._process_analysis_queue())

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self.is_active = False
        logger.info(f"Stopping agent {self.agent_id} ({self.role.value})")

    @abstractmethod
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform agent-specific analysis."""
        pass

    async def _process_messages(self) -> None:
        """Process incoming messages from the message bus."""
        if not self._message_bus:
            return

        async for message in self._message_bus.subscribe(self.agent_id):
            if not self.is_active:
                break

            try:
                await self._handle_message(message)
            except Exception as e:
                logger.error(f"Error processing message in agent {self.agent_id}: {e}")

    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle incoming message based on type."""
        message_type = message.message_type

        if message_type == "analysis_request":
            request = AnalysisRequest(**message.content)
            await self._processing_queue.put(request)

        elif message_type == "market_data_update":
            await self._handle_market_data_update(message.content)

        elif message_type == "signal_update":
            await self._handle_signal_update(message.content)

        else:
            logger.warning(f"Unknown message type: {message_type}")

    async def _process_analysis_queue(self) -> None:
        """Process analysis requests from the queue."""
        while self.is_active:
            try:
                # Wait for requests with timeout
                request = await asyncio.wait_for(
                    self._processing_queue.get(),
                    timeout=1.0
                )

                await self._process_analysis_request(request)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in analysis queue processing: {e}")

    async def _process_analysis_request(self, request: AnalysisRequest) -> None:
        """Process a single analysis request with resilience."""
        start_time = time.time()

        try:
            # Use circuit breaker for resilience
            result = await self._circuit_breaker.call_async(
                self.analyze, request
            )

            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time

            self._update_performance_metrics(processing_time, success=True)

            # Send result back via message bus
            if self._message_bus:
                response_message = AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=None,  # Broadcast result
                    message_type="analysis_result",
                    content=result.__dict__,
                    correlation_id=request.request_id
                )
                await self._message_bus.publish(response_message)

        except Exception as e:
            logger.error(f"Analysis failed for agent {self.agent_id}: {e}")
            self._update_performance_metrics(0.0, success=False)

            # Send error response
            if self._message_bus:
                error_result = AnalysisResult(
                    request_id=request.request_id,
                    agent_id=self.agent_id,
                    symbols=request.symbols,
                    signals=[],
                    errors=[str(e)]
                )
                error_message = AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=None,
                    message_type="analysis_error",
                    content=error_result.__dict__,
                    correlation_id=request.request_id
                )
                await self._message_bus.publish(error_message)

    def _update_performance_metrics(self, processing_time_ms: float, success: bool) -> None:
        """Update agent performance metrics."""
        self.total_analyses += 1
        if success:
            self.successful_analyses += 1

        # Update rolling average processing time
        if self.average_processing_time == 0:
            self.average_processing_time = processing_time_ms
        else:
            # Simple exponential moving average
            alpha = 0.1
            self.average_processing_time = (
                alpha * processing_time_ms +
                (1 - alpha) * self.average_processing_time
            )

        self.last_analysis_time = datetime.now(timezone.utc)

    async def _handle_market_data_update(self, data: Dict[str, Any]) -> None:
        """Handle market data updates."""
        # Default implementation - can be overridden by specific agents
        pass

    async def _handle_signal_update(self, data: Dict[str, Any]) -> None:
        """Handle trading signal updates from other agents."""
        # Default implementation - can be overridden by specific agents
        pass

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        success_rate = (
            self.successful_analyses / max(1, self.total_analyses)
        ) * 100

        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "is_active": self.is_active,
            "total_analyses": self.total_analyses,
            "successful_analyses": self.successful_analyses,
            "success_rate_percent": round(success_rate, 2),
            "average_processing_time_ms": round(self.average_processing_time, 2),
            "last_analysis_time": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "circuit_breaker_stats": self._circuit_breaker.get_stats()
        }

    @retry(RetryConfig(max_attempts=3))
    async def get_market_data(self, symbols: List[str]) -> List[MarketData]:
        """Get market data with retry logic."""
        if not self._data_provider:
            raise RuntimeError("No data provider configured")

        return await self._data_provider.get_market_data(symbols)

    async def send_message(self, recipient_id: Optional[str], message_type: str, content: Dict[str, Any]) -> None:
        """Send message to another agent or broadcast."""
        if not self._message_bus:
            raise RuntimeError("No message bus configured")

        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content
        )

        await self._message_bus.publish(message)

    def __repr__(self) -> str:
        return f"BaseAgent(id={self.agent_id}, role={self.role.value}, active={self.is_active})"


__all__ = [
    "BaseAgent",
    "AgentRole",
    "ConfidenceLevel",
    "SignalDirection",
    "MarketData",
    "TradingSignal",
    "AgentMessage",
    "AnalysisRequest",
    "AnalysisResult",
    "MessageBus",
    "DataProvider"
]