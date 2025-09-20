"""
Core agent framework for multi-agent trading system.
"""

from .base_agent import (
    BaseAgent,
    AgentRole,
    ConfidenceLevel,
    SignalDirection,
    MarketData,
    TradingSignal,
    AgentMessage,
    AnalysisRequest,
    AnalysisResult,
    MessageBus,
    DataProvider
)

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