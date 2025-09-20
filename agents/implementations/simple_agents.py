"""
Simplified agent implementations that don't require external dependencies.

These are minimal implementations for testing the orchestration system
without needing numpy, pandas, scipy, etc.
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.core import (
    BaseAgent, AgentRole, AnalysisRequest, AnalysisResult,
    TradingSignal, SignalDirection, ConfidenceLevel
)

logger = logging.getLogger(__name__)


class SimpleFundamentalAgent(BaseAgent):
    """Simplified fundamental agent for testing."""

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentRole.FUNDAMENTAL, config)

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Simplified fundamental analysis."""
        signals = []
        analysis_data = {}

        for symbol in request.symbols:
            # Simple mock fundamental analysis
            pe_ratio = 15 + (hash(symbol) % 20)  # PE between 15-35
            growth_rate = (hash(symbol + "growth") % 20) / 100  # 0-20% growth

            # Simple decision logic
            if pe_ratio < 20 and growth_rate > 0.1:
                signal_direction = SignalDirection.BUY
                confidence = ConfidenceLevel.HIGH
                reasoning = f"Low PE ({pe_ratio:.1f}) and strong growth ({growth_rate:.1%})"
            elif pe_ratio > 30:
                signal_direction = SignalDirection.SELL
                confidence = ConfidenceLevel.MEDIUM
                reasoning = f"High PE ({pe_ratio:.1f}) suggests overvaluation"
            else:
                signal_direction = SignalDirection.HOLD
                confidence = ConfidenceLevel.MEDIUM
                reasoning = f"Fair valuation: PE {pe_ratio:.1f}, growth {growth_rate:.1%}"

            signal = TradingSignal(
                agent_id=self.agent_id,
                agent_role=self.role,
                symbol=symbol,
                signal=signal_direction,
                confidence=confidence,
                reasoning=reasoning,
                supporting_data={"pe_ratio": pe_ratio, "growth_rate": growth_rate}
            )
            signals.append(signal)

            analysis_data[symbol] = {
                "pe_ratio": pe_ratio,
                "growth_rate": growth_rate,
                "valuation": "cheap" if pe_ratio < 20 else "expensive" if pe_ratio > 30 else "fair"
            }

        return AnalysisResult(
            request_id=request.request_id,
            agent_id=self.agent_id,
            symbols=request.symbols,
            signals=signals,
            analysis_data=analysis_data
        )


class SimpleTechnicalAgent(BaseAgent):
    """Simplified technical agent for testing."""

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentRole.TECHNICAL, config)

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Simplified technical analysis."""
        signals = []
        analysis_data = {}

        for symbol in request.symbols:
            # Simple mock technical indicators
            rsi = 30 + (hash(symbol + "rsi") % 40)  # RSI between 30-70
            ma_signal = (hash(symbol + "ma") % 3) - 1  # -1, 0, 1

            # Simple decision logic
            if rsi < 35 and ma_signal > 0:
                signal_direction = SignalDirection.BUY
                confidence = ConfidenceLevel.HIGH
                reasoning = f"Oversold RSI ({rsi:.1f}) with bullish MA crossover"
            elif rsi > 65 and ma_signal < 0:
                signal_direction = SignalDirection.SELL
                confidence = ConfidenceLevel.HIGH
                reasoning = f"Overbought RSI ({rsi:.1f}) with bearish MA crossover"
            elif ma_signal != 0:
                signal_direction = SignalDirection.BUY if ma_signal > 0 else SignalDirection.SELL
                confidence = ConfidenceLevel.MEDIUM
                reasoning = f"MA crossover signal: {'bullish' if ma_signal > 0 else 'bearish'}"
            else:
                signal_direction = SignalDirection.HOLD
                confidence = ConfidenceLevel.LOW
                reasoning = f"Neutral: RSI {rsi:.1f}, no clear MA signal"

            signal = TradingSignal(
                agent_id=self.agent_id,
                agent_role=self.role,
                symbol=symbol,
                signal=signal_direction,
                confidence=confidence,
                reasoning=reasoning,
                supporting_data={"rsi": rsi, "ma_signal": ma_signal}
            )
            signals.append(signal)

            analysis_data[symbol] = {
                "rsi": rsi,
                "ma_signal": ma_signal,
                "trend": "bullish" if ma_signal > 0 else "bearish" if ma_signal < 0 else "neutral"
            }

        return AnalysisResult(
            request_id=request.request_id,
            agent_id=self.agent_id,
            symbols=request.symbols,
            signals=signals,
            analysis_data=analysis_data
        )


class SimpleSentimentAgent(BaseAgent):
    """Simplified sentiment agent for testing."""

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentRole.SENTIMENT, config)

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Simplified sentiment analysis."""
        signals = []
        analysis_data = {}

        for symbol in request.symbols:
            # Simple mock sentiment scores
            news_sentiment = (hash(symbol + "news") % 100) / 100  # 0-1
            social_sentiment = (hash(symbol + "social") % 100) / 100  # 0-1
            analyst_sentiment = (hash(symbol + "analyst") % 100) / 100  # 0-1

            # Weighted average sentiment
            overall_sentiment = (news_sentiment * 0.4 + social_sentiment * 0.3 + analyst_sentiment * 0.3)

            # Simple decision logic
            if overall_sentiment > 0.7:
                signal_direction = SignalDirection.BUY
                confidence = ConfidenceLevel.HIGH
                reasoning = f"Strongly positive sentiment ({overall_sentiment:.2f})"
            elif overall_sentiment < 0.3:
                signal_direction = SignalDirection.SELL
                confidence = ConfidenceLevel.HIGH
                reasoning = f"Strongly negative sentiment ({overall_sentiment:.2f})"
            elif overall_sentiment > 0.6:
                signal_direction = SignalDirection.BUY
                confidence = ConfidenceLevel.MEDIUM
                reasoning = f"Positive sentiment ({overall_sentiment:.2f})"
            elif overall_sentiment < 0.4:
                signal_direction = SignalDirection.SELL
                confidence = ConfidenceLevel.MEDIUM
                reasoning = f"Negative sentiment ({overall_sentiment:.2f})"
            else:
                signal_direction = SignalDirection.HOLD
                confidence = ConfidenceLevel.LOW
                reasoning = f"Neutral sentiment ({overall_sentiment:.2f})"

            signal = TradingSignal(
                agent_id=self.agent_id,
                agent_role=self.role,
                symbol=symbol,
                signal=signal_direction,
                confidence=confidence,
                reasoning=reasoning,
                supporting_data={
                    "news_sentiment": news_sentiment,
                    "social_sentiment": social_sentiment,
                    "analyst_sentiment": analyst_sentiment,
                    "overall_sentiment": overall_sentiment
                }
            )
            signals.append(signal)

            analysis_data[symbol] = {
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
                "analyst_sentiment": analyst_sentiment,
                "overall_sentiment": overall_sentiment,
                "sentiment_category": "positive" if overall_sentiment > 0.6 else "negative" if overall_sentiment < 0.4 else "neutral"
            }

        return AnalysisResult(
            request_id=request.request_id,
            agent_id=self.agent_id,
            symbols=request.symbols,
            signals=signals,
            analysis_data=analysis_data
        )


class SimpleQuantitativeAgent(BaseAgent):
    """Simplified quantitative agent for testing."""

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentRole.QUANTITATIVE, config)

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Simplified quantitative analysis."""
        signals = []
        analysis_data = {}

        for symbol in request.symbols:
            # Simple mock factor scores
            momentum_score = (hash(symbol + "momentum") % 100) / 100  # 0-1
            value_score = (hash(symbol + "value") % 100) / 100  # 0-1
            quality_score = (hash(symbol + "quality") % 100) / 100  # 0-1

            # Combined quantitative score
            quant_score = (momentum_score * 0.4 + value_score * 0.3 + quality_score * 0.3)

            # Simple decision logic
            if quant_score > 0.75:
                signal_direction = SignalDirection.STRONG_BUY
                confidence = ConfidenceLevel.VERY_HIGH
                reasoning = f"Strong quantitative signals ({quant_score:.2f})"
            elif quant_score > 0.6:
                signal_direction = SignalDirection.BUY
                confidence = ConfidenceLevel.HIGH
                reasoning = f"Positive quantitative signals ({quant_score:.2f})"
            elif quant_score < 0.25:
                signal_direction = SignalDirection.STRONG_SELL
                confidence = ConfidenceLevel.VERY_HIGH
                reasoning = f"Weak quantitative signals ({quant_score:.2f})"
            elif quant_score < 0.4:
                signal_direction = SignalDirection.SELL
                confidence = ConfidenceLevel.HIGH
                reasoning = f"Negative quantitative signals ({quant_score:.2f})"
            else:
                signal_direction = SignalDirection.HOLD
                confidence = ConfidenceLevel.MEDIUM
                reasoning = f"Mixed quantitative signals ({quant_score:.2f})"

            signal = TradingSignal(
                agent_id=self.agent_id,
                agent_role=self.role,
                symbol=symbol,
                signal=signal_direction,
                confidence=confidence,
                reasoning=reasoning,
                supporting_data={
                    "momentum_score": momentum_score,
                    "value_score": value_score,
                    "quality_score": quality_score,
                    "combined_score": quant_score
                }
            )
            signals.append(signal)

            analysis_data[symbol] = {
                "momentum_score": momentum_score,
                "value_score": value_score,
                "quality_score": quality_score,
                "combined_score": quant_score,
                "quantitative_rating": "strong" if quant_score > 0.7 else "weak" if quant_score < 0.3 else "moderate"
            }

        return AnalysisResult(
            request_id=request.request_id,
            agent_id=self.agent_id,
            symbols=request.symbols,
            signals=signals,
            analysis_data=analysis_data
        )


__all__ = [
    "SimpleFundamentalAgent",
    "SimpleTechnicalAgent",
    "SimpleSentimentAgent",
    "SimpleQuantitativeAgent"
]