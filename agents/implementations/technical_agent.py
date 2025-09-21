"""
Technical analyst agent implementation.

Performs chart pattern recognition, momentum analysis, support/resistance identification,
and multi-timeframe technical indicator analysis.
"""
from __future__ import annotations

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from agents.core import (
    BaseAgent, AgentRole, AnalysisRequest, AnalysisResult,
    TradingSignal, SignalDirection, ConfidenceLevel, MarketData
)
from mcp.market_data.real_data_provider import RealMarketDataProvider

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """Container for technical indicator values."""
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_middle: float
    atr_14: float
    stoch_k: float
    stoch_d: float
    obv: float
    volume_sma_20: float


@dataclass
class PatternSignal:
    """Chart pattern recognition result."""
    pattern_name: str
    confidence: float
    direction: str  # "bullish", "bearish", "neutral"
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    time_horizon: str = "short"  # "short", "medium", "long"


class TechnicalAgent(BaseAgent):
    """Technical analysis agent for pattern recognition and momentum analysis."""

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentRole.TECHNICAL, config)

        # Configuration parameters
        self.lookback_periods = self.config.get("lookback_periods", 200)  # Bars to analyze
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.volume_threshold = self.config.get("volume_threshold", 1.5)  # Volume spike threshold

        # Pattern recognition settings
        self.pattern_min_confidence = self.config.get("pattern_min_confidence", 0.6)
        self.trend_strength_threshold = self.config.get("trend_strength_threshold", 0.3)

        # Real market data provider (replaces synthetic data generation)
        self.data_provider = RealMarketDataProvider()

        logger.info(f"Technical agent initialized with {self.lookback_periods} lookback periods and real market data provider")

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform technical analysis on requested symbols."""
        signals = []
        analysis_data = {}

        for symbol in request.symbols:
            try:
                # Get price data
                price_data = await self._get_price_data(symbol)

                # Calculate technical indicators
                indicators = await self._calculate_indicators(price_data)

                # Perform pattern recognition
                patterns = await self._detect_patterns(price_data, indicators)

                # Analyze momentum and trend
                momentum_analysis = await self._analyze_momentum(price_data, indicators)

                # Identify support and resistance
                support_resistance = await self._identify_support_resistance(price_data)

                # Generate trading signal
                signal = await self._generate_technical_signal(
                    symbol, indicators, patterns, momentum_analysis, support_resistance
                )

                if signal:
                    signals.append(signal)

                # Store analysis data
                analysis_data[symbol] = {
                    "indicators": indicators.__dict__ if indicators else {},
                    "patterns": [p.__dict__ for p in patterns],
                    "momentum_analysis": momentum_analysis,
                    "support_resistance": support_resistance,
                    "current_price": price_data[-1]["close"] if price_data else None
                }

            except Exception as e:
                logger.error(f"Technical analysis failed for {symbol}: {e}")
                analysis_data[symbol] = {"error": str(e)}

        return AnalysisResult(
            request_id=request.request_id,
            agent_id=self.agent_id,
            symbols=request.symbols,
            signals=signals,
            analysis_data=analysis_data
        )

    async def _get_price_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get real price data for technical analysis using RealMarketDataProvider."""
        try:
            # Calculate date range for lookback
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_periods + 30)  # Extra buffer for weekends/holidays

            logger.info(f"Fetching real market data for {symbol} from {start_date.date()} to {end_date.date()}")

            # Get real market data
            stock_data = await self.data_provider.get_stock_data([symbol], start_date, end_date)

            if symbol not in stock_data or not stock_data[symbol].get('prices'):
                logger.warning(f"No real data available for {symbol}, using fallback data")
                return self._generate_fallback_data(symbol)

            # Convert real data to technical analysis format
            symbol_data = stock_data[symbol]
            prices = symbol_data['prices']
            volumes = symbol_data.get('volumes', [1000000] * len(prices))  # Default volume if not available
            dates = symbol_data.get('dates', [])

            # Take only the most recent lookback_periods
            data_length = min(len(prices), self.lookback_periods)
            prices = prices[-data_length:]
            volumes = volumes[-data_length:] if len(volumes) >= data_length else volumes

            data = []
            for i in range(data_length):
                # For simplicity, treat prices as close prices and generate OHLC
                close_price = prices[i]

                # Simple OHLC generation (in real system, would get actual OHLC)
                volatility = 0.01  # 1% intraday volatility assumption
                high = close_price * (1 + volatility)
                low = close_price * (1 - volatility)
                open_price = prices[i-1] if i > 0 else close_price

                # Parse date if available
                if i < len(dates):
                    try:
                        timestamp = datetime.strptime(dates[i], "%Y-%m-%d")
                    except (ValueError, TypeError):
                        timestamp = end_date - timedelta(days=data_length - i)
                else:
                    timestamp = end_date - timedelta(days=data_length - i)

                volume = volumes[i] if i < len(volumes) else 1000000

                data.append({
                    "timestamp": timestamp,
                    "open": float(open_price),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close_price),
                    "volume": int(volume)
                })

            logger.info(f"Retrieved {len(data)} real data points for {symbol} from {symbol_data.get('source', 'unknown')}")
            return data

        except Exception as e:
            logger.error(f"Error fetching real market data for {symbol}: {e}")
            return self._generate_fallback_data(symbol)

    def _generate_fallback_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate fallback data when real data is unavailable."""
        logger.warning(f"Using fallback synthetic data for {symbol}")

        # Generate sample OHLCV data (kept minimal for fallback only)
        np.random.seed(hash(symbol) % 2147483647)  # Deterministic but symbol-specific

        base_price = 100.0
        data = []

        for i in range(min(self.lookback_periods, 50)):  # Limit fallback data
            # Random walk with trend
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            trend = 0.0005  # Slight upward trend

            if i == 0:
                price = base_price
            else:
                price = data[-1]["close"] * (1 + change + trend)

            # Generate OHLC from close
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = data[-1]["close"] if i > 0 else price

            # Ensure OHLC consistency
            high = max(high, open_price, price)
            low = min(low, open_price, price)

            volume = int(np.random.lognormal(12, 0.5))  # Log-normal volume distribution

            data.append({
                "timestamp": datetime.now() - timedelta(days=len(data) + 1),
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume
            })

        return data

    async def _calculate_indicators(self, price_data: List[Dict[str, Any]]) -> Optional[TechnicalIndicators]:
        """Calculate technical indicators from price data."""
        try:
            if len(price_data) < 50:  # Need minimum data
                return None

            # Convert to pandas for easier calculation
            df = pd.DataFrame(price_data)
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])

            # Simple Moving Averages
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            sma_200 = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else sma_50

            # Exponential Moving Averages
            ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = df['close'].ewm(span=26).mean().iloc[-1]

            # RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]

            # MACD
            ema_12_series = df['close'].ewm(span=12).mean()
            ema_26_series = df['close'].ewm(span=26).mean()
            macd_line = (ema_12_series - ema_26_series).iloc[-1]
            macd_signal = (ema_12_series - ema_26_series).ewm(span=9).mean().iloc[-1]
            macd_histogram = macd_line - macd_signal

            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = df['close'].rolling(bb_period).mean().iloc[-1]
            bb_std_val = df['close'].rolling(bb_period).std().iloc[-1]
            bollinger_upper = bb_middle + (bb_std * bb_std_val)
            bollinger_lower = bb_middle - (bb_std * bb_std_val)

            # Average True Range
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = true_range.rolling(14).mean().iloc[-1]

            # Stochastic Oscillator
            lowest_low = df['low'].rolling(14).min()
            highest_high = df['high'].rolling(14).max()
            stoch_k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            stoch_d = stoch_k.rolling(3).mean()
            stoch_k_val = stoch_k.iloc[-1]
            stoch_d_val = stoch_d.iloc[-1]

            # On-Balance Volume (simplified)
            obv = ((df['close'].diff() > 0).astype(int) * df['volume']).cumsum().iloc[-1]

            # Volume SMA
            volume_sma_20 = df['volume'].rolling(20).mean().iloc[-1]

            return TechnicalIndicators(
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                ema_12=ema_12,
                ema_26=ema_26,
                rsi_14=rsi_14,
                macd_line=macd_line,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                bollinger_upper=bollinger_upper,
                bollinger_lower=bollinger_lower,
                bollinger_middle=bb_middle,
                atr_14=atr_14,
                stoch_k=stoch_k_val,
                stoch_d=stoch_d_val,
                obv=obv,
                volume_sma_20=volume_sma_20
            )

        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            return None

    async def _detect_patterns(self, price_data: List[Dict[str, Any]], indicators: Optional[TechnicalIndicators]) -> List[PatternSignal]:
        """Detect chart patterns."""
        patterns = []

        try:
            if not indicators or len(price_data) < 50:
                return patterns

            current_price = price_data[-1]["close"]
            recent_prices = [d["close"] for d in price_data[-20:]]  # Last 20 periods

            # Moving Average Crossover Patterns
            if indicators.sma_20 > indicators.sma_50:
                if abs(indicators.sma_20 - indicators.sma_50) / indicators.sma_50 > 0.02:  # 2% separation
                    patterns.append(PatternSignal(
                        pattern_name="Golden Cross (20/50)",
                        confidence=0.7,
                        direction="bullish",
                        target_price=current_price * 1.05,
                        time_horizon="medium"
                    ))

            # MACD Signal
            if indicators.macd_line > indicators.macd_signal and indicators.macd_histogram > 0:
                patterns.append(PatternSignal(
                    pattern_name="MACD Bullish Crossover",
                    confidence=0.6,
                    direction="bullish",
                    time_horizon="short"
                ))

            # Bollinger Band Squeeze/Breakout
            bb_squeeze = (indicators.bollinger_upper - indicators.bollinger_lower) / indicators.bollinger_middle
            if bb_squeeze < 0.1:  # Tight bands indicate low volatility
                if current_price > indicators.bollinger_upper:
                    patterns.append(PatternSignal(
                        pattern_name="Bollinger Band Breakout (Upper)",
                        confidence=0.75,
                        direction="bullish",
                        target_price=current_price * 1.03,
                        stop_loss=indicators.bollinger_middle,
                        time_horizon="short"
                    ))
                elif current_price < indicators.bollinger_lower:
                    patterns.append(PatternSignal(
                        pattern_name="Bollinger Band Breakout (Lower)",
                        confidence=0.75,
                        direction="bearish",
                        target_price=current_price * 0.97,
                        stop_loss=indicators.bollinger_middle,
                        time_horizon="short"
                    ))

            # RSI Divergence (simplified)
            if indicators.rsi_14 > 70:
                patterns.append(PatternSignal(
                    pattern_name="RSI Overbought",
                    confidence=0.5,
                    direction="bearish",
                    time_horizon="short"
                ))
            elif indicators.rsi_14 < 30:
                patterns.append(PatternSignal(
                    pattern_name="RSI Oversold",
                    confidence=0.5,
                    direction="bullish",
                    time_horizon="short"
                ))

            # Simple trend detection
            short_trend = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5]
            medium_trend = (recent_prices[-1] - recent_prices[-10]) / recent_prices[-10]

            if short_trend > 0.02 and medium_trend > 0.03:  # Strong uptrend
                patterns.append(PatternSignal(
                    pattern_name="Strong Uptrend",
                    confidence=0.8,
                    direction="bullish",
                    time_horizon="medium"
                ))
            elif short_trend < -0.02 and medium_trend < -0.03:  # Strong downtrend
                patterns.append(PatternSignal(
                    pattern_name="Strong Downtrend",
                    confidence=0.8,
                    direction="bearish",
                    time_horizon="medium"
                ))

        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")

        return patterns

    async def _analyze_momentum(self, price_data: List[Dict[str, Any]], indicators: Optional[TechnicalIndicators]) -> Dict[str, Any]:
        """Analyze momentum indicators."""
        try:
            if not indicators or len(price_data) < 20:
                return {"error": "Insufficient data for momentum analysis"}

            current_price = price_data[-1]["close"]
            current_volume = price_data[-1]["volume"]

            # Price momentum
            price_change_1d = (current_price - price_data[-2]["close"]) / price_data[-2]["close"]
            price_change_5d = (current_price - price_data[-6]["close"]) / price_data[-6]["close"]
            price_change_20d = (current_price - price_data[-21]["close"]) / price_data[-21]["close"]

            # Volume analysis
            volume_ratio = current_volume / indicators.volume_sma_20
            volume_trend = "increasing" if volume_ratio > 1.2 else "decreasing" if volume_ratio < 0.8 else "normal"

            # RSI momentum
            rsi_momentum = "overbought" if indicators.rsi_14 > self.rsi_overbought else \
                          "oversold" if indicators.rsi_14 < self.rsi_oversold else "neutral"

            # MACD momentum
            macd_momentum = "bullish" if indicators.macd_line > indicators.macd_signal else "bearish"

            # Stochastic momentum
            stoch_momentum = "overbought" if indicators.stoch_k > 80 else \
                            "oversold" if indicators.stoch_k < 20 else "neutral"

            # Overall momentum score (-1 to 1)
            momentum_score = 0
            momentum_score += 0.3 if price_change_5d > 0.02 else -0.3 if price_change_5d < -0.02 else 0
            momentum_score += 0.2 if indicators.macd_line > indicators.macd_signal else -0.2
            momentum_score += 0.2 if 30 < indicators.rsi_14 < 70 else 0  # Healthy RSI range
            momentum_score += 0.1 if volume_ratio > 1.1 else -0.1 if volume_ratio < 0.9 else 0
            momentum_score += 0.2 if indicators.sma_20 > indicators.sma_50 else -0.2

            return {
                "price_change_1d": price_change_1d,
                "price_change_5d": price_change_5d,
                "price_change_20d": price_change_20d,
                "volume_ratio": volume_ratio,
                "volume_trend": volume_trend,
                "rsi_momentum": rsi_momentum,
                "macd_momentum": macd_momentum,
                "stoch_momentum": stoch_momentum,
                "momentum_score": momentum_score,
                "momentum_direction": "bullish" if momentum_score > 0.2 else "bearish" if momentum_score < -0.2 else "neutral"
            }

        except Exception as e:
            logger.error(f"Momentum analysis failed: {e}")
            return {"error": str(e)}

    async def _identify_support_resistance(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify support and resistance levels."""
        try:
            if len(price_data) < 50:
                return {"error": "Insufficient data for support/resistance analysis"}

            # Extract price arrays
            highs = np.array([d["high"] for d in price_data[-50:]])  # Last 50 periods
            lows = np.array([d["low"] for d in price_data[-50:]])
            closes = np.array([d["close"] for d in price_data[-50:]])

            current_price = closes[-1]

            # Simple support/resistance identification using pivot points
            def find_pivot_points(prices: np.ndarray, window: int = 5) -> List[float]:
                pivots = []
                for i in range(window, len(prices) - window):
                    if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
                       all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                        pivots.append(prices[i])  # Resistance
                    elif all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
                         all(prices[i] <= prices[i+j] for j in range(1, window+1)):
                        pivots.append(prices[i])  # Support
                return pivots

            resistance_levels = find_pivot_points(highs)
            support_levels = find_pivot_points(lows)

            # Filter levels close to current price
            nearby_resistance = [r for r in resistance_levels if current_price < r < current_price * 1.1]
            nearby_support = [s for s in support_levels if current_price * 0.9 < s < current_price]

            # Find nearest levels
            nearest_resistance = min(nearby_resistance) if nearby_resistance else None
            nearest_support = max(nearby_support) if nearby_support else None

            return {
                "current_price": current_price,
                "nearest_resistance": nearest_resistance,
                "nearest_support": nearest_support,
                "resistance_levels": sorted(resistance_levels, reverse=True)[:3],  # Top 3
                "support_levels": sorted(support_levels, reverse=True)[:3],  # Top 3
                "distance_to_resistance": (nearest_resistance / current_price - 1) if nearest_resistance else None,
                "distance_to_support": (1 - nearest_support / current_price) if nearest_support else None
            }

        except Exception as e:
            logger.error(f"Support/resistance analysis failed: {e}")
            return {"error": str(e)}

    async def _generate_technical_signal(
        self,
        symbol: str,
        indicators: Optional[TechnicalIndicators],
        patterns: List[PatternSignal],
        momentum_analysis: Dict[str, Any],
        support_resistance: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Generate trading signal based on technical analysis."""
        try:
            if not indicators:
                return None

            signal_strength = 0.0
            confidence_score = 0.0
            reasoning_parts = []

            # Pattern-based signals
            bullish_patterns = [p for p in patterns if p.direction == "bullish" and p.confidence >= self.pattern_min_confidence]
            bearish_patterns = [p for p in patterns if p.direction == "bearish" and p.confidence >= self.pattern_min_confidence]

            if bullish_patterns:
                pattern_strength = sum(p.confidence for p in bullish_patterns) / len(bullish_patterns)
                signal_strength += pattern_strength * 0.4
                reasoning_parts.append(f"Bullish patterns: {', '.join(p.pattern_name for p in bullish_patterns)}")

            if bearish_patterns:
                pattern_strength = sum(p.confidence for p in bearish_patterns) / len(bearish_patterns)
                signal_strength -= pattern_strength * 0.4
                reasoning_parts.append(f"Bearish patterns: {', '.join(p.pattern_name for p in bearish_patterns)}")

            # Momentum signals
            momentum_score = momentum_analysis.get("momentum_score", 0)
            signal_strength += momentum_score * 0.3

            if abs(momentum_score) > 0.2:
                reasoning_parts.append(f"Momentum: {momentum_analysis.get('momentum_direction', 'neutral')}")

            # Moving average signals
            ma_signal = 0
            if indicators.sma_20 > indicators.sma_50:
                ma_signal += 0.15
                reasoning_parts.append("SMA20 > SMA50 (bullish)")
            else:
                ma_signal -= 0.15

            if indicators.sma_50 > indicators.sma_200:
                ma_signal += 0.1
                reasoning_parts.append("SMA50 > SMA200 (long-term bullish)")
            else:
                ma_signal -= 0.1

            signal_strength += ma_signal

            # RSI signals
            if indicators.rsi_14 < self.rsi_oversold:
                signal_strength += 0.2
                reasoning_parts.append(f"RSI oversold: {indicators.rsi_14:.1f}")
            elif indicators.rsi_14 > self.rsi_overbought:
                signal_strength -= 0.2
                reasoning_parts.append(f"RSI overbought: {indicators.rsi_14:.1f}")

            # Volume confirmation
            volume_ratio = momentum_analysis.get("volume_ratio", 1.0)
            if volume_ratio > self.volume_threshold and abs(signal_strength) > 0.2:
                signal_strength *= 1.2  # Amplify signal with volume confirmation
                reasoning_parts.append(f"Volume confirmation: {volume_ratio:.1f}x average")

            # Support/resistance context
            distance_to_resistance = support_resistance.get("distance_to_resistance")
            distance_to_support = support_resistance.get("distance_to_support")

            if distance_to_resistance and distance_to_resistance < 0.02:  # Near resistance
                signal_strength -= 0.1
                reasoning_parts.append("Near resistance level")
            elif distance_to_support and distance_to_support < 0.02:  # Near support
                signal_strength += 0.1
                reasoning_parts.append("Near support level")

            # Base confidence
            confidence_score = 0.6

            # Adjust confidence based on signal strength and confirmations
            if abs(signal_strength) > 0.5:
                confidence_score += 0.2
            if len(reasoning_parts) >= 3:  # Multiple confirmations
                confidence_score += 0.1
            if volume_ratio > self.volume_threshold:
                confidence_score += 0.1

            confidence_score = min(max(confidence_score, 0.1), 0.9)

            # Determine signal direction
            if signal_strength > 0.3:
                signal_direction = SignalDirection.BUY if signal_strength < 0.6 else SignalDirection.STRONG_BUY
            elif signal_strength < -0.3:
                signal_direction = SignalDirection.SELL if signal_strength > -0.6 else SignalDirection.STRONG_SELL
            else:
                signal_direction = SignalDirection.HOLD

            # Determine confidence level
            if confidence_score > 0.8:
                confidence = ConfidenceLevel.VERY_HIGH
            elif confidence_score > 0.6:
                confidence = ConfidenceLevel.HIGH
            elif confidence_score > 0.4:
                confidence = ConfidenceLevel.MEDIUM
            elif confidence_score > 0.2:
                confidence = ConfidenceLevel.LOW
            else:
                confidence = ConfidenceLevel.VERY_LOW

            # Only generate signal if we have meaningful conviction
            if abs(signal_strength) < 0.25 or confidence_score < 0.4:
                return None

            reasoning = f"Technical analysis: {'; '.join(reasoning_parts)}"

            # Set target price based on patterns or indicators
            current_price = support_resistance.get("current_price")
            target_price = None
            stop_loss = None

            if signal_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                # Look for bullish pattern targets
                pattern_targets = [p.target_price for p in bullish_patterns if p.target_price]
                if pattern_targets:
                    target_price = max(pattern_targets)
                elif current_price:
                    target_price = current_price * (1 + indicators.atr_14 / current_price * 2)  # 2x ATR target

                # Set stop loss below support or using ATR
                nearest_support = support_resistance.get("nearest_support")
                if nearest_support and current_price:
                    stop_loss = min(nearest_support * 0.98, current_price * (1 - indicators.atr_14 / current_price))

            elif signal_direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
                # Look for bearish pattern targets
                pattern_targets = [p.target_price for p in bearish_patterns if p.target_price]
                if pattern_targets:
                    target_price = min(pattern_targets)
                elif current_price:
                    target_price = current_price * (1 - indicators.atr_14 / current_price * 2)  # 2x ATR target

                # Set stop loss above resistance or using ATR
                nearest_resistance = support_resistance.get("nearest_resistance")
                if nearest_resistance and current_price:
                    stop_loss = max(nearest_resistance * 1.02, current_price * (1 + indicators.atr_14 / current_price))

            return TradingSignal(
                agent_id=self.agent_id,
                agent_role=self.role,
                symbol=symbol,
                signal=signal_direction,
                confidence=confidence,
                reasoning=reasoning,
                target_price=target_price,
                stop_loss=stop_loss,
                supporting_data={
                    "signal_strength": signal_strength,
                    "pattern_count": len(patterns),
                    "momentum_score": momentum_score,
                    "rsi": indicators.rsi_14,
                    "volume_ratio": volume_ratio,
                    "near_support": distance_to_support is not None and distance_to_support < 0.02,
                    "near_resistance": distance_to_resistance is not None and distance_to_resistance < 0.02
                }
            )

        except Exception as e:
            logger.error(f"Technical signal generation failed for {symbol}: {e}")
            return None


__all__ = ["TechnicalAgent", "TechnicalIndicators", "PatternSignal"]