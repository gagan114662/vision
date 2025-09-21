"""
Fundamental analyst agent implementation.

Performs DCF modeling, earnings analysis, sector rotation, and valuation assessments
using financial data and economic indicators.
"""
from __future__ import annotations

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from agents.core import (
    BaseAgent, AgentRole, AnalysisRequest, AnalysisResult,
    TradingSignal, SignalDirection, ConfidenceLevel, MarketData
)

logger = logging.getLogger(__name__)


class FundamentalAgent(BaseAgent):
    """Fundamental analysis agent for long-term value assessment."""

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentRole.FUNDAMENTAL, config)

        # Configuration parameters
        self.dcf_growth_rate = self.config.get("dcf_growth_rate", 0.05)  # 5% default growth
        self.dcf_terminal_growth = self.config.get("dcf_terminal_growth", 0.02)  # 2% terminal growth
        self.discount_rate = self.config.get("discount_rate", 0.10)  # 10% WACC
        self.safety_margin = self.config.get("safety_margin", 0.20)  # 20% margin of safety

        # Valuation thresholds
        self.undervalued_threshold = self.config.get("undervalued_threshold", 0.80)  # 80% of fair value
        self.overvalued_threshold = self.config.get("overvalued_threshold", 1.20)  # 120% of fair value

        logger.info(f"Fundamental agent initialized with DCF growth: {self.dcf_growth_rate}")

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform fundamental analysis on requested symbols."""
        signals = []
        analysis_data = {}

        for symbol in request.symbols:
            try:
                # Get fundamental data
                fundamental_data = await self._get_fundamental_data(symbol)

                # Perform DCF valuation
                dcf_value = await self._calculate_dcf_value(symbol, fundamental_data)

                # Analyze earnings trends
                earnings_analysis = await self._analyze_earnings_trends(symbol, fundamental_data)

                # Assess sector positioning
                sector_analysis = await self._analyze_sector_position(symbol, fundamental_data)

                # Generate trading signal
                signal = await self._generate_fundamental_signal(
                    symbol, dcf_value, earnings_analysis, sector_analysis
                )

                if signal:
                    signals.append(signal)

                # Store analysis data
                analysis_data[symbol] = {
                    "dcf_valuation": dcf_value,
                    "earnings_analysis": earnings_analysis,
                    "sector_analysis": sector_analysis,
                    "current_price": fundamental_data.get("current_price"),
                    "fair_value": dcf_value.get("fair_value"),
                    "margin_of_safety": dcf_value.get("margin_of_safety")
                }

            except Exception as e:
                logger.error(f"Fundamental analysis failed for {symbol}: {e}")
                analysis_data[symbol] = {"error": str(e)}

        return AnalysisResult(
            request_id=request.request_id,
            agent_id=self.agent_id,
            symbols=request.symbols,
            signals=signals,
            analysis_data=analysis_data
        )

    async def _get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for a symbol."""
        # Get real current price from market data
        current_price = await self._get_current_price(symbol)

        # TODO: Integrate with real fundamental data providers
        # For now, use realistic mock data with real current price
        mock_data = {
            "current_price": current_price,
            "market_cap": 2.5e12,  # $2.5T
            "revenue_ttm": 400e9,   # $400B TTM revenue
            "revenue_growth_3y": 0.08,  # 8% 3-year average growth
            "net_income_ttm": 100e9,    # $100B net income
            "free_cash_flow_ttm": 90e9,  # $90B FCF
            "total_debt": 120e9,    # $120B total debt
            "cash_and_equivalents": 200e9,  # $200B cash
            "shares_outstanding": 16e9,  # 16B shares
            "book_value_per_share": 4.5,
            "roe": 0.28,  # 28% ROE
            "debt_to_equity": 0.21,
            "current_ratio": 1.07,
            "pe_ratio": 24.0,
            "peg_ratio": 1.2,
            "dividend_yield": 0.005,  # 0.5%
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "beta": 1.2,
            # Earnings history (quarterly)
            "earnings_history": [
                {"quarter": "Q1-2024", "eps": 1.53, "revenue": 110e9, "beat_estimate": True},
                {"quarter": "Q4-2023", "eps": 2.18, "revenue": 119e9, "beat_estimate": True},
                {"quarter": "Q3-2023", "eps": 1.46, "revenue": 89e9, "beat_estimate": False},
                {"quarter": "Q2-2023", "eps": 1.26, "revenue": 81e9, "beat_estimate": True},
            ]
        }

        return mock_data

    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        try:
            # Use market data provider to get real current price
            if hasattr(self, 'data_provider') and self.data_provider:
                market_data = await self.data_provider.get_market_data([symbol])
                if market_data and len(market_data) > 0:
                    price = market_data[0].price
                    logger.info(f"Retrieved real current price for {symbol}: ${price:.2f}")
                    return price
                else:
                    logger.warning(f"No market data available for {symbol}")

            # Fallback to deterministic mock price
            import hashlib
            seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
            fallback_price = 50 + (seed % 200)  # $50-$250 range
            logger.info(f"Using fallback price for {symbol}: ${fallback_price:.2f}")
            return float(fallback_price)

        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 150.0  # Default fallback

    async def _calculate_dcf_value(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Discounted Cash Flow valuation."""
        try:
            # Extract key metrics
            fcf_ttm = data["free_cash_flow_ttm"]
            shares_outstanding = data["shares_outstanding"]
            current_price = data["current_price"]

            # Project future cash flows (5-year explicit forecast)
            forecast_years = 5
            projected_fcf = []

            for year in range(1, forecast_years + 1):
                # Declining growth rate over time
                growth_rate = self.dcf_growth_rate * (0.9 ** (year - 1))
                fcf_year = fcf_ttm * ((1 + growth_rate) ** year)
                projected_fcf.append(fcf_year)

            # Terminal value calculation
            terminal_fcf = projected_fcf[-1] * (1 + self.dcf_terminal_growth)
            terminal_value = terminal_fcf / (self.discount_rate - self.dcf_terminal_growth)

            # Discount cash flows to present value
            pv_fcf = []
            for year, fcf in enumerate(projected_fcf, 1):
                pv = fcf / ((1 + self.discount_rate) ** year)
                pv_fcf.append(pv)

            # Discount terminal value
            pv_terminal = terminal_value / ((1 + self.discount_rate) ** forecast_years)

            # Enterprise value
            enterprise_value = sum(pv_fcf) + pv_terminal

            # Equity value (subtract net debt)
            net_debt = data["total_debt"] - data["cash_and_equivalents"]
            equity_value = enterprise_value - net_debt

            # Per share value
            fair_value_per_share = equity_value / shares_outstanding

            # Margin of safety
            margin_of_safety = (fair_value_per_share - current_price) / fair_value_per_share

            return {
                "fair_value": fair_value_per_share,
                "current_price": current_price,
                "margin_of_safety": margin_of_safety,
                "enterprise_value": enterprise_value,
                "projected_fcf": projected_fcf,
                "terminal_value": terminal_value,
                "discount_rate": self.discount_rate,
                "upside_potential": (fair_value_per_share / current_price) - 1
            }

        except Exception as e:
            logger.error(f"DCF calculation failed for {symbol}: {e}")
            return {"error": str(e)}

    async def _analyze_earnings_trends(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze earnings trends and quality."""
        try:
            earnings_history = data["earnings_history"]

            # Calculate earnings growth
            if len(earnings_history) >= 2:
                recent_eps = earnings_history[0]["eps"]
                prior_eps = earnings_history[-1]["eps"]
                earnings_growth = (recent_eps / prior_eps) - 1 if prior_eps != 0 else 0
            else:
                earnings_growth = 0

            # Beat rate calculation
            beats = sum(1 for e in earnings_history if e["beat_estimate"])
            beat_rate = beats / len(earnings_history) if earnings_history else 0

            # Revenue growth calculation
            revenue_growth_3y = data.get("revenue_growth_3y", 0)

            # Quality metrics
            roe = data.get("roe", 0)
            debt_to_equity = data.get("debt_to_equity", 0)
            current_ratio = data.get("current_ratio", 1)

            # Earnings quality score (0-100)
            quality_score = (
                (min(roe * 100, 30)) * 0.3 +  # ROE component (max 30 points)
                (min(revenue_growth_3y * 100, 20)) * 0.25 +  # Growth component (max 20 points)
                (beat_rate * 100) * 0.2 +  # Consistency component (max 20 points)
                (min(2.0 / max(debt_to_equity, 0.1), 1) * 15) * 0.15 +  # Leverage component (max 15 points)
                (min(current_ratio, 2.0) / 2.0 * 15) * 0.1  # Liquidity component (max 15 points)
            )

            return {
                "earnings_growth": earnings_growth,
                "beat_rate": beat_rate,
                "revenue_growth_3y": revenue_growth_3y,
                "roe": roe,
                "debt_to_equity": debt_to_equity,
                "current_ratio": current_ratio,
                "quality_score": min(quality_score, 100),
                "trend_direction": "positive" if earnings_growth > 0.05 else "negative" if earnings_growth < -0.05 else "neutral"
            }

        except Exception as e:
            logger.error(f"Earnings analysis failed for {symbol}: {e}")
            return {"error": str(e)}

    async def _analyze_sector_position(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sector positioning and relative valuation."""
        try:
            sector = data.get("sector", "Unknown")
            pe_ratio = data.get("pe_ratio", 0)
            peg_ratio = data.get("peg_ratio", 0)
            beta = data.get("beta", 1.0)

            # Sector-specific valuation benchmarks (simplified)
            sector_pe_benchmarks = {
                "Technology": 28.0,
                "Healthcare": 22.0,
                "Financials": 12.0,
                "Consumer Discretionary": 20.0,
                "Industrials": 18.0,
                "Energy": 15.0,
                "Utilities": 16.0,
                "Materials": 14.0,
                "Consumer Staples": 19.0,
                "Real Estate": 25.0,
                "Telecommunications": 13.0
            }

            sector_benchmark = sector_pe_benchmarks.get(sector, 18.0)
            relative_valuation = pe_ratio / sector_benchmark if sector_benchmark > 0 else 1.0

            # Sector momentum (simplified - would use real sector data)
            sector_momentum = "neutral"  # Would calculate from sector ETF performance

            return {
                "sector": sector,
                "pe_ratio": pe_ratio,
                "sector_pe_benchmark": sector_benchmark,
                "relative_valuation": relative_valuation,
                "peg_ratio": peg_ratio,
                "beta": beta,
                "sector_momentum": sector_momentum,
                "valuation_vs_sector": "undervalued" if relative_valuation < 0.9 else "overvalued" if relative_valuation > 1.1 else "fair"
            }

        except Exception as e:
            logger.error(f"Sector analysis failed for {symbol}: {e}")
            return {"error": str(e)}

    async def _generate_fundamental_signal(
        self,
        symbol: str,
        dcf_analysis: Dict[str, Any],
        earnings_analysis: Dict[str, Any],
        sector_analysis: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Generate trading signal based on fundamental analysis."""
        try:
            # Extract key metrics
            margin_of_safety = dcf_analysis.get("margin_of_safety", 0)
            quality_score = earnings_analysis.get("quality_score", 50)
            relative_valuation = sector_analysis.get("relative_valuation", 1.0)

            # Decision logic
            signal_strength = 0.0
            confidence_score = 0.0
            reasoning_parts = []

            # DCF-based assessment
            if margin_of_safety > self.safety_margin:
                signal_strength += 0.4
                reasoning_parts.append(f"Undervalued by DCF: {margin_of_safety:.1%} margin of safety")
            elif margin_of_safety < -self.safety_margin:
                signal_strength -= 0.4
                reasoning_parts.append(f"Overvalued by DCF: {abs(margin_of_safety):.1%} overvaluation")

            # Quality assessment
            if quality_score > 75:
                signal_strength += 0.2
                confidence_score += 0.3
                reasoning_parts.append(f"High quality score: {quality_score:.0f}/100")
            elif quality_score < 40:
                signal_strength -= 0.2
                confidence_score -= 0.2
                reasoning_parts.append(f"Low quality score: {quality_score:.0f}/100")

            # Sector relative valuation
            if relative_valuation < 0.8:
                signal_strength += 0.15
                reasoning_parts.append(f"Cheap vs sector: {relative_valuation:.2f}x relative P/E")
            elif relative_valuation > 1.3:
                signal_strength -= 0.15
                reasoning_parts.append(f"Expensive vs sector: {relative_valuation:.2f}x relative P/E")

            # Base confidence from analysis completeness
            confidence_score += 0.5

            # Determine signal direction
            if signal_strength > 0.3:
                signal_direction = SignalDirection.BUY if signal_strength < 0.6 else SignalDirection.STRONG_BUY
            elif signal_strength < -0.3:
                signal_direction = SignalDirection.SELL if signal_strength > -0.6 else SignalDirection.STRONG_SELL
            else:
                signal_direction = SignalDirection.HOLD

            # Determine confidence level
            confidence_score = min(max(confidence_score, 0.1), 0.9)

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
            if abs(signal_strength) < 0.2 or confidence_score < 0.3:
                return None

            reasoning = f"Fundamental analysis: {'; '.join(reasoning_parts)}"

            # Set target price based on DCF
            target_price = None
            if signal_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                fair_value = dcf_analysis.get("fair_value")
                if fair_value:
                    target_price = fair_value * 0.95  # Conservative target

            return TradingSignal(
                agent_id=self.agent_id,
                agent_role=self.role,
                symbol=symbol,
                signal=signal_direction,
                confidence=confidence,
                reasoning=reasoning,
                target_price=target_price,
                supporting_data={
                    "dcf_fair_value": dcf_analysis.get("fair_value"),
                    "margin_of_safety": margin_of_safety,
                    "quality_score": quality_score,
                    "relative_valuation": relative_valuation,
                    "signal_strength": signal_strength
                }
            )

        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return None


__all__ = ["FundamentalAgent"]