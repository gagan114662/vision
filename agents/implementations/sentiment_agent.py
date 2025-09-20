"""
Sentiment analyst agent implementation.

Performs news sentiment analysis, social media monitoring, market psychology assessment,
and fear/greed indicator calculation for trading decisions.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from agents.core import (
    BaseAgent, AgentRole, AnalysisRequest, AnalysisResult,
    TradingSignal, SignalDirection, ConfidenceLevel, MarketData
)

logger = logging.getLogger(__name__)


class SentimentScore(Enum):
    """Sentiment score levels."""
    VERY_NEGATIVE = -1.0
    NEGATIVE = -0.5
    NEUTRAL = 0.0
    POSITIVE = 0.5
    VERY_POSITIVE = 1.0


class NewsCategory(Enum):
    """News category classifications."""
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    ANALYST = "analyst_rating"
    MERGER = "merger_acquisition"
    REGULATORY = "regulatory"
    MACRO = "macroeconomic"
    PRODUCT = "product_announcement"
    MANAGEMENT = "management_change"
    LEGAL = "legal_issues"
    OTHER = "other"


@dataclass
class NewsItem:
    """Individual news item with sentiment analysis."""
    headline: str
    content: str
    timestamp: datetime
    source: str
    category: NewsCategory
    sentiment_score: float  # -1.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    credibility_score: float  # 0.0 to 1.0
    impact_potential: str  # "low", "medium", "high"


@dataclass
class SocialMediaSignal:
    """Social media sentiment signal."""
    platform: str
    mention_count: int
    sentiment_average: float
    volume_change_24h: float
    trending_topics: List[str]
    influencer_mentions: int
    sentiment_distribution: Dict[str, float]  # positive, negative, neutral percentages


@dataclass
class MarketPsychologyMetrics:
    """Market psychology indicators."""
    fear_greed_index: float  # 0-100 scale
    vix_level: float
    put_call_ratio: float
    insider_trading_sentiment: float
    analyst_revisions_trend: float
    social_sentiment_momentum: float


class SentimentAgent(BaseAgent):
    """Sentiment analysis agent for market psychology and news impact assessment."""

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentRole.SENTIMENT, config)

        # Configuration parameters
        self.news_lookback_hours = self.config.get("news_lookback_hours", 24)
        self.sentiment_threshold = self.config.get("sentiment_threshold", 0.3)
        self.high_impact_threshold = self.config.get("high_impact_threshold", 0.7)
        self.social_weight = self.config.get("social_weight", 0.4)
        self.news_weight = self.config.get("news_weight", 0.6)

        # Sentiment lexicons (simplified - in production would use comprehensive NLP models)
        self.positive_keywords = {
            'earnings', 'beat', 'exceed', 'strong', 'growth', 'profit', 'revenue', 'bullish',
            'upgrade', 'outperform', 'buy', 'positive', 'optimistic', 'rally', 'surge',
            'breakthrough', 'innovation', 'partnership', 'expansion', 'acquisition'
        }

        self.negative_keywords = {
            'miss', 'weak', 'decline', 'loss', 'bearish', 'downgrade', 'sell', 'negative',
            'pessimistic', 'crash', 'plunge', 'concern', 'warning', 'litigation', 'scandal',
            'layoffs', 'bankruptcy', 'investigation', 'regulatory', 'fine'
        }

        self.impact_keywords = {
            'high': {'earnings', 'fda', 'acquisition', 'merger', 'bankruptcy', 'breakthrough', 'lawsuit'},
            'medium': {'guidance', 'upgrade', 'downgrade', 'partnership', 'expansion', 'layoffs'},
            'low': {'insider', 'conference', 'interview', 'rumor', 'speculation'}
        }

        logger.info(f"Sentiment agent initialized with {self.news_lookback_hours}h lookback")

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform sentiment analysis on requested symbols."""
        signals = []
        analysis_data = {}

        for symbol in request.symbols:
            try:
                # Get news data
                news_items = await self._get_news_data(symbol)

                # Get social media sentiment
                social_signals = await self._get_social_sentiment(symbol)

                # Calculate market psychology metrics
                psychology_metrics = await self._calculate_market_psychology(symbol)

                # Analyze news sentiment
                news_sentiment = await self._analyze_news_sentiment(news_items)

                # Combine sentiment sources
                combined_sentiment = await self._combine_sentiment_sources(
                    news_sentiment, social_signals, psychology_metrics
                )

                # Generate trading signal
                signal = await self._generate_sentiment_signal(
                    symbol, combined_sentiment, news_items, social_signals
                )

                if signal:
                    signals.append(signal)

                # Store analysis data
                analysis_data[symbol] = {
                    "news_sentiment": news_sentiment,
                    "social_sentiment": social_signals.__dict__ if social_signals else {},
                    "psychology_metrics": psychology_metrics.__dict__ if psychology_metrics else {},
                    "combined_sentiment": combined_sentiment,
                    "news_count": len(news_items),
                    "high_impact_news": len([n for n in news_items if n.impact_potential == "high"])
                }

            except Exception as e:
                logger.error(f"Sentiment analysis failed for {symbol}: {e}")
                analysis_data[symbol] = {"error": str(e)}

        return AnalysisResult(
            request_id=request.request_id,
            agent_id=self.agent_id,
            symbols=request.symbols,
            signals=signals,
            analysis_data=analysis_data
        )

    async def _get_news_data(self, symbol: str) -> List[NewsItem]:
        """Get news data for sentiment analysis."""
        # In a real implementation, this would fetch from news APIs like:
        # - Bloomberg API, Reuters, Alpha Vantage News, NewsAPI
        # - Financial news aggregators like RavenPack, Thomson Reuters

        # Simulated news data based on symbol
        import hashlib
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        import random
        random.seed(seed)

        news_items = []
        current_time = datetime.now(timezone.utc)

        # Generate realistic news items
        news_templates = [
            {
                "headline": f"{symbol} reports Q3 earnings beat expectations",
                "category": NewsCategory.EARNINGS,
                "impact": "high",
                "sentiment_bias": 0.7
            },
            {
                "headline": f"Analyst upgrades {symbol} to Buy rating",
                "category": NewsCategory.ANALYST,
                "impact": "medium",
                "sentiment_bias": 0.5
            },
            {
                "headline": f"{symbol} CEO discusses growth strategy in interview",
                "category": NewsCategory.MANAGEMENT,
                "impact": "low",
                "sentiment_bias": 0.2
            },
            {
                "headline": f"Regulatory concerns emerge for {symbol}",
                "category": NewsCategory.REGULATORY,
                "impact": "medium",
                "sentiment_bias": -0.4
            },
            {
                "headline": f"{symbol} announces new product launch",
                "category": NewsCategory.PRODUCT,
                "impact": "medium",
                "sentiment_bias": 0.3
            }
        ]

        # Generate 3-7 news items
        num_items = random.randint(3, 7)
        for i in range(num_items):
            template = random.choice(news_templates)

            # Add some noise to sentiment
            sentiment_noise = random.uniform(-0.2, 0.2)
            sentiment_score = max(-1.0, min(1.0, template["sentiment_bias"] + sentiment_noise))

            # Generate realistic timestamps
            hours_ago = random.uniform(0, self.news_lookback_hours)
            timestamp = current_time - timedelta(hours=hours_ago)

            # Credibility based on source
            sources = ["Reuters", "Bloomberg", "CNBC", "WSJ", "MarketWatch", "Yahoo Finance"]
            source = random.choice(sources)
            credibility_map = {
                "Reuters": 0.95, "Bloomberg": 0.95, "WSJ": 0.90,
                "CNBC": 0.85, "MarketWatch": 0.80, "Yahoo Finance": 0.75
            }
            credibility = credibility_map.get(source, 0.75)

            # Relevance score
            relevance = random.uniform(0.7, 1.0)

            news_item = NewsItem(
                headline=template["headline"],
                content=f"Detailed analysis of {template['headline'].lower()}...",
                timestamp=timestamp,
                source=source,
                category=template["category"],
                sentiment_score=sentiment_score,
                relevance_score=relevance,
                credibility_score=credibility,
                impact_potential=template["impact"]
            )

            news_items.append(news_item)

        return sorted(news_items, key=lambda x: x.timestamp, reverse=True)

    async def _get_social_sentiment(self, symbol: str) -> Optional[SocialMediaSignal]:
        """Get social media sentiment data."""
        # In a real implementation, this would integrate with:
        # - Twitter API for tweet analysis
        # - Reddit API for r/investing, r/stocks discussions
        # - StockTwits for financial social sentiment
        # - Discord/Telegram financial channels

        import hashlib
        import random

        seed = int(hashlib.md5(f"social_{symbol}".encode()).hexdigest()[:8], 16)
        random.seed(seed)

        # Simulate social sentiment data
        mention_count = random.randint(50, 500)
        sentiment_average = random.uniform(-0.5, 0.5)
        volume_change = random.uniform(-0.3, 0.8)  # Can spike significantly
        influencer_mentions = random.randint(0, 5)

        # Trending topics
        topics = [f"${symbol}", "earnings", "buy_the_dip", "technical_analysis", "options_play"]
        trending_topics = random.sample(topics, random.randint(1, 3))

        # Sentiment distribution
        positive_pct = max(0, min(1, (sentiment_average + 1) / 2 + random.uniform(-0.1, 0.1)))
        negative_pct = max(0, min(1, (-sentiment_average + 1) / 2 + random.uniform(-0.1, 0.1)))
        neutral_pct = max(0, 1 - positive_pct - negative_pct)

        # Normalize to 100%
        total = positive_pct + negative_pct + neutral_pct
        if total > 0:
            positive_pct /= total
            negative_pct /= total
            neutral_pct /= total

        return SocialMediaSignal(
            platform="aggregated",
            mention_count=mention_count,
            sentiment_average=sentiment_average,
            volume_change_24h=volume_change,
            trending_topics=trending_topics,
            influencer_mentions=influencer_mentions,
            sentiment_distribution={
                "positive": positive_pct,
                "negative": negative_pct,
                "neutral": neutral_pct
            }
        )

    async def _calculate_market_psychology(self, symbol: str) -> Optional[MarketPsychologyMetrics]:
        """Calculate market psychology indicators."""
        import hashlib
        import random

        seed = int(hashlib.md5(f"psychology_{symbol}".encode()).hexdigest()[:8], 16)
        random.seed(seed)

        # Simulate market psychology metrics
        # In reality, these would come from:
        # - VIX data from CBOE
        # - Put/Call ratios from options exchanges
        # - Insider trading data from SEC filings
        # - Analyst revision data from financial data providers

        return MarketPsychologyMetrics(
            fear_greed_index=random.uniform(20, 80),  # CNN Fear & Greed Index
            vix_level=random.uniform(12, 35),  # VIX volatility index
            put_call_ratio=random.uniform(0.7, 1.3),  # Options sentiment
            insider_trading_sentiment=random.uniform(-0.3, 0.3),  # Insider buy/sell
            analyst_revisions_trend=random.uniform(-0.2, 0.2),  # Estimate revisions
            social_sentiment_momentum=random.uniform(-0.5, 0.5)  # Social trend
        )

    async def _analyze_news_sentiment(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """Analyze sentiment from news items."""
        if not news_items:
            return {
                "overall_sentiment": 0.0,
                "sentiment_strength": 0.0,
                "high_impact_sentiment": 0.0,
                "sentiment_trend": "neutral",
                "credibility_weighted_sentiment": 0.0
            }

        # Calculate various sentiment metrics
        sentiments = [item.sentiment_score for item in news_items]
        credibility_weights = [item.credibility_score for item in news_items]

        # Overall sentiment (simple average)
        overall_sentiment = sum(sentiments) / len(sentiments)

        # Credibility-weighted sentiment
        if sum(credibility_weights) > 0:
            credibility_weighted = sum(s * w for s, w in zip(sentiments, credibility_weights)) / sum(credibility_weights)
        else:
            credibility_weighted = overall_sentiment

        # High-impact news sentiment
        high_impact_items = [item for item in news_items if item.impact_potential == "high"]
        high_impact_sentiment = 0.0
        if high_impact_items:
            high_impact_sentiment = sum(item.sentiment_score for item in high_impact_items) / len(high_impact_items)

        # Sentiment strength (volatility of sentiment)
        if len(sentiments) > 1:
            sentiment_variance = sum((s - overall_sentiment) ** 2 for s in sentiments) / len(sentiments)
            sentiment_strength = sentiment_variance ** 0.5
        else:
            sentiment_strength = abs(overall_sentiment)

        # Sentiment trend (recent vs older news)
        if len(news_items) >= 3:
            recent_items = news_items[:len(news_items)//2]  # First half (most recent)
            older_items = news_items[len(news_items)//2:]   # Second half (older)

            recent_sentiment = sum(item.sentiment_score for item in recent_items) / len(recent_items)
            older_sentiment = sum(item.sentiment_score for item in older_items) / len(older_items)

            trend_change = recent_sentiment - older_sentiment
            if trend_change > 0.1:
                sentiment_trend = "improving"
            elif trend_change < -0.1:
                sentiment_trend = "deteriorating"
            else:
                sentiment_trend = "stable"
        else:
            sentiment_trend = "neutral"

        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_strength": sentiment_strength,
            "high_impact_sentiment": high_impact_sentiment,
            "sentiment_trend": sentiment_trend,
            "credibility_weighted_sentiment": credibility_weighted,
            "news_count": len(news_items),
            "high_impact_count": len(high_impact_items),
            "average_credibility": sum(credibility_weights) / len(credibility_weights) if credibility_weights else 0
        }

    async def _combine_sentiment_sources(
        self,
        news_sentiment: Dict[str, Any],
        social_signals: Optional[SocialMediaSignal],
        psychology_metrics: Optional[MarketPsychologyMetrics]
    ) -> Dict[str, Any]:
        """Combine sentiment from multiple sources."""

        # News sentiment component
        news_score = news_sentiment.get("credibility_weighted_sentiment", 0.0)
        news_impact = news_sentiment.get("high_impact_sentiment", 0.0)

        # Social sentiment component
        social_score = 0.0
        if social_signals:
            social_score = social_signals.sentiment_average
            # Amplify if high volume or influencer mentions
            if social_signals.volume_change_24h > 0.5:
                social_score *= 1.2
            if social_signals.influencer_mentions > 2:
                social_score *= 1.1

        # Market psychology adjustment
        psychology_adjustment = 0.0
        if psychology_metrics:
            # Fear/Greed index normalization (50 = neutral)
            fg_normalized = (psychology_metrics.fear_greed_index - 50) / 50

            # VIX adjustment (higher VIX = more fear)
            vix_adjustment = -((psychology_metrics.vix_level - 20) / 20) * 0.1

            # Put/Call ratio (>1.0 = bearish, <1.0 = bullish)
            pc_adjustment = -(psychology_metrics.put_call_ratio - 1.0) * 0.1

            psychology_adjustment = (fg_normalized + vix_adjustment + pc_adjustment) / 3

        # Weighted combination
        combined_score = (
            news_score * self.news_weight +
            social_score * self.social_weight +
            psychology_adjustment * 0.2
        )

        # Calculate confidence based on consistency
        sentiment_sources = [news_score, social_score]
        if psychology_metrics:
            sentiment_sources.append(psychology_adjustment)

        # Confidence higher when sources agree
        sentiment_consistency = 1.0 - (max(sentiment_sources) - min(sentiment_sources)) / 2.0
        confidence = max(0.1, sentiment_consistency)

        return {
            "combined_sentiment": combined_score,
            "news_component": news_score,
            "social_component": social_score,
            "psychology_component": psychology_adjustment,
            "sentiment_confidence": confidence,
            "high_impact_factor": abs(news_impact) if abs(news_impact) > 0.3 else 0,
            "source_agreement": sentiment_consistency
        }

    async def _generate_sentiment_signal(
        self,
        symbol: str,
        combined_sentiment: Dict[str, Any],
        news_items: List[NewsItem],
        social_signals: Optional[SocialMediaSignal]
    ) -> Optional[TradingSignal]:
        """Generate trading signal based on sentiment analysis."""
        try:
            sentiment_score = combined_sentiment.get("combined_sentiment", 0.0)
            confidence_score = combined_sentiment.get("sentiment_confidence", 0.5)
            high_impact_factor = combined_sentiment.get("high_impact_factor", 0.0)

            # Adjust signal strength based on high-impact news
            adjusted_sentiment = sentiment_score
            if high_impact_factor > 0:
                adjusted_sentiment *= (1 + high_impact_factor)

            # Signal strength thresholds
            signal_strength = 0.0
            reasoning_parts = []

            # Sentiment-based signal
            if abs(adjusted_sentiment) > self.sentiment_threshold:
                signal_strength = adjusted_sentiment

                if adjusted_sentiment > 0:
                    reasoning_parts.append(f"Positive sentiment: {adjusted_sentiment:.2f}")
                else:
                    reasoning_parts.append(f"Negative sentiment: {adjusted_sentiment:.2f}")

            # High-impact news amplification
            if high_impact_factor > 0:
                reasoning_parts.append(f"High-impact news factor: {high_impact_factor:.2f}")

            # Social media momentum
            if social_signals and abs(social_signals.sentiment_average) > 0.3:
                if social_signals.volume_change_24h > 0.5:
                    reasoning_parts.append(f"Social momentum: {social_signals.volume_change_24h:.1f}x volume")

            # Recent news trend
            news_sentiment = await self._analyze_news_sentiment(news_items)
            sentiment_trend = news_sentiment.get("sentiment_trend", "neutral")
            if sentiment_trend in ["improving", "deteriorating"]:
                reasoning_parts.append(f"Sentiment trend: {sentiment_trend}")

            # Determine signal direction
            if signal_strength > 0.4:
                signal_direction = SignalDirection.STRONG_BUY
            elif signal_strength > 0.2:
                signal_direction = SignalDirection.BUY
            elif signal_strength < -0.4:
                signal_direction = SignalDirection.STRONG_SELL
            elif signal_strength < -0.2:
                signal_direction = SignalDirection.SELL
            else:
                signal_direction = SignalDirection.HOLD

            # Adjust confidence based on source agreement and impact
            final_confidence = confidence_score
            if high_impact_factor > 0.5:
                final_confidence += 0.2
            if len(reasoning_parts) >= 3:  # Multiple confirmations
                final_confidence += 0.1

            final_confidence = min(max(final_confidence, 0.1), 0.9)

            # Determine confidence level
            if final_confidence > 0.8:
                confidence = ConfidenceLevel.VERY_HIGH
            elif final_confidence > 0.6:
                confidence = ConfidenceLevel.HIGH
            elif final_confidence > 0.4:
                confidence = ConfidenceLevel.MEDIUM
            elif final_confidence > 0.2:
                confidence = ConfidenceLevel.LOW
            else:
                confidence = ConfidenceLevel.VERY_LOW

            # Only generate signal if we have meaningful conviction
            if abs(signal_strength) < 0.15 or final_confidence < 0.3:
                return None

            reasoning = f"Sentiment analysis: {'; '.join(reasoning_parts)}"

            # Set expiry based on news recency and social momentum
            expiry_hours = 24  # Default 24 hour expiry
            if high_impact_factor > 0.5:
                expiry_hours = 72  # High impact news lasts longer
            elif social_signals and social_signals.volume_change_24h > 1.0:
                expiry_hours = 12  # Social momentum fades quickly

            expiry_time = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

            return TradingSignal(
                agent_id=self.agent_id,
                agent_role=self.role,
                symbol=symbol,
                signal=signal_direction,
                confidence=confidence,
                reasoning=reasoning,
                expiry_time=expiry_time,
                supporting_data={
                    "sentiment_score": sentiment_score,
                    "adjusted_sentiment": adjusted_sentiment,
                    "high_impact_factor": high_impact_factor,
                    "news_count": len(news_items),
                    "social_volume_change": social_signals.volume_change_24h if social_signals else 0,
                    "sentiment_trend": sentiment_trend,
                    "source_agreement": combined_sentiment.get("source_agreement", 0.5)
                }
            )

        except Exception as e:
            logger.error(f"Sentiment signal generation failed for {symbol}: {e}")
            return None


__all__ = [
    "SentimentAgent",
    "SentimentScore",
    "NewsCategory",
    "NewsItem",
    "SocialMediaSignal",
    "MarketPsychologyMetrics"
]