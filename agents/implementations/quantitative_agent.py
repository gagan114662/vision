"""
Quantitative researcher agent implementation.

Performs factor model analysis, statistical arbitrage, pairs trading,
cointegration testing, and quantitative strategy development.
"""
from __future__ import annotations

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.optimize import minimize

from agents.core import (
    BaseAgent, AgentRole, AnalysisRequest, AnalysisResult,
    TradingSignal, SignalDirection, ConfidenceLevel, MarketData
)

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """Factor model types."""
    MARKET = "market"
    SIZE = "size"
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    PROFITABILITY = "profitability"


@dataclass
class FactorExposure:
    """Factor exposure for a security."""
    symbol: str
    market_beta: float
    size_factor: float  # SMB (Small Minus Big)
    value_factor: float  # HML (High Minus Low)
    momentum_factor: float  # UMD (Up Minus Down)
    quality_factor: float  # RMW (Robust Minus Weak)
    volatility_factor: float  # LTR (Low minus High)
    profitability_factor: float  # CMA (Conservative Minus Aggressive)
    r_squared: float
    alpha: float


@dataclass
class PairsAnalysis:
    """Pairs trading analysis result."""
    symbol_a: str
    symbol_b: str
    correlation: float
    cointegration_pvalue: float
    hedge_ratio: float
    half_life: float  # Mean reversion half-life in days
    z_score_current: float
    z_score_mean: float
    z_score_std: float
    entry_threshold: float
    exit_threshold: float
    spread_current: float


@dataclass
class StatisticalArbitrageSignal:
    """Statistical arbitrage opportunity."""
    strategy_type: str  # "pairs", "mean_reversion", "momentum"
    symbols: List[str]
    direction: Dict[str, str]  # symbol -> "long"/"short"
    confidence: float
    expected_return: float
    risk_estimate: float
    time_horizon_days: int


class QuantitativeAgent(BaseAgent):
    """Quantitative analysis agent for factor models and statistical arbitrage."""

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, AgentRole.QUANTITATIVE, config)

        # Configuration parameters
        self.lookback_days = self.config.get("lookback_days", 252)  # 1 year of data
        self.min_correlation = self.config.get("min_correlation", 0.7)
        self.cointegration_threshold = self.config.get("cointegration_threshold", 0.05)
        self.z_score_entry = self.config.get("z_score_entry", 2.0)
        self.z_score_exit = self.config.get("z_score_exit", 0.5)

        # Factor model parameters
        self.factor_window = self.config.get("factor_window", 60)  # 60-day rolling window
        self.min_r_squared = self.config.get("min_r_squared", 0.3)

        logger.info(f"Quantitative agent initialized with {self.lookback_days} days lookback")

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform quantitative analysis on requested symbols."""
        signals = []
        analysis_data = {}

        try:
            # Get price data for all symbols
            price_data = await self._get_multi_symbol_data(request.symbols)

            # Perform factor analysis
            factor_exposures = await self._calculate_factor_exposures(price_data)

            # Identify pairs trading opportunities
            pairs_opportunities = await self._identify_pairs_opportunities(price_data)

            # Perform mean reversion analysis
            mean_reversion_signals = await self._analyze_mean_reversion(price_data)

            # Statistical arbitrage detection
            stat_arb_opportunities = await self._detect_statistical_arbitrage(
                price_data, factor_exposures, pairs_opportunities
            )

            # Generate trading signals
            for symbol in request.symbols:
                signal = await self._generate_quantitative_signal(
                    symbol, factor_exposures.get(symbol), stat_arb_opportunities, mean_reversion_signals.get(symbol)
                )

                if signal:
                    signals.append(signal)

            # Store analysis data
            analysis_data = {
                "factor_exposures": {k: v.__dict__ for k, v in factor_exposures.items()},
                "pairs_opportunities": [p.__dict__ for p in pairs_opportunities],
                "mean_reversion_signals": mean_reversion_signals,
                "stat_arb_opportunities": [s.__dict__ for s in stat_arb_opportunities],
                "universe_correlation_matrix": await self._calculate_correlation_matrix(price_data)
            }

        except Exception as e:
            logger.error(f"Quantitative analysis failed: {e}")
            analysis_data = {"error": str(e)}

        return AnalysisResult(
            request_id=request.request_id,
            agent_id=self.agent_id,
            symbols=request.symbols,
            signals=signals,
            analysis_data=analysis_data
        )

    async def _get_multi_symbol_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Get price data for multiple symbols."""
        price_data = {}

        for symbol in symbols:
            # Generate synthetic price data (in production, would fetch real data)
            np.random.seed(hash(symbol) % 2147483647)

            dates = pd.date_range(
                end=datetime.now(),
                periods=self.lookback_days,
                freq='D'
            )

            # Generate correlated returns for realistic pair relationships
            base_return = 0.0008  # 0.08% daily return
            volatility = 0.02  # 2% daily volatility

            returns = np.random.normal(base_return, volatility, len(dates))

            # Add sector correlation
            if symbol.startswith('A'):  # Tech sector proxy
                sector_factor = np.random.normal(0, 0.01, len(dates))
                returns += sector_factor * 0.5

            # Calculate prices from returns
            prices = [100.0]  # Starting price
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            # Create DataFrame
            df = pd.DataFrame({
                'date': dates,
                'price': prices,
                'returns': [0] + list(np.diff(np.log(prices)))
            })
            df.set_index('date', inplace=True)

            price_data[symbol] = df

        return price_data

    async def _calculate_factor_exposures(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, FactorExposure]:
        """Calculate factor model exposures using Fama-French factors."""
        factor_exposures = {}

        # Generate market factor (simplified)
        market_returns = []
        for symbol_data in price_data.values():
            market_returns.append(symbol_data['returns'].values)

        if not market_returns:
            return factor_exposures

        # Equal-weighted market return
        market_factor = np.mean(market_returns, axis=0)

        # Generate style factors (simplified proxies)
        size_factor = np.random.normal(0, 0.005, len(market_factor))  # SMB
        value_factor = np.random.normal(0, 0.004, len(market_factor))  # HML
        momentum_factor = np.random.normal(0, 0.006, len(market_factor))  # UMD
        quality_factor = np.random.normal(0, 0.003, len(market_factor))  # RMW
        volatility_factor = np.random.normal(0, 0.004, len(market_factor))  # LTR
        profitability_factor = np.random.normal(0, 0.003, len(market_factor))  # CMA

        for symbol, data in price_data.items():
            try:
                returns = data['returns'].values[1:]  # Skip first zero return

                # Ensure same length
                min_length = min(len(returns), len(market_factor))
                y = returns[-min_length:]
                X = np.column_stack([
                    market_factor[-min_length:],
                    size_factor[-min_length:],
                    value_factor[-min_length:],
                    momentum_factor[-min_length:],
                    quality_factor[-min_length:],
                    volatility_factor[-min_length:],
                    profitability_factor[-min_length:]
                ])

                # Add constant for alpha
                X = np.column_stack([np.ones(len(X)), X])

                # Multiple regression
                if len(y) > 10 and np.linalg.matrix_rank(X) == X.shape[1]:
                    coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

                    # Calculate R-squared
                    y_mean = np.mean(y)
                    ss_tot = np.sum((y - y_mean) ** 2)
                    ss_res = np.sum(residuals) if len(residuals) > 0 else np.sum((y - X @ coefficients) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    factor_exposures[symbol] = FactorExposure(
                        symbol=symbol,
                        market_beta=coefficients[1],
                        size_factor=coefficients[2],
                        value_factor=coefficients[3],
                        momentum_factor=coefficients[4],
                        quality_factor=coefficients[5],
                        volatility_factor=coefficients[6],
                        profitability_factor=coefficients[7],
                        r_squared=max(0, min(1, r_squared)),
                        alpha=coefficients[0]
                    )

            except Exception as e:
                logger.warning(f"Factor exposure calculation failed for {symbol}: {e}")

        return factor_exposures

    async def _identify_pairs_opportunities(self, price_data: Dict[str, pd.DataFrame]) -> List[PairsAnalysis]:
        """Identify pairs trading opportunities through cointegration analysis."""
        pairs_opportunities = []
        symbols = list(price_data.keys())

        for i, symbol_a in enumerate(symbols):
            for symbol_b in symbols[i+1:]:
                try:
                    data_a = price_data[symbol_a]['price'].values
                    data_b = price_data[symbol_b]['price'].values

                    # Ensure same length
                    min_length = min(len(data_a), len(data_b))
                    prices_a = data_a[-min_length:]
                    prices_b = data_b[-min_length:]

                    if min_length < 30:  # Need minimum data
                        continue

                    # Calculate correlation
                    correlation = np.corrcoef(prices_a, prices_b)[0, 1]

                    if abs(correlation) < self.min_correlation:
                        continue

                    # Cointegration test (Engle-Granger)
                    # Step 1: Find hedge ratio through OLS
                    X = np.column_stack([np.ones(len(prices_a)), prices_a])
                    hedge_ratio_coef = np.linalg.lstsq(X, prices_b, rcond=None)[0]
                    hedge_ratio = hedge_ratio_coef[1]

                    # Step 2: Calculate spread
                    spread = prices_b - hedge_ratio * prices_a - hedge_ratio_coef[0]

                    # Step 3: Test spread for stationarity (simplified ADF test)
                    spread_diff = np.diff(spread)
                    spread_lagged = spread[:-1]

                    if len(spread_diff) > 5 and np.std(spread_lagged) > 0:
                        # Simple unit root test
                        X_adf = np.column_stack([np.ones(len(spread_diff)), spread_lagged])
                        try:
                            adf_coef = np.linalg.lstsq(X_adf, spread_diff, rcond=None)[0]
                            # Simplified p-value calculation
                            cointegration_pvalue = max(0.001, min(0.999, abs(adf_coef[1])))
                        except:
                            cointegration_pvalue = 1.0
                    else:
                        cointegration_pvalue = 1.0

                    if cointegration_pvalue > self.cointegration_threshold:
                        continue

                    # Calculate mean reversion properties
                    spread_mean = np.mean(spread)
                    spread_std = np.std(spread)
                    z_score_current = (spread[-1] - spread_mean) / spread_std if spread_std > 0 else 0

                    # Half-life calculation (simplified)
                    if len(spread) > 10:
                        spread_centered = spread - spread_mean
                        spread_lag = spread_centered[:-1]
                        spread_diff = np.diff(spread_centered)

                        if np.std(spread_lag) > 0:
                            try:
                                ar_coef = np.linalg.lstsq(
                                    spread_lag.reshape(-1, 1), spread_diff, rcond=None
                                )[0][0]
                                half_life = -np.log(2) / ar_coef if ar_coef < 0 else float('inf')
                                half_life = max(1, min(252, half_life))  # Cap between 1 day and 1 year
                            except:
                                half_life = 30  # Default 30 days
                        else:
                            half_life = 30
                    else:
                        half_life = 30

                    pairs_analysis = PairsAnalysis(
                        symbol_a=symbol_a,
                        symbol_b=symbol_b,
                        correlation=correlation,
                        cointegration_pvalue=cointegration_pvalue,
                        hedge_ratio=hedge_ratio,
                        half_life=half_life,
                        z_score_current=z_score_current,
                        z_score_mean=0.0,
                        z_score_std=1.0,
                        entry_threshold=self.z_score_entry,
                        exit_threshold=self.z_score_exit,
                        spread_current=spread[-1]
                    )

                    pairs_opportunities.append(pairs_analysis)

                except Exception as e:
                    logger.warning(f"Pairs analysis failed for {symbol_a}-{symbol_b}: {e}")

        # Sort by significance (low p-value, high correlation)
        pairs_opportunities.sort(
            key=lambda x: x.cointegration_pvalue + (1 - abs(x.correlation))
        )

        return pairs_opportunities[:10]  # Return top 10 pairs

    async def _analyze_mean_reversion(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Analyze mean reversion characteristics for individual securities."""
        mean_reversion_signals = {}

        for symbol, data in price_data.items():
            try:
                prices = data['price'].values
                returns = data['returns'].values[1:]  # Skip first zero

                if len(prices) < 30:
                    continue

                # Calculate rolling z-score of prices
                window = min(60, len(prices) // 3)
                if window < 10:
                    continue

                price_series = pd.Series(prices)
                rolling_mean = price_series.rolling(window).mean()
                rolling_std = price_series.rolling(window).std()

                z_scores = (price_series - rolling_mean) / rolling_std
                current_z_score = z_scores.iloc[-1] if not pd.isna(z_scores.iloc[-1]) else 0

                # Hurst exponent for mean reversion detection
                hurst_exponent = await self._calculate_hurst_exponent(prices)

                # Ornstein-Uhlenbeck test
                ou_params = await self._estimate_ou_parameters(prices)

                # Mean reversion strength
                mr_strength = 0.5 - hurst_exponent  # H < 0.5 indicates mean reversion
                mr_strength = max(-0.5, min(0.5, mr_strength))

                mean_reversion_signals[symbol] = {
                    "current_z_score": float(current_z_score),
                    "hurst_exponent": hurst_exponent,
                    "mean_reversion_strength": mr_strength,
                    "ou_kappa": ou_params.get("kappa", 0),
                    "ou_theta": ou_params.get("theta", prices[-1]),
                    "ou_sigma": ou_params.get("sigma", np.std(returns)),
                    "half_life_days": ou_params.get("half_life", float('inf')),
                    "current_price": prices[-1],
                    "fair_value": ou_params.get("theta", prices[-1])
                }

            except Exception as e:
                logger.warning(f"Mean reversion analysis failed for {symbol}: {e}")

        return mean_reversion_signals

    async def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent for mean reversion detection."""
        try:
            if len(prices) < 20:
                return 0.5

            # R/S analysis (simplified)
            lags = range(2, min(len(prices) // 4, 50))
            rs_values = []

            log_prices = np.log(prices)

            for lag in lags:
                # Split into chunks
                chunks = len(log_prices) // lag
                if chunks < 2:
                    continue

                rs_chunk = []
                for i in range(chunks):
                    chunk = log_prices[i*lag:(i+1)*lag]
                    if len(chunk) < lag:
                        continue

                    # Calculate mean
                    mean_chunk = np.mean(chunk)

                    # Cumulative deviations
                    cumdev = np.cumsum(chunk - mean_chunk)

                    # Range
                    R = np.max(cumdev) - np.min(cumdev)

                    # Standard deviation
                    S = np.std(chunk)

                    if S > 0:
                        rs_chunk.append(R / S)

                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))

            if len(rs_values) < 3:
                return 0.5

            # Fit log(R/S) = H * log(lag) + constant
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)

            # Remove infinite or NaN values
            valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
            if np.sum(valid_mask) < 3:
                return 0.5

            log_lags = log_lags[valid_mask]
            log_rs = log_rs[valid_mask]

            # Linear regression
            X = np.column_stack([np.ones(len(log_lags)), log_lags])
            hurst = np.linalg.lstsq(X, log_rs, rcond=None)[0][1]

            return max(0.0, min(1.0, hurst))

        except Exception:
            return 0.5  # Default to random walk

    async def _estimate_ou_parameters(self, prices: np.ndarray) -> Dict[str, float]:
        """Estimate Ornstein-Uhlenbeck process parameters."""
        try:
            if len(prices) < 10:
                return {}

            log_prices = np.log(prices)
            dt = 1.0  # Daily data

            # Calculate price differences
            dx = np.diff(log_prices)
            x = log_prices[:-1]

            # Estimate parameters using least squares
            # dx = kappa * (theta - x) * dt + sigma * dW
            # Rearrange: dx = (kappa * theta * dt) - (kappa * dt) * x + sigma * dW

            if len(x) < 3:
                return {}

            # Linear regression: dx = a + b * x + error
            X = np.column_stack([np.ones(len(x)), x])
            try:
                coeffs = np.linalg.lstsq(X, dx, rcond=None)[0]
                a, b = coeffs[0], coeffs[1]

                # Extract OU parameters
                kappa = -b / dt
                theta = -a / b if b != 0 else np.mean(log_prices)
                residuals = dx - (a + b * x)
                sigma = np.std(residuals) / np.sqrt(dt)

                # Half-life calculation
                half_life = np.log(2) / kappa if kappa > 0 else float('inf')

                return {
                    "kappa": float(kappa),
                    "theta": float(np.exp(theta)),  # Convert back to price level
                    "sigma": float(sigma),
                    "half_life": float(half_life)
                }

            except np.linalg.LinAlgError:
                return {}

        except Exception:
            return {}

    async def _detect_statistical_arbitrage(
        self,
        price_data: Dict[str, pd.DataFrame],
        factor_exposures: Dict[str, FactorExposure],
        pairs_opportunities: List[PairsAnalysis]
    ) -> List[StatisticalArbitrageSignal]:
        """Detect statistical arbitrage opportunities."""
        stat_arb_signals = []

        # Pairs trading opportunities
        for pair in pairs_opportunities:
            if abs(pair.z_score_current) > pair.entry_threshold:
                direction = {}
                if pair.z_score_current > 0:  # Spread too high
                    direction[pair.symbol_a] = "long"   # Buy undervalued
                    direction[pair.symbol_b] = "short"  # Sell overvalued
                else:  # Spread too low
                    direction[pair.symbol_a] = "short"  # Sell overvalued
                    direction[pair.symbol_b] = "long"   # Buy undervalued

                confidence = min(0.9, abs(pair.z_score_current) / pair.entry_threshold * 0.7)
                expected_return = abs(pair.z_score_current) * 0.01  # 1% per z-score unit
                risk_estimate = 0.02  # 2% risk estimate

                stat_arb_signals.append(StatisticalArbitrageSignal(
                    strategy_type="pairs",
                    symbols=[pair.symbol_a, pair.symbol_b],
                    direction=direction,
                    confidence=confidence,
                    expected_return=expected_return,
                    risk_estimate=risk_estimate,
                    time_horizon_days=int(pair.half_life * 2)  # 2x half-life
                ))

        # Mean reversion opportunities
        for symbol, data in price_data.items():
            if symbol in factor_exposures:
                exposure = factor_exposures[symbol]

                # Look for high alpha with good factor model fit
                if abs(exposure.alpha) > 0.001 and exposure.r_squared > self.min_r_squared:
                    direction = {"long" if exposure.alpha > 0 else "short": symbol}
                    confidence = min(0.8, exposure.r_squared * abs(exposure.alpha) * 1000)

                    if confidence > 0.3:
                        stat_arb_signals.append(StatisticalArbitrageSignal(
                            strategy_type="mean_reversion",
                            symbols=[symbol],
                            direction={symbol: "long" if exposure.alpha > 0 else "short"},
                            confidence=confidence,
                            expected_return=abs(exposure.alpha) * 252,  # Annualized alpha
                            risk_estimate=0.03,
                            time_horizon_days=60  # 2-3 months
                        ))

        return stat_arb_signals

    async def _calculate_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate correlation matrix for the universe."""
        try:
            symbols = list(price_data.keys())
            returns_matrix = []

            for symbol in symbols:
                returns = price_data[symbol]['returns'].values[1:]  # Skip first zero
                returns_matrix.append(returns)

            if not returns_matrix:
                return {}

            # Ensure all series have same length
            min_length = min(len(r) for r in returns_matrix)
            returns_matrix = [r[-min_length:] for r in returns_matrix]

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(returns_matrix)

            # Convert to dictionary format
            corr_dict = {}
            for i, symbol_i in enumerate(symbols):
                corr_dict[symbol_i] = {}
                for j, symbol_j in enumerate(symbols):
                    corr_dict[symbol_i][symbol_j] = float(correlation_matrix[i, j])

            return {
                "correlation_matrix": corr_dict,
                "average_correlation": float(np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])),
                "max_correlation": float(np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])),
                "min_correlation": float(np.min(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
            }

        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            return {}

    async def _generate_quantitative_signal(
        self,
        symbol: str,
        factor_exposure: Optional[FactorExposure],
        stat_arb_opportunities: List[StatisticalArbitrageSignal],
        mean_reversion_data: Optional[Dict[str, Any]]
    ) -> Optional[TradingSignal]:
        """Generate trading signal based on quantitative analysis."""
        try:
            signal_strength = 0.0
            confidence_score = 0.0
            reasoning_parts = []

            # Factor model alpha signal
            if factor_exposure and factor_exposure.r_squared > self.min_r_squared:
                alpha_signal = factor_exposure.alpha * 1000  # Scale up
                if abs(alpha_signal) > 0.5:
                    signal_strength += alpha_signal * 0.4
                    reasoning_parts.append(f"Factor alpha: {factor_exposure.alpha:.4f}")
                    confidence_score += factor_exposure.r_squared * 0.3

            # Statistical arbitrage signals
            symbol_stat_arb = [s for s in stat_arb_opportunities if symbol in s.symbols]
            for stat_arb in symbol_stat_arb:
                if symbol in stat_arb.direction:
                    direction_multiplier = 1 if stat_arb.direction[symbol] == "long" else -1
                    arb_strength = stat_arb.expected_return * direction_multiplier * 2  # Scale up
                    signal_strength += arb_strength * 0.5
                    confidence_score += stat_arb.confidence * 0.4
                    reasoning_parts.append(f"{stat_arb.strategy_type}: {stat_arb.direction[symbol]}")

            # Mean reversion signal
            if mean_reversion_data:
                z_score = mean_reversion_data.get("current_z_score", 0)
                mr_strength = mean_reversion_data.get("mean_reversion_strength", 0)

                if abs(z_score) > 1.5 and mr_strength > 0.1:
                    # Mean reversion: buy when below mean, sell when above
                    mr_signal = -z_score * mr_strength * 0.3
                    signal_strength += mr_signal
                    reasoning_parts.append(f"Mean reversion: z={z_score:.2f}, strength={mr_strength:.2f}")
                    confidence_score += min(0.3, abs(z_score) / 3 * mr_strength)

            # Determine signal direction
            if signal_strength > 0.3:
                signal_direction = SignalDirection.BUY if signal_strength < 0.6 else SignalDirection.STRONG_BUY
            elif signal_strength < -0.3:
                signal_direction = SignalDirection.SELL if signal_strength > -0.6 else SignalDirection.STRONG_SELL
            else:
                signal_direction = SignalDirection.HOLD

            # Confidence calculation
            confidence_score = min(max(confidence_score, 0.1), 0.9)

            if confidence_score > 0.7:
                confidence = ConfidenceLevel.VERY_HIGH
            elif confidence_score > 0.5:
                confidence = ConfidenceLevel.HIGH
            elif confidence_score > 0.3:
                confidence = ConfidenceLevel.MEDIUM
            elif confidence_score > 0.15:
                confidence = ConfidenceLevel.LOW
            else:
                confidence = ConfidenceLevel.VERY_LOW

            # Only generate signal if we have meaningful conviction
            if abs(signal_strength) < 0.2 or confidence_score < 0.2:
                return None

            reasoning = f"Quantitative analysis: {'; '.join(reasoning_parts)}"

            # Set time horizon based on strategy type
            time_horizon_days = 60  # Default 2 months
            if symbol_stat_arb:
                avg_horizon = sum(s.time_horizon_days for s in symbol_stat_arb) / len(symbol_stat_arb)
                time_horizon_days = int(avg_horizon)

            expiry_time = datetime.now(timezone.utc) + timedelta(days=time_horizon_days)

            return TradingSignal(
                agent_id=self.agent_id,
                agent_role=self.role,
                symbol=symbol,
                signal=signal_direction,
                confidence=confidence,
                reasoning=reasoning,
                expiry_time=expiry_time,
                supporting_data={
                    "signal_strength": signal_strength,
                    "factor_alpha": factor_exposure.alpha if factor_exposure else 0,
                    "factor_r_squared": factor_exposure.r_squared if factor_exposure else 0,
                    "stat_arb_count": len(symbol_stat_arb),
                    "mean_reversion_z_score": mean_reversion_data.get("current_z_score", 0) if mean_reversion_data else 0,
                    "time_horizon_days": time_horizon_days
                }
            )

        except Exception as e:
            logger.error(f"Quantitative signal generation failed for {symbol}: {e}")
            return None


__all__ = [
    "QuantitativeAgent",
    "FactorType",
    "FactorExposure",
    "PairsAnalysis",
    "StatisticalArbitrageSignal"
]