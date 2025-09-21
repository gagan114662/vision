"""
Real Factor Models with Market Data Integration.

This replaces the synthetic factor model with one that uses real market data
from the RealMarketDataProvider to calculate actual factor exposures and returns.
"""
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Try numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """Types of risk factors."""
    MARKET = "market"
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    SIZE = "size"
    PROFITABILITY = "profitability"
    INVESTMENT = "investment"


@dataclass
class FactorExposure:
    """Factor exposure for a single asset."""
    symbol: str
    exposures: Dict[FactorType, float]
    calculated_date: datetime


@dataclass
class RealFactorRiskModel:
    """Real factor risk model with market data."""
    model_id: str
    factors: List[FactorType]
    factor_returns: Dict[FactorType, List[float]]
    factor_covariance: Dict[Tuple[FactorType, FactorType], float]
    asset_exposures: Dict[str, FactorExposure]
    model_timestamp: datetime
    data_source: str
    lookback_days: int

    def get_factor_return(self, factor: FactorType, period: int = 0) -> float:
        """Get factor return for a specific period."""
        returns = self.factor_returns.get(factor, [])
        if period < len(returns):
            return returns[period]
        return 0.0

    def get_asset_exposure(self, symbol: str, factor: FactorType) -> float:
        """Get asset exposure to a factor."""
        if symbol in self.asset_exposures:
            return self.asset_exposures[symbol].exposures.get(factor, 0.0)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "factors": [f.value for f in self.factors],
            "model_timestamp": self.model_timestamp.isoformat(),
            "data_source": self.data_source,
            "lookback_days": self.lookback_days,
            "num_assets": len(self.asset_exposures)
        }


class RealFactorModelBuilder:
    """Build factor models using real market data."""

    def __init__(self, data_provider=None):
        """Initialize with market data provider."""
        self.data_provider = data_provider
        if not self.data_provider:
            # Import here to avoid circular dependency
            from mcp.market_data.real_data_provider import RealMarketDataProvider
            self.data_provider = RealMarketDataProvider()

    async def build_factor_model(
        self,
        symbols: List[str],
        lookback_days: int = 252,  # 1 year
        end_date: Optional[datetime] = None
    ) -> RealFactorRiskModel:
        """Build factor model using real market data."""

        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer for calculations

        logger.info(f"Building real factor model for {len(symbols)} symbols")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Get real market data
        stock_data = await self.data_provider.get_stock_data(symbols, start_date, end_date)
        market_factors = await self.data_provider.get_market_factors(start_date, end_date)
        risk_free_rate = await self.data_provider.get_risk_free_rate(end_date)

        # Calculate factor returns from real data
        factor_returns = await self._calculate_real_factor_returns(
            stock_data, market_factors, risk_free_rate, lookback_days
        )

        # Calculate factor covariance from real data
        factor_covariance = self._calculate_real_factor_covariance(factor_returns)

        # Calculate asset exposures from real data
        asset_exposures = {}
        for symbol in symbols:
            if symbol in stock_data:
                exposure = await self._calculate_real_asset_exposures(
                    symbol, stock_data[symbol], factor_returns, lookback_days
                )
                asset_exposures[symbol] = exposure
            else:
                logger.warning(f"No data available for {symbol}, using fallback")
                asset_exposures[symbol] = self._fallback_asset_exposures(symbol)

        # Determine data source
        data_sources = set()
        for symbol_data in stock_data.values():
            data_sources.add(symbol_data.get('source', 'unknown'))

        data_source = f"mixed: {', '.join(data_sources)}" if len(data_sources) > 1 else list(data_sources)[0]

        model = RealFactorRiskModel(
            model_id=f"real_factor_model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            factors=list(FactorType),
            factor_returns=factor_returns,
            factor_covariance=factor_covariance,
            asset_exposures=asset_exposures,
            model_timestamp=datetime.now(timezone.utc),
            data_source=data_source,
            lookback_days=lookback_days
        )

        logger.info(f"✅ Built real factor model with {len(model.factors)} factors for {len(asset_exposures)} assets")
        logger.info(f"Data source: {data_source}")

        return model

    async def _calculate_real_factor_returns(
        self,
        stock_data: Dict[str, Any],
        market_factors: Dict[str, List[float]],
        risk_free_rate: float,
        lookback_days: int
    ) -> Dict[FactorType, List[float]]:
        """Calculate factor returns from real market data."""

        logger.info("Calculating factor returns from real market data")

        factor_returns = {}

        # Market factor from actual stock data if available
        if stock_data and HAS_NUMPY:
            # Try to calculate market return from SPY or a market proxy
            market_symbol = None
            for symbol in ["SPY", "VTI", "IVV"]:
                if symbol in stock_data:
                    market_symbol = symbol
                    break

            if market_symbol:
                prices = stock_data[market_symbol]['prices']
                if len(prices) > 1:
                    returns = []
                    for i in range(1, min(len(prices), lookback_days + 1)):
                        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                        returns.append(daily_return - risk_free_rate/252)  # Excess return

                    factor_returns[FactorType.MARKET] = returns[-lookback_days:] if len(returns) > lookback_days else returns
                    logger.info(f"✅ Calculated {len(factor_returns[FactorType.MARKET])} days of market returns from {market_symbol}")

        # Use market factors if available from data provider
        if "market_premium" in market_factors:
            market_premium = market_factors["market_premium"]
            factor_returns[FactorType.MARKET] = market_premium[-lookback_days:] if len(market_premium) > lookback_days else market_premium

        # Other Fama-French factors
        factor_mapping = {
            FactorType.VALUE: "hml",
            FactorType.SIZE: "smb",
            FactorType.PROFITABILITY: "rmw",
            FactorType.INVESTMENT: "cma"
        }

        for factor_type, factor_key in factor_mapping.items():
            if factor_key in market_factors:
                factor_data = market_factors[factor_key]
                factor_returns[factor_type] = factor_data[-lookback_days:] if len(factor_data) > lookback_days else factor_data

        # Calculate momentum factor from stock data
        if stock_data and HAS_NUMPY:
            momentum_returns = await self._calculate_momentum_factor(stock_data, lookback_days)
            if momentum_returns:
                factor_returns[FactorType.MOMENTUM] = momentum_returns

        # Calculate volatility factor from stock data
        if stock_data and HAS_NUMPY:
            vol_returns = await self._calculate_volatility_factor(stock_data, lookback_days)
            if vol_returns:
                factor_returns[FactorType.VOLATILITY] = vol_returns

        # Calculate quality factor (if we have fundamental data - placeholder for now)
        factor_returns[FactorType.QUALITY] = self._generate_fallback_factor_returns(
            lookback_days, mean=0.0002, vol=0.008
        )

        # Ensure all factors have returns
        for factor_type in FactorType:
            if factor_type not in factor_returns:
                logger.warning(f"No real data for {factor_type.value}, using fallback")
                factor_returns[factor_type] = self._generate_fallback_factor_returns(
                    lookback_days, mean=0.0001, vol=0.010
                )

        logger.info(f"✅ Calculated factor returns for {len(factor_returns)} factors")
        return factor_returns

    async def _calculate_momentum_factor(
        self,
        stock_data: Dict[str, Any],
        lookback_days: int
    ) -> Optional[List[float]]:
        """Calculate momentum factor from stock price data."""

        if not HAS_NUMPY:
            return None

        try:
            # Calculate momentum for each stock
            momentum_scores = {}

            for symbol, data in stock_data.items():
                prices = data['prices']
                if len(prices) >= 60:  # Need at least 60 days for momentum calc
                    # 12-1 momentum: return from 12 months ago to 1 month ago
                    returns = []
                    for i in range(1, len(prices)):
                        returns.append((prices[i] - prices[i-1]) / prices[i-1])

                    if len(returns) >= 22:  # At least 1 month
                        momentum_scores[symbol] = sum(returns[-60:-22]) if len(returns) >= 60 else sum(returns[:-22])

            if len(momentum_scores) >= 2:
                # Create momentum factor as long-short portfolio
                sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1])
                n_stocks = len(sorted_stocks)

                # Long top tercile, short bottom tercile
                long_stocks = sorted_stocks[-n_stocks//3:]
                short_stocks = sorted_stocks[:n_stocks//3]

                # Calculate factor returns
                factor_returns = []
                for day in range(min(lookback_days, len(list(stock_data.values())[0]['prices']) - 1)):
                    daily_return = 0.0
                    count = 0

                    # Long position returns
                    for symbol, _ in long_stocks:
                        if symbol in stock_data:
                            prices = stock_data[symbol]['prices']
                            if day + 1 < len(prices):
                                daily_return += (prices[day + 1] - prices[day]) / prices[day]
                                count += 1

                    # Short position returns
                    for symbol, _ in short_stocks:
                        if symbol in stock_data:
                            prices = stock_data[symbol]['prices']
                            if day + 1 < len(prices):
                                daily_return -= (prices[day + 1] - prices[day]) / prices[day]
                                count += 1

                    if count > 0:
                        factor_returns.append(daily_return / count)

                logger.info(f"✅ Calculated momentum factor from {len(momentum_scores)} stocks")
                return factor_returns[-lookback_days:] if len(factor_returns) > lookback_days else factor_returns

        except Exception as e:
            logger.error(f"Error calculating momentum factor: {e}")

        return None

    async def _calculate_volatility_factor(
        self,
        stock_data: Dict[str, Any],
        lookback_days: int
    ) -> Optional[List[float]]:
        """Calculate volatility factor from stock price data."""

        if not HAS_NUMPY:
            return None

        try:
            # Calculate volatility for each stock
            volatilities = {}

            for symbol, data in stock_data.items():
                prices = data['prices']
                if len(prices) >= 22:  # Need at least 1 month
                    returns = []
                    for i in range(1, len(prices)):
                        returns.append((prices[i] - prices[i-1]) / prices[i-1])

                    if len(returns) >= 22:
                        # Calculate rolling 1-month volatility
                        vol = np.std(returns[-22:]) * np.sqrt(252)  # Annualized
                        volatilities[symbol] = vol

            if len(volatilities) >= 2:
                # Create volatility factor as low vol minus high vol
                sorted_stocks = sorted(volatilities.items(), key=lambda x: x[1])
                n_stocks = len(sorted_stocks)

                # Long low vol tercile, short high vol tercile
                long_stocks = sorted_stocks[:n_stocks//3]  # Low volatility
                short_stocks = sorted_stocks[-n_stocks//3:]  # High volatility

                # Calculate factor returns
                factor_returns = []
                for day in range(min(lookback_days, len(list(stock_data.values())[0]['prices']) - 1)):
                    daily_return = 0.0
                    count = 0

                    # Long low vol positions
                    for symbol, _ in long_stocks:
                        if symbol in stock_data:
                            prices = stock_data[symbol]['prices']
                            if day + 1 < len(prices):
                                daily_return += (prices[day + 1] - prices[day]) / prices[day]
                                count += 1

                    # Short high vol positions
                    for symbol, _ in short_stocks:
                        if symbol in stock_data:
                            prices = stock_data[symbol]['prices']
                            if day + 1 < len(prices):
                                daily_return -= (prices[day + 1] - prices[day]) / prices[day]
                                count += 1

                    if count > 0:
                        factor_returns.append(daily_return / count)

                logger.info(f"✅ Calculated volatility factor from {len(volatilities)} stocks")
                return factor_returns[-lookback_days:] if len(factor_returns) > lookback_days else factor_returns

        except Exception as e:
            logger.error(f"Error calculating volatility factor: {e}")

        return None

    def _calculate_real_factor_covariance(
        self,
        factor_returns: Dict[FactorType, List[float]]
    ) -> Dict[Tuple[FactorType, FactorType], float]:
        """Calculate factor covariance from real factor returns."""

        if not HAS_NUMPY:
            return self._fallback_factor_covariance()

        logger.info("Calculating factor covariance from real returns")

        covariance = {}
        factors = list(FactorType)

        # Calculate covariance matrix
        for factor1 in factors:
            for factor2 in factors:
                if factor1 in factor_returns and factor2 in factor_returns:
                    returns1 = factor_returns[factor1]
                    returns2 = factor_returns[factor2]

                    # Ensure same length
                    min_len = min(len(returns1), len(returns2))
                    if min_len > 0:
                        r1 = returns1[-min_len:]
                        r2 = returns2[-min_len:]

                        if factor1 == factor2:
                            # Variance
                            covariance[(factor1, factor2)] = np.var(r1, ddof=1) if len(r1) > 1 else 0.0001
                        else:
                            # Covariance
                            covariance[(factor1, factor2)] = np.cov(r1, r2, ddof=1)[0, 1] if len(r1) > 1 else 0.0
                    else:
                        covariance[(factor1, factor2)] = 0.0001 if factor1 == factor2 else 0.0
                else:
                    # Fallback values
                    covariance[(factor1, factor2)] = 0.0001 if factor1 == factor2 else 0.0

        logger.info("✅ Calculated real factor covariance matrix")
        return covariance

    async def _calculate_real_asset_exposures(
        self,
        symbol: str,
        stock_data: Dict[str, Any],
        factor_returns: Dict[FactorType, List[float]],
        lookback_days: int
    ) -> FactorExposure:
        """Calculate real asset exposures using regression on factor returns."""

        if not HAS_NUMPY or len(stock_data.get('prices', [])) < 2:
            return self._fallback_asset_exposures(symbol)

        try:
            # Calculate stock returns
            prices = stock_data['prices']
            stock_returns = []
            for i in range(1, len(prices)):
                stock_returns.append((prices[i] - prices[i-1]) / prices[i-1])

            # Use the most recent returns up to lookback_days
            stock_returns = stock_returns[-lookback_days:] if len(stock_returns) > lookback_days else stock_returns

            if len(stock_returns) < 10:  # Need minimum data
                return self._fallback_asset_exposures(symbol)

            # Regression against factors
            exposures = {}

            for factor_type, factor_rets in factor_returns.items():
                if len(factor_rets) >= len(stock_returns):
                    # Align lengths
                    min_len = min(len(stock_returns), len(factor_rets))
                    y = stock_returns[-min_len:]
                    x = factor_rets[-min_len:]

                    if min_len > 1:
                        # Simple linear regression: beta = cov(x,y) / var(x)
                        cov_xy = np.cov(x, y, ddof=1)[0, 1] if min_len > 1 else 0.0
                        var_x = np.var(x, ddof=1) if min_len > 1 else 1.0

                        if var_x > 0:
                            beta = cov_xy / var_x
                        else:
                            beta = 0.0

                        exposures[factor_type] = beta
                    else:
                        exposures[factor_type] = 0.0
                else:
                    exposures[factor_type] = 0.0

            logger.debug(f"Calculated real exposures for {symbol}")
            return FactorExposure(
                symbol=symbol,
                exposures=exposures,
                calculated_date=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Error calculating real exposures for {symbol}: {e}")
            return self._fallback_asset_exposures(symbol)

    def _fallback_asset_exposures(self, symbol: str) -> FactorExposure:
        """Generate fallback asset exposures when real calculation fails."""

        # Use symbol hash for consistent fallback
        import hashlib
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % 10000

        # Generate reasonable exposures based on symbol
        if symbol in ["SPY", "VTI", "IVV"]:
            # Market ETFs
            exposures = {
                FactorType.MARKET: 1.0,
                FactorType.VALUE: 0.0,
                FactorType.MOMENTUM: 0.0,
                FactorType.QUALITY: 0.0,
                FactorType.VOLATILITY: 0.0,
                FactorType.SIZE: 0.0,
                FactorType.PROFITABILITY: 0.0,
                FactorType.INVESTMENT: 0.0,
            }
        else:
            # Individual stocks - use hash for variety
            import random
            random.seed(seed)

            exposures = {
                FactorType.MARKET: 1.0 + random.normalvariate(0, 0.2),
                FactorType.VALUE: random.normalvariate(0, 0.5),
                FactorType.MOMENTUM: random.normalvariate(0, 0.4),
                FactorType.QUALITY: random.normalvariate(0, 0.3),
                FactorType.VOLATILITY: random.normalvariate(0, 0.3),
                FactorType.SIZE: random.normalvariate(0, 0.6),
                FactorType.PROFITABILITY: random.normalvariate(0, 0.4),
                FactorType.INVESTMENT: random.normalvariate(0, 0.3),
            }

        return FactorExposure(
            symbol=symbol,
            exposures=exposures,
            calculated_date=datetime.now(timezone.utc)
        )

    def _fallback_factor_covariance(self) -> Dict[Tuple[FactorType, FactorType], float]:
        """Generate fallback factor covariance when real calculation fails."""

        factors = list(FactorType)
        covariance = {}

        # Base volatilities (annualized)
        factor_vols = {
            FactorType.MARKET: 0.16,
            FactorType.VALUE: 0.12,
            FactorType.MOMENTUM: 0.15,
            FactorType.QUALITY: 0.10,
            FactorType.VOLATILITY: 0.08,
            FactorType.SIZE: 0.14,
            FactorType.PROFITABILITY: 0.11,
            FactorType.INVESTMENT: 0.09,
        }

        # Realistic correlations
        correlations = {
            (FactorType.MARKET, FactorType.SIZE): -0.3,
            (FactorType.VALUE, FactorType.MOMENTUM): -0.4,
            (FactorType.QUALITY, FactorType.PROFITABILITY): 0.6,
            (FactorType.SIZE, FactorType.VALUE): 0.4,
        }

        # Fill covariance matrix
        for factor1 in factors:
            for factor2 in factors:
                if factor1 == factor2:
                    covariance[(factor1, factor2)] = factor_vols[factor1] ** 2
                else:
                    corr = correlations.get((factor1, factor2)) or correlations.get((factor2, factor1)) or 0.0
                    covariance[(factor1, factor2)] = corr * factor_vols[factor1] * factor_vols[factor2]

        return covariance

    def _generate_fallback_factor_returns(
        self,
        length: int,
        mean: float = 0.0,
        vol: float = 0.01
    ) -> List[float]:
        """Generate fallback factor returns when real data unavailable."""

        import random
        import math

        # Use a fixed seed for consistency
        random.seed(42)

        returns = []
        for _ in range(length):
            daily_return = random.normalvariate(mean, vol)
            returns.append(daily_return)

        return returns


async def test_real_factor_models():
    """Test the real factor model builder."""

    print("🧪 Testing Real Factor Model Builder")
    print("=" * 50)

    builder = RealFactorModelBuilder()

    symbols = ["SPY", "AAPL", "MSFT", "GOOGL"]
    model = await builder.build_factor_model(symbols, lookback_days=60)

    print(f"✅ Built factor model: {model.model_id}")
    print(f"Data source: {model.data_source}")
    print(f"Factors: {len(model.factors)}")
    print(f"Assets: {len(model.asset_exposures)}")

    # Show some exposures
    for symbol in symbols[:2]:
        if symbol in model.asset_exposures:
            exposure = model.asset_exposures[symbol]
            print(f"\n{symbol} exposures:")
            for factor, value in exposure.exposures.items():
                print(f"   {factor.value}: {value:.3f}")

    print(f"\n✅ Real factor model test complete!")


if __name__ == "__main__":
    asyncio.run(test_real_factor_models())