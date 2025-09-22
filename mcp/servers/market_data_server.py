"""Real-time market data MCP server with multiple data source integrations.

Provides real market data from multiple sources including Alpha Vantage, Yahoo Finance,
and IEX Cloud with fallback mechanisms and circuit breaker protection.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import aiohttp
except ImportError:  # pragma: no cover - optional dependency in tests
    aiohttp = None  # type: ignore

try:
    from mcp.server import register_tool
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
    from mcp.common.server_config import get_server_config, get_tool_config
except ImportError:  # pragma: no cover
    def register_tool(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator

    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available market data sources."""
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    MOCK = "mock"  # Fallback for testing


@dataclass
class MarketDataPoint:
    """Single market data point."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None
    source: DataSource = DataSource.MOCK
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: DataSource
    api_key: Optional[str] = None
    base_url: str = ""
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 10
    priority: int = 1  # Lower number = higher priority
    enabled: bool = True


class MarketDataProvider:
    """Real-time market data provider with multiple source support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiters: Dict[DataSource, List[float]] = {}

        # Initialize data sources
        self._data_sources = self._initialize_data_sources()

        # Circuit breaker configurations
        self._circuit_breakers = {}
        for source in self._data_sources:
            cb_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_seconds=60.0,
                expected_exception=Exception
            )
            self._circuit_breakers[source.name] = cb_config

        logger.info(f"Market data provider initialized with {len(self._data_sources)} sources")

    def _initialize_data_sources(self) -> List[DataSourceConfig]:
        """Initialize and prioritize data sources."""
        sources = []
        network_available = aiohttp is not None

        # Alpha Vantage
        av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if av_key and network_available:
            sources.append(DataSourceConfig(
                name=DataSource.ALPHA_VANTAGE,
                api_key=av_key,
                base_url="https://www.alphavantage.co/query",
                rate_limit_per_minute=5,  # Free tier limit
                priority=1,
                enabled=True
            ))

        # Yahoo Finance (no API key required)
        sources.append(DataSourceConfig(
            name=DataSource.YAHOO_FINANCE,
            base_url="https://query1.finance.yahoo.com/v8/finance/chart",
            rate_limit_per_minute=60,
            priority=2,
            enabled=network_available
        ))

        # IEX Cloud
        iex_key = os.getenv("IEX_CLOUD_API_KEY")
        if iex_key and network_available:
            sources.append(DataSourceConfig(
                name=DataSource.IEX_CLOUD,
                api_key=iex_key,
                base_url="https://cloud.iexapis.com/stable",
                rate_limit_per_minute=50,
                priority=3,
                enabled=True
            ))

        # Polygon.io
        polygon_key = os.getenv("POLYGON_API_KEY")
        if polygon_key and network_available:
            sources.append(DataSourceConfig(
                name=DataSource.POLYGON,
                api_key=polygon_key,
                base_url="https://api.polygon.io",
                rate_limit_per_minute=100,
                priority=4,
                enabled=True
            ))

        # Always include mock source as fallback
        sources.append(DataSourceConfig(
            name=DataSource.MOCK,
            priority=99,
            enabled=True
        ))

        # Sort by priority
        sources.sort(key=lambda x: x.priority)
        return sources

    async def __aenter__(self):
        """Async context manager entry."""
        if aiohttp is None:
            return self
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    def _check_rate_limit(self, source: DataSource) -> bool:
        """Check if we can make a request to this source."""
        now = time.time()
        source_config = next((s for s in self._data_sources if s.name == source), None)
        if not source_config:
            return False

        # Initialize rate limiter for source
        if source not in self._rate_limiters:
            self._rate_limiters[source] = []

        # Clean old timestamps (older than 1 minute)
        self._rate_limiters[source] = [
            ts for ts in self._rate_limiters[source]
            if now - ts < 60
        ]

        # Check if we're under the limit
        if len(self._rate_limiters[source]) >= source_config.rate_limit_per_minute:
            return False

        # Record this request
        self._rate_limiters[source].append(now)
        return True

    @circuit_breaker
    async def get_real_time_data(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Get real-time market data for symbols."""
        if aiohttp is None:
            # Offline mode - return deterministic mock data
            return [await self._fetch_mock_data(symbol.upper()) for symbol in symbols]

        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        results = []

        for symbol in symbols:
            data_point = await self._get_symbol_data(symbol.upper())
            if data_point:
                results.append(data_point)

        logger.info(f"Retrieved real-time data for {len(results)}/{len(symbols)} symbols")
        return results

    async def _get_symbol_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get data for a single symbol with fallback through data sources."""

        for source_config in self._data_sources:
            if not source_config.enabled:
                continue

            # Check rate limiting
            if not self._check_rate_limit(source_config.name):
                logger.debug(f"Rate limit reached for {source_config.name.value}")
                continue

            try:
                # Circuit breaker check
                if source_config.name in self._circuit_breakers:
                    # In a real implementation, this would use the actual circuit breaker
                    pass

                data_point = await self._fetch_from_source(symbol, source_config)
                if data_point:
                    logger.debug(f"Got data for {symbol} from {source_config.name.value}")
                    return data_point

            except Exception as e:
                logger.warning(f"Failed to get {symbol} from {source_config.name.value}: {e}")
                continue

        logger.error(f"Failed to get data for {symbol} from all sources")
        return None

    async def _fetch_from_source(self, symbol: str, source_config: DataSourceConfig) -> Optional[MarketDataPoint]:
        """Fetch data from a specific source."""

        if source_config.name == DataSource.ALPHA_VANTAGE:
            return await self._fetch_alpha_vantage(symbol, source_config)
        elif source_config.name == DataSource.YAHOO_FINANCE:
            return await self._fetch_yahoo_finance(symbol, source_config)
        elif source_config.name == DataSource.IEX_CLOUD:
            return await self._fetch_iex_cloud(symbol, source_config)
        elif source_config.name == DataSource.POLYGON:
            return await self._fetch_polygon(symbol, source_config)
        elif source_config.name == DataSource.MOCK:
            return await self._fetch_mock_data(symbol)

        return None

    async def _fetch_alpha_vantage(self, symbol: str, config: DataSourceConfig) -> Optional[MarketDataPoint]:
        """Fetch data from Alpha Vantage."""
        try:
            url = f"{config.base_url}?function=GLOBAL_QUOTE&symbol={symbol}&apikey={config.api_key}"

            async with self._session.get(url, timeout=config.timeout_seconds) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                quote = data.get("Global Quote", {})

                if not quote:
                    return None

                return MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    price=float(quote.get("05. price", 0)),
                    volume=int(float(quote.get("06. volume", 0))),
                    open_price=float(quote.get("02. open", 0)),
                    high_24h=float(quote.get("03. high", 0)),
                    low_24h=float(quote.get("04. low", 0)),
                    close_price=float(quote.get("08. previous close", 0)),
                    source=DataSource.ALPHA_VANTAGE,
                    metadata={
                        "change_percent": quote.get("10. change percent", "0%"),
                        "last_trading_day": quote.get("07. latest trading day")
                    }
                )

        except Exception as e:
            logger.error(f"Alpha Vantage fetch error for {symbol}: {e}")
            return None

    async def _fetch_yahoo_finance(self, symbol: str, config: DataSourceConfig) -> Optional[MarketDataPoint]:
        """Fetch data from Yahoo Finance."""
        try:
            url = f"{config.base_url}/{symbol}?interval=1m&range=1d"

            async with self._session.get(url, timeout=config.timeout_seconds) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                chart = data.get("chart", {})
                results = chart.get("result", [])

                if not results:
                    return None

                result = results[0]
                meta = result.get("meta", {})
                timestamps = result.get("timestamp", [])
                quotes = result.get("indicators", {}).get("quote", [{}])[0]

                if not timestamps:
                    return None

                # Get the latest data point
                latest_idx = -1
                latest_timestamp = datetime.fromtimestamp(timestamps[latest_idx], tz=timezone.utc)

                return MarketDataPoint(
                    symbol=symbol,
                    timestamp=latest_timestamp,
                    price=float(meta.get("regularMarketPrice", 0)),
                    volume=int(meta.get("regularMarketVolume", 0)),
                    open_price=float(quotes.get("open", [0])[latest_idx] or 0),
                    high_24h=float(quotes.get("high", [0])[latest_idx] or 0),
                    low_24h=float(quotes.get("low", [0])[latest_idx] or 0),
                    close_price=float(quotes.get("close", [0])[latest_idx] or 0),
                    bid=float(meta.get("bid", 0)),
                    ask=float(meta.get("ask", 0)),
                    source=DataSource.YAHOO_FINANCE,
                    metadata={
                        "currency": meta.get("currency"),
                        "exchange": meta.get("exchangeName"),
                        "market_state": meta.get("marketState")
                    }
                )

        except Exception as e:
            logger.error(f"Yahoo Finance fetch error for {symbol}: {e}")
            return None

    async def _fetch_iex_cloud(self, symbol: str, config: DataSourceConfig) -> Optional[MarketDataPoint]:
        """Fetch data from IEX Cloud."""
        try:
            url = f"{config.base_url}/stock/{symbol}/quote?token={config.api_key}"

            async with self._session.get(url, timeout=config.timeout_seconds) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                return MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(data.get("latestUpdate", 0) / 1000, tz=timezone.utc),
                    price=float(data.get("latestPrice", 0)),
                    volume=int(data.get("latestVolume", 0)),
                    open_price=float(data.get("open", 0)),
                    high_24h=float(data.get("high", 0)),
                    low_24h=float(data.get("low", 0)),
                    close_price=float(data.get("close", 0)),
                    bid=float(data.get("iexBidPrice", 0)),
                    ask=float(data.get("iexAskPrice", 0)),
                    source=DataSource.IEX_CLOUD,
                    metadata={
                        "change": data.get("change"),
                        "change_percent": data.get("changePercent"),
                        "market_cap": data.get("marketCap"),
                        "pe_ratio": data.get("peRatio")
                    }
                )

        except Exception as e:
            logger.error(f"IEX Cloud fetch error for {symbol}: {e}")
            return None

    async def _fetch_polygon(self, symbol: str, config: DataSourceConfig) -> Optional[MarketDataPoint]:
        """Fetch data from Polygon.io."""
        try:
            url = f"{config.base_url}/v2/aggs/ticker/{symbol}/prev?adjusted=true&apikey={config.api_key}"

            async with self._session.get(url, timeout=config.timeout_seconds) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                results = data.get("results", [])

                if not results:
                    return None

                result = results[0]

                return MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(result.get("t", 0) / 1000, tz=timezone.utc),
                    price=float(result.get("c", 0)),  # Close price as current
                    volume=int(result.get("v", 0)),
                    open_price=float(result.get("o", 0)),
                    high_24h=float(result.get("h", 0)),
                    low_24h=float(result.get("l", 0)),
                    close_price=float(result.get("c", 0)),
                    source=DataSource.POLYGON,
                    metadata={
                        "vwap": result.get("vw"),
                        "transactions": result.get("n")
                    }
                )

        except Exception as e:
            logger.error(f"Polygon fetch error for {symbol}: {e}")
            return None

    async def _fetch_mock_data(self, symbol: str) -> MarketDataPoint:
        """Generate mock data as fallback."""
        # Generate deterministic but varied mock data
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        base_price = 50 + (seed % 200)  # $50-$250 range

        # Add some time-based variation
        time_factor = int(time.time()) % 1000
        price_variation = (time_factor % 21 - 10) / 100  # ±10%
        current_price = base_price * (1 + price_variation)

        return MarketDataPoint(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            price=current_price,
            volume=1000000 + (seed % 5000000),
            bid=current_price - 0.01,
            ask=current_price + 0.01,
            high_24h=current_price * 1.05,
            low_24h=current_price * 0.95,
            open_price=current_price * 0.98,
            close_price=current_price * 1.01,
            source=DataSource.MOCK,
            metadata={
                "warning": "Mock data - not for production use",
                "seed": seed
            }
        )


# Global provider instance
_market_data_provider = None


async def _get_provider() -> MarketDataProvider:
    """Get or create market data provider instance."""
    global _market_data_provider
    if _market_data_provider is None:
        _market_data_provider = MarketDataProvider()
    return _market_data_provider


_INTERVAL_SECONDS = {
    "minute": 60,
    "hour": 3600,
    "day": 86400,
}


def _parse_timestamp(value: str) -> datetime:
    """Parse ISO-8601 timestamp into aware UTC datetime."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _generate_mock_ohlcv_rows(
    symbol: str,
    start: datetime,
    end: datetime,
    interval_seconds: int,
    *,
    base_price: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Generate deterministic mock OHLCV rows for offline/test scenarios."""

    if end <= start:
        raise ValueError("end must be after start")

    total_seconds = (end - start).total_seconds()
    max_points = 5000
    estimated_points = int(total_seconds // interval_seconds) + 1
    if estimated_points > max_points:
        raise ValueError("Requested window exceeds maximum of 5000 data points")

    seed_src = f"{symbol}:{start.isoformat()}:{interval_seconds}"
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    if base_price is None:
        base_price = 75.0 + (seed % 100)

    rows: List[Dict[str, Any]] = []
    price = float(base_price)
    current = start
    idx = 0
    while current <= end and len(rows) < max_points:
        drift = 0.0005 * ((seed >> (idx % 8)) & 0x7) - 0.0015
        shock = ((seed >> (idx % 12)) & 0xF) / 32.0 - 0.25
        open_price = price
        close_price = max(1.0, open_price * (1 + drift + shock * 0.02))
        high = max(open_price, close_price) * (1 + abs(shock) * 0.03)
        low = min(open_price, close_price) * (1 - abs(shock) * 0.03)
        volume = int(300000 + (seed % 120000) + idx * 850)
        provenance_id = f"mock::{symbol}::{current.isoformat()}"

        rows.append({
            "timestamp": current.isoformat(),
            "open": round(open_price, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "close": round(close_price, 4),
            "volume": volume,
            "provenance_id": provenance_id,
        })

        price = close_price
        current += timedelta(seconds=interval_seconds)
        idx += 1

    return rows


@register_tool(
    name="market-data.pricing.get_ohlcv",
    schema="./schemas/tool.market-data.pricing.get_ohlcv.schema.json",
)
async def get_ohlcv(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return OHLCV bars for the specified symbol and window."""

    symbol = params["symbol"].upper()
    start = _parse_timestamp(params["start"])
    end = _parse_timestamp(params["end"])
    interval = params["interval"]
    adjustment = params.get("adjustment", "split")
    vendor = params.get("vendor")

    if interval not in _INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval '{interval}'")

    interval_seconds = _INTERVAL_SECONDS[interval]
    if adjustment not in {"raw", "split", "dividend"}:
        raise ValueError("adjustment must be one of 'raw', 'split', or 'dividend'")

    provider = await _get_provider()
    try:
        async with provider as active_provider:
            # Get historical OHLCV data - NO FALLBACKS
            rows = await active_provider.get_historical_ohlcv(
                symbol, start, end, interval
            )

            if not rows:
                from mcp.common.exceptions import DataUnavailableError
                raise DataUnavailableError(
                    f"No historical data available for {symbol} from {start} to {end}. "
                    f"Please ensure market data provider has this symbol and date range."
                )

            # Generate provenance IDs for real data
            provenance_ids = [f"real_{vendor or 'default'}_{i}" for i in range(len(rows))]

            return {
                "symbol": symbol,
                "interval": interval,
                "rows": rows,
                "provenance_ids": provenance_ids,
                "data_source": "real",
                "vendor": vendor or active_provider.__class__.__name__
            }

    except Exception as exc:
        # DO NOT fall back to synthetic data - this is a real data-only endpoint
        logger.error(f"Failed to get real OHLCV data for {symbol}: {exc}")
        from mcp.common.exceptions import DataUnavailableError
        raise DataUnavailableError(
            f"Unable to retrieve market data for {symbol}: {str(exc)}. "
            f"This is a real data-only endpoint - synthetic fallbacks have been removed. "
            f"Please ensure market data provider credentials are configured correctly."
        )


async def get_real_time_market_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get real-time market data for given symbols."""
    try:
        symbols = params.get("symbols", [])
        if not symbols:
            return {"error": "No symbols provided"}

        if len(symbols) > 50:  # Reasonable limit
            return {"error": f"Too many symbols: {len(symbols)} (max 50)"}

        provider = await _get_provider()

        async with provider as p:
            data_points = await p.get_real_time_data(symbols)

            # Convert to dict format for JSON serialization
            result_data = []
            for point in data_points:
                point_dict = asdict(point)
                # Convert datetime to ISO string
                point_dict["timestamp"] = point.timestamp.isoformat()
                result_data.append(point_dict)

            return {
                "success": True,
                "data": result_data,
                "count": len(result_data),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    except Exception as e:
        logger.error(f"Real-time market data error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
async def get_historical_market_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get historical market data for analysis."""
    try:
        symbol = params.get("symbol")
        days = params.get("days", 30)

        if not symbol:
            return {"error": "Symbol is required"}

        if days > 365:  # Reasonable limit
            return {"error": f"Too many days: {days} (max 365)"}

        # For now, return mock historical data
        # In production, this would fetch from data sources
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        historical_data = []
        current_time = start_time
        base_price = 100.0

        while current_time <= end_time:
            # Simple random walk for mock data
            price_change = (hash(current_time.isoformat()) % 100) / 500 - 0.1
            base_price = max(base_price * (1 + price_change), 1.0)

            point = MarketDataPoint(
                symbol=symbol,
                timestamp=current_time,
                price=base_price,
                volume=1000000,
                open_price=base_price * 0.99,
                high_24h=base_price * 1.02,
                low_24h=base_price * 0.98,
                close_price=base_price,
                source=DataSource.MOCK,
                metadata={"historical": True}
            )

            point_dict = asdict(point)
            point_dict["timestamp"] = point.timestamp.isoformat()
            historical_data.append(point_dict)

            current_time += timedelta(hours=1)

        return {
            "success": True,
            "symbol": symbol,
            "data": historical_data,
            "count": len(historical_data),
            "period": f"{days} days"
        }

    except Exception as e:
        logger.error(f"Historical market data error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


__all__ = [
    "MarketDataProvider",
    "MarketDataPoint",
    "DataSource",
    "get_ohlcv",
    "get_real_time_market_data",
    "get_historical_market_data"
]
