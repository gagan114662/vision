"""
Production Portfolio Engine with Real Factor Model Integration.

This connects the factor models to the trading workflow with real portfolio
construction, rebalancing logic, and performance attribution.
"""
from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json

try:
    from .factor_models import (
        FactorModelBuilder, PortfolioOptimizer, FactorRiskModel,
        PortfolioOptimizationResult, FactorType
    )
except ImportError:
    from factor_models import (
        FactorModelBuilder, PortfolioOptimizer, FactorRiskModel,
        PortfolioOptimizationResult, FactorType
    )

logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class PortfolioPosition:
    """Individual position in portfolio."""
    symbol: str
    target_weight: float
    current_weight: float
    market_value: float
    shares: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    factor_exposures: Dict[FactorType, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_return(self) -> float:
        """Calculate total return for position."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    def update_price(self, new_price: float) -> None:
        """Update position with new price."""
        self.current_price = new_price
        self.market_value = self.shares * new_price
        self.unrealized_pnl = self.shares * (new_price - self.entry_price)
        self.last_updated = datetime.now(timezone.utc)


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""
    snapshot_id: str
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, PortfolioPosition]
    factor_exposures: Dict[FactorType, float]
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]

    def get_weights(self) -> Dict[str, float]:
        """Get current position weights."""
        if self.total_value == 0:
            return {}
        return {
            symbol: position.market_value / self.total_value
            for symbol, position in self.positions.items()
        }


@dataclass
class RebalanceSignal:
    """Signal to rebalance portfolio."""
    signal_id: str
    timestamp: datetime
    trigger_reason: str
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    weight_deviations: Dict[str, float]
    estimated_turnover: float
    estimated_costs: float

    @property
    def max_deviation(self) -> float:
        """Maximum weight deviation."""
        return max(abs(dev) for dev in self.weight_deviations.values()) if self.weight_deviations else 0.0

    def should_rebalance(self, threshold: float = 0.05) -> bool:
        """Check if rebalancing is needed."""
        return self.max_deviation > threshold


class PortfolioEngine:
    """Production portfolio engine with factor model integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.engine_id = self.config.get("engine_id", f"portfolio_engine_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")

        # Configuration
        self.initial_capital = self.config.get("initial_capital", 1000000)  # $1M default
        self.rebalance_frequency = RebalanceFrequency(self.config.get("rebalance_frequency", "monthly"))
        self.rebalance_threshold = self.config.get("rebalance_threshold", 0.05)  # 5% weight deviation
        self.transaction_cost = self.config.get("transaction_cost", 0.001)  # 10 bps
        self.cash_buffer = self.config.get("cash_buffer", 0.02)  # 2% cash buffer

        # Components
        self.factor_model_builder = FactorModelBuilder(self.config.get("factor_model", {}))
        self.optimizer = PortfolioOptimizer(self.config.get("optimization", {}))

        # State
        self.current_portfolio: Optional[PortfolioSnapshot] = None
        self.factor_model: Optional[FactorRiskModel] = None
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.rebalance_history: List[RebalanceSignal] = []

        logger.info(f"Portfolio engine initialized: {self.engine_id}")

    async def initialize_portfolio(
        self,
        target_symbols: List[str],
        optimization_method: str = "black_litterman",
        initial_weights: Optional[Dict[str, float]] = None,
        agent_views: Optional[Dict[str, float]] = None
    ) -> PortfolioSnapshot:
        """Initialize portfolio with optimal weights."""

        logger.info(f"Initializing portfolio with {len(target_symbols)} symbols using {optimization_method}")

        # Build factor model
        self.factor_model = await self.factor_model_builder.build_factor_model(target_symbols)

        # Get optimal portfolio
        optimization_result = await self.optimizer.optimize_portfolio(
            factor_model=self.factor_model,
            target_symbols=target_symbols,
            method=optimization_method,
            views=agent_views
        )

        # Create initial positions
        positions = {}
        investable_capital = self.initial_capital * (1 - self.cash_buffer)

        for symbol in target_symbols:
            target_weight = optimization_result.optimal_weights.get(symbol, 0.0)

            if target_weight > 0:
                target_value = investable_capital * target_weight
                # Simulate entry price
                entry_price = self._get_current_price(symbol)
                shares = int(target_value / entry_price)
                actual_value = shares * entry_price

                position = PortfolioPosition(
                    symbol=symbol,
                    target_weight=target_weight,
                    current_weight=actual_value / self.initial_capital,
                    market_value=actual_value,
                    shares=shares,
                    entry_price=entry_price,
                    current_price=entry_price,
                    unrealized_pnl=0.0,
                    factor_exposures=self._get_position_factor_exposures(symbol)
                )
                positions[symbol] = position

        # Calculate actual cash after purchases
        total_invested = sum(pos.market_value for pos in positions.values())
        cash = self.initial_capital - total_invested

        # Create initial snapshot
        portfolio_snapshot = PortfolioSnapshot(
            snapshot_id=f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc),
            total_value=self.initial_capital,
            cash=cash,
            positions=positions,
            factor_exposures=self._calculate_portfolio_factor_exposures(positions),
            risk_metrics=self._calculate_risk_metrics(positions),
            performance_metrics=self._calculate_performance_metrics(positions)
        )

        self.current_portfolio = portfolio_snapshot
        self.portfolio_history.append(portfolio_snapshot)

        logger.info(f"Portfolio initialized with {len(positions)} positions, {cash/self.initial_capital:.1%} cash")
        return portfolio_snapshot

    async def update_portfolio_prices(self, price_updates: Dict[str, float]) -> PortfolioSnapshot:
        """Update portfolio with new market prices."""

        if not self.current_portfolio:
            raise ValueError("Portfolio not initialized")

        logger.debug(f"Updating portfolio prices for {len(price_updates)} symbols")

        # Update position prices
        updated_positions = {}
        for symbol, position in self.current_portfolio.positions.items():
            updated_position = PortfolioPosition(
                symbol=position.symbol,
                target_weight=position.target_weight,
                current_weight=position.current_weight,  # Will recalculate below
                market_value=position.market_value,
                shares=position.shares,
                entry_price=position.entry_price,
                current_price=position.current_price,
                unrealized_pnl=position.unrealized_pnl,
                factor_exposures=position.factor_exposures,
                last_updated=position.last_updated
            )

            if symbol in price_updates:
                updated_position.update_price(price_updates[symbol])

            updated_positions[symbol] = updated_position

        # Recalculate portfolio totals
        total_market_value = sum(pos.market_value for pos in updated_positions.values())
        total_portfolio_value = total_market_value + self.current_portfolio.cash

        # Update current weights
        for position in updated_positions.values():
            position.current_weight = position.market_value / total_portfolio_value if total_portfolio_value > 0 else 0

        # Create new snapshot
        updated_snapshot = PortfolioSnapshot(
            snapshot_id=f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc),
            total_value=total_portfolio_value,
            cash=self.current_portfolio.cash,
            positions=updated_positions,
            factor_exposures=self._calculate_portfolio_factor_exposures(updated_positions),
            risk_metrics=self._calculate_risk_metrics(updated_positions),
            performance_metrics=self._calculate_performance_metrics(updated_positions)
        )

        self.current_portfolio = updated_snapshot
        self.portfolio_history.append(updated_snapshot)

        return updated_snapshot

    async def check_rebalance_signal(self) -> Optional[RebalanceSignal]:
        """Check if portfolio needs rebalancing."""

        if not self.current_portfolio or not self.factor_model:
            return None

        current_weights = self.current_portfolio.get_weights()
        symbols = list(self.current_portfolio.positions.keys())

        # Get fresh optimal weights
        optimization_result = await self.optimizer.optimize_portfolio(
            factor_model=self.factor_model,
            target_symbols=symbols,
            method="black_litterman"  # Could make configurable
        )

        target_weights = optimization_result.optimal_weights

        # Calculate deviations
        weight_deviations = {}
        for symbol in symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            weight_deviations[symbol] = current_weight - target_weight

        # Estimate turnover and costs
        turnover = sum(abs(dev) for dev in weight_deviations.values()) / 2
        estimated_costs = turnover * self.transaction_cost * self.current_portfolio.total_value

        rebalance_signal = RebalanceSignal(
            signal_id=f"rebalance_signal_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc),
            trigger_reason="periodic_check",
            current_weights=current_weights,
            target_weights=target_weights,
            weight_deviations=weight_deviations,
            estimated_turnover=turnover,
            estimated_costs=estimated_costs
        )

        if rebalance_signal.should_rebalance(self.rebalance_threshold):
            logger.info(f"Rebalance signal generated: max deviation {rebalance_signal.max_deviation:.1%}")
            self.rebalance_history.append(rebalance_signal)
            return rebalance_signal

        return None

    async def execute_rebalance(self, rebalance_signal: RebalanceSignal) -> PortfolioSnapshot:
        """Execute portfolio rebalancing."""

        if not self.current_portfolio:
            raise ValueError("Portfolio not initialized")

        logger.info(f"Executing rebalance with {rebalance_signal.estimated_turnover:.1%} turnover")

        target_weights = rebalance_signal.target_weights
        total_value = self.current_portfolio.total_value
        transaction_costs = rebalance_signal.estimated_costs

        # Adjust available capital for transaction costs
        available_capital = total_value - transaction_costs - (self.cash_buffer * total_value)

        # Calculate new positions
        new_positions = {}
        total_invested = 0

        for symbol, target_weight in target_weights.items():
            if target_weight <= 0:
                continue

            target_value = available_capital * target_weight
            current_price = self._get_current_price(symbol)
            new_shares = int(target_value / current_price)
            actual_value = new_shares * current_price

            # Get existing position for entry price tracking
            existing_position = self.current_portfolio.positions.get(symbol)
            entry_price = existing_position.entry_price if existing_position else current_price

            new_position = PortfolioPosition(
                symbol=symbol,
                target_weight=target_weight,
                current_weight=actual_value / total_value,
                market_value=actual_value,
                shares=new_shares,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=new_shares * (current_price - entry_price),
                factor_exposures=self._get_position_factor_exposures(symbol)
            )

            new_positions[symbol] = new_position
            total_invested += actual_value

        # Update cash after rebalancing
        new_cash = total_value - total_invested - transaction_costs

        # Create rebalanced snapshot
        rebalanced_snapshot = PortfolioSnapshot(
            snapshot_id=f"rebalanced_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc),
            total_value=total_value - transaction_costs,  # Account for costs
            cash=new_cash,
            positions=new_positions,
            factor_exposures=self._calculate_portfolio_factor_exposures(new_positions),
            risk_metrics=self._calculate_risk_metrics(new_positions),
            performance_metrics=self._calculate_performance_metrics(new_positions)
        )

        self.current_portfolio = rebalanced_snapshot
        self.portfolio_history.append(rebalanced_snapshot)

        logger.info(f"Rebalancing complete: {len(new_positions)} positions, {new_cash/rebalanced_snapshot.total_value:.1%} cash")
        return rebalanced_snapshot

    async def run_performance_attribution(self) -> Dict[str, Any]:
        """Run factor-based performance attribution."""

        if len(self.portfolio_history) < 2:
            return {"error": "Insufficient history for attribution"}

        current = self.portfolio_history[-1]
        previous = self.portfolio_history[-2]

        # Calculate returns
        portfolio_return = (current.total_value - previous.total_value) / previous.total_value

        # Factor attribution
        factor_returns = {}
        if self.factor_model:
            for factor in FactorType:
                current_exposure = current.factor_exposures.get(factor, 0.0)
                factor_return = self.factor_model.factor_returns.get(factor, 0.0)
                factor_returns[factor] = current_exposure * factor_return

        # Security selection (residual return)
        total_factor_return = sum(factor_returns.values())
        security_selection = portfolio_return - total_factor_return

        attribution = {
            "portfolio_return": portfolio_return,
            "factor_attribution": {factor.value: ret for factor, ret in factor_returns.items()},
            "security_selection": security_selection,
            "total_factor_contribution": total_factor_return,
            "period": {
                "start": previous.timestamp.isoformat(),
                "end": current.timestamp.isoformat()
            }
        }

        logger.info(f"Performance attribution: {portfolio_return:.2%} total return, {security_selection:.2%} alpha")
        return attribution

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""

        if not self.current_portfolio:
            return {"error": "Portfolio not initialized"}

        portfolio = self.current_portfolio
        summary = {
            "portfolio_id": self.engine_id,
            "timestamp": portfolio.timestamp.isoformat(),
            "total_value": portfolio.total_value,
            "cash": portfolio.cash,
            "cash_percentage": portfolio.cash / portfolio.total_value if portfolio.total_value > 0 else 0,
            "position_count": len(portfolio.positions),

            "holdings": {
                symbol: {
                    "weight": position.current_weight,
                    "market_value": position.market_value,
                    "shares": position.shares,
                    "unrealized_pnl": position.unrealized_pnl,
                    "total_return": position.total_return
                }
                for symbol, position in portfolio.positions.items()
            },

            "factor_exposures": {factor.value: exposure for factor, exposure in portfolio.factor_exposures.items()},
            "risk_metrics": portfolio.risk_metrics,
            "performance_metrics": portfolio.performance_metrics,

            "statistics": {
                "total_unrealized_pnl": sum(pos.unrealized_pnl for pos in portfolio.positions.values()),
                "largest_position": max(portfolio.positions.items(), key=lambda x: x[1].current_weight)[0] if portfolio.positions else None,
                "rebalance_count": len(self.rebalance_history),
                "last_rebalance": self.rebalance_history[-1].timestamp.isoformat() if self.rebalance_history else None
            }
        }

        return summary

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (simulated)."""
        # In production, this would connect to market data feed
        base_prices = {
            "AAPL": 175.0,
            "GOOGL": 135.0,
            "MSFT": 380.0,
            "TSLA": 250.0,
            "NVDA": 450.0,
            "AMZN": 145.0,
            "META": 320.0
        }

        # Add some randomness for simulation
        base_price = base_prices.get(symbol, 100.0)
        variation = (hash(symbol + str(datetime.now().hour)) % 200 - 100) / 1000  # ±10%
        return base_price * (1 + variation)

    def _get_position_factor_exposures(self, symbol: str) -> Dict[FactorType, float]:
        """Get factor exposures for position."""
        if not self.factor_model or symbol not in self.factor_model.asset_exposures:
            return {}

        exposure = self.factor_model.asset_exposures[symbol]
        return exposure.factor_loadings

    def _calculate_portfolio_factor_exposures(self, positions: Dict[str, PortfolioPosition]) -> Dict[FactorType, float]:
        """Calculate portfolio-level factor exposures."""
        if not self.factor_model:
            return {}

        portfolio_exposures = {}
        total_value = sum(pos.market_value for pos in positions.values())

        if total_value == 0:
            return portfolio_exposures

        for factor in FactorType:
            weighted_exposure = sum(
                (pos.market_value / total_value) * pos.factor_exposures.get(factor, 0.0)
                for pos in positions.values()
            )
            portfolio_exposures[factor] = weighted_exposure

        return portfolio_exposures

    def _calculate_risk_metrics(self, positions: Dict[str, PortfolioPosition]) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""

        if not self.factor_model or not positions:
            return {}

        weights = {}
        total_value = sum(pos.market_value for pos in positions.values())

        if total_value == 0:
            return {}

        for symbol, position in positions.items():
            weights[symbol] = position.market_value / total_value

        portfolio_risk = self.factor_model.calculate_portfolio_risk(weights)

        # Additional risk metrics
        concentration = max(weights.values()) if weights else 0
        effective_positions = 1 / sum(w**2 for w in weights.values()) if weights else 0

        return {
            "portfolio_volatility": portfolio_risk,
            "concentration_risk": concentration,
            "effective_positions": effective_positions,
            "diversification_ratio": len(positions) / effective_positions if effective_positions > 0 else 0
        }

    def _calculate_performance_metrics(self, positions: Dict[str, PortfolioPosition]) -> Dict[str, float]:
        """Calculate performance metrics."""

        if not positions:
            return {}

        total_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        total_invested = sum(pos.shares * pos.entry_price for pos in positions.values())

        if total_invested == 0:
            return {}

        unrealized_return = total_pnl / total_invested

        # Winner/loser ratio
        winners = sum(1 for pos in positions.values() if pos.unrealized_pnl > 0)
        losers = sum(1 for pos in positions.values() if pos.unrealized_pnl < 0)
        winner_ratio = winners / len(positions) if positions else 0

        return {
            "unrealized_return": unrealized_return,
            "total_unrealized_pnl": total_pnl,
            "winner_ratio": winner_ratio,
            "position_count": len(positions)
        }


async def main():
    """Demonstrate portfolio engine functionality."""
    print("🏗️ Portfolio Engine Demonstration")
    print("=" * 60)

    # Initialize engine
    config = {
        "initial_capital": 1000000,
        "rebalance_threshold": 0.03,  # 3% threshold
        "transaction_cost": 0.0005,   # 5 bps
        "optimization": {"risk_aversion": 2.5}
    }

    engine = PortfolioEngine(config)
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

    # Initialize portfolio
    print(f"\n🚀 Initializing portfolio with {len(symbols)} symbols...")

    # Simulate agent views
    agent_views = {
        "AAPL": 0.12,   # 12% expected return
        "NVDA": 0.18,   # 18% expected return
        "TSLA": 0.10    # 10% expected return
    }

    initial_portfolio = await engine.initialize_portfolio(
        target_symbols=symbols,
        optimization_method="black_litterman",
        agent_views=agent_views
    )

    print(f"✅ Portfolio initialized with ${initial_portfolio.total_value:,.0f}")

    # Show initial allocation
    print(f"\n📊 Initial Allocation:")
    for symbol, position in initial_portfolio.positions.items():
        print(f"   {symbol}: {position.current_weight:.1%} (${position.market_value:,.0f})")
    print(f"   Cash: {initial_portfolio.cash/initial_portfolio.total_value:.1%} (${initial_portfolio.cash:,.0f})")

    # Simulate price changes
    print(f"\n📈 Simulating market movements...")
    price_changes = {
        "AAPL": 180.0,   # +2.9%
        "GOOGL": 130.0,  # -3.7%
        "MSFT": 390.0,   # +2.6%
        "TSLA": 270.0,   # +8.0%
        "NVDA": 480.0    # +6.7%
    }

    updated_portfolio = await engine.update_portfolio_prices(price_changes)
    print(f"✅ Portfolio updated: ${updated_portfolio.total_value:,.0f} (+{((updated_portfolio.total_value/initial_portfolio.total_value)-1)*100:.1f}%)")

    # Check rebalancing
    print(f"\n⚖️ Checking rebalancing signal...")
    rebalance_signal = await engine.check_rebalance_signal()

    if rebalance_signal:
        print(f"🔄 Rebalance needed: {rebalance_signal.max_deviation:.1%} max deviation")
        print(f"   Estimated turnover: {rebalance_signal.estimated_turnover:.1%}")
        print(f"   Estimated costs: ${rebalance_signal.estimated_costs:,.0f}")

        # Execute rebalance
        rebalanced_portfolio = await engine.execute_rebalance(rebalance_signal)
        print(f"✅ Rebalanced: ${rebalanced_portfolio.total_value:,.0f}")
    else:
        print("✅ No rebalancing needed")

    # Performance attribution
    print(f"\n🎯 Performance Attribution:")
    attribution = await engine.run_performance_attribution()

    if "error" not in attribution:
        print(f"   Portfolio Return: {attribution['portfolio_return']:.2%}")
        print(f"   Security Selection: {attribution['security_selection']:.2%}")
        print(f"   Factor Contributions:")
        for factor, contribution in attribution["factor_attribution"].items():
            if abs(contribution) > 0.001:  # Only show significant contributions
                print(f"      {factor}: {contribution:.2%}")

    # Final summary
    print(f"\n📋 Portfolio Summary:")
    summary = engine.get_portfolio_summary()
    print(f"   Total Value: ${summary['total_value']:,.0f}")
    print(f"   Positions: {summary['position_count']}")
    print(f"   Cash: {summary['cash_percentage']:.1%}")
    print(f"   Total P&L: ${summary['statistics']['total_unrealized_pnl']:,.0f}")
    print(f"   Largest Position: {summary['statistics']['largest_position']}")

    print(f"\n✅ Portfolio Engine demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())