from __future__ import annotations

from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from AlgorithmImports import *


class MonthlyUniverseAlpha(QCAlgorithm):
    """Monthly rebalanced universe strategy combining momentum and volatility filters."""

    def Initialize(self) -> None:
        self.SetStartDate(2010, 1, 1)
        self.SetCash(1_000_000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetBenchmark("SPY")

        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.Leverage = 1.0

        self.rebalance_period = timedelta(days=30)
        self.next_rebalance = self.Time
        self.next_selection = self.Time
        self.last_selected: List[Symbol] = []

        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        self.symbol_scores: Dict[Symbol, float] = {}
        self.active: Dict[Symbol, Security] = {}

    def CoarseSelectionFunction(self, coarse: List[CoarseFundamental]) -> List[Symbol]:
        if coarse is None:
            return self.last_selected

        if self.Time < self.next_selection and self.last_selected:
            return self.last_selected

        filtered = [
            c for c in coarse
            if c.HasFundamentalData and c.Price > 5 and c.Volume > 1_000_000
        ]
        top = sorted(filtered, key=lambda c: c.DollarVolume, reverse=True)[:200]
        symbols = [c.Symbol for c in top]

        self.next_selection = self.Time + self.rebalance_period
        self.last_selected = symbols
        return symbols

    def FineSelectionFunction(self, fine: List[FineFundamental]) -> List[Symbol]:
        if fine is None or (self.Time < self.next_selection and self.last_selected):
            return self.last_selected

        filtered = [
            f for f in fine
            if f.CompanyReference.CountryId == "USA"
            and f.MarketCap and f.MarketCap > 2_000_000_000
            and f.FinancialStatements.IncomeStatement.NetIncome.TwelveMonths > 0
        ]
        top = sorted(filtered, key=lambda f: f.MarketCap, reverse=True)[:100]
        symbols = [f.Symbol for f in top]
        self.last_selected = symbols
        return symbols

    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            self.active[symbol] = security
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.Portfolio:
                self.Liquidate(symbol)
            self.active.pop(symbol, None)
            self.symbol_scores.pop(symbol, None)

    def OnData(self, data: Slice) -> None:
        if self.Time < self.next_rebalance:
            return

        self.next_rebalance = self.Time + self.rebalance_period

        scores: Dict[Symbol, float] = {}
        history_period = 252
        for symbol in list(self.active.keys()):
        history = self.History(symbol, history_period, Resolution.Daily)
        if history.empty:
            continue
        close = self._extract_close_series(history)
        close = close.dropna()
            if len(close) < 60:
                continue
            momentum = close.iloc[-1] / close.iloc[-63] - 1  # 3-month momentum
            daily_returns = close.pct_change().dropna()
            if daily_returns.empty:
                continue
            volatility = daily_returns.std() * np.sqrt(252)
            score = momentum - 0.5 * volatility
            scores[symbol] = score

        if not scores:
            return

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_symbols = [symbol for symbol, _ in ranked[:20]]
        weight = 1.0 / len(top_symbols)

        invested = set()
        for symbol in top_symbols:
            if symbol in data and data[symbol]:
                self.SetHoldings(symbol, weight)
                invested.add(symbol)

        for symbol in list(self.Portfolio.Keys):
            if symbol not in invested and self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)

        self.symbol_scores = scores

    @staticmethod
    def _extract_close_series(history: pd.DataFrame) -> pd.Series:
        if isinstance(history, pd.Series):
            return history
        if isinstance(history.columns, pd.MultiIndex):
            return history.xs("close", level=1, axis=1).iloc[:, 0]
        if "close" in history.columns:
            return history["close"]
        return history.squeeze()


# Required algorithm reference for Lean CLI
class MonthlyUniverseAlphaAlgorithm(MonthlyUniverseAlpha):
    pass
