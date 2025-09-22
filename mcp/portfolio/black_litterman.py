"""
Black-Litterman portfolio optimization with multi-agent views integration.

Implements the Black-Litterman model enhanced with agent-generated views
from fundamental, technical, sentiment, and quantitative analysis agents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

logger = logging.getLogger(__name__)


class ViewType(Enum):
    """Types of investment views."""
    ABSOLUTE_RETURN = "absolute_return"        # Asset will return X%
    RELATIVE_RETURN = "relative_return"        # Asset A will outperform Asset B by X%
    VOLATILITY = "volatility"                  # Asset volatility expectation
    CORRELATION = "correlation"                # Correlation between assets
    SECTOR_ROTATION = "sector_rotation"        # Sector-level views


class ConfidenceLevel(Enum):
    """Confidence levels for views."""
    VERY_LOW = 0.1
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 0.9


@dataclass
class AgentView:
    """Investment view from an agent with confidence and rationale."""
    view_id: str
    agent_source: str                          # fundamental, technical, sentiment, quantitative
    view_type: ViewType
    assets: List[str]                          # Assets involved in the view
    expected_return: float                     # Expected return (annualized)
    confidence: ConfidenceLevel                # Confidence in the view
    time_horizon: int                          # Days for view realization
    rationale: str                             # Explanation for the view
    created_at: datetime = field(default_factory=datetime.now)
    weight: float = 1.0                        # Relative importance weight
    sector: Optional[str] = None               # Sector if applicable
    factor_exposure: Dict[str, float] = field(default_factory=dict)


@dataclass
class BlackLittermanParameters:
    """Black-Litterman model parameters."""
    risk_aversion: float = 3.0                 # Risk aversion coefficient
    tau: float = 0.05                          # Scaling factor for uncertainty
    max_weight: float = 0.2                    # Maximum position weight
    min_weight: float = 0.001                  # Minimum position weight
    confidence_scaling: float = 1.0            # Scale factor for confidence levels
    view_decay_halflife: int = 30              # Days for view confidence decay
    equilibrium_method: str = "market_cap"     # market_cap, equal_weight, custom
    shrinkage_intensity: float = 0.1           # Covariance shrinkage intensity


@dataclass
class BlackLittermanResults:
    """Results from Black-Litterman optimization."""
    optimal_weights: Dict[str, float]
    expected_returns: Dict[str, float]
    posterior_covariance: np.ndarray
    implied_returns: Dict[str, float]
    view_contributions: Dict[str, float]
    expected_portfolio_return: float
    expected_portfolio_volatility: float
    sharpe_ratio: float
    optimization_date: datetime
    active_views: List[AgentView]
    parameters: BlackLittermanParameters


class EquilibriumCalculator:
    """Calculate equilibrium (prior) expected returns."""

    @staticmethod
    def market_cap_equilibrium(
        market_caps: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        risk_aversion: float = 3.0
    ) -> pd.Series:
        """Calculate equilibrium returns using market capitalization weights."""
        total_market_cap = sum(market_caps.values())
        market_weights = {asset: cap / total_market_cap for asset, cap in market_caps.items()}

        # Convert to pandas Series aligned with covariance matrix
        weight_series = pd.Series(market_weights)[covariance_matrix.index]
        weight_series = weight_series.fillna(0)  # Handle missing assets

        # Equilibrium returns: μ = λ * Σ * w
        equilibrium_returns = risk_aversion * covariance_matrix.dot(weight_series)
        return equilibrium_returns

    @staticmethod
    def equal_weight_equilibrium(
        assets: List[str],
        covariance_matrix: pd.DataFrame,
        risk_aversion: float = 3.0
    ) -> pd.Series:
        """Calculate equilibrium returns using equal weights."""
        n_assets = len(assets)
        equal_weights = pd.Series([1.0 / n_assets] * n_assets, index=assets)

        # Align with covariance matrix
        weight_series = equal_weights[covariance_matrix.index]
        weight_series = weight_series.fillna(0)

        equilibrium_returns = risk_aversion * covariance_matrix.dot(weight_series)
        return equilibrium_returns

    @staticmethod
    def custom_equilibrium(
        custom_weights: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        risk_aversion: float = 3.0
    ) -> pd.Series:
        """Calculate equilibrium returns using custom weights."""
        weight_series = pd.Series(custom_weights)[covariance_matrix.index]
        weight_series = weight_series.fillna(0)

        equilibrium_returns = risk_aversion * covariance_matrix.dot(weight_series)
        return equilibrium_returns


class ViewMatrixBuilder:
    """Build P (picking) and Q (view) matrices from agent views."""

    def __init__(self, assets: List[str]):
        self.assets = assets
        self.asset_to_index = {asset: i for i, asset in enumerate(assets)}

    def build_matrices(self, views: List[AgentView]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build P (picking), Q (view), and Ω (uncertainty) matrices."""
        n_assets = len(self.assets)
        n_views = len(views)

        if n_views == 0:
            # Return empty matrices
            return np.zeros((0, n_assets)), np.zeros(0), np.zeros((0, 0))

        P_matrix = np.zeros((n_views, n_assets))
        Q_vector = np.zeros(n_views)
        Omega_matrix = np.zeros((n_views, n_views))

        for i, view in enumerate(views):
            # Build picking matrix row
            P_row = self._build_picking_row(view)
            P_matrix[i, :] = P_row

            # Set view return
            Q_vector[i] = view.expected_return

            # Set uncertainty (diagonal element of Omega)
            # Higher confidence → lower uncertainty
            base_uncertainty = 0.01  # 1% base uncertainty
            confidence_factor = 1.0 - view.confidence.value
            time_decay = self._calculate_time_decay(view)

            uncertainty = base_uncertainty * (1 + confidence_factor) * time_decay
            Omega_matrix[i, i] = uncertainty ** 2  # Variance

        return P_matrix, Q_vector, Omega_matrix

    def _build_picking_row(self, view: AgentView) -> np.ndarray:
        """Build a single row of the picking matrix for a view."""
        n_assets = len(self.assets)
        picking_row = np.zeros(n_assets)

        if view.view_type == ViewType.ABSOLUTE_RETURN:
            # Single asset absolute return view
            if len(view.assets) == 1 and view.assets[0] in self.asset_to_index:
                asset_idx = self.asset_to_index[view.assets[0]]
                picking_row[asset_idx] = 1.0

        elif view.view_type == ViewType.RELATIVE_RETURN:
            # Relative return view: Asset A - Asset B
            if len(view.assets) >= 2:
                if view.assets[0] in self.asset_to_index:
                    asset_a_idx = self.asset_to_index[view.assets[0]]
                    picking_row[asset_a_idx] = 1.0

                if view.assets[1] in self.asset_to_index:
                    asset_b_idx = self.asset_to_index[view.assets[1]]
                    picking_row[asset_b_idx] = -1.0

        elif view.view_type == ViewType.SECTOR_ROTATION:
            # Sector rotation view: equal weight within sector
            sector_assets = [asset for asset in view.assets if asset in self.asset_to_index]
            if sector_assets:
                weight_per_asset = 1.0 / len(sector_assets)
                for asset in sector_assets:
                    asset_idx = self.asset_to_index[asset]
                    picking_row[asset_idx] = weight_per_asset

        return picking_row

    def _calculate_time_decay(self, view: AgentView) -> float:
        """Calculate time decay factor for view confidence."""
        days_since_creation = (datetime.now() - view.created_at).days
        decay_factor = np.exp(-days_since_creation / 30)  # 30-day half-life
        return max(0.1, decay_factor)  # Minimum 10% confidence retention


class AgentViewAggregator:
    """Aggregate and consolidate views from multiple agents."""

    def __init__(self, parameters: BlackLittermanParameters):
        self.parameters = parameters

    def aggregate_views(self, agent_views: List[AgentView]) -> List[AgentView]:
        """Aggregate similar views from different agents."""
        if not agent_views:
            return []

        # Group views by asset and type
        view_groups = {}
        for view in agent_views:
            # Create grouping key
            assets_key = tuple(sorted(view.assets))
            group_key = (assets_key, view.view_type)

            if group_key not in view_groups:
                view_groups[group_key] = []
            view_groups[group_key].append(view)

        # Aggregate each group
        aggregated_views = []
        for group_key, views in view_groups.items():
            if len(views) == 1:
                aggregated_views.append(views[0])
            else:
                aggregated_view = self._aggregate_view_group(views)
                aggregated_views.append(aggregated_view)

        return aggregated_views

    def _aggregate_view_group(self, views: List[AgentView]) -> AgentView:
        """Aggregate multiple views into a single consensus view."""
        # Calculate weighted average return expectation
        total_weight = sum(view.weight * view.confidence.value for view in views)
        if total_weight == 0:
            total_weight = 1.0

        weighted_return = sum(
            view.expected_return * view.weight * view.confidence.value
            for view in views
        ) / total_weight

        # Calculate consensus confidence
        confidence_values = [view.confidence.value for view in views]
        avg_confidence = np.mean(confidence_values)

        # Map back to ConfidenceLevel enum
        confidence_level = ConfidenceLevel.MEDIUM
        for conf_enum in ConfidenceLevel:
            if abs(conf_enum.value - avg_confidence) < 0.125:  # Within range
                confidence_level = conf_enum
                break

        # Combine rationales
        agent_sources = list(set(view.agent_source for view in views))
        combined_rationale = f"Consensus from {', '.join(agent_sources)}: " + \
                           "; ".join(view.rationale for view in views[:3])  # Limit length

        # Use the most recent view as template
        latest_view = max(views, key=lambda v: v.created_at)

        return AgentView(
            view_id=f"consensus_{latest_view.view_id}",
            agent_source="consensus",
            view_type=latest_view.view_type,
            assets=latest_view.assets,
            expected_return=weighted_return,
            confidence=confidence_level,
            time_horizon=int(np.mean([view.time_horizon for view in views])),
            rationale=combined_rationale,
            created_at=latest_view.created_at,
            weight=np.mean([view.weight for view in views]),
            sector=latest_view.sector
        )


class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimizer with agent views integration."""

    def __init__(self, parameters: BlackLittermanParameters):
        self.parameters = parameters
        self.view_aggregator = AgentViewAggregator(parameters)
        self._agent_integrator = AgentViewIntegrator()

    def optimize(
        self,
        returns: pd.DataFrame,
        agent_views: List[AgentView],
        market_caps: Optional[Dict[str, float]] = None,
        custom_equilibrium: Optional[Dict[str, float]] = None,
        agent_outputs: Optional[Dict[str, Any]] = None
    ) -> BlackLittermanResults:
        """Perform Black-Litterman optimization with agent views."""
        try:
            logger.info(f"Starting Black-Litterman optimization with {len(agent_views)} views")

            # Integrate real-time agent outputs if provided
            if agent_outputs:
                real_time_views = self._agent_integrator.register_agent_views(agent_outputs)
                agent_views.extend(real_time_views)
                logger.info(f"Added {len(real_time_views)} real-time agent views")

            # Prepare data
            assets = returns.columns.tolist()
            covariance_matrix = self._calculate_covariance_matrix(returns)

            # Calculate equilibrium returns
            equilibrium_returns = self._calculate_equilibrium_returns(
                assets, covariance_matrix, market_caps, custom_equilibrium
            )

            # Aggregate agent views
            active_views = self.view_aggregator.aggregate_views(agent_views)
            logger.info(f"Aggregated to {len(active_views)} active views")

            # Build view matrices
            view_builder = ViewMatrixBuilder(assets)
            P_matrix, Q_vector, Omega_matrix = view_builder.build_matrices(active_views)

            # Apply Black-Litterman formula
            posterior_returns, posterior_covariance = self._apply_black_litterman(
                equilibrium_returns, covariance_matrix, P_matrix, Q_vector, Omega_matrix
            )

            # Optimize portfolio
            optimal_weights = self._optimize_portfolio(posterior_returns, posterior_covariance)

            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                optimal_weights, posterior_returns, posterior_covariance
            )

            # Calculate view contributions
            view_contributions = self._calculate_view_contributions(
                active_views, optimal_weights, equilibrium_returns, posterior_returns
            )

            results = BlackLittermanResults(
                optimal_weights=optimal_weights,
                expected_returns=posterior_returns.to_dict(),
                posterior_covariance=posterior_covariance.values,
                implied_returns=equilibrium_returns.to_dict(),
                view_contributions=view_contributions,
                expected_portfolio_return=portfolio_metrics['expected_return'],
                expected_portfolio_volatility=portfolio_metrics['expected_volatility'],
                sharpe_ratio=portfolio_metrics['sharpe_ratio'],
                optimization_date=datetime.now(),
                active_views=active_views,
                parameters=self.parameters
            )

            logger.info(f"Black-Litterman optimization completed. Sharpe ratio: {results.sharpe_ratio:.3f}")
            return results

        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            # Return equal weight portfolio as fallback
            n_assets = len(returns.columns)
            equal_weights = {asset: 1.0/n_assets for asset in returns.columns}

            return BlackLittermanResults(
                optimal_weights=equal_weights,
                expected_returns={},
                posterior_covariance=np.eye(n_assets),
                implied_returns={},
                view_contributions={},
                expected_portfolio_return=0.0,
                expected_portfolio_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_date=datetime.now(),
                active_views=[],
                parameters=self.parameters
            )

    def _calculate_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate shrunk covariance matrix."""
        sample_cov = returns.cov() * 252  # Annualize

        # Apply shrinkage towards identity matrix
        if self.parameters.shrinkage_intensity > 0:
            n_assets = len(sample_cov)
            identity = np.eye(n_assets) * np.trace(sample_cov) / n_assets

            shrunk_cov = (
                (1 - self.parameters.shrinkage_intensity) * sample_cov +
                self.parameters.shrinkage_intensity * identity
            )
            return pd.DataFrame(shrunk_cov, index=sample_cov.index, columns=sample_cov.columns)

        return sample_cov

    def _calculate_equilibrium_returns(
        self,
        assets: List[str],
        covariance_matrix: pd.DataFrame,
        market_caps: Optional[Dict[str, float]],
        custom_equilibrium: Optional[Dict[str, float]]
    ) -> pd.Series:
        """Calculate equilibrium (prior) expected returns."""
        if custom_equilibrium:
            return EquilibriumCalculator.custom_equilibrium(
                custom_equilibrium, covariance_matrix, self.parameters.risk_aversion
            )
        elif market_caps and self.parameters.equilibrium_method == "market_cap":
            return EquilibriumCalculator.market_cap_equilibrium(
                market_caps, covariance_matrix, self.parameters.risk_aversion
            )
        else:
            return EquilibriumCalculator.equal_weight_equilibrium(
                assets, covariance_matrix, self.parameters.risk_aversion
            )

    def _apply_black_litterman(
        self,
        equilibrium_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        P_matrix: np.ndarray,
        Q_vector: np.ndarray,
        Omega_matrix: np.ndarray
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Apply Black-Litterman formula to get posterior distribution."""
        if len(P_matrix) == 0:  # No views
            return equilibrium_returns, covariance_matrix

        # Convert to numpy arrays
        mu_eq = equilibrium_returns.values
        Sigma = covariance_matrix.values

        # Black-Litterman posterior mean:
        # μ_bl = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1 * μ_eq + P'Ω^-1 * Q]

        tau_sigma_inv = linalg.inv(self.parameters.tau * Sigma)
        omega_inv = linalg.inv(Omega_matrix)

        # Calculate the bracketed inverse term
        bracket_inv = linalg.inv(tau_sigma_inv + P_matrix.T @ omega_inv @ P_matrix)

        # Calculate posterior mean
        term1 = tau_sigma_inv @ mu_eq
        term2 = P_matrix.T @ omega_inv @ Q_vector
        mu_bl = bracket_inv @ (term1 + term2)

        # Black-Litterman posterior covariance:
        # Σ_bl = [(τΣ)^-1 + P'Ω^-1P]^-1
        sigma_bl = bracket_inv

        # Convert back to pandas
        posterior_returns = pd.Series(mu_bl, index=equilibrium_returns.index)
        posterior_covariance = pd.DataFrame(
            sigma_bl, index=covariance_matrix.index, columns=covariance_matrix.columns
        )

        return posterior_returns, posterior_covariance

    def _optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Optimize portfolio weights given expected returns and covariance."""
        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()

        # Objective function: maximize utility = μ'w - 0.5 * λ * w'Σw
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns.values)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix.values, weights))
            utility = portfolio_return - 0.5 * self.parameters.risk_aversion * portfolio_variance
            return -utility  # Minimize negative utility

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.parameters.min_weight, self.parameters.max_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )

        if result.success:
            optimal_weights = result.x
        else:
            logger.warning("Optimization failed, using equal weights")
            optimal_weights = np.ones(n_assets) / n_assets

        # Convert to dictionary
        return {asset: weight for asset, weight in zip(assets, optimal_weights)}

    def _calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        weight_array = np.array([weights[asset] for asset in expected_returns.index])

        expected_return = np.dot(weight_array, expected_returns.values)
        expected_variance = np.dot(weight_array, np.dot(covariance_matrix.values, weight_array))
        expected_volatility = np.sqrt(expected_variance)

        sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0.0

        return {
            'expected_return': expected_return,
            'expected_volatility': expected_volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def _calculate_view_contributions(
        self,
        views: List[AgentView],
        optimal_weights: Dict[str, float],
        equilibrium_returns: pd.Series,
        posterior_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate the contribution of each view to the final portfolio."""
        view_contributions = {}

        for view in views:
            # Calculate impact of view on expected returns
            view_impact = 0.0
            for asset in view.assets:
                if asset in equilibrium_returns.index:
                    equilibrium_return = equilibrium_returns[asset]
                    posterior_return = posterior_returns[asset]
                    weight = optimal_weights.get(asset, 0.0)

                    # View contribution = weight * (posterior - equilibrium) return
                    view_impact += weight * (posterior_return - equilibrium_return)

            view_contributions[view.view_id] = view_impact

        return view_contributions


class AgentViewIntegrator:
    """Integration layer for real-time agent views from trading agents."""

    def __init__(self):
        self._cached_views: Dict[str, AgentView] = {}
        self._view_history: List[AgentView] = []

    def register_agent_views(self, agent_outputs: Dict[str, Any]) -> List[AgentView]:
        """Convert agent outputs to standardized AgentView objects."""
        views = []

        # Process fundamental agent views
        if 'fundamental_agent' in agent_outputs:
            fundamental_views = self._process_fundamental_views(agent_outputs['fundamental_agent'])
            views.extend(fundamental_views)

        # Process technical agent views
        if 'technical_agent' in agent_outputs:
            technical_views = self._process_technical_views(agent_outputs['technical_agent'])
            views.extend(technical_views)

        # Process sentiment agent views
        if 'sentiment_agent' in agent_outputs:
            sentiment_views = self._process_sentiment_views(agent_outputs['sentiment_agent'])
            views.extend(sentiment_views)

        # Process quantitative agent views
        if 'quantitative_agent' in agent_outputs:
            quant_views = self._process_quantitative_views(agent_outputs['quantitative_agent'])
            views.extend(quant_views)

        # Cache and track views
        for view in views:
            self._cached_views[view.view_id] = view
            self._view_history.append(view)

        # Prune old views (keep last 1000)
        if len(self._view_history) > 1000:
            self._view_history = self._view_history[-1000:]

        logger.info(f"Registered {len(views)} new agent views")
        return views

    def _process_fundamental_views(self, fundamental_output: Dict[str, Any]) -> List[AgentView]:
        """Process fundamental agent analysis into views."""
        views = []

        if 'stock_analysis' in fundamental_output:
            for symbol, analysis in fundamental_output['stock_analysis'].items():
                # Convert fundamental rating to expected return
                rating = analysis.get('recommendation', 'HOLD')
                expected_return = self._rating_to_return(rating)

                # Convert confidence score to enum
                confidence_score = analysis.get('confidence', 0.5)
                confidence = self._score_to_confidence(confidence_score)

                view = AgentView(
                    view_id=f"fund_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    agent_source="fundamental",
                    view_type=ViewType.ABSOLUTE_RETURN,
                    assets=[symbol],
                    expected_return=expected_return,
                    confidence=confidence,
                    time_horizon=90,  # 3 months for fundamental views
                    rationale=analysis.get('rationale', 'Fundamental analysis'),
                    sector=analysis.get('sector'),
                    factor_exposure=analysis.get('factor_exposure', {})
                )
                views.append(view)

        return views

    def _process_technical_views(self, technical_output: Dict[str, Any]) -> List[AgentView]:
        """Process technical agent signals into views."""
        views = []

        if 'signals' in technical_output:
            for symbol, signal_data in technical_output['signals'].items():
                signal_strength = signal_data.get('signal_strength', 0.0)
                signal_direction = signal_data.get('direction', 'neutral')

                # Convert signal to expected return
                if signal_direction == 'bullish':
                    expected_return = abs(signal_strength) * 0.15  # Scale to reasonable return
                elif signal_direction == 'bearish':
                    expected_return = -abs(signal_strength) * 0.15
                else:
                    expected_return = 0.0

                # Technical signals typically have medium confidence and shorter horizon
                confidence = ConfidenceLevel.MEDIUM
                if abs(signal_strength) > 0.7:
                    confidence = ConfidenceLevel.HIGH
                elif abs(signal_strength) < 0.3:
                    confidence = ConfidenceLevel.LOW

                view = AgentView(
                    view_id=f"tech_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    agent_source="technical",
                    view_type=ViewType.ABSOLUTE_RETURN,
                    assets=[symbol],
                    expected_return=expected_return,
                    confidence=confidence,
                    time_horizon=30,  # 1 month for technical views
                    rationale=signal_data.get('rationale', 'Technical analysis signal'),
                    weight=abs(signal_strength)
                )
                views.append(view)

        return views

    def _process_sentiment_views(self, sentiment_output: Dict[str, Any]) -> List[AgentView]:
        """Process sentiment agent analysis into views."""
        views = []

        if 'sentiment_scores' in sentiment_output:
            for symbol, sentiment_data in sentiment_output['sentiment_scores'].items():
                sentiment_score = sentiment_data.get('composite_score', 0.0)

                # Convert sentiment score to expected return
                # Sentiment scores typically range from -1 to 1
                expected_return = sentiment_score * 0.10  # Scale to 10% max impact

                # Sentiment confidence based on data quality
                data_quality = sentiment_data.get('data_quality', 0.5)
                confidence = self._score_to_confidence(data_quality)

                view = AgentView(
                    view_id=f"sent_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    agent_source="sentiment",
                    view_type=ViewType.ABSOLUTE_RETURN,
                    assets=[symbol],
                    expected_return=expected_return,
                    confidence=confidence,
                    time_horizon=14,  # 2 weeks for sentiment views
                    rationale=sentiment_data.get('summary', 'Market sentiment analysis'),
                    weight=data_quality
                )
                views.append(view)

        return views

    def _process_quantitative_views(self, quant_output: Dict[str, Any]) -> List[AgentView]:
        """Process quantitative agent models into views."""
        views = []

        # Process factor model predictions
        if 'factor_predictions' in quant_output:
            for symbol, prediction in quant_output['factor_predictions'].items():
                expected_return = prediction.get('expected_return', 0.0)
                prediction_confidence = prediction.get('confidence', 0.5)

                confidence = self._score_to_confidence(prediction_confidence)

                view = AgentView(
                    view_id=f"quant_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    agent_source="quantitative",
                    view_type=ViewType.ABSOLUTE_RETURN,
                    assets=[symbol],
                    expected_return=expected_return,
                    confidence=confidence,
                    time_horizon=60,  # 2 months for quant views
                    rationale=prediction.get('model_explanation', 'Quantitative factor model'),
                    factor_exposure=prediction.get('factor_exposures', {})
                )
                views.append(view)

        # Process relative value opportunities
        if 'pairs_trading' in quant_output:
            for pair_id, pair_data in quant_output['pairs_trading'].items():
                asset_a, asset_b = pair_data['assets']
                spread_prediction = pair_data.get('spread_prediction', 0.0)

                view = AgentView(
                    view_id=f"quant_pair_{pair_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    agent_source="quantitative",
                    view_type=ViewType.RELATIVE_RETURN,
                    assets=[asset_a, asset_b],
                    expected_return=spread_prediction,
                    confidence=ConfidenceLevel.MEDIUM,
                    time_horizon=45,
                    rationale=pair_data.get('rationale', 'Statistical arbitrage opportunity')
                )
                views.append(view)

        return views

    def _rating_to_return(self, rating: str) -> float:
        """Convert fundamental rating to expected return."""
        rating_map = {
            'STRONG_BUY': 0.20,
            'BUY': 0.10,
            'HOLD': 0.0,
            'SELL': -0.10,
            'STRONG_SELL': -0.20
        }
        return rating_map.get(rating.upper(), 0.0)

    def _score_to_confidence(self, score: float) -> ConfidenceLevel:
        """Convert numeric confidence score to ConfidenceLevel enum."""
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def get_active_views(self, max_age_hours: int = 24) -> List[AgentView]:
        """Get all active views within the specified age limit."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        active_views = [
            view for view in self._view_history
            if view.created_at >= cutoff_time
        ]

        return active_views

    def get_view_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered views."""
        if not self._view_history:
            return {"total_views": 0}

        agent_counts = {}
        view_type_counts = {}

        for view in self._view_history:
            agent_counts[view.agent_source] = agent_counts.get(view.agent_source, 0) + 1
            view_type_counts[view.view_type.value] = view_type_counts.get(view.view_type.value, 0) + 1

        return {
            "total_views": len(self._view_history),
            "agent_breakdown": agent_counts,
            "view_type_breakdown": view_type_counts,
            "cached_views": len(self._cached_views)
        }


__all__ = [
    "BlackLittermanOptimizer",
    "AgentView",
    "BlackLittermanParameters",
    "BlackLittermanResults",
    "ViewType",
    "ConfidenceLevel",
    "EquilibriumCalculator",
    "ViewMatrixBuilder",
    "AgentViewAggregator",
    "AgentViewIntegrator"
]