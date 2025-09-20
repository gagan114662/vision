"""
Hierarchical Risk Parity (HRP) portfolio optimization implementation.

Based on Marcos López de Prado's research, this module implements
advanced portfolio optimization without matrix inversion, providing
superior out-of-sample performance and stability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import minimize
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


@dataclass
class HRPParameters:
    """Hierarchical Risk Parity optimization parameters."""
    distance_metric: str = "correlation"  # correlation, mutual_info, variation_info
    linkage_method: str = "ward"          # ward, single, complete, average
    min_weight: float = 0.001             # Minimum position weight
    max_weight: float = 0.5               # Maximum position weight
    risk_tolerance: float = 1.0           # Risk scaling factor
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    lookback_period: int = 252            # Days of historical data
    stability_threshold: float = 0.1      # Maximum weight change for stability


@dataclass
class ClusteringResults:
    """Results from hierarchical clustering analysis."""
    distance_matrix: np.ndarray
    linkage_matrix: np.ndarray
    cluster_labels: np.ndarray
    dendrogram_order: List[int]
    clustered_series: pd.DataFrame
    cluster_variance: Dict[int, float]


@dataclass
class PortfolioAllocation:
    """Portfolio allocation results from HRP optimization."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    cluster_contributions: Dict[int, float]
    optimization_date: datetime
    allocation_stability: float
    parameters: HRPParameters


class DistanceCalculator:
    """Advanced distance metrics for asset correlation analysis."""

    @staticmethod
    def correlation_distance(returns: pd.DataFrame) -> np.ndarray:
        """Calculate correlation-based distance matrix."""
        correlation_matrix = returns.corr()
        # Convert correlation to distance: d = sqrt(0.5 * (1 - correlation))
        distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
        return distance_matrix.values

    @staticmethod
    def mutual_information_distance(returns: pd.DataFrame, bins: int = 10) -> np.ndarray:
        """Calculate mutual information-based distance matrix."""
        n_assets = len(returns.columns)
        distance_matrix = np.zeros((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(i, n_assets):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    # Calculate mutual information
                    mi = DistanceCalculator._mutual_information(
                        returns.iloc[:, i], returns.iloc[:, j], bins
                    )
                    # Convert to distance
                    distance_matrix[i, j] = distance_matrix[j, i] = 1.0 - mi

        return distance_matrix

    @staticmethod
    def _mutual_information(x: pd.Series, y: pd.Series, bins: int) -> float:
        """Calculate mutual information between two series."""
        try:
            # Create joint histogram
            hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

            # Convert to probabilities
            hist_2d = hist_2d / np.sum(hist_2d)

            # Calculate marginal probabilities
            px = np.sum(hist_2d, axis=1)
            py = np.sum(hist_2d, axis=0)

            # Calculate mutual information
            mi = 0.0
            for i in range(bins):
                for j in range(bins):
                    if hist_2d[i, j] > 0:
                        mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (px[i] * py[j]))

            return mi
        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {e}")
            return 0.0

    @staticmethod
    def variation_of_information_distance(returns: pd.DataFrame, bins: int = 10) -> np.ndarray:
        """Calculate variation of information distance matrix."""
        n_assets = len(returns.columns)
        distance_matrix = np.zeros((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(i, n_assets):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    # Calculate variation of information
                    vi = DistanceCalculator._variation_of_information(
                        returns.iloc[:, i], returns.iloc[:, j], bins
                    )
                    distance_matrix[i, j] = distance_matrix[j, i] = vi

        return distance_matrix

    @staticmethod
    def _variation_of_information(x: pd.Series, y: pd.Series, bins: int) -> float:
        """Calculate variation of information between two series."""
        try:
            # Calculate entropy and mutual information
            h_x = DistanceCalculator._entropy(x, bins)
            h_y = DistanceCalculator._entropy(y, bins)
            mi = DistanceCalculator._mutual_information(x, y, bins)

            # Variation of information = H(X) + H(Y) - 2*MI(X,Y)
            vi = h_x + h_y - 2 * mi
            return max(0.0, vi)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Variation of information calculation failed: {e}")
            return 1.0

    @staticmethod
    def _entropy(x: pd.Series, bins: int) -> float:
        """Calculate entropy of a series."""
        try:
            hist, _ = np.histogram(x, bins=bins)
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log(hist + 1e-10))  # Add small value to avoid log(0)
            return entropy
        except Exception as e:
            logger.warning(f"Entropy calculation failed: {e}")
            return 0.0


class HierarchicalClustering:
    """Advanced hierarchical clustering for portfolio construction."""

    def __init__(self, parameters: HRPParameters):
        self.parameters = parameters

    def cluster_assets(self, returns: pd.DataFrame) -> ClusteringResults:
        """Perform hierarchical clustering on asset returns."""
        try:
            # Calculate distance matrix
            if self.parameters.distance_metric == "correlation":
                distance_matrix = DistanceCalculator.correlation_distance(returns)
            elif self.parameters.distance_metric == "mutual_info":
                distance_matrix = DistanceCalculator.mutual_information_distance(returns)
            elif self.parameters.distance_metric == "variation_info":
                distance_matrix = DistanceCalculator.variation_of_information_distance(returns)
            else:
                raise ValueError(f"Unknown distance metric: {self.parameters.distance_metric}")

            # Ensure valid distance matrix
            distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=0.0)
            np.fill_diagonal(distance_matrix, 0.0)

            # Convert to condensed distance matrix for clustering
            condensed_distances = squareform(distance_matrix, checks=False)

            # Perform hierarchical clustering
            linkage_matrix = sch.linkage(condensed_distances, method=self.parameters.linkage_method)

            # Get cluster labels and optimal ordering
            cluster_labels = sch.fcluster(linkage_matrix, t=0.5, criterion='distance')

            # Get dendrogram ordering for quasi-diagonalization
            dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
            dendrogram_order = dendrogram['leaves']

            # Reorder correlation matrix based on clustering
            clustered_series = returns.iloc[:, dendrogram_order]

            # Calculate cluster variance contributions
            cluster_variance = self._calculate_cluster_variance(returns, cluster_labels)

            return ClusteringResults(
                distance_matrix=distance_matrix,
                linkage_matrix=linkage_matrix,
                cluster_labels=cluster_labels,
                dendrogram_order=dendrogram_order,
                clustered_series=clustered_series,
                cluster_variance=cluster_variance
            )

        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}")
            # Fallback to simple ordering
            n_assets = len(returns.columns)
            return ClusteringResults(
                distance_matrix=np.eye(n_assets),
                linkage_matrix=np.zeros((n_assets-1, 4)),
                cluster_labels=np.arange(n_assets),
                dendrogram_order=list(range(n_assets)),
                clustered_series=returns,
                cluster_variance={i: 1.0/n_assets for i in range(n_assets)}
            )

    def _calculate_cluster_variance(self, returns: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[int, float]:
        """Calculate variance contribution of each cluster."""
        cluster_variance = {}
        total_variance = 0.0

        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_assets = returns.columns[cluster_labels == cluster_id]
            cluster_returns = returns[cluster_assets]

            if len(cluster_assets) > 1:
                # Calculate equal-weighted cluster portfolio variance
                equal_weights = np.ones(len(cluster_assets)) / len(cluster_assets)
                cluster_cov = cluster_returns.cov().values
                cluster_var = np.dot(equal_weights, np.dot(cluster_cov, equal_weights))
            else:
                cluster_var = cluster_returns.var().iloc[0]

            cluster_variance[cluster_id] = max(cluster_var, 1e-8)  # Avoid zero variance
            total_variance += cluster_var

        # Normalize to sum to 1
        if total_variance > 0:
            for cluster_id in cluster_variance:
                cluster_variance[cluster_id] /= total_variance

        return cluster_variance


class HRPOptimizer:
    """Hierarchical Risk Parity portfolio optimizer."""

    def __init__(self, parameters: HRPParameters):
        self.parameters = parameters
        self.clustering = HierarchicalClustering(parameters)

    def optimize(self, returns: pd.DataFrame, previous_weights: Optional[Dict[str, float]] = None) -> PortfolioAllocation:
        """Perform HRP optimization on asset returns."""
        try:
            logger.info(f"Starting HRP optimization for {len(returns.columns)} assets")

            # Validate input data
            if len(returns) < self.parameters.lookback_period // 4:
                logger.warning(f"Insufficient data: {len(returns)} < {self.parameters.lookback_period // 4}")

            # Clean data
            returns = self._clean_returns_data(returns)

            # Perform hierarchical clustering
            clustering_results = self.clustering.cluster_assets(returns)

            # Calculate HRP weights
            hrp_weights = self._calculate_hrp_weights(returns, clustering_results)

            # Apply constraints
            constrained_weights = self._apply_constraints(hrp_weights)

            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, constrained_weights)

            # Calculate allocation stability
            stability = self._calculate_allocation_stability(constrained_weights, previous_weights)

            allocation = PortfolioAllocation(
                weights=constrained_weights,
                expected_return=portfolio_metrics['expected_return'],
                expected_volatility=portfolio_metrics['expected_volatility'],
                sharpe_ratio=portfolio_metrics['sharpe_ratio'],
                max_drawdown=portfolio_metrics['max_drawdown'],
                cluster_contributions=clustering_results.cluster_variance,
                optimization_date=datetime.now(),
                allocation_stability=stability,
                parameters=self.parameters
            )

            logger.info(f"HRP optimization completed. Sharpe ratio: {allocation.sharpe_ratio:.3f}")
            return allocation

        except Exception as e:
            logger.error(f"HRP optimization failed: {e}")
            # Return equal weight portfolio as fallback
            n_assets = len(returns.columns)
            equal_weights = {asset: 1.0/n_assets for asset in returns.columns}
            return PortfolioAllocation(
                weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                cluster_contributions={},
                optimization_date=datetime.now(),
                allocation_stability=0.0,
                parameters=self.parameters
            )

    def _clean_returns_data(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate returns data."""
        # Remove assets with insufficient data
        min_observations = max(30, len(returns) // 4)
        valid_assets = []

        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            if len(asset_returns) >= min_observations:
                valid_assets.append(asset)

        if len(valid_assets) < 2:
            raise ValueError("Insufficient assets with valid data")

        cleaned_returns = returns[valid_assets].dropna()

        # Remove extreme outliers (beyond 5 standard deviations)
        for asset in cleaned_returns.columns:
            mean_return = cleaned_returns[asset].mean()
            std_return = cleaned_returns[asset].std()
            threshold = 5 * std_return

            outlier_mask = np.abs(cleaned_returns[asset] - mean_return) > threshold
            cleaned_returns.loc[outlier_mask, asset] = np.nan

        # Forward fill any remaining NaN values
        cleaned_returns = cleaned_returns.fillna(method='ffill').dropna()

        logger.info(f"Cleaned data: {len(cleaned_returns)} observations for {len(cleaned_returns.columns)} assets")
        return cleaned_returns

    def _calculate_hrp_weights(self, returns: pd.DataFrame, clustering_results: ClusteringResults) -> Dict[str, float]:
        """Calculate HRP weights using recursive bisection."""
        covariance_matrix = returns.cov().values
        assets = returns.columns.tolist()

        # Get quasi-diagonal covariance matrix
        ordered_assets = [assets[i] for i in clustering_results.dendrogram_order]
        ordered_indices = [assets.index(asset) for asset in ordered_assets]

        quasi_diag_cov = covariance_matrix[np.ix_(ordered_indices, ordered_indices)]

        # Calculate weights using recursive bisection
        weights_array = self._recursive_bisection(quasi_diag_cov)

        # Map back to original asset order
        weights_dict = {}
        for i, asset in enumerate(ordered_assets):
            weights_dict[asset] = weights_array[i]

        return weights_dict

    def _recursive_bisection(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Perform recursive bisection to calculate HRP weights."""
        n_assets = cov_matrix.shape[0]

        if n_assets == 1:
            return np.array([1.0])

        # Split into two groups
        split_point = n_assets // 2

        # Calculate weights for each group
        left_cov = cov_matrix[:split_point, :split_point]
        right_cov = cov_matrix[split_point:, split_point:]

        left_weights = self._recursive_bisection(left_cov)
        right_weights = self._recursive_bisection(right_cov)

        # Calculate group variances for allocation between groups
        left_var = self._calculate_group_variance(left_cov, left_weights)
        right_var = self._calculate_group_variance(right_cov, right_weights)

        # Allocate based on inverse variance
        total_inv_var = (1.0 / left_var) + (1.0 / right_var)
        left_allocation = (1.0 / left_var) / total_inv_var
        right_allocation = (1.0 / right_var) / total_inv_var

        # Combine weights
        combined_weights = np.zeros(n_assets)
        combined_weights[:split_point] = left_weights * left_allocation
        combined_weights[split_point:] = right_weights * right_allocation

        return combined_weights

    def _calculate_group_variance(self, cov_matrix: np.ndarray, weights: np.ndarray) -> float:
        """Calculate portfolio variance for a group of assets."""
        try:
            return np.dot(weights, np.dot(cov_matrix, weights))
        except Exception:
            return np.trace(cov_matrix) / len(weights)  # Fallback to average variance

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply position size and other constraints to weights."""
        constrained_weights = weights.copy()

        # Apply min/max weight constraints
        for asset, weight in constrained_weights.items():
            constrained_weights[asset] = np.clip(
                weight, self.parameters.min_weight, self.parameters.max_weight
            )

        # Renormalize to sum to 1
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            for asset in constrained_weights:
                constrained_weights[asset] /= total_weight

        return constrained_weights

    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        try:
            # Calculate portfolio returns
            weight_series = pd.Series(weights)
            portfolio_returns = (returns * weight_series).sum(axis=1)

            # Calculate metrics
            expected_return = portfolio_returns.mean() * 252  # Annualized
            expected_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0.0

            # Calculate maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())

            return {
                'expected_return': expected_return,
                'expected_volatility': expected_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }

        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return {
                'expected_return': 0.0,
                'expected_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }

    def _calculate_allocation_stability(
        self,
        current_weights: Dict[str, float],
        previous_weights: Optional[Dict[str, float]]
    ) -> float:
        """Calculate allocation stability compared to previous weights."""
        if previous_weights is None:
            return 1.0

        # Calculate weight changes
        weight_changes = []
        all_assets = set(current_weights.keys()) | set(previous_weights.keys())

        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            previous_weight = previous_weights.get(asset, 0.0)
            weight_changes.append(abs(current_weight - previous_weight))

        # Stability is 1 - (sum of absolute weight changes / 2)
        total_change = sum(weight_changes) / 2  # Divide by 2 since changes sum to twice the turnover
        stability = max(0.0, 1.0 - total_change)

        return stability


class HRPPortfolioManager:
    """Portfolio manager using Hierarchical Risk Parity optimization."""

    def __init__(self, parameters: HRPParameters):
        self.parameters = parameters
        self.optimizer = HRPOptimizer(parameters)
        self.current_allocation: Optional[PortfolioAllocation] = None
        self.allocation_history: List[PortfolioAllocation] = []

    def update_allocation(self, returns: pd.DataFrame) -> PortfolioAllocation:
        """Update portfolio allocation using HRP optimization."""
        previous_weights = self.current_allocation.weights if self.current_allocation else None

        # Perform optimization
        new_allocation = self.optimizer.optimize(returns, previous_weights)

        # Store allocation
        self.current_allocation = new_allocation
        self.allocation_history.append(new_allocation)

        # Keep only recent history
        if len(self.allocation_history) > 100:
            self.allocation_history = self.allocation_history[-100:]

        logger.info(f"Portfolio allocation updated. Stability: {new_allocation.allocation_stability:.3f}")
        return new_allocation

    def get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        if self.current_allocation:
            return self.current_allocation.weights
        return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get portfolio performance summary."""
        if not self.allocation_history:
            return {}

        recent_allocations = self.allocation_history[-10:]  # Last 10 allocations

        return {
            "total_allocations": len(self.allocation_history),
            "current_sharpe_ratio": self.current_allocation.sharpe_ratio if self.current_allocation else 0,
            "average_stability": np.mean([a.allocation_stability for a in recent_allocations]),
            "average_sharpe_ratio": np.mean([a.sharpe_ratio for a in recent_allocations]),
            "current_volatility": self.current_allocation.expected_volatility if self.current_allocation else 0,
            "average_max_drawdown": np.mean([a.max_drawdown for a in recent_allocations]),
            "current_weights": self.get_current_weights(),
            "optimization_parameters": {
                "distance_metric": self.parameters.distance_metric,
                "linkage_method": self.parameters.linkage_method,
                "lookback_period": self.parameters.lookback_period,
                "rebalance_frequency": self.parameters.rebalance_frequency
            }
        }


__all__ = [
    "HRPOptimizer",
    "HRPPortfolioManager",
    "HRPParameters",
    "PortfolioAllocation",
    "ClusteringResults",
    "DistanceCalculator",
    "HierarchicalClustering"
]