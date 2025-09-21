import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class CovarianceEstimator(Enum):
    EMPIRICAL = "empirical"
    LEDOIT_WOLF = "ledoit_wolf"
    OAS = "oas"  # Oracle Approximating Shrinkage
    ROBUST = "robust"

class ClusteringMethod(Enum):
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"

@dataclass
class HRPConfig:
    """Configuration for Hierarchical Risk Parity optimization"""
    covariance_estimator: CovarianceEstimator = CovarianceEstimator.LEDOIT_WOLF
    clustering_method: ClusteringMethod = ClusteringMethod.SINGLE
    min_cluster_size: int = 2
    max_clusters: Optional[int] = None
    risk_free_rate: float = 0.02
    lookback_days: int = 252
    min_weight: float = 0.001
    max_weight: float = 0.3
    rebalance_frequency: str = "monthly"

@dataclass
class FactorData:
    """Real factor data for enhanced HRP"""
    returns: pd.DataFrame
    factors: pd.DataFrame  # Factor loadings/exposures
    factor_names: List[str]
    risk_factors: Optional[pd.DataFrame] = None  # Risk factor returns
    style_factors: Optional[pd.DataFrame] = None  # Style factor exposures

class HierarchicalRiskParity:
    """
    Enhanced Hierarchical Risk Parity implementation with real factor data integration
    """

    def __init__(self, config: HRPConfig = None):
        self.config = config or HRPConfig()
        self._correlation_matrix = None
        self._covariance_matrix = None
        self._linkage_matrix = None
        self._cluster_labels = None

    def fit(self, returns: pd.DataFrame, factor_data: Optional[FactorData] = None) -> 'HierarchicalRiskParity':
        """
        Fit the HRP model to return data with optional factor enhancement

        Args:
            returns: DataFrame with asset returns (assets as columns, dates as rows)
            factor_data: Optional factor data for enhanced risk modeling
        """
        logger.info(f"Fitting HRP model with {len(returns.columns)} assets, {len(returns)} observations")

        # Validate input data
        self._validate_returns(returns)

        # Enhanced covariance estimation with factor data
        if factor_data is not None:
            self._covariance_matrix = self._estimate_factor_enhanced_covariance(returns, factor_data)
        else:
            self._covariance_matrix = self._estimate_covariance(returns)

        # Calculate correlation matrix
        self._correlation_matrix = self._cov_to_corr(self._covariance_matrix)

        # Perform hierarchical clustering
        self._linkage_matrix = self._cluster_assets()

        # Generate cluster labels
        self._cluster_labels = self._get_cluster_labels()

        logger.info(f"HRP model fitted successfully with {len(np.unique(self._cluster_labels))} clusters")
        return self

    def optimize(self, returns: pd.DataFrame, factor_data: Optional[FactorData] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights using Hierarchical Risk Parity

        Returns:
            Dictionary mapping asset symbols to weights
        """
        if self._covariance_matrix is None:
            self.fit(returns, factor_data)

        logger.info("Starting HRP optimization")

        # Step 1: Quasi-diagonalization
        sorted_assets = self._quasi_diagonalize()
        logger.debug(f"Asset ordering after quasi-diagonalization: {sorted_assets[:10]}...")

        # Step 2: Recursive bisection
        weights = self._recursive_bisection(sorted_assets)

        # Step 3: Apply constraints
        weights = self._apply_constraints(weights)

        # Step 4: Factor-based adjustments if available
        if factor_data is not None:
            weights = self._apply_factor_adjustments(weights, returns, factor_data)

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {asset: weight / total_weight for asset, weight in weights.items()}

        logger.info(f"HRP optimization completed. Portfolio weights: min={min(weights.values()):.4f}, max={max(weights.values()):.4f}")

        return weights

    def _validate_returns(self, returns: pd.DataFrame):
        """Validate input return data"""
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")

        if returns.isnull().any().any():
            logger.warning("Missing values detected in returns data. Forward filling...")
            returns.fillna(method='ffill', inplace=True)

        if len(returns) < self.config.lookback_days:
            logger.warning(f"Only {len(returns)} observations available, less than configured lookback of {self.config.lookback_days}")

        # Check for constant columns
        constant_cols = returns.columns[returns.var() == 0].tolist()
        if constant_cols:
            logger.warning(f"Constant return columns detected: {constant_cols}")

    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance matrix using configured estimator"""
        logger.debug(f"Estimating covariance using {self.config.covariance_estimator.value} estimator")

        if self.config.covariance_estimator == CovarianceEstimator.EMPIRICAL:
            estimator = EmpiricalCovariance()
        elif self.config.covariance_estimator == CovarianceEstimator.LEDOIT_WOLF:
            estimator = LedoitWolf()
        elif self.config.covariance_estimator == CovarianceEstimator.OAS:
            estimator = OAS()
        else:
            # Robust estimator fallback
            estimator = LedoitWolf()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cov_matrix = estimator.fit(returns.values).covariance_

        # Ensure positive definite
        cov_matrix = self._make_positive_definite(cov_matrix)

        return cov_matrix

    def _estimate_factor_enhanced_covariance(self, returns: pd.DataFrame, factor_data: FactorData) -> np.ndarray:
        """
        Estimate covariance matrix enhanced with factor model
        Uses the factor model: Cov = B * F * B' + D
        where B is factor loadings, F is factor covariance, D is specific risk
        """
        logger.debug("Estimating factor-enhanced covariance matrix")

        # Align data
        common_dates = returns.index.intersection(factor_data.returns.index)
        if len(common_dates) < len(returns) * 0.8:
            logger.warning("Limited overlap between returns and factor data")

        aligned_returns = returns.loc[common_dates]
        aligned_factors = factor_data.returns.loc[common_dates]

        # Estimate factor loadings via regression
        factor_loadings = self._estimate_factor_loadings(aligned_returns, aligned_factors)

        # Estimate factor covariance
        factor_cov = self._estimate_covariance(aligned_factors)

        # Estimate specific risk (residual variance)
        specific_risk = self._estimate_specific_risk(aligned_returns, aligned_factors, factor_loadings)

        # Construct factor model covariance
        # Cov = B * F * B' + D
        factor_component = factor_loadings @ factor_cov @ factor_loadings.T
        factor_enhanced_cov = factor_component + np.diag(specific_risk)

        # Blend with sample covariance for robustness
        sample_cov = self._estimate_covariance(aligned_returns)
        blend_weight = 0.7  # Weight on factor model

        final_cov = blend_weight * factor_enhanced_cov + (1 - blend_weight) * sample_cov

        return self._make_positive_definite(final_cov)

    def _estimate_factor_loadings(self, returns: pd.DataFrame, factors: pd.DataFrame) -> np.ndarray:
        """Estimate factor loadings using multivariate regression"""
        n_assets = len(returns.columns)
        n_factors = len(factors.columns)
        loadings = np.zeros((n_assets, n_factors))

        X = factors.values
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept

        for i, asset in enumerate(returns.columns):
            y = returns[asset].values
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                loadings[i, :] = beta[1:]  # Exclude intercept
            except np.linalg.LinAlgError:
                logger.warning(f"Failed to estimate loadings for {asset}, using zeros")
                loadings[i, :] = 0

        return loadings

    def _estimate_specific_risk(self, returns: pd.DataFrame, factors: pd.DataFrame, loadings: np.ndarray) -> np.ndarray:
        """Estimate asset-specific risk (residual variance)"""
        n_assets = len(returns.columns)
        specific_variance = np.zeros(n_assets)

        X = factors.values
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept

        for i, asset in enumerate(returns.columns):
            y = returns[asset].values
            try:
                # Predict returns using factor model
                beta = np.concatenate([[0], loadings[i, :]])  # Add zero intercept
                y_pred = X @ beta
                residuals = y - y_pred
                specific_variance[i] = np.var(residuals, ddof=1)
            except:
                # Fallback to total variance
                specific_variance[i] = np.var(returns[asset].values, ddof=1)

        # Ensure minimum specific risk
        min_specific_risk = 1e-6
        specific_variance = np.maximum(specific_variance, min_specific_risk)

        return specific_variance

    def _make_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite"""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)

        # Set minimum eigenvalue
        min_eigenval = 1e-8
        eigenvals = np.maximum(eigenvals, min_eigenval)

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def _cov_to_corr(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix"""
        diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(cov_matrix)))
        return diag_inv_sqrt @ cov_matrix @ diag_inv_sqrt

    def _cluster_assets(self) -> np.ndarray:
        """Perform hierarchical clustering on assets"""
        # Convert correlation to distance
        distance_matrix = np.sqrt(0.5 * (1 - self._correlation_matrix))

        # Ensure distance matrix is valid
        distance_matrix = np.clip(distance_matrix, 0, np.inf)

        # Convert to condensed distance matrix
        distance_vector = squareform(distance_matrix, checks=False)

        # Perform linkage
        linkage_matrix = linkage(
            distance_vector,
            method=self.config.clustering_method.value,
            optimal_ordering=True
        )

        return linkage_matrix

    def _get_cluster_labels(self) -> np.ndarray:
        """Get cluster labels from linkage matrix"""
        n_assets = len(self._correlation_matrix)

        if self.config.max_clusters is not None:
            n_clusters = min(self.config.max_clusters, n_assets)
        else:
            # Automatic cluster selection based on dendrogram
            n_clusters = max(2, n_assets // 4)  # Heuristic: ~4 assets per cluster

        cluster_labels = fcluster(
            self._linkage_matrix,
            n_clusters,
            criterion='maxclust'
        )

        return cluster_labels - 1  # Convert to 0-based indexing

    def _quasi_diagonalize(self) -> List[int]:
        """
        Quasi-diagonalize the correlation matrix using hierarchical clustering
        Returns the sorted asset indices
        """
        def _get_cluster_order(linkage_matrix: np.ndarray, n_assets: int) -> List[int]:
            """Get asset order from hierarchical clustering"""
            # Simple implementation - can be enhanced with more sophisticated ordering
            cluster_order = []

            # Use dendrogram ordering
            from scipy.cluster.hierarchy import dendrogram
            dendro = dendrogram(linkage_matrix, no_plot=True)
            cluster_order = dendro['leaves']

            return cluster_order

        sorted_indices = _get_cluster_order(self._linkage_matrix, len(self._correlation_matrix))

        logger.debug(f"Asset ordering from quasi-diagonalization: {sorted_indices}")
        return sorted_indices

    def _recursive_bisection(self, sorted_assets: List[int]) -> Dict[int, float]:
        """
        Perform recursive bisection to allocate weights
        """
        def _calculate_cluster_variance(asset_indices: List[int]) -> float:
            """Calculate the variance of an asset cluster"""
            if len(asset_indices) == 1:
                return self._covariance_matrix[asset_indices[0], asset_indices[0]]

            cluster_cov = self._covariance_matrix[np.ix_(asset_indices, asset_indices)]
            # Equal-weighted cluster variance
            ones = np.ones((len(asset_indices), 1)) / len(asset_indices)
            return float(ones.T @ cluster_cov @ ones)

        def _bisect(asset_indices: List[int], weight: float) -> Dict[int, float]:
            """Recursively bisect clusters and allocate weights"""
            if len(asset_indices) == 1:
                return {asset_indices[0]: weight}

            # Find the optimal split point
            n = len(asset_indices)
            best_split = n // 2
            min_variance_diff = float('inf')

            # Try different split points
            for split in range(1, n):
                left_cluster = asset_indices[:split]
                right_cluster = asset_indices[split:]

                left_var = _calculate_cluster_variance(left_cluster)
                right_var = _calculate_cluster_variance(right_cluster)
                variance_diff = abs(left_var - right_var)

                if variance_diff < min_variance_diff:
                    min_variance_diff = variance_diff
                    best_split = split

            # Split at optimal point
            left_cluster = asset_indices[:best_split]
            right_cluster = asset_indices[best_split:]

            # Calculate inverse variance weights
            left_var = _calculate_cluster_variance(left_cluster)
            right_var = _calculate_cluster_variance(right_cluster)

            # Avoid division by zero
            left_var = max(left_var, 1e-8)
            right_var = max(right_var, 1e-8)

            inv_left = 1.0 / left_var
            inv_right = 1.0 / right_var
            total_inv = inv_left + inv_right

            left_weight = weight * (inv_left / total_inv)
            right_weight = weight * (inv_right / total_inv)

            # Recursively allocate weights
            weights = {}
            weights.update(_bisect(left_cluster, left_weight))
            weights.update(_bisect(right_cluster, right_weight))

            return weights

        return _bisect(sorted_assets, 1.0)

    def _apply_constraints(self, weights: Dict[int, float]) -> Dict[int, float]:
        """Apply weight constraints (min/max weights)"""
        # Apply minimum weight constraint
        for asset in weights:
            if weights[asset] < self.config.min_weight:
                weights[asset] = self.config.min_weight

        # Apply maximum weight constraint
        for asset in weights:
            if weights[asset] > self.config.max_weight:
                weights[asset] = self.config.max_weight

        # Renormalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {asset: weight / total_weight for asset, weight in weights.items()}

        return weights

    def _apply_factor_adjustments(self, weights: Dict[int, float], returns: pd.DataFrame, factor_data: FactorData) -> Dict[int, float]:
        """Apply factor-based adjustments to weights"""
        logger.debug("Applying factor-based weight adjustments")

        if factor_data.style_factors is not None:
            # Adjust for style factor exposures
            weights = self._adjust_for_style_factors(weights, returns, factor_data)

        if factor_data.risk_factors is not None:
            # Adjust for risk factor exposures
            weights = self._adjust_for_risk_factors(weights, returns, factor_data)

        return weights

    def _adjust_for_style_factors(self, weights: Dict[int, float], returns: pd.DataFrame, factor_data: FactorData) -> Dict[int, float]:
        """Adjust weights based on style factor exposures"""
        # Simple implementation - can be enhanced with more sophisticated factor models
        adjustment_factor = 0.1  # Conservative adjustment

        asset_names = returns.columns.tolist()

        for i, asset in enumerate(asset_names):
            if asset in factor_data.style_factors.index:
                # Example: Reduce weight for high-momentum stocks (assuming momentum factor exists)
                if 'momentum' in factor_data.style_factors.columns:
                    momentum_exposure = factor_data.style_factors.loc[asset, 'momentum']
                    if momentum_exposure > 1.0:  # High momentum
                        weights[i] *= (1 - adjustment_factor * (momentum_exposure - 1))

        return weights

    def _adjust_for_risk_factors(self, weights: Dict[int, float], returns: pd.DataFrame, factor_data: FactorData) -> Dict[int, float]:
        """Adjust weights based on risk factor exposures"""
        # Implement risk factor adjustments
        # This is a placeholder for more sophisticated risk factor modeling
        return weights

    def get_cluster_analysis(self) -> Dict[str, Any]:
        """Get detailed cluster analysis"""
        if self._cluster_labels is None:
            raise ValueError("Model must be fitted before getting cluster analysis")

        unique_clusters = np.unique(self._cluster_labels)
        cluster_info = {}

        for cluster_id in unique_clusters:
            cluster_assets = np.where(self._cluster_labels == cluster_id)[0]
            cluster_corr = self._correlation_matrix[np.ix_(cluster_assets, cluster_assets)]

            cluster_info[f"cluster_{cluster_id}"] = {
                "assets": cluster_assets.tolist(),
                "size": len(cluster_assets),
                "avg_correlation": np.mean(cluster_corr[np.triu_indices_from(cluster_corr, k=1)]),
                "min_correlation": np.min(cluster_corr[np.triu_indices_from(cluster_corr, k=1)]),
                "max_correlation": np.max(cluster_corr[np.triu_indices_from(cluster_corr, k=1)])
            }

        return {
            "n_clusters": len(unique_clusters),
            "cluster_details": cluster_info,
            "silhouette_score": self._calculate_silhouette_score()
        }

    def _calculate_silhouette_score(self) -> float:
        """Calculate silhouette score for clustering quality"""
        try:
            from sklearn.metrics import silhouette_score
            distance_matrix = np.sqrt(0.5 * (1 - self._correlation_matrix))
            return silhouette_score(distance_matrix, self._cluster_labels, metric='precomputed')
        except ImportError:
            logger.warning("scikit-learn not available for silhouette score calculation")
            return 0.0

    def get_risk_contributions(self, weights: Dict[int, float]) -> Dict[str, float]:
        """Calculate risk contributions for each asset"""
        weight_array = np.array([weights.get(i, 0) for i in range(len(self._covariance_matrix))])
        portfolio_variance = weight_array.T @ self._covariance_matrix @ weight_array

        # Marginal risk contributions
        marginal_contrib = self._covariance_matrix @ weight_array
        risk_contrib = weight_array * marginal_contrib / portfolio_variance if portfolio_variance > 0 else weight_array * 0

        return {f"asset_{i}": float(contrib) for i, contrib in enumerate(risk_contrib)}

def create_factor_data_from_real_sources(
    asset_returns: pd.DataFrame,
    factor_returns: Optional[pd.DataFrame] = None,
    style_exposures: Optional[pd.DataFrame] = None
) -> FactorData:
    """
    Create FactorData from real data sources

    Args:
        asset_returns: Asset return data
        factor_returns: Factor return data (e.g., Fama-French factors)
        style_exposures: Style factor exposures (e.g., from Barra, MSCI)
    """
    # Default factor construction if not provided
    if factor_returns is None:
        # Create simple market factor
        market_factor = asset_returns.mean(axis=1)
        factor_returns = pd.DataFrame({'market': market_factor}, index=asset_returns.index)

    factor_names = factor_returns.columns.tolist()

    return FactorData(
        returns=asset_returns,
        factors=factor_returns,
        factor_names=factor_names,
        risk_factors=factor_returns,
        style_factors=style_exposures
    )