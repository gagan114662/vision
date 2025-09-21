"""
Advanced portfolio optimization and risk management modules.

Implements Hierarchical Risk Parity (HRP), Black-Litterman enhancement,
factor models, and other sophisticated portfolio construction techniques.
"""

from .hrp_optimizer import (
    HRPOptimizer,
    HRPPortfolioManager,
    HRPParameters,
    PortfolioAllocation,
    ClusteringResults,
    DistanceCalculator,
    HierarchicalClustering
)

from .black_litterman import (
    BlackLittermanOptimizer,
    AgentView,
    BlackLittermanParameters,
    BlackLittermanResults,
    ViewType,
    ConfidenceLevel,
    EquilibriumCalculator,
    ViewMatrixBuilder,
    AgentViewAggregator
)

from .factor_models import (
    FactorType,
    FactorExposure,
    FactorRiskModel,
    FactorModelBuilder,
    PortfolioOptimizer,
    PortfolioOptimizationResult
)

from .portfolio_engine import (
    RebalanceFrequency,
    PortfolioPosition,
    PortfolioSnapshot,
    RebalanceSignal,
    PortfolioEngine
)

__all__ = [
    # HRP Optimization
    "HRPOptimizer",
    "HRPPortfolioManager",
    "HRPParameters",
    "PortfolioAllocation",
    "ClusteringResults",
    "DistanceCalculator",
    "HierarchicalClustering",

    # Black-Litterman Enhancement
    "BlackLittermanOptimizer",
    "AgentView",
    "BlackLittermanParameters",
    "BlackLittermanResults",
    "ViewType",
    "ConfidenceLevel",
    "EquilibriumCalculator",
    "ViewMatrixBuilder",
    "AgentViewAggregator",

    # Factor Models
    "FactorType",
    "FactorExposure",
    "FactorRiskModel",
    "FactorModelBuilder",
    "PortfolioOptimizer",
    "PortfolioOptimizationResult",

    # Portfolio Engine
    "RebalanceFrequency",
    "PortfolioPosition",
    "PortfolioSnapshot",
    "RebalanceSignal",
    "PortfolioEngine"
]