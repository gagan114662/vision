"""
Advanced portfolio optimization and risk management modules.

Implements Hierarchical Risk Parity (HRP), Black-Litterman enhancement,
and other sophisticated portfolio construction techniques.
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
    "AgentViewAggregator"
]