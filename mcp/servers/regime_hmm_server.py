"""Hidden Markov Model regime detection MCP tool."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:  # pragma: no cover - executed when hmmlearn is available
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when hmmlearn missing
    _HMM_AVAILABLE = False
    GaussianHMM = None  # type: ignore

try:
    from mcp.server import register_tool
except ImportError:  # pragma: no cover
    def register_tool(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func

        return decorator


def _ensure_hmm() -> None:
    if not _HMM_AVAILABLE:
        raise RuntimeError(
            "hmmlearn is required for strategy.regime.detect_states. Install with 'pip install hmmlearn'."
        )


def _validate_inputs(prices: List[float], volumes: List[float] | None) -> np.ndarray:
    if len(prices) < 10:
        raise ValueError("At least 10 price observations required for regime detection")
    prices_arr = np.asarray(prices, dtype=float)
    if np.any(~np.isfinite(prices_arr)):
        raise ValueError("Prices contain non-finite values")

    # compute log returns
    returns = np.diff(np.log(prices_arr))

    feature_columns = [returns.reshape(-1, 1)]

    if volumes is not None:
        if len(volumes) != len(prices):
            raise ValueError("Volumes must align with prices")
        vol_arr = np.asarray(volumes[1:], dtype=float)
        if np.any(~np.isfinite(vol_arr)):
            raise ValueError("Volumes contain non-finite values")
        vol_z = (vol_arr - np.mean(vol_arr)) / (np.std(vol_arr) + 1e-9)
        feature_columns.append(vol_z.reshape(-1, 1))

    # realized volatility using absolute returns as simple proxy
    abs_returns = np.abs(returns)
    feature_columns.append(abs_returns.reshape(-1, 1))

    features = np.hstack(feature_columns)
    return features


@register_tool(
    name="strategy.regime.detect_states",
    schema="./schemas/tool.strategy.regime.detect_states.schema.json",
)
def detect_regimes(params: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_hmm()

    prices = params["prices"]
    volumes = params.get("volumes")
    n_regimes = params.get("n_regimes", 3)
    covariance_type = params.get("covariance_type", "full")

    features = _validate_inputs(prices, volumes)
    if len(features) <= n_regimes:
        raise ValueError("Insufficient data length for requested number of regimes")

    model = GaussianHMM(n_components=n_regimes, covariance_type=covariance_type, n_iter=200)
    model.fit(features)

    states = model.predict(features)
    transition = model.transmat_.tolist()
    means = model.means_.tolist()
    log_likelihood = float(model.score(features))

    return {
        "states": states.tolist(),
        "transition_matrix": transition,
        "state_means": means,
        "log_likelihood": log_likelihood,
    }


__all__ = ["detect_regimes"]
