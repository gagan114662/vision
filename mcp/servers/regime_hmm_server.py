"""Hidden Markov Model regime detection MCP tool."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from mcp.server import register_tool

try:
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
except ImportError:
    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator


logger = logging.getLogger(__name__)


def _ensure_hmm():
    """Return hmmlearn GaussianHMM or a deterministic fallback implementation."""
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
        logger.info("Using hmmlearn GaussianHMM implementation")
        return GaussianHMM
    except ImportError:
        logger.warning("hmmlearn not available; using deterministic fallback for regime detection")

        class GaussianHMMFallback:
            def __init__(self, n_components: int, covariance_type: str = "full", n_iter: int = 100, **_: Any) -> None:
                self.n_components = n_components
                self.covariance_type = covariance_type
                self.n_iter = n_iter
                self._state_means: np.ndarray | None = None
                self._transitions: np.ndarray | None = None

            def fit(self, features: np.ndarray) -> "GaussianHMMFallback":
                if features.size == 0:
                    raise ValueError("No features provided")
                primary = features[:, 0]
                percentiles = np.linspace(0, 100, self.n_components + 1)
                bins = np.percentile(primary, percentiles)
                bins[0] = -np.inf
                bins[-1] = np.inf
                states = np.digitize(primary, bins[1:-1], right=True)
                self._state_means = np.vstack([
                    features[states == idx].mean(axis=0) if np.any(states == idx) else np.zeros(features.shape[1])
                    for idx in range(self.n_components)
                ])

                transitions = np.zeros((self.n_components, self.n_components), dtype=float)
                for prev, nxt in zip(states[:-1], states[1:]):
                    transitions[prev, nxt] += 1
                transitions += 1  # Laplace smoothing
                transitions /= transitions.sum(axis=1, keepdims=True)
                self._transitions = transitions
                self._last_states = states
                return self

            def predict(self, features: np.ndarray) -> np.ndarray:
                if self._state_means is None:
                    raise RuntimeError("Model must be fitted before prediction")
                dists = np.linalg.norm(features[:, None, :] - self._state_means[None, :, :], axis=2)
                return np.argmin(dists, axis=1)

            def score(self, features: np.ndarray) -> float:
                if self._state_means is None:
                    raise RuntimeError("Model must be fitted before scoring")
                preds = self.predict(features)
                errors = features - self._state_means[preds]
                return -float(np.mean(np.sum(errors ** 2, axis=1)))

            @property
            def transmat_(self) -> np.ndarray:
                if self._transitions is None:
                    raise RuntimeError("Model must be fitted before accessing transmat_")
                return self._transitions

            @property
            def means_(self) -> np.ndarray:
                if self._state_means is None:
                    raise RuntimeError("Model must be fitted before accessing means_")
                return self._state_means

        return GaussianHMMFallback


GaussianHMM = _ensure_hmm()


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
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def detect_regimes(params: Dict[str, Any]) -> Dict[str, Any]:
    prices = params["prices"]
    volumes = params.get("volumes")
    n_regimes = params.get("n_regimes", 3)
    covariance_type = params.get("covariance_type", "full")

    features = _validate_inputs(prices, volumes)
    if len(features) <= n_regimes:
        raise ValueError("Insufficient data length for requested number of regimes")

    # Use the autonomous HMM class (real or fallback)
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
