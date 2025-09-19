"""Robustness evaluation utilities for strategy returns."""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class RobustnessResult:
    sharpe: float
    cross_validation: List[Dict[str, float]]
    walk_forward: List[Dict[str, float]]
    bootstrap: Dict[str, float]
    benchmark_correlation: float | None
    buy_and_hold_similarity: float | None
    verdict: str


def _compute_sharpe(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    mean = statistics.mean(returns)
    stdev = statistics.pstdev(returns) or 1e-9
    return (mean / stdev) * math.sqrt(252)


def _chunk_sequence(seq: Sequence[float], folds: int) -> List[Sequence[float]]:
    chunk_size = max(1, len(seq) // folds)
    chunks: List[Sequence[float]] = []
    for i in range(folds):
        start = i * chunk_size
        end = len(seq) if i == folds - 1 else (i + 1) * chunk_size
        if start >= len(seq):
            break
        chunk = seq[start:end]
        if chunk:
            chunks.append(chunk)
    return chunks


def _walk_forward(seq: Sequence[float], window: int) -> List[Sequence[float]]:
    window = max(window, 10)
    segments: List[Sequence[float]] = []
    for start in range(0, len(seq) - window + 1, window // 2 or 1):
        segment = seq[start : start + window]
        if len(segment) == window:
            segments.append(segment)
    return segments


def _bootstrap(seq: Sequence[float], iterations: int) -> Tuple[float, float]:
    if not seq:
        return 0.0, 0.0
    means: List[float] = []
    for _ in range(iterations):
        sample = [random.choice(seq) for _ in range(len(seq))]
        means.append(statistics.mean(sample))
    means.sort()
    lower = means[int(0.025 * (iterations - 1))]
    upper = means[int(0.975 * (iterations - 1))]
    return lower, upper


def _correlation(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("nan")
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    denom = math.sqrt(var_a * var_b)
    if denom == 0:
        return float("nan")
    return cov / denom


def evaluate_returns(
    returns: Sequence[float],
    folds: int = 3,
    walk_forward_window: int = 60,
    bootstrap_iterations: int = 100,
    benchmark_returns: Sequence[float] | None = None,
) -> RobustnessResult:
    returns = list(returns)
    if len(returns) < 60:
        raise ValueError("At least 60 return observations required for robustness evaluation")

    sharpe = _compute_sharpe(returns)

    cross_validation = []
    for idx, chunk in enumerate(_chunk_sequence(returns, folds), start=1):
        cross_validation.append({"fold": float(idx), "sharpe": _compute_sharpe(chunk)})

    walk_forward = []
    for idx, segment in enumerate(_walk_forward(returns, walk_forward_window), start=1):
        walk_forward.append(
            {
                "window": float(idx),
                "cumulative_return": float(math.prod(1 + r for r in segment) - 1),
                "sharpe": _compute_sharpe(segment),
            }
        )

    lower_ci, upper_ci = _bootstrap(returns, bootstrap_iterations)
    bootstrap_summary = {
        "mean": float(statistics.mean(returns)),
        "lower_95": float(lower_ci),
        "upper_95": float(upper_ci),
    }

    benchmark_corr = None
    buy_and_hold_similarity = None
    if benchmark_returns:
        benchmark_returns = list(benchmark_returns)
        min_len = min(len(returns), len(benchmark_returns))
        if min_len >= 60:
            strat = returns[-min_len:]
            bench = benchmark_returns[-min_len:]
            corr = _correlation(strat, bench)
            if not math.isnan(corr):
                benchmark_corr = float(corr)
            cum_strat = math.prod(1 + r for r in strat) - 1
            cum_bench = math.prod(1 + r for r in bench) - 1
            if cum_bench != 0:
                buy_and_hold_similarity = float(cum_strat / cum_bench)

    verdict = "pass"
    if abs(sharpe) < 0.5:
        verdict = "concerns"
    if any(result["sharpe"] < 0 for result in cross_validation):
        verdict = "concerns"
    if benchmark_corr is not None and abs(benchmark_corr) > 0.85:
        verdict = "fail"
    if buy_and_hold_similarity is not None and 0.8 < buy_and_hold_similarity < 1.2:
        verdict = "fail"

    return RobustnessResult(
        sharpe=float(sharpe),
        cross_validation=cross_validation,
        walk_forward=walk_forward,
        bootstrap=bootstrap_summary,
        benchmark_correlation=benchmark_corr,
        buy_and_hold_similarity=buy_and_hold_similarity,
        verdict=verdict,
    )


def to_dict(result: RobustnessResult) -> Dict[str, Any]:  # type: ignore[name-defined]
    return {
        "sharpe": result.sharpe,
        "cross_validation": result.cross_validation,
        "walk_forward": result.walk_forward,
        "bootstrap": result.bootstrap,
        "benchmark_correlation": result.benchmark_correlation,
        "buy_and_hold_similarity": result.buy_and_hold_similarity,
        "verdict": result.verdict,
    }
