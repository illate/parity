"""Confidence intervals by bootstrap.

Why paired: model A and model B answer the *same* test items. Some items are hard
for both. Resampling items (not models) keeps that pairing, so the interval on
(B - A) is much tighter than comparing two separate intervals, and honest.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(y == p))


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: Metric = _acc,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, low, high) percentile CI for one model."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric(y[idx], p[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return metric(y, p), float(lo), float(hi)


def paired_bootstrap(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: Metric = _acc,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """CI for metric(B) - metric(A), resampling the same item indices for both."""
    if not (len(y) == len(p_a) == len(p_b)):
        raise ValueError("y, p_a and p_b must have the same length")
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = metric(y[idx], p_b[idx]) - metric(y[idx], p_a[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "diff": metric(y, p_b) - metric(y, p_a),
        "low": float(lo),
        "high": float(hi),
        "p_b_worse": float(np.mean(diffs < 0)),  # share of resamples where B loses
        "n_boot": n_boot,
        "alpha": alpha,
    }


def fast_paired_accuracy(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Same result as paired_bootstrap(metric=accuracy) but vectorised.

    Accuracy difference only depends on the per-item score d_i = [B right] - [A right],
    so we resample d directly in one matrix op."""
    d = (y == p_b).astype(np.float64) - (y == p_a).astype(np.float64)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n_boot)
    chunk = max(1, 2_000_000 // max(n, 1))  # keep memory bounded
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        idx = rng.integers(0, n, (e - s, n))
        means[s:e] = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "diff": float(d.mean()),
        "low": float(lo),
        "high": float(hi),
        "p_b_worse": float(np.mean(means < 0)),
        "n_boot": n_boot,
        "alpha": alpha,
    }
