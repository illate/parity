import numpy as np
import pytest

from illate_parity.bootstrap import bootstrap_ci, fast_paired_accuracy, paired_bootstrap


def _data(n=800, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)
    pa = np.where(rng.random(n) < 0.85, y, (y + 1) % 5)
    pb = np.where(rng.random(n) < 0.80, y, (y + 2) % 5)
    return y, pa, pb


def test_fast_matches_generic():
    y, pa, pb = _data()
    slow = paired_bootstrap(y, pa, pb, n_boot=3000, seed=7)
    fast = fast_paired_accuracy(y, pa, pb, n_boot=3000, seed=7)
    assert fast["diff"] == pytest.approx(slow["diff"])
    # different random streams, so compare loosely
    assert fast["low"] == pytest.approx(slow["low"], abs=0.006)
    assert fast["high"] == pytest.approx(slow["high"], abs=0.006)


def test_ci_contains_point_and_is_sane():
    y, pa, _ = _data()
    acc, lo, hi = bootstrap_ci(y, pa, n_boot=2000)
    assert lo < acc < hi
    # normal approximation: half-width ~ 1.96*sqrt(p(1-p)/n)
    expected = 1.96 * np.sqrt(acc * (1 - acc) / len(y))
    assert (hi - lo) / 2 == pytest.approx(expected, rel=0.15)


def test_paired_is_tighter_than_unpaired():
    """Two models that make mostly the same mistakes: pairing should shrink the CI."""
    rng = np.random.default_rng(3)
    n = 1000
    y = rng.integers(0, 4, n)
    pa = np.where(rng.random(n) < 0.8, y, (y + 1) % 4)
    pb = pa.copy()
    flip = rng.random(n) < 0.05
    pb[flip] = y[flip]  # B fixes a few of A's mistakes
    paired = fast_paired_accuracy(y, pa, pb, n_boot=3000)
    _, alo, ahi = bootstrap_ci(y, pa, n_boot=3000)
    _, blo, bhi = bootstrap_ci(y, pb, n_boot=3000, seed=1)
    assert (paired["high"] - paired["low"]) < 0.5 * ((ahi - alo) + (bhi - blo))
    assert paired["low"] > 0  # B is reliably better
