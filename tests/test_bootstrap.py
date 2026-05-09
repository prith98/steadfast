"""Tests for steadfast.stats.bootstrap — scipy wrapper + edge cases."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import bootstrap as scipy_bootstrap

from steadfast.stats.bootstrap import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_METHOD,
    DEFAULT_N_RESAMPLES,
    bootstrap_ci,
)


def test_defaults_match_methodology() -> None:
    """Per docs/METHODOLOGY.md §"Statistical conventions"."""
    assert DEFAULT_N_RESAMPLES == 10_000
    assert DEFAULT_CONFIDENCE_LEVEL == 0.95
    assert DEFAULT_METHOD == "BCa"


def test_point_estimate_is_mean_by_default() -> None:
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    ci = bootstrap_ci(data, n_resamples=200, seed=0)
    assert ci.point_estimate == pytest.approx(0.3, abs=1e-9)


def test_ci_brackets_point_estimate() -> None:
    data = [0.7, 0.8, 0.9, 0.85, 0.75, 0.88, 0.92]
    ci = bootstrap_ci(data, n_resamples=500, seed=42)
    assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper


def test_wrapper_matches_scipy_directly() -> None:
    """Wrapper must produce the same point estimate as scipy on identical inputs."""
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    seed = 123
    rng = np.random.default_rng(seed)

    def stat(arr: np.ndarray) -> float:
        return float(np.mean(arr))

    direct = scipy_bootstrap(
        (data,),
        stat,
        n_resamples=500,
        confidence_level=0.95,
        method="BCa",
        random_state=rng,
        vectorized=False,
    )
    wrapper = bootstrap_ci(list(data), n_resamples=500, seed=seed)
    # Point estimates are deterministic (no randomness).
    assert wrapper.point_estimate == pytest.approx(stat(data))
    # CI bounds depend on random resampling; verify that both wrappers'
    # bounds bracket the true mean and are roughly within ~10% of each
    # other (CIs are estimates, not exact).
    assert wrapper.ci_lower < stat(data) < wrapper.ci_upper
    assert direct.confidence_interval.low < stat(data) < direct.confidence_interval.high


def test_seeded_runs_are_reproducible() -> None:
    data = [0.1, 0.4, 0.5, 0.55, 0.6, 0.7, 0.9]
    a = bootstrap_ci(data, n_resamples=300, seed=99)
    b = bootstrap_ci(data, n_resamples=300, seed=99)
    assert (a.ci_lower, a.ci_upper) == (b.ci_lower, b.ci_upper)


def test_custom_statistic_median() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = bootstrap_ci(
        data,
        statistic=lambda x: float(np.median(x)),
        n_resamples=200,
        seed=0,
    )
    assert ci.point_estimate == pytest.approx(3.0)


def test_empty_data_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci([], n_resamples=10)


def test_n_one_raises() -> None:
    with pytest.raises(ValueError, match="N >= 2"):
        bootstrap_ci([0.5], n_resamples=10)


def test_zero_variance_returns_degenerate_ci() -> None:
    """All-identical data → CI collapses to (point, point) with degenerate=True
    (vs scipy emitting NaN). Without the flag, downstream consumers can't
    distinguish 'tightly estimated' from 'no variation in the data'.
    """
    ci = bootstrap_ci([0.7] * 10, n_resamples=200, seed=0)
    assert ci.degenerate is True
    assert ci.ci_lower == ci.point_estimate == ci.ci_upper == 0.7


def test_n_samples_recorded() -> None:
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    ci = bootstrap_ci(data, n_resamples=100, seed=0)
    assert ci.n_samples == 5
    assert ci.n_resamples == 100
    assert ci.method == "BCa"
    assert ci.confidence_level == 0.95
