"""Tests for steadfast.stats.wilson — Wilson 95% CI for binomial proportions."""

from __future__ import annotations

import pytest
from scipy.stats import binomtest

from steadfast.stats.wilson import DEFAULT_CONFIDENCE_LEVEL, wilson_ci


def test_default_confidence_level() -> None:
    assert DEFAULT_CONFIDENCE_LEVEL == 0.95


def test_proportion_correct() -> None:
    ci = wilson_ci(8, 10)
    assert ci.proportion == 0.8
    assert ci.successes == 8
    assert ci.trials == 10


def test_ci_brackets_proportion() -> None:
    ci = wilson_ci(8, 10)
    assert ci.ci_lower <= ci.proportion <= ci.ci_upper


def test_wrapper_matches_scipy_binomtest() -> None:
    """Wilson CI bounds must match a direct scipy.stats.binomtest call."""
    direct = binomtest(k=7, n=10).proportion_ci(method="wilson")
    ci = wilson_ci(7, 10)
    assert ci.ci_lower == pytest.approx(direct.low)
    assert ci.ci_upper == pytest.approx(direct.high)


def test_perfect_pass_rate_has_nonzero_lower_bound() -> None:
    """10/10 successes → Wilson CI lower bound > 0 (vs Wald which gives 1.0/1.0)."""
    ci = wilson_ci(10, 10)
    assert ci.proportion == 1.0
    assert ci.ci_lower < 1.0
    assert ci.ci_upper == pytest.approx(1.0, abs=1e-9)


def test_zero_pass_rate_has_nonzero_upper_bound() -> None:
    """0/10 successes → Wilson CI upper bound > 0 (Wald collapses)."""
    ci = wilson_ci(0, 10)
    assert ci.proportion == 0.0
    assert ci.ci_lower == pytest.approx(0.0, abs=1e-9)
    assert ci.ci_upper > 0.0


def test_zero_trials_raises() -> None:
    with pytest.raises(ValueError, match="trials >= 1"):
        wilson_ci(0, 0)


def test_successes_exceeds_trials_raises() -> None:
    with pytest.raises(ValueError, match=r"successes must be in \[0, trials\]"):
        wilson_ci(11, 10)


def test_negative_inputs_raise() -> None:
    with pytest.raises(ValueError, match=r"successes must be in \[0, trials\]"):
        wilson_ci(-1, 10)
    with pytest.raises(ValueError, match="trials must be non-negative"):
        wilson_ci(5, -1)
