"""Tests for steadfast.stats.paired_bootstrap — paired-delta CI primitive.

Hand-computed expected values per ADR-0006 §F. The paired bootstrap is
defined as ``mean(perturbed_rates - clean_rates)`` plus a BCa CI on
that mean; equivalence to bootstrapping over (task, both-arms) pairs
is the §F derivation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from steadfast.stats.paired_bootstrap import PairedBootstrapCI, paired_bootstrap_ci

# ---------------------------------------------------------------------------
# Hand-computed point estimates (no randomness in the mean — only in the CI).
# ---------------------------------------------------------------------------


def test_identical_arms_delta_zero() -> None:
    """Both arms identical → delta = 0; degenerate flag set (zero-variance
    delta vector); CI collapses to (0, 0)."""
    clean = [0.7, 0.8, 0.6, 0.9, 0.75]
    perturbed = list(clean)
    result = paired_bootstrap_ci(clean, perturbed, n_resamples=200, seed=0)
    assert result.delta == pytest.approx(0.0, abs=1e-12)
    assert result.degenerate is True
    assert result.delta_ci_lower == result.delta_ci_upper == result.delta
    assert result.clean_mean == pytest.approx(0.75, abs=1e-12)
    assert result.perturbed_mean == pytest.approx(0.75, abs=1e-12)


def test_all_clean_pass_all_perturbed_fail_delta_minus_one() -> None:
    """Per ADR-0006 §F worst-case: clean=1.0 across all tasks, perturbed=0.0
    across all tasks → delta = -1.0; per-task delta vector is constant so
    the CI is degenerate at -1.0."""
    clean = [1.0] * 5
    perturbed = [0.0] * 5
    result = paired_bootstrap_ci(clean, perturbed, n_resamples=200, seed=0)
    assert result.delta == pytest.approx(-1.0)
    assert result.degenerate is True
    assert result.delta_ci_lower == result.delta_ci_upper == -1.0
    assert result.clean_mean == 1.0
    assert result.perturbed_mean == 0.0


def test_three_task_synthetic_hand_computed_delta() -> None:
    """The ADR-0006 §F worked example.

    clean      = [1.0, 0.8, 0.6]
    perturbed  = [0.7, 0.5, 0.4]
    deltas     = [-0.3, -0.3, -0.2]
    mean delta = -0.8 / 3 ≈ -0.26666...

    With three observations and two distinct delta values (variance > 0),
    the CI is non-degenerate and brackets the point estimate.
    """
    clean = [1.0, 0.8, 0.6]
    perturbed = [0.7, 0.5, 0.4]
    result = paired_bootstrap_ci(clean, perturbed, n_resamples=500, seed=42)
    expected_delta = -0.8 / 3
    assert result.delta == pytest.approx(expected_delta, abs=1e-12)
    assert result.clean_mean == pytest.approx(0.8, abs=1e-12)
    assert result.perturbed_mean == pytest.approx(8.0 / 15, abs=1e-12)
    assert result.degenerate is False
    assert result.delta_ci_lower <= result.delta <= result.delta_ci_upper
    # Both bounds in the realizable per-task delta range [-0.3, -0.2].
    assert result.delta_ci_lower >= -0.3 - 1e-9
    assert result.delta_ci_upper <= -0.2 + 1e-9


def test_n_tasks_recorded() -> None:
    result = paired_bootstrap_ci(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.0, 0.1, 0.2, 0.3, 0.4],
        n_resamples=100,
        seed=0,
    )
    assert result.n_tasks == 5
    assert result.n_resamples == 100
    assert result.method == "BCa"
    assert result.confidence_level == 0.95


def test_seeded_runs_are_reproducible() -> None:
    clean = [0.9, 0.7, 0.5, 0.6, 0.8]
    perturbed = [0.7, 0.6, 0.4, 0.5, 0.6]
    a = paired_bootstrap_ci(clean, perturbed, n_resamples=300, seed=99)
    b = paired_bootstrap_ci(clean, perturbed, n_resamples=300, seed=99)
    assert (a.delta_ci_lower, a.delta_ci_upper) == (b.delta_ci_lower, b.delta_ci_upper)


def test_result_is_frozen_pydantic() -> None:
    """PairedBootstrapCI must be frozen (no field mutation post-construction)
    — matches BootstrapCI's contract from ADR-0004 §H."""
    result = paired_bootstrap_ci(
        [0.5, 0.7, 0.6],
        [0.4, 0.5, 0.5],
        n_resamples=100,
        seed=0,
    )
    assert isinstance(result, PairedBootstrapCI)
    with pytest.raises(ValidationError):
        result.delta = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases (mirror stats/bootstrap.py per ADR-0004 §H).
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_ci([0.5, 0.7], [0.4], n_resamples=10)


def test_empty_data_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_ci([], [], n_resamples=10)


def test_n_one_raises() -> None:
    with pytest.raises(ValueError, match="N >= 2"):
        paired_bootstrap_ci([0.5], [0.4], n_resamples=10)


# ---------------------------------------------------------------------------
# Methodology consistency — defaults match the canonical entry point.
# ---------------------------------------------------------------------------


def test_defaults_match_methodology() -> None:
    """Per docs/METHODOLOGY.md §"Statistical conventions" via ADR-0006 §F.

    The paired-bootstrap defaults are inherited from stats/bootstrap.py;
    this test pins the inheritance so a future drift in either module is
    caught.
    """
    from steadfast.stats.bootstrap import (
        DEFAULT_CONFIDENCE_LEVEL,
        DEFAULT_METHOD,
        DEFAULT_N_RESAMPLES,
    )

    result = paired_bootstrap_ci(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.0, 0.1, 0.2, 0.3, 0.4],
        n_resamples=DEFAULT_N_RESAMPLES,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        method=DEFAULT_METHOD,
        seed=0,
    )
    assert result.method == DEFAULT_METHOD
    assert result.n_resamples == DEFAULT_N_RESAMPLES
    assert result.confidence_level == DEFAULT_CONFIDENCE_LEVEL


def test_independent_consumer_can_recover_per_task_deltas() -> None:
    """The metric layer (tomorrow's measure_typo_robustness) recomputes per-task
    deltas for its diagnostic table without the stats primitive having to
    expose them. Sanity-check the math the metric layer will rely on."""
    clean = [1.0, 0.8, 0.6]
    perturbed = [0.7, 0.5, 0.4]
    result = paired_bootstrap_ci(clean, perturbed, n_resamples=100, seed=0)
    per_task_deltas = [p - c for c, p in zip(clean, perturbed, strict=True)]
    # All three list elements need approx() — float subtraction yields e.g.
    # -0.30000000000000004 for 0.7 - 1.0 (IEEE-754 representation).
    assert per_task_deltas == [pytest.approx(-0.3), pytest.approx(-0.3), pytest.approx(-0.2)]
    assert result.delta == pytest.approx(sum(per_task_deltas) / len(per_task_deltas))
