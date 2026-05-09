"""Tests for steadfast.stats.calibration — Brier and ECE math primitives."""

from __future__ import annotations

import numpy as np
import pytest

from steadfast.stats.calibration import (
    DEFAULT_ECE_BINS,
    BinStats,
    brier_score,
    brier_squared_errors,
    equal_mass_bin_indices,
    expected_calibration_error,
)

# ---------------------------------------------------------------------------
# brier_squared_errors / brier_score
# ---------------------------------------------------------------------------


def test_brier_squared_errors_basic() -> None:
    """Hand-computed: forecasts [0.8, 0.6, 0.4] outcomes [1, 0, 0].

    Squared errors: (0.8-1)^2=0.04, (0.6-0)^2=0.36, (0.4-0)^2=0.16
    Mean = (0.04 + 0.36 + 0.16) / 3 = 0.18667
    """
    errs = brier_squared_errors([0.8, 0.6, 0.4], [1, 0, 0])
    assert errs == pytest.approx([0.04, 0.36, 0.16])
    assert brier_score([0.8, 0.6, 0.4], [1, 0, 0]) == pytest.approx(0.18667, abs=1e-3)


def test_brier_perfect_forecasts_score_zero() -> None:
    """Perfect forecasts (1.0 for correct, 0.0 for incorrect) → Brier = 0."""
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0


def test_brier_worst_forecasts_score_one() -> None:
    """Maximally wrong forecasts → Brier = 1.0."""
    assert brier_score([0.0, 1.0, 0.0], [1, 0, 1]) == pytest.approx(1.0)


def test_brier_uniform_half_forecast() -> None:
    """Forecasting 0.5 always → Brier = 0.25 regardless of outcomes."""
    assert brier_score([0.5] * 10, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) == pytest.approx(0.25)


def test_brier_squared_errors_empty_returns_empty_array() -> None:
    errs = brier_squared_errors([], [])
    assert isinstance(errs, np.ndarray)
    assert errs.size == 0


def test_brier_squared_errors_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        brier_squared_errors([0.5, 0.5], [1])


def test_brier_squared_errors_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        brier_squared_errors([1.5], [1])


def test_brier_squared_errors_non_binary_outcomes() -> None:
    with pytest.raises(ValueError, match="must be 0 or 1"):
        brier_squared_errors([0.5], [2])


def test_brier_score_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        brier_score([], [])


# ---------------------------------------------------------------------------
# equal_mass_bin_indices
# ---------------------------------------------------------------------------


def test_equal_mass_bins_evenly_divides() -> None:
    """N=15 with 5 bins → 3 items each."""
    bins = equal_mass_bin_indices(15, 5)
    assert bins == [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]


def test_equal_mass_bins_uneven_distribution() -> None:
    """N=10, n_bins=3 → first n_bins remainder=1 gets the extra item: 4, 3, 3."""
    bins = equal_mass_bin_indices(10, 3)
    sizes = [end - start for start, end in bins]
    assert sizes == [4, 3, 3]
    assert sum(sizes) == 10


def test_equal_mass_bins_all_items_covered() -> None:
    """The union of bins always covers every index in [0, N)."""
    for n in [1, 7, 13, 100]:
        for n_bins in [1, 2, 3, 7]:
            if n_bins > n:
                continue
            bins = equal_mass_bin_indices(n, n_bins)
            covered = sum(end - start for start, end in bins)
            assert covered == n


def test_equal_mass_bins_zero_items() -> None:
    assert equal_mass_bin_indices(0, 5) == []


def test_equal_mass_bins_invalid_n_bins() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        equal_mass_bin_indices(10, 0)


def test_equal_mass_bins_too_few_items() -> None:
    with pytest.raises(ValueError, match="use a smaller bin count"):
        equal_mass_bin_indices(5, 10)


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration_is_zero() -> None:
    """A perfectly calibrated forecaster: 0.5 with outcomes 50/50 within each bin."""
    # Construct: 30 forecasts at 0.3 with 30% positive; 30 at 0.7 with 70% positive.
    # Within each bin, accuracy == mean_confidence → gap = 0 → ECE = 0.
    confidences = [0.3] * 30 + [0.7] * 30
    outcomes = [1] * 9 + [0] * 21 + [1] * 21 + [0] * 9
    ece, bin_stats = expected_calibration_error(confidences, outcomes, n_bins=2)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert all(bs.gap == pytest.approx(0.0) for bs in bin_stats)


def test_ece_total_miscalibration() -> None:
    """All forecasts at 1.0; all outcomes 0 → bin gap = 1.0 → ECE = 1.0."""
    confidences = [1.0] * 10
    outcomes = [0] * 10
    ece, _ = expected_calibration_error(confidences, outcomes, n_bins=2)
    assert ece == pytest.approx(1.0)


def test_ece_known_two_bin_value() -> None:
    """Hand-computed two-bin ECE.

    After sort: [(0.1,0), (0.2,1), (0.3,0), (0.7,1), (0.8,0), (0.9,1)]
    bin0 = first 3: mean_conf = 0.2, accuracy = 1/3, gap = |1/3 - 0.2| = 0.1333
    bin1 = last 3:  mean_conf = 0.8, accuracy = 2/3, gap = |2/3 - 0.8| = 0.1333
    Weighted ECE = 0.5 * 0.1333 + 0.5 * 0.1333 = 0.1333
    """
    confidences = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    outcomes = [0, 1, 0, 1, 0, 1]
    ece, bin_stats = expected_calibration_error(confidences, outcomes, n_bins=2)
    assert ece == pytest.approx(2 / 15)  # exact: 1/3 - 1/5 = 2/15
    assert bin_stats[0].mean_confidence == pytest.approx(0.2)
    assert bin_stats[0].accuracy == pytest.approx(1 / 3)
    assert bin_stats[1].mean_confidence == pytest.approx(0.8)
    assert bin_stats[1].accuracy == pytest.approx(2 / 3)


def test_ece_with_default_15_bins() -> None:
    """Sanity: N=30 with 15 bins → 2 items per bin; doesn't raise."""
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0, 1, size=30).tolist()
    outcomes = rng.integers(0, 2, size=30).tolist()
    ece, bin_stats = expected_calibration_error(confidences, outcomes, n_bins=DEFAULT_ECE_BINS)
    assert 0.0 <= ece <= 1.0
    assert len(bin_stats) == DEFAULT_ECE_BINS
    assert all(bs.n == 2 for bs in bin_stats)


def test_ece_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        expected_calibration_error([], [], n_bins=15)


def test_bin_stats_fields() -> None:
    """BinStats carries n / mean_confidence / accuracy / gap and is frozen."""
    from pydantic import ValidationError

    bs = BinStats(n=5, mean_confidence=0.6, accuracy=0.4, gap=0.2)
    with pytest.raises(ValidationError):
        bs.n = 10  # type: ignore[misc]
    assert bs.n == 5
    assert bs.gap == pytest.approx(0.2)
