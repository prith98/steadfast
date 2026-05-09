"""Calibration math — Brier score and Expected Calibration Error.

These are the *statistical primitives* used by ``steadfast.metrics.calibration``;
keeping them here lets the metric module focus on data plumbing and reporting
while this module owns the math.

* **Brier score** (Brier 1950): ``mean((p - y) ** 2)`` over a flat array
  of (forecast, outcome) pairs. The metric layer is responsible for
  pooling reps across tasks before calling :func:`brier_score` /
  :func:`brier_squared_errors`.
* **ECE with equal-mass bins** (Guo et al. 2017 / Nixon et al. 2019):
  bins of equal sample mass (rather than equal width) — equal-mass is
  more robust when confidence concentrates near 1.0, which it always
  does for current LLMs. The implementation sorts the (confidence,
  outcome) pairs and partitions into ``n_bins`` contiguous chunks of
  ``floor(N / n_bins)`` or ``ceil(N / n_bins)`` samples each, then
  computes ``Σ_b (n_b / N) · |acc_b - conf_b|``.

Both functions are pure over numpy arrays — no I/O, no LLM calls, no
side effects. The bootstrap CI is layered on top by the metric module
via :func:`steadfast.stats.bootstrap.bootstrap_ci`.

Citations:

* Brier, G. W. (1950). "Verification of forecasts expressed in terms of
  probability." *Monthly Weather Review* 78(1), 1-3.
* Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On
  calibration of modern neural networks." *ICML*.
  https://arxiv.org/abs/1706.04599
* Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G., & Tran, D.
  (2019). "Measuring calibration in deep learning." *CVPR Workshops*.
  https://arxiv.org/abs/1904.01685
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Methodology default per docs/METHODOLOGY.md §3.3 / ADR-0005 §D.
DEFAULT_ECE_BINS = 15
# Below ECE_FALLBACK_MIN_BINS the result is N/A; between
# ECE_FALLBACK_MIN_BINS and DEFAULT_ECE_BINS the implementation uses
# floor(N / 3) bins instead of 15.
ECE_FALLBACK_MIN_BINS = 3


class BinStats(BaseModel):
    """One ECE bin's accuracy / mean-confidence / population.

    Lands in :class:`ECEResult` so consumers can render reliability
    diagrams without recomputing.
    """

    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=0)
    mean_confidence: float
    accuracy: float
    gap: float  # |accuracy - mean_confidence|


def brier_squared_errors(
    confidences: Sequence[float],
    outcomes: Sequence[int],
) -> np.ndarray:
    """Return the per-prediction squared errors ``(p_i - y_i) ** 2``.

    ``confidences`` are forecast probabilities in [0, 1]; ``outcomes`` are
    binary 0/1. The returned array is what the metric module hands to
    :func:`steadfast.stats.bootstrap.bootstrap_ci` so the CI is over the
    pool of squared errors (per ADR-0005 §D).
    """
    if len(confidences) != len(outcomes):
        raise ValueError(
            f"confidences/outcomes length mismatch: {len(confidences)} vs {len(outcomes)}"
        )
    p = np.asarray(confidences, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return np.zeros(0, dtype=float)
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("confidences must be in [0, 1]")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("outcomes must be 0 or 1")
    errors: np.ndarray = (p - y) ** 2
    return errors


def brier_score(
    confidences: Sequence[float],
    outcomes: Sequence[int],
) -> float:
    """Mean Brier score over the (forecast, outcome) pool.

    Convenience around :func:`brier_squared_errors`. The metric module's
    bootstrap CI is computed by resampling the squared-error array, not
    by calling this function repeatedly.
    """
    errs = brier_squared_errors(confidences, outcomes)
    if errs.size == 0:
        raise ValueError("brier_score requires at least one (forecast, outcome) pair")
    return float(np.mean(errs))


def equal_mass_bin_indices(n: int, n_bins: int) -> list[tuple[int, int]]:
    """Return half-open ``(start, end)`` index pairs for ``n_bins`` equal-mass bins.

    "Equal-mass" means each bin contains either ``floor(n / n_bins)`` or
    ``ceil(n / n_bins)`` items; the first ``n % n_bins`` bins get the
    extra item (per Nixon et al. 2019). Pure index computation — does not
    look at the data — so the metric module can sort the data once and
    slice per-bin.

    Edge cases: ``n_bins <= 0`` raises; ``n == 0`` returns ``[]``;
    ``n_bins > n`` would yield empty bins and is rejected (caller should
    use the small-N fallback documented in METHODOLOGY §3.3).
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be >= 1")
    if n == 0:
        return []
    if n_bins > n:
        raise ValueError(f"n_bins ({n_bins}) > n ({n}); use a smaller bin count for small samples")
    base = n // n_bins
    extra = n % n_bins
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for i in range(n_bins):
        size = base + (1 if i < extra else 0)
        boundaries.append((cursor, cursor + size))
        cursor += size
    return boundaries


def expected_calibration_error(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_bins: int = DEFAULT_ECE_BINS,
) -> tuple[float, list[BinStats]]:
    """ECE with equal-mass bins. Returns ``(ece, bin_stats)``.

    Implements ``Σ_b (n_b / N) · |acc_b - conf_b|`` with equal-mass
    binning per Nixon et al. 2019. Sort is by confidence ascending; ties
    fall into adjacent bins per their array order (numpy ``argsort`` is
    stable, so ordering is reproducible across runs).

    Caller should handle the small-N fallback — when ``len(confidences)
    < n_bins`` the function will raise via ``equal_mass_bin_indices``.
    The metric module wraps this with the documented fallback to
    ``floor(N / 3)`` bins (METHODOLOGY §3.3 / ADR-0005 §D).
    """
    p = np.asarray(confidences, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size != y.size:
        raise ValueError("confidences/outcomes length mismatch")
    if p.size == 0:
        raise ValueError("ECE requires at least one (forecast, outcome) pair")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("confidences must be in [0, 1]")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("outcomes must be 0 or 1")

    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    y_sorted = y[order]

    boundaries = equal_mass_bin_indices(p.size, n_bins)
    bin_stats: list[BinStats] = []
    weighted_gaps: list[float] = []
    n = p.size
    for start, end in boundaries:
        if start == end:  # skip empty bin (shouldn't happen given guards above)
            continue
        bin_confidences = p_sorted[start:end]
        bin_outcomes = y_sorted[start:end]
        mean_conf = float(np.mean(bin_confidences))
        accuracy = float(np.mean(bin_outcomes))
        gap = abs(accuracy - mean_conf)
        bin_stats.append(
            BinStats(
                n=int(end - start),
                mean_confidence=mean_conf,
                accuracy=accuracy,
                gap=gap,
            )
        )
        weighted_gaps.append((end - start) / n * gap)

    ece = float(sum(weighted_gaps))
    return ece, bin_stats


__all__ = [
    "DEFAULT_ECE_BINS",
    "ECE_FALLBACK_MIN_BINS",
    "BinStats",
    "brier_score",
    "brier_squared_errors",
    "equal_mass_bin_indices",
    "expected_calibration_error",
]
