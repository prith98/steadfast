"""Bootstrap confidence intervals — wraps :func:`scipy.stats.bootstrap`.

Per ``docs/METHODOLOGY.md`` §"Statistical conventions": **95% CI, BCa
method, 10,000 resamples**. This wrapper is the canonical entry point —
no other module is allowed to call ``scipy.stats.bootstrap`` directly,
and we never hand-roll bootstrap code (``CLAUDE.md`` "Tech stack").

Edge cases per ADR-0004 §H:

* Empty data → :class:`ValueError`.
* ``N < 2`` → :class:`ValueError` (BCa needs at least two samples).
* Zero-variance data (all identical) → returns ``(point, point)`` with
  ``degenerate=True`` rather than the NaN scipy emits.

References:

* Efron & Tibshirani, *An Introduction to the Bootstrap* (1993).
* Efron (1987), "Better bootstrap confidence intervals", *JASA*
  82(397), 171-185 (BCa formulation).
* SciPy docs:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import DegenerateDataWarning, bootstrap

# Methodology defaults (ADR-0004 §H, METHODOLOGY §"Statistical conventions").
DEFAULT_N_RESAMPLES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_METHOD: Literal["BCa", "percentile", "basic"] = "BCa"


class BootstrapCI(BaseModel):
    """Bootstrap confidence interval result.

    ``degenerate`` is True when the input had zero variance and the
    interval collapses to ``(point_estimate, point_estimate)``. Without
    the flag, a downstream consumer can't distinguish "tightly estimated"
    from "no variation in the data."
    """

    model_config = ConfigDict(frozen=True)

    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    method: str
    n_resamples: int
    n_samples: int
    degenerate: bool = False


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float] | None = None,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    method: Literal["BCa", "percentile", "basic"] = DEFAULT_METHOD,
    seed: int | None = None,
) -> BootstrapCI:
    """Bootstrap CI for ``statistic`` over ``data`` with the v0.1 defaults.

    ``statistic`` defaults to :func:`numpy.mean`; pass any callable that
    takes a 1-D ndarray and returns a scalar (e.g.,
    ``lambda x: np.median(x)``).
    """
    if len(data) == 0:
        raise ValueError("bootstrap requires non-empty data")
    if len(data) < 2:
        raise ValueError("bootstrap requires N >= 2 samples")

    arr = np.asarray(data, dtype=float)
    stat: Callable[[np.ndarray], float] = (
        statistic if statistic is not None else (lambda x: float(np.mean(x)))
    )
    point = float(stat(arr))

    # BCa's acceleration term is undefined when variance is zero;
    # without this short-circuit scipy lets NaN propagate.
    if float(np.ptp(arr)) == 0.0:
        return BootstrapCI(
            point_estimate=point,
            ci_lower=point,
            ci_upper=point,
            confidence_level=confidence_level,
            method=method,
            n_resamples=n_resamples,
            n_samples=len(arr),
            degenerate=True,
        )

    rng = np.random.default_rng(seed)

    # SciPy emits a DegenerateDataWarning in some borderline near-degenerate
    # cases that aren't caught by our ptp==0 check; treat them as warnings,
    # not errors, and let the resulting CI flow through.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegenerateDataWarning)
        result = bootstrap(
            (arr,),
            stat,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method=method,
            random_state=rng,
            vectorized=False,
        )

    lo = float(result.confidence_interval.low)
    hi = float(result.confidence_interval.high)
    # NaN can still appear if scipy's BCa internals fail; clamp to the
    # point estimate and flag degenerate so the caller can decide.
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return BootstrapCI(
            point_estimate=point,
            ci_lower=point,
            ci_upper=point,
            confidence_level=confidence_level,
            method=method,
            n_resamples=n_resamples,
            n_samples=len(arr),
            degenerate=True,
        )
    return BootstrapCI(
        point_estimate=point,
        ci_lower=lo,
        ci_upper=hi,
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        n_samples=len(arr),
        degenerate=False,
    )
