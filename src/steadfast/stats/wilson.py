"""Wilson confidence interval for binomial proportions.

Used by any pass-rate / success-rate metric (format consistency,
overconfidence rate, catastrophic-failure rate, refusal calibration
cells). Wilson is the standard-of-care for binomial CIs at small N —
better coverage than Wald and tighter than Clopper-Pearson at the
sample sizes Steadfast typically reports (N=10 reps).

Implementation wraps :func:`scipy.stats.binomtest` (per CLAUDE.md "no
hand-rolled statistical functions"); we expose a typed Pydantic result
shape for downstream serialization.

References:

* Wilson, E. B. (1927). "Probable inference, the law of succession,
  and statistical inference." *JASA* 22(158), 209-212.
* Brown, Cai & DasGupta (2001). "Interval estimation for a binomial
  proportion." *Statistical Science* 16(2), 101-133 (comparison vs
  Wald and Clopper-Pearson).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import binomtest

DEFAULT_CONFIDENCE_LEVEL = 0.95


class WilsonCI(BaseModel):
    """Wilson-score confidence interval for a binomial proportion."""

    model_config = ConfigDict(frozen=True)

    successes: int = Field(ge=0)
    trials: int = Field(ge=0)
    proportion: float = Field(ge=0.0, le=1.0)
    ci_lower: float = Field(ge=0.0, le=1.0)
    ci_upper: float = Field(ge=0.0, le=1.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)


def wilson_ci(
    successes: int,
    trials: int,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> WilsonCI:
    """Wilson 95% (default) CI for ``successes / trials``.

    Raises :class:`ValueError` if ``trials == 0`` — there is no
    interpretable interval for the empty sample. Caller should report
    "N/A" in that case.
    """
    if trials < 0:
        raise ValueError("trials must be non-negative")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be in [0, trials]")
    if trials == 0:
        raise ValueError("wilson_ci requires trials >= 1")

    test = binomtest(k=successes, n=trials)
    interval = test.proportion_ci(confidence_level=confidence_level, method="wilson")
    return WilsonCI(
        successes=successes,
        trials=trials,
        proportion=successes / trials,
        ci_lower=float(interval.low),
        ci_upper=float(interval.high),
        confidence_level=confidence_level,
    )
