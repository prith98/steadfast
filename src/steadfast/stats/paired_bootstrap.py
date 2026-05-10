"""Paired bootstrap CI for robustness deltas.

Per ``docs/METHODOLOGY.md`` §2 and ADR-0006 §F: robustness sub-metrics
report a 95% CI on the **delta** (perturbed success rate minus clean
success rate), not on the two endpoints separately. The pairing is
intrinsic to the metric definition — without per-task pairing, the
within-task correlation between clean and perturbed outcomes (driven by
per-task difficulty) doesn't get factored out of the CI.

This module exposes :func:`paired_bootstrap_ci`, which takes per-task
clean and perturbed success rates (paired by index — ``clean_rates[i]``
and ``perturbed_rates[i]`` are the same task's two arms) and returns a
bootstrap CI on the mean per-task delta.

Implementation is delegated to :func:`steadfast.stats.bootstrap.bootstrap_ci`
because pre-computing deltas and bootstrapping over the delta array is
mathematically equivalent to bootstrapping over (task, both-arms) pairs
and taking the per-resample mean delta — a 1-1 function of the original
data. This keeps the BCa, edge-case, and degenerate-data handling in one
place (``stats/bootstrap.py`` per ADR-0004 §H).

References:

* Efron & Tibshirani (1993), *An Introduction to the Bootstrap*,
  §10 ("Confidence intervals based on bootstrap percentiles") — paired
  / matched-pairs bootstrap baseline.
* Field & Welsh (2007), *Bootstrapping clustered data*, *JRSS-B* 69(3),
  369-390 — cluster bootstrap rationale that underwrites resampling at
  the task (cluster) level.
* ADR-0006 §F — the v0.1 contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from steadfast.stats.bootstrap import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_METHOD,
    DEFAULT_N_RESAMPLES,
    bootstrap_ci,
)


class PairedBootstrapCI(BaseModel):
    """Paired bootstrap CI for the mean per-task delta.

    ``delta`` is the point estimate ``mean(perturbed_rates - clean_rates)``;
    ``delta_ci_lower`` / ``delta_ci_upper`` are the bootstrap interval on
    that mean. ``clean_mean`` and ``perturbed_mean`` are reported alongside
    so the leaderboard row can show both endpoints without recomputing.

    ``degenerate`` is True when the per-task delta vector had zero
    variance (e.g., every task degraded by exactly the same amount, or
    both arms produced identical rates per task) and the CI collapsed to
    ``(delta, delta)``. Same semantics as
    :class:`steadfast.stats.bootstrap.BootstrapCI`.
    """

    model_config = ConfigDict(frozen=True)

    clean_mean: float
    perturbed_mean: float
    delta: float
    delta_ci_lower: float
    delta_ci_upper: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    method: str
    n_resamples: int
    n_tasks: int
    degenerate: bool = False


def paired_bootstrap_ci(
    clean_rates: Sequence[float],
    perturbed_rates: Sequence[float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    method: Literal["BCa", "percentile", "basic"] = DEFAULT_METHOD,
    seed: int | None = None,
) -> PairedBootstrapCI:
    """Paired bootstrap CI on the mean per-task delta.

    ``clean_rates[i]`` and ``perturbed_rates[i]`` must be the same task's
    two arms (paired by index). The delta vector
    ``perturbed_rates - clean_rates`` is computed once, then bootstrapped
    via :func:`steadfast.stats.bootstrap.bootstrap_ci`.

    Edge cases mirror ``stats/bootstrap.py`` per ADR-0004 §H:

    * Length mismatch between arms → :class:`ValueError`.
    * Empty input → :class:`ValueError`.
    * ``N < 2`` paired observations → :class:`ValueError` (BCa needs at
      least two samples for the acceleration term).
    * Zero-variance delta vector → degenerate CI flagged on the result
      (delegated to :func:`bootstrap_ci`).
    """
    if len(clean_rates) != len(perturbed_rates):
        raise ValueError(
            f"paired bootstrap requires equal-length arms; "
            f"got len(clean_rates)={len(clean_rates)}, "
            f"len(perturbed_rates)={len(perturbed_rates)}"
        )
    if len(clean_rates) == 0:
        raise ValueError("paired bootstrap requires non-empty data")
    if len(clean_rates) < 2:
        raise ValueError("paired bootstrap requires N >= 2 paired observations")

    clean_arr = np.asarray(clean_rates, dtype=float)
    perturbed_arr = np.asarray(perturbed_rates, dtype=float)
    deltas = (perturbed_arr - clean_arr).tolist()

    delta_ci = bootstrap_ci(
        deltas,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        seed=seed,
    )

    return PairedBootstrapCI(
        clean_mean=float(np.mean(clean_arr)),
        perturbed_mean=float(np.mean(perturbed_arr)),
        delta=delta_ci.point_estimate,
        delta_ci_lower=delta_ci.ci_lower,
        delta_ci_upper=delta_ci.ci_upper,
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        n_tasks=len(clean_arr),
        degenerate=delta_ci.degenerate,
    )
