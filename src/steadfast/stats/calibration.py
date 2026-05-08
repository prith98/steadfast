"""Calibration math — Brier score and Expected Calibration Error.

These are the *statistical primitives* used by ``steadfast.metrics.calibration``;
keeping them here lets the metric module focus on data plumbing and reporting
while this module owns the math.

* **Brier score**: ``mean((p - y) ** 2)``. Standard reference: Brier (1950).
* **ECE with equal-mass bins**: 15 bins of equal sample mass, weighted mean
  absolute gap between bin accuracy and bin mean confidence. Equal-mass
  binning per Nixon et al. 2019, https://arxiv.org/abs/1904.01685; original
  ECE formulation per Guo et al. 2017, https://arxiv.org/abs/1706.04599.

Implementation in ``docs/WEEK_1.md`` §"Friday".
"""

from __future__ import annotations
