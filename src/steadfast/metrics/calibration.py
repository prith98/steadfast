"""Calibration dimension — Brier, ECE, refusal calibration, overconfidence rate.

Per ``docs/METHODOLOGY.md`` §3:

* **Brier score** (§3.2): ``mean((p - y) ** 2)``, BCa bootstrap CI.
* **ECE** (§3.3): 15 **equal-mass** bins (Nixon et al. 2019,
  https://arxiv.org/abs/1904.01685); equal-mass is robust when confidence
  concentrates near 1.0, which it always does for current LLMs.
* **Refusal calibration** (§3.4): {refused | answered} x {hard | normal}
  confusion matrix on the curated unanswerable subset (10% of the benchmark).
* **Overconfidence rate** (§3.5): ``P(incorrect ∧ confidence ≥ 0.9)``,
  Wilson 95% CI.

Verbalized and (where available) logprob-derived confidence are reported
separately — see §3.1 on the Tian et al. 2023 caveat.

Implementation in ``docs/WEEK_1.md`` §"Friday".
"""

from __future__ import annotations
