"""Bootstrap confidence intervals — wraps ``scipy.stats.bootstrap`` with the
Steadfast default config.

Per ``docs/METHODOLOGY.md`` §"Statistical conventions": **95% CI, BCa method,
10,000 resamples**. The wrapper is the canonical entry point — no other
module is allowed to call ``scipy.stats.bootstrap`` directly, and we never
hand-roll bootstrap code (``CLAUDE.md`` "Tech stack").

References:
* Efron & Tibshirani, *An Introduction to the Bootstrap* (1993).
* SciPy docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html

Implementation in ``docs/WEEK_1.md`` §"Thursday".
"""

from __future__ import annotations
