"""Wilson confidence interval for binomial proportions.

Used for any pass-rate / success-rate metric (format consistency,
overconfidence rate, catastrophic-failure rate, refusal calibration cells).
Wilson is the standard-of-care for binomial CIs at small N — see Wilson
(1927); Brown, Cai, DasGupta (2001) for the comparison with Wald and
Clopper-Pearson.

Implementation in ``docs/WEEK_1.md`` §"Thursday/Friday" as needed.
"""

from __future__ import annotations
