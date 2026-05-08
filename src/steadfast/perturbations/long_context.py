"""Long-context degradation perturbation.

Per ``docs/METHODOLOGY.md`` §2.4: pad input with neutral filler to reach
context lengths of 4k, 16k, 64k, 128k tokens. Report success-rate at each
length as a curve, plus the slope coefficient from a logistic fit.

Implementation in **week 2**. Stub on Monday.
"""

from __future__ import annotations
