"""Robustness dimension — typo, distractor, contradiction, long-context.

Per ``docs/METHODOLOGY.md`` §2, each sub-metric reports a **success-rate
delta** (perturbed - clean) with bootstrap CI on the delta itself, not the
endpoints. Contradiction handling is reported as a 3-way categorical (detect /
retry-or-escalate / hallucinate), deliberately not collapsed to a scalar.

Implementation in **week 2** per ``docs/WEEK_1.md``. Stub on Monday.
"""

from __future__ import annotations
