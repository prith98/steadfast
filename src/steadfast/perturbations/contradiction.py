"""Contradiction perturbation — corrupt tool outputs at probability 0.3.

Per ``docs/METHODOLOGY.md`` §2.3: when the agent calls a tool, return a
contradictory or partially-corrupted response with p=0.3. Measure whether the
agent (a) detects the contradiction, (b) retries or escalates, (c)
hallucinates a coherent answer. Reported as a **3-way categorical** —
collapsing this to a scalar would lose the most interesting signal.

Implementation in **week 2**. Stub on Monday.
"""

from __future__ import annotations
