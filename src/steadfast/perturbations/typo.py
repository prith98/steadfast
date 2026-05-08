"""Typo perturbation — character-level noise.

Per ``docs/METHODOLOGY.md`` §2.1: 5% per-character noise rate, with a
constraint that no individual word exceeds 25% character corruption (so no
word is rendered fully unrecognizable). Deterministic given a seed derived
from the task ID.

Inspired by NLP robustness literature (CheckList; Ribeiro et al. 2020).

Implementation in **week 2**. Stub on Monday.
"""

from __future__ import annotations
