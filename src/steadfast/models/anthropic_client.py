"""Anthropic client wrapper — Claude family.

Implementation in ``docs/WEEK_1.md`` §"Tuesday". Note: logprob-derived
confidence is limited on Anthropic's API; calibration §3.1 in
``docs/METHODOLOGY.md`` documents the asymmetry across providers.
"""

from __future__ import annotations
