"""Ensemble judge — 3 different judge models, majority vote.

Used for high-stakes metrics (catastrophic-failure trap cases, refusal
accuracy) where the residual judge bias of a single LLM-as-judge is most
costly. See ``docs/METHODOLOGY.md`` §"Known limitations and threats to
validity".

Inspired by Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena" (2023, https://arxiv.org/abs/2306.05685).

Promotion to default judge for *all* rubric paths is the v0.2 path documented
in ``docs/adr/0001-infrastructure-model.md``.

Implementation late week 1 / week 2.
"""

from __future__ import annotations
