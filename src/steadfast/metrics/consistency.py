"""Consistency dimension — output, trajectory, and format consistency.

Per ``docs/METHODOLOGY.md`` §1:

* **Output consistency** (§1.1): K=5 paraphrases, pairwise (embedding cosine
  + LLM-judge 0-4 Likert rubric); mean rubric normalized to [0, 1] with BCa
  bootstrap CI.
* **Trajectory consistency** (§1.2): N=10 same-input runs; ``1 - mean(normalized
  Levenshtein)`` over tool-name sequence + agentevals ``superset`` arg matching.
* **Format consistency** (§1.3): schema-validation pass-rate, Wilson 95% CI.

Implementation in ``docs/WEEK_1.md`` §"Thursday".
"""

from __future__ import annotations
