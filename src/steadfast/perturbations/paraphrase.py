"""Paraphrase generation for output-consistency measurement.

Per ``docs/METHODOLOGY.md`` §1.1: K=5 paraphrases of each task input, generated
by ``gpt-5.2`` (per ADR-0001) with the frozen prompt at
``prompts/paraphrase_v1.txt``, temperature 0.7, deterministic seed per task.
A second LLM call validates semantic equivalence to the original; rejected
paraphrases are regenerated up to 3 times. Rejection rate is reported as a
quality metric on each run.

Implementation in ``docs/WEEK_1.md`` §"Thursday".
"""

from __future__ import annotations
