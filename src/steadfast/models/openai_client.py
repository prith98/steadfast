"""OpenAI client wrapper — GPT family + embeddings.

Used both as a target-model client *and* as the v0.1 benchmark infrastructure
(paraphrase generation, ``text-embedding-3-large`` similarity, default rubric
judge — see ``docs/adr/0001-infrastructure-model.md``).

Implementation in ``docs/WEEK_1.md`` §"Tuesday".
"""

from __future__ import annotations
