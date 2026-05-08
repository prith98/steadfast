"""Async N-run executor with checkpointing.

Methodological commitment (``docs/METHODOLOGY.md`` §"Multi-run by default"):
every task is executed N=10 times per ``(model, framework)`` configuration.
The runner owns parallelism, retry, semaphore-bounded provider concurrency,
and per-task SQLite checkpointing so a crashed run resumes cleanly.

Implementation in ``docs/WEEK_1.md`` §"Tuesday".
"""

from __future__ import annotations
