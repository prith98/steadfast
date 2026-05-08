"""SQLite-backed checkpoint and run-result store.

Per ``docs/METHODOLOGY.md`` §"Reproducibility" and ``docs/WEEK_1.md``
§"Tuesday" — a crashed benchmark run resumes by re-reading the per-task
checkpoint table; final results land in a structured ``runs`` table that
the reporting module reads from.

Implementation Tuesday. Uses ``aiosqlite`` (added Tuesday, not in Monday's
dep set).
"""

from __future__ import annotations
