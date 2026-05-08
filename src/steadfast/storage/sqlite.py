"""SQLite-backed checkpoint and run-result store.

The store is intentionally a "dumb persistence" layer — it takes/returns
strings, ints, and JSON blobs, with no knowledge of higher-level Pydantic
models. :mod:`steadfast.runner` owns ``RepRecord`` ↔ row conversion. This
keeps storage decoupled from runner data types and avoids circular imports.

Per ``docs/METHODOLOGY.md`` §"Reproducibility", every benchmark run writes a
``runs`` row with a frozen manifest plus one ``reps`` row per (task, rep_idx)
combination. Crashed runs resume by re-executing reps with status ``pending``
or ``in_flight``; ``completed`` reps are preserved.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    started_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reps (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    rep_idx INTEGER NOT NULL,
    status TEXT NOT NULL,
    response_json TEXT,
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    cost_usd TEXT NOT NULL DEFAULT '0',
    PRIMARY KEY (run_id, task_id, rep_idx),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS reps_status_idx ON reps(run_id, status);
"""


@dataclass(frozen=True, slots=True)
class RepRow:
    """One row from the ``reps`` table.

    Carries primitive types only — the runner converts to/from ``RepRecord``.
    """

    run_id: str
    task_id: str
    rep_idx: int
    status: str
    response_json: str | None
    error: str | None
    started_at: str | None
    completed_at: str | None
    cost_usd: str


class CheckpointStore:
    """aiosqlite-backed persistence for runs and per-rep status.

    Use :meth:`init` once before the first read/write to create the schema.
    The store opens a fresh connection per operation; an internal
    ``asyncio.Lock`` serializes writes so concurrent ``asyncio.gather`` fanout
    in the runner cannot trigger SQLite's "database is locked" error. Reads
    are not lock-protected (SQLite handles concurrent readers natively).
    """

    def __init__(self, path: str | Path) -> None:
        # Accept ":memory:" for in-process testing or a filesystem path.
        self.path = str(path)
        # Lazy-bound to whichever event loop first acquires it.
        self._write_lock = asyncio.Lock()

    async def init(self) -> None:
        async with self._write_lock, aiosqlite.connect(self.path) as db:
            # WAL mode permits concurrent readers + a single writer and is
            # persistent (set once, applies to the database file).
            # busy_timeout is per-connection; we set it on every open below.
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            await db.executescript(_SCHEMA)
            await db.commit()

    async def upsert_run(self, *, run_id: str, manifest_json: str, started_at: str) -> None:
        async with self._write_lock, aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute(
                """
                INSERT INTO runs (run_id, manifest_json, started_at)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    manifest_json = excluded.manifest_json
                """,
                (run_id, manifest_json, started_at),
            )
            await db.commit()

    async def upsert_rep(
        self,
        *,
        run_id: str,
        task_id: str,
        rep_idx: int,
        status: str,
        response_json: str | None,
        error: str | None,
        started_at: str | None,
        completed_at: str | None,
        cost_usd: str,
    ) -> None:
        async with self._write_lock, aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute(
                """
                INSERT INTO reps (
                    run_id, task_id, rep_idx, status,
                    response_json, error, started_at, completed_at, cost_usd
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, task_id, rep_idx) DO UPDATE SET
                    status = excluded.status,
                    response_json = excluded.response_json,
                    error = excluded.error,
                    started_at = COALESCE(excluded.started_at, reps.started_at),
                    completed_at = excluded.completed_at,
                    cost_usd = excluded.cost_usd
                """,
                (
                    run_id,
                    task_id,
                    rep_idx,
                    status,
                    response_json,
                    error,
                    started_at,
                    completed_at,
                    cost_usd,
                ),
            )
            await db.commit()

    async def list_reps_for_run(self, run_id: str) -> list[RepRow]:
        async with (
            aiosqlite.connect(self.path) as db,
            db.execute(
                """
                SELECT run_id, task_id, rep_idx, status, response_json,
                       error, started_at, completed_at, cost_usd
                FROM reps
                WHERE run_id = ?
                ORDER BY task_id, rep_idx
                """,
                (run_id,),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [_row_from_tuple(row) for row in rows]

    async def list_pending_reps(self, run_id: str) -> list[RepRow]:
        """Return reps with ``status IN ('pending', 'in_flight')`` — those that
        a resumed run should re-execute.
        """
        async with (
            aiosqlite.connect(self.path) as db,
            db.execute(
                """
                SELECT run_id, task_id, rep_idx, status, response_json,
                       error, started_at, completed_at, cost_usd
                FROM reps
                WHERE run_id = ? AND status IN ('pending', 'in_flight')
                ORDER BY task_id, rep_idx
                """,
                (run_id,),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [_row_from_tuple(row) for row in rows]


def _row_from_tuple(row: Sequence[Any]) -> RepRow:
    return RepRow(
        run_id=str(row[0]),
        task_id=str(row[1]),
        rep_idx=int(row[2]),
        status=str(row[3]),
        response_json=str(row[4]) if row[4] is not None else None,
        error=str(row[5]) if row[5] is not None else None,
        started_at=str(row[6]) if row[6] is not None else None,
        completed_at=str(row[7]) if row[7] is not None else None,
        cost_usd=str(row[8]),
    )
