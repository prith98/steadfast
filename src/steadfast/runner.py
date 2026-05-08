"""Async N-rep executor with SQLite checkpointing.

Per ``docs/METHODOLOGY.md`` §"Multi-run by default", every task is executed
N=10 times per ``(model, framework)`` configuration. The runner owns rep-level
parallelism (bounded by the model client's semaphore), retry orchestration
(delegated to the client layer), and per-task SQLite checkpointing so a
crashed run resumes cleanly.

Per ADR-0002, ``run_id`` is a deterministic SHA-256 of canonicalized inputs;
identical configurations produce identical IDs, so resumption is automatic.
Failed reps are not auto-retried (failures are signal); a future
``--retry-failed`` flag handles deliberate re-runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from steadfast import __version__
from steadfast.agent import Agent, AgentResponse, Task
from steadfast.storage.sqlite import CheckpointStore, RepRow


class RepStatus(StrEnum):
    """Lifecycle status for a single rep within a run."""

    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    FAILED = "failed"


class RepRecord(BaseModel):
    """One repetition of a task within a run.

    Pydantic-typed counterpart to :class:`steadfast.storage.sqlite.RepRow`.
    The runner converts between the two; storage is the dumb persistence layer.
    """

    run_id: str
    task_id: str
    rep_idx: int
    status: RepStatus
    response: AgentResponse | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cost_usd: Decimal = Field(default=Decimal("0"))


class RunResult(BaseModel):
    """Final result of executing a task N times.

    Persisted to ``<output>/<task_id>.json`` by the CLI.
    """

    run_id: str
    task: Task
    reps: list[RepRecord]
    package_version: str = __version__


def compute_run_id(
    *,
    task: Task,
    agent_class: str,
    model: str,
    reps: int,
    package_version: str,
) -> str:
    """Deterministic 16-hex-char run identifier.

    Identical inputs produce identical IDs, which is how the runner detects
    a resumable prior run. The full task content (not just ``task.id``) is
    hashed so editing a task's text invalidates the cached run.
    """
    canonical_task = task.model_dump_json()
    inputs = {
        "task_id": task.id,
        "task_content_sha256": hashlib.sha256(canonical_task.encode()).hexdigest(),
        "agent_class": agent_class,
        "model": model,
        "reps": reps,
        "package_version": package_version,
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _rep_row_to_record(row: RepRow) -> RepRecord:
    return RepRecord(
        run_id=row.run_id,
        task_id=row.task_id,
        rep_idx=row.rep_idx,
        status=RepStatus(row.status),
        response=(
            AgentResponse.model_validate_json(row.response_json) if row.response_json else None
        ),
        error=row.error,
        started_at=(datetime.fromisoformat(row.started_at) if row.started_at else None),
        completed_at=(datetime.fromisoformat(row.completed_at) if row.completed_at else None),
        cost_usd=Decimal(row.cost_usd or "0"),
    )


async def run_task(
    *,
    agent: Agent,
    task: Task,
    reps: int,
    model: str,
    checkpoint_path: str | Path,
) -> RunResult:
    """Execute ``task`` ``reps`` times against ``agent``, persisting checkpoints.

    Resumes any prior incomplete run with the same ``run_id`` automatically.
    Returns a :class:`RunResult` with one :class:`RepRecord` per rep_idx.
    """
    run_id = compute_run_id(
        task=task,
        agent_class=agent.__class__.__name__,
        model=model,
        reps=reps,
        package_version=__version__,
    )

    store = CheckpointStore(checkpoint_path)
    await store.init()

    manifest = {
        "run_id": run_id,
        "task_id": task.id,
        "task_content_sha256": hashlib.sha256(task.model_dump_json().encode()).hexdigest(),
        "agent_class": agent.__class__.__name__,
        "model": model,
        "reps": reps,
        "package_version": __version__,
    }
    await store.upsert_run(
        run_id=run_id,
        manifest_json=json.dumps(manifest, sort_keys=True),
        started_at=_now_iso(),
    )

    existing = {r.rep_idx: r for r in await store.list_reps_for_run(run_id) if r.task_id == task.id}

    pending_indices: list[int] = []
    for i in range(reps):
        existing_row = existing.get(i)
        if existing_row is None or existing_row.status in {
            RepStatus.PENDING.value,
            RepStatus.IN_FLIGHT.value,
        }:
            await store.upsert_rep(
                run_id=run_id,
                task_id=task.id,
                rep_idx=i,
                status=RepStatus.PENDING.value,
                response_json=None,
                error=None,
                started_at=None,
                completed_at=None,
                cost_usd="0",
            )
            pending_indices.append(i)

    async def execute_rep(rep_idx: int) -> None:
        await store.upsert_rep(
            run_id=run_id,
            task_id=task.id,
            rep_idx=rep_idx,
            status=RepStatus.IN_FLIGHT.value,
            response_json=None,
            error=None,
            started_at=_now_iso(),
            completed_at=None,
            cost_usd="0",
        )
        try:
            response = await agent.arun(task)
        except Exception as exc:
            await store.upsert_rep(
                run_id=run_id,
                task_id=task.id,
                rep_idx=rep_idx,
                status=RepStatus.FAILED.value,
                response_json=None,
                error=f"{type(exc).__name__}: {exc}",
                started_at=None,  # COALESCE preserves prior started_at
                completed_at=_now_iso(),
                cost_usd="0",
            )
            return

        cost = str(response.cost_usd) if response.cost_usd is not None else "0"
        await store.upsert_rep(
            run_id=run_id,
            task_id=task.id,
            rep_idx=rep_idx,
            status=RepStatus.COMPLETED.value,
            response_json=response.model_dump_json(),
            error=None,
            started_at=None,  # preserved by COALESCE
            completed_at=_now_iso(),
            cost_usd=cost,
        )

    if pending_indices:
        await asyncio.gather(*(execute_rep(i) for i in pending_indices))

    final_rows = sorted(
        (r for r in await store.list_reps_for_run(run_id) if r.task_id == task.id),
        key=lambda r: r.rep_idx,
    )
    final_records = [_rep_row_to_record(r) for r in final_rows]

    return RunResult(run_id=run_id, task=task, reps=final_records)
