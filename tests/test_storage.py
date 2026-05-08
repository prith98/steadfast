"""Tests for steadfast.storage.sqlite — CheckpointStore round-trips and filtering."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from steadfast.storage.sqlite import CheckpointStore


def _async_run(coro: object) -> object:
    return asyncio.get_event_loop().run_until_complete(coro)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> CheckpointStore:
    s = CheckpointStore(tmp_path / "test.sqlite")
    asyncio.run(s.init())
    return s


def test_init_idempotent(tmp_path: Path) -> None:
    s = CheckpointStore(tmp_path / "x.sqlite")
    asyncio.run(s.init())
    asyncio.run(s.init())  # second call must not error


async def _seed_run_and_reps(store: CheckpointStore) -> None:
    await store.upsert_run(
        run_id="r1", manifest_json='{"k":"v"}', started_at="2026-05-08T00:00:00+00:00"
    )
    for i in range(3):
        await store.upsert_rep(
            run_id="r1",
            task_id="t1",
            rep_idx=i,
            status="pending",
            response_json=None,
            error=None,
            started_at=None,
            completed_at=None,
            cost_usd="0",
        )


def test_upsert_run_and_list_reps_empty(store: CheckpointStore) -> None:
    asyncio.run(
        store.upsert_run(
            run_id="r1",
            manifest_json='{"k":"v"}',
            started_at="2026-05-08T00:00:00+00:00",
        )
    )
    rows = asyncio.run(store.list_reps_for_run("r1"))
    assert rows == []


def test_upsert_rep_then_list(store: CheckpointStore) -> None:
    asyncio.run(_seed_run_and_reps(store))
    rows = asyncio.run(store.list_reps_for_run("r1"))
    assert len(rows) == 3
    assert {r.rep_idx for r in rows} == {0, 1, 2}
    assert all(r.status == "pending" for r in rows)


def test_list_pending_filters_by_status(store: CheckpointStore) -> None:
    asyncio.run(_seed_run_and_reps(store))
    # Mark rep 0 completed, rep 1 in_flight, rep 2 stays pending
    asyncio.run(
        store.upsert_rep(
            run_id="r1",
            task_id="t1",
            rep_idx=0,
            status="completed",
            response_json='{"answer":"x"}',
            error=None,
            started_at="2026-05-08T01:00:00+00:00",
            completed_at="2026-05-08T01:00:01+00:00",
            cost_usd="0.001",
        )
    )
    asyncio.run(
        store.upsert_rep(
            run_id="r1",
            task_id="t1",
            rep_idx=1,
            status="in_flight",
            response_json=None,
            error=None,
            started_at="2026-05-08T01:00:00+00:00",
            completed_at=None,
            cost_usd="0",
        )
    )

    pending = asyncio.run(store.list_pending_reps("r1"))
    assert {r.rep_idx for r in pending} == {1, 2}
    assert {r.status for r in pending} == {"pending", "in_flight"}


def test_upsert_preserves_started_at_via_coalesce(store: CheckpointStore) -> None:
    asyncio.run(_seed_run_and_reps(store))
    # First update sets started_at
    asyncio.run(
        store.upsert_rep(
            run_id="r1",
            task_id="t1",
            rep_idx=0,
            status="in_flight",
            response_json=None,
            error=None,
            started_at="2026-05-08T01:00:00+00:00",
            completed_at=None,
            cost_usd="0",
        )
    )
    # Second update passes started_at=None; COALESCE should preserve the earlier value
    asyncio.run(
        store.upsert_rep(
            run_id="r1",
            task_id="t1",
            rep_idx=0,
            status="completed",
            response_json='{"answer":"x"}',
            error=None,
            started_at=None,
            completed_at="2026-05-08T01:00:05+00:00",
            cost_usd="0.001",
        )
    )
    rows = asyncio.run(store.list_reps_for_run("r1"))
    rep0 = next(r for r in rows if r.rep_idx == 0)
    assert rep0.started_at == "2026-05-08T01:00:00+00:00"
    assert rep0.completed_at == "2026-05-08T01:00:05+00:00"
    assert rep0.cost_usd == "0.001"
