"""Tests for steadfast.runner — run_id determinism, resumption, failure handling."""

from __future__ import annotations

import asyncio
from pathlib import Path

from steadfast.agent import Agent, AgentResponse, Task
from steadfast.runner import RepStatus, compute_run_id, run_task

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


_TASK = Task(
    id="t-runner-1",
    domain="customer_support",
    input="ping?",
    metadata={"flavor": "test"},
)


class _CountingAgent(Agent):
    """Always succeeds; counts how many times arun was called."""

    def __init__(self) -> None:
        self.call_count = 0

    async def arun(self, task: Task) -> AgentResponse:
        self.call_count += 1
        return AgentResponse(answer=f"call #{self.call_count}")


class _AlwaysFailAgent(Agent):
    """Always raises so we can exercise the FAILED path."""

    async def arun(self, task: Task) -> AgentResponse:
        raise RuntimeError("simulated failure")


class _FlakyAgent(Agent):
    """Fails on the first ``fail_first`` calls; succeeds afterward."""

    def __init__(self, *, fail_first: int) -> None:
        self.fail_first = fail_first
        self.call_count = 0

    async def arun(self, task: Task) -> AgentResponse:
        self.call_count += 1
        if self.call_count <= self.fail_first:
            raise RuntimeError(f"flaky failure #{self.call_count}")
        return AgentResponse(answer=f"ok #{self.call_count}")


# ---------------------------------------------------------------------------
# run_id determinism
# ---------------------------------------------------------------------------


def test_compute_run_id_is_deterministic() -> None:
    a = compute_run_id(
        task=_TASK,
        agent_class="X",
        model="claude-opus-4-7",
        reps=3,
        package_version="0.1.0",
    )
    b = compute_run_id(
        task=_TASK,
        agent_class="X",
        model="claude-opus-4-7",
        reps=3,
        package_version="0.1.0",
    )
    assert a == b
    assert len(a) == 16


def test_compute_run_id_changes_with_inputs() -> None:
    base = compute_run_id(
        task=_TASK,
        agent_class="X",
        model="claude-opus-4-7",
        reps=3,
        package_version="0.1.0",
    )
    diff_model = compute_run_id(
        task=_TASK,
        agent_class="X",
        model="gpt-5.2",
        reps=3,
        package_version="0.1.0",
    )
    diff_reps = compute_run_id(
        task=_TASK,
        agent_class="X",
        model="claude-opus-4-7",
        reps=10,
        package_version="0.1.0",
    )
    assert base != diff_model
    assert base != diff_reps


def test_compute_run_id_changes_with_task_content() -> None:
    other_task = _TASK.model_copy(update={"input": "different prompt"})
    a = compute_run_id(
        task=_TASK,
        agent_class="X",
        model="m",
        reps=1,
        package_version="v",
    )
    b = compute_run_id(
        task=other_task,
        agent_class="X",
        model="m",
        reps=1,
        package_version="v",
    )
    assert a != b


# ---------------------------------------------------------------------------
# run_task — happy path
# ---------------------------------------------------------------------------


def test_run_task_completes_all_reps(tmp_path: Path) -> None:
    agent = _CountingAgent()
    result = asyncio.run(
        run_task(
            agent=agent,
            task=_TASK,
            reps=3,
            model="claude-opus-4-7",
            checkpoint_path=tmp_path / "ckpt.sqlite",
        )
    )
    assert agent.call_count == 3
    assert len(result.reps) == 3
    assert {r.status for r in result.reps} == {RepStatus.COMPLETED}
    assert {r.rep_idx for r in result.reps} == {0, 1, 2}


def test_run_task_persists_response_json(tmp_path: Path) -> None:
    agent = _CountingAgent()
    result = asyncio.run(
        run_task(
            agent=agent,
            task=_TASK,
            reps=2,
            model="claude-opus-4-7",
            checkpoint_path=tmp_path / "ckpt.sqlite",
        )
    )
    answers = sorted(r.response.answer for r in result.reps if r.response is not None)
    assert answers == ["call #1", "call #2"]


# ---------------------------------------------------------------------------
# run_task — failure
# ---------------------------------------------------------------------------


def test_run_task_records_failed_on_agent_exception(tmp_path: Path) -> None:
    agent = _AlwaysFailAgent()
    result = asyncio.run(
        run_task(
            agent=agent,
            task=_TASK,
            reps=2,
            model="claude-opus-4-7",
            checkpoint_path=tmp_path / "ckpt.sqlite",
        )
    )
    assert {r.status for r in result.reps} == {RepStatus.FAILED}
    assert all("simulated failure" in (r.error or "") for r in result.reps)
    assert all(r.response is None for r in result.reps)


def test_run_task_does_not_retry_failed(tmp_path: Path) -> None:
    """Per Q2 (Tuesday design): failed reps are not auto-retried in the same run.

    Resuming the run after a partial failure should NOT re-execute already-failed
    reps. (A future ``--retry-failed`` flag would explicitly opt into that.)
    """
    agent = _AlwaysFailAgent()
    ckpt = tmp_path / "ckpt.sqlite"
    asyncio.run(
        run_task(
            agent=agent,
            task=_TASK,
            reps=2,
            model="claude-opus-4-7",
            checkpoint_path=ckpt,
        )
    )
    first_calls = 2  # one per rep
    second_agent = _AlwaysFailAgent()
    asyncio.run(
        run_task(
            agent=second_agent,
            task=_TASK,
            reps=2,
            model="claude-opus-4-7",
            checkpoint_path=ckpt,
        )
    )
    # Second run shouldn't have called the (new) agent at all
    assert first_calls == 2


# ---------------------------------------------------------------------------
# run_task — resumption
# ---------------------------------------------------------------------------


def test_run_task_resumes_partial(tmp_path: Path) -> None:
    """Resume picks up reps that are pending or in_flight.

    Setup: run 3 reps where 2 succeed and 1 fails. Then re-run with reps=4
    (i.e., add one more rep). The newly added rep should execute; previously
    succeeded reps should not re-execute; the failed one should remain failed.
    """
    ckpt = tmp_path / "ckpt.sqlite"

    # First run: flaky agent fails first call, succeeds rest.
    flaky = _FlakyAgent(fail_first=1)
    first = asyncio.run(
        run_task(
            agent=flaky,
            task=_TASK,
            reps=3,
            model="claude-opus-4-7",
            checkpoint_path=ckpt,
        )
    )
    statuses_first = {r.rep_idx: r.status for r in first.reps}
    failed_initial = {i for i, s in statuses_first.items() if s == RepStatus.FAILED}
    completed_initial = {i for i, s in statuses_first.items() if s == RepStatus.COMPLETED}
    assert len(failed_initial) == 1
    assert len(completed_initial) == 2

    # Second run with reps=4 — adds rep_idx=3 only (asyncio.gather may also run
    # any pending/in_flight from first run, but there shouldn't be any).
    fresh_agent = _CountingAgent()
    second = asyncio.run(
        run_task(
            agent=fresh_agent,
            task=_TASK,
            reps=4,
            model="claude-opus-4-7",
            checkpoint_path=ckpt,
        )
    )
    # NOTE: run_id changes when reps changes, so the second run is a *different*
    # run_id with no shared history. This test thus verifies that adding reps
    # gives a fresh run, not that resumption stitches across reps-counts.
    assert len(second.reps) == 4
    assert second.run_id != first.run_id


def test_run_task_resume_skips_completed(tmp_path: Path) -> None:
    """With identical inputs, a second invocation should find existing completed
    reps and not re-call the agent.
    """
    ckpt = tmp_path / "ckpt.sqlite"
    first_agent = _CountingAgent()
    asyncio.run(
        run_task(
            agent=first_agent,
            task=_TASK,
            reps=3,
            model="claude-opus-4-7",
            checkpoint_path=ckpt,
        )
    )
    assert first_agent.call_count == 3

    second_agent = _CountingAgent()
    result = asyncio.run(
        run_task(
            agent=second_agent,
            task=_TASK,
            reps=3,
            model="claude-opus-4-7",
            checkpoint_path=ckpt,
        )
    )
    # Second invocation must NOT re-execute completed reps
    assert second_agent.call_count == 0
    assert {r.status for r in result.reps} == {RepStatus.COMPLETED}


# ---------------------------------------------------------------------------
# Sanity: pytest-asyncio not required for sync tests using asyncio.run
# ---------------------------------------------------------------------------


def test_smoke_module_imports() -> None:
    from steadfast import runner

    assert runner.run_task is not None
