"""Live integration tests — gated by ``@pytest.mark.live`` and require API keys.

These tests hit the real provider APIs. They are excluded from CI (which runs
``pytest -m "not slow and not live"``) and from the default local test run.
Run explicitly with::

    uv run pytest tests/test_live_integration.py -m live

Each test skips automatically if its required API key is not set.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from steadfast.agent import SimplePromptingAgent, Task
from steadfast.runner import RepStatus, run_task


def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        pytest.skip(f"skipping live test; missing env: {missing}")


PILOT_PATH = Path(__file__).parent.parent / "benchmarks" / "customer_support" / "pilot_001.json"


@pytest.mark.live
def test_anthropic_pilot_three_reps_end_to_end(tmp_path: Path) -> None:
    """Tuesday DOD — runs pilot_001 against Anthropic for 3 reps."""
    _require_env("ANTHROPIC_API_KEY")

    from steadfast.models.anthropic_client import AnthropicClient

    task = Task.model_validate_json(PILOT_PATH.read_text())
    client = AnthropicClient()
    agent = SimplePromptingAgent(client=client, model="claude-haiku-4-5-20251001")

    result = asyncio.run(
        run_task(
            agent=agent,
            task=task,
            reps=3,
            model="claude-haiku-4-5-20251001",
            checkpoint_path=tmp_path / "live.sqlite",
        )
    )

    assert len(result.reps) == 3
    assert {r.status for r in result.reps} == {RepStatus.COMPLETED}
    # Every completed rep must carry a non-None response and a positive cost.
    for r in result.reps:
        assert r.response is not None
        assert r.response.answer
        assert r.response.cost_usd is not None
        assert r.response.cost_usd > 0
        assert r.response.metadata.get("input_tokens", 0) > 0
        assert r.response.metadata.get("output_tokens", 0) > 0

    # The serialized run.json must round-trip via the same model.
    serialized = result.model_dump_json()
    parsed = json.loads(serialized)
    assert parsed["run_id"] == result.run_id
    assert len(parsed["reps"]) == 3


@pytest.mark.live
def test_openai_pilot_three_reps_end_to_end(tmp_path: Path) -> None:
    """Same shape as the Anthropic test, against OpenAI."""
    _require_env("OPENAI_API_KEY")

    from steadfast.models.openai_client import OpenAIClient

    task = Task.model_validate_json(PILOT_PATH.read_text())
    client = OpenAIClient()
    agent = SimplePromptingAgent(client=client, model="gpt-5-mini")

    result = asyncio.run(
        run_task(
            agent=agent,
            task=task,
            reps=3,
            model="gpt-5-mini",
            checkpoint_path=tmp_path / "live.sqlite",
        )
    )

    assert len(result.reps) == 3
    assert {r.status for r in result.reps} == {RepStatus.COMPLETED}
    for r in result.reps:
        assert r.response is not None
        assert r.response.cost_usd is not None
        assert r.response.cost_usd > 0


@pytest.mark.live
def test_google_pilot_three_reps_end_to_end(tmp_path: Path) -> None:
    """Same shape, against Google Gemini."""
    _require_env("GOOGLE_API_KEY")

    from steadfast.models.google_client import GoogleClient

    task = Task.model_validate_json(PILOT_PATH.read_text())
    client = GoogleClient()
    agent = SimplePromptingAgent(client=client, model="gemini-2.5-flash")

    result = asyncio.run(
        run_task(
            agent=agent,
            task=task,
            reps=3,
            model="gemini-2.5-flash",
            checkpoint_path=tmp_path / "live.sqlite",
        )
    )

    assert len(result.reps) == 3
    assert {r.status for r in result.reps} == {RepStatus.COMPLETED}
    for r in result.reps:
        assert r.response is not None
        assert r.response.cost_usd is not None
        assert r.response.cost_usd > 0
