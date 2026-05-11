"""Live-skipped integration test for the LangGraph adapter.

Per ``docs/WEEK_2.md`` §Thursday item 4: a trivial calculator-tool
LangGraph agent runs ``pilot_001`` end-to-end. Marked
``@pytest.mark.live``; CI excludes the ``live`` marker per the existing
pattern in :mod:`tests.test_live_integration`.

Run locally with::

    uv run pytest tests/test_adapter_langgraph.py -m live

Requires:

* ``ANTHROPIC_API_KEY`` exported in the environment.
* The ``langgraph`` optional extra: ``uv pip install 'steadfast[langgraph]'``.
* The Anthropic chat-model integration package (not a Steadfast
  optional extra, since the adapter itself doesn't care which provider
  the graph uses internally): ``uv pip install langchain-anthropic``.

The test skips cleanly when any of those are missing, so the live-test
marker behaves the way ``tests.test_live_integration`` does for the
direct-SDK clients.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path

import pytest

from steadfast.agent import Task

PILOT_PATH = Path(__file__).parent.parent / "benchmarks" / "customer_support" / "pilot_001.json"


def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        pytest.skip(f"skipping live test; missing env: {missing}")


def _require_packages(*names: str) -> None:
    missing = [n for n in names if importlib.util.find_spec(n) is None]
    if missing:
        pytest.skip(
            f"skipping live test; missing packages: {missing}. "
            f"Install with: uv pip install 'steadfast[langgraph]' {' '.join(missing)}"
        )


@pytest.mark.live
def test_langgraph_adapter_pilot_with_calculator_tool() -> None:
    """End-to-end: a calculator-tool LangGraph agent runs pilot_001.

    Builds a minimal LangChain react-style agent with one tool
    (``add(a, b)``) and a Claude backend. The customer-support pilot
    task does not require the calculator, so the agent typically
    produces an answer without invoking the tool — but the trajectory
    contract from ADR-0006 §A is exercised either way (the empty-
    trajectory case is just as important to verify as the populated
    one).
    """
    _require_env("ANTHROPIC_API_KEY")
    _require_packages("langgraph", "langchain_core", "langchain_anthropic")

    # The modern (langchain 1.0+) API. Imported lazily because the
    # symbol's existence is gated on the langchain version present in
    # the user's environment.
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    from steadfast.adapters.langgraph import LangGraphAdapter

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    # Use a Haiku-class model to keep the live cost minimal; matches the
    # SimplePromptingAgent live tests' model choice.
    graph = create_agent(
        model="anthropic:claude-haiku-4-5-20251001",
        tools=[add],
    )
    adapter = LangGraphAdapter(graph)
    task = Task.model_validate_json(PILOT_PATH.read_text())

    response = asyncio.run(adapter.arun(task))

    # Core contract assertions per ADR-0006 §A.
    assert response.answer
    assert isinstance(response.trajectory, list)  # not None, even if empty
    assert response.metadata.get("adapter") == "langgraph"
    # The adapter records the message count and the tool-call count;
    # both are >= 0 and consistent with each other.
    n_messages = int(response.metadata.get("n_messages", 0))
    n_tool_calls = int(response.metadata.get("n_tool_calls", 0))
    assert n_messages >= 1, "graph must have produced at least one message"
    assert n_tool_calls == len(response.trajectory)


def test_langgraph_adapter_requires_extra_when_missing() -> None:
    """Construction must fail loudly when the langgraph extra is not installed.

    Skipped when ``langgraph`` is installed (the failure-mode test is
    only meaningful in the no-extra environment); CI running without
    the extra will execute this branch, and a contributor running with
    the extra installed sees a clean skip.
    """
    if importlib.util.find_spec("langgraph") is not None:
        pytest.skip("langgraph is installed; ImportError gate cannot be exercised")

    from steadfast.adapters.langgraph import LangGraphAdapter

    with pytest.raises(ImportError, match="steadfast\\[langgraph\\]"):
        LangGraphAdapter(graph=object())  # type: ignore[arg-type]
