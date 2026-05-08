"""Tests for the public Agent contract (steadfast.agent)."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from pydantic import ValidationError

from steadfast.agent import (
    Agent,
    AgentResponse,
    GroundTruth,
    SimplePromptingAgent,
    Task,
    ToolCall,
)
from steadfast.models.base import (
    BaseModelClient,
    ChatMessage,
    ChatResponse,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Task / GroundTruth
# ---------------------------------------------------------------------------


def test_task_round_trip() -> None:
    task = Task(
        id="t1",
        domain="customer_support",
        input="How long is the return window?",
        ground_truth=GroundTruth(kind="exact", value="30 days"),
        judge="exact_match",
        confidence_suffix="Report your confidence as a probability in [0, 1].",
        metadata={"difficulty": "trivial", "version_int": 1, "weight": 1.5, "live": True},
    )
    payload = task.model_dump_json()
    rebuilt = Task.model_validate_json(payload)
    assert rebuilt == task


def test_task_metadata_rejects_non_scalar() -> None:
    with pytest.raises(ValidationError):
        Task(
            id="t1",
            domain="d",
            input="x",
            metadata={"nested": {"foo": "bar"}},  # type: ignore[dict-item]
        )


def test_task_judge_default_is_rubric() -> None:
    t = Task(id="t1", domain="d", input="x")
    assert t.judge == "rubric"


def test_ground_truth_is_frozen() -> None:
    gt = GroundTruth(kind="exact", value="42")
    with pytest.raises(ValidationError):
        gt.value = "43"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


def test_agent_response_defaults() -> None:
    r = AgentResponse(answer="ok")
    assert r.confidence is None
    assert r.trajectory == []
    assert r.raw_output is None
    assert r.cost_usd is None
    assert r.metadata == {}


def test_agent_response_confidence_validates_range() -> None:
    AgentResponse(answer="ok", confidence=0.0)
    AgentResponse(answer="ok", confidence=1.0)
    with pytest.raises(ValidationError):
        AgentResponse(answer="ok", confidence=-0.01)
    with pytest.raises(ValidationError):
        AgentResponse(answer="ok", confidence=1.01)


def test_agent_response_round_trip_with_decimal_cost() -> None:
    r = AgentResponse(
        answer="ok",
        confidence=0.7,
        trajectory=[ToolCall(name="search", args={"q": "foo"}, result="bar")],
        cost_usd=Decimal("0.0042"),
        metadata={"model": "claude-opus-4-7"},
    )
    rebuilt = AgentResponse.model_validate_json(r.model_dump_json())
    assert rebuilt.cost_usd == Decimal("0.0042")
    assert rebuilt.trajectory[0].name == "search"
    assert rebuilt.confidence == 0.7


# ---------------------------------------------------------------------------
# Agent ABC + SimplePromptingAgent
# ---------------------------------------------------------------------------


class _EchoAgent(Agent):
    async def arun(self, task: Task) -> AgentResponse:
        return AgentResponse(answer=f"echo: {task.input}")


def test_agent_run_wraps_arun_synchronously() -> None:
    agent = _EchoAgent()
    task = Task(id="t1", domain="d", input="hello")
    result = agent.run(task)
    assert result.answer == "echo: hello"


def test_agent_abc_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# SimplePromptingAgent uses the client.acomplete and forwards usage/cost
# ---------------------------------------------------------------------------


class _StubClient(BaseModelClient):
    """Records the prompt it received; returns a canned response."""

    def __init__(self, *, canned_text: str = "stub answer") -> None:
        super().__init__(max_concurrent=1, max_retries=1)
        self.canned_text = canned_text
        self.last_messages: list[ChatMessage] | None = None

    async def _achat_provider(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: object,
    ) -> ChatResponse:
        del kwargs
        self.last_messages = messages
        return ChatResponse(
            text=self.canned_text,
            usage=TokenUsage(input_tokens=12, output_tokens=34),
            cost_usd=Decimal("0.0001"),
            model=model,
            finish_reason="stop",
        )


def test_simple_prompting_agent_concatenates_confidence_suffix() -> None:
    client = _StubClient()
    agent = SimplePromptingAgent(client=client, model="claude-opus-4-7")
    task = Task(
        id="t1",
        domain="d",
        input="What is 2+2?",
        confidence_suffix="State your confidence as a probability.",
    )
    response = asyncio.run(agent.arun(task))

    assert client.last_messages is not None
    assert len(client.last_messages) == 1
    assert "What is 2+2?" in client.last_messages[0].content
    assert "confidence" in client.last_messages[0].content.lower()
    assert response.answer == "stub answer"
    assert response.cost_usd == Decimal("0.0001")
    assert response.metadata["input_tokens"] == 12
    assert response.metadata["output_tokens"] == 34
    assert response.confidence is None  # Tuesday: parser lands Friday


def test_simple_prompting_agent_skips_suffix_when_none() -> None:
    client = _StubClient()
    agent = SimplePromptingAgent(client=client, model="claude-opus-4-7")
    task = Task(id="t1", domain="d", input="bare prompt")
    asyncio.run(agent.arun(task))
    assert client.last_messages is not None
    assert client.last_messages[0].content == "bare prompt"
