"""Agent ABC and Pydantic models for the public Steadfast contract.

This module defines the surface that user-supplied agents must conform to,
plus a built-in :class:`SimplePromptingAgent` used by ``steadfast bench``
when no ``--agent`` is given.

See ``docs/adr/0002-v01-core-abstractions.md`` for the design rationale,
including the Q1 confidence-elicitation contract and the Q5 metadata typing
policy.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from steadfast.models.base import BaseModelClient


# Scalar union for metadata bags. Per Q5 (project kickoff), the public surface
# forbids ``dict[str, Any]``; richer types belong on AgentResponse subclasses.
MetadataValue: TypeAlias = str | int | float | bool


class GroundTruth(BaseModel):
    """Reference answer used by judges to score an agent response.

    ``kind="exact"`` is consumed by :class:`steadfast.judges.exact_match.ExactMatchJudge`
    after canonicalization. ``kind="rubric"`` carries criteria consumed by
    :class:`steadfast.judges.rubric.RubricJudge` (LLM-as-judge with a frozen rubric).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["exact", "rubric"]
    value: str


class Task(BaseModel):
    """A single benchmark task.

    The harness sets ``confidence_suffix`` to the frozen prompt suffix that
    elicits a probability estimate. Per Q1 (project kickoff), the primary
    contract requires the agent to concatenate the suffix into its prompt and
    populate :attr:`AgentResponse.confidence` itself.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    domain: str
    input: str
    ground_truth: GroundTruth | None = None
    judge: Literal["exact_match", "rubric"] = "rubric"
    confidence_suffix: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """One tool invocation in an agent's trajectory.

    Consumed by trajectory-consistency measurement
    (``docs/METHODOLOGY.md`` §1.2) — ``args`` uses the metadata scalar union so
    arguments can be serialized and compared structurally.
    """

    name: str
    args: dict[str, MetadataValue] = Field(default_factory=dict)
    result: str | None = None


class AgentResponse(BaseModel):
    """Output of one agent execution.

    Field semantics:

    * ``confidence``: required for the calibration dimension. Agents that
      cannot or do not report confidence leave it ``None``; calibration
      metrics skip those reps with a logged warning rather than failing.
    * ``trajectory``: may be empty for agents that do not expose tool calls.
      Trajectory consistency returns N/A for empty trajectories. Open
      methodology question (see auto-memory) on toolless-agent contract.
    * ``raw_output``: verbatim model response, kept for the post-hoc
      confidence parser and for debug spans.
    * ``cost_usd``: best-effort. Built-in agents that wrap a Steadfast
      :class:`~steadfast.models.base.BaseModelClient` populate this; user
      agents may leave it ``None``.
    """

    answer: str
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    trajectory: list[ToolCall] = Field(default_factory=list)
    raw_output: str | None = None
    cost_usd: Decimal | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class Agent(ABC):
    """Abstract base for benchmarkable agents.

    Subclasses implement :meth:`arun`. :meth:`run` is a synchronous convenience
    wrapper; the harness itself always calls :meth:`arun` directly.
    """

    @abstractmethod
    async def arun(self, task: Task) -> AgentResponse:
        """Execute ``task`` and return a structured response."""

    def run(self, task: Task) -> AgentResponse:
        return asyncio.run(self.arun(task))


class SimplePromptingAgent(Agent):
    """Direct single-shot agent: send the task input as one user message.

    Used by ``steadfast bench`` as the default when no ``--agent`` is given,
    so the harness runs end-to-end without forcing every user to write an
    Agent subclass on day one. Not intended as a strong reference baseline.

    Confidence parsing lands Friday (``docs/WEEK_1.md`` §"Friday"); Tuesday's
    version always returns ``confidence=None`` and lets the calibration
    pipeline log a warning.
    """

    def __init__(self, *, client: BaseModelClient, model: str) -> None:
        self._client = client
        self._model = model

    async def arun(self, task: Task) -> AgentResponse:
        prompt = task.input
        if task.confidence_suffix:
            prompt = f"{task.input}\n\n{task.confidence_suffix}"
        response = await self._client.acomplete(prompt, model=self._model)
        return AgentResponse(
            answer=response.text.strip(),
            confidence=None,
            trajectory=[],
            raw_output=response.text,
            cost_usd=response.cost_usd,
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "model": response.model,
                "finish_reason": response.finish_reason or "",
            },
        )
