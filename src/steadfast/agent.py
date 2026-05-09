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
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from steadfast.perturbations.confidence import parse_verbalized_confidence

if TYPE_CHECKING:
    from steadfast.models.base import BaseModelClient, ChatResponse


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

    ``output_schema`` is an optional JSON-Schema string consumed by the
    format-consistency metric (``docs/METHODOLOGY.md`` §1.3). Tasks
    without a schema return ``N/A`` from format consistency. Stored as a
    string rather than ``dict[str, Any]`` to keep the public contract
    Pydantic-typed (ADR-0002 §A.3 / ADR-0004 §D).

    ``difficulty`` drives refusal calibration (METHODOLOGY §3.4 / ADR-0005
    §E). Tasks marked ``"hard"`` either have no determinable ground truth
    from the prompt or are deliberately under-specified; refusal or
    hedging is the correct behavior. Default ``"normal"`` so existing
    tasks pass through unchanged.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    domain: str
    input: str
    ground_truth: GroundTruth | None = None
    judge: Literal["exact_match", "rubric"] = "rubric"
    confidence_suffix: str | None = None
    output_schema: str | None = None
    difficulty: Literal["normal", "hard"] = "normal"
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

    * ``confidence``: verbalized confidence parsed from the agent's
      structured ANSWER/CONFIDENCE tail (ADR-0005 §B). Agents that cannot or
      do not report confidence leave it ``None``; calibration metrics skip
      those reps with a logged warning rather than failing.
    * ``refused``: True iff the agent emitted the literal ``REFUSE`` token
      on the answer line of the elicitation tail (METHODOLOGY §3.4 / ADR-0005
      §E). Refusal calibration consumes this; Brier / ECE / overconfidence
      pools exclude refused reps.
    * ``logprob_avg``: mean per-token logprob over the model's response,
      where the provider's API exposes it (OpenAI yes; Anthropic, Google
      ``None`` for v0.1 per ADR-0005 §A). Calibration's secondary
      logprob-derived column applies the ``exp(logprob_avg)`` transform
      (Kadavath et al. 2022).
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
    refused: bool = False
    logprob_avg: float | None = None
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

    When ``Task.confidence_suffix`` is set, the agent appends the suffix to
    the prompt, requests logprobs from clients that support them
    (per ADR-0005 §A — OpenAI populates, others return None), and parses the
    response with :func:`steadfast.perturbations.confidence.parse_verbalized_confidence`.
    On a parse failure the agent retries once with a stricter "your previous
    output did not include a CONFIDENCE: line" reminder; on a second failure
    the rep stays COMPLETED with ``confidence=None`` and the calibration
    metric layer skips it (per ADR-0002 §A.2 / ADR-0005 §C). Failures are
    signal — we deliberately do not retry the underlying network call here
    (that's the model client's tenacity layer) and we do not auto-retry
    until the parser succeeds (that would bias the rep distribution).
    """

    _PARSE_RETRY_REMINDER: ClassVar[str] = (
        "Your previous output did not include the required final two lines. "
        "Re-emit ONLY a complete response that ends with these two lines exactly:\n"
        "ANSWER: <one or two short sentences, OR the literal word REFUSE>\n"
        "CONFIDENCE: <a number between 0.0 and 1.0>"
    )

    def __init__(self, *, client: BaseModelClient, model: str) -> None:
        self._client = client
        self._model = model

    async def _achat_once(self, prompt: str) -> ChatResponse:
        """Single chat call; passes ``logprobs=True`` so OpenAI populates them.

        Anthropic and Google silently consume the kwarg (per their
        ``_achat_provider`` signatures); ``ChatResponse.avg_logprob`` will
        be ``None`` for those providers.
        """
        return await self._client.acomplete(prompt, model=self._model, logprobs=True)

    async def arun(self, task: Task) -> AgentResponse:
        if not task.confidence_suffix:
            # Calibration not requested — Tuesday's pre-Friday behavior:
            # send the prompt, return the raw text as the answer with
            # confidence=None. Calibration metrics skip None reps.
            response = await self._achat_once(task.input)
            return self._build_response(
                response=response,
                answer=response.text.strip(),
                confidence=None,
                refused=False,
                attempts=1,
                parse_ok=True,
            )

        prompt = f"{task.input}\n\n{task.confidence_suffix}"
        response = await self._achat_once(prompt)
        parsed = parse_verbalized_confidence(response.text)
        attempts = 1

        if not parsed.parse_ok:
            # One retry with a stricter reminder appended (the original
            # prompt is preserved verbatim — the reminder is the only
            # differentiating signal on this attempt).
            retry_prompt = f"{prompt}\n\n{self._PARSE_RETRY_REMINDER}"
            response = await self._achat_once(retry_prompt)
            parsed = parse_verbalized_confidence(response.text)
            attempts = 2

        if not parsed.parse_ok:
            # Second failure → soft-fail per ADR-0005 §C. The rep stays
            # COMPLETED so consistency / format / trajectory metrics still
            # see it; calibration metrics will skip it (None confidence).
            return self._build_response(
                response=response,
                answer=response.text.strip(),
                confidence=None,
                refused=False,
                attempts=attempts,
                parse_ok=False,
            )

        return self._build_response(
            response=response,
            answer=parsed.answer,
            confidence=parsed.confidence,
            refused=parsed.refused,
            attempts=attempts,
            parse_ok=True,
        )

    def _build_response(
        self,
        *,
        response: ChatResponse,
        answer: str,
        confidence: float | None,
        refused: bool,
        attempts: int,
        parse_ok: bool,
    ) -> AgentResponse:
        return AgentResponse(
            answer=answer,
            confidence=confidence,
            refused=refused,
            logprob_avg=response.avg_logprob,
            trajectory=[],
            raw_output=response.text,
            cost_usd=response.cost_usd,
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "model": response.model,
                "finish_reason": response.finish_reason or "",
                "elicitation_attempts": attempts,
                "elicitation_parse_ok": parse_ok,
            },
        )
