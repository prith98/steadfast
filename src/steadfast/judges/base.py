"""Judge protocol and ``Verdict`` Pydantic model.

Per ADR-0003 §B, a :class:`Verdict` carries a continuous ``score`` in
[0, 1], a binary ``passed`` flag, and a free-text ``reason``. Both fields
are populated unconditionally so downstream metrics don't have to redo
the threshold decision.

The :class:`Judge` ABC exposes a single async :meth:`Judge.ajudge` —
async because rubric judges hit an LLM. The future ensemble path
(ADR-0001 §"Path to v0.2") composes multiple :class:`Judge` instances
behind the same surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from steadfast.agent import AgentResponse, Task


class Verdict(BaseModel):
    """Outcome of judging one (task, response) pair.

    Frozen so a verdict cannot mutate after creation — keeps the metric
    pipeline reasoning straightforward and prevents accidental
    write-after-judge bugs.
    """

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str


class Judge(ABC):
    """Abstract base for outcome judges."""

    @abstractmethod
    async def ajudge(self, task: Task, response: AgentResponse) -> Verdict:
        """Score ``response`` against ``task`` and return a :class:`Verdict`."""


class JudgeError(RuntimeError):
    """Base class for judge-side failures.

    Per ADR-0003 §B.4, a judge raises rather than producing a soft-failed
    verdict. Failures are signal — silent fallbacks would corrupt
    downstream metric distributions.
    """


class JudgeParseError(JudgeError):
    """A rubric judge's LLM produced output that did not validate.

    Raised after the rubric retry budget (1 retry per ADR-0003 §B.4) is
    exhausted.
    """
