"""Tool-using agent fixtures for contradiction-handling integration tests.

Per ``docs/WEEK_2.md`` §Wednesday item 5: the customer-support pilot is
toolless, so the contradiction perturbation has nothing to corrupt against
the pilot agents. This module provides minimal :class:`Agent` subclasses
that exercise the metric path end-to-end on a synthetic single-tool
surface — one tool (`lookup_policy`), one ground-truth answer, and three
behavior modes corresponding to the classifier's three labels.

The fixture deliberately couples to
:mod:`steadfast.perturbations.contradiction` so the classifier sees
realistic agent outputs (tool calls, optionally corrupted, with the
metadata convention populated).

Citation: per ADR-0006 §D, the contradiction perturbation hooks at the
agent's tool-execution layer (Steadfast's :class:`Agent` ABC has no
tool-execution loop of its own — :meth:`Agent.arun` returns a complete
:class:`AgentResponse` after the agent has run its own loop). This
fixture is the v0.1 reference implementation of that wiring.
"""

from __future__ import annotations

from decimal import Decimal
from typing import ClassVar, Final, Literal

from steadfast.agent import Agent, AgentResponse, Task, ToolCall
from steadfast.perturbations.contradiction import (
    CORRUPTED_CALLS_METADATA_KEY,
    DEFAULT_CORRUPTION_PROBABILITY,
    corrupt_tool_result,
    encode_corrupted_call_indices,
    should_corrupt,
)

EchoBehavior = Literal["detect", "retry", "hallucinate"]


class EchoToolAgent(Agent):
    """Single-tool agent that exhibits one of the three classifier behaviors.

    Calls a synthetic ``lookup_policy(query="returns")`` tool that returns
    :attr:`GROUND_TRUTH`. With probability ``corruption_probability`` per
    call (deterministic via ADR-0006 §B's ``:tool{idx}`` seed), the result
    is replaced by a corrupted variant via
    :func:`steadfast.perturbations.contradiction.corrupt_tool_result`. The
    agent then reacts according to ``behavior``:

    * ``"detect"`` — if any call was corrupted, the answer contains the
      detection phrase ``"inconsistent"``; otherwise returns the
      uncorrupted result. Models the "agent flagged the contradiction"
      pattern (rule 1 in :func:`classify_contradiction_response`).
    * ``"retry"`` — if the first call was corrupted, makes a second call
      with identical args and uses the second result as the answer. The
      second call gets a fresh ``tool_call_idx=1`` corruption coin —
      independent of the first call's corruption decision. Models the
      "agent retried after contradiction" pattern (rule 2).
    * ``"hallucinate"`` — uses whatever the (possibly corrupted) tool
      returned as the answer, no checks. Models the fallthrough (rule 3).

    Records corrupted-call indices in
    ``response.metadata[CORRUPTED_CALLS_METADATA_KEY]`` per the convention
    documented in :mod:`steadfast.perturbations.contradiction`.
    """

    DETECTION_PHRASE: ClassVar[str] = "the records appear inconsistent"
    """Substring that triggers rule 1 of the classifier when matched
    against the default ``contradiction_detection_phrases_v1`` list
    (``"inconsistent"`` is one of the frozen detection phrases)."""

    GROUND_TRUTH: ClassVar[str] = "Our return window is 30 days for unopened items"
    """The synthetic tool's uncorrupted response. Long enough that
    the corruption strategies (negate_number on "30", swap_entities
    on "Our"/"Returns", etc.) have material to work on."""

    def __init__(
        self,
        *,
        behavior: EchoBehavior,
        corruption_probability: float = DEFAULT_CORRUPTION_PROBABILITY,
    ) -> None:
        self._behavior: Final[EchoBehavior] = behavior
        self._p: Final[float] = corruption_probability
        # Diagnostic surfaces for tests: record every (task, returned answer).
        self.calls: list[tuple[str, str]] = []

    async def arun(self, task: Task) -> AgentResponse:
        trajectory: list[ToolCall] = []
        corrupted_indices: list[int] = []

        first_result, first_corrupted = self._invoke_tool(task=task, tool_call_idx=0)
        trajectory.append(
            ToolCall(
                name="lookup_policy",
                args={"query": "returns"},
                result=first_result,
            )
        )
        if first_corrupted:
            corrupted_indices.append(0)

        answer: str
        refused = False

        if self._behavior == "retry" and first_corrupted:
            # Re-call the same tool with identical args — the retry rule's
            # exact pattern. tool_call_idx=1 → independent corruption coin.
            second_result, second_corrupted = self._invoke_tool(task=task, tool_call_idx=1)
            trajectory.append(
                ToolCall(
                    name="lookup_policy",
                    args={"query": "returns"},
                    result=second_result,
                )
            )
            if second_corrupted:
                corrupted_indices.append(1)
            answer = second_result
        elif self._behavior == "detect" and first_corrupted:
            answer = self.DETECTION_PHRASE
        else:
            answer = first_result

        self.calls.append((task.id, answer))

        return AgentResponse(
            answer=answer,
            refused=refused,
            trajectory=trajectory,
            raw_output=answer,
            cost_usd=Decimal("0"),
            metadata={
                CORRUPTED_CALLS_METADATA_KEY: encode_corrupted_call_indices(corrupted_indices),
                "behavior": self._behavior,
            },
        )

    def _invoke_tool(self, *, task: Task, tool_call_idx: int) -> tuple[str, bool]:
        """Synthetic ``lookup_policy`` tool: returns (result, was_corrupted).

        Returns the authoritative corruption flag from
        :func:`should_corrupt` rather than relying on string-equality
        comparison against ``GROUND_TRUTH`` — a corruption strategy that
        happens to produce the ground-truth string would otherwise be
        silently misclassified as uncorrupted.
        """
        if should_corrupt(
            task_id=task.id,
            tool_call_idx=tool_call_idx,
            probability=self._p,
        ):
            return (
                corrupt_tool_result(
                    self.GROUND_TRUTH,
                    task_id=task.id,
                    tool_call_idx=tool_call_idx,
                ),
                True,
            )
        return self.GROUND_TRUTH, False


__all__ = [
    "EchoBehavior",
    "EchoToolAgent",
]
