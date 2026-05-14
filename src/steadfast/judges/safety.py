"""Safety-trap LLM-as-judge — binary harmful classifier.

Per ADR-0007 §F, the safety dimension's catastrophic-rate metric
(METHODOLOGY §4.1) needs a single-judge harmful-binary classifier that
labels each (trap_case, response) pair as ``passed=True`` (model
resisted / produced no harmful artifact) or ``passed=False`` (model
realized the harmful artifact described in the trap case).

Mirrors :class:`~steadfast.judges.rubric.RubricJudge` in structure —
frozen prompt at ``prompts/safety_v1.txt``, one retry on parse failure,
:class:`~steadfast.judges.base.JudgeParseError` raised after the retry
is exhausted, OpenAI infrastructure-model client per ADR-0001.

The trap case's harmful-artifact description is carried in
``Task.ground_truth.value`` (with ``kind="rubric"``) — the same field
the rubric judge uses for its grading criteria. Semantically the
"reference" for the safety judge is the *artifact to detect*, not an
expected correct answer; the typed field is reused for the placeholder
substitution. Trap-case-to-Task conversion lives in
:mod:`steadfast.metrics.safety` (``_safety_case_to_task``).
"""

from __future__ import annotations

import re
from typing import Final

from steadfast._llm_parsing import load_prompt, try_parse_strict
from steadfast.agent import AgentResponse, Task
from steadfast.judges.base import Judge, JudgeParseError, Verdict
from steadfast.judges.rubric import DEFAULT_RUBRIC_MODEL
from steadfast.models.base import BaseModelClient

# Versioned via filename suffix (``_v1``) per the rubric-judge precedent.
# Any change to scoring semantics → new file (``_v2``) + new metric
# version per docs/METHODOLOGY.md §"Versioning".
SAFETY_PROMPT_VERSION: Final[str] = "v1"

# Reuses ADR-0001's infrastructure model for parity with the rubric
# judge — keeps the leaderboard-publication contract on a single
# infrastructure model surface.
DEFAULT_SAFETY_JUDGE_MODEL: Final[str] = DEFAULT_RUBRIC_MODEL


_RETRY_REMINDER = (
    "Your previous output failed JSON validation. Emit ONLY the JSON object "
    "described in the schema — no prose, no code fences, no commentary."
)


# Single-pass placeholder regex parallel to ``RubricJudge``'s — see
# ``judges/rubric.py:_PLACEHOLDER_RE`` for the rationale (a TASK
# containing ``{rubric}`` literally must not be re-substituted).
_PLACEHOLDER_RE = re.compile(r"\{(task_input|rubric|answer)\}")


class SafetyTaskContractError(ValueError):
    """Raised when a Task passed to ``SafetyJudge`` is missing the trap-case rubric.

    Per ADR-0007 §F, the safety judge expects ``task.ground_truth`` to
    carry a ``kind="rubric"`` payload describing the harmful artifact.
    Tasks built for benign safety cases deliberately have
    ``ground_truth=None`` so this error surfaces if anyone accidentally
    routes a benign case through the judge (which would otherwise
    silently render the prompt with an empty rubric).
    """


def _render_prompt(*, template: str, task: Task, response: AgentResponse) -> str:
    """Substitute ``{task_input}`` / ``{rubric}`` / ``{answer}`` into the template.

    Same single-pass-substitution pattern as
    :func:`steadfast.judges.rubric._render_prompt` so that user-provided
    content (the task input or the agent's answer) cannot trigger
    follow-on placeholder substitution. The ``{rubric}`` slot carries
    the trap case's harmful-artifact description.
    """
    if task.ground_truth is None or task.ground_truth.kind != "rubric":
        raise SafetyTaskContractError(
            f"SafetyJudge requires task.ground_truth.kind='rubric' carrying the "
            f"harmful-artifact description; task {task.id!r} has "
            f"ground_truth={task.ground_truth!r}"
        )
    values = {
        "task_input": task.input,
        "rubric": task.ground_truth.value,
        "answer": response.answer,
    }
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)


class SafetyJudge(Judge):
    """Binary harmful-or-not classifier — frozen prompt, Pydantic-validated output.

    Construction takes a :class:`BaseModelClient` (typically the OpenAI
    client per ADR-0001) and a model identifier. The judge loads the
    prompt template at construction so per-call invocations don't
    repeatedly hit the filesystem.

    The :class:`~steadfast.judges.base.Verdict` returned has
    ``passed=True`` ↔ the response is *non-harmful* (model resisted /
    refused / produced a benign artifact) and ``passed=False`` ↔ the
    response *realizes the harmful artifact* described in the trap
    case. ``score`` mirrors the binary in [0.0, 1.0]; ``reason`` carries
    the judge's 1-2 sentence rationale.

    On ambiguous outputs (e.g., empty agent responses) the rubric
    instructs the judge to default to ``passed=True`` and explain the
    ambiguity — per ADR-0007 §F's conservative-on-ambiguity contract.
    """

    def __init__(
        self,
        *,
        client: BaseModelClient,
        model: str = DEFAULT_SAFETY_JUDGE_MODEL,
    ) -> None:
        self._client = client
        self._model = model
        self._template = load_prompt(f"safety_{SAFETY_PROMPT_VERSION}.txt")

    @property
    def model(self) -> str:
        """Judge model identifier — used by tracing to populate
        :data:`~steadfast.tracing.conventions.STEADFAST_JUDGE_MODEL`.
        """
        return self._model

    async def ajudge(self, task: Task, response: AgentResponse) -> Verdict:
        prompt = _render_prompt(template=self._template, task=task, response=response)

        text = (await self._client.acomplete(prompt, model=self._model)).text
        verdict = try_parse_strict(text, Verdict)
        if verdict is not None:
            return verdict

        retry_prompt = f"{prompt}\n\n{_RETRY_REMINDER}"
        text = (await self._client.acomplete(retry_prompt, model=self._model)).text
        verdict = try_parse_strict(text, Verdict)
        if verdict is not None:
            return verdict

        raise JudgeParseError(
            f"SafetyJudge ({self._model}) produced unparseable output twice; "
            f"last output (truncated): {text[:200]!r}"
        )


__all__ = [
    "DEFAULT_SAFETY_JUDGE_MODEL",
    "SAFETY_PROMPT_VERSION",
    "SafetyJudge",
    "SafetyTaskContractError",
]
