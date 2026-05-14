"""Outcome judges — exact match, LLM-as-judge rubric, ensemble (v0.2).

Public surface:

* :class:`Verdict` — Pydantic outcome of judging one (task, response) pair.
* :class:`Judge` — abstract base class.
* :class:`ExactMatchJudge`, :class:`RubricJudge` — concrete judges per
  ADR-0003 §B.
* :class:`JudgeError`, :class:`JudgeParseError` — failure surfaces.
* :func:`build_default_judge` — factory dispatching on ``task.judge``.
* :func:`judge_run_result` — batch dispatcher; populates each rep's
  ``verdict`` field and emits ``score`` spans.

See ``docs/adr/0003-tracing-and-judges.md`` for the design rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from steadfast.agent import Task
from steadfast.judges.base import Judge, JudgeError, JudgeParseError, Verdict
from steadfast.judges.exact_match import ExactMatchJudge, canonicalize
from steadfast.judges.rubric import (
    DEFAULT_RUBRIC_MODEL,
    RUBRIC_PROMPT_VERSION,
    RubricJudge,
)
from steadfast.judges.safety import (
    DEFAULT_SAFETY_JUDGE_MODEL,
    SAFETY_PROMPT_VERSION,
    SafetyJudge,
    SafetyTaskContractError,
)
from steadfast.models.base import BaseModelClient
from steadfast.tracing import record_verdict, score_span

if TYPE_CHECKING:
    # Importing RunResult / RepStatus at module load time would create a cycle:
    # ``steadfast.runner`` imports ``Verdict`` from ``steadfast.judges.base``,
    # which forces ``steadfast.judges/__init__.py`` to execute. Use the
    # ``from __future__ import annotations`` lazy-evaluation form for the
    # type hints; the runtime ``RepStatus`` lookup is deferred into the
    # function body below.
    from steadfast.runner import RunResult

_log = logging.getLogger(__name__)


def build_default_judge(
    task: Task,
    *,
    rubric_client: BaseModelClient | None = None,
    rubric_model: str = DEFAULT_RUBRIC_MODEL,
    safety_model: str = DEFAULT_SAFETY_JUDGE_MODEL,
) -> Judge:
    """Return the judge appropriate to ``task.judge``.

    For ``task.judge == "rubric"`` or ``"safety_harmful"`` callers must
    provide ``rubric_client`` (typically the OpenAI client per
    ADR-0001). For ``"exact_match"`` no client is needed.

    ``rubric_model`` and ``safety_model`` are kept as separate
    parameters even though both default to ``gpt-5.2`` in v0.1 — the
    rubric and safety judges are independent prompt surfaces and may
    diverge in v0.2 (per ADR-0007 §H ensemble path). Letting callers
    override them independently keeps the dispatch defensible against
    that divergence.
    """
    if task.judge == "exact_match":
        return ExactMatchJudge()
    if task.judge == "rubric":
        if rubric_client is None:
            raise ValueError(
                "RubricJudge requires a BaseModelClient (typically OpenAIClient "
                "per ADR-0001); pass rubric_client= explicitly."
            )
        return RubricJudge(client=rubric_client, model=rubric_model)
    if task.judge == "safety_harmful":
        if rubric_client is None:
            raise ValueError(
                "SafetyJudge requires a BaseModelClient (typically OpenAIClient "
                "per ADR-0001); pass rubric_client= explicitly."
            )
        return SafetyJudge(client=rubric_client, model=safety_model)
    raise ValueError(f"unknown task.judge value: {task.judge!r}")


async def judge_run_result(
    result: RunResult,
    *,
    rubric_client: BaseModelClient | None = None,
    rubric_model: str = DEFAULT_RUBRIC_MODEL,
) -> RunResult:
    """Run the appropriate :class:`Judge` over every completed rep in ``result``.

    Mutates each rep's ``verdict`` field in place and returns ``result``
    so call sites can chain. Failed reps and reps with no response are
    left unjudged.

    Per ADR-0003 §B.4 a judge raises rather than soft-failing. We catch
    :class:`JudgeError` at the batch boundary so a single judge failure
    doesn't abort scoring of the rest of the run; the rep's verdict
    stays ``None`` and the caller can distinguish "judge failed" from
    "judge said score=0.0".
    """
    # Deferred to break the ``runner`` ↔ ``judges`` import cycle (see the
    # TYPE_CHECKING note at the top of this file).
    from steadfast.runner import RepStatus

    judge = build_default_judge(result.task, rubric_client=rubric_client, rubric_model=rubric_model)
    judge_model = getattr(judge, "model", None)

    for rep in result.reps:
        if rep.status != RepStatus.COMPLETED or rep.response is None:
            continue
        try:
            # Letting JudgeError propagate out of score_span lets the span's
            # own except branch mark it ERROR with the right error.type;
            # we catch outside so the loop proceeds with sibling reps.
            with score_span(
                judge_kind=result.task.judge,
                task_id=rep.task_id,
                rep_idx=rep.rep_idx,
                run_id=rep.run_id,
                judge_model=judge_model,
            ) as span:
                verdict = await judge.ajudge(result.task, rep.response)
                rep.verdict = verdict
                record_verdict(span, score=verdict.score, passed=verdict.passed)
        except JudgeError as exc:
            _log.warning(
                "judge failed on task=%s rep=%d: %s",
                rep.task_id,
                rep.rep_idx,
                exc,
            )

    return result


__all__ = [
    "DEFAULT_RUBRIC_MODEL",
    "DEFAULT_SAFETY_JUDGE_MODEL",
    "RUBRIC_PROMPT_VERSION",
    "SAFETY_PROMPT_VERSION",
    "ExactMatchJudge",
    "Judge",
    "JudgeError",
    "JudgeParseError",
    "RubricJudge",
    "SafetyJudge",
    "SafetyTaskContractError",
    "Verdict",
    "build_default_judge",
    "canonicalize",
    "judge_run_result",
]
