"""Consistency dimension — output, trajectory, and format consistency.

Per ``docs/METHODOLOGY.md`` §1 and ADR-0004 §E, three independent
measurement functions:

* :func:`measure_output_consistency` — K=5 paraphrases run once each;
  pairwise embedding cosine + LLM-judge 0-4 Likert rubric (normalized
  to [0, 1] per ADR-0004 §B). Mean rubric with bootstrap CI is the
  reported headline metric.
* :func:`measure_trajectory_consistency` — N=10 same-input reps;
  ``1 - mean(normalized Levenshtein)`` over tool-name sequences plus
  ``agentevals`` superset arg-equivalence rate.
* :func:`measure_format_consistency` — N=10 same-input reps; pass-rate
  of strict JSON-schema validation, Wilson 95% CI.

Each function returns a typed Pydantic result; downstream reporting and
aggregation consume them uniformly.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import re
from collections.abc import Sequence
from typing import Any, Final

import jsonschema
from agentevals.trajectory.match import create_trajectory_match_evaluator
from pydantic import BaseModel, ConfigDict, Field

from steadfast._llm_parsing import load_prompt, try_parse_strict
from steadfast.agent import Agent, Task, ToolCall
from steadfast.models.base import BaseModelClient
from steadfast.models.openai_client import DEFAULT_EMBEDDING_MODEL, OpenAIClient
from steadfast.perturbations.paraphrase import (
    DEFAULT_PARAPHRASE_MODEL,
    generate_paraphrases,
)
from steadfast.runner import RepRecord, RepStatus
from steadfast.stats.bootstrap import BootstrapCI, bootstrap_ci
from steadfast.stats.wilson import WilsonCI, wilson_ci

CONSISTENCY_RUBRIC_PROMPT_VERSION: Final[str] = "v1"
DEFAULT_RUBRIC_JUDGE_MODEL: Final[str] = "gpt-5.2"
LIKERT_MAX: Final[int] = 4

# Substituted in for empty agent answers before embedding / rubric calls.
# OpenAI's embedding endpoint rejects empty-string inputs (HTTP 400
# "input cannot be an empty string"), so a metric over a Gemini run that
# hits a safety filter would crash without this sentinel. The rubric
# judge and the embedding model will both score the placeholder low
# against any real answer, which is the correct calibration signal — an
# empty response from the agent is *not* consistent with a real one.
_EMPTY_ANSWER_PLACEHOLDER: Final[str] = "(no answer)"

_RUBRIC_PLACEHOLDER_RE = re.compile(r"\{(task_input|answer_a|answer_b)\}")


# ---------------------------------------------------------------------------
# Result Pydantic models
# ---------------------------------------------------------------------------


class OutputConsistencyResult(BaseModel):
    """Per-task output-consistency measurement (METHODOLOGY §1.1).

    ``n_empty_answers`` counts how many of the K paraphrase responses were
    empty (e.g., model safety filter or empty content). Empty answers are
    substituted with a placeholder before the embedding call (OpenAI's
    embedding endpoint rejects ``""``); rubric and cosine scores against
    the placeholder are correctly low, which the bootstrap CI absorbs.
    Surfaced in the report so a high count signals a target-model issue
    rather than a consistency-metric issue.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    k: int
    paraphrase_rejection_rate: float
    rubric_scores: list[float]  # length C(K, 2); each in [0, 1]
    embedding_cosines: list[float]  # length C(K, 2)
    mean_rubric: float
    mean_embedding_cosine: float
    rubric_ci: BootstrapCI
    embedding_ci: BootstrapCI
    n_empty_answers: int = 0
    rubric_prompt_version: str = CONSISTENCY_RUBRIC_PROMPT_VERSION
    embedding_model: str
    rubric_model: str


class TrajectoryConsistencyResult(BaseModel):
    """Per-task trajectory-consistency measurement (METHODOLOGY §1.2)."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    n_reps: int
    # Headline metric: 1 - mean(normalized Levenshtein) over tool-name
    # sequences. None when undefined (e.g., empty trajectories everywhere
    # or fewer than 2 completed reps).
    value: float | None
    ci: BootstrapCI | None
    # Auxiliary: rate of pairs whose argument tuples superset-match
    # (per agentevals).
    arg_match_rate: float | None
    reason: str | None = None


class FormatConsistencyResult(BaseModel):
    """Per-task format-consistency measurement (METHODOLOGY §1.3)."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    n_reps: int
    pass_rate: float | None
    ci: WilsonCI | None
    reason: str | None = None


# ---------------------------------------------------------------------------
# Output consistency — K=5 paraphrases x pairwise rubric + embedding
# ---------------------------------------------------------------------------


class _RubricOutput(BaseModel):
    """JSON schema for the consistency rubric LLM output."""

    score: int = Field(ge=0, le=LIKERT_MAX)
    reason: str


def _render_rubric_prompt(*, template: str, task_input: str, a: str, b: str) -> str:
    values = {"task_input": task_input, "answer_a": a, "answer_b": b}
    return _RUBRIC_PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)


async def _judge_pair_consistency(
    *,
    client: BaseModelClient,
    model: str,
    template: str,
    task_input: str,
    answer_a: str,
    answer_b: str,
) -> float:
    """Score a pair of answers on the 0-4 Likert rubric, normalized to [0, 1].

    Pattern adapted from the Pydantic Evals LLM-as-judge guide
    (https://pydantic.dev/articles/llm-as-a-judge); see METHODOLOGY §1.1
    for the rationale on combining LLM-judge rubric with embedding
    cosine.

    Conservative on parse failure: returns 0.5 (the maximum-uncertainty
    score) and lets the bootstrap CI absorb the noise. We don't raise
    here — a single-pair rubric failure shouldn't abort the whole
    output-consistency measurement, and the CI will widen appropriately
    if many pairs fail.
    """
    prompt = _render_rubric_prompt(template=template, task_input=task_input, a=answer_a, b=answer_b)
    response = await client.acomplete(prompt, model=model, temperature=0.0)
    parsed = try_parse_strict(response.text, _RubricOutput)
    if parsed is None:
        return 0.5
    return parsed.score / LIKERT_MAX


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    arr_a = list(a)
    arr_b = list(b)
    if len(arr_a) != len(arr_b):
        raise ValueError(f"vector length mismatch: {len(arr_a)} vs {len(arr_b)}")
    dot = sum(x * y for x, y in zip(arr_a, arr_b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in arr_a))
    norm_b = math.sqrt(sum(y * y for y in arr_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def measure_output_consistency(
    *,
    task: Task,
    agent: Agent,
    infra_client: OpenAIClient,
    k: int = 5,
    seed: int = 0,
    rubric_model: str = DEFAULT_RUBRIC_JUDGE_MODEL,
    paraphrase_model: str = DEFAULT_PARAPHRASE_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> OutputConsistencyResult:
    """Per METHODOLOGY §1.1: K paraphrases, pairwise rubric + embedding.

    The infrastructure client (``infra_client``) is the OpenAI client per
    ADR-0001's lock — the same instance is reused for paraphrase
    generation, the rubric judge, and embeddings so concurrency is
    bounded by a single semaphore.
    """
    if k < 3:
        # K=2 yields only C(2,2)=1 pair — bootstrap requires N>=2. K=3 is
        # the smallest meaningful K. Methodology default is K=5.
        raise ValueError("output consistency requires k >= 3 paraphrases")

    paraphrase_set = await generate_paraphrases(
        original=task.input,
        k=k,
        client=infra_client,
        model=paraphrase_model,
        seed=seed,
    )

    # Build paraphrased Tasks and run the agent on each. We do NOT go
    # through ``run_task`` (per ADR-0004 §E): output consistency is K=5
    # different inputs, conceptually orthogonal to N=10 same-input reps.
    paraphrase_tasks = [
        task.model_copy(update={"input": paraphrase}) for paraphrase in paraphrase_set.paraphrases
    ]
    responses = await asyncio.gather(*(agent.arun(t) for t in paraphrase_tasks))
    raw_answers = [r.answer for r in responses]

    # Substitute empty answers with a placeholder. OpenAI's embedding
    # endpoint rejects empty strings (HTTP 400), so even one empty Gemini
    # safety-filtered response would crash the whole consistency
    # measurement. The placeholder scores low against any real answer in
    # both the rubric and the cosine pass — that's the correct
    # calibration signal: an empty response is *not* consistent with a
    # real one. ``n_empty_answers`` rides on the result so the report
    # surfaces high-empty cases as a target-model issue.
    n_empty_answers = sum(1 for a in raw_answers if not a.strip())
    answers = [a if a.strip() else _EMPTY_ANSWER_PLACEHOLDER for a in raw_answers]

    # Concurrent across pairs — the model client's semaphore bounds the
    # fan-out so this can't exceed the configured per-client concurrency.
    rubric_template = load_prompt("consistency_rubric_v1.txt")
    pairs: list[tuple[int, int]] = list(itertools.combinations(range(k), 2))
    rubric_scores = await asyncio.gather(
        *(
            _judge_pair_consistency(
                client=infra_client,
                model=rubric_model,
                template=rubric_template,
                task_input=task.input,
                answer_a=answers[i],
                answer_b=answers[j],
            )
            for i, j in pairs
        )
    )

    # Batched embedding call → pairwise cosines are local arithmetic.
    vectors, _, _ = await infra_client.aembed(answers, model=embedding_model)
    embedding_cosines = [_cosine_similarity(vectors[i], vectors[j]) for i, j in pairs]

    rubric_ci = bootstrap_ci(rubric_scores, seed=seed)
    embedding_ci = bootstrap_ci(embedding_cosines, seed=seed)

    return OutputConsistencyResult(
        task_id=task.id,
        k=k,
        paraphrase_rejection_rate=paraphrase_set.rejection_rate,
        rubric_scores=list(rubric_scores),
        embedding_cosines=embedding_cosines,
        mean_rubric=rubric_ci.point_estimate,
        mean_embedding_cosine=embedding_ci.point_estimate,
        rubric_ci=rubric_ci,
        embedding_ci=embedding_ci,
        n_empty_answers=n_empty_answers,
        embedding_model=embedding_model,
        rubric_model=rubric_model,
    )


# ---------------------------------------------------------------------------
# Trajectory consistency — Wagner-Fischer Levenshtein + agentevals superset
# ---------------------------------------------------------------------------


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Wagner-Fischer edit distance between two sequences of tokens.

    Tokens are tool names (strings); the algorithm is the textbook DP
    over a (m+1) x (n+1) matrix with O(min(m, n)) space.

    Citation: Wagner, R. A. & Fischer, M. J. (1974). "The string-to-string
    correction problem." *J. ACM* 21(1), 168-173.
    """
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,  # deletion
                cur[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = cur
    return prev[n]


def _normalized_levenshtein(a: Sequence[str], b: Sequence[str]) -> float:
    """Edit distance normalized to [0, 1] by the longer sequence length."""
    if not a and not b:
        return 0.0
    return _levenshtein(a, b) / max(len(a), len(b))


def _trajectory_to_openai_messages(trajectory: list[ToolCall]) -> list[dict[str, Any]]:
    """Adapt a Steadfast trajectory to the OpenAI message format agentevals expects.

    agentevals reads tool calls from ``message["tool_calls"]`` on
    assistant messages, where each entry has the OpenAI-style
    ``{"function": {"name": ..., "arguments": ...}}`` shape.
    """
    if not trajectory:
        return []
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args, sort_keys=True),
                    }
                }
                for tc in trajectory
            ],
        }
    ]


def _agentevals_superset_match_rate(trajectories: list[list[ToolCall]]) -> float | None:
    """Pairwise rate at which two trajectories share the same tool-call set.

    Uses ``agentevals.create_trajectory_match_evaluator(trajectory_match_mode=
    "superset", tool_args_match_mode="exact")`` in BOTH directions per
    pair: a pair is "matched" iff trajectory A is a superset of B AND B
    is a superset of A — i.e., they contain the same tool calls (possibly
    in different orders, possibly with duplicates handled per
    ``agentevals`` semantics). The bidirectional check makes the metric
    direction-independent, which matters because
    ``itertools.combinations`` is unordered: a unidirectional check
    would yield results that depend on iteration order rather than on
    the trajectories themselves.
    """
    if len(trajectories) < 2:
        return None
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",
        tool_args_match_mode="exact",
    )
    pairs = list(itertools.combinations(trajectories, 2))
    matches = 0
    for a, b in pairs:
        msgs_a = _trajectory_to_openai_messages(a)
        msgs_b = _trajectory_to_openai_messages(b)
        a_superset_b = bool(evaluator(outputs=msgs_a, reference_outputs=msgs_b).get("score"))
        b_superset_a = bool(evaluator(outputs=msgs_b, reference_outputs=msgs_a).get("score"))
        if a_superset_b and b_superset_a:
            matches += 1
    return matches / len(pairs)


def measure_trajectory_consistency(
    reps: list[RepRecord],
) -> TrajectoryConsistencyResult:
    """Per METHODOLOGY §1.2 / ADR-0004 §G: pairwise normalized-Levenshtein.

    Pure function over completed reps from the runner. Returns
    ``value=None`` when there are fewer than two completed reps or
    every completed rep has an empty trajectory (per ADR-0002 §A.2 the
    metric is N/A for toolless agents).
    """
    completed = [r for r in reps if r.status == RepStatus.COMPLETED and r.response is not None]
    task_id = reps[0].task_id if reps else ""

    if len(completed) < 2:
        return TrajectoryConsistencyResult(
            task_id=task_id,
            n_reps=len(completed),
            value=None,
            ci=None,
            arg_match_rate=None,
            reason="trajectory consistency requires at least 2 completed reps",
        )

    # mypy: completed is filtered by ``r.response is not None``, but the
    # generator below loses that narrowing — assert at the boundary.
    trajectories: list[list[ToolCall]] = []
    for r in completed:
        assert r.response is not None
        trajectories.append(list(r.response.trajectory))

    if all(len(t) == 0 for t in trajectories):
        return TrajectoryConsistencyResult(
            task_id=task_id,
            n_reps=len(completed),
            value=None,
            ci=None,
            arg_match_rate=None,
            reason="trajectory not exposed by agent",
        )

    similarities: list[float] = [
        1.0 - _normalized_levenshtein([tc.name for tc in a], [tc.name for tc in b])
        for a, b in itertools.combinations(trajectories, 2)
    ]
    ci = bootstrap_ci(similarities)
    arg_match_rate = _agentevals_superset_match_rate(trajectories)

    return TrajectoryConsistencyResult(
        task_id=task_id,
        n_reps=len(completed),
        value=ci.point_estimate,
        ci=ci,
        arg_match_rate=arg_match_rate,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Format consistency — JSON-schema pass rate + Wilson CI
# ---------------------------------------------------------------------------


def _validates_against(answer: str, schema_obj: dict[str, Any]) -> bool:
    """Return True iff ``answer`` parses as JSON AND validates against ``schema_obj``.

    Both parse failure and validation failure count as "fail" — there is
    no useful distinction at the pass-rate aggregation layer.
    """
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return False
    try:
        jsonschema.validate(parsed, schema_obj)
    except jsonschema.ValidationError:
        return False
    return True


def measure_format_consistency(
    reps: list[RepRecord],
    schema: str,
) -> FormatConsistencyResult:
    """Per METHODOLOGY §1.3: schema-validation pass rate with Wilson CI.

    ``schema`` is a JSON Schema **string** (matching ``Task.output_schema``
    per ADR-0004 §D); we parse it once here. Tasks with no schema should
    short-circuit at the call site.
    """
    completed = [r for r in reps if r.status == RepStatus.COMPLETED and r.response is not None]
    task_id = reps[0].task_id if reps else ""

    if not completed:
        return FormatConsistencyResult(
            task_id=task_id,
            n_reps=0,
            pass_rate=None,
            ci=None,
            reason="format consistency requires at least 1 completed rep",
        )

    try:
        schema_obj = json.loads(schema)
    except json.JSONDecodeError as exc:
        # A malformed schema is a task-authoring bug; surface it loudly
        # rather than silently scoring 0% across every rep.
        raise ValueError(f"task.output_schema is not valid JSON: {exc}") from exc

    successes = sum(
        1
        for r in completed
        if r.response is not None and _validates_against(r.response.answer, schema_obj)
    )
    ci = wilson_ci(successes=successes, trials=len(completed))
    return FormatConsistencyResult(
        task_id=task_id,
        n_reps=len(completed),
        pass_rate=ci.proportion,
        ci=ci,
        reason=None,
    )
