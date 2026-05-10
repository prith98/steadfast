"""Tests for steadfast.metrics.consistency — output, trajectory, format."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any

import pytest

from steadfast.agent import Agent, AgentResponse, Task, ToolCall
from steadfast.metrics.consistency import (
    LIKERT_MAX,
    OutputConsistencyResult,
    _agentevals_superset_match_rate,
    _cosine_similarity,
    _levenshtein,
    _normalized_levenshtein,
    _trajectory_to_openai_messages,
    measure_format_consistency,
    measure_output_consistency,
    measure_trajectory_consistency,
)
from steadfast.models.base import ChatMessage, ChatResponse, TokenUsage
from steadfast.models.openai_client import OpenAIClient
from steadfast.runner import RepRecord, RepStatus


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from tenacity import wait_none

    monkeypatch.setattr("steadfast.models.base.wait_exponential", lambda **_kw: wait_none())


# ---------------------------------------------------------------------------
# Levenshtein and cosine helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([], [], 0),
        ([], ["x"], 1),
        (["x"], [], 1),
        (["a", "b"], ["a", "b"], 0),
        (["a", "b", "c"], ["a", "b", "d"], 1),
        (["a"], ["b"], 1),
        (["a", "b", "c"], ["c", "b", "a"], 2),
        (["a", "b", "c", "d"], ["a", "x", "c"], 2),  # substitute b→x, delete d
    ],
)
def test_levenshtein_known_values(a: list[str], b: list[str], expected: int) -> None:
    assert _levenshtein(a, b) == expected


def test_normalized_levenshtein_in_unit_interval() -> None:
    assert _normalized_levenshtein([], []) == 0.0
    assert _normalized_levenshtein(["a"], ["a"]) == 0.0
    assert _normalized_levenshtein(["a"], ["b"]) == 1.0
    assert _normalized_levenshtein(["a", "b", "c"], ["x", "y", "z"]) == 1.0
    assert _normalized_levenshtein(["a", "b", "c"], ["a", "b", "d"]) == pytest.approx(1 / 3)


def test_cosine_similarity_basic() -> None:
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine_similarity([1.0, 1.0], [1.0, 1.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_cosine_similarity_handles_zero_vector() -> None:
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        _cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Trajectory consistency
# ---------------------------------------------------------------------------


def _completed_rep(rep_idx: int, trajectory: list[ToolCall]) -> RepRecord:
    return RepRecord(
        run_id="r",
        task_id="t1",
        rep_idx=rep_idx,
        status=RepStatus.COMPLETED,
        response=AgentResponse(answer="ans", trajectory=trajectory),
    )


def test_trajectory_consistency_identical_trajectories_score_one() -> None:
    traj = [ToolCall(name="search", args={"q": "foo"}), ToolCall(name="parse", args={})]
    reps = [_completed_rep(i, traj) for i in range(3)]
    result = measure_trajectory_consistency(reps)
    assert result.value == pytest.approx(1.0)
    assert result.ci is not None
    assert result.arg_match_rate == pytest.approx(1.0)
    assert result.n_reps == 3


def test_trajectory_consistency_empty_everywhere_returns_na() -> None:
    reps = [_completed_rep(i, []) for i in range(3)]
    result = measure_trajectory_consistency(reps)
    assert result.value is None
    assert result.ci is None
    assert "trajectory not exposed" in (result.reason or "")


def test_trajectory_consistency_too_few_reps_returns_na() -> None:
    reps = [_completed_rep(0, [ToolCall(name="x", args={})])]
    result = measure_trajectory_consistency(reps)
    assert result.value is None
    assert "at least 2" in (result.reason or "")


def test_trajectory_consistency_partial_overlap() -> None:
    """Hand-computed: 3 reps with trajectories of length 3 each, one differs.

    rep 0: [a, b, c]   rep 1: [a, b, c]   rep 2: [a, b, d]
    Pair distances: (0,1)=0, (0,2)=1, (1,2)=1. Normalized: 0, 1/3, 1/3.
    Similarities (1 - normalized): 1, 2/3, 2/3.
    Mean = (1 + 2/3 + 2/3) / 3 = 7/9 ≈ 0.778.
    """
    reps = [
        _completed_rep(
            0,
            [ToolCall(name="a", args={}), ToolCall(name="b", args={}), ToolCall(name="c", args={})],
        ),
        _completed_rep(
            1,
            [ToolCall(name="a", args={}), ToolCall(name="b", args={}), ToolCall(name="c", args={})],
        ),
        _completed_rep(
            2,
            [ToolCall(name="a", args={}), ToolCall(name="b", args={}), ToolCall(name="d", args={})],
        ),
    ]
    result = measure_trajectory_consistency(reps)
    assert result.value == pytest.approx(7 / 9, abs=1e-6)


def test_trajectory_consistency_failed_reps_excluded() -> None:
    """Failed reps don't count toward the n_reps; only completed ones contribute."""
    completed = _completed_rep(0, [ToolCall(name="x", args={})])
    failed = RepRecord(
        run_id="r",
        task_id="t1",
        rep_idx=1,
        status=RepStatus.FAILED,
        error="boom",
    )
    result = measure_trajectory_consistency([completed, failed])
    assert result.n_reps == 1
    assert result.value is None  # need 2 completed


# ---------------------------------------------------------------------------
# agentevals adapter
# ---------------------------------------------------------------------------


def test_trajectory_to_openai_messages_format() -> None:
    traj = [ToolCall(name="search", args={"q": "foo"}), ToolCall(name="parse", args={"n": 3})]
    msgs = _trajectory_to_openai_messages(traj)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    tool_calls = msgs[0]["tool_calls"]
    assert tool_calls[0]["function"]["name"] == "search"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"q": "foo"}


def test_trajectory_to_openai_messages_empty() -> None:
    assert _trajectory_to_openai_messages([]) == []


def test_agentevals_superset_match_identical_trajectories() -> None:
    traj = [[ToolCall(name="search", args={"q": "x"})] for _ in range(3)]
    rate = _agentevals_superset_match_rate(traj)
    assert rate == pytest.approx(1.0)


def test_agentevals_superset_match_returns_none_for_singleton() -> None:
    assert _agentevals_superset_match_rate([[ToolCall(name="x", args={})]]) is None


def test_agentevals_superset_match_is_symmetric() -> None:
    """A pair where A ⊃ B but B ⊄ A must NOT count as a match — the metric
    is symmetric (both directions must hold) so its result is independent
    of iteration order in itertools.combinations."""
    a = [ToolCall(name="search", args={"q": "x"}), ToolCall(name="parse", args={})]
    b = [ToolCall(name="search", args={"q": "x"})]
    rate = _agentevals_superset_match_rate([a, b])
    assert rate == 0.0  # asymmetric pair → no match


# ---------------------------------------------------------------------------
# Format consistency
# ---------------------------------------------------------------------------


_INT_SCHEMA = json.dumps(
    {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
)


def _format_rep(rep_idx: int, answer: str) -> RepRecord:
    return RepRecord(
        run_id="r",
        task_id="t1",
        rep_idx=rep_idx,
        status=RepStatus.COMPLETED,
        response=AgentResponse(answer=answer),
    )


def test_format_consistency_all_pass() -> None:
    reps = [_format_rep(i, '{"n": 1}') for i in range(5)]
    result = measure_format_consistency(reps, _INT_SCHEMA)
    assert result.pass_rate == 1.0
    assert result.ci is not None
    assert result.ci.successes == 5
    assert result.ci.trials == 5


def test_format_consistency_mixed() -> None:
    reps = [
        _format_rep(0, '{"n": 1}'),  # ok
        _format_rep(1, '{"n": "wrong type"}'),  # schema fail
        _format_rep(2, "not json"),  # parse fail
        _format_rep(3, '{"n": 2}'),  # ok
    ]
    result = measure_format_consistency(reps, _INT_SCHEMA)
    assert result.pass_rate == pytest.approx(2 / 4)
    assert result.ci is not None
    assert result.ci.ci_lower < result.pass_rate < result.ci.ci_upper


def test_format_consistency_no_completed_reps_returns_na() -> None:
    failed = RepRecord(run_id="r", task_id="t1", rep_idx=0, status=RepStatus.FAILED, error="boom")
    result = measure_format_consistency([failed], _INT_SCHEMA)
    assert result.pass_rate is None
    assert result.ci is None


def test_format_consistency_invalid_schema_raises() -> None:
    """A malformed schema is a task-authoring bug — fail loudly."""
    reps = [_format_rep(0, '{"n": 1}')]
    with pytest.raises(ValueError, match="not valid JSON"):
        measure_format_consistency(reps, "not a json schema")


# ---------------------------------------------------------------------------
# Output consistency (end-to-end with stub clients/agent)
# ---------------------------------------------------------------------------


class _StubAgent(Agent):
    """Returns a canned answer per task input."""

    def __init__(self, answer_map: dict[str, str]) -> None:
        self._answers = answer_map

    async def arun(self, task: Task) -> AgentResponse:
        return AgentResponse(answer=self._answers.get(task.input, "default"))


class _StubInfraClient(OpenAIClient):
    """OpenAIClient with the network methods replaced by canned outputs.

    We subclass so the type checker / measure_output_consistency type
    contract is satisfied (it expects an OpenAIClient), while overriding
    _achat_provider and aembed to return scripted data.
    """

    PROVIDER_NAME = "test"

    def __init__(self, *, chat_outputs: list[str], embeddings: list[list[float]]) -> None:
        # Skip the AsyncOpenAI init — we don't make real network calls.
        super(OpenAIClient, self).__init__(max_concurrent=1, max_retries=1)
        self._chat_outputs = list(chat_outputs)
        self._embeddings = list(embeddings)

    async def _achat_provider(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        del messages, kwargs
        text = self._chat_outputs.pop(0) if self._chat_outputs else "{}"
        return ChatResponse(
            text=text,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            cost_usd=Decimal("0"),
            model=model,
            finish_reason="stop",
        )

    async def aembed(  # type: ignore[override]
        self,
        texts: Any,
        *,
        model: str = "text-embedding-3-large",
    ) -> tuple[list[list[float]], TokenUsage, Decimal]:
        # Return one vector per text in input order.
        n = len(list(texts))
        if len(self._embeddings) < n:
            raise RuntimeError("scripted embeddings exhausted")
        out = self._embeddings[:n]
        self._embeddings = self._embeddings[n:]
        return out, TokenUsage(input_tokens=1, output_tokens=0), Decimal("0")


def test_output_consistency_end_to_end_identical_answers() -> None:
    """K=3 paraphrases all yielding the same agent answer → max consistency.

    Hand-computed: 3 paraphrases x 1 answer each = 3 identical answers.
    Pairwise rubric (Likert 4 → normalized 1.0) for all C(3,2)=3 pairs → mean = 1.0.
    Pairwise embedding cosine of identical vectors = 1.0 → mean = 1.0.
    """
    paraphrases = ["q1", "q2", "q3"]
    chat_outputs = [
        # Generator: 3 paraphrases.
        '{"paraphrases": ["q1", "q2", "q3"]}',
        # Validator: yes x 3.
        '{"equivalent": true, "reason": "ok"}',
        '{"equivalent": true, "reason": "ok"}',
        '{"equivalent": true, "reason": "ok"}',
        # Rubric: 4 x C(3,2)=3 pairs.
        '{"score": 4, "reason": "same"}',
        '{"score": 4, "reason": "same"}',
        '{"score": 4, "reason": "same"}',
    ]
    embeddings = [[1.0, 0.0, 0.0]] * len(paraphrases)
    infra = _StubInfraClient(chat_outputs=chat_outputs, embeddings=embeddings)
    agent = _StubAgent(dict.fromkeys(paraphrases, "ANSWER"))
    task = Task(id="t1", domain="d", input="original?")

    result = asyncio.run(
        measure_output_consistency(task=task, agent=agent, infra_client=infra, k=3, seed=0)
    )
    assert isinstance(result, OutputConsistencyResult)
    assert result.k == 3
    assert len(result.rubric_scores) == 3  # C(3,2)
    assert result.mean_rubric == pytest.approx(1.0)
    assert result.mean_embedding_cosine == pytest.approx(1.0)
    assert result.paraphrase_rejection_rate == 0.0
    # Both CIs must be populated — the embedding CI is the secondary
    # reported metric per METHODOLOGY §1.1 and CLAUDE.md "Confidence
    # intervals on everything".
    assert result.rubric_ci is not None
    assert result.embedding_ci is not None
    assert result.embedding_ci.point_estimate == pytest.approx(1.0)


def test_output_consistency_rubric_normalization() -> None:
    """All pairs scored 2/4 → normalized 0.5; mean rubric = 0.5."""
    chat_outputs = [
        '{"paraphrases": ["q1", "q2", "q3"]}',
        '{"equivalent": true, "reason": "ok"}',
        '{"equivalent": true, "reason": "ok"}',
        '{"equivalent": true, "reason": "ok"}',
        # C(3,2)=3 pairs, all scored 2/4 → normalized 0.5.
        '{"score": 2, "reason": "partial"}',
        '{"score": 2, "reason": "partial"}',
        '{"score": 2, "reason": "partial"}',
    ]
    # Three orthogonal vectors → all pairwise cosines = 0.
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    infra = _StubInfraClient(chat_outputs=chat_outputs, embeddings=embeddings)
    agent = _StubAgent({"q1": "A", "q2": "B", "q3": "C"})
    task = Task(id="t1", domain="d", input="original?")
    result = asyncio.run(
        measure_output_consistency(task=task, agent=agent, infra_client=infra, k=3, seed=0)
    )
    assert result.mean_rubric == pytest.approx(2 / LIKERT_MAX)
    assert result.mean_embedding_cosine == pytest.approx(0.0)


def test_output_consistency_invalid_k_raises() -> None:
    task = Task(id="t1", domain="d", input="x")
    agent = _StubAgent({})
    infra = _StubInfraClient(chat_outputs=[], embeddings=[])
    with pytest.raises(ValueError, match="k >= 3"):
        asyncio.run(
            measure_output_consistency(task=task, agent=agent, infra_client=infra, k=2, seed=0)
        )


def test_output_consistency_substitutes_empty_answers() -> None:
    """Empty agent responses must not crash the embedding call.

    OpenAI's embedding endpoint rejects ``input=[""]`` with HTTP 400.
    Real-world Gemini target runs hit safety filters that produce empty
    text on hard prompts; the metric must substitute a placeholder so
    the bootstrap CI absorbs the empty-response signal rather than
    crashing the whole run.
    """
    chat_outputs = [
        '{"paraphrases": ["q1", "q2", "q3"]}',
        '{"equivalent": true, "reason": "ok"}',
        '{"equivalent": true, "reason": "ok"}',
        '{"equivalent": true, "reason": "ok"}',
        '{"score": 1, "reason": "low"}',
        '{"score": 1, "reason": "low"}',
        '{"score": 4, "reason": "high"}',
    ]
    embeddings = [[1.0, 0.0, 0.0]] * 3
    infra = _StubInfraClient(chat_outputs=chat_outputs, embeddings=embeddings)
    # Agent returns "" for q1 (simulating a safety-filter block) and real
    # answers for q2, q3.
    agent = _StubAgent({"q1": "", "q2": "real answer", "q3": "real answer"})
    task = Task(id="t1", domain="d", input="original?")

    result = asyncio.run(
        measure_output_consistency(task=task, agent=agent, infra_client=infra, k=3, seed=0)
    )
    assert result.n_empty_answers == 1
    # The metric still returned a value rather than crashing.
    assert result.mean_rubric is not None
    assert result.mean_embedding_cosine is not None
