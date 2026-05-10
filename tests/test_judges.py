"""Tests for steadfast.judges — Verdict, ExactMatchJudge, RubricJudge."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from steadfast.agent import AgentResponse, GroundTruth, Task
from steadfast.judges import (
    ExactMatchJudge,
    JudgeParseError,
    RubricJudge,
    Verdict,
    build_default_judge,
    canonicalize,
    judge_run_result,
)
from steadfast.models.base import (
    BaseModelClient,
    ChatMessage,
    ChatResponse,
    TokenUsage,
)
from steadfast.runner import RepRecord, RepStatus, RunResult

# ---------------------------------------------------------------------------
# Verdict — schema and frozen-ness.
# ---------------------------------------------------------------------------


def test_verdict_score_must_be_in_unit_interval() -> None:
    Verdict(score=0.0, passed=False, reason="r")
    Verdict(score=1.0, passed=True, reason="r")
    with pytest.raises(ValidationError):
        Verdict(score=-0.01, passed=False, reason="r")
    with pytest.raises(ValidationError):
        Verdict(score=1.01, passed=True, reason="r")


def test_verdict_is_frozen() -> None:
    v = Verdict(score=0.5, passed=False, reason="r")
    with pytest.raises(ValidationError):
        v.score = 0.7  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExactMatchJudge canonicalization rules (ADR-0003 §B.3).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("30 days", "30 days"),
        ("30 Days", "30 days"),
        ("  30   days  ", "30 days"),
        ("30 days.", "30 days"),
        ("30 days!?", "30 days"),
        ("30\tdays\n", "30 days"),
        # NFKC canonicalizes fullwidth digits (U+FF13 / U+FF10) to their
        # ASCII counterparts. Written as escape sequences so the source
        # stays ASCII (ruff confusable lints clean) but the test still
        # exercises the unicode path at runtime.
        ("\uff13\uff10 days", "30 days"),
        # casefold collapses German eszett (U+00DF) to "ss".
        ("Stra\u00dfe", "strasse"),
        # Hyphenation clarification fix 2026-05-11 (WEEK_2.md \u00a7O.1):
        # ASCII hyphens between word characters become a single space.
        ("30-day", "30 day"),
        ("30-Day", "30 day"),
        ("multi-step process", "multi step process"),
        ("state-of-the-art", "state of the art"),
        # Hyphen NOT between word characters is preserved (leading/
        # trailing dash, double-hyphen em-dash substitute).
        ("-foo", "-foo"),
        ("foo-", "foo-"),
        ("foo--bar", "foo--bar"),
    ],
)
def test_canonicalize_rules(raw: str, expected: str) -> None:
    assert canonicalize(raw) == expected


def test_canonicalize_idempotent_punctuation_strip() -> None:
    """Repeated trailing punctuation is fully stripped."""
    assert canonicalize("answer.!?") == "answer"
    assert canonicalize("answer; .  ") == "answer"


def test_canonicalize_hyphen_then_whitespace_collapse() -> None:
    """Hyphen-replacement runs before whitespace collapse.

    Without the order guarantee, ``"30-day  return"`` would become
    ``"30 day  return"`` and then collapse \u2014 which is what we want, but
    the test pins the contract so a future refactor that reorders the
    rules can't silently break it.
    """
    assert canonicalize("30-day  return") == "30 day return"
    assert canonicalize("foo-bar\tbaz") == "foo bar baz"


# ---------------------------------------------------------------------------
# ExactMatchJudge — substring containment after canonicalization.
# ---------------------------------------------------------------------------


def _exact_task(value: str = "30 days") -> Task:
    return Task(
        id="t1",
        domain="d",
        input="x",
        ground_truth=GroundTruth(kind="exact", value=value),
        judge="exact_match",
    )


@pytest.mark.parametrize(
    ("answer", "passed"),
    [
        ("30 days", True),
        ("The return window is 30 days for unopened items.", True),
        ("30 Days.", True),
        ("Thirty days", False),
        ("60 days", False),
        ("", False),
    ],
)
def test_exact_match_judge_pass_fail(answer: str, passed: bool) -> None:
    task = _exact_task()
    judge = ExactMatchJudge()
    verdict = asyncio.run(judge.ajudge(task, AgentResponse(answer=answer)))
    assert verdict.passed is passed
    assert verdict.score == (1.0 if passed else 0.0)


@pytest.mark.parametrize(
    "answer",
    [
        # The actual GPT-5.2 response text from the 2026-05-10 pilot run on
        # `pilot_001`. Pre-fix this scored 0/10 against ground truth
        # `"30 days"`; post-fix (canonicalize hyphens, ground truth tightened
        # to `"30-day"` which canonicalizes to `"30 day"`) it scores 10/10.
        "Our store offers a 30-day return window for unopened items.",
        # Plural form — also passes because `"30 day"` is a substring of
        # `"30 days"` (the trailing s on the answer doesn't break containment).
        "The return window is 30 days for unopened items.",
        # Unhyphenated singular form.
        "Returns accepted within 30 day window.",
    ],
)
def test_exact_match_judge_pilot_001_hyphenation_regression(answer: str) -> None:
    """Regression for the 2026-05-10 pilot-run finding (WEEK_2.md §O.1).

    Ground truth `"30-day"` canonicalizes to `"30 day"`. The hyphenation
    canonicalize fix means the singular hyphen-adjective form (which is
    how GPT-5.2 phrased its answer 10/10 reps) now matches; the plural
    form continues to match via substring containment. The mismatch case
    (different number of days) still fails — see the negative-case
    parametrize in :func:`test_exact_match_judge_pass_fail`.
    """
    task = _exact_task("30-day")
    verdict = asyncio.run(ExactMatchJudge().ajudge(task, AgentResponse(answer=answer)))
    assert verdict.passed is True
    assert verdict.score == 1.0


def test_exact_match_judge_rejects_missing_ground_truth() -> None:
    task = Task(id="t1", domain="d", input="x", ground_truth=None, judge="exact_match")
    judge = ExactMatchJudge()
    with pytest.raises(ValueError, match="kind='exact'"):
        asyncio.run(judge.ajudge(task, AgentResponse(answer="anything")))


def test_exact_match_judge_rejects_rubric_ground_truth() -> None:
    """Mis-routing (kind='rubric' on an exact_match task) raises rather than silently scoring 0."""
    task = Task(
        id="t1",
        domain="d",
        input="x",
        ground_truth=GroundTruth(kind="rubric", value="some criteria"),
        judge="exact_match",
    )
    with pytest.raises(ValueError, match="kind='exact'"):
        asyncio.run(ExactMatchJudge().ajudge(task, AgentResponse(answer="x")))


# ---------------------------------------------------------------------------
# RubricJudge — Pydantic parse + 1 retry + raise.
# ---------------------------------------------------------------------------


class _ScriptedClient(BaseModelClient):
    """BaseModelClient stub that returns canned outputs in sequence."""

    PROVIDER_NAME = "test"

    def __init__(self, *, outputs: list[str]) -> None:
        super().__init__(max_concurrent=1, max_retries=1)
        self._outputs = list(outputs)
        self.calls = 0

    async def _achat_provider(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        del messages, kwargs
        self.calls += 1
        text = self._outputs.pop(0) if self._outputs else "{}"
        return ChatResponse(
            text=text,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            cost_usd=Decimal("0.0001"),
            model=model,
            finish_reason="stop",
        )


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same as test_models — force retry waits to zero."""
    from tenacity import wait_none

    monkeypatch.setattr("steadfast.models.base.wait_exponential", lambda **_kw: wait_none())


def _rubric_task() -> Task:
    return Task(
        id="t1",
        domain="d",
        input="What is 2+2?",
        ground_truth=GroundTruth(kind="rubric", value="The answer must be 4."),
        judge="rubric",
    )


def test_rubric_judge_parses_first_attempt() -> None:
    canned = '{"score": 1.0, "passed": true, "reason": "correct"}'
    client = _ScriptedClient(outputs=[canned])
    judge = RubricJudge(client=client, model="gpt-test")
    verdict = asyncio.run(judge.ajudge(_rubric_task(), AgentResponse(answer="4")))
    assert verdict.score == 1.0
    assert verdict.passed is True
    assert client.calls == 1


def test_rubric_judge_extracts_json_from_surrounding_prose() -> None:
    """LLM emits prose alongside the JSON object — extraction must still parse."""
    prose = (
        "Sure, here is my verdict:\n\n"
        '{"score": 0.8, "passed": true, "reason": "mostly correct"}\n\n'
        "Let me know if you need clarification."
    )
    client = _ScriptedClient(outputs=[prose])
    judge = RubricJudge(client=client, model="gpt-test")
    verdict = asyncio.run(judge.ajudge(_rubric_task(), AgentResponse(answer="4")))
    assert verdict.score == 0.8
    assert verdict.passed is True
    assert client.calls == 1


def test_rubric_judge_strips_code_fences() -> None:
    """Real LLMs often wrap JSON in ```json fences — judge must strip them."""
    canned = '```json\n{"score": 0.7, "passed": true, "reason": "mostly right"}\n```'
    client = _ScriptedClient(outputs=[canned])
    judge = RubricJudge(client=client, model="gpt-test")
    verdict = asyncio.run(judge.ajudge(_rubric_task(), AgentResponse(answer="4")))
    assert verdict.score == 0.7
    assert client.calls == 1


def test_rubric_judge_retries_once_on_invalid_json() -> None:
    """ADR-0003 §B.4: invalid output triggers exactly one retry, then succeeds."""
    bad = "not json at all"
    good = '{"score": 0.5, "passed": false, "reason": "partial"}'
    client = _ScriptedClient(outputs=[bad, good])
    judge = RubricJudge(client=client, model="gpt-test")
    verdict = asyncio.run(judge.ajudge(_rubric_task(), AgentResponse(answer="3")))
    assert verdict.score == 0.5
    assert verdict.passed is False
    assert client.calls == 2  # one bad + one good


def test_rubric_judge_raises_after_two_failures() -> None:
    """ADR-0003 §B.4: a second parse failure raises rather than soft-failing."""
    client = _ScriptedClient(outputs=["junk", "still junk"])
    judge = RubricJudge(client=client, model="gpt-test")
    with pytest.raises(JudgeParseError):
        asyncio.run(judge.ajudge(_rubric_task(), AgentResponse(answer="x")))
    assert client.calls == 2


def test_rubric_judge_does_not_double_substitute_placeholders() -> None:
    """A task input containing a literal ``{rubric}`` must NOT cause the rubric
    text to be injected at two locations after rendering. Single-pass
    substitution prevents the silent prompt-corruption bug.
    """
    from steadfast._llm_parsing import load_prompt
    from steadfast.judges.rubric import _render_prompt

    template = load_prompt("rubric_v1.txt")
    task = Task(
        id="t1",
        domain="d",
        input="Please summarize: {rubric} should appear once.",
        ground_truth=GroundTruth(kind="rubric", value="RUBRIC_TEXT"),
        judge="rubric",
    )
    response = AgentResponse(answer="ANSWER_TEXT")
    rendered = _render_prompt(template=template, task=task, response=response)

    # The rubric text must appear exactly once (in the RUBRIC: section),
    # NOT a second time inside the substituted task input.
    assert rendered.count("RUBRIC_TEXT") == 1
    # The literal "{rubric}" from the task input survives as itself.
    assert "{rubric} should appear once." in rendered


def test_rubric_judge_validation_error_on_score_out_of_range() -> None:
    """A judge that returns score=1.5 fails Pydantic validation, gets retried, and on second
    failure raises JudgeParseError. This guards against the metric pipeline silently
    accepting nonsense scores."""
    client = _ScriptedClient(
        outputs=[
            '{"score": 1.5, "passed": true, "reason": "x"}',
            '{"score": 2.0, "passed": true, "reason": "x"}',
        ]
    )
    judge = RubricJudge(client=client, model="gpt-test")
    with pytest.raises(JudgeParseError):
        asyncio.run(judge.ajudge(_rubric_task(), AgentResponse(answer="x")))


# ---------------------------------------------------------------------------
# build_default_judge — task.judge dispatch.
# ---------------------------------------------------------------------------


def test_build_default_judge_exact_match() -> None:
    judge = build_default_judge(_exact_task())
    assert isinstance(judge, ExactMatchJudge)


def test_build_default_judge_rubric_requires_client() -> None:
    with pytest.raises(ValueError, match="rubric_client"):
        build_default_judge(_rubric_task())


def test_build_default_judge_rubric_with_client() -> None:
    client = _ScriptedClient(outputs=[])
    judge = build_default_judge(_rubric_task(), rubric_client=client, rubric_model="gpt-test")
    assert isinstance(judge, RubricJudge)


# ---------------------------------------------------------------------------
# judge_run_result — batch dispatcher attaches verdicts and skips failed reps.
# ---------------------------------------------------------------------------


def _build_run_result(*, statuses: list[RepStatus]) -> RunResult:
    task = _exact_task("30 days")
    reps = [
        RepRecord(
            run_id="r",
            task_id=task.id,
            rep_idx=i,
            status=s,
            response=(AgentResponse(answer="30 days") if s == RepStatus.COMPLETED else None),
        )
        for i, s in enumerate(statuses)
    ]
    return RunResult(run_id="r", task=task, reps=reps)


def test_judge_run_result_attaches_verdicts_to_completed_reps() -> None:
    rr = _build_run_result(statuses=[RepStatus.COMPLETED, RepStatus.FAILED, RepStatus.COMPLETED])
    asyncio.run(judge_run_result(rr))
    assert rr.reps[0].verdict is not None
    assert rr.reps[0].verdict.passed
    assert rr.reps[1].verdict is None  # failed rep is left unjudged
    assert rr.reps[2].verdict is not None


def test_judge_run_result_swallows_judge_errors_at_batch_boundary(
    monkeypatch: pytest.MonkeyPatch,
    memory_exporter: Any,
) -> None:
    """A judge that raises JudgeParseError on rep 0 must not abort scoring of rep 1.

    Also verifies the failed rep's ``score`` span is marked ERROR (per
    ADR-0003 §B.4 "failures are signal" — a green span on a judge
    failure would hide the signal in Phoenix).
    """
    from opentelemetry.trace import StatusCode

    rr = _build_run_result(statuses=[RepStatus.COMPLETED, RepStatus.COMPLETED])

    call_count = {"n": 0}

    async def _flaky_ajudge(self: Any, task: Task, response: AgentResponse) -> Verdict:
        del self, task, response
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise JudgeParseError("simulated parse failure")
        return Verdict(score=1.0, passed=True, reason="ok")

    monkeypatch.setattr(ExactMatchJudge, "ajudge", _flaky_ajudge)

    asyncio.run(judge_run_result(rr))
    assert rr.reps[0].verdict is None  # judge raised; verdict left None
    assert rr.reps[1].verdict is not None
    assert rr.reps[1].verdict.passed

    # Verify the failed rep's score span is ERROR; the successful one is OK.
    score_spans = [s for s in memory_exporter.get_finished_spans() if s.name == "score exact_match"]
    statuses = [s.status.status_code for s in score_spans]
    assert StatusCode.ERROR in statuses
    assert any(s != StatusCode.ERROR for s in statuses)
