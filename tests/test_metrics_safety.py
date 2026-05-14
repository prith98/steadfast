"""Tests for steadfast.metrics.safety — catastrophic rate + refusal F1.

End-to-end tests use stub agents + a stub safety judge so the metric
behavior is hand-verifiable on known input distributions. Bootstrap CI
inclusion (point estimate within bounds) replaces exact-bound checks
where the bounds depend on the random seed.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from steadfast.agent import Agent, AgentResponse, Task
from steadfast.judges.base import JudgeParseError, Verdict
from steadfast.metrics.safety import (
    SafetyBank,
    SafetyCase,
    SafetyCaseResult,
    _compute_f1_components,
    _f1_from_components,
    _safety_case_to_task,
    load_safety_bank,
    measure_catastrophic_rate,
    measure_refusal_f1,
    measure_safety,
    write_safety_bank,
)
from steadfast.stats.bootstrap import BootstrapCI
from steadfast.stats.wilson import WilsonCI

FIXTURE_BANK_PATH = Path(__file__).parent / "fixtures" / "safety_bank_audited.json"


# ---------------------------------------------------------------------------
# SafetyCase / SafetyBank validation
# ---------------------------------------------------------------------------


def test_safety_case_trap_requires_category() -> None:
    with pytest.raises(ValidationError, match="must have a category"):
        SafetyCase(id="t1", kind="trap", prompt="x", expected_harmful_artifact="y")


def test_safety_case_trap_category_must_be_in_taxonomy() -> None:
    with pytest.raises(ValidationError, match=r"not in the v0\.1 taxonomy"):
        SafetyCase(
            id="t1",
            kind="trap",
            prompt="x",
            category="weapons",
            expected_harmful_artifact="y",
        )


def test_safety_case_trap_requires_harmful_artifact() -> None:
    with pytest.raises(ValidationError, match="expected_harmful_artifact"):
        SafetyCase(id="t1", kind="trap", prompt="x", category="data_destruction")


def test_safety_case_benign_forbids_category() -> None:
    with pytest.raises(ValidationError, match="must have category=None"):
        SafetyCase(id="b1", kind="benign", prompt="x", category="data_destruction")


def test_safety_case_benign_forbids_harmful_artifact() -> None:
    with pytest.raises(ValidationError, match="must have expected_harmful_artifact=None"):
        SafetyCase(id="b1", kind="benign", prompt="x", expected_harmful_artifact="y")


def test_safety_case_trap_valid() -> None:
    case = SafetyCase(
        id="t1",
        kind="trap",
        prompt="bad request",
        category="data_destruction",
        expected_harmful_artifact="rm -rf",
    )
    assert case.kind == "trap"
    assert case.category == "data_destruction"


def test_safety_case_benign_valid() -> None:
    case = SafetyCase(id="b1", kind="benign", prompt="list files")
    assert case.kind == "benign"
    assert case.category is None


def test_safety_bank_traps_benigns_partition() -> None:
    bank = SafetyBank(
        review_status="reviewed",
        cases=[
            SafetyCase(
                id="t1",
                kind="trap",
                prompt="x",
                category="data_destruction",
                expected_harmful_artifact="y",
            ),
            SafetyCase(id="b1", kind="benign", prompt="x"),
            SafetyCase(
                id="t2",
                kind="trap",
                prompt="x",
                category="illegal_advice",
                expected_harmful_artifact="y",
            ),
        ],
    )
    assert len(bank.traps) == 2
    assert len(bank.benigns) == 1
    assert bank.traps[0].id == "t1"
    assert bank.benigns[0].id == "b1"


def test_load_safety_bank_refuses_draft(tmp_path: Path) -> None:
    bank = SafetyBank(review_status="draft", cases=[])
    bank_path = tmp_path / "bank.json"
    write_safety_bank(bank, bank_path)
    with pytest.raises(ValueError, match="review_status='draft'"):
        load_safety_bank(bank_path)


def test_load_safety_bank_accepts_reviewed(tmp_path: Path) -> None:
    bank = SafetyBank(review_status="reviewed", cases=[])
    bank_path = tmp_path / "bank.json"
    write_safety_bank(bank, bank_path)
    loaded = load_safety_bank(bank_path)
    assert loaded.review_status == "reviewed"


def test_load_safety_bank_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="safety bank not found"):
        load_safety_bank(tmp_path / "nope.json")


def test_committed_draft_bank_fails_loud() -> None:
    """The committed production bank under benchmarks/safety/ ships as
    draft per ADR-0007 §G; loading it must fail-loud until the
    operator flips review_status."""
    bank_path = Path(__file__).parent.parent / "benchmarks" / "safety" / "cases_v1.json"
    # Defensive: skip if the file isn't present in this checkout.
    if not bank_path.is_file():
        pytest.skip("committed safety bank not present in checkout")
    payload = json.loads(bank_path.read_text())
    if payload.get("review_status") != "draft":
        pytest.skip(
            "committed safety bank is no longer 'draft' — operator audit has been "
            "completed; this fail-loud test was for the pre-audit state"
        )
    with pytest.raises(ValueError, match="review_status='draft'"):
        load_safety_bank(bank_path)


def test_load_fixture_bank() -> None:
    """The audited test fixture under tests/fixtures loads cleanly."""
    bank = load_safety_bank(FIXTURE_BANK_PATH)
    assert bank.review_status == "reviewed"
    assert len(bank.traps) == 3
    assert len(bank.benigns) == 3
    categories = {c.category for c in bank.traps}
    assert categories == {"data_destruction", "illegal_advice", "social_engineering"}


# ---------------------------------------------------------------------------
# F1 helper math
# ---------------------------------------------------------------------------


def test_f1_components_all_true_positive() -> None:
    tp, fp, fn, tn = _compute_f1_components([True, True, True], [True, True, True])
    assert (tp, fp, fn, tn) == (3, 0, 0, 0)


def test_f1_components_mixed() -> None:
    # truth = [T, T, F, F], pred = [T, F, T, F]
    # TP=1 (T,T), FN=1 (T,F), FP=1 (F,T), TN=1 (F,F)
    tp, fp, fn, tn = _compute_f1_components([True, True, False, False], [True, False, True, False])
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)


def test_f1_perfect_returns_one() -> None:
    assert _f1_from_components(10, 0, 0) == pytest.approx(1.0)


def test_f1_zero_tp_with_other_positives_returns_zero() -> None:
    """sklearn convention: F1 is 0 (not None) when tp=0 but fp/fn>0."""
    assert _f1_from_components(0, 5, 0) == pytest.approx(0.0)
    assert _f1_from_components(0, 0, 5) == pytest.approx(0.0)


def test_f1_undefined_when_no_positives() -> None:
    """F1 is undefined (None) when tp+fp+fn == 0 — no refusals predicted
    and no truth positives."""
    assert _f1_from_components(0, 0, 0) is None


def test_f1_half_half() -> None:
    # tp=5, fp=5, fn=5 → p=0.5, r=0.5, F1=0.5
    assert _f1_from_components(5, 5, 5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# measure_catastrophic_rate
# ---------------------------------------------------------------------------


def _case(
    *,
    case_id: str = "c1",
    kind: str = "trap",
    category: str | None = "data_destruction",
    refused: bool | None = False,
    harmful: bool | None = False,
) -> SafetyCaseResult:
    return SafetyCaseResult(
        case_id=case_id,
        case_kind=kind,  # type: ignore[arg-type]
        category=category if kind == "trap" else None,
        refused=refused,
        harmful=harmful if kind == "trap" else None,
        answer_preview="",
    )


def test_catastrophic_rate_all_safe() -> None:
    cases = [_case(case_id=f"t{i}", harmful=False) for i in range(10)]
    result = measure_catastrophic_rate(cases)
    assert result.n_harmful == 0
    assert result.n_traps_judged == 10
    assert result.rate == pytest.approx(0.0)
    assert isinstance(result.ci, WilsonCI)
    # Wilson 95% CI on 0/10 — upper bound should be the documented
    # methodology width (~0.278 per ADR-0007 §E).
    assert result.ci.ci_lower == pytest.approx(0.0, abs=1e-9)
    assert 0.25 < result.ci.ci_upper < 0.31


def test_catastrophic_rate_all_harmful() -> None:
    cases = [_case(case_id=f"t{i}", harmful=True) for i in range(10)]
    result = measure_catastrophic_rate(cases)
    assert result.n_harmful == 10
    assert result.rate == pytest.approx(1.0)
    assert result.ci is not None
    # Wilson 95% CI on 10/10 — lower bound strictly < 1.0.
    assert result.ci.ci_lower < 1.0
    assert result.ci.ci_upper == pytest.approx(1.0, abs=1e-9)


def test_catastrophic_rate_half_half() -> None:
    cases = [_case(case_id=f"t{i}", harmful=(i % 2 == 0)) for i in range(10)]
    result = measure_catastrophic_rate(cases)
    assert result.n_harmful == 5
    assert result.rate == pytest.approx(0.5)
    assert result.ci is not None


def test_catastrophic_rate_skips_unjudged_traps() -> None:
    cases = [
        _case(case_id="t1", harmful=True),
        _case(case_id="t2", harmful=None),  # judge failed
        _case(case_id="t3", harmful=False),
    ]
    result = measure_catastrophic_rate(cases)
    assert result.n_traps_total == 3
    assert result.n_traps_judged == 2
    assert result.n_harmful == 1
    assert result.rate == pytest.approx(0.5)


def test_catastrophic_rate_skips_benign_cases() -> None:
    cases = [
        _case(case_id="t1", harmful=True),
        _case(case_id="b1", kind="benign", category=None, refused=False, harmful=None),
    ]
    result = measure_catastrophic_rate(cases)
    assert result.n_traps_total == 1
    assert result.n_harmful == 1


def test_catastrophic_rate_na_when_no_judged_traps() -> None:
    cases = [_case(case_id=f"t{i}", harmful=None) for i in range(3)]
    result = measure_catastrophic_rate(cases)
    assert result.rate is None
    assert result.ci is None
    assert result.reason is not None
    assert "no trap cases" in result.reason


# ---------------------------------------------------------------------------
# measure_refusal_f1
# ---------------------------------------------------------------------------


def test_refusal_f1_perfect() -> None:
    """All traps refused, all benigns answered → F1 = 1.0."""
    cases = [_case(case_id=f"t{i}", kind="trap", refused=True, harmful=False) for i in range(5)]
    cases.extend(
        _case(case_id=f"b{i}", kind="benign", category=None, refused=False) for i in range(5)
    )
    result = measure_refusal_f1(cases, seed=0)
    assert result.f1 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.tp == 5
    assert result.fp == 0
    assert result.fn == 0
    assert result.tn == 5
    assert isinstance(result.f1_ci, BootstrapCI)


def test_refusal_f1_all_refuse_inflates_fp() -> None:
    """Model that refuses everything → precision = 5/10 = 0.5, recall = 1."""
    cases = [_case(case_id=f"t{i}", kind="trap", refused=True, harmful=False) for i in range(5)]
    cases.extend(
        _case(case_id=f"b{i}", kind="benign", category=None, refused=True) for i in range(5)
    )
    result = measure_refusal_f1(cases, seed=0)
    assert result.tp == 5
    assert result.fp == 5
    assert result.fn == 0
    assert result.tn == 0
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_refusal_f1_all_comply_collapses_recall_to_zero() -> None:
    """Model that refuses nothing → tp=0, fp=0, fn=5 → F1 = 0.0."""
    cases = [_case(case_id=f"t{i}", kind="trap", refused=False, harmful=True) for i in range(5)]
    cases.extend(
        _case(case_id=f"b{i}", kind="benign", category=None, refused=False) for i in range(5)
    )
    result = measure_refusal_f1(cases, seed=0)
    assert result.tp == 0
    assert result.fn == 5
    assert result.f1 == pytest.approx(0.0)


def test_refusal_f1_excludes_agent_failed_cases() -> None:
    cases = [
        _case(case_id="t1", kind="trap", refused=True, harmful=False),
        _case(case_id="t2", kind="trap", refused=None, harmful=None),  # agent failed
        _case(case_id="b1", kind="benign", category=None, refused=False),
    ]
    result = measure_refusal_f1(cases, seed=0)
    assert result.n_total == 2  # t2 excluded
    assert result.tp == 1
    assert result.fp == 0


def test_refusal_f1_empty_pool_returns_na() -> None:
    result = measure_refusal_f1([], seed=0)
    assert result.f1 is None
    assert result.f1_ci is None
    assert result.reason is not None
    assert "at least one judged case" in result.reason


def test_refusal_f1_no_positives_anywhere_returns_na() -> None:
    """All benigns answered → tp=fp=fn=0 → F1 undefined."""
    cases = [_case(case_id=f"b{i}", kind="benign", category=None, refused=False) for i in range(5)]
    result = measure_refusal_f1(cases, seed=0)
    assert result.f1 is None
    assert result.reason is not None
    assert "undefined" in result.reason


def test_refusal_f1_bootstrap_ci_brackets_point() -> None:
    """The bootstrap CI must bracket the point estimate."""
    cases = [_case(case_id=f"t{i}", kind="trap", refused=(i < 4), harmful=False) for i in range(5)]
    cases.extend(
        _case(case_id=f"b{i}", kind="benign", category=None, refused=(i < 1)) for i in range(5)
    )
    result = measure_refusal_f1(cases, n_resamples=1000, seed=42)
    assert result.f1 is not None
    assert result.f1_ci is not None
    assert result.f1_ci.ci_lower <= result.f1 <= result.f1_ci.ci_upper


def test_refusal_f1_single_case_pool_returns_no_ci() -> None:
    """With n=1 in the post-filter pool, the bootstrap helper short-circuits
    (n<2 guard), F1 carries the point estimate, the CI is None.

    The HTML cell renderer is responsible for surfacing the value
    without the CI; this test pins the metric-layer contract that
    leaves both populated independently.
    """
    cases = [_case(case_id="t1", kind="trap", refused=True, harmful=False)]
    result = measure_refusal_f1(cases, n_resamples=200, seed=0)
    # tp=1, fp=0, fn=0 → F1=1.0 defined; CI undefined for n<2.
    assert result.f1 == pytest.approx(1.0)
    assert result.f1_ci is None
    assert result.tp == 1


def test_refusal_f1_bootstrap_handles_unbalanced_pool_without_crashing() -> None:
    """Heavily unbalanced pool (1 trap among 99 benigns) should still
    produce a valid F1 + CI rather than the bootstrap raising.

    The degenerate-threshold branch in ``_bootstrap_f1_ci`` is hard to
    trip from real input distributions (it fires only when >99% of
    resamples produce undefined F1). This test confirms that the
    happy path holds under the unbalanced regime, and that the CI
    brackets the point estimate.
    """
    cases = [_case(case_id="t1", kind="trap", refused=True, harmful=False)]
    cases.extend(
        _case(case_id=f"b{i}", kind="benign", category=None, refused=False) for i in range(99)
    )
    result = measure_refusal_f1(cases, n_resamples=1000, seed=7)
    assert result.f1 is not None
    assert result.f1_ci is not None
    assert result.f1_ci.ci_lower <= result.f1 <= result.f1_ci.ci_upper


def test_refusal_f1_deterministic_under_same_seed() -> None:
    cases = [_case(case_id=f"t{i}", kind="trap", refused=(i < 3), harmful=False) for i in range(5)]
    cases.extend(
        _case(case_id=f"b{i}", kind="benign", category=None, refused=(i < 2)) for i in range(5)
    )
    a = measure_refusal_f1(cases, n_resamples=500, seed=123)
    b = measure_refusal_f1(cases, n_resamples=500, seed=123)
    assert a.f1_ci is not None
    assert b.f1_ci is not None
    assert a.f1_ci.ci_lower == b.f1_ci.ci_lower
    assert a.f1_ci.ci_upper == b.f1_ci.ci_upper


# ---------------------------------------------------------------------------
# _safety_case_to_task
# ---------------------------------------------------------------------------


def test_safety_case_to_task_trap() -> None:
    case = SafetyCase(
        id="t1",
        kind="trap",
        prompt="bad",
        category="data_destruction",
        expected_harmful_artifact="rm -rf",
    )
    task = _safety_case_to_task(case, confidence_suffix="ANSWER: ...")
    assert task.id == "t1"
    assert task.judge == "safety_harmful"
    assert task.ground_truth is not None
    assert task.ground_truth.kind == "rubric"
    assert task.ground_truth.value == "rm -rf"
    assert task.difficulty == "hard"
    assert task.confidence_suffix == "ANSWER: ..."


def test_safety_case_to_task_benign() -> None:
    case = SafetyCase(id="b1", kind="benign", prompt="list files")
    task = _safety_case_to_task(case, confidence_suffix=None)
    assert task.id == "b1"
    assert task.judge == "exact_match"
    assert task.ground_truth is None
    assert task.difficulty == "normal"


# ---------------------------------------------------------------------------
# measure_safety — end-to-end with stub agent + stub judge
# ---------------------------------------------------------------------------


class _StaticAgent(Agent):
    """Agent that returns the same canned response for every task.

    Useful for testing the metric's reduce path independently of any
    real LLM. The response can be hand-shaped to exercise specific
    (refused, harmful) combinations.
    """

    def __init__(self, *, answer: str, refused: bool) -> None:
        self._answer = answer
        self._refused = refused
        self.calls: list[Task] = []

    async def arun(self, task: Task) -> AgentResponse:
        self.calls.append(task)
        return AgentResponse(
            answer=self._answer,
            refused=self._refused,
            raw_output=self._answer,
            cost_usd=Decimal("0"),
        )


class _PerCaseAgent(Agent):
    """Agent that maps case id → (answer, refused) for fine-grained fixtures."""

    def __init__(self, responses: dict[str, tuple[str, bool]]) -> None:
        self._responses = responses

    async def arun(self, task: Task) -> AgentResponse:
        ans, refused = self._responses[task.id]
        return AgentResponse(
            answer=ans,
            refused=refused,
            raw_output=ans,
            cost_usd=Decimal("0"),
        )


class _RaisingAgent(Agent):
    """Agent that raises on every call — exercises the failure path."""

    async def arun(self, task: Task) -> AgentResponse:
        raise RuntimeError(f"agent failure on {task.id}")


class _FakeClient:
    """Stand-in for a BaseModelClient — never invoked because SafetyJudge
    is monkey-patched in the tests that need it."""


class _AlwaysSafeJudge:
    """Stand-in SafetyJudge that always returns passed=True."""

    model = "fake-judge"

    async def ajudge(self, task: Task, response: AgentResponse) -> Verdict:
        return Verdict(score=1.0, passed=True, reason="stub: marked safe")


class _AlwaysHarmfulJudge:
    """Stand-in SafetyJudge that always returns passed=False."""

    model = "fake-judge"

    async def ajudge(self, task: Task, response: AgentResponse) -> Verdict:
        return Verdict(score=0.0, passed=False, reason="stub: marked harmful")


class _RaisingJudge:
    """Stand-in SafetyJudge that always raises."""

    model = "fake-judge"

    async def ajudge(self, task: Task, response: AgentResponse) -> Verdict:
        raise RuntimeError("judge failure")


def _patch_safety_judge(monkeypatch: pytest.MonkeyPatch, judge: Any) -> None:
    """Replace SafetyJudge construction in metrics.safety with a stub."""
    monkeypatch.setattr(
        "steadfast.metrics.safety.SafetyJudge",
        lambda *_, **__: judge,
    )


@pytest.mark.asyncio
async def test_measure_safety_perfect_resistor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent that refuses every trap and answers every benign → catastrophic 0, F1 1.0."""
    bank = load_safety_bank(FIXTURE_BANK_PATH)
    responses = {
        case.id: ("REFUSE" if case.kind == "trap" else "here you go", case.kind == "trap")
        for case in bank.cases
    }
    agent = _PerCaseAgent(responses)
    _patch_safety_judge(monkeypatch, _AlwaysSafeJudge())

    result = await measure_safety(
        model="test-model",
        bank=bank,
        agent=agent,
        safety_judge_client=_FakeClient(),  # type: ignore[arg-type]
        confidence_suffix=None,
        n_resamples=500,
    )
    assert result.catastrophic.rate == pytest.approx(0.0)
    assert result.refusal.f1 == pytest.approx(1.0)
    assert result.n_traps == 3
    assert result.n_benigns == 3
    assert result.n_agent_failures == 0
    assert result.n_judge_failures == 0


@pytest.mark.asyncio
async def test_measure_safety_complete_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent that complies with everything → catastrophic 1.0, F1 0."""
    bank = load_safety_bank(FIXTURE_BANK_PATH)
    agent = _StaticAgent(answer="sure, here is the harmful artifact", refused=False)
    _patch_safety_judge(monkeypatch, _AlwaysHarmfulJudge())

    result = await measure_safety(
        model="test-model",
        bank=bank,
        agent=agent,
        safety_judge_client=_FakeClient(),  # type: ignore[arg-type]
        confidence_suffix=None,
        n_resamples=500,
    )
    assert result.catastrophic.rate == pytest.approx(1.0)
    assert result.refusal.f1 == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_measure_safety_over_refuser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent that refuses every case → catastrophic 0, but F1 < 1 (false-positive
    refusals on benigns)."""
    bank = load_safety_bank(FIXTURE_BANK_PATH)
    agent = _StaticAgent(answer="REFUSE", refused=True)
    _patch_safety_judge(monkeypatch, _AlwaysSafeJudge())

    result = await measure_safety(
        model="test-model",
        bank=bank,
        agent=agent,
        safety_judge_client=_FakeClient(),  # type: ignore[arg-type]
        confidence_suffix=None,
        n_resamples=500,
    )
    assert result.catastrophic.rate == pytest.approx(0.0)
    # Refusing everything: precision = 3/6 = 0.5, recall = 1.0, F1 = 2/3.
    assert result.refusal.f1 == pytest.approx(2 / 3, abs=1e-3)


@pytest.mark.asyncio
async def test_measure_safety_judge_failure_drops_trap_from_catastrophic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge failures shrink the catastrophic denominator but don't kill the metric.

    Per ADR-0007 §F's broadened-Exception contract. Refusal F1 still
    sees the trap because it only needs ``response.refused``.
    """
    bank = load_safety_bank(FIXTURE_BANK_PATH)
    agent = _StaticAgent(answer="REFUSE", refused=True)
    _patch_safety_judge(monkeypatch, _RaisingJudge())

    result = await measure_safety(
        model="test-model",
        bank=bank,
        agent=agent,
        safety_judge_client=_FakeClient(),  # type: ignore[arg-type]
        confidence_suffix=None,
        n_resamples=200,
    )
    assert result.catastrophic.rate is None
    assert result.catastrophic.ci is None
    assert result.n_judge_failures == 3
    # Refusal F1 still computed: agent refused everything (and all
    # cases passed through), so we get the same over-refusal-style F1.
    assert result.refusal.f1 is not None


@pytest.mark.asyncio
async def test_measure_safety_agent_failure_drops_case_from_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When agent.arun raises, the case drops from both sub-metric pools."""
    bank = load_safety_bank(FIXTURE_BANK_PATH)
    agent = _RaisingAgent()
    _patch_safety_judge(monkeypatch, _AlwaysSafeJudge())

    result = await measure_safety(
        model="test-model",
        bank=bank,
        agent=agent,
        safety_judge_client=_FakeClient(),  # type: ignore[arg-type]
        confidence_suffix=None,
        n_resamples=200,
    )
    # Every case errored → both metrics surface their N/A paths.
    assert result.catastrophic.rate is None
    assert result.refusal.f1 is None
    assert result.n_agent_failures == 6
    # Every per_case row should carry the error annotation.
    assert all(c.error is not None for c in result.per_case)


@pytest.mark.asyncio
async def test_measure_safety_judge_parse_error_is_broadened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JudgeParseError (a JudgeError subclass) is also absorbed by the broad catch.

    This regression-tests the broadened-Exception contract — we don't
    just catch ``JudgeError``, we catch ``Exception``. A
    ``JudgeParseError`` raised after the retry budget exhausts must not
    take down the whole metric.
    """
    bank = load_safety_bank(FIXTURE_BANK_PATH)
    agent = _StaticAgent(answer="REFUSE", refused=True)

    class _ParseErrorJudge:
        model = "fake-judge"

        async def ajudge(self, task: Task, response: AgentResponse) -> Verdict:
            raise JudgeParseError("simulated parse failure")

    _patch_safety_judge(monkeypatch, _ParseErrorJudge())

    result = await measure_safety(
        model="test-model",
        bank=bank,
        agent=agent,
        safety_judge_client=_FakeClient(),  # type: ignore[arg-type]
        confidence_suffix=None,
        n_resamples=200,
    )
    assert result.n_judge_failures == 3
    assert result.catastrophic.rate is None
    # Refusal F1 should still compute because refused flags are read
    # off the AgentResponse independently of the judge.
    assert result.refusal.f1 is not None


@pytest.mark.asyncio
async def test_measure_safety_writes_per_case_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    bank = load_safety_bank(FIXTURE_BANK_PATH)
    agent = _StaticAgent(answer="here it is", refused=False)
    _patch_safety_judge(monkeypatch, _AlwaysHarmfulJudge())

    result = await measure_safety(
        model="test-model",
        bank=bank,
        agent=agent,
        safety_judge_client=_FakeClient(),  # type: ignore[arg-type]
        confidence_suffix=None,
        n_resamples=200,
    )
    assert len(result.per_case) == 6
    trap_rows = [r for r in result.per_case if r.case_kind == "trap"]
    benign_rows = [r for r in result.per_case if r.case_kind == "benign"]
    assert len(trap_rows) == 3
    assert len(benign_rows) == 3
    # Trap rows carry harmful labels; benign rows leave it None.
    assert all(r.harmful is True for r in trap_rows)
    assert all(r.harmful is None for r in benign_rows)
    # Every row has the answer preview.
    assert all(r.answer_preview for r in result.per_case)


# ---------------------------------------------------------------------------
# Cost aggregation (v0.1.x clarification fix, ADR-0008 §G item 1)
# ---------------------------------------------------------------------------


class _CostedAgent(Agent):
    """Agent that returns a configured per-task cost on every response."""

    def __init__(self, *, cost: Decimal | None) -> None:
        self._cost = cost

    async def arun(self, task: Task) -> AgentResponse:
        return AgentResponse(
            answer="ok",
            refused=False,
            raw_output="ok",
            cost_usd=self._cost,
        )


@pytest.mark.asyncio
async def test_measure_safety_aggregates_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-case cost flows from AgentResponse through SafetyCaseResult into the dimension total."""
    bank = load_safety_bank(FIXTURE_BANK_PATH)
    agent = _CostedAgent(cost=Decimal("0.012"))
    _patch_safety_judge(monkeypatch, _AlwaysSafeJudge())

    result = await measure_safety(
        model="test-model",
        bank=bank,
        agent=agent,
        safety_judge_client=_FakeClient(),  # type: ignore[arg-type]
        confidence_suffix=None,
        n_resamples=200,
    )
    # 6 cases * 0.012 = 0.072
    assert result.total_cost_usd == Decimal("0.072")
    # Every per-case row carries the cost.
    assert all(c.cost_usd == Decimal("0.012") for c in result.per_case)


@pytest.mark.asyncio
async def test_measure_safety_cost_none_when_unpopulated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agents that don't expose cost_usd surface as total_cost_usd=None, not 0."""
    bank = load_safety_bank(FIXTURE_BANK_PATH)
    agent = _CostedAgent(cost=None)
    _patch_safety_judge(monkeypatch, _AlwaysSafeJudge())

    result = await measure_safety(
        model="test-model",
        bank=bank,
        agent=agent,
        safety_judge_client=_FakeClient(),  # type: ignore[arg-type]
        confidence_suffix=None,
        n_resamples=200,
    )
    assert result.total_cost_usd is None
    assert all(c.cost_usd is None for c in result.per_case)


def test_safety_dimension_total_cost_excludes_failed_cases() -> None:
    """Failed-agent cases have cost_usd=None and don't affect the dimension sum."""
    per_case = [
        SafetyCaseResult(
            case_id="t1",
            case_kind="trap",
            category="data_destruction",
            refused=False,
            harmful=True,
            answer_preview="…",
            cost_usd=Decimal("0.05"),
        ),
        SafetyCaseResult(
            case_id="t2",
            case_kind="trap",
            category="data_destruction",
            refused=None,
            harmful=None,
            answer_preview="",
            error="RuntimeError: boom",
            cost_usd=None,
        ),
        SafetyCaseResult(
            case_id="b1",
            case_kind="benign",
            category=None,
            refused=False,
            harmful=None,
            answer_preview="ok",
            cost_usd=Decimal("0.02"),
        ),
    ]
    from steadfast.metrics.safety import (
        CatastrophicRateResult,
        RefusalF1Result,
        SafetyDimension,
    )

    dim = SafetyDimension(
        model="m",
        bank_version="v1",
        n_traps=2,
        n_benigns=1,
        catastrophic=CatastrophicRateResult(
            n_traps_total=2,
            n_traps_judged=1,
            n_harmful=1,
            rate=1.0,
            ci=None,
        ),
        refusal=RefusalF1Result(
            n_total=2,
            n_traps=1,
            n_benigns=1,
            tp=0,
            fp=0,
            fn=1,
            tn=1,
            precision=None,
            recall=0.0,
            f1=0.0,
            f1_ci=None,
            n_resamples=None,
            confidence_level=None,
            method=None,
        ),
        per_case=per_case,
    )
    assert dim.total_cost_usd == Decimal("0.07")
