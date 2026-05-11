"""Tests for steadfast.metrics.robustness — typo + distractor sub-metrics.

End-to-end tests use stub agents and the real :class:`ExactMatchJudge` so
the per-rep ``passed`` flag is deterministic on agent output. Stat
behavior (paired bootstrap CI) is exercised via the underlying primitive
in ``tests/test_paired_bootstrap.py``; here we verify the metric layer's
plumbing — per-rep distinct perturbations, n_tasks=1 N/A, multi-kind
aggregation.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from steadfast.agent import Agent, AgentResponse, GroundTruth, Task, ToolCall
from steadfast.judges.base import Verdict
from steadfast.metrics.robustness import (
    SUPPORTED_KINDS,
    ContradictionResult,
    RobustnessSubMetricResult,
    _aggregate_sub_metric,
    classify_contradiction_response,
    measure_contradiction_handling,
    measure_distractor_robustness,
    measure_robustness,
    measure_typo_robustness,
)
from steadfast.perturbations.contradiction import CORRUPTED_CALLS_METADATA_KEY
from steadfast.perturbations.distractor import DistractorBank, DistractorSnippet
from steadfast.runner import RepRecord, RepStatus, RunResult
from tests.fixtures.contradiction_agents import EchoToolAgent

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _exact_task(task_id: str, *, ground_truth: str, input_text: str = "") -> Task:
    return Task(
        id=task_id,
        domain="test_support",
        input=input_text or f"Q for {task_id}",
        ground_truth=GroundTruth(kind="exact", value=ground_truth),
        judge="exact_match",
    )


def _make_clean_run_result(
    task: Task, *, passes: list[bool], run_id: str = "run-test"
) -> RunResult:
    """Build a judged RunResult with one rep per ``passes`` entry."""
    reps: list[RepRecord] = []
    now = datetime.now(UTC)
    for i, passed in enumerate(passes):
        reps.append(
            RepRecord(
                run_id=run_id,
                task_id=task.id,
                rep_idx=i,
                status=RepStatus.COMPLETED,
                response=AgentResponse(
                    answer="GROUND_TRUTH" if passed else "WRONG",
                    raw_output="GROUND_TRUTH" if passed else "WRONG",
                    cost_usd=Decimal("0"),
                ),
                error=None,
                started_at=now,
                completed_at=now,
                cost_usd=Decimal("0"),
                verdict=Verdict(
                    score=1.0 if passed else 0.0,
                    passed=passed,
                    reason="test",
                ),
            )
        )
    return RunResult(run_id=run_id, task=task, reps=reps)


class _AlwaysPassAgent(Agent):
    """Agent that returns the task's ground-truth string verbatim.

    Combined with :class:`ExactMatchJudge`, every rep passes regardless
    of input perturbation — useful for the "perturbed arm matches clean"
    fixture (delta = 0).
    """

    def __init__(self, ground_truth: str) -> None:
        self._gt = ground_truth
        self.calls: list[str] = []

    async def arun(self, task: Task) -> AgentResponse:
        self.calls.append(task.input)
        return AgentResponse(
            answer=self._gt,
            raw_output=self._gt,
            cost_usd=Decimal("0"),
        )


class _AlwaysFailAgent(Agent):
    """Agent that emits a fixed wrong answer regardless of input.

    Combined with :class:`ExactMatchJudge`, every rep fails — the
    "perturbed arm fully degrades" fixture (delta = -1 when paired with
    an all-pass clean arm).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def arun(self, task: Task) -> AgentResponse:
        self.calls.append(task.input)
        return AgentResponse(
            answer="DEFINITELY_WRONG",
            raw_output="DEFINITELY_WRONG",
            cost_usd=Decimal("0"),
        )


class _PerInputAgent(Agent):
    """Agent that maps a substring marker in the input to a canned answer.

    Useful for "agent is brittle to a specific perturbation" fixtures —
    e.g., if any perturbed input contains ``"!"`` (a fictional brittleness
    trigger), the agent returns a wrong answer. Otherwise returns the
    ground truth.
    """

    def __init__(self, *, trigger: str, ground_truth: str) -> None:
        self._trigger = trigger
        self._gt = ground_truth
        self.calls: list[str] = []

    async def arun(self, task: Task) -> AgentResponse:
        self.calls.append(task.input)
        if self._trigger in task.input:
            return AgentResponse(
                answer="WRONG",
                raw_output="WRONG",
                cost_usd=Decimal("0"),
            )
        return AgentResponse(
            answer=self._gt,
            raw_output=self._gt,
            cost_usd=Decimal("0"),
        )


# ---------------------------------------------------------------------------
# Per-task behavior
# ---------------------------------------------------------------------------


def test_perturbed_arm_invokes_agent_n_times_with_distinct_inputs() -> None:
    """N=10 perturbed reps must hit ``agent.arun`` ten times with ten different inputs."""
    # Long-enough input that the 5%-rate perturbation has mutation budget
    # (n_target = round(0.05 * n_alphanum) > 0). Short prompts (few chars,
    # short words) collapse n_target to 0 and produce identical perturbed
    # copies — correct behavior per the per-word floor cap, but it would
    # mask the per-rep-seed signal we want to test here.
    task = _exact_task(
        "t_distinct",
        ground_truth="GROUND_TRUTH",
        input_text=(
            "What is the return window for unopened items at our store, "
            "and does the policy differ for opened items or items that have been "
            "used or have any visible signs of wear from regular customer use?"
        ),
    )
    agent = _AlwaysPassAgent("GROUND_TRUTH")
    clean = _make_clean_run_result(task, passes=[True] * 10)

    asyncio.run(
        measure_typo_robustness(
            tasks=[task],
            clean_run_results=[clean],
            agent=agent,
            rubric_client=None,
            reps=10,
        )
    )
    assert len(agent.calls) == 10
    distinct_inputs = set(agent.calls)
    # Per-rep seeding (ADR-0006 §B) should make essentially every input
    # distinct on a long-enough prompt; allow at most one collision for
    # short inputs where the typo perturbation might no-op.
    assert len(distinct_inputs) >= 9, f"only {len(distinct_inputs)} distinct inputs"


def test_typo_clean_pass_perturbed_pass_yields_zero_delta() -> None:
    """An agent that always passes regardless of perturbation gives delta = 0."""
    task = _exact_task("t_zero", ground_truth="GROUND_TRUTH", input_text="What policy applies?")
    agent = _AlwaysPassAgent("GROUND_TRUTH")
    clean = _make_clean_run_result(task, passes=[True] * 10)

    result = asyncio.run(
        measure_typo_robustness(
            tasks=[task],
            clean_run_results=[clean],
            agent=agent,
            rubric_client=None,
            reps=10,
        )
    )
    assert result.kind == "typo"
    assert result.n_tasks == 1
    assert result.clean_mean == pytest.approx(1.0)
    assert result.perturbed_mean == pytest.approx(1.0)
    assert result.delta == pytest.approx(0.0)
    assert len(result.per_task) == 1
    assert result.per_task[0].clean_passes == [True] * 10
    assert result.per_task[0].perturbed_passes == [True] * 10


def test_typo_clean_pass_perturbed_fail_yields_delta_minus_one() -> None:
    """ADR-0006 §F worst case: clean=1.0, perturbed=0.0 → delta = -1."""
    task = _exact_task("t_full_fail", ground_truth="GROUND_TRUTH")
    agent = _AlwaysFailAgent()
    clean = _make_clean_run_result(task, passes=[True] * 10)

    result = asyncio.run(
        measure_typo_robustness(
            tasks=[task],
            clean_run_results=[clean],
            agent=agent,
            rubric_client=None,
            reps=10,
        )
    )
    assert result.delta == pytest.approx(-1.0)
    assert result.per_task[0].clean_rate == pytest.approx(1.0)
    assert result.per_task[0].perturbed_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# n_tasks=1 N/A path (ADR-0004 §G surface)
# ---------------------------------------------------------------------------


def test_n_tasks_1_skips_ci_with_reason() -> None:
    """Single-task measurement reports point estimate but N/As the CI."""
    task = _exact_task("t_single", ground_truth="GROUND_TRUTH")
    agent = _AlwaysPassAgent("GROUND_TRUTH")
    clean = _make_clean_run_result(task, passes=[True] * 10)
    result = asyncio.run(
        measure_typo_robustness(
            tasks=[task],
            clean_run_results=[clean],
            agent=agent,
            rubric_client=None,
            reps=10,
        )
    )
    assert result.n_tasks == 1
    assert result.delta is not None
    assert result.delta_ci_lower is None
    assert result.delta_ci_upper is None
    assert result.confidence_level is None
    assert result.method is None
    assert result.reason is not None
    assert "n_tasks >= 2" in result.reason


def test_n_tasks_0_returns_full_na() -> None:
    """Zero tasks (e.g., distractor with no banks) returns a fully-N/A sub-metric."""
    result = _aggregate_sub_metric(kind="typo", per_task=[], seed=0)
    assert result.n_tasks == 0
    assert result.clean_mean is None
    assert result.delta is None
    assert result.reason == "no tasks measured"


# ---------------------------------------------------------------------------
# Cross-task aggregation with paired bootstrap
# ---------------------------------------------------------------------------


def test_three_task_aggregate_brackets_realizable_delta() -> None:
    """Three-task happy-path aggregation with a robust agent (delta = 0).

    The always-pass agent makes every perturbed rep succeed, so per-task
    delta is exactly zero across all three tasks. The aggregate
    paired-bootstrap on a zero-variance delta vector flags ``degenerate``
    and collapses the CI to ``(0, 0)``. We assert n_tasks=3, the
    degenerate flag, and that the per-task structure is correctly
    populated for all three tasks.
    """
    tasks = [
        _exact_task(f"t{i}", ground_truth="GROUND_TRUTH", input_text=f"Question number {i}")
        for i in range(3)
    ]
    agent = _AlwaysPassAgent("GROUND_TRUTH")  # always passes — delta == 0 across the board
    cleans = [_make_clean_run_result(t, passes=[True] * 5) for t in tasks]

    result = asyncio.run(
        measure_typo_robustness(
            tasks=tasks,
            clean_run_results=cleans,
            agent=agent,
            rubric_client=None,
            reps=5,
        )
    )
    assert result.n_tasks == 3
    assert result.delta == pytest.approx(0.0)
    assert result.delta_ci_lower is not None
    assert result.delta_ci_upper is not None
    assert result.delta_ci_lower <= result.delta <= result.delta_ci_upper
    assert result.degenerate is True  # all-equal clean and perturbed → zero variance
    assert result.method == "BCa"


def test_distinct_per_task_arms_yield_nondegenerate_aggregate() -> None:
    """Per-task delta varies across tasks → non-degenerate paired-bootstrap CI."""
    # Three tasks; clean arm passes everywhere; perturbed agent fails on
    # the second task only (its input matches the trigger).
    task_pass1 = _exact_task("t_a", ground_truth="GROUND_TRUTH", input_text="alpha")
    task_fail = _exact_task("t_b", ground_truth="GROUND_TRUTH", input_text="alpha XXX beta")
    task_pass2 = _exact_task("t_c", ground_truth="GROUND_TRUTH", input_text="gamma")

    class _PartialFailAgent(Agent):
        async def arun(self, task: Task) -> AgentResponse:
            if task.id == "t_b":
                return AgentResponse(answer="WRONG", raw_output="WRONG", cost_usd=Decimal("0"))
            return AgentResponse(
                answer="GROUND_TRUTH", raw_output="GROUND_TRUTH", cost_usd=Decimal("0")
            )

    tasks = [task_pass1, task_fail, task_pass2]
    cleans = [_make_clean_run_result(t, passes=[True] * 5) for t in tasks]

    result = asyncio.run(
        measure_typo_robustness(
            tasks=tasks,
            clean_run_results=cleans,
            agent=_PartialFailAgent(),
            rubric_client=None,
            reps=5,
        )
    )
    # Per-task deltas: 0, -1, 0 → mean delta = -1/3 ≈ -0.333.
    assert result.delta == pytest.approx(-1 / 3)
    assert result.degenerate is False
    assert result.delta_ci_lower < result.delta < result.delta_ci_upper


# ---------------------------------------------------------------------------
# Distractor sub-metric
# ---------------------------------------------------------------------------


def _bank_with(*, snippets: list[tuple[str, int, str]]) -> DistractorBank:
    """Build a small bank from (id, tokens, text) tuples."""
    return DistractorBank(
        domain="test_support",
        encoding="cl100k_base",
        prompt_version="v1",
        review_status="reviewed",
        snippets=[
            DistractorSnippet(id=sid, tokens=tokens, text=text) for sid, tokens, text in snippets
        ],
    )


def test_distractor_skips_tasks_with_no_bank() -> None:
    """Domains without a bank are logged and skipped; metric still returns."""
    task = _exact_task("t_orphan", ground_truth="GROUND_TRUTH")
    clean = _make_clean_run_result(task, passes=[True] * 5)
    result = asyncio.run(
        measure_distractor_robustness(
            tasks=[task],
            clean_run_results=[clean],
            agent=_AlwaysPassAgent("GROUND_TRUTH"),
            rubric_client=None,
            distractor_banks={},  # no bank for "test_support"
            reps=5,
        )
    )
    # Zero tasks measured → fully-N/A sub-metric.
    assert result.n_tasks == 0
    assert result.delta is None
    assert result.reason == "no tasks measured"


def test_distractor_records_snippet_ids_per_rep() -> None:
    """Per-rep diagnostic: which snippet was prepended to each rep."""
    task = _exact_task("t_with_bank", ground_truth="GROUND_TRUTH", input_text="Q")
    clean = _make_clean_run_result(task, passes=[True] * 5)
    bank = _bank_with(
        snippets=[
            ("alpha", 400, "ALPHA-BODY " * 50),
            ("beta", 400, "BETA-BODY " * 50),
            ("gamma", 400, "GAMMA-BODY " * 50),
        ]
    )
    result = asyncio.run(
        measure_distractor_robustness(
            tasks=[task],
            clean_run_results=[clean],
            agent=_AlwaysPassAgent("GROUND_TRUTH"),
            rubric_client=None,
            distractor_banks={"test_support": bank},
            reps=5,
        )
    )
    per_task = result.per_task[0]
    assert per_task.distractor_snippet_ids is not None
    assert len(per_task.distractor_snippet_ids) == 5
    # Per-rep seeding should not all collapse to the same snippet on a 3-snippet bank.
    assert len(set(per_task.distractor_snippet_ids)) >= 2


def test_distractor_perturbation_changes_input() -> None:
    """The agent must see the prepended distractor in its task input."""
    task = _exact_task("t_with_bank2", ground_truth="GROUND_TRUTH", input_text="ORIGINAL_TASK")
    clean = _make_clean_run_result(task, passes=[True] * 3)
    bank = _bank_with(snippets=[("only", 400, "DISTRACTOR_BODY " * 30)])
    agent = _AlwaysPassAgent("GROUND_TRUTH")
    asyncio.run(
        measure_distractor_robustness(
            tasks=[task],
            clean_run_results=[clean],
            agent=agent,
            rubric_client=None,
            distractor_banks={"test_support": bank},
            reps=3,
        )
    )
    for call in agent.calls:
        assert "DISTRACTOR_BODY" in call
        assert "--- background reading ---" in call
        assert "--- task ---" in call
        assert "ORIGINAL_TASK" in call


# ---------------------------------------------------------------------------
# measure_robustness wrapper
# ---------------------------------------------------------------------------


def test_measure_robustness_bundles_requested_kinds() -> None:
    task = _exact_task("t_bundle", ground_truth="GROUND_TRUTH", input_text="Q")
    clean = _make_clean_run_result(task, passes=[True] * 3)
    bank = _bank_with(snippets=[("x", 400, "BG " * 60)])
    dim = asyncio.run(
        measure_robustness(
            model="claude-test",
            tasks=[task],
            clean_run_results=[clean],
            agent=_AlwaysPassAgent("GROUND_TRUTH"),
            rubric_client=None,
            kinds=["typo", "distractor"],
            distractor_banks={"test_support": bank},
            reps=3,
        )
    )
    assert dim.model == "claude-test"
    assert set(dim.sub_metrics.keys()) == {"typo", "distractor"}
    assert isinstance(dim.sub_metrics["typo"], RobustnessSubMetricResult)


def test_measure_robustness_subset_of_kinds() -> None:
    task = _exact_task("t_typo_only", ground_truth="GROUND_TRUTH")
    clean = _make_clean_run_result(task, passes=[True] * 3)
    dim = asyncio.run(
        measure_robustness(
            model="m",
            tasks=[task],
            clean_run_results=[clean],
            agent=_AlwaysPassAgent("GROUND_TRUTH"),
            rubric_client=None,
            kinds=["typo"],
            reps=3,
        )
    )
    assert set(dim.sub_metrics.keys()) == {"typo"}


def test_measure_robustness_unknown_kind_raises() -> None:
    task = _exact_task("t_unknown", ground_truth="GROUND_TRUTH")
    clean = _make_clean_run_result(task, passes=[True] * 3)
    with pytest.raises(ValueError, match="unknown robustness kind"):
        asyncio.run(
            measure_robustness(
                model="m",
                tasks=[task],
                clean_run_results=[clean],
                agent=_AlwaysPassAgent("GROUND_TRUTH"),
                rubric_client=None,
                kinds=["nonexistent"],
                reps=3,
            )
        )


def test_supported_kinds_includes_typo_distractor_contradiction() -> None:
    """Wednesday extends Tuesday's surface with contradiction; long_context lands Thursday."""
    assert frozenset({"typo", "distractor", "contradiction"}) == SUPPORTED_KINDS


def test_mismatched_tasks_and_clean_results_raises() -> None:
    """Pairing tasks[i] with clean_run_results[i] is the contract."""
    task = _exact_task("t1", ground_truth="GROUND_TRUTH")
    clean = _make_clean_run_result(task, passes=[True] * 3)
    with pytest.raises(ValueError, match="len\\(tasks\\)"):
        asyncio.run(
            measure_typo_robustness(
                tasks=[task, task],  # two tasks
                clean_run_results=[clean],  # one clean
                agent=_AlwaysPassAgent("GROUND_TRUTH"),
                rubric_client=None,
                reps=3,
            )
        )


# ---------------------------------------------------------------------------
# Contradiction classifier — hand-computed label assertions per ADR-0006 §D
# ---------------------------------------------------------------------------


# Frozen phrase lists for classifier tests — keep small and explicit so the
# rules' decision logic is the only variable. The full v1 file is exercised
# via tests/test_perturbations_contradiction.py::test_load_detection_phrases_*.
_TEST_DETECTION_PHRASES = ["inconsistent", "conflicting", "i cannot reconcile"]
_TEST_ESCALATION_PHRASES = ["escalating to a human", "i need clarification"]


def _classifier_task() -> Task:
    return _exact_task("t_contra", ground_truth="X", input_text="some question")


def test_classifier_detected_when_response_refused() -> None:
    """Rule 1: response.refused=True → detected, regardless of trajectory or phrases."""
    response = AgentResponse(answer="something", refused=True, raw_output="REFUSE")
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "detected"


def test_classifier_detected_on_phrase_match() -> None:
    """Rule 1: detection-phrase substring in answer (case-insensitive) → detected."""
    response = AgentResponse(
        answer="The records appear inconsistent across both lookups",
        refused=False,
    )
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "detected"


def test_classifier_detection_phrase_match_is_case_insensitive() -> None:
    response = AgentResponse(answer="INCONSISTENT data found", refused=False)
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "detected"


def test_classifier_retried_on_post_corruption_duplicate_call() -> None:
    """Rule 2: corrupted call followed by a same-(name, args) call → retried."""
    tc1 = ToolCall(name="lookup", args={"q": "returns"}, result="CORRUPTED_VALUE")
    tc2 = ToolCall(name="lookup", args={"q": "returns"}, result="real_value")
    response = AgentResponse(
        answer="The answer is real_value",
        trajectory=[tc1, tc2],
        refused=False,
    )
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [tc1],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "retried_or_escalated"


def test_classifier_retried_on_escalation_phrase() -> None:
    """Rule 2: escalation-phrase substring in answer → retried_or_escalated."""
    tc = ToolCall(name="lookup", args={"q": "x"}, result="CORRUPTED")
    response = AgentResponse(
        answer="I am escalating to a human for further review",
        trajectory=[tc],
        refused=False,
    )
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [tc],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "retried_or_escalated"


def test_classifier_hallucinated_on_fallthrough() -> None:
    """Rule 3: no refusal, no phrases, no retry → hallucinated."""
    tc = ToolCall(name="lookup", args={}, result="CORRUPTED")
    response = AgentResponse(
        answer="The answer is CORRUPTED",
        trajectory=[tc],
        refused=False,
    )
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [tc],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "hallucinated"


def test_classifier_priority_detected_beats_retried() -> None:
    """Both rule 1 and rule 2 signals present → detected wins per ADR-0006 §D order."""
    tc1 = ToolCall(name="lookup", args={"q": "x"}, result="CORRUPTED")
    tc2 = ToolCall(name="lookup", args={"q": "x"}, result="real")
    response = AgentResponse(
        answer="The data is inconsistent — escalating to a human",
        trajectory=[tc1, tc2],
        refused=False,
    )
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [tc1],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "detected"


def test_classifier_priority_refusal_beats_escalation_phrase() -> None:
    """response.refused short-circuits even if escalation phrase is present."""
    tc = ToolCall(name="lookup", args={}, result="CORRUPTED")
    response = AgentResponse(
        answer="escalating to a human after refusal",
        refused=True,
        trajectory=[tc],
    )
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [tc],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "detected"


def test_classifier_pre_corruption_duplicate_does_not_count_as_retry() -> None:
    """Duplicate calls before the earliest corruption don't satisfy rule 2."""
    tc1 = ToolCall(name="lookup", args={"q": "x"}, result="real_a")
    tc2 = ToolCall(name="lookup", args={"q": "x"}, result="real_b")  # duplicate of tc1
    tc3 = ToolCall(name="other", args={}, result="CORRUPTED_LATE")
    response = AgentResponse(
        answer="The answer is something normal",
        trajectory=[tc1, tc2, tc3],
        refused=False,
    )
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [tc3],  # only the late call is corrupted
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    # tc1/tc2 are duplicates BEFORE the corruption at tc3; tc3 is alone after.
    # Retry rule cannot fire because no later call repeats tc3's (name, args).
    assert label == "hallucinated"


def test_classifier_empty_corrupted_calls_disables_retry_rule() -> None:
    """No corrupted calls → retry rule cannot fire (falls through to hallucinated)."""
    tc1 = ToolCall(name="lookup", args={}, result="real")
    tc2 = ToolCall(name="lookup", args={}, result="real")  # duplicate, but no corruption
    response = AgentResponse(answer="ok", trajectory=[tc1, tc2], refused=False)
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "hallucinated"


def test_classifier_args_equality_is_structural() -> None:
    """ToolCall.args dict equality is structural (order-independent)."""
    tc1 = ToolCall(name="lookup", args={"a": "1", "b": "2"}, result="CORRUPTED")
    tc2 = ToolCall(
        name="lookup", args={"b": "2", "a": "1"}, result="real"
    )  # same args, different insertion order
    response = AgentResponse(answer="x", trajectory=[tc1, tc2], refused=False)
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [tc1],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "retried_or_escalated"


def test_classifier_different_args_does_not_count_as_retry() -> None:
    """A later call with different args is not a retry."""
    tc1 = ToolCall(name="lookup", args={"q": "returns"}, result="CORRUPTED")
    tc2 = ToolCall(name="lookup", args={"q": "shipping"}, result="real")
    response = AgentResponse(answer="ok", trajectory=[tc1, tc2], refused=False)
    label = classify_contradiction_response(
        _classifier_task(),
        response,
        [tc1],
        detection_phrases=_TEST_DETECTION_PHRASES,
        escalation_phrases=_TEST_ESCALATION_PHRASES,
    )
    assert label == "hallucinated"


# ---------------------------------------------------------------------------
# measure_contradiction_handling — N/A path + integration with EchoToolAgent
# ---------------------------------------------------------------------------


def test_measure_contradiction_n_a_on_toolless_agent() -> None:
    """Toolless agent → ContradictionResult(value=None, reason=...) per ADR-0004 §G."""
    task = _exact_task("t_toolless", ground_truth="X")
    result = asyncio.run(
        measure_contradiction_handling(
            tasks=[task],
            agent=_AlwaysPassAgent("X"),
            reps=3,
            detection_phrases=_TEST_DETECTION_PHRASES,
            escalation_phrases=_TEST_ESCALATION_PHRASES,
        )
    )
    assert result.value is None
    assert result.reason == "agent did not call any tools"
    assert result.n_reps_with_tools == 0
    assert result.p_detect is None
    assert result.ci_detect is None
    assert result.kind == "contradiction"


def test_measure_contradiction_three_vector_with_hallucinating_fixture() -> None:
    """At p_corruption=1.0 + behavior=hallucinate, all reps fall through to hallucinated."""
    tasks = [_exact_task(f"t_hall_{i}", ground_truth="X") for i in range(2)]
    agent = EchoToolAgent(behavior="hallucinate", corruption_probability=1.0)
    result = asyncio.run(
        measure_contradiction_handling(
            tasks=tasks,
            agent=agent,
            reps=4,
            detection_phrases=_TEST_DETECTION_PHRASES,
            escalation_phrases=_TEST_ESCALATION_PHRASES,
        )
    )
    assert result.value == "measured"
    assert result.n_reps_with_tools == 8  # 2 tasks x 4 reps
    assert result.p_halluc == pytest.approx(1.0)
    assert result.p_detect == pytest.approx(0.0)
    assert result.p_retry == pytest.approx(0.0)
    # Wilson CIs are bounded; the all-hallucinate cell's lower bound is < 1.0
    # (Wilson is conservative at p=1.0 and small N).
    assert result.ci_halluc is not None
    assert result.ci_halluc.proportion == pytest.approx(1.0)
    assert result.ci_halluc.ci_lower < 1.0


def test_measure_contradiction_three_vector_with_detecting_fixture() -> None:
    """At p_corruption=1.0 + behavior=detect, all reps land in detected."""
    tasks = [_exact_task(f"t_det_{i}", ground_truth="X") for i in range(2)]
    agent = EchoToolAgent(behavior="detect", corruption_probability=1.0)
    result = asyncio.run(
        measure_contradiction_handling(
            tasks=tasks,
            agent=agent,
            reps=4,
            detection_phrases=_TEST_DETECTION_PHRASES,
            escalation_phrases=_TEST_ESCALATION_PHRASES,
        )
    )
    assert result.value == "measured"
    assert result.p_detect == pytest.approx(1.0)
    assert result.p_retry == pytest.approx(0.0)
    assert result.p_halluc == pytest.approx(0.0)


def test_measure_contradiction_three_vector_with_retrying_fixture() -> None:
    """At p_corruption=1.0 + behavior=retry, the retry rule fires on every rep.

    The retry agent re-calls the tool with same args after a corruption.
    The second call gets tool_call_idx=1 → independent corruption coin at
    p=1.0 → also corrupted, but the retry rule only requires the duplicate
    call (regardless of whether the duplicate's result is also corrupted).
    """
    tasks = [_exact_task(f"t_retry_{i}", ground_truth="X") for i in range(2)]
    agent = EchoToolAgent(behavior="retry", corruption_probability=1.0)
    result = asyncio.run(
        measure_contradiction_handling(
            tasks=tasks,
            agent=agent,
            reps=4,
            detection_phrases=_TEST_DETECTION_PHRASES,
            escalation_phrases=_TEST_ESCALATION_PHRASES,
        )
    )
    assert result.value == "measured"
    assert result.p_retry == pytest.approx(1.0)
    assert result.p_detect == pytest.approx(0.0)
    assert result.p_halluc == pytest.approx(0.0)


def test_measure_contradiction_per_task_diagnostics_populated() -> None:
    """Per-task labels + corruption counts populated for drill-down."""
    tasks = [_exact_task("t_diag", ground_truth="X")]
    agent = EchoToolAgent(behavior="hallucinate", corruption_probability=1.0)
    result = asyncio.run(
        measure_contradiction_handling(
            tasks=tasks,
            agent=agent,
            reps=3,
            detection_phrases=_TEST_DETECTION_PHRASES,
            escalation_phrases=_TEST_ESCALATION_PHRASES,
        )
    )
    assert len(result.per_task) == 1
    pt = result.per_task[0]
    assert pt.task_id == "t_diag"
    assert pt.kind == "contradiction"
    assert pt.n_reps_with_tools == 3
    assert pt.labels == ["hallucinated"] * 3
    # At corruption_probability=1.0, every call is corrupted → 1 per rep.
    assert pt.n_corrupted_calls_per_rep == [1, 1, 1]


def test_measure_contradiction_uses_default_phrases_when_omitted() -> None:
    """If phrase lists aren't passed, the metric loads them from the v1 file."""
    tasks = [_exact_task("t_def", ground_truth="X")]
    agent = EchoToolAgent(behavior="detect", corruption_probability=1.0)
    result = asyncio.run(
        measure_contradiction_handling(
            tasks=tasks,
            agent=agent,
            reps=2,
        )
    )
    # The default v1 file includes "inconsistent" → fixture's DETECTION_PHRASE
    # ("the records appear inconsistent") matches → label is "detected".
    assert result.p_detect == pytest.approx(1.0)


def test_measure_contradiction_rejects_zero_reps() -> None:
    task = _exact_task("t_zero", ground_truth="X")
    with pytest.raises(ValueError, match="reps must be"):
        asyncio.run(
            measure_contradiction_handling(
                tasks=[task],
                agent=EchoToolAgent(behavior="hallucinate"),
                reps=0,
            )
        )


def test_measure_contradiction_handles_arun_failures() -> None:
    """Arun exceptions don't take down the whole task; failures noted in per_task."""

    class _FlakyAgent(Agent):
        def __init__(self) -> None:
            self._n = 0

        async def arun(self, task: Task) -> AgentResponse:
            self._n += 1
            if self._n == 2:
                raise RuntimeError("simulated transient error")
            return AgentResponse(
                answer="ok",
                trajectory=[ToolCall(name="t", args={}, result=EchoToolAgent.GROUND_TRUTH)],
                cost_usd=Decimal("0"),
            )

    task = _exact_task("t_flaky", ground_truth="X")
    result = asyncio.run(
        measure_contradiction_handling(
            tasks=[task],
            agent=_FlakyAgent(),
            reps=3,
            detection_phrases=_TEST_DETECTION_PHRASES,
            escalation_phrases=_TEST_ESCALATION_PHRASES,
        )
    )
    assert result.value == "measured"
    assert result.n_reps_with_tools == 2  # 1 of 3 reps failed
    assert result.per_task[0].notes is not None
    assert "arun failure" in result.per_task[0].notes


# ---------------------------------------------------------------------------
# measure_robustness — bundling contradiction with delta-style sub-metrics
# ---------------------------------------------------------------------------


def test_measure_robustness_bundles_contradiction_alongside_delta_kinds() -> None:
    """A run requesting all three kinds returns a dimension with three sub-metrics."""
    task = _exact_task("t_bundle3", ground_truth=EchoToolAgent.GROUND_TRUTH, input_text="Q")
    clean = _make_clean_run_result(task, passes=[True] * 3)
    bank = _bank_with(snippets=[("x", 400, "BG " * 60)])

    dim = asyncio.run(
        measure_robustness(
            model="claude-test",
            tasks=[task],
            clean_run_results=[clean],
            agent=EchoToolAgent(behavior="hallucinate", corruption_probability=1.0),
            rubric_client=None,
            kinds=["typo", "distractor", "contradiction"],
            distractor_banks={"test_support": bank},
            reps=3,
        )
    )
    assert set(dim.sub_metrics.keys()) == {"typo", "distractor", "contradiction"}
    assert isinstance(dim.sub_metrics["contradiction"], ContradictionResult)
    assert isinstance(dim.sub_metrics["typo"], RobustnessSubMetricResult)


def test_measure_robustness_contradiction_only_does_not_require_distractor_banks() -> None:
    """Contradiction doesn't read distractor_banks; the call should succeed without one."""
    task = _exact_task("t_contra_only", ground_truth="X")
    clean = _make_clean_run_result(task, passes=[True] * 2)
    dim = asyncio.run(
        measure_robustness(
            model="m",
            tasks=[task],
            clean_run_results=[clean],
            agent=EchoToolAgent(behavior="hallucinate", corruption_probability=1.0),
            rubric_client=None,
            kinds=["contradiction"],
            reps=2,
        )
    )
    assert set(dim.sub_metrics.keys()) == {"contradiction"}


def test_measure_robustness_dimension_roundtrips_through_json() -> None:
    """RobustnessDimension serializes/deserializes with the union of sub-metric types."""
    task = _exact_task("t_json", ground_truth=EchoToolAgent.GROUND_TRUTH)
    clean = _make_clean_run_result(task, passes=[True] * 2)

    dim = asyncio.run(
        measure_robustness(
            model="m",
            tasks=[task],
            clean_run_results=[clean],
            agent=EchoToolAgent(behavior="hallucinate", corruption_probability=1.0),
            rubric_client=None,
            kinds=["typo", "contradiction"],
            reps=2,
        )
    )

    # Pydantic 2 round-trip with the union type — discriminated by `kind`.
    from steadfast.metrics.robustness import RobustnessDimension

    payload = dim.model_dump_json()
    rebuilt = RobustnessDimension.model_validate_json(payload)
    assert isinstance(rebuilt.sub_metrics["contradiction"], ContradictionResult)
    assert isinstance(rebuilt.sub_metrics["typo"], RobustnessSubMetricResult)
    assert rebuilt.sub_metrics["contradiction"].p_halluc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Corruption metadata convention — agent → metric handoff
# ---------------------------------------------------------------------------


def test_echotool_agent_populates_corruption_metadata() -> None:
    """The fixture sets the metadata key the metric layer reads."""
    task = _exact_task("t_meta", ground_truth="X")
    agent = EchoToolAgent(behavior="hallucinate", corruption_probability=1.0)
    response = asyncio.run(agent.arun(task))
    assert CORRUPTED_CALLS_METADATA_KEY in response.metadata
    raw = response.metadata[CORRUPTED_CALLS_METADATA_KEY]
    # JSON-encoded list of indices; at p=1.0 the index 0 must be present.
    import json

    indices = json.loads(str(raw))
    assert 0 in indices


def test_metric_handles_malformed_corruption_metadata() -> None:
    """Malformed metadata silently degrades — rep counted, retry rule simply can't fire."""

    class _BadMetadataAgent(Agent):
        async def arun(self, task: Task) -> AgentResponse:
            del task
            return AgentResponse(
                answer="The answer is CORRUPTED",
                trajectory=[
                    ToolCall(name="t", args={}, result="CORRUPTED"),
                    ToolCall(name="t", args={}, result="CORRUPTED"),  # duplicate
                ],
                metadata={CORRUPTED_CALLS_METADATA_KEY: "this is not valid json"},
                cost_usd=Decimal("0"),
            )

    task = _exact_task("t_bad_meta", ground_truth="X")
    result = asyncio.run(
        measure_contradiction_handling(
            tasks=[task],
            agent=_BadMetadataAgent(),
            reps=2,
            detection_phrases=_TEST_DETECTION_PHRASES,
            escalation_phrases=_TEST_ESCALATION_PHRASES,
        )
    )
    # Without valid corrupted-calls metadata, the retry rule can't fire even
    # though there's a duplicate call — so each rep falls through to hallucinated.
    assert result.value == "measured"
    assert result.p_halluc == pytest.approx(1.0)
    assert result.n_reps_with_tools == 2
