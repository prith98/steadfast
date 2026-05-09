"""Tests for steadfast.metrics.calibration — Brier / ECE / refusal / overconfidence.

End-to-end tests use hand-constructed CalibrationRep lists so the metric
behavior is hand-verifiable. Bootstrap CIs are sanity-checked via
inclusion (point estimate within bounds) since the bounds themselves are
random-seed-dependent.
"""

from __future__ import annotations

import math

import pytest

from steadfast.agent import AgentResponse, Task
from steadfast.judges.base import Verdict
from steadfast.metrics.calibration import (
    CalibrationRep,
    measure_brier,
    measure_calibration,
    measure_ece,
    measure_overconfidence,
    measure_refusal_calibration,
    reps_from_run_results,
)
from steadfast.runner import RepRecord, RepStatus


def _verdict(passed: bool) -> Verdict:
    return Verdict(score=1.0 if passed else 0.0, passed=passed, reason="test")


def _rep(
    *,
    task_id: str = "t1",
    confidence: float | None = 0.8,
    refused: bool = False,
    passed: bool = True,
    logprob_avg: float | None = None,
    difficulty: str = "normal",
) -> CalibrationRep:
    return CalibrationRep(
        task=Task(
            id=task_id,
            domain="d",
            input="x",
            difficulty=difficulty,  # type: ignore[arg-type]
        ),
        response=AgentResponse(
            answer="ans",
            confidence=confidence,
            refused=refused,
            logprob_avg=logprob_avg,
        ),
        verdict=_verdict(passed),
    )


# ---------------------------------------------------------------------------
# Brier
# ---------------------------------------------------------------------------


def test_brier_perfect_calibration_returns_zero() -> None:
    reps = [_rep(confidence=1.0, passed=True) for _ in range(10)] + [
        _rep(confidence=0.0, passed=False) for _ in range(10)
    ]
    result = measure_brier(reps, seed=0)
    assert result.verbalized is not None
    assert result.verbalized.point_estimate == pytest.approx(0.0)
    assert result.n == 20
    assert result.n_refused == 0
    assert result.n_no_confidence == 0


def test_brier_uniform_half_returns_quarter() -> None:
    """Forecasting 0.5 always → Brier = 0.25."""
    reps = [_rep(confidence=0.5, passed=(i % 2 == 0)) for i in range(20)]
    result = measure_brier(reps, seed=0)
    assert result.verbalized is not None
    assert result.verbalized.point_estimate == pytest.approx(0.25)


def test_brier_excludes_refused_and_none_confidence() -> None:
    reps = [_rep(confidence=0.8, passed=True) for _ in range(5)]
    reps.append(_rep(refused=True, confidence=0.0))
    reps.append(_rep(confidence=None))
    result = measure_brier(reps, seed=0)
    assert result.n == 5
    assert result.n_refused == 1
    assert result.n_no_confidence == 1
    assert result.n_total == 7


def test_brier_logprob_column_populated_when_available() -> None:
    """Mixed pool: half the reps carry a logprob, half don't.

    The verbalized column uses every answered rep; the logprob column
    uses only the subset with logprob_avg.
    """
    # 10 reps with logprob_avg = log(0.7) (so exp = 0.7), perfect outcomes
    reps_with_lp = [_rep(confidence=0.7, passed=True, logprob_avg=math.log(0.7)) for _ in range(10)]
    reps_without_lp = [_rep(confidence=0.7, passed=True) for _ in range(10)]
    result = measure_brier([*reps_with_lp, *reps_without_lp], seed=0)
    assert result.n == 20
    assert result.n_logprob == 10
    assert result.verbalized is not None
    assert result.logprob is not None
    # Verbalized: all 0.7 forecasts, all correct → (0.7 - 1)^2 = 0.09
    assert result.verbalized.point_estimate == pytest.approx(0.09)
    # Logprob implied = exp(log(0.7)) = 0.7 too → same number
    assert result.logprob.point_estimate == pytest.approx(0.09)


def test_brier_logprob_none_when_no_provider_supports_it() -> None:
    """Anthropic-only run: no logprob_avg anywhere → logprob column is None."""
    reps = [_rep(confidence=0.7, passed=True) for _ in range(10)]
    result = measure_brier(reps, seed=0)
    assert result.n_logprob == 0
    assert result.logprob is None


def test_brier_too_few_returns_none_with_reason() -> None:
    result = measure_brier([_rep(confidence=0.5, passed=True)])
    assert result.verbalized is None
    assert result.reason is not None
    assert "at least 2" in result.reason


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def test_ece_default_15_bins_when_pool_large_enough() -> None:
    """N=30 (≥15) → uses the methodology default 15 bins, no fallback."""
    reps = [_rep(confidence=0.5 + 0.01 * i, passed=(i % 2 == 0)) for i in range(30)]
    result = measure_ece(reps)
    assert result.n == 30
    assert result.n_bins == 15
    assert result.fallback_used is False
    assert result.verbalized is not None
    assert 0.0 <= result.verbalized <= 1.0


def test_ece_small_n_falls_back_to_floor_n_div_3() -> None:
    """N=12 (<15) → fallback to floor(12/3) = 4 bins."""
    reps = [_rep(confidence=0.5 + 0.01 * i, passed=(i % 2 == 0)) for i in range(12)]
    result = measure_ece(reps)
    assert result.n_bins == 4
    assert result.fallback_used is True
    assert result.verbalized is not None


def test_ece_too_small_returns_none() -> None:
    """N<9 (so floor(N/3) < 3 bins) → returns None with a reason."""
    reps = [_rep(confidence=0.5, passed=True) for _ in range(5)]
    result = measure_ece(reps)
    assert result.verbalized is None
    assert result.reason is not None
    assert "too small" in result.reason


def test_ece_perfect_calibration_is_zero_in_two_groups() -> None:
    """30 reps at conf=0.3 with 30% positive + 30 at conf=0.7 with 70% positive.

    With 2 bins (forced via n_bins=2), each bin has gap=0 → ECE=0.
    """
    reps = (
        [_rep(confidence=0.3, passed=False) for _ in range(21)]
        + [_rep(confidence=0.3, passed=True) for _ in range(9)]
        + [_rep(confidence=0.7, passed=False) for _ in range(9)]
        + [_rep(confidence=0.7, passed=True) for _ in range(21)]
    )
    result = measure_ece(reps, n_bins=2)
    assert result.verbalized == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Refusal calibration
# ---------------------------------------------------------------------------


def test_refusal_calibration_perfect() -> None:
    """Agent refuses every hard task and answers every normal task → sens=1, spec=1."""
    reps = [_rep(refused=True, difficulty="hard", confidence=0.0) for _ in range(10)] + [
        _rep(refused=False, difficulty="normal", confidence=0.8) for _ in range(20)
    ]
    result = measure_refusal_calibration(reps)
    assert result.refused_hard == 10
    assert result.answered_normal == 20
    assert result.refused_normal == 0
    assert result.answered_hard == 0
    assert result.sensitivity == pytest.approx(1.0)
    assert result.specificity == pytest.approx(1.0)
    assert result.sensitivity_ci is not None
    assert result.specificity_ci is not None


def test_refusal_calibration_worst_case() -> None:
    """Agent answers every hard task and refuses every normal task → sens=0, spec=0."""
    reps = [_rep(refused=False, difficulty="hard", confidence=0.95) for _ in range(10)] + [
        _rep(refused=True, difficulty="normal", confidence=0.0) for _ in range(20)
    ]
    result = measure_refusal_calibration(reps)
    assert result.sensitivity == pytest.approx(0.0)
    assert result.specificity == pytest.approx(0.0)


def test_refusal_calibration_no_hard_tasks_returns_none_sensitivity() -> None:
    reps = [_rep(refused=False, difficulty="normal", confidence=0.8) for _ in range(10)]
    result = measure_refusal_calibration(reps)
    assert result.sensitivity is None
    assert result.sensitivity_ci is None
    assert result.n_hard == 0
    assert result.specificity == pytest.approx(1.0)


def test_refusal_calibration_no_normal_tasks_returns_none_specificity() -> None:
    reps = [_rep(refused=True, difficulty="hard", confidence=0.0) for _ in range(5)]
    result = measure_refusal_calibration(reps)
    assert result.specificity is None
    assert result.specificity_ci is None
    assert result.n_normal == 0


# ---------------------------------------------------------------------------
# Overconfidence rate
# ---------------------------------------------------------------------------


def test_overconfidence_perfect_calibration_rate_zero() -> None:
    """All correct answers regardless of confidence → no incorrect, rate=0."""
    reps = [_rep(confidence=0.95, passed=True) for _ in range(10)]
    result = measure_overconfidence(reps)
    assert result.n_overconfident == 0
    assert result.rate == 0.0
    assert result.ci is not None


def test_overconfidence_high_when_wrong_at_high_confidence() -> None:
    """All forecasts at 0.95, all wrong → rate=1.0."""
    reps = [_rep(confidence=0.95, passed=False) for _ in range(10)]
    result = measure_overconfidence(reps)
    assert result.n_overconfident == 10
    assert result.rate == pytest.approx(1.0)


def test_overconfidence_excludes_low_confidence_wrong_answers() -> None:
    """Wrong at confidence=0.5 doesn't count as overconfident; wrong at 0.95 does."""
    reps = (
        [_rep(confidence=0.95, passed=False) for _ in range(3)]  # overconfident
        + [_rep(confidence=0.5, passed=False) for _ in range(5)]  # wrong but not overconfident
        + [_rep(confidence=0.95, passed=True) for _ in range(2)]  # confident and right
    )
    result = measure_overconfidence(reps)
    assert result.n_answered == 10
    assert result.n_overconfident == 3
    assert result.rate == pytest.approx(0.3)


def test_overconfidence_excludes_refused_reps() -> None:
    reps = [_rep(refused=True, confidence=0.0) for _ in range(5)]
    result = measure_overconfidence(reps)
    assert result.n_answered == 0
    assert result.rate is None
    assert result.reason is not None


def test_overconfidence_threshold_is_inclusive() -> None:
    """Per METHODOLOGY §3.5: confidence >= 0.9 (not strictly greater)."""
    reps = [_rep(confidence=0.9, passed=False) for _ in range(5)]
    result = measure_overconfidence(reps)
    assert result.n_overconfident == 5


def test_overconfidence_invalid_threshold() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        measure_overconfidence([_rep()], threshold=1.5)


# ---------------------------------------------------------------------------
# reps_from_run_results / measure_calibration end-to-end
# ---------------------------------------------------------------------------


def test_reps_from_run_results_filters_by_status_and_verdict() -> None:
    task = Task(id="t1", domain="d", input="x")
    completed_with_verdict = RepRecord(
        run_id="r",
        task_id="t1",
        rep_idx=0,
        status=RepStatus.COMPLETED,
        response=AgentResponse(answer="ans", confidence=0.8),
        verdict=_verdict(True),
    )
    completed_no_verdict = RepRecord(
        run_id="r",
        task_id="t1",
        rep_idx=1,
        status=RepStatus.COMPLETED,
        response=AgentResponse(answer="ans"),
        verdict=None,  # judge failed
    )
    failed = RepRecord(run_id="r", task_id="t1", rep_idx=2, status=RepStatus.FAILED, error="boom")
    out = reps_from_run_results([completed_with_verdict, completed_no_verdict, failed], task)
    assert len(out) == 1
    assert out[0].task is task
    assert out[0].verdict.passed is True


def test_measure_calibration_bundles_all_four() -> None:
    """End-to-end: hand-built reps produce a CalibrationDimension with sensible numbers."""
    reps = (
        [_rep(confidence=0.95, passed=True, difficulty="normal") for _ in range(20)]
        + [_rep(refused=True, confidence=0.0, difficulty="hard") for _ in range(5)]
        + [_rep(confidence=0.95, passed=False, difficulty="normal") for _ in range(5)]
    )
    result = measure_calibration(reps, model="claude-test", n_tasks=3, seed=0)
    assert result.model == "claude-test"
    assert result.n_reps_total == 30
    # Brier: 25 answered reps; 20 correct at 0.95 → (0.95-1)^2 = 0.0025;
    # 5 wrong at 0.95 → (0.95)^2 = 0.9025. Mean = (20*0.0025 + 5*0.9025)/25 = 0.1825
    assert result.brier.verbalized is not None
    assert result.brier.verbalized.point_estimate == pytest.approx(0.1825, abs=1e-3)
    # Refusal: 5 hard refused + 0 hard answered + 0 normal refused + 25 normal answered.
    # Sensitivity = 5/5 = 1.0, specificity = 25/25 = 1.0.
    assert result.refusal.sensitivity == pytest.approx(1.0)
    assert result.refusal.specificity == pytest.approx(1.0)
    # Overconfidence: 5 incorrect at 0.95 / 25 answered = 0.2.
    assert result.overconfidence.rate == pytest.approx(0.2)
