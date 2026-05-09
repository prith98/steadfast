"""Calibration dimension — Brier, ECE, refusal calibration, overconfidence rate.

Per ``docs/METHODOLOGY.md`` §3 and ADR-0005 §D-E, four measurement
functions:

* :func:`measure_brier` — pooled (task, rep) squared errors with bootstrap
  CI; verbalized + (where available) logprob-derived parallel columns.
* :func:`measure_ece` — 15 equal-mass bins (Nixon et al. 2019) with the
  small-N fallback documented in METHODOLOGY §3.3.
* :func:`measure_refusal_calibration` — {refused | answered} x {hard |
  normal} confusion matrix with Wilson CIs per cell.
* :func:`measure_overconfidence` — Wilson CI on
  ``count(incorrect ∧ confidence ≥ 0.9) / count(answered)``.

Each function returns a frozen Pydantic result; the headline calibration
column is **verbalized** (universal across providers) and the secondary
**logprob** column carries explicit ``None`` for providers that don't
expose per-token logprobs (Anthropic, Google in v0.1 — see ADR-0005 §A).

The metric layer accepts a flat ``CalibrationInputs`` view of all reps
across all tasks for a single (model, run) configuration; the CLI is
responsible for assembling that view from per-task ``RunResult`` JSONs.
This separation keeps the metric pure (no JSON parsing, no I/O) and
testable on hand-constructed inputs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from steadfast.agent import AgentResponse, Task
from steadfast.judges.base import Verdict
from steadfast.runner import RepRecord, RepStatus
from steadfast.stats.bootstrap import BootstrapCI, bootstrap_ci
from steadfast.stats.calibration import (
    DEFAULT_ECE_BINS,
    ECE_FALLBACK_MIN_BINS,
    BinStats,
    brier_squared_errors,
    expected_calibration_error,
)
from steadfast.stats.wilson import WilsonCI, wilson_ci

# Per METHODOLOGY §3.5 — the threshold above which an incorrect answer
# counts as "overconfident".
OVERCONFIDENCE_THRESHOLD: Final[float] = 0.9

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input view: a flat per-rep tuple of (task, response, verdict).
# ---------------------------------------------------------------------------


class CalibrationRep(BaseModel):
    """One (task, rep) row used by every calibration measurement.

    Carries everything a calibration function needs: the rep's task
    (for ``difficulty``), the agent's response (for ``confidence``,
    ``refused``, ``logprob_avg``), and the rep's outcome verdict (for the
    binary correct/incorrect signal). The CLI builds this list once from
    the persisted ``RunResult`` JSONs and passes it to each measurement
    function.

    The ``RepRecord`` inputs that contribute to this view are filtered to
    ``status == COMPLETED`` and a non-``None`` verdict — failed reps and
    un-judged reps don't contribute to calibration. A separate counter on
    each result Pydantic model surfaces how many reps were excluded so the
    HTML report can flag low-yield runs.
    """

    model_config = ConfigDict(frozen=True)

    task: Task
    response: AgentResponse
    verdict: Verdict


class _Pool(BaseModel):
    """Flat numpy-ready view of a CalibrationRep list, post-filtering."""

    model_config = ConfigDict(frozen=True)

    confidences: list[float]
    logprob_implied: list[float | None]  # exp(avg_logprob) or None per rep
    outcomes: list[int]  # 1 if rep passed, 0 otherwise
    n_total: int  # total CalibrationReps before filtering
    n_refused: int
    n_no_confidence: int  # reps with confidence=None (parser soft-failed)


# ---------------------------------------------------------------------------
# Result Pydantic models
# ---------------------------------------------------------------------------


class BrierResult(BaseModel):
    """Pooled-bootstrap Brier score with a parallel logprob-derived column.

    The headline ``verbalized`` value is computed from
    :attr:`AgentResponse.confidence`; the ``logprob`` value is computed
    from ``exp(AgentResponse.logprob_avg)`` over the subset of reps whose
    provider exposed logprobs. ``logprob`` is ``None`` when no rep in the
    pool carried a logprob (e.g., Anthropic-only models in v0.1).
    """

    model_config = ConfigDict(frozen=True)

    n: int  # answered reps with non-None confidence
    n_logprob: int  # subset of n that also carried logprob
    verbalized: BootstrapCI | None
    logprob: BootstrapCI | None
    n_total: int
    n_refused: int
    n_no_confidence: int
    reason: str | None = None


class ECEResult(BaseModel):
    """ECE with equal-mass bins; bin_stats let the report render a reliability diagram."""

    model_config = ConfigDict(frozen=True)

    n: int
    n_bins: int
    fallback_used: bool  # True iff small-N fallback to floor(N/3) bins fired
    verbalized: float | None
    verbalized_bins: list[BinStats]
    logprob: float | None
    logprob_bins: list[BinStats]
    n_logprob: int
    reason: str | None = None


class RefusalCalibrationResult(BaseModel):
    """{refused | answered} x {hard | normal} confusion matrix with Wilson cell CIs.

    Headline scalars are sensitivity (TR / (TR + FA)) and specificity
    (TA / (TA + FR)). When a row's denominator is zero, the corresponding
    Wilson CI is None and the scalar is None (see METHODOLOGY §3.4).
    """

    model_config = ConfigDict(frozen=True)

    refused_hard: int  # TR
    refused_normal: int  # FR
    answered_hard: int  # FA
    answered_normal: int  # TA
    n_hard: int = Field(ge=0)
    n_normal: int = Field(ge=0)
    sensitivity: float | None
    sensitivity_ci: WilsonCI | None
    specificity: float | None
    specificity_ci: WilsonCI | None


class OverconfidenceResult(BaseModel):
    """Wilson 95% CI on count(incorrect ∧ confidence ≥ threshold) / count(answered)."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    n_answered: int
    n_overconfident: int
    rate: float | None
    ci: WilsonCI | None
    reason: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pool(reps: Sequence[CalibrationRep]) -> _Pool:
    """Filter to answered reps with parsed confidence; carry logprob alongside.

    Reps where ``response.refused == True`` or ``response.confidence is
    None`` are excluded from Brier / ECE / overconfidence pools per
    ADR-0005 §C-D. Counters preserve the bookkeeping so the HTML report
    can show "Brier computed over X of Y reps; Z refused, W parser-failed".
    """
    confidences: list[float] = []
    logprob_implied: list[float | None] = []
    outcomes: list[int] = []
    n_refused = 0
    n_no_confidence = 0
    for rep in reps:
        if rep.response.refused:
            n_refused += 1
            continue
        if rep.response.confidence is None:
            n_no_confidence += 1
            continue
        confidences.append(float(rep.response.confidence))
        outcome = 1 if rep.verdict.passed else 0
        outcomes.append(outcome)
        lp_avg = rep.response.logprob_avg
        # exp(avg_logprob) is the geometric mean of per-token probabilities
        # — a calibration heuristic per Kadavath et al. 2022. We take the
        # *implied probability* per ADR-0005 §A; the metric layer then
        # reports both verbalized and logprob columns side-by-side.
        logprob_implied.append(math.exp(lp_avg) if lp_avg is not None else None)
    return _Pool(
        confidences=confidences,
        logprob_implied=logprob_implied,
        outcomes=outcomes,
        n_total=len(reps),
        n_refused=n_refused,
        n_no_confidence=n_no_confidence,
    )


def reps_from_run_results(reps: Sequence[RepRecord], task: Task) -> list[CalibrationRep]:
    """Project ``RunResult.reps`` + the originating ``task`` into ``CalibrationRep``s.

    Filters to completed reps with a populated verdict (per the
    "calibration consumes scored reps" rule in ADR-0005). The CLI calls
    this once per task, then concatenates across tasks before invoking
    the calibration metrics so the pool spans the full benchmark.
    """
    out: list[CalibrationRep] = []
    for rep in reps:
        if rep.status != RepStatus.COMPLETED:
            continue
        if rep.response is None or rep.verdict is None:
            continue
        out.append(CalibrationRep(task=task, response=rep.response, verdict=rep.verdict))
    return out


# ---------------------------------------------------------------------------
# Brier
# ---------------------------------------------------------------------------


def measure_brier(
    reps: Sequence[CalibrationRep],
    *,
    seed: int = 0,
) -> BrierResult:
    """Per METHODOLOGY §3.2 / ADR-0005 §D: pooled bootstrap Brier with parallel logprob column.

    Bootstrap is BCa over the pooled per-rep squared-error array (not a
    cluster bootstrap; see ADR-0005 §D for the rationale and the v0.2
    upgrade path).
    """
    pool = _build_pool(reps)
    n = len(pool.confidences)
    if n < 2:
        return BrierResult(
            n=n,
            n_logprob=0,
            verbalized=None,
            logprob=None,
            n_total=pool.n_total,
            n_refused=pool.n_refused,
            n_no_confidence=pool.n_no_confidence,
            reason="brier requires at least 2 (forecast, outcome) pairs",
        )

    verbalized_errs = brier_squared_errors(pool.confidences, pool.outcomes)
    verbalized_ci = bootstrap_ci(verbalized_errs.tolist(), seed=seed)

    # Logprob column: subset to reps whose provider exposed logprobs. The
    # CLI may sequence runs across providers in any order so we filter
    # each rep individually rather than gating the whole metric on a
    # provider check.
    logprob_pairs = [
        (lp, y) for lp, y in zip(pool.logprob_implied, pool.outcomes, strict=True) if lp is not None
    ]
    logprob_ci: BootstrapCI | None
    if len(logprob_pairs) < 2:
        logprob_ci = None
    else:
        logprob_confidences = [lp for lp, _ in logprob_pairs]
        logprob_outcomes = [y for _, y in logprob_pairs]
        logprob_errs = brier_squared_errors(logprob_confidences, logprob_outcomes)
        logprob_ci = bootstrap_ci(logprob_errs.tolist(), seed=seed)

    return BrierResult(
        n=n,
        n_logprob=len(logprob_pairs),
        verbalized=verbalized_ci,
        logprob=logprob_ci,
        n_total=pool.n_total,
        n_refused=pool.n_refused,
        n_no_confidence=pool.n_no_confidence,
        reason=None,
    )


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def _ece_with_fallback(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_bins_default: int = DEFAULT_ECE_BINS,
) -> tuple[float | None, list[BinStats], int, bool, str | None]:
    """ECE with the documented small-N fallback.

    Returns ``(ece, bin_stats, n_bins, fallback_used, reason)``. When the
    sample is too small for even the fallback (< ``ECE_FALLBACK_MIN_BINS``
    bins), ``ece`` is ``None`` and ``reason`` carries the explanation.
    """
    n = len(confidences)
    if n == 0:
        return None, [], 0, False, "ECE requires at least one (forecast, outcome) pair"

    if n >= n_bins_default:
        ece, bins = expected_calibration_error(confidences, outcomes, n_bins=n_bins_default)
        return ece, bins, n_bins_default, False, None

    # Small-N fallback per METHODOLOGY §3.3 / ADR-0005 §D: floor(N / 3)
    # bins; below 3 bins return None.
    fallback_bins = n // 3
    if fallback_bins < ECE_FALLBACK_MIN_BINS:
        min_n = ECE_FALLBACK_MIN_BINS * 3
        return (
            None,
            [],
            0,
            True,
            f"ECE pool of {n} forecasts is too small even for the floor(N/3) fallback "
            f"(need at least {min_n} forecasts for {ECE_FALLBACK_MIN_BINS} bins)",
        )
    ece, bins = expected_calibration_error(confidences, outcomes, n_bins=fallback_bins)
    return ece, bins, fallback_bins, True, None


def measure_ece(
    reps: Sequence[CalibrationRep],
    *,
    n_bins: int = DEFAULT_ECE_BINS,
) -> ECEResult:
    """Per METHODOLOGY §3.3 / ADR-0005 §D: equal-mass ECE with parallel logprob column."""
    pool = _build_pool(reps)
    verbalized_ece, verbalized_bins, used_bins, fallback, reason = _ece_with_fallback(
        pool.confidences, pool.outcomes, n_bins_default=n_bins
    )

    logprob_pairs = [
        (lp, y) for lp, y in zip(pool.logprob_implied, pool.outcomes, strict=True) if lp is not None
    ]
    if logprob_pairs:
        logprob_confidences = [lp for lp, _ in logprob_pairs]
        logprob_outcomes = [y for _, y in logprob_pairs]
        logprob_ece, logprob_bins, _, _, _ = _ece_with_fallback(
            logprob_confidences, logprob_outcomes, n_bins_default=n_bins
        )
    else:
        logprob_ece = None
        logprob_bins = []

    return ECEResult(
        n=len(pool.confidences),
        n_bins=used_bins,
        fallback_used=fallback,
        verbalized=verbalized_ece,
        verbalized_bins=verbalized_bins,
        logprob=logprob_ece,
        logprob_bins=logprob_bins,
        n_logprob=len(logprob_pairs),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Refusal calibration
# ---------------------------------------------------------------------------


def measure_refusal_calibration(reps: Sequence[CalibrationRep]) -> RefusalCalibrationResult:
    """Per METHODOLOGY §3.4 / ADR-0005 §E.

    Computes the 2x2 confusion matrix on ``(task.difficulty, response.refused)``
    and reports refusal sensitivity / specificity each with a Wilson CI.
    Empty rows yield ``None`` for the corresponding scalar / CI rather
    than raising — a leaderboard entry with no hard tasks shouldn't
    crash this metric, but it should also not pretend to have a value.
    """
    refused_hard = 0
    refused_normal = 0
    answered_hard = 0
    answered_normal = 0
    for rep in reps:
        is_hard = rep.task.difficulty == "hard"
        if rep.response.refused:
            if is_hard:
                refused_hard += 1
            else:
                refused_normal += 1
        else:
            if is_hard:
                answered_hard += 1
            else:
                answered_normal += 1
    n_hard = refused_hard + answered_hard
    n_normal = refused_normal + answered_normal

    sensitivity_ci: WilsonCI | None
    sensitivity: float | None
    if n_hard > 0:
        sensitivity_ci = wilson_ci(successes=refused_hard, trials=n_hard)
        sensitivity = sensitivity_ci.proportion
    else:
        sensitivity_ci = None
        sensitivity = None

    specificity_ci: WilsonCI | None
    specificity: float | None
    if n_normal > 0:
        specificity_ci = wilson_ci(successes=answered_normal, trials=n_normal)
        specificity = specificity_ci.proportion
    else:
        specificity_ci = None
        specificity = None

    return RefusalCalibrationResult(
        refused_hard=refused_hard,
        refused_normal=refused_normal,
        answered_hard=answered_hard,
        answered_normal=answered_normal,
        n_hard=n_hard,
        n_normal=n_normal,
        sensitivity=sensitivity,
        sensitivity_ci=sensitivity_ci,
        specificity=specificity,
        specificity_ci=specificity_ci,
    )


# ---------------------------------------------------------------------------
# Overconfidence rate
# ---------------------------------------------------------------------------


def measure_overconfidence(
    reps: Sequence[CalibrationRep],
    *,
    threshold: float = OVERCONFIDENCE_THRESHOLD,
) -> OverconfidenceResult:
    """Per METHODOLOGY §3.5: Wilson CI on incorrect-with-high-confidence rate.

    ``count(incorrect ∧ confidence ≥ threshold) / count(answered)``. The
    pool is "answered" reps (refused + parser-failed reps excluded);
    ``threshold`` defaults to 0.9 per the methodology spec.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    pool = _build_pool(reps)
    n_answered = len(pool.confidences)
    if n_answered == 0:
        return OverconfidenceResult(
            threshold=threshold,
            n_answered=0,
            n_overconfident=0,
            rate=None,
            ci=None,
            reason="overconfidence requires at least 1 answered rep",
        )
    n_overconfident = sum(
        1 for c, y in zip(pool.confidences, pool.outcomes, strict=True) if y == 0 and c >= threshold
    )
    ci = wilson_ci(successes=n_overconfident, trials=n_answered)
    return OverconfidenceResult(
        threshold=threshold,
        n_answered=n_answered,
        n_overconfident=n_overconfident,
        rate=ci.proportion,
        ci=ci,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Aggregate result — what the HTML report consumes per (model, run)
# ---------------------------------------------------------------------------


class CalibrationDimension(BaseModel):
    """Combined calibration result for one (model, run) configuration.

    The CLI assembles one of these per model after running the benchmark
    and judging every rep. The HTML report (``reporting.html``) consumes
    this directly.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    n_tasks: int
    n_reps_total: int
    brier: BrierResult
    ece: ECEResult
    refusal: RefusalCalibrationResult
    overconfidence: OverconfidenceResult


def measure_calibration(
    reps: Sequence[CalibrationRep],
    *,
    model: str,
    n_tasks: int,
    seed: int = 0,
) -> CalibrationDimension:
    """Run all four calibration measurements over ``reps`` and bundle the result."""
    return CalibrationDimension(
        model=model,
        n_tasks=n_tasks,
        n_reps_total=len(reps),
        brier=measure_brier(reps, seed=seed),
        ece=measure_ece(reps),
        refusal=measure_refusal_calibration(reps),
        overconfidence=measure_overconfidence(reps),
    )


__all__ = [
    "OVERCONFIDENCE_THRESHOLD",
    "BrierResult",
    "CalibrationDimension",
    "CalibrationRep",
    "ECEResult",
    "OverconfidenceResult",
    "RefusalCalibrationResult",
    "measure_brier",
    "measure_calibration",
    "measure_ece",
    "measure_overconfidence",
    "measure_refusal_calibration",
    "reps_from_run_results",
]
