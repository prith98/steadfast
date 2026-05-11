"""Robustness dimension — typo / distractor / contradiction / long-context sub-metrics.

Per ``docs/METHODOLOGY.md`` §2 and ADR-0006, robustness sub-metrics fall
into three reporting shapes:

* **Delta-style** (typo, distractor): success-rate delta (perturbed minus
  clean) on the same task set, with a 95% paired-bootstrap CI on the
  delta itself per ADR-0006 §F.
* **3-way categorical** (contradiction): marginal proportions
  ``(p_detect, p_retry, p_halluc)`` with per-cell Wilson 95% CIs per
  ADR-0006 §D. The three CIs are not jointly bounded (sum-to-1 only at
  the point estimate); the result's ``notes`` field documents this honestly.
* **Curve + fit** (long_context): empirical success rate at each of
  ``[4k, 16k, 64k, 128k]`` token tiers with Wilson CIs, plus a sigmoid
  fit on ``log10(tokens)`` summarized by a slope coefficient and the
  derived ``L_50`` token count at which fitted success drops to 50%.
  Both summary statistics carry bootstrap CIs via task-level resampling
  per ADR-0006 §E.

The clean arm reuses the existing per-task ``RunResult`` produced by the
main bench loop — clean reps were already executed and judged before
this metric runs, so we don't re-spend tokens. The perturbed arm runs
N=10 distinct perturbations through ``agent.arun`` via
:func:`asyncio.gather` (the same precedent
:func:`steadfast.metrics.consistency.measure_output_consistency` set for
multi-input metric paths) and judges each response with the per-task
:class:`steadfast.judges.base.Judge` selected by ``Task.judge``.

For delta-style sub-metrics the CI is computed at the dimension level
across tasks. For ``n_tasks < 2`` (single-task ``--task`` invocation)
the paired bootstrap is undefined; the metric reports the per-task
point estimate and N/A's the CI with a populated ``reason`` field —
the same N/A pattern ADR-0004 §G uses for trajectory consistency on
toolless agents. Contradiction reuses the same N/A pattern when no rep
across any task had a non-empty trajectory (toolless agents).

References:

* ADR-0006 §B (seeds), §C (distractor bank), §D (contradiction labels +
  rule-based classifier), §F (paired bootstrap).
* METHODOLOGY §2 (robustness contract).
"""

from __future__ import annotations

import asyncio
import json
import logging
import warnings
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import OptimizeWarning, curve_fit

from steadfast.agent import Agent, AgentResponse, Task, ToolCall
from steadfast.judges import build_default_judge
from steadfast.models.base import BaseModelClient
from steadfast.perturbations import derive_seed
from steadfast.perturbations.contradiction import (
    CORRUPTED_CALLS_METADATA_KEY,
    load_detection_phrases,
)
from steadfast.perturbations.distractor import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DistractorBank,
    perturb_distractor,
    pick_distractor,
)
from steadfast.perturbations.long_context import (
    DEFAULT_FILLER_PATH as LONG_CONTEXT_DEFAULT_FILLER_PATH,
)
from steadfast.perturbations.long_context import (
    perturb_long_context,
)
from steadfast.perturbations.typo import (
    DEFAULT_MAX_WORD_CORRUPTION,
    DEFAULT_RATE,
    perturb_typo,
)
from steadfast.runner import RepStatus, RunResult
from steadfast.stats.bootstrap import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_METHOD,
    DEFAULT_N_RESAMPLES,
    bootstrap_ci,
)
from steadfast.stats.paired_bootstrap import paired_bootstrap_ci
from steadfast.stats.wilson import WilsonCI, wilson_ci

_log = logging.getLogger(__name__)

# Robustness sub-metric kinds shipped to date. Thursday adds "long_context".
# This is the user-facing union (CLI ``--robustness-types`` + SUPPORTED_KINDS
# membership). The delta-shaped result classes narrow ``kind`` to
# ``Literal["typo", "distractor"]`` so the discriminated union with
# ``ContradictionResult`` (kind="contradiction") works cleanly.
RobustnessKind = Literal["typo", "distractor", "contradiction", "long_context"]
SUPPORTED_KINDS: Final[frozenset[str]] = frozenset(
    {"typo", "distractor", "contradiction", "long_context"}
)

# Methodology-specified long-context tier ladder (METHODOLOGY §2.4 /
# ADR-0006 §E). The log10-spaced 4x-jump ladder maps to evenly-spaced
# x-axis points under the sigmoid fit.
DEFAULT_LONG_CONTEXT_LENGTHS: Final[tuple[int, ...]] = (4_000, 16_000, 64_000, 128_000)

# Three-way categorical labels per ADR-0006 §D. Decision rules in the
# classifier are evaluated in this priority order: detected wins over
# retried_or_escalated wins over hallucinated.
ContradictionLabel = Literal["detected", "retried_or_escalated", "hallucinated"]

# How many characters of the perturbed input to surface in per-rep diagnostics.
# Long enough to spot whether the perturbation worked at a glance; short
# enough to keep the JSON file sane at 10 reps x 5 tasks x 2 kinds.
_PERTURBED_PREVIEW_CHARS: Final[int] = 120


# ---------------------------------------------------------------------------
# Result Pydantic models
# ---------------------------------------------------------------------------


class RobustnessTaskResult(BaseModel):
    """Per-(task, kind) measurement.

    The per-rep boolean vectors are surfaced so a downstream debug pass
    or a future cluster-bootstrap path can recover them without rerunning
    the metric. ``perturbed_input_previews`` is the first
    ``_PERTURBED_PREVIEW_CHARS`` characters of each rep's perturbed input,
    a low-cost diagnostic that lets a reader visually confirm the
    perturbation took effect.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    # Narrower than ``RobustnessKind`` (which also includes "contradiction"):
    # contradiction has its own per-task result type because its shape is
    # categorical, not delta-style.
    kind: Literal["typo", "distractor"]
    n_reps_clean: int
    n_reps_perturbed: int
    clean_rate: float
    perturbed_rate: float
    delta: float
    clean_passes: list[bool]
    perturbed_passes: list[bool]
    perturbed_input_previews: list[str]
    seed: int
    distractor_snippet_ids: list[str] | None = None  # populated for kind=distractor
    notes: str | None = None


class RobustnessSubMetricResult(BaseModel):
    """Aggregate per-(model, kind) result with the paired-bootstrap CI.

    For ``n_tasks >= 2`` the headline numbers are populated and
    ``delta_ci_lower`` / ``delta_ci_upper`` carry the bootstrap CI. For
    ``n_tasks < 2`` (single-task invocation), the headline scalars are
    populated from the lone task's per-task result but the CI is
    ``None`` and ``reason`` explains why — matches ADR-0004 §G's N/A
    pattern.
    """

    model_config = ConfigDict(frozen=True)

    # Narrower than ``RobustnessKind``: contradiction has its own aggregate
    # result type. Pydantic's union discrimination uses this narrower
    # Literal to disambiguate from :class:`ContradictionResult`.
    kind: Literal["typo", "distractor"]
    n_tasks: int
    clean_mean: float | None
    perturbed_mean: float | None
    delta: float | None
    delta_ci_lower: float | None
    delta_ci_upper: float | None
    confidence_level: float | None
    method: str | None
    n_resamples: int | None
    degenerate: bool = False
    per_task: list[RobustnessTaskResult]
    reason: str | None = None


class ContradictionTaskResult(BaseModel):
    """Per-task contradiction labels and per-rep diagnostics.

    Parallel to :class:`RobustnessTaskResult` but for the 3-way categorical
    metric. ``labels`` carries one entry per rep that had a non-empty
    trajectory (reps with empty trajectories are excluded — toolless reps
    are not measured per ADR-0006 §D / ADR-0004 §G); ``n_reps_with_tools``
    is ``len(labels)``. ``n_corrupted_calls_per_rep`` is the diagnostic
    parallel to :attr:`RobustnessTaskResult.distractor_snippet_ids`.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    kind: Literal["contradiction"] = "contradiction"
    n_reps_with_tools: int
    n_reps_completed: int
    labels: list[ContradictionLabel]
    n_corrupted_calls_per_rep: list[int]
    seed: int
    notes: str | None = None


class ContradictionResult(BaseModel):
    """Aggregate contradiction result across tasks for one model.

    Reports the three marginal proportions ``(p_detect, p_retry, p_halluc)``
    with per-cell Wilson 95% CIs (ADR-0006 §D). The CIs are not jointly
    bounded — the three proportions sum to 1 only at the point estimate.
    The :attr:`notes` field documents this honestly so a leaderboard reader
    doesn't mistake the three intervals for a Dirichlet credible region.

    For toolless runs (no rep across any task had a non-empty trajectory),
    :attr:`value` is ``None`` and :attr:`reason` is populated, matching the
    N/A pattern :class:`RobustnessSubMetricResult` uses (ADR-0004 §G).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["contradiction"] = "contradiction"
    n_tasks: int
    n_reps_with_tools: int
    p_detect: float | None
    p_retry: float | None
    p_halluc: float | None
    ci_detect: WilsonCI | None
    ci_retry: WilsonCI | None
    ci_halluc: WilsonCI | None
    per_task: list[ContradictionTaskResult]
    # ``value="measured"`` when n_reps_with_tools > 0; ``None`` on the N/A
    # path (no rep had any tool call across any task). Kept as an explicit
    # field so HTML / leaderboard consumers can branch on it without
    # re-deriving from p_detect.
    value: Literal["measured"] | None = "measured"
    reason: str | None = None
    # ``notes`` is populated only on the measured path — it documents the
    # marginal-CI semantics. On the N/A path the field is ``None``; the
    # ``reason`` field carries the diagnostic instead, matching the pattern
    # :class:`RobustnessSubMetricResult` uses.
    notes: str | None = None


class LongContextTaskResult(BaseModel):
    """Per-task long-context measurement at each length tier.

    ``passes_per_length[i]`` is the per-rep boolean vector at
    ``lengths[i]``; ``rates_per_length[i]`` is its mean. A length tier
    that was skipped for this task (because the task input is already
    longer than the tier) appears as an empty inner list at that index
    and the corresponding rate is ``None``. ``notes`` records any
    per-tier skips so the diagnostic surface is preserved.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    kind: Literal["long_context"] = "long_context"
    lengths: list[int]
    passes_per_length: list[list[bool]]
    rates_per_length: list[float | None]
    perturbed_input_previews_per_length: list[list[str]]
    seed: int  # base seed = derive_seed(task_id, "long_context") (per-rep extends this)
    notes: str | None = None


class LongContextResult(BaseModel):
    """Aggregate long-context degradation curve + sigmoid fit per ADR-0006 §E.

    The empirical curve is the methodologically primary artifact;
    ``success_rates`` and ``success_cis`` pool successes/trials across
    tasks at each length tier with one Wilson 95% CI per tier. The
    sigmoid fit summarizes the curve with two derived quantities:

    * ``slope`` — coefficient on ``log10(tokens)`` in
      ``p(L) = 1 / (1 + exp(-(a + b · log10(L))))``. Negative ``slope``
      = the agent degrades with length (the expected case); positive
      ``slope`` = the agent gets better with more context (a real
      surprise per ADR-0006 §E's "warrants investigation" rule).
    * ``l50`` — derived ``10^(-a/b)``, the token count at which fitted
      success probability drops to 0.5.

    Both come with bootstrap CIs: resample tasks (with their per-tier
    rate arrays as a unit), refit per resample, take the 2.5/97.5
    percentiles. Implementation reuses
    :func:`steadfast.stats.bootstrap.bootstrap_ci` with a custom
    statistic per ADR-0006 §E.

    On fit-convergence failure (e.g., a flat curve where the sigmoid is
    ill-conditioned, or a curve with no monotone trend), ``slope`` /
    ``slope_ci_*`` / ``l50`` / ``l50_ci_*`` are all ``None`` and
    ``fit_converged`` is ``False``; the empirical curve still reports.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["long_context"] = "long_context"
    n_tasks: int
    lengths: list[int]

    # Pooled empirical curve — one Wilson CI per length tier. Tiers
    # where no task produced a measurement are omitted from these two
    # parallel lists; ``measured_length_indices`` (below) records which
    # entries of ``lengths`` were actually pooled so consumers can
    # align the curve back to the input tier ladder.
    success_rates: list[float]
    success_cis: list[WilsonCI]
    measured_length_indices: list[int]  # indices into ``lengths`` parallel to success_rates

    # Sigmoid fit summary (all None when ``fit_converged is False``)
    slope: float | None
    slope_ci_lower: float | None
    slope_ci_upper: float | None
    l50: float | None
    l50_ci_lower: float | None
    l50_ci_upper: float | None
    fit_converged: bool

    confidence_level: float | None
    n_resamples: int | None

    per_task: list[LongContextTaskResult]
    reason: str | None = None  # populated on N/A paths (n_tasks<2, no measured tiers)
    notes: str | None = None


class RobustnessDimension(BaseModel):
    """Combined robustness result for one (model, run) configuration.

    Mirrors :class:`steadfast.metrics.calibration.CalibrationDimension`
    in shape — the HTML report consumes a single ``robustness.json`` per
    model with all sub-metrics nested. The dict value is a Pydantic 2
    "smart" union of the delta-style, contradiction, and long-context
    shapes, discriminated by each member's ``kind`` Literal.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    n_tasks: int
    sub_metrics: dict[
        str,
        RobustnessSubMetricResult | ContradictionResult | LongContextResult,
    ] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completed_passes(run_result: RunResult) -> list[bool]:
    """Extract per-rep ``passed`` booleans from a judged ``RunResult``.

    Filters to reps that completed AND were judged. Reps that failed
    (network error, parse failure) or never received a verdict are
    excluded — their contribution would bias the rate downward without
    being a real "model failed the perturbation" signal.
    """
    out: list[bool] = []
    for rep in run_result.reps:
        if rep.status != RepStatus.COMPLETED:
            continue
        if rep.verdict is None:
            continue
        out.append(bool(rep.verdict.passed))
    return out


PerturbFn = Callable[[str, int], str]
"""Signature for a perturbation: ``(text, seed) -> perturbed_text``.

Both implementations bind their methodology constants (typo's ``rate`` /
``max_word_corruption``, distractor's ``min_tokens`` / ``max_tokens``) at
:mod:`measure_*_robustness` call time before passing through this seam,
so the per-task helper stays kind-agnostic.
"""


async def _measure_per_task(
    *,
    task: Task,
    kind: Literal["typo", "distractor"],
    clean_passes: list[bool],
    perturb_fn: PerturbFn,
    agent: Agent,
    rubric_client: BaseModelClient | None,
    reps: int,
    distractor_snippet_ids: list[str] | None = None,
) -> RobustnessTaskResult:
    """Measure one (task, kind) — runs the perturbed arm and pairs with the clean arm.

    The perturbed arm: 10 distinct perturbed inputs (per-rep seeds via
    :func:`steadfast.perturbations.derive_seed`), agent.arun on each via
    gather, judge each response sequentially, reduce to a per-rep
    ``passed`` vector. Reps that fail mid-pipeline (transient agent
    error, rubric-judge parse failure) drop out of the rate's denominator
    *and* out of the per-rep diagnostic lists, which stay parallel by
    construction.
    """
    if reps < 1:
        raise ValueError(f"reps must be >= 1; got {reps}")

    # Build N=reps perturbed inputs deterministically. Per ADR-0006 §B,
    # the per-rep seed is sha256(f"{task.id}:{kind}:v1:rep{idx}")[:8].
    perturbed_inputs: list[str] = []
    for rep_idx in range(reps):
        rep_seed = derive_seed(task.id, kind, rep_idx=rep_idx)
        perturbed_inputs.append(perturb_fn(task.input, rep_seed))

    # Confidence suffix is preserved on the perturbed arm: the clean arm
    # ran with whatever suffix the CLI applied (depends on whether
    # calibration was co-requested). The perturbed arm must match so the
    # arm-delta isolates input perturbation rather than conflating it with
    # elicitation-presence. See notes/tradeoffs_log.md §T6.
    perturbed_tasks = [task.model_copy(update={"input": pi}) for pi in perturbed_inputs]

    # Run the perturbed arm. ``return_exceptions=True`` so a single transient
    # failure (rate-limit retry exhaustion, Gemini safety filter on a
    # particular distractor) doesn't take the whole task's measurement
    # down — it just shrinks the perturbed-arm denominator. The skipped
    # reps are recorded in the result's ``notes``.
    raw_results: list[AgentResponse | BaseException] = await asyncio.gather(
        *(agent.arun(t) for t in perturbed_tasks),
        return_exceptions=True,
    )

    # Single sequential pass over agent results: filter out arun failures,
    # judge survivors, filter out judge failures, and grow the per-rep
    # diagnostic lists in lockstep with the surviving rep set. This
    # guarantees ``perturbed_passes[i]``, ``perturbed_input_previews[i]``,
    # and ``snippet_ids_aligned[i]`` (when present) refer to the same
    # rep — without it, an agent.arun failure mid-arm would leave the
    # diagnostic indices misaligned with the verdict indices.
    judge = build_default_judge(task, rubric_client=rubric_client)
    perturbed_passes: list[bool] = []
    perturbed_input_previews: list[str] = []
    snippet_ids_aligned: list[str] | None = [] if distractor_snippet_ids is not None else None
    n_arun_failures = 0
    n_judge_failures = 0
    for rep_idx, raw in enumerate(raw_results):
        if isinstance(raw, BaseException):
            n_arun_failures += 1
            _log.warning(
                "agent.arun failed on perturbed rep for task=%s kind=%s rep=%d: %s",
                task.id,
                kind,
                rep_idx,
                raw,
            )
            continue
        try:
            verdict = await judge.ajudge(task, raw)
        except Exception as exc:
            # Broaden beyond JudgeError to also absorb transient
            # underlying-LLM failures the rubric judge can leak through
            # (e.g., an OpenAI rate-limit cascade after tenacity retries
            # are exhausted). Per the same fail-soft contract as the
            # arun path above: a failed rep drops out of the denominator
            # rather than killing the whole metric. Catches both the
            # JudgeError surface and provider-SDK exceptions.
            n_judge_failures += 1
            _log.warning(
                "judge failed on perturbed rep for task=%s kind=%s rep=%d: %s — excluding from arm",
                task.id,
                kind,
                rep_idx,
                exc,
            )
            continue
        perturbed_passes.append(bool(verdict.passed))
        perturbed_input_previews.append(perturbed_inputs[rep_idx][:_PERTURBED_PREVIEW_CHARS])
        if snippet_ids_aligned is not None and distractor_snippet_ids is not None:
            snippet_ids_aligned.append(distractor_snippet_ids[rep_idx])

    n_clean = len(clean_passes)
    n_perturbed = len(perturbed_passes)

    clean_rate = sum(clean_passes) / n_clean if n_clean > 0 else 0.0
    perturbed_rate = sum(perturbed_passes) / n_perturbed if n_perturbed > 0 else 0.0
    delta = perturbed_rate - clean_rate

    notes_parts: list[str] = []
    if n_arun_failures:
        notes_parts.append(f"{n_arun_failures} perturbed rep(s) failed during agent.arun")
    if n_judge_failures:
        notes_parts.append(f"{n_judge_failures} perturbed rep(s) failed during judging")
    if n_clean < reps:
        notes_parts.append(
            f"clean arm has {n_clean} judged reps (expected {reps}); "
            "rate computed over completed-and-judged subset"
        )

    return RobustnessTaskResult(
        task_id=task.id,
        kind=kind,
        n_reps_clean=n_clean,
        n_reps_perturbed=n_perturbed,
        clean_rate=clean_rate,
        perturbed_rate=perturbed_rate,
        delta=delta,
        clean_passes=list(clean_passes),
        perturbed_passes=list(perturbed_passes),
        perturbed_input_previews=perturbed_input_previews,
        seed=derive_seed(task.id, kind),
        distractor_snippet_ids=snippet_ids_aligned,
        notes="; ".join(notes_parts) if notes_parts else None,
    )


def _aggregate_sub_metric(
    *,
    kind: Literal["typo", "distractor"],
    per_task: list[RobustnessTaskResult],
    seed: int,
) -> RobustnessSubMetricResult:
    """Aggregate per-task results into a sub-metric result with the paired-bootstrap CI.

    For ``n_tasks >= 2`` calls :func:`paired_bootstrap_ci` over the per-
    task (clean_rate, perturbed_rate) pairs. For ``n_tasks < 2`` returns
    a structured N/A per ADR-0004 §G — point estimate populated from
    the single task, CI fields ``None``, ``reason`` populated.
    """
    n_tasks = len(per_task)
    if n_tasks == 0:
        return RobustnessSubMetricResult(
            kind=kind,
            n_tasks=0,
            clean_mean=None,
            perturbed_mean=None,
            delta=None,
            delta_ci_lower=None,
            delta_ci_upper=None,
            confidence_level=None,
            method=None,
            n_resamples=None,
            per_task=[],
            reason="no tasks measured",
        )

    if n_tasks < 2:
        only = per_task[0]
        return RobustnessSubMetricResult(
            kind=kind,
            n_tasks=1,
            clean_mean=only.clean_rate,
            perturbed_mean=only.perturbed_rate,
            delta=only.delta,
            delta_ci_lower=None,
            delta_ci_upper=None,
            confidence_level=None,
            method=None,
            n_resamples=None,
            per_task=per_task,
            reason="paired bootstrap CI requires n_tasks >= 2; reported point estimate only",
        )

    clean_rates = [r.clean_rate for r in per_task]
    perturbed_rates = [r.perturbed_rate for r in per_task]
    ci = paired_bootstrap_ci(clean_rates, perturbed_rates, seed=seed)
    return RobustnessSubMetricResult(
        kind=kind,
        n_tasks=n_tasks,
        clean_mean=ci.clean_mean,
        perturbed_mean=ci.perturbed_mean,
        delta=ci.delta,
        delta_ci_lower=ci.delta_ci_lower,
        delta_ci_upper=ci.delta_ci_upper,
        confidence_level=ci.confidence_level,
        method=ci.method,
        n_resamples=ci.n_resamples,
        degenerate=ci.degenerate,
        per_task=per_task,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Public API — per-kind measurement, plus the bundling wrapper
# ---------------------------------------------------------------------------


async def measure_typo_robustness(
    *,
    tasks: Sequence[Task],
    clean_run_results: Sequence[RunResult],
    agent: Agent,
    rubric_client: BaseModelClient | None,
    reps: int = 10,
    rate: float = DEFAULT_RATE,
    max_word_corruption: float = DEFAULT_MAX_WORD_CORRUPTION,
    aggregate_seed: int = 0,
) -> RobustnessSubMetricResult:
    """Per METHODOLOGY §2.1: 5%-rate character-level noise, 25% per-word cap.

    ``tasks`` and ``clean_run_results`` are paired by index — element ``i``
    of each is the same task's ``Task`` object and its already-judged
    clean ``RunResult``. ``aggregate_seed`` seeds the paired bootstrap
    over per-task rates (``rate`` and ``max_word_corruption`` configure
    the perturbation; their defaults match METHODOLOGY §2.1).
    """
    if len(tasks) != len(clean_run_results):
        raise ValueError(
            f"len(tasks)={len(tasks)} != len(clean_run_results)="
            f"{len(clean_run_results)}; pair task[i] with clean_run_results[i]"
        )

    def _typo(text: str, seed: int) -> str:
        return perturb_typo(
            text,
            rate=rate,
            max_word_corruption=max_word_corruption,
            seed=seed,
        )

    per_task: list[RobustnessTaskResult] = []
    for task, clean in zip(tasks, clean_run_results, strict=True):
        clean_passes = _completed_passes(clean)
        result = await _measure_per_task(
            task=task,
            kind="typo",
            clean_passes=clean_passes,
            perturb_fn=_typo,
            agent=agent,
            rubric_client=rubric_client,
            reps=reps,
        )
        per_task.append(result)

    return _aggregate_sub_metric(kind="typo", per_task=per_task, seed=aggregate_seed)


async def measure_distractor_robustness(
    *,
    tasks: Sequence[Task],
    clean_run_results: Sequence[RunResult],
    agent: Agent,
    rubric_client: BaseModelClient | None,
    distractor_banks: dict[str, DistractorBank],
    reps: int = 10,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    aggregate_seed: int = 0,
) -> RobustnessSubMetricResult:
    """Per METHODOLOGY §2.2: prepend a 200-800-token distractor per (task, rep).

    ``distractor_banks`` is keyed by ``Task.domain``; each task picks its
    bank by domain. Tasks whose domain has no bank are skipped (with a
    log warning) — partial coverage is preferred to a hard failure on an
    unsupplied domain, since the rest of the benchmark may still produce
    signal.
    """
    if len(tasks) != len(clean_run_results):
        raise ValueError(
            f"len(tasks)={len(tasks)} != len(clean_run_results)={len(clean_run_results)}"
        )

    per_task: list[RobustnessTaskResult] = []
    for task, clean in zip(tasks, clean_run_results, strict=True):
        bank = distractor_banks.get(task.domain)
        if bank is None:
            _log.warning(
                "no distractor bank for domain=%r — skipping distractor robustness for task=%s",
                task.domain,
                task.id,
            )
            continue

        clean_passes = _completed_passes(clean)

        # Pre-pick snippet IDs for diagnostic output. The actual perturbation
        # happens inside the closure below, but recording the ids here keeps
        # them available even if the closure is never called (e.g., reps=0
        # smoke check).
        snippet_ids: list[str] = []
        for rep_idx in range(reps):
            rep_seed = derive_seed(task.id, "distractor", rep_idx=rep_idx)
            chosen = pick_distractor(
                bank, seed=rep_seed, min_tokens=min_tokens, max_tokens=max_tokens
            )
            snippet_ids.append(chosen.id)

        def _distractor(text: str, seed: int, _bank: DistractorBank = bank) -> str:
            return perturb_distractor(
                text,
                bank=_bank,
                seed=seed,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )

        result = await _measure_per_task(
            task=task,
            kind="distractor",
            clean_passes=clean_passes,
            perturb_fn=_distractor,
            agent=agent,
            rubric_client=rubric_client,
            reps=reps,
            distractor_snippet_ids=snippet_ids,
        )
        per_task.append(result)

    return _aggregate_sub_metric(kind="distractor", per_task=per_task, seed=aggregate_seed)


# ---------------------------------------------------------------------------
# Contradiction (3-way categorical) — Wednesday's surface per ADR-0006 §D
# ---------------------------------------------------------------------------


def _args_key(args: dict[str, str | int | float | bool]) -> frozenset[tuple[str, object]]:
    """Hashable structural key for ``ToolCall.args`` equality matching.

    ``ToolCall.args`` is a ``dict[str, MetadataValue]`` (scalar union) per
    ADR-0002 §A.3. ``frozenset(items)`` is the standard canonicalization
    for dict-as-key under Python's hashable semantics.
    """
    return frozenset(args.items())


def _has_post_corruption_retry(trajectory: list[ToolCall], corrupted_calls: list[ToolCall]) -> bool:
    """Return True iff some corrupted call's ``(name, args)`` repeats later.

    The retry-rule helper for :func:`classify_contradiction_response`. Per
    ADR-0006 §D's rule 2: "trajectory shows at least one tool call repeated
    with the same args after a corrupted response." The "after" is index
    order; identity by structural ``(name, args)``.

    A duplicate call appearing **before** the earliest corruption does not
    count — the retry must be a reaction to a corruption the agent has
    already seen.
    """
    if not corrupted_calls or not trajectory:
        return False
    is_corrupted = [tc in corrupted_calls for tc in trajectory]
    for i, corrupted in enumerate(is_corrupted):
        if not corrupted:
            continue
        target = (trajectory[i].name, _args_key(trajectory[i].args))
        for j in range(i + 1, len(trajectory)):
            later = trajectory[j]
            if (later.name, _args_key(later.args)) == target:
                return True
    return False


def _extract_corrupted_calls(response: AgentResponse) -> list[ToolCall]:
    """Read the corruption-indices metadata convention into a list of ToolCalls.

    The agent populates
    ``response.metadata[CORRUPTED_CALLS_METADATA_KEY]`` with a JSON-encoded
    ``list[int]`` of indices into ``response.trajectory`` whose result was
    corrupted. Malformed metadata (missing key, non-string value, JSON
    decode error, non-list payload, out-of-range indices, non-int entries)
    silently degrades to an empty list with a logged warning — the rep is
    still counted in ``n_reps_with_tools`` but the retry rule simply can't
    fire (which is the right calibration: an agent that doesn't signal
    corruption can never get the retry credit).
    """
    raw = response.metadata.get(CORRUPTED_CALLS_METADATA_KEY)
    if raw is None:
        return []
    if not isinstance(raw, str):
        _log.warning(
            "ignoring %s metadata: expected JSON-string, got %s",
            CORRUPTED_CALLS_METADATA_KEY,
            type(raw).__name__,
        )
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        _log.warning("ignoring %s metadata: not valid JSON: %r", CORRUPTED_CALLS_METADATA_KEY, raw)
        return []
    if not isinstance(decoded, list):
        return []
    n_traj = len(response.trajectory)
    return [response.trajectory[i] for i in decoded if isinstance(i, int) and 0 <= i < n_traj]


def classify_contradiction_response(
    task: Task,
    response: AgentResponse,
    corrupted_calls: list[ToolCall],
    *,
    detection_phrases: list[str],
    escalation_phrases: list[str],
) -> ContradictionLabel:
    """Rule-based classifier per ADR-0006 §D, evaluated in priority order.

    1. ``detected`` if ``response.refused`` is True OR the lowercased answer
       text contains any phrase from ``detection_phrases``.
    2. ``retried_or_escalated`` if the trajectory shows a same-args retry
       of a corrupted tool call (per :func:`_has_post_corruption_retry`)
       OR the lowercased answer text contains any phrase from
       ``escalation_phrases``.
    3. ``hallucinated`` otherwise.

    Phrase matching is case-insensitive substring containment; phrase
    lists are pre-lowercased by :func:`load_detection_phrases`. The
    classifier is deterministic — a given ``(task, response,
    corrupted_calls)`` triple always produces the same label.

    Why rule-based not LLM-judged (per ADR-0006 §D): adds a fourth
    infrastructure-LLM judge surface to maintain (cost + bias risk per
    ADR-0001), and the rules above key on signals already in
    :class:`AgentResponse` (no new parsing surface). v0.2 path:
    :class:`ContradictionRubricJudge` for cases where the rule-based
    classifier returns ``hallucinated`` but a human would say ``detected``.
    """
    del task  # signature kept judge-shape compatible per ADR-0006 §D
    answer_lower = response.answer.lower()

    # Rule 1: detected.
    if response.refused:
        return "detected"
    if any(phrase in answer_lower for phrase in detection_phrases):
        return "detected"

    # Rule 2: retried_or_escalated.
    if _has_post_corruption_retry(response.trajectory, corrupted_calls):
        return "retried_or_escalated"
    if any(phrase in answer_lower for phrase in escalation_phrases):
        return "retried_or_escalated"

    # Rule 3: hallucinated (fallthrough).
    return "hallucinated"


async def measure_contradiction_handling(
    *,
    tasks: Sequence[Task],
    agent: Agent,
    reps: int = 10,
    detection_phrases: list[str] | None = None,
    escalation_phrases: list[str] | None = None,
) -> ContradictionResult:
    """Per METHODOLOGY §2.3 / ADR-0006 §D: 3-bar marginal with per-cell Wilson CIs.

    Runs ``reps`` invocations of ``agent.arun(task)`` per task. The agent is
    expected to wire the contradiction perturbation into its tool-execution
    loop (corrupting tool results per
    :func:`steadfast.perturbations.contradiction.should_corrupt`) and to
    set ``response.metadata[CORRUPTED_CALLS_METADATA_KEY]`` with the
    JSON-encoded list of corrupted call indices per the metadata convention.

    Reps with an empty ``trajectory`` are excluded from the per-task
    ``labels`` list (toolless reps don't measure contradiction handling).
    Reps with a non-empty trajectory but empty ``corrupted_calls`` (e.g.,
    p=0.3 happened to land all "no corruption" coins on the rep) still
    count toward ``n_reps_with_tools`` and naturally fall through to the
    ``hallucinated`` bucket — a model that produces an answer when no
    contradiction is present is correctly characterized as not having
    detected / retried (because there was nothing to detect or retry).

    Toolless run (no rep across any task had a non-empty trajectory)
    returns ``ContradictionResult(value=None, reason="agent did not call
    any tools")`` per ADR-0004 §G's N/A pattern.

    Parameters
    ----------
    tasks:
        The benchmark tasks. Each is run independently for ``reps``
        invocations.
    agent:
        The :class:`Agent` under measurement. Must populate
        ``response.metadata[CORRUPTED_CALLS_METADATA_KEY]`` for the retry
        rule to fire; agents that don't signal corruption can still be
        measured (the rule simply never fires for them, which biases their
        scoring toward ``hallucinated``).
    reps:
        Per-task repetition count. Methodology default is N=10.
    detection_phrases, escalation_phrases:
        Optional pre-loaded phrase lists. Default to the frozen file at
        ``prompts/contradiction_detection_phrases_v1.txt``. Tests pass
        explicit lists to keep the unit tests file-system-independent.
    """
    if reps < 1:
        raise ValueError(f"reps must be >= 1; got {reps}")

    # Both phrase lists default to the frozen v1 file (loaded together so the
    # detection / escalation pair stays consistent). Tests pass explicit lists
    # to keep them file-system-independent; production callers omit both.
    if detection_phrases is None:
        detection_phrases, default_escalation = load_detection_phrases()
        if escalation_phrases is None:
            escalation_phrases = default_escalation
    elif escalation_phrases is None:
        _, escalation_phrases = load_detection_phrases()

    per_task: list[ContradictionTaskResult] = []
    total_with_tools = 0
    total_detect = 0
    total_retry = 0
    total_halluc = 0

    for task in tasks:
        raw_results: list[AgentResponse | BaseException] = await asyncio.gather(
            *(agent.arun(task) for _ in range(reps)),
            return_exceptions=True,
        )

        labels: list[ContradictionLabel] = []
        n_corrupted_per_rep: list[int] = []
        n_arun_failures = 0
        n_empty_trajectory = 0

        for rep_idx, raw in enumerate(raw_results):
            if isinstance(raw, BaseException):
                n_arun_failures += 1
                _log.warning(
                    "agent.arun failed on contradiction rep for task=%s rep=%d: %s",
                    task.id,
                    rep_idx,
                    raw,
                )
                continue
            if not raw.trajectory:
                # Toolless rep — does not measure contradiction handling.
                n_empty_trajectory += 1
                continue

            corrupted_calls = _extract_corrupted_calls(raw)
            label = classify_contradiction_response(
                task,
                raw,
                corrupted_calls,
                detection_phrases=detection_phrases,
                escalation_phrases=escalation_phrases,
            )
            labels.append(label)
            n_corrupted_per_rep.append(len(corrupted_calls))

            if label == "detected":
                total_detect += 1
            elif label == "retried_or_escalated":
                total_retry += 1
            else:
                total_halluc += 1

        n_with_tools_this_task = len(labels)
        total_with_tools += n_with_tools_this_task

        notes_parts: list[str] = []
        if n_arun_failures:
            notes_parts.append(f"{n_arun_failures} arun failure(s)")
        if n_empty_trajectory:
            notes_parts.append(f"{n_empty_trajectory} rep(s) had empty trajectory")

        per_task.append(
            ContradictionTaskResult(
                task_id=task.id,
                n_reps_with_tools=n_with_tools_this_task,
                n_reps_completed=reps - n_arun_failures,
                labels=labels,
                n_corrupted_calls_per_rep=n_corrupted_per_rep,
                seed=derive_seed(task.id, "contradiction"),
                notes="; ".join(notes_parts) if notes_parts else None,
            )
        )

    if total_with_tools == 0:
        return ContradictionResult(
            n_tasks=len(tasks),
            n_reps_with_tools=0,
            p_detect=None,
            p_retry=None,
            p_halluc=None,
            ci_detect=None,
            ci_retry=None,
            ci_halluc=None,
            per_task=per_task,
            value=None,
            reason="agent did not call any tools",
        )

    return ContradictionResult(
        n_tasks=len(tasks),
        n_reps_with_tools=total_with_tools,
        p_detect=total_detect / total_with_tools,
        p_retry=total_retry / total_with_tools,
        p_halluc=total_halluc / total_with_tools,
        ci_detect=wilson_ci(total_detect, total_with_tools),
        ci_retry=wilson_ci(total_retry, total_with_tools),
        ci_halluc=wilson_ci(total_halluc, total_with_tools),
        per_task=per_task,
        value="measured",
        notes=(
            "Wilson 95% CIs are per-cell marginals; the three CIs are not "
            "jointly bounded — sum-to-1 holds at the point estimate but not "
            "within the intervals. v0.2 may add a Dirichlet-multinomial "
            "credible region (ADR-0006 §D)."
        ),
    )


# ---------------------------------------------------------------------------
# Long-context (curve + sigmoid fit) — Thursday's surface per ADR-0006 §E
# ---------------------------------------------------------------------------


def _sigmoid_log10(log10_l: np.ndarray, a: float, b: float) -> np.ndarray:
    """``p(L) = 1 / (1 + exp(-(a + b · log10(L))))`` evaluated on a log10 axis.

    The fit is parameterized in ``log10(L)`` rather than ``L`` so the
    four METHODOLOGY-specified length tiers (4k, 16k, 64k, 128k) sit
    evenly on the x-axis (each 4x jump = one unit on log10). Caller
    passes the *already-log10ed* tier values per ADR-0006 §E.
    """
    return 1.0 / (1.0 + np.exp(-(a + b * log10_l)))


def _fit_sigmoid(
    lengths: Sequence[int], rates: Sequence[float]
) -> tuple[float, float, float] | None:
    """Fit the long-context sigmoid; return ``(a, b, l50)`` or ``None``.

    ``rates`` are aggregated success rates at each entry of ``lengths``;
    NaN-valued entries (length tiers with no measurement) are filtered
    out before fitting. Returns ``None`` if fewer than two finite
    points remain (fit ill-conditioned), if ``curve_fit`` raises
    :class:`RuntimeError`, or if ``OptimizeWarning`` was emitted, or
    if the fitted slope is too close to zero for ``l50 = 10^(-a/b)`` to
    be numerically meaningful.

    Initial guesses: ``a=0`` (centered around 50%-success at log10=0),
    ``b=-1`` (mild negative slope on the log10 axis). The slope's sign
    is intentionally guessed negative so the optimizer doesn't land on
    a mirror-image positive-slope basin when the curve actually
    degrades.
    """
    finite_pairs = [(lo, r) for lo, r in zip(lengths, rates, strict=True) if np.isfinite(r)]
    if len(finite_pairs) < 2:
        return None
    finite_lengths = np.array([p[0] for p in finite_pairs], dtype=float)
    finite_rates = np.array([p[1] for p in finite_pairs], dtype=float)
    log10_l = np.log10(finite_lengths)

    try:
        with warnings.catch_warnings():
            # Treat optimizer warnings (covariance estimation issues,
            # ill-conditioned Jacobians) as failures rather than letting
            # them silently leak through.
            warnings.simplefilter("error", OptimizeWarning)
            popt, _pcov = curve_fit(
                _sigmoid_log10,
                log10_l,
                finite_rates,
                p0=(0.0, -1.0),
                maxfev=5000,
            )
    except (RuntimeError, OptimizeWarning, ValueError):
        return None

    a = float(popt[0])
    b = float(popt[1])
    # l50 is the token count where the fitted probability crosses 0.5.
    # ``log10(l50) = -a/b`` is well-defined only when ``b`` is non-zero;
    # near-zero slope means the curve is essentially flat and l50 would
    # blow up. Treat as non-convergence.
    if abs(b) < 1e-9:
        return None
    log10_l50 = -a / b
    # Guard against pathological fits that place l50 absurdly far from
    # the measured tier range (e.g., 10^100). The tier ladder spans
    # ~4 orders of magnitude; clamp the well-defined range to
    # log10 ∈ [0, 9] (1 token to ~1B tokens).
    if not (0.0 <= log10_l50 <= 9.0):
        return None
    return a, b, 10**log10_l50


def _bootstrap_curve_cis(
    *,
    per_task_rates: np.ndarray,
    lengths: Sequence[int],
    seed: int,
    n_resamples: int,
    confidence_level: float,
) -> tuple[float, float, float, float] | None:
    """Bootstrap CI on ``(slope, l50)`` per ADR-0006 §E.

    ``per_task_rates`` has shape ``(n_tasks, n_lengths)`` with NaN at
    tiers that weren't measured for a given task. Resamples task rows
    with replacement, computes ``nanmean`` per tier on the resampled
    set, refits the sigmoid, collects ``(slope, l50)``. Reuses
    :func:`steadfast.stats.bootstrap.bootstrap_ci` with a custom
    statistic that closes over the rate matrix — per ADR-0006 §E's
    "Reuses ``stats.bootstrap.bootstrap_ci`` with a custom statistic"
    contract.

    Returns ``(slope_lo, slope_hi, l50_lo, l50_hi)`` or ``None`` if the
    bootstrap couldn't produce a finite CI (e.g., the resampled fits
    failed too often to estimate a percentile reliably).
    """
    n_tasks = per_task_rates.shape[0]
    if n_tasks < 2:
        return None

    # Pre-suppress NaN warnings — many resamples produce all-NaN columns
    # by chance on small N, which is the expected degenerate path; we
    # surface them as failed fits, not as runtime warnings.
    def _resampled_rates(idx_arr: np.ndarray) -> np.ndarray:
        idx = idx_arr.astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.asarray(np.nanmean(per_task_rates[idx], axis=0), dtype=float)

    def fit_slope(idx_arr: np.ndarray) -> float:
        rates = _resampled_rates(idx_arr)
        fit = _fit_sigmoid(lengths, rates.tolist())
        return float("nan") if fit is None else fit[1]

    def fit_l50(idx_arr: np.ndarray) -> float:
        rates = _resampled_rates(idx_arr)
        fit = _fit_sigmoid(lengths, rates.tolist())
        return float("nan") if fit is None else fit[2]

    # The bootstrap reuses the canonical bootstrap_ci wrapper per ADR-0006 §E.
    # We pass task indices [0..n-1] as the data; the statistic closes over
    # `per_task_rates` to recover the per-resample rate vector.
    indices = list(range(n_tasks))
    try:
        with warnings.catch_warnings():
            # scipy.stats.bootstrap's BCa path divides by an acceleration-
            # term denominator that goes to zero when every resample yields
            # the same slope (degenerate case — all tasks share an identical
            # curve). The bootstrap_ci wrapper already NaN-clamps the
            # resulting interval to the point estimate, so the runtime
            # warning is pure noise.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            slope_ci = bootstrap_ci(
                indices,
                fit_slope,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                method=DEFAULT_METHOD,
                seed=seed,
            )
            l50_ci = bootstrap_ci(
                indices,
                fit_l50,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                method=DEFAULT_METHOD,
                seed=seed,
            )
    except (ValueError, RuntimeError):
        return None

    if not (
        np.isfinite(slope_ci.ci_lower)
        and np.isfinite(slope_ci.ci_upper)
        and np.isfinite(l50_ci.ci_lower)
        and np.isfinite(l50_ci.ci_upper)
    ):
        return None
    return slope_ci.ci_lower, slope_ci.ci_upper, l50_ci.ci_lower, l50_ci.ci_upper


def _aggregate_long_context_result(
    *,
    per_task: list[LongContextTaskResult],
    lengths: Sequence[int],
    aggregate_seed: int,
    n_resamples: int,
    confidence_level: float,
) -> LongContextResult:
    """Aggregate per-task curves into the dimension result.

    Steps per ADR-0006 §E:

    1. Pool successes/trials across tasks at each tier → empirical
       curve + Wilson CIs (skip tiers where total trials = 0).
    2. Fit the sigmoid on the pooled rates at the measured tiers.
    3. Bootstrap CI on slope + l50 by resampling tasks (with their
       per-tier rate arrays as a unit). Each resample refits.
    4. On fit failure, the empirical curve still reports;
       ``fit_converged = False`` and the slope/L50 fields are None.
    """
    lengths_list = list(lengths)
    n_tasks = len(per_task)
    n_tiers = len(lengths_list)

    # Step 1: pool empirical curve.
    successes_per_tier = [0] * n_tiers
    trials_per_tier = [0] * n_tiers
    for ptr in per_task:
        # Each per-task result must agree with the dimension's ``lengths``
        # list — assemble defensively in case a future caller passes
        # heterogeneous tier ladders.
        for tier_idx, tier_passes in enumerate(ptr.passes_per_length):
            if tier_idx >= n_tiers:
                break
            successes_per_tier[tier_idx] += sum(tier_passes)
            trials_per_tier[tier_idx] += len(tier_passes)

    measured_indices: list[int] = []
    success_rates: list[float] = []
    success_cis: list[WilsonCI] = []
    for tier_idx in range(n_tiers):
        if trials_per_tier[tier_idx] == 0:
            continue
        measured_indices.append(tier_idx)
        success_rates.append(successes_per_tier[tier_idx] / trials_per_tier[tier_idx])
        success_cis.append(wilson_ci(successes_per_tier[tier_idx], trials_per_tier[tier_idx]))

    if not measured_indices:
        # No data at any tier — every task was too long for every tier.
        return LongContextResult(
            kind="long_context",
            n_tasks=n_tasks,
            lengths=lengths_list,
            success_rates=[],
            success_cis=[],
            measured_length_indices=[],
            slope=None,
            slope_ci_lower=None,
            slope_ci_upper=None,
            l50=None,
            l50_ci_lower=None,
            l50_ci_upper=None,
            fit_converged=False,
            confidence_level=None,
            n_resamples=None,
            per_task=per_task,
            reason="no length tier produced any judged rep (all tasks exceeded every tier)",
        )

    measured_lengths = [lengths_list[i] for i in measured_indices]

    # Step 2: point fit on pooled rates.
    point_fit = _fit_sigmoid(measured_lengths, success_rates)

    # Step 3: bootstrap CI on slope + l50 if we have enough tasks to
    # resample. We need at least 2 tasks AND at least 2 measured tiers
    # for the sigmoid to be identifiable.
    notes_parts: list[str] = []
    reason: str | None = None

    if point_fit is None:
        fit_converged = False
        slope = None
        l50 = None
        slope_ci_lower = slope_ci_upper = None
        l50_ci_lower = l50_ci_upper = None
        reason = (
            "sigmoid fit did not converge — typically a flat curve "
            "(graceful degradation across all tiers) or a curve with no "
            "monotone trend. Empirical curve still reports per ADR-0006 §E."
        )
    elif n_tasks < 2 or len(measured_indices) < 2:
        # Point fit succeeded but bootstrap is undefined — report the
        # point estimate and N/A the CI, parallel to the n_tasks<2 path
        # in `_aggregate_sub_metric`.
        _, slope, l50 = point_fit
        fit_converged = True
        slope_ci_lower = slope_ci_upper = None
        l50_ci_lower = l50_ci_upper = None
        reason = (
            "bootstrap CI on slope/L_50 requires n_tasks >= 2 AND >= 2 "
            "measured tiers; reported point fit only"
        )
    else:
        _, slope, l50 = point_fit
        fit_converged = True

        # Build the per-task rate matrix (n_tasks, n_measured_tiers).
        # NaN at any (task, tier) the task didn't measure — bootstrap
        # closure handles via ``nanmean``.
        rate_matrix = np.full((n_tasks, len(measured_indices)), np.nan, dtype=float)
        for task_idx, ptr in enumerate(per_task):
            for out_idx, tier_idx in enumerate(measured_indices):
                if tier_idx >= len(ptr.rates_per_length):
                    continue
                rate = ptr.rates_per_length[tier_idx]
                if rate is not None:
                    rate_matrix[task_idx, out_idx] = rate

        cis = _bootstrap_curve_cis(
            per_task_rates=rate_matrix,
            lengths=measured_lengths,
            seed=aggregate_seed,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
        )
        if cis is None:
            slope_ci_lower = slope_ci_upper = None
            l50_ci_lower = l50_ci_upper = None
            notes_parts.append(
                "bootstrap CI on slope/L_50 unavailable (too many resamples failed to fit)"
            )
        else:
            slope_ci_lower, slope_ci_upper, l50_ci_lower, l50_ci_upper = cis

    aggregate_notes = "; ".join(notes_parts) if notes_parts else None

    return LongContextResult(
        kind="long_context",
        n_tasks=n_tasks,
        lengths=lengths_list,
        success_rates=success_rates,
        success_cis=success_cis,
        measured_length_indices=measured_indices,
        slope=slope,
        slope_ci_lower=slope_ci_lower,
        slope_ci_upper=slope_ci_upper,
        l50=l50,
        l50_ci_lower=l50_ci_lower,
        l50_ci_upper=l50_ci_upper,
        fit_converged=fit_converged,
        confidence_level=confidence_level if fit_converged else None,
        n_resamples=n_resamples if fit_converged else None,
        per_task=per_task,
        reason=reason,
        notes=aggregate_notes,
    )


async def measure_long_context_degradation(
    *,
    tasks: Sequence[Task],
    agent: Agent,
    rubric_client: BaseModelClient | None,
    reps: int = 10,
    lengths: Sequence[int] = DEFAULT_LONG_CONTEXT_LENGTHS,
    filler_path: str | Path = LONG_CONTEXT_DEFAULT_FILLER_PATH,
    aggregate_seed: int = 0,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> LongContextResult:
    """Per METHODOLOGY §2.4 / ADR-0006 §E: curve + sigmoid + bootstrap CIs.

    For each (task, tier) cell, the metric builds ``reps`` perturbed
    inputs (per-rep seeds extending the long-context base seed via
    ADR-0006 §B's ``rep{idx}`` extension; the tier itself contributes to
    the seed via the ``tool_call_idx`` slot so reps at different tiers
    pull different filler windows). Each perturbed input is run through
    ``agent.arun`` via :func:`asyncio.gather` and judged by the per-task
    :class:`steadfast.judges.base.Judge`. The aggregate empirical curve
    pools successes/trials across tasks at each tier with Wilson 95%
    CIs; the sigmoid fit summarizes the curve with slope and L_50, both
    with bootstrap CIs.

    Unlike :func:`measure_typo_robustness`, this function does not take
    a ``clean_run_results`` argument — the long-context metric reports
    an absolute curve over tier ladder, not a clean-vs-perturbed delta.

    Parameters
    ----------
    tasks:
        Benchmark tasks. Each is measured at every tier in ``lengths``;
        a task that's already too long for a given tier (after the
        delimiter) is skipped for that tier with a per-task ``notes``
        entry.
    agent:
        The :class:`Agent` under measurement.
    rubric_client:
        Optional rubric judge client; ``None`` is fine for
        ``judge="exact_match"`` tasks.
    reps:
        Per (task, tier) repetition count. METHODOLOGY default is N=10.
    lengths:
        The tier ladder. Default is :data:`DEFAULT_LONG_CONTEXT_LENGTHS`
        (4k / 16k / 64k / 128k per METHODOLOGY §2.4).
    filler_path:
        Path to the frozen filler corpus. Defaults to
        ``prompts/longcontext_filler_v1.txt``.
    aggregate_seed:
        Seed for the bootstrap CI on slope and L_50.
    n_resamples, confidence_level:
        Standard bootstrap parameters; defaults from
        :mod:`steadfast.stats.bootstrap`.
    """
    if reps < 1:
        raise ValueError(f"reps must be >= 1; got {reps}")
    if not lengths:
        raise ValueError("lengths must be non-empty")
    lengths_list = list(lengths)
    if any(length <= 0 for length in lengths_list):
        raise ValueError(f"all lengths must be > 0; got {lengths_list}")

    per_task_results: list[LongContextTaskResult] = []

    for task in tasks:
        passes_per_length: list[list[bool]] = []
        previews_per_length: list[list[str]] = []
        rates_per_length: list[float | None] = []
        task_notes: list[str] = []

        judge = build_default_judge(task, rubric_client=rubric_client)

        for tier_idx, target_tokens in enumerate(lengths_list):
            # Per-tier rep seeding: use the ADR-0006 §B per-rep extension
            # and ride the tier index on the ``tool_call_idx`` slot. The
            # _seed.py docstring explicitly anticipates both extensions
            # stacking; semantically the tier identifies a corruption-
            # like axis (which filler window to pull) parallel to the
            # contradiction perturbation's per-tool-call coin.
            perturbed_inputs: list[str] = []
            tier_skipped_reason: str | None = None
            for rep_idx in range(reps):
                rep_seed = derive_seed(
                    task.id,
                    "long_context",
                    rep_idx=rep_idx,
                    tool_call_idx=tier_idx,
                )
                try:
                    p_input = perturb_long_context(
                        task.input,
                        target_tokens=target_tokens,
                        filler_path=filler_path,
                        seed=rep_seed,
                    )
                except ValueError as exc:
                    # Task too long for this tier — skip the whole tier.
                    tier_skipped_reason = (
                        f"tier {target_tokens} skipped: task input exceeds tier ({exc})"
                    )
                    break
                perturbed_inputs.append(p_input)

            if tier_skipped_reason is not None:
                task_notes.append(tier_skipped_reason)
                passes_per_length.append([])
                previews_per_length.append([])
                rates_per_length.append(None)
                continue

            perturbed_tasks = [task.model_copy(update={"input": pi}) for pi in perturbed_inputs]
            raw_results: list[AgentResponse | BaseException] = await asyncio.gather(
                *(agent.arun(t) for t in perturbed_tasks),
                return_exceptions=True,
            )

            tier_passes: list[bool] = []
            tier_previews: list[str] = []
            n_arun_failures = 0
            n_judge_failures = 0
            for rep_idx, raw in enumerate(raw_results):
                if isinstance(raw, BaseException):
                    n_arun_failures += 1
                    _log.warning(
                        "agent.arun failed on long_context rep for task=%s tier=%d rep=%d: %s",
                        task.id,
                        target_tokens,
                        rep_idx,
                        raw,
                    )
                    continue
                try:
                    verdict = await judge.ajudge(task, raw)
                except Exception as exc:
                    # See `_measure_per_task` rationale: catch broader
                    # than JudgeError so rate-limit cascades on the
                    # rubric judge don't crash the long-context tier
                    # loop mid-run.
                    n_judge_failures += 1
                    _log.warning(
                        "judge failed on long_context rep for task=%s tier=%d rep=%d: %s",
                        task.id,
                        target_tokens,
                        rep_idx,
                        exc,
                    )
                    continue
                tier_passes.append(bool(verdict.passed))
                tier_previews.append(perturbed_inputs[rep_idx][:_PERTURBED_PREVIEW_CHARS])

            passes_per_length.append(tier_passes)
            previews_per_length.append(tier_previews)
            rates_per_length.append(sum(tier_passes) / len(tier_passes) if tier_passes else None)

            if n_arun_failures or n_judge_failures:
                task_notes.append(
                    f"tier {target_tokens}: {n_arun_failures} arun fail(s), "
                    f"{n_judge_failures} judge fail(s)"
                )

        per_task_results.append(
            LongContextTaskResult(
                task_id=task.id,
                lengths=lengths_list,
                passes_per_length=passes_per_length,
                rates_per_length=rates_per_length,
                perturbed_input_previews_per_length=previews_per_length,
                seed=derive_seed(task.id, "long_context"),
                notes="; ".join(task_notes) if task_notes else None,
            )
        )

    return _aggregate_long_context_result(
        per_task=per_task_results,
        lengths=lengths_list,
        aggregate_seed=aggregate_seed,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )


# ---------------------------------------------------------------------------
# Wrapper that bundles all sub-metrics into a single dimension result
# ---------------------------------------------------------------------------


async def measure_robustness(
    *,
    model: str,
    tasks: Sequence[Task],
    clean_run_results: Sequence[RunResult],
    agent: Agent,
    rubric_client: BaseModelClient | None,
    kinds: Iterable[str],
    distractor_banks: dict[str, DistractorBank] | None = None,
    long_context_filler_path: str | Path = LONG_CONTEXT_DEFAULT_FILLER_PATH,
    long_context_lengths: Sequence[int] = DEFAULT_LONG_CONTEXT_LENGTHS,
    reps: int = 10,
    aggregate_seed: int = 0,
) -> RobustnessDimension:
    """Run the requested robustness sub-metrics and bundle into a dimension.

    ``kinds`` is the subset of :data:`SUPPORTED_KINDS` to measure
    (typo / distractor / contradiction / long_context). Unknown kinds
    raise :class:`ValueError`. Contradiction and long_context do not
    consume ``clean_run_results`` (contradiction is a 3-way categorical
    per ADR-0006 §D; long_context is an absolute curve per ADR-0006 §E);
    the dispatch threads only the inputs each kind needs.
    """
    requested = frozenset(kinds)
    unknown = requested - SUPPORTED_KINDS
    if unknown:
        raise ValueError(
            f"unknown robustness kind(s): {sorted(unknown)} — supported: {sorted(SUPPORTED_KINDS)}"
        )

    sub_metrics: dict[
        str,
        RobustnessSubMetricResult | ContradictionResult | LongContextResult,
    ] = {}

    delta_runners: list[tuple[str, Awaitable[RobustnessSubMetricResult]]] = []
    if "typo" in requested:
        delta_runners.append(
            (
                "typo",
                measure_typo_robustness(
                    tasks=tasks,
                    clean_run_results=clean_run_results,
                    agent=agent,
                    rubric_client=rubric_client,
                    reps=reps,
                    aggregate_seed=aggregate_seed,
                ),
            )
        )
    if "distractor" in requested:
        delta_runners.append(
            (
                "distractor",
                measure_distractor_robustness(
                    tasks=tasks,
                    clean_run_results=clean_run_results,
                    agent=agent,
                    rubric_client=rubric_client,
                    distractor_banks=distractor_banks or {},
                    reps=reps,
                    aggregate_seed=aggregate_seed,
                ),
            )
        )

    # Run sub-metrics serially. Parallel execution would multiply the
    # in-flight perturbed-rep fanout against the model client's semaphore;
    # the sub-metrics each saturate the semaphore on their own, so
    # interleaving them just adds contention without a wall-clock win.
    for kind, awaitable in delta_runners:
        sub_metrics[kind] = await awaitable

    if "contradiction" in requested:
        sub_metrics["contradiction"] = await measure_contradiction_handling(
            tasks=tasks,
            agent=agent,
            reps=reps,
        )

    if "long_context" in requested:
        sub_metrics["long_context"] = await measure_long_context_degradation(
            tasks=tasks,
            agent=agent,
            rubric_client=rubric_client,
            reps=reps,
            lengths=long_context_lengths,
            filler_path=long_context_filler_path,
            aggregate_seed=aggregate_seed,
        )

    return RobustnessDimension(
        model=model,
        n_tasks=len(tasks),
        sub_metrics=sub_metrics,
    )


__all__ = [
    "DEFAULT_LONG_CONTEXT_LENGTHS",
    "SUPPORTED_KINDS",
    "ContradictionLabel",
    "ContradictionResult",
    "ContradictionTaskResult",
    "LongContextResult",
    "LongContextTaskResult",
    "RobustnessDimension",
    "RobustnessKind",
    "RobustnessSubMetricResult",
    "RobustnessTaskResult",
    "classify_contradiction_response",
    "measure_contradiction_handling",
    "measure_distractor_robustness",
    "measure_long_context_degradation",
    "measure_robustness",
    "measure_typo_robustness",
]
