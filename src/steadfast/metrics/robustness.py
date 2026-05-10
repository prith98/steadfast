"""Robustness dimension — typo and distractor sub-metrics (week 2 / Tuesday).

Per ``docs/METHODOLOGY.md`` §2 and ADR-0006, each robustness sub-metric
reports a **success-rate delta** (perturbed minus clean) on the same
task set, with a 95% paired-bootstrap CI on the delta itself per
ADR-0006 §F. Contradiction (week 2 / Wednesday) and long-context (week 2
/ Thursday) land in follow-up commits per ``docs/WEEK_2.md``.

The clean arm reuses the existing per-task ``RunResult`` produced by the
main bench loop — clean reps were already executed and judged before
this metric runs, so we don't re-spend tokens. The perturbed arm runs
N=10 distinct perturbations through ``agent.arun`` via
:func:`asyncio.gather` (the same precedent
:func:`steadfast.metrics.consistency.measure_output_consistency` set for
multi-input metric paths) and judges each response with the per-task
:class:`steadfast.judges.base.Judge` selected by ``Task.judge``.

The CI is computed at the dimension level, across tasks. For
``n_tasks < 2`` (single-task ``--task`` invocation) the paired bootstrap
is undefined; the metric reports the per-task point estimate and N/A's
the CI with a populated ``reason`` field — the same N/A pattern
ADR-0004 §G uses for trajectory consistency on toolless agents.

References:

* ADR-0006 §B (seeds), §C (distractor bank), §F (paired bootstrap).
* METHODOLOGY §2 (robustness contract).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from steadfast.agent import Agent, AgentResponse, Task
from steadfast.judges import build_default_judge
from steadfast.judges.base import JudgeError
from steadfast.models.base import BaseModelClient
from steadfast.perturbations import derive_seed
from steadfast.perturbations.distractor import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DistractorBank,
    perturb_distractor,
    pick_distractor,
)
from steadfast.perturbations.typo import (
    DEFAULT_MAX_WORD_CORRUPTION,
    DEFAULT_RATE,
    perturb_typo,
)
from steadfast.runner import RepStatus, RunResult
from steadfast.stats.paired_bootstrap import paired_bootstrap_ci

_log = logging.getLogger(__name__)

# Robustness sub-metric kinds shipped this week. Wednesday adds
# "contradiction"; Thursday adds "long_context".
RobustnessKind = Literal["typo", "distractor"]
SUPPORTED_KINDS: Final[frozenset[str]] = frozenset({"typo", "distractor"})

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
    kind: RobustnessKind
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

    kind: RobustnessKind
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


class RobustnessDimension(BaseModel):
    """Combined robustness result for one (model, run) configuration.

    Mirrors :class:`steadfast.metrics.calibration.CalibrationDimension`
    in shape — the HTML report consumes a single
    ``robustness.json`` per model with all sub-metrics nested.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    n_tasks: int
    sub_metrics: dict[str, RobustnessSubMetricResult] = Field(default_factory=dict)


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
    kind: RobustnessKind,
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
        except JudgeError as exc:
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
    kind: RobustnessKind,
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


async def measure_robustness(
    *,
    model: str,
    tasks: Sequence[Task],
    clean_run_results: Sequence[RunResult],
    agent: Agent,
    rubric_client: BaseModelClient | None,
    kinds: Iterable[str],
    distractor_banks: dict[str, DistractorBank] | None = None,
    reps: int = 10,
    aggregate_seed: int = 0,
) -> RobustnessDimension:
    """Run the requested robustness sub-metrics and bundle into a dimension.

    ``kinds`` is the subset of ``{"typo", "distractor"}`` to measure
    (week 2 / Tuesday surface — contradiction and long_context land
    later in the week). Unknown kinds raise :class:`ValueError`.
    """
    requested = frozenset(kinds)
    unknown = requested - SUPPORTED_KINDS
    if unknown:
        raise ValueError(
            f"unknown robustness kind(s): {sorted(unknown)} — supported: {sorted(SUPPORTED_KINDS)}"
        )

    sub_metrics: dict[str, RobustnessSubMetricResult] = {}

    runners: list[tuple[str, Awaitable[RobustnessSubMetricResult]]] = []
    if "typo" in requested:
        runners.append(
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
        runners.append(
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
    for kind, awaitable in runners:
        sub_metrics[kind] = await awaitable

    return RobustnessDimension(
        model=model,
        n_tasks=len(tasks),
        sub_metrics=sub_metrics,
    )


__all__ = [
    "SUPPORTED_KINDS",
    "RobustnessDimension",
    "RobustnessKind",
    "RobustnessSubMetricResult",
    "RobustnessTaskResult",
    "measure_distractor_robustness",
    "measure_robustness",
    "measure_typo_robustness",
]
