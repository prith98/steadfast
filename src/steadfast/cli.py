"""Typer CLI surface.

Two invocation shapes:

* **Single task / single model** (Tuesday surface, preserved for inner-loop
  development): ``steadfast bench --task path/to/task.json --model claude-opus-4-7``.
* **Benchmark / multiple models** (Friday surface, ADR-0005 §G): ``steadfast
  bench --benchmark customer_support_pilot --models claude-opus-4-7,gpt-5.2,
  gemini-2.5-pro --metrics consistency,calibration``. The CLI iterates models
  sequentially (parallelism is per-rep within a model, bounded by the
  per-client semaphore — adding cross-model parallelism would multiply API
  spend without improving statistics).

``--metrics`` resolves to per-(model) calibration measurement (Brier / ECE /
refusal calibration / overconfidence rate per ADR-0005 §D-E) and
per-(model, task) output-consistency measurement (K=5 paraphrases per
ADR-0004 §E). The HTML report aggregates across models for side-by-side
comparison.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Final

import typer

from steadfast import __version__
from steadfast.agent import Agent, SimplePromptingAgent, Task
from steadfast.judges import judge_run_result
from steadfast.metrics.calibration import (
    CalibrationDimension,
    CalibrationRep,
    measure_calibration,
    reps_from_run_results,
)
from steadfast.metrics.consistency import (
    OutputConsistencyResult,
    measure_output_consistency,
)
from steadfast.metrics.robustness import (
    SUPPORTED_KINDS as _SUPPORTED_ROBUSTNESS_KINDS,
)
from steadfast.metrics.robustness import (
    RobustnessDimension,
    measure_robustness,
)
from steadfast.metrics.safety import (
    SafetyDimension,
    load_safety_bank,
    measure_safety,
)
from steadfast.models.base import BaseModelClient
from steadfast.models.openai_client import OpenAIClient
from steadfast.models.pricing import provider_for_model
from steadfast.perturbations.confidence import load_confidence_suffix_v1
from steadfast.perturbations.distractor import (
    DistractorBank,
    load_distractor_bank,
)
from steadfast.reporting.html import write_html_report
from steadfast.runner import RunResult, run_task
from steadfast.tracing import benchmark_span, configure_tracing
from steadfast.tracing.exporters import ExporterKind

# Metric dimensions that ``--metrics`` accepts. Robustness sub-metrics
# (typo, distractor, contradiction, long_context) are selected via the
# ``--robustness-types`` flag and validated against
# ``_SUPPORTED_ROBUSTNESS_KINDS`` from ``metrics.robustness``. Safety is
# the week-3 addition per ADR-0007 and is special — it requires
# ``--benchmark safety`` (the only benchmark that resolves to a
# :class:`SafetyBank` rather than per-task JSON files).
_VALID_METRICS: Final[frozenset[str]] = frozenset(
    {"consistency", "calibration", "robustness", "safety"}
)

# Reserved benchmark name that resolves to the safety case bank instead
# of the standard ``benchmarks/<name>/*.json`` resolution. ADR-0007 §G's
# audit gate enforced by :func:`load_safety_bank`.
_SAFETY_BENCHMARK_NAME: Final[str] = "safety"

# Benchmarks ship under ``benchmarks/`` at the repo root. Resolution rule
# per ADR-0005 §F: ``customer_support_pilot`` → every
# ``benchmarks/customer_support/pilot_*.json`` file. ``customer_support`` →
# all task files in the directory. The mapping is intentionally simple so
# the resolution logic is easy to reason about; v0.2 may add a
# manifest-driven resolver.
_BENCHMARK_BASE = Path(__file__).resolve().parents[2] / "benchmarks"


app = typer.Typer(
    name="steadfast",
    help="Reliability benchmarking for AI agents.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"steadfast {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the steadfast version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Steadfast — reliability benchmarking for AI agents."""


def _build_openai_client() -> OpenAIClient:
    """Construct an :class:`OpenAIClient` with rate-limit-tier env knobs applied.

    ``STEADFAST_OPENAI_MAX_CONCURRENT`` and ``STEADFAST_OPENAI_MAX_RETRIES``
    override the constructor defaults (5 / 5). Lower these for accounts on a
    low rate-limit tier — gpt-5.2 on the free tier is 3 RPM, which collides
    with the default 5-way concurrent fanout (paraphrase generator +
    validator + pairwise rubric + outcome judge). Setting
    ``STEADFAST_OPENAI_MAX_CONCURRENT=1`` and ``STEADFAST_OPENAI_MAX_RETRIES=10``
    makes the run finish on the free tier at the cost of wall-clock time;
    the existing tenacity exponential backoff (capped at 30s) covers the
    20s rate-limit window.
    """
    max_concurrent = int(os.environ.get("STEADFAST_OPENAI_MAX_CONCURRENT", "5"))
    max_retries = int(os.environ.get("STEADFAST_OPENAI_MAX_RETRIES", "5"))
    return OpenAIClient(max_concurrent=max_concurrent, max_retries=max_retries)


def _build_client(provider: str) -> BaseModelClient:
    """Instantiate the right :class:`BaseModelClient` for a provider name."""
    if provider == "anthropic":
        from steadfast.models.anthropic_client import AnthropicClient

        return AnthropicClient()
    if provider == "openai":
        return _build_openai_client()
    if provider == "google":
        from steadfast.models.google_client import GoogleClient

        return GoogleClient()
    raise typer.BadParameter(f"unknown provider: {provider!r}")


def _build_rubric_client(target_provider: str, target_client: BaseModelClient) -> BaseModelClient:
    """Return the OpenAI client to use for the rubric judge.

    Per ADR-0001, the v0.1 rubric judge is locked to ``gpt-5.2`` on OpenAI.
    If the target model is also on OpenAI we reuse the existing client to
    share its asyncio semaphore (avoids fan-in of N target reps + N judge
    calls each holding their own slot).
    """
    # Best-effort: reuse the target client when it's the OpenAIClient on
    # an OpenAI target model (so the asyncio semaphore is shared).
    # Otherwise build a fresh instance — consistency / rubric judge
    # type-contracts expect an OpenAIClient regardless.
    if target_provider == "openai" and isinstance(target_client, OpenAIClient):
        return target_client
    return _build_openai_client()


def resolve_benchmark(name: str) -> list[Path]:
    """Resolve a benchmark name to a sorted list of task JSON paths.

    ``customer_support_pilot`` → ``benchmarks/customer_support/pilot_*.json``.
    Other ``<domain>_<suffix>`` names are reserved; v0.1 only ships the
    pilot. Bare domain names (``customer_support``) resolve to every
    ``*.json`` task in the domain directory.

    Raises :class:`typer.BadParameter` if no tasks match — silent empty
    resolution would be worse than a loud failure.
    """
    if not _BENCHMARK_BASE.is_dir():
        raise typer.BadParameter(
            f"benchmarks directory not found at {_BENCHMARK_BASE} — are you running "
            "from a Steadfast checkout?"
        )

    pilot_suffix = "_pilot"
    if name.endswith(pilot_suffix):
        domain = name[: -len(pilot_suffix)]
        domain_dir = _BENCHMARK_BASE / domain
        glob = "pilot_*.json"
    else:
        domain_dir = _BENCHMARK_BASE / name
        glob = "*.json"

    if not domain_dir.is_dir():
        raise typer.BadParameter(
            f"benchmark {name!r} did not resolve to a directory under {_BENCHMARK_BASE}"
        )
    paths = sorted(domain_dir.glob(glob))
    if not paths:
        raise typer.BadParameter(
            f"benchmark {name!r} resolved to {domain_dir} but no task files matched {glob!r}"
        )
    return paths


def parse_metrics(spec: str | None) -> frozenset[str]:
    """Parse the comma-separated ``--metrics`` argument into a set.

    Empty / None → empty set (no metric dispatch). Unknown metric names
    raise :class:`typer.BadParameter` with the list of valid dimensions
    so users discover the surface.
    """
    if not spec:
        return frozenset()
    requested = {part.strip() for part in spec.split(",") if part.strip()}
    invalid = requested - _VALID_METRICS
    if invalid:
        raise typer.BadParameter(
            f"unknown metric(s): {sorted(invalid)} — valid: {sorted(_VALID_METRICS)}"
        )
    return frozenset(requested)


def parse_robustness_types(spec: str | None) -> frozenset[str]:
    """Parse the comma-separated ``--robustness-types`` argument.

    Empty / None → all supported kinds (the methodology-default coverage
    when ``--metrics robustness`` is on its own). Unknown kinds raise
    :class:`typer.BadParameter` with the supported set. The v0.1
    supported set is typo / distractor / contradiction / long_context
    per METHODOLOGY §2.
    """
    if not spec:
        return frozenset(_SUPPORTED_ROBUSTNESS_KINDS)
    requested = {part.strip() for part in spec.split(",") if part.strip()}
    invalid = requested - _SUPPORTED_ROBUSTNESS_KINDS
    if invalid:
        raise typer.BadParameter(
            f"unknown robustness type(s): {sorted(invalid)} — "
            f"supported: {sorted(_SUPPORTED_ROBUSTNESS_KINDS)}"
        )
    return frozenset(requested)


def parse_models(spec: str) -> list[str]:
    """Parse the comma-separated ``--models`` argument into an ordered list.

    Order matters: the CLI iterates models in the given order, and per-model
    output directories are named after the model ID, so users get a
    predictable layout.
    """
    models = [m.strip() for m in spec.split(",") if m.strip()]
    if not models:
        raise typer.BadParameter("--models must contain at least one model ID")
    # Validate provider lookup before we start spending money.
    for m in models:
        try:
            provider_for_model(m)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    return models


def _apply_confidence_suffix(tasks: Iterable[Task], suffix: str) -> list[Task]:
    """Inject the frozen confidence-elicitation suffix into every task.

    Per ADR-0002 §A.1, the harness sets ``Task.confidence_suffix``; the
    agent reads it. Tasks are frozen Pydantic models so we copy with the
    update applied rather than mutate.
    """
    return [t.model_copy(update={"confidence_suffix": suffix}) for t in tasks]


@app.command()
def bench(
    task: Annotated[
        Path | None,
        typer.Option("--task", help="Path to a single task JSON file (single-task surface)."),
    ] = None,
    benchmark: Annotated[
        str | None,
        typer.Option(
            "--benchmark",
            help=(
                "Curated benchmark suite (e.g. 'customer_support_pilot'). "
                "Mutually exclusive with --task."
            ),
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Single target model identifier."),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            help=("Comma-separated list of target model IDs. Mutually exclusive with --model."),
        ),
    ] = None,
    reps: Annotated[
        int,
        typer.Option("--reps", help="Repetitions per task (methodology default N=10)."),
    ] = 10,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Directory for results JSON and SQLite checkpoint.",
        ),
    ] = Path("results"),
    agent: Annotated[
        str | None,
        typer.Option(
            "--agent",
            help="Import path to a custom Agent subclass (not yet supported).",
        ),
    ] = None,
    metrics: Annotated[
        str | None,
        typer.Option(
            "--metrics",
            help=(
                "Comma-separated metric dimensions to compute after the run. "
                "Valid: consistency, calibration, robustness."
            ),
        ),
    ] = None,
    robustness_types: Annotated[
        str | None,
        typer.Option(
            "--robustness-types",
            help=(
                "Comma-separated robustness sub-metrics. Default: all supported. "
                "Valid (v0.1): typo, distractor, contradiction, long_context."
            ),
        ),
    ] = None,
    exporter: Annotated[
        str,
        typer.Option(
            "--exporter",
            help=(
                "OpenTelemetry exporter — 'console' (default; spans → stdout), "
                "'otlp' (HTTP to Phoenix at http://localhost:6006/v1/traces by "
                "default; override with OTEL_EXPORTER_OTLP_ENDPOINT), or 'none' "
                "(no spans exported)."
            ),
        ),
    ] = "console",
) -> None:
    """Run the reliability benchmark against one or more wrapped Agents."""
    # ---- argument validation ----
    if (task is None) == (benchmark is None):
        typer.echo(
            "error: exactly one of --task / --benchmark is required.",
            err=True,
        )
        raise typer.Exit(2)
    if (model is None) == (models is None):
        typer.echo(
            "error: exactly one of --model / --models is required.",
            err=True,
        )
        raise typer.Exit(2)
    if agent is not None:
        typer.echo(
            "error: --agent is not yet supported (built-in SimplePromptingAgent in use).",
            err=True,
        )
        raise typer.Exit(2)
    if exporter not in {"console", "otlp", "none"}:
        typer.echo(
            f"error: --exporter={exporter!r} must be one of: console, otlp, none.",
            err=True,
        )
        raise typer.Exit(2)
    exporter_kind: ExporterKind = exporter  # type: ignore[assignment]

    requested_metrics = parse_metrics(metrics)
    requested_robustness_kinds = (
        parse_robustness_types(robustness_types)
        if "robustness" in requested_metrics
        else frozenset()
    )
    target_models: list[str] = [model] if model is not None else parse_models(models or "")

    # ---- safety-vs-everything-else coherence check ----
    # Safety needs the SafetyBank rather than per-task JSONs, so it
    # can't be mixed with other --benchmark surfaces in the v0.1 CLI.
    # Surfacing both directions as parser errors avoids silent misuse:
    # someone who specifies --metrics safety against customer_support_pilot
    # would get a working run that didn't actually measure safety, and
    # someone who specifies --benchmark safety with --metrics calibration
    # would get a confused mid-run failure when the safety case Tasks
    # don't carry the expected ground-truth shape.
    safety_metric_requested = "safety" in requested_metrics
    safety_benchmark_requested = benchmark == _SAFETY_BENCHMARK_NAME
    if safety_metric_requested != safety_benchmark_requested:
        typer.echo(
            "error: --metrics safety requires --benchmark safety, and "
            "--benchmark safety requires --metrics safety (no other --metrics "
            "are supported on the safety bank in v0.1 per ADR-0007).",
            err=True,
        )
        raise typer.Exit(2)
    if safety_metric_requested and len(requested_metrics) > 1:
        typer.echo(
            "error: --metrics safety is mutually exclusive with other metric "
            "dimensions in v0.1 (the safety bank doesn't carry the ground "
            "truth needed by consistency / calibration / robustness).",
            err=True,
        )
        raise typer.Exit(2)
    if safety_benchmark_requested and task is not None:
        typer.echo(
            "error: --benchmark safety cannot be combined with --task.",
            err=True,
        )
        raise typer.Exit(2)

    output.mkdir(parents=True, exist_ok=True)

    # ---- safety branch — its own dispatch (no run_task, no judge_run_result) ----
    if safety_benchmark_requested:
        _run_safety_bench(
            target_models=target_models,
            output=output,
            exporter_kind=exporter_kind,
            requested_metrics=requested_metrics,
        )
        return

    # ---- task resolution (normal benchmarks) ----
    if task is not None:
        if not task.is_file():
            typer.echo(f"error: --task {task} is not a file.", err=True)
            raise typer.Exit(2)
        task_paths = [task]
        benchmark_name = task.stem
    else:
        assert benchmark is not None  # narrowed by the (task is None) == (benchmark is None) check
        task_paths = resolve_benchmark(benchmark)
        benchmark_name = benchmark
    tasks: list[Task] = [Task.model_validate_json(p.read_text()) for p in task_paths]
    if "calibration" in requested_metrics:
        # Inject the frozen suffix; the agent reads it.
        tasks = _apply_confidence_suffix(tasks, load_confidence_suffix_v1())

    typer.echo(
        f"running benchmark {benchmark_name!r} with {len(tasks)} task(s) on "
        f"{len(target_models)} model(s); reps={reps} metrics="
        f"{sorted(requested_metrics) or '[none]'} exporter={exporter}...",
        err=True,
    )

    provider_obj = configure_tracing(exporter=exporter_kind)
    try:
        for target_model in target_models:
            try:
                provider_name = provider_for_model(target_model)
            except ValueError as e:
                typer.echo(f"error: {e}", err=True)
                raise typer.Exit(2) from e
            target_client = _build_client(provider_name)
            sf_agent = SimplePromptingAgent(client=target_client, model=target_model)

            model_dir = output / _slug(target_model)
            model_dir.mkdir(parents=True, exist_ok=True)

            run_results = asyncio.run(
                _run_one_model(
                    benchmark_name=benchmark_name,
                    target_model=target_model,
                    target_provider=provider_name,
                    target_client=target_client,
                    sf_agent=sf_agent,
                    tasks=tasks,
                    reps=reps,
                    model_dir=model_dir,
                    requested_metrics=requested_metrics,
                    requested_robustness_kinds=requested_robustness_kinds,
                )
            )

            _summarize_model_run(target_model, run_results)
    finally:
        provider_obj.force_flush()
        provider_obj.shutdown()

    # ---- HTML report aggregating across all models ----
    if requested_metrics:
        report_path = output / "report.html"
        write_html_report(
            output_dir=output,
            benchmark_name=benchmark_name,
            target_models=target_models,
            requested_metrics=requested_metrics,
            report_path=report_path,
        )
        typer.echo(f"wrote {report_path}", err=True)


async def _run_one_model(
    *,
    benchmark_name: str,
    target_model: str,
    target_provider: str,
    target_client: BaseModelClient,
    sf_agent: Agent,
    tasks: list[Task],
    reps: int,
    model_dir: Path,
    requested_metrics: frozenset[str],
    requested_robustness_kinds: frozenset[str],
) -> list[RunResult]:
    """Execute every task for a single model, judge, and write per-task results.

    After all tasks complete the function dispatches the requested metrics
    (calibration is per-model, computed over the pooled reps; consistency
    is per-task, written separately; robustness is per-model with per-task
    detail nested in a single robustness.json).
    """
    rubric_client = _build_rubric_client(target_provider, target_client)
    run_results: list[RunResult] = []
    calibration_reps: list[CalibrationRep] = []

    with benchmark_span(name=f"{benchmark_name}/{target_model}", package_version=__version__):
        for task in tasks:
            checkpoint_path = model_dir / f"{task.id}.sqlite"
            result = await run_task(
                agent=sf_agent,
                task=task,
                reps=reps,
                model=target_model,
                checkpoint_path=checkpoint_path,
            )
            await judge_run_result(result, rubric_client=rubric_client)
            (model_dir / f"{task.id}.json").write_text(result.model_dump_json(indent=2))
            run_results.append(result)
            if "calibration" in requested_metrics:
                calibration_reps.extend(reps_from_run_results(result.reps, task))

        if "consistency" in requested_metrics:
            for task, result in zip(tasks, run_results, strict=True):
                # Consistency uses K=5 paraphrases of the *bare* task input
                # (without the confidence suffix) — paraphrasing the
                # elicitation tail itself would conflate output drift with
                # confidence-prompt drift. Strip the suffix on a copy.
                bare_task = task.model_copy(update={"confidence_suffix": None})
                if not isinstance(rubric_client, OpenAIClient):
                    raise RuntimeError(
                        "consistency requires an OpenAI infrastructure client per ADR-0001"
                    )
                consistency = await measure_output_consistency(
                    task=bare_task,
                    agent=sf_agent,
                    infra_client=rubric_client,
                )
                _write_consistency(model_dir, task.id, consistency)
                # mypy: keep the loop body type-narrow.
                _ = result

        if "calibration" in requested_metrics and calibration_reps:
            calibration = measure_calibration(
                calibration_reps,
                model=target_model,
                n_tasks=len(tasks),
            )
            _write_calibration(model_dir, calibration)

        if "robustness" in requested_metrics:
            distractor_banks: dict[str, DistractorBank] = {}
            if "distractor" in requested_robustness_kinds:
                distractor_banks = _load_distractor_banks_for(tasks)
            robustness = await measure_robustness(
                model=target_model,
                tasks=tasks,
                clean_run_results=run_results,
                agent=sf_agent,
                rubric_client=rubric_client,
                kinds=requested_robustness_kinds,
                distractor_banks=distractor_banks,
                reps=reps,
            )
            _write_robustness(model_dir, robustness)

    return run_results


def _summarize_model_run(model: str, results: list[RunResult]) -> None:
    """Print a one-line summary of a model's run for stdout consumption."""
    statuses: Counter[str] = Counter()
    cost = Decimal("0")
    judged = 0
    passed = 0
    for r in results:
        statuses.update(rec.status.value for rec in r.reps)
        for rec in r.reps:
            if rec.response is not None and rec.response.cost_usd is not None:
                cost += Decimal(str(rec.response.cost_usd))
            if rec.verdict is not None:
                judged += 1
                if rec.verdict.passed:
                    passed += 1
    typer.echo(
        f"model={model} tasks={len(results)} statuses={dict(statuses)} "
        f"cost_usd={cost} verdicts={passed}/{judged}"
    )


def _slug(model: str) -> str:
    """Filesystem-safe slug for a model identifier.

    Replaces characters that some shells or filesystems treat specially
    (slash on POSIX, backslash on Windows, colon on macOS-classic-mode).
    Model identifiers in v0.1 don't actually contain any of those, but
    being defensive here costs nothing.
    """
    safe = []
    for ch in model:
        safe.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(safe)


def _write_consistency(model_dir: Path, task_id: str, result: OutputConsistencyResult) -> None:
    """Write a per-task consistency JSON to ``<model_dir>/consistency_<task_id>.json``."""
    path = model_dir / f"consistency_{task_id}.json"
    path.write_text(result.model_dump_json(indent=2))


def _write_calibration(model_dir: Path, result: CalibrationDimension) -> None:
    """Write the per-model calibration JSON to ``<model_dir>/calibration.json``."""
    path = model_dir / "calibration.json"
    path.write_text(result.model_dump_json(indent=2))


def _write_robustness(model_dir: Path, result: RobustnessDimension) -> None:
    """Write the per-model robustness JSON to ``<model_dir>/robustness.json``."""
    path = model_dir / "robustness.json"
    path.write_text(result.model_dump_json(indent=2))


def _write_safety(model_dir: Path, result: SafetyDimension) -> None:
    """Write the per-model safety JSON to ``<model_dir>/safety.json``."""
    path = model_dir / "safety.json"
    path.write_text(result.model_dump_json(indent=2))


def _run_safety_bench(
    *,
    target_models: list[str],
    output: Path,
    exporter_kind: ExporterKind,
    requested_metrics: frozenset[str],
) -> None:
    """Safety dispatch — its own bench loop, separate from the normal flow.

    The safety dimension loads a curated :class:`SafetyBank` rather
    than per-task JSON files (one bank file holds both trap and benign
    cases), runs the agent once per case (N=1 per ADR-0007 §E), and
    invokes :class:`~steadfast.judges.safety.SafetyJudge` directly on
    each trap response. No ``run_task`` checkpointing and no
    ``judge_run_result`` dispatch — :func:`measure_safety` owns the
    loop end-to-end.

    The audit gate (ADR-0007 §G) is enforced by
    :func:`load_safety_bank`; a ``review_status="draft"`` bank raises
    ``ValueError`` and the CLI surfaces the error verbatim so the
    operator sees the audit-pending message rather than a partial run.
    """
    bank_path = _BENCHMARK_BASE / _SAFETY_BENCHMARK_NAME / "cases_v1.json"
    try:
        bank = load_safety_bank(bank_path)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except ValueError as exc:
        # The audit-gate error per ADR-0007 §G — surface the message
        # verbatim so the operator sees the path forward without
        # having to dig through the traceback.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    typer.echo(
        f"running safety bank version={bank.version!r} "
        f"({len(bank.traps)} trap + {len(bank.benigns)} benign cases) on "
        f"{len(target_models)} model(s); metrics="
        f"{sorted(requested_metrics)}...",
        err=True,
    )

    # Apply the confidence suffix to every safety case so the agent's
    # parser populates ``response.refused`` from the REFUSE token on
    # the ANSWER line. Refusal F1 (METHODOLOGY §4.2) depends on this.
    confidence_suffix = load_confidence_suffix_v1()

    provider_obj = configure_tracing(exporter=exporter_kind)
    try:
        for target_model in target_models:
            try:
                provider_name = provider_for_model(target_model)
            except ValueError as e:
                typer.echo(f"error: {e}", err=True)
                raise typer.Exit(2) from e
            target_client = _build_client(provider_name)
            sf_agent = SimplePromptingAgent(client=target_client, model=target_model)
            safety_judge_client = _build_rubric_client(provider_name, target_client)

            model_dir = output / _slug(target_model)
            model_dir.mkdir(parents=True, exist_ok=True)

            with benchmark_span(
                name=f"{_SAFETY_BENCHMARK_NAME}/{target_model}",
                package_version=__version__,
            ):
                result = asyncio.run(
                    measure_safety(
                        model=target_model,
                        bank=bank,
                        agent=sf_agent,
                        safety_judge_client=safety_judge_client,
                        confidence_suffix=confidence_suffix,
                    )
                )
            _write_safety(model_dir, result)
            _summarize_safety_run(target_model, result)
    finally:
        provider_obj.force_flush()
        provider_obj.shutdown()

    report_path = output / "report.html"
    write_html_report(
        output_dir=output,
        benchmark_name=_SAFETY_BENCHMARK_NAME,
        target_models=target_models,
        requested_metrics=requested_metrics,
        report_path=report_path,
    )
    typer.echo(f"wrote {report_path}", err=True)


def _summarize_safety_run(model: str, result: SafetyDimension) -> None:
    """One-line per-model summary for the safety dispatch."""
    cat_rate = f"{result.catastrophic.rate:.3f}" if result.catastrophic.rate is not None else "N/A"
    f1 = f"{result.refusal.f1:.3f}" if result.refusal.f1 is not None else "N/A"
    typer.echo(
        f"model={model} catastrophic_rate={cat_rate} "
        f"({result.catastrophic.n_harmful}/{result.catastrophic.n_traps_judged}) "
        f"refusal_f1={f1} "
        f"(tp={result.refusal.tp} fp={result.refusal.fp} "
        f"fn={result.refusal.fn} tn={result.refusal.tn}) "
        f"agent_failures={result.n_agent_failures} "
        f"judge_failures={result.n_judge_failures}"
    )


def _load_distractor_banks_for(tasks: Iterable[Task]) -> dict[str, DistractorBank]:
    """Load the distractor bank for each distinct domain in ``tasks``.

    Missing-bank tasks are not silently ignored at this layer — the metric
    layer logs and skips them, but we surface the missing-bank list at
    CLI time too so the operator sees it before spending money on a run
    that can't produce distractor signal for some tasks.
    """
    domains = {t.domain for t in tasks}
    banks: dict[str, DistractorBank] = {}
    for domain in sorted(domains):
        bank_path = _BENCHMARK_BASE / domain / "distractors_v1.json"
        try:
            banks[domain] = load_distractor_bank(bank_path)
        except FileNotFoundError:
            typer.echo(
                f"warning: no distractor bank for domain={domain!r} at {bank_path}; "
                "distractor robustness will be skipped for tasks in this domain. "
                "Run scripts/generate_distractor_bank.py to create one.",
                err=True,
            )
    return banks


def main() -> None:
    """Console-script entry point declared in ``pyproject.toml``."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    app()
