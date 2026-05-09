"""Typer CLI surface.

``steadfast bench`` configures OpenTelemetry tracing per ``--exporter``
(``console`` | ``otlp`` | ``none``), wraps the run in a ``benchmark``
span, executes the runner, dispatches the appropriate
:class:`~steadfast.judges.base.Judge`, and writes the verdict-augmented
:class:`~steadfast.runner.RunResult` JSON.

``--metrics`` is accepted but ignored — the metric dimensions are
implemented in :mod:`steadfast.metrics`.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer

from steadfast import __version__
from steadfast.agent import Agent, SimplePromptingAgent, Task
from steadfast.judges import judge_run_result
from steadfast.models.base import BaseModelClient
from steadfast.models.pricing import provider_for_model
from steadfast.runner import RunResult, run_task
from steadfast.tracing import benchmark_span, configure_tracing
from steadfast.tracing.exporters import ExporterKind

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


def _build_client(provider: str) -> BaseModelClient:
    """Instantiate the right :class:`BaseModelClient` for a provider name."""
    if provider == "anthropic":
        from steadfast.models.anthropic_client import AnthropicClient

        return AnthropicClient()
    if provider == "openai":
        from steadfast.models.openai_client import OpenAIClient

        return OpenAIClient()
    if provider == "google":
        from steadfast.models.google_client import GoogleClient

        return GoogleClient()
    raise typer.BadParameter(f"unknown provider: {provider!r}")


def _build_rubric_client(target_provider: str, target_client: BaseModelClient) -> BaseModelClient:
    """Return the OpenAI client to use for the rubric judge.

    Per ADR-0001, the v0.1 rubric judge is locked to ``gpt-5.2`` on OpenAI.
    If the target model is also on OpenAI we reuse the existing client to
    share its asyncio semaphore (avoids fan-in of N target reps + N judge
    calls each holding their own slot). For other providers we lazily
    construct a fresh OpenAI client.
    """
    if target_provider == "openai":
        return target_client
    from steadfast.models.openai_client import OpenAIClient

    return OpenAIClient()


@app.command()
def bench(
    task: Annotated[
        Path | None,
        typer.Option("--task", help="Path to a single task JSON file."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Target model identifier (e.g., claude-opus-4-7, gpt-5.4, gemini-2.5-pro).",
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
    benchmark: Annotated[
        str | None,
        typer.Option("--benchmark", help="Curated benchmark suite (not yet supported)."),
    ] = None,
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
            help="Metric dimensions (not yet wired into the CLI).",
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
    """Run the reliability benchmark against a wrapped Agent.

    Single-task wiring: runs the Task at ``--task`` against ``--model``
    for ``--reps`` iterations using :class:`SimplePromptingAgent`,
    emits OTel GenAI spans (``benchmark`` → ``task`` → ``rep`` →
    ``chat {model}``) plus per-rep ``score`` spans, and writes the
    verdict-augmented ``RunResult`` JSON to ``<output>/<task_id>.json``.
    """
    if task is None:
        typer.echo("error: --task is required.", err=True)
        raise typer.Exit(2)
    if model is None:
        typer.echo("error: --model is required.", err=True)
        raise typer.Exit(2)
    if benchmark is not None:
        typer.echo(
            "error: --benchmark is not yet supported. Pass --task <path>.",
            err=True,
        )
        raise typer.Exit(2)
    if agent is not None:
        typer.echo(
            "error: --agent is not yet supported (built-in SimplePromptingAgent in use).",
            err=True,
        )
        raise typer.Exit(2)
    if metrics is not None:
        typer.echo(f"warning: --metrics={metrics!r} is not yet wired; ignoring.", err=True)
    if exporter not in {"console", "otlp", "none"}:
        typer.echo(
            f"error: --exporter={exporter!r} must be one of: console, otlp, none.",
            err=True,
        )
        raise typer.Exit(2)
    exporter_kind: ExporterKind = exporter  # type: ignore[assignment]

    task_obj = Task.model_validate_json(task.read_text())

    try:
        provider = provider_for_model(model)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e

    target_client = _build_client(provider)
    sf_agent = SimplePromptingAgent(client=target_client, model=model)

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"{task_obj.id}.sqlite"

    typer.echo(
        f"running task {task_obj.id} on {model} ({provider}) with reps={reps} "
        f"exporter={exporter}...",
        err=True,
    )

    # Configure tracing once per CLI invocation. The provider is held so we
    # can flush+shut down at the end (otherwise BatchSpanProcessor may drop
    # spans on process exit).
    provider_obj = configure_tracing(exporter=exporter_kind)
    try:
        result = asyncio.run(
            _run_and_judge(
                agent=sf_agent,
                task_obj=task_obj,
                reps=reps,
                model=model,
                checkpoint_path=checkpoint_path,
                target_provider=provider,
                target_client=target_client,
            )
        )
    finally:
        # force_flush before shutdown so console/OTLP exporters drain on
        # exit. Both methods are best-effort and tolerant of "no processor".
        provider_obj.force_flush()
        provider_obj.shutdown()

    out_path = output / f"{task_obj.id}.json"
    out_path.write_text(result.model_dump_json(indent=2))

    statuses = Counter(r.status.value for r in result.reps)
    total_cost = sum(
        (
            Decimal(str(r.response.cost_usd))
            for r in result.reps
            if r.response and r.response.cost_usd is not None
        ),
        Decimal("0"),
    )

    judged_reps = [r for r in result.reps if r.verdict is not None]
    passed_reps = [r for r in judged_reps if r.verdict and r.verdict.passed]
    if judged_reps:
        mean_score = sum(r.verdict.score for r in judged_reps if r.verdict) / len(judged_reps)
        verdict_summary = (
            f"verdicts={len(passed_reps)}/{len(judged_reps)} passed mean_score={mean_score:.3f}"
        )
    else:
        verdict_summary = "verdicts=0/0 (no reps scored)"

    typer.echo(f"wrote {out_path}", err=True)
    typer.echo(
        f"run_id={result.run_id} reps={len(result.reps)} "
        f"statuses={dict(statuses)} cost_usd={total_cost} {verdict_summary}"
    )


async def _run_and_judge(
    *,
    agent: Agent,
    task_obj: Task,
    reps: int,
    model: str,
    checkpoint_path: Path,
    target_provider: str,
    target_client: BaseModelClient,
) -> RunResult:
    """Combined run + judge so they share a single asyncio loop and a single
    ``benchmark`` span context.
    """
    rubric_client: BaseModelClient | None = None
    if task_obj.judge == "rubric":
        rubric_client = _build_rubric_client(target_provider, target_client)

    with benchmark_span(name=task_obj.id, package_version=__version__):
        result = await run_task(
            agent=agent,
            task=task_obj,
            reps=reps,
            model=model,
            checkpoint_path=checkpoint_path,
        )
        await judge_run_result(result, rubric_client=rubric_client)
    return result


def main() -> None:
    """Console-script entry point declared in ``pyproject.toml``."""
    # Default logging at WARNING so judge-failure messages from
    # judges.judge_run_result are visible without flooding the user.
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    app()
