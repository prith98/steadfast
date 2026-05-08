"""Typer CLI surface.

The Tuesday surface wires ``steadfast bench`` to the runner with a
provider-aware model client and the built-in :class:`SimplePromptingAgent`.
``--metrics`` and ``--exporter`` are accepted but ignored with a warning;
they land Wednesday alongside OTel tracing and Thursday/Friday alongside
the consistency / calibration metrics.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from steadfast import __version__
from steadfast.agent import SimplePromptingAgent, Task
from steadfast.models.base import BaseModelClient
from steadfast.models.pricing import provider_for_model
from steadfast.runner import run_task

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
        typer.Option("--benchmark", help="Curated benchmark suite (Wednesday+; ignored Tuesday)."),
    ] = None,
    agent: Annotated[
        str | None,
        typer.Option(
            "--agent",
            help="Import path to a custom Agent subclass (Wednesday+; ignored Tuesday).",
        ),
    ] = None,
    metrics: Annotated[
        str | None,
        typer.Option(
            "--metrics",
            help="Metric dimensions (Thursday/Friday; ignored Tuesday).",
        ),
    ] = None,
    exporter: Annotated[
        str,
        typer.Option(
            "--exporter",
            help="OTel exporter (Wednesday; only 'console' is functional Tuesday).",
        ),
    ] = "console",
) -> None:
    """Run the reliability benchmark against a wrapped Agent.

    Tuesday's wiring runs a single Task against a single model for ``reps``
    iterations using :class:`SimplePromptingAgent`. Output JSON lands at
    ``<output>/<task_id>.json``; checkpoint SQLite at
    ``<output>/<task_id>.sqlite`` (resumes automatically on subsequent runs).
    """
    if task is None:
        typer.echo("error: --task is required.", err=True)
        raise typer.Exit(2)
    if model is None:
        typer.echo("error: --model is required.", err=True)
        raise typer.Exit(2)
    if benchmark is not None:
        typer.echo(
            "error: --benchmark is not yet supported (Tuesday skeleton). "
            "Pass --task <path> to run a single task.",
            err=True,
        )
        raise typer.Exit(2)
    if agent is not None:
        typer.echo(
            "error: --agent is not yet supported (Tuesday skeleton uses the "
            "built-in SimplePromptingAgent).",
            err=True,
        )
        raise typer.Exit(2)
    if metrics is not None:
        typer.echo(f"warning: --metrics={metrics!r} is not yet wired; ignoring.", err=True)
    if exporter != "console":
        typer.echo(
            f"warning: --exporter={exporter!r} is not yet wired (Wednesday); using console.",
            err=True,
        )

    task_obj = Task.model_validate_json(task.read_text())

    try:
        provider = provider_for_model(model)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e

    client = _build_client(provider)
    sf_agent = SimplePromptingAgent(client=client, model=model)

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"{task_obj.id}.sqlite"

    typer.echo(
        f"running task {task_obj.id} on {model} ({provider}) with reps={reps}...",
        err=True,
    )

    result = asyncio.run(
        run_task(
            agent=sf_agent,
            task=task_obj,
            reps=reps,
            model=model,
            checkpoint_path=checkpoint_path,
        )
    )

    out_path = output / f"{task_obj.id}.json"
    out_path.write_text(result.model_dump_json(indent=2))

    statuses = Counter(r.status.value for r in result.reps)
    total_cost = sum(
        (r.response.cost_usd or 0)
        for r in result.reps
        if r.response is not None and r.response.cost_usd is not None
    )

    typer.echo(f"wrote {out_path}", err=True)
    typer.echo(
        f"run_id={result.run_id} reps={len(result.reps)} "
        f"statuses={dict(statuses)} cost_usd={total_cost}"
    )


def main() -> None:
    """Console-script entry point declared in ``pyproject.toml``."""
    app()
