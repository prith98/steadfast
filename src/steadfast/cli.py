"""Typer CLI surface.

Monday's skeleton exposes ``steadfast --help``, ``steadfast --version``, and
``steadfast bench --help`` with placeholder flags. Tuesday wires ``bench`` to
``steadfast.runner``; later weeks add metric and report subcommands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from steadfast import __version__

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


@app.command()
def bench(
    agent: Annotated[
        str | None,
        typer.Option(
            "--agent",
            help="Import path to the Agent subclass under test (e.g., my_pkg:MyAgent).",
        ),
    ] = None,
    task: Annotated[
        Path | None,
        typer.Option("--task", help="Path to a single task JSON file."),
    ] = None,
    benchmark: Annotated[
        str | None,
        typer.Option("--benchmark", help="Name of a curated benchmark suite."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Target model identifier."),
    ] = None,
    reps: Annotated[
        int,
        typer.Option("--reps", help="Number of repetitions per task (methodology default N=10)."),
    ] = 10,
    metrics: Annotated[
        str | None,
        typer.Option(
            "--metrics",
            help="Comma-separated dimensions: consistency,robustness,calibration,safety.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output directory for results and reports."),
    ] = None,
    exporter: Annotated[
        str,
        typer.Option(
            "--exporter",
            help="OTel span exporter: 'console' (default) or 'otlp'.",
        ),
    ] = "console",
) -> None:
    """Run the reliability benchmark against a wrapped Agent.

    Not implemented in the Monday skeleton — see ``docs/WEEK_1.md`` for the
    schedule. Tuesday wires the runner; Wednesday wires tracing; Thursday
    adds consistency metrics; Friday adds calibration and the pilot run.
    """
    del agent, task, benchmark, model, reps, metrics, output, exporter
    typer.echo("steadfast bench: not implemented (skeleton).", err=True)
    raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point declared in ``pyproject.toml``."""
    app()
