"""HTML report generation — single-file output for ``steadfast bench``.

The HTML report is the human-facing artifact for local runs; the
leaderboard (week 3) consumes the same JSON that this report renders.
v0.1 prioritizes information density and unambiguous CI display over
visual polish. Inline CSS keeps the file self-contained — no JavaScript,
no external assets, no network requests.

The report layout:

* Run header (benchmark, models, date).
* Per-(model) calibration table — Brier / ECE / refusal sensitivity-
  specificity / overconfidence rate, with bootstrap or Wilson CIs.
* Per-(model, task) output-consistency table — mean rubric (CI) and
  mean embedding cosine.
* Per-task run summary — pass-rate per model with Wilson CI.
* Reproducibility footer — package version, methodology version,
  ADR references.

The renderer reads the per-model JSON files written by the CLI; each
file is a Pydantic round-trip of the corresponding result model. Reading
JSON rather than threading the in-memory result objects keeps the HTML
generator decoupled from the runner — a future leaderboard ingest path
can render the same report from JSON the leaderboard backend produces.
"""

from __future__ import annotations

import datetime
import html
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from steadfast import __version__
from steadfast.metrics.calibration import (
    BrierResult,
    CalibrationDimension,
    ECEResult,
    OverconfidenceResult,
)
from steadfast.metrics.consistency import OutputConsistencyResult
from steadfast.runner import RunResult
from steadfast.stats.bootstrap import BootstrapCI
from steadfast.stats.wilson import WilsonCI, wilson_ci

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 2em auto; padding: 0 1.5em; color: #222; }
h1, h2, h3 { color: #111; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { padding: 0.5em 0.75em; border-bottom: 1px solid #e0e0e0; text-align: left;
         font-variant-numeric: tabular-nums; }
th { background: #f5f5f5; font-weight: 600; }
tr:hover td { background: #fafafa; }
.ci { color: #666; font-size: 0.9em; }
.na { color: #999; font-style: italic; }
.warn { color: #b25700; }
.passed { color: #1a7a1a; }
.failed { color: #a01a1a; }
footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #ddd;
         color: #666; font-size: 0.85em; }
code { background: #f5f5f5; padding: 0.1em 0.3em; border-radius: 3px;
       font-family: ui-monospace, SF Mono, Menlo, monospace; }
.section { margin-bottom: 2.5em; }
.subtle { color: #666; font-size: 0.9em; }
""".strip()


def _slug(model: str) -> str:
    """Filesystem-safe slug — must match the CLI's :func:`steadfast.cli._slug`.

    Duplicated here rather than imported because pulling in cli.py would
    create a cycle at import time (cli imports this module).
    """
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in model)


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return _na()
    return f"{x:.3f}"


def _fmt_ci(ci: BootstrapCI | WilsonCI | None) -> str:
    if ci is None:
        return _na()
    if isinstance(ci, BootstrapCI):
        return (
            f"<span class='ci'>[{ci.ci_lower:.3f}, {ci.ci_upper:.3f}]"
            f"{' degenerate' if ci.degenerate else ''}</span>"
        )
    return f"<span class='ci'>[{ci.ci_lower:.3f}, {ci.ci_upper:.3f}]</span>"


def _na() -> str:
    return "<span class='na'>N/A</span>"


_M = TypeVar("_M", bound=BaseModel)


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        result: dict[str, object] = json.loads(path.read_text())
        return result
    except json.JSONDecodeError:
        return None


def _safe_load(path: Path, model_cls: type[_M]) -> _M | None:
    if not path.is_file():
        return None
    return model_cls.model_validate_json(path.read_text())


def _h(s: str) -> str:
    """HTML-escape user-controlled strings before inlining."""
    return html.escape(s, quote=True)


def _render_calibration_section(
    target_models: list[str],
    output_dir: Path,
) -> str:
    """Per-model calibration table — Brier, ECE, refusal cells, overconfidence."""
    rows: list[str] = []
    rows.append(
        "<tr>"
        "<th>Model</th>"
        "<th>Brier (verbalized)</th>"
        "<th>Brier (logprob)</th>"
        "<th>ECE (verbalized)</th>"
        "<th>Refusal sens.</th>"
        "<th>Refusal spec.</th>"
        "<th>Overconfidence</th>"
        "<th>n / n_total</th>"
        "</tr>"
    )
    any_calibration = False
    for model in target_models:
        calib = _safe_load(output_dir / _slug(model) / "calibration.json", CalibrationDimension)
        if calib is None:
            continue
        any_calibration = True
        rows.append(_render_calibration_row(model, calib))
    if not any_calibration:
        return ""
    return f"""
<section class="section">
  <h2>Calibration</h2>
  <p class="subtle">Pooled bootstrap (BCa, 10k resamples) per ADR-0005 §D over (task, rep) squared
  errors. Logprob-derived Brier shows N/A where the provider's API doesn't expose per-token
  logprobs (per METHODOLOGY §3.1 / ADR-0005 §A — Anthropic and Google return N/A in v0.1).</p>
  <table>
    {"".join(rows)}
  </table>
</section>
"""


def _render_calibration_row(model: str, calib: CalibrationDimension) -> str:
    brier = calib.brier
    ece = calib.ece
    refusal = calib.refusal
    over = calib.overconfidence
    return (
        "<tr>"
        f"<td><code>{_h(model)}</code></td>"
        f"<td>{_render_brier_cell(brier, kind='verbalized')}</td>"
        f"<td>{_render_brier_cell(brier, kind='logprob')}</td>"
        f"<td>{_render_ece_cell(ece)}</td>"
        f"<td>{_render_proportion_cell(refusal.sensitivity, refusal.sensitivity_ci)}"
        f" <span class='subtle'>(n_hard={refusal.n_hard})</span></td>"
        f"<td>{_render_proportion_cell(refusal.specificity, refusal.specificity_ci)}"
        f" <span class='subtle'>(n_normal={refusal.n_normal})</span></td>"
        f"<td>{_render_overconfidence_cell(over)}</td>"
        f"<td><span class='subtle'>{brier.n} / {brier.n_total}</span></td>"
        "</tr>"
    )


def _render_brier_cell(brier: BrierResult, *, kind: str) -> str:
    ci = brier.verbalized if kind == "verbalized" else brier.logprob
    if ci is None:
        return _na()
    return f"{ci.point_estimate:.3f} {_fmt_ci(ci)}"


def _render_ece_cell(ece: ECEResult) -> str:
    if ece.verbalized is None:
        warn = f"<span class='warn'>{_h(ece.reason or '')}</span>" if ece.reason else _na()
        return warn
    fallback_note = (
        f" <span class='warn'>(fallback {ece.n_bins} bins; pool too small for 15)</span>"
        if ece.fallback_used
        else ""
    )
    return f"{ece.verbalized:.3f}{fallback_note}"


def _render_proportion_cell(value: float | None, ci: WilsonCI | None) -> str:
    if value is None or ci is None:
        return _na()
    return f"{value:.3f} {_fmt_ci(ci)}"


def _render_overconfidence_cell(over: OverconfidenceResult) -> str:
    if over.rate is None or over.ci is None:
        return _na()
    return (
        f"{over.rate:.3f} {_fmt_ci(over.ci)} "
        f"<span class='subtle'>({over.n_overconfident}/{over.n_answered})</span>"
    )


def _render_consistency_section(
    target_models: list[str],
    output_dir: Path,
) -> str:
    """Per-(model, task) consistency table — mean rubric and embedding cosine."""
    # Discover task IDs by listing consistency_*.json under each model dir.
    task_ids: set[str] = set()
    rows_by_model: dict[str, dict[str, OutputConsistencyResult]] = {}
    for model in target_models:
        model_results: dict[str, OutputConsistencyResult] = {}
        model_dir = output_dir / _slug(model)
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.glob("consistency_*.json")):
            stem = path.stem
            task_id = stem[len("consistency_") :]
            consistency = _safe_load(path, OutputConsistencyResult)
            if consistency is not None:
                model_results[task_id] = consistency
                task_ids.add(task_id)
        rows_by_model[model] = model_results

    if not task_ids:
        return ""

    sorted_tasks = sorted(task_ids)
    header_cols = "".join(f"<th>{_h(model)}</th>" for model in target_models)
    rows = [f"<tr><th>Task</th>{header_cols}</tr>"]
    for task_id in sorted_tasks:
        cells: list[str] = [f"<td><code>{_h(task_id)}</code></td>"]
        for model in target_models:
            consistency = rows_by_model.get(model, {}).get(task_id)
            if consistency is None:
                cells.append(f"<td>{_na()}</td>")
                continue
            cells.append(
                "<td>"
                f"rubric {consistency.mean_rubric:.3f} "
                f"{_fmt_ci(consistency.rubric_ci)} <br/>"
                f"<span class='subtle'>cosine {consistency.mean_embedding_cosine:.3f}</span>"
                "</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="section">
  <h2>Output consistency</h2>
  <p class="subtle">K=5 paraphrases per task; pairwise 0-4 Likert rubric (normalized to [0, 1])
  with bootstrap CI; embedding cosine on the same answer pool. METHODOLOGY §1.1.</p>
  <table>{"".join(rows)}</table>
</section>
"""


def _render_pass_rate_section(
    target_models: list[str],
    output_dir: Path,
    benchmark_name: str,
) -> str:
    """Per-task pass-rate table across models with Wilson CI."""
    del benchmark_name  # name is in the header; not needed for this section
    task_ids: set[str] = set()
    pass_rates: dict[str, dict[str, tuple[int, int]]] = {}  # model → task_id → (passed, total)
    for model in target_models:
        per_task: dict[str, tuple[int, int]] = {}
        model_dir = output_dir / _slug(model)
        if not model_dir.is_dir():
            continue
        for run_path in sorted(model_dir.glob("*.json")):
            if run_path.name in {"calibration.json"} or run_path.name.startswith("consistency_"):
                continue
            run_data = _safe_load(run_path, RunResult)
            if run_data is None:
                continue
            judged = [r for r in run_data.reps if r.verdict is not None]
            passed_reps = [r for r in judged if r.verdict and r.verdict.passed]
            per_task[run_data.task.id] = (len(passed_reps), len(judged))
            task_ids.add(run_data.task.id)
        pass_rates[model] = per_task

    if not task_ids:
        return ""

    sorted_tasks = sorted(task_ids)
    header = (
        "<tr><th>Task</th>" + "".join(f"<th>{_h(model)}</th>" for model in target_models) + "</tr>"
    )
    rows = [header]
    for task_id in sorted_tasks:
        cells = [f"<td><code>{_h(task_id)}</code></td>"]
        for model in target_models:
            data = pass_rates.get(model, {}).get(task_id)
            if data is None or data[1] == 0:
                cells.append(f"<td>{_na()}</td>")
                continue
            n_passed, n_total = data
            ci = wilson_ci(successes=n_passed, trials=n_total)
            cells.append(
                f"<td>{n_passed}/{n_total} <br/>"
                f"<span class='ci'>{ci.proportion:.2f} {_fmt_ci(ci)}</span></td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="section">
  <h2>Per-task pass rate (Wilson 95% CI)</h2>
  <p class="subtle">N reps per (model, task) per ADR-0002. Pass = rubric/exact-match verdict
  passed; failed and unjudged reps not in the denominator.</p>
  <table>{"".join(rows)}</table>
</section>
"""


def _render_header(
    benchmark_name: str,
    target_models: Iterable[str],
    requested_metrics: Iterable[str],
) -> str:
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    models = ", ".join(f"<code>{_h(m)}</code>" for m in target_models)
    metrics = ", ".join(sorted(requested_metrics)) or "[none]"
    return f"""
<header>
  <h1>Steadfast — {_h(benchmark_name)}</h1>
  <p class="subtle">
    Models: {models} &middot; metrics: {_h(metrics)} &middot; generated: {_h(now)} &middot;
    package: <code>steadfast {__version__}</code>
  </p>
</header>
"""


def _render_footer() -> str:
    return f"""
<footer>
  <p>
    Methodology: see <code>docs/METHODOLOGY.md</code>. Statistical conventions: 95% CIs,
    BCa bootstrap with 10k resamples (per <code>docs/adr/0004-consistency-and-stats.md</code>),
    Wilson 95% CIs for binomial proportions. Calibration follows
    <code>docs/adr/0005-calibration-and-confidence.md</code>: pooled-bootstrap Brier with parallel
    verbalized + logprob columns; ECE with 15 equal-mass bins (Nixon et al. 2019); refusal
    confusion matrix on (Task.difficulty, AgentResponse.refused). Verbalized confidence is the
    leaderboard headline; logprob is N/A for Anthropic and Google in v0.1.
  </p>
  <p>
    Generated by <code>steadfast {__version__}</code>.
  </p>
</footer>
"""


def write_html_report(
    *,
    output_dir: Path,
    benchmark_name: str,
    target_models: list[str],
    requested_metrics: frozenset[str],
    report_path: Path,
) -> None:
    """Render the HTML report from the JSON files under ``output_dir``.

    The report is a single self-contained ``.html`` file. Failure modes:

    * If a model's calibration / consistency JSON is missing the
      corresponding row in that table renders as N/A — the report is
      partial but doesn't crash.
    * If no JSON files exist (e.g., the run failed before any task
      completed), the body sections are empty but the header and footer
      render so the user has something to look at.
    """
    body_parts: list[str] = [
        _render_header(benchmark_name, target_models, requested_metrics),
        _render_calibration_section(target_models, output_dir),
        _render_consistency_section(target_models, output_dir),
        _render_pass_rate_section(target_models, output_dir, benchmark_name),
        _render_footer(),
    ]
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Steadfast — {_h(benchmark_name)}</title>
  <style>{_CSS}</style>
</head>
<body>
{"".join(body_parts)}
</body>
</html>
"""
    report_path.write_text(page)
