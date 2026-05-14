"""Tests for steadfast.reporting.html — HTML report generation.

The report reads JSON files written by the CLI; tests construct those
JSONs by serializing in-memory result models, write them to a temp dir,
and inspect the rendered HTML for required structural elements. We don't
parse the HTML formally — substring assertions are sufficient at v0.1
fidelity.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from steadfast.agent import AgentResponse, Task
from steadfast.judges.base import Verdict
from steadfast.metrics.calibration import CalibrationDimension, measure_calibration
from steadfast.metrics.consistency import OutputConsistencyResult
from steadfast.metrics.robustness import (
    RobustnessDimension,
    RobustnessSubMetricResult,
    RobustnessTaskResult,
)
from steadfast.reporting.html import write_html_report
from steadfast.runner import RepRecord, RepStatus, RunResult
from steadfast.stats.bootstrap import BootstrapCI


def _calibration_for(model: str, mostly_right: bool = True) -> CalibrationDimension:
    """Build a CalibrationDimension with hand-constructed inputs."""
    from steadfast.metrics.calibration import CalibrationRep

    reps = []
    for i in range(20):
        passed = (i % 5 != 0) if mostly_right else (i % 5 == 0)
        reps.append(
            CalibrationRep(
                task=Task(
                    id=f"t{i % 5}",
                    domain="d",
                    input="x",
                    difficulty="hard" if i % 5 == 4 else "normal",
                ),
                response=AgentResponse(answer="ans", confidence=0.85, refused=False),
                verdict=Verdict(
                    score=1.0 if passed else 0.0,
                    passed=passed,
                    reason="t",
                ),
            )
        )
    return measure_calibration(reps, model=model, n_tasks=5, seed=0)


def _run_result_for(task_id: str, *, n_passed: int, n_total: int) -> RunResult:
    """Build a RunResult with ``n_passed`` of ``n_total`` reps marked as passing."""
    task = Task(id=task_id, domain="d", input="x")
    reps = []
    for i in range(n_total):
        passed = i < n_passed
        reps.append(
            RepRecord(
                run_id="r",
                task_id=task_id,
                rep_idx=i,
                status=RepStatus.COMPLETED,
                response=AgentResponse(answer="ans"),
                cost_usd=Decimal("0"),
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                verdict=Verdict(score=1.0 if passed else 0.0, passed=passed, reason="t"),
            )
        )
    return RunResult(run_id="r", task=task, reps=reps)


def _consistency_for(task_id: str) -> OutputConsistencyResult:
    rubric_ci = BootstrapCI(
        point_estimate=0.78,
        ci_lower=0.6,
        ci_upper=0.95,
        confidence_level=0.95,
        method="BCa",
        n_resamples=10000,
        n_samples=10,
    )
    embed_ci = BootstrapCI(
        point_estimate=0.91,
        ci_lower=0.85,
        ci_upper=0.98,
        confidence_level=0.95,
        method="BCa",
        n_resamples=10000,
        n_samples=10,
    )
    return OutputConsistencyResult(
        task_id=task_id,
        k=5,
        paraphrase_rejection_rate=0.0,
        rubric_scores=[1.0] * 10,
        embedding_cosines=[0.91] * 10,
        mean_rubric=0.78,
        mean_embedding_cosine=0.91,
        rubric_ci=rubric_ci,
        embedding_ci=embed_ci,
        embedding_model="text-embedding-3-large",
        rubric_model="gpt-5.2",
    )


@pytest.fixture
def report_workspace(tmp_path: Path) -> Path:
    """Construct a results directory with two models' JSON files."""
    for model in ["claude-opus-4-7", "gpt-5.2"]:
        slug = model.replace("/", "_")
        model_dir = tmp_path / slug
        model_dir.mkdir(parents=True)
        # Calibration
        calib = _calibration_for(model, mostly_right=(model == "claude-opus-4-7"))
        (model_dir / "calibration.json").write_text(calib.model_dump_json(indent=2))
        # Consistency for two tasks
        for task_id in ["t1", "t2"]:
            consistency = _consistency_for(task_id)
            (model_dir / f"consistency_{task_id}.json").write_text(
                consistency.model_dump_json(indent=2)
            )
        # Run results for two tasks
        for task_id in ["t1", "t2"]:
            run = _run_result_for(task_id, n_passed=8, n_total=10)
            (model_dir / f"{task_id}.json").write_text(run.model_dump_json(indent=2))
    return tmp_path


def test_write_html_report_creates_file(report_workspace: Path) -> None:
    report_path = report_workspace / "report.html"
    write_html_report(
        output_dir=report_workspace,
        benchmark_name="customer_support_pilot",
        target_models=["claude-opus-4-7", "gpt-5.2"],
        requested_metrics=frozenset({"calibration", "consistency"}),
        report_path=report_path,
    )
    assert report_path.is_file()
    contents = report_path.read_text()
    # Header must mention the benchmark name and both models.
    assert "customer_support_pilot" in contents
    assert "claude-opus-4-7" in contents
    assert "gpt-5.2" in contents


def test_html_report_calibration_table_renders(report_workspace: Path) -> None:
    report_path = report_workspace / "report.html"
    write_html_report(
        output_dir=report_workspace,
        benchmark_name="b",
        target_models=["claude-opus-4-7", "gpt-5.2"],
        requested_metrics=frozenset({"calibration"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    # Calibration table header columns must appear.
    assert "Brier (verbalized)" in contents
    assert "Brier (logprob)" in contents
    assert "ECE" in contents
    assert "Refusal sens." in contents
    assert "Overconfidence" in contents


def test_html_report_consistency_table_renders(report_workspace: Path) -> None:
    report_path = report_workspace / "report.html"
    write_html_report(
        output_dir=report_workspace,
        benchmark_name="b",
        target_models=["claude-opus-4-7", "gpt-5.2"],
        requested_metrics=frozenset({"consistency"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert "Output consistency" in contents
    assert "rubric" in contents
    assert "0.780" in contents  # mean_rubric value


def test_html_report_pass_rate_section_renders(report_workspace: Path) -> None:
    report_path = report_workspace / "report.html"
    write_html_report(
        output_dir=report_workspace,
        benchmark_name="b",
        target_models=["claude-opus-4-7", "gpt-5.2"],
        requested_metrics=frozenset({"calibration"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert "Per-task pass rate" in contents
    assert "8/10" in contents


def test_html_report_handles_missing_files_gracefully(tmp_path: Path) -> None:
    """Empty workspace → report renders with N/A cells, no crash."""
    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="empty",
        target_models=["claude-opus-4-7"],
        requested_metrics=frozenset({"calibration", "consistency"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    # Header + footer always render even with no data.
    assert "Steadfast" in contents
    assert "claude-opus-4-7" in contents


def _robustness_for(
    model: str,
    *,
    typo_delta: float = -0.10,
    typo_ci: tuple[float, float] | None = (-0.20, 0.00),
    distractor_delta: float | None = -0.05,
    distractor_ci: tuple[float, float] | None = (-0.15, 0.05),
    n_tasks: int = 5,
) -> RobustnessDimension:
    """Build a RobustnessDimension fixture for the HTML render test."""

    def _per_task(task_id: str, kind: str, clean: float, perturbed: float) -> RobustnessTaskResult:
        return RobustnessTaskResult(
            task_id=task_id,
            kind=kind,  # type: ignore[arg-type]
            n_reps_clean=10,
            n_reps_perturbed=10,
            clean_rate=clean,
            perturbed_rate=perturbed,
            delta=perturbed - clean,
            clean_passes=[True] * 10,
            perturbed_passes=[True] * 10,
            perturbed_input_previews=["..."] * 10,
            seed=0,
        )

    def _sub(
        kind: str, delta: float | None, ci: tuple[float, float] | None
    ) -> RobustnessSubMetricResult:
        return RobustnessSubMetricResult(
            kind=kind,  # type: ignore[arg-type]
            n_tasks=n_tasks,
            clean_mean=0.9 if delta is not None else None,
            perturbed_mean=(0.9 + delta) if delta is not None else None,
            delta=delta,
            delta_ci_lower=ci[0] if ci is not None else None,
            delta_ci_upper=ci[1] if ci is not None else None,
            confidence_level=0.95 if ci is not None else None,
            method="BCa" if ci is not None else None,
            n_resamples=10000 if ci is not None else None,
            per_task=[_per_task(f"t{i}", kind, 0.9, 0.9 + (delta or 0.0)) for i in range(n_tasks)],
            reason=None if delta is not None else "no tasks measured",
        )

    return RobustnessDimension(
        model=model,
        n_tasks=n_tasks,
        sub_metrics={
            "typo": _sub("typo", typo_delta, typo_ci),
            "distractor": _sub("distractor", distractor_delta, distractor_ci),
        },
    )


def test_html_report_robustness_section_renders(tmp_path: Path) -> None:
    """End-to-end: a robustness JSON in a model dir → a robustness section."""
    model = "claude-opus-4-7"
    model_dir = tmp_path / model.replace("/", "_")
    model_dir.mkdir(parents=True)
    dim = _robustness_for(model)
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert "<h2>Robustness</h2>" in contents
    assert "typo" in contents
    assert "distractor" in contents
    # Point estimate + CI must render with sign and bracketed bounds.
    assert "-0.100" in contents
    assert "[-0.200, +0.000]" in contents
    # Means line shown for context.
    assert "clean 0.900" in contents


def test_html_report_robustness_handles_na_ci(tmp_path: Path) -> None:
    """N/A CI (n_tasks=1 single-task surface) renders the point estimate + N/A."""
    model = "gpt-5.2"
    model_dir = tmp_path / model.replace("/", "_")
    model_dir.mkdir(parents=True)
    dim = _robustness_for(
        model,
        typo_delta=-0.10,
        typo_ci=None,
        distractor_delta=None,
        distractor_ci=None,
    )
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert "<h2>Robustness</h2>" in contents
    # Typo: point estimate present, CI N/A.
    assert "-0.100" in contents
    # Distractor: fully None → reason shown.
    assert "no tasks measured" in contents


def test_html_report_robustness_section_skipped_when_no_files(tmp_path: Path) -> None:
    """Empty workspace → no Robustness section header in the rendered HTML."""
    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=["m"],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert "<h2>Robustness</h2>" not in contents


def _contradiction_for(
    model: str,
    *,
    p_detect: float | None = 0.4,
    p_retry: float | None = 0.3,
    p_halluc: float | None = 0.3,
    n_with_tools: int = 30,
    n_tasks: int = 3,
    value: str | None = "measured",
    reason: str | None = None,
) -> RobustnessDimension:
    """Build a RobustnessDimension carrying a ContradictionResult sub-metric."""
    from steadfast.metrics.robustness import (
        ContradictionResult,
        ContradictionTaskResult,
    )
    from steadfast.stats.wilson import wilson_ci

    has_marginals = (
        value == "measured"
        and p_detect is not None
        and p_retry is not None
        and p_halluc is not None
    )
    if has_marginals:
        # Type narrowing for mypy: the conjunctive guard above pins all three
        # to non-None, but the variable annotations here make that explicit.
        assert p_detect is not None
        assert p_retry is not None
        assert p_halluc is not None
        n_detect = round(p_detect * n_with_tools)
        n_retry = round(p_retry * n_with_tools)
        n_halluc = n_with_tools - n_detect - n_retry
        ci_detect = wilson_ci(n_detect, n_with_tools)
        ci_retry = wilson_ci(n_retry, n_with_tools)
        ci_halluc = wilson_ci(n_halluc, n_with_tools)
    else:
        ci_detect = ci_retry = ci_halluc = None

    sub = ContradictionResult(
        n_tasks=n_tasks,
        n_reps_with_tools=n_with_tools,
        p_detect=p_detect,
        p_retry=p_retry,
        p_halluc=p_halluc,
        ci_detect=ci_detect,
        ci_retry=ci_retry,
        ci_halluc=ci_halluc,
        per_task=[
            ContradictionTaskResult(
                task_id=f"t{i}",
                n_reps_with_tools=n_with_tools // n_tasks if n_tasks else 0,
                n_reps_completed=10,
                labels=[],
                n_corrupted_calls_per_rep=[],
                seed=0,
            )
            for i in range(n_tasks)
        ],
        value="measured" if value == "measured" else None,  # type: ignore[arg-type]
        reason=reason,
    )

    return RobustnessDimension(
        model=model,
        n_tasks=n_tasks,
        sub_metrics={"contradiction": sub},
    )


def test_html_report_contradiction_section_renders_three_bars(tmp_path: Path) -> None:
    """Contradiction sub-metric renders three labeled lines with Wilson CIs."""
    model = "claude-opus-4-7"
    model_dir = tmp_path / model.replace("/", "_")
    model_dir.mkdir(parents=True)
    dim = _contradiction_for(model, p_detect=0.4, p_retry=0.3, p_halluc=0.3, n_with_tools=30)
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert "<h2>Robustness</h2>" in contents
    # All three label lines should appear in the cell.
    assert "detect 0.400" in contents
    assert "retry 0.300" in contents
    assert "halluc 0.300" in contents
    # n=30 footer line.
    assert "n=30" in contents
    # Section copy mentions the marginal-CI semantics.
    assert "not jointly bounded" in contents


def test_html_report_contradiction_n_a_renders_reason(tmp_path: Path) -> None:
    """Toolless-agent contradiction surface renders the reason string in warn style."""
    model = "gpt-5.2"
    model_dir = tmp_path / model.replace("/", "_")
    model_dir.mkdir(parents=True)
    dim = _contradiction_for(
        model,
        p_detect=None,
        p_retry=None,
        p_halluc=None,
        n_with_tools=0,
        value=None,
        reason="agent did not call any tools",
    )
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert "agent did not call any tools" in contents
    assert "warn" in contents  # the warn-styled span carries the reason


def test_html_report_robustness_mixed_kinds_render_correctly(tmp_path: Path) -> None:
    """A model with both delta and contradiction sub-metrics renders both shapes."""
    model = "claude-opus-4-7"
    model_dir = tmp_path / model.replace("/", "_")
    model_dir.mkdir(parents=True)

    delta_dim = _robustness_for(model)
    contra_dim = _contradiction_for(model, p_detect=0.5, p_retry=0.3, p_halluc=0.2, n_with_tools=20)
    # Merge the two dimensions' sub_metrics into one RobustnessDimension JSON.
    combined = RobustnessDimension(
        model=model,
        n_tasks=delta_dim.n_tasks,
        sub_metrics={**delta_dim.sub_metrics, **contra_dim.sub_metrics},
    )
    (model_dir / "robustness.json").write_text(combined.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    # Delta cell from typo
    assert "-0.100" in contents
    # Contradiction cell.
    assert "detect 0.500" in contents
    assert "halluc 0.200" in contents


def test_html_report_escapes_user_strings(tmp_path: Path) -> None:
    """Benchmark / model names are inlined; unsafe HTML must be escaped."""
    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="<script>alert(1)</script>",
        target_models=["m<x>"],
        requested_metrics=frozenset({"calibration"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert "<script>alert(1)</script>" not in contents
    assert "&lt;script&gt;" in contents
    assert "m&lt;x&gt;" in contents


# ---------------------------------------------------------------------------
# Long-context rendering: SVG curve + per-task drill-down
# ---------------------------------------------------------------------------


def _long_context_for(
    model: str,
    *,
    fit_converged: bool = True,
    slope: float | None = -1.5,
    l50: float | None = 32_000,
    n_tasks: int = 3,
    lengths: tuple[int, ...] = (4_000, 16_000, 64_000, 128_000),
    skip_tier_for_first_task: bool = False,
) -> RobustnessDimension:
    """Build a RobustnessDimension carrying a LongContextResult.

    The synthetic curve: ``[1.0, 1.0, 0.0, 0.0]`` pooled across tasks,
    which is exactly the step-curve fixture used in the metric tests.
    The slope / l50 inputs let the test override the fit values
    independently of the empirical points.
    """
    from steadfast.metrics.robustness import (
        LongContextResult,
        LongContextTaskResult,
    )
    from steadfast.stats.wilson import wilson_ci

    rates = [1.0, 1.0, 0.0, 0.0]
    n_per_tier = 4 * n_tasks
    success_cis = []
    for rate in rates:
        n_pass = round(rate * n_per_tier)
        success_cis.append(wilson_ci(n_pass, n_per_tier))

    per_task = []
    for i in range(n_tasks):
        passes_per_length: list[list[bool]] = []
        rates_per_length: list[float | None] = []
        for tier_idx, rate in enumerate(rates):
            if skip_tier_for_first_task and i == 0 and tier_idx == len(lengths) - 1:
                # First task skipped the largest tier (input too long).
                passes_per_length.append([])
                rates_per_length.append(None)
            else:
                passes_per_length.append([rate >= 0.5] * 4)
                rates_per_length.append(rate)
        per_task.append(
            LongContextTaskResult(
                task_id=f"t{i}",
                lengths=list(lengths),
                passes_per_length=passes_per_length,
                rates_per_length=rates_per_length,
                perturbed_input_previews_per_length=[[""] * 4 for _ in range(len(lengths))],
                seed=i,
            )
        )

    sub = LongContextResult(
        kind="long_context",
        n_tasks=n_tasks,
        lengths=list(lengths),
        success_rates=rates,
        success_cis=success_cis,
        measured_length_indices=list(range(len(lengths))),
        slope=slope if fit_converged else None,
        slope_ci_lower=-2.0 if fit_converged else None,
        slope_ci_upper=-1.0 if fit_converged else None,
        l50=l50 if fit_converged else None,
        l50_ci_lower=25_000.0 if fit_converged else None,
        l50_ci_upper=40_000.0 if fit_converged else None,
        fit_converged=fit_converged,
        confidence_level=0.95 if fit_converged else None,
        n_resamples=10_000 if fit_converged else None,
        per_task=per_task,
        reason=None if fit_converged else "sigmoid fit did not converge",
    )
    return RobustnessDimension(
        model=model,
        n_tasks=n_tasks,
        sub_metrics={"long_context": sub},
    )


def test_html_report_long_context_renders_svg(tmp_path: Path) -> None:
    """Long-context cell embeds an inline SVG with curve, points, and fit overlay."""
    model = "claude-opus-4-7"
    model_dir = tmp_path / model
    model_dir.mkdir(parents=True)
    dim = _long_context_for(model)
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()

    # The SVG element renders inline with the right structural classes.
    assert 'class="lc-plot"' in contents
    assert "viewBox=" in contents
    # Tier labels are formatted compactly.
    assert ">4k</text>" in contents
    assert ">128k</text>" in contents
    # The fit overlay polyline renders when fit_converged.
    assert 'class="lc-fit"' in contents
    # The L_50 marker renders inside the plotted range (L_50=32k is between 4k and 128k).
    assert 'class="lc-l50"' in contents
    assert "L50 32k" in contents
    # Summary line under the plot carries slope and L_50 with CIs.
    assert "slope -1.50" in contents
    assert "L<sub>50</sub> 32,000" in contents


def test_html_report_long_context_fit_failure_hides_overlay(tmp_path: Path) -> None:
    """fit_converged=False → no sigmoid polyline, no L_50 marker, warn reason shown."""
    model = "gpt-5.2"
    model_dir = tmp_path / model
    model_dir.mkdir(parents=True)
    dim = _long_context_for(
        model,
        fit_converged=False,
        slope=None,
        l50=None,
    )
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()

    # SVG still renders (empirical points + axes) but no fit polyline / L_50 marker.
    assert 'class="lc-plot"' in contents
    assert 'class="lc-point"' in contents
    assert 'class="lc-fit"' not in contents
    assert 'class="lc-l50"' not in contents
    # Reason line surfaces the non-convergence.
    assert "did not converge" in contents


def test_html_report_long_context_l50_out_of_range_not_drawn(tmp_path: Path) -> None:
    """L_50 outside the plotted x-range → marker omitted from the SVG.

    A fit that places L_50 at e.g. 1k (below the 4k smallest tier) shouldn't
    paint a marker that visually claims a degradation point the data can't
    support.
    """
    model = "m"
    model_dir = tmp_path / model
    model_dir.mkdir(parents=True)
    dim = _long_context_for(model, slope=-1.5, l50=500.0)  # 500 < 4k smallest tier
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    # Curve still drawn, but L_50 marker line / label omitted.
    assert 'class="lc-fit"' in contents
    assert 'class="lc-l50"' not in contents
    # Summary strip still reports the value textually — only the plot
    # marker is suppressed, not the numeric report.
    assert "L<sub>50</sub> 500" in contents


def test_html_report_long_context_empty_renders_warn(tmp_path: Path) -> None:
    """No measured tiers → warn cell with reason, no SVG."""
    from steadfast.metrics.robustness import LongContextResult

    model = "m"
    model_dir = tmp_path / model
    model_dir.mkdir(parents=True)
    empty = LongContextResult(
        kind="long_context",
        n_tasks=0,
        lengths=[4_000, 16_000, 64_000, 128_000],
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
        per_task=[],
        reason="no length tier produced any judged rep",
    )
    dim = RobustnessDimension(
        model=model,
        n_tasks=0,
        sub_metrics={"long_context": empty},
    )
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert 'class="lc-plot"' not in contents
    assert "no length tier" in contents


# ---------------------------------------------------------------------------
# Per-task robustness drill-down section
# ---------------------------------------------------------------------------


def test_html_report_per_task_section_includes_typo_distractor_tables(tmp_path: Path) -> None:
    """Per-task drill-down renders a table per (kind, model) for typo + distractor."""
    model = "claude-opus-4-7"
    model_dir = tmp_path / model
    model_dir.mkdir(parents=True)
    dim = _robustness_for(model)  # carries typo + distractor with 5 per-task entries each
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    # New section heading appears.
    assert "Robustness — per-task detail" in contents
    # typo header + at least one per-task row.
    assert "typo — per task" in contents
    assert "distractor — per task" in contents
    # Per-task rows are labeled by task ID.
    assert "t0" in contents
    assert "t4" in contents
    # Column headers for the delta-style table.
    assert "Clean rate" in contents
    assert "Perturbed rate" in contents
    assert "Delta" in contents


def test_html_report_per_task_section_includes_long_context_tier_columns(tmp_path: Path) -> None:
    """Long-context per-task table has one column per measured tier."""
    model = "m"
    model_dir = tmp_path / model
    model_dir.mkdir(parents=True)
    dim = _long_context_for(model, n_tasks=2)
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert "long-context — per task" in contents
    # Each tier header is present.
    assert ">4k</th>" in contents
    assert ">16k</th>" in contents
    assert ">64k</th>" in contents
    assert ">128k</th>" in contents
    # Per-task rates rendered as ratio (n_pass/N).
    assert "(4/4)" in contents  # the 4k tier pass row
    assert "(0/4)" in contents  # the 128k tier fail row


def test_html_report_per_task_section_handles_skipped_tier(tmp_path: Path) -> None:
    """A task that skipped a tier (input too long) renders N/A in that cell."""
    model = "m"
    model_dir = tmp_path / model
    model_dir.mkdir(parents=True)
    dim = _long_context_for(model, n_tasks=2, skip_tier_for_first_task=True)
    (model_dir / "robustness.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=[model],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    # The 128k tier for t0 should render N/A.
    assert "N/A" in contents
    # The other tasks at the same tier still have a numeric cell.
    assert "(0/4)" in contents


def test_html_report_per_task_section_skipped_when_no_robustness_files(tmp_path: Path) -> None:
    """No per-model robustness.json → no per-task drill-down section."""
    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="b",
        target_models=["m"],
        requested_metrics=frozenset({"robustness"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    assert "Robustness — per-task detail" not in contents


def test_html_report_long_context_format_tokens_branches() -> None:
    """Unit test for _format_tokens covering all three branches."""
    from steadfast.reporting.html import _format_tokens

    assert _format_tokens(500) == "500"
    assert _format_tokens(4_000) == "4k"
    assert _format_tokens(128_000) == "128k"
    assert _format_tokens(2_000_000) == "2M"
    assert _format_tokens(1_500_000) == "1.5M"


def _build_safety_dim(model: str, traps: list[tuple[str, str, bool]]) -> object:
    """Build a SafetyDimension from a list of (case_id, category, harmful) trap triples."""
    from steadfast.metrics.safety import (
        CatastrophicRateResult,
        RefusalF1Result,
        SafetyCaseResult,
        SafetyDimension,
    )

    per_case = []
    n_harmful = 0
    for case_id, category, harmful in traps:
        per_case.append(
            SafetyCaseResult(
                case_id=case_id,
                case_kind="trap",
                category=category,
                refused=False,
                harmful=harmful,
                answer_preview="…",
            )
        )
        if harmful:
            n_harmful += 1
    return SafetyDimension(
        model=model,
        bank_version="v1",
        n_traps=len(traps),
        n_benigns=0,
        catastrophic=CatastrophicRateResult(
            n_traps_total=len(traps),
            n_traps_judged=len(traps),
            n_harmful=n_harmful,
            rate=n_harmful / len(traps) if traps else None,
            ci=None,
        ),
        refusal=RefusalF1Result(
            n_total=len(traps),
            n_traps=len(traps),
            n_benigns=0,
            tp=0,
            fp=0,
            fn=len(traps),
            tn=0,
            precision=None,
            recall=0.0,
            f1=0.0,
            f1_ci=None,
            n_resamples=None,
            confidence_level=None,
            method=None,
        ),
        per_case=per_case,
    )


def test_html_report_safety_per_category_breakdown(tmp_path: Path) -> None:
    """v0.1.x clarification fix per ADR-0008 §G item 2: per-harm-category sub-table."""
    model = "test-model"
    dim = _build_safety_dim(
        model,
        traps=[
            # 2/2 harmful on data_destruction
            ("td1", "data_destruction", True),
            ("td2", "data_destruction", True),
            # 0/2 harmful on illegal_advice
            ("ti1", "illegal_advice", False),
            ("ti2", "illegal_advice", False),
            # 1/2 harmful on pii_exfiltration
            ("tp1", "pii_exfiltration", True),
            ("tp2", "pii_exfiltration", False),
            # 0/2 harmful on social_engineering
            ("ts1", "social_engineering", False),
            ("ts2", "social_engineering", False),
        ],
    )
    slug = model.replace("/", "_")
    (tmp_path / slug).mkdir()
    (tmp_path / slug / "safety.json").write_text(dim.model_dump_json(indent=2))

    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="safety",
        target_models=[model],
        requested_metrics=frozenset({"safety"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    # Per-category sub-table header
    assert "Catastrophic rate by harm category" in contents
    # All four categories surface as rows
    for cat in ("data_destruction", "illegal_advice", "pii_exfiltration", "social_engineering"):
        assert cat in contents
    # data_destruction at 2/2 → 1.000 cell
    assert "1.000" in contents
    # illegal_advice and social_engineering at 0/2 → 0.000 cell
    assert "0.000" in contents
    # pii_exfiltration at 1/2 → 0.500 cell
    assert "0.500" in contents


def test_html_report_safety_per_category_handles_missing_judge(tmp_path: Path) -> None:
    """A category where every trap had a judge failure → N/A cell, not crash."""
    from steadfast.metrics.safety import (
        CatastrophicRateResult,
        RefusalF1Result,
        SafetyCaseResult,
        SafetyDimension,
    )

    per_case = [
        # data_destruction: both judge failed → N/A
        SafetyCaseResult(
            case_id="td1",
            case_kind="trap",
            category="data_destruction",
            refused=False,
            harmful=None,
            answer_preview="…",
            judge_reason="judge failed",
        ),
        SafetyCaseResult(
            case_id="td2",
            case_kind="trap",
            category="data_destruction",
            refused=False,
            harmful=None,
            answer_preview="…",
            judge_reason="judge failed",
        ),
    ]
    dim = SafetyDimension(
        model="m",
        bank_version="v1",
        n_traps=2,
        n_benigns=0,
        n_judge_failures=2,
        catastrophic=CatastrophicRateResult(
            n_traps_total=2,
            n_traps_judged=0,
            n_harmful=0,
            rate=None,
            ci=None,
            reason="no judged traps",
        ),
        refusal=RefusalF1Result(
            n_total=2,
            n_traps=2,
            n_benigns=0,
            tp=0,
            fp=0,
            fn=2,
            tn=0,
            precision=None,
            recall=0.0,
            f1=0.0,
            f1_ci=None,
            n_resamples=None,
            confidence_level=None,
            method=None,
        ),
        per_case=per_case,
    )
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "safety.json").write_text(dim.model_dump_json(indent=2))
    report_path = tmp_path / "report.html"
    write_html_report(
        output_dir=tmp_path,
        benchmark_name="safety",
        target_models=["m"],
        requested_metrics=frozenset({"safety"}),
        report_path=report_path,
    )
    contents = report_path.read_text()
    # Per-category table renders despite zero judged traps everywhere.
    assert "Catastrophic rate by harm category" in contents
    assert "N/A (0/2)" in contents
    # Other categories with zero traps at all show N/A (0/0).
    assert "N/A (0/0)" in contents
