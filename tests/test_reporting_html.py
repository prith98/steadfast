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
