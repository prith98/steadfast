"""Tests for benchmark task loading + the ADR-0008 §F audit gate.

The CLI's :func:`~steadfast.cli.resolve_benchmark` is the single entry
point that surfaces task files to the bench loop. These tests verify:

* Every reviewed task in every domain loads as a valid
  :class:`~steadfast.agent.Task`.
* The audit gate filters out drafted tasks with a warning.
* Files that aren't tasks (manifests, bank artifacts) don't leak in.
* Each domain hits the ≥10% hard-task floor from METHODOLOGY §3.4.

Per ADR-0008 §F the gate fires on bare-domain slugs only; the
``_pilot`` slug pre-dates the manifest and stays in legacy mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steadfast.agent import Task
from steadfast.cli import BenchmarkAuditManifest, resolve_benchmark


def _reviewed_domains() -> list[str]:
    """Return the domains that ship a `_review.json` manifest."""
    base = Path(__file__).resolve().parents[1] / "benchmarks"
    return sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and (p / "_review.json").is_file()
    )


def test_audit_manifest_loads_for_each_domain() -> None:
    """Every domain that ships a manifest produces a valid BenchmarkAuditManifest."""
    base = Path(__file__).resolve().parents[1] / "benchmarks"
    domains = _reviewed_domains()
    assert "customer_support" in domains
    assert "code_repair" in domains
    assert "multi_hop_research" in domains
    for d in domains:
        manifest = BenchmarkAuditManifest.model_validate_json(
            (base / d / "_review.json").read_text(encoding="utf-8")
        )
        # Lists are not None.
        assert isinstance(manifest.reviewed_tasks, list)
        assert isinstance(manifest.draft_tasks, list)


def test_customer_support_bare_slug_resolves_all_17_reviewed() -> None:
    """Bare-domain `customer_support` returns all 17 reviewed tasks (5 pilots + cs_001-012)."""
    paths = resolve_benchmark("customer_support")
    names = sorted(p.name for p in paths)
    expected = sorted(
        [f"cs_{i:03d}.json" for i in range(1, 13)]
        + [f"pilot_{i:03d}.json" for i in range(1, 6)]
    )
    assert names == expected


def test_pilot_slug_bypasses_audit_gate() -> None:
    """`customer_support_pilot` returns all pilot_*.json regardless of manifest state."""
    paths = resolve_benchmark("customer_support_pilot")
    names = sorted(p.name for p in paths)
    assert names == [
        "pilot_001.json",
        "pilot_002.json",
        "pilot_003.json",
        "pilot_004.json",
        "pilot_005.json",
    ]


def test_code_repair_bare_slug_resolves_all_17_reviewed() -> None:
    """Bare-domain `code_repair` returns all 17 reviewed cr_001-017 tasks."""
    paths = resolve_benchmark("code_repair")
    names = sorted(p.name for p in paths)
    assert names == [f"cr_{i:03d}.json" for i in range(1, 18)]


def test_multi_hop_research_bare_slug_resolves_all_17_reviewed() -> None:
    """Bare-domain `multi_hop_research` returns all 17 reviewed mhr_001-017 tasks."""
    paths = resolve_benchmark("multi_hop_research")
    names = sorted(p.name for p in paths)
    assert names == [f"mhr_{i:03d}.json" for i in range(1, 18)]


def test_audit_gate_passes_through_when_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Domains without a `_review.json` get legacy behavior (back-compat)."""
    # Mock _BENCHMARK_BASE to a tmp dir with a single task file.
    base = tmp_path / "benchmarks"
    domain = base / "legacy_domain"
    domain.mkdir(parents=True)
    (domain / "t1.json").write_text(
        Task(id="t1", domain="legacy_domain", input="x").model_dump_json()
    )
    monkeypatch.setattr("steadfast.cli._BENCHMARK_BASE", base)
    paths = resolve_benchmark("legacy_domain")
    assert len(paths) == 1


def test_audit_gate_filters_out_drafts_silently_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Partial-audit state: reviewed tasks load; drafts filter with stderr warning."""
    base = tmp_path / "benchmarks"
    domain = base / "test_domain"
    domain.mkdir(parents=True)
    (domain / "t1.json").write_text(
        Task(id="t1", domain="test_domain", input="x").model_dump_json()
    )
    (domain / "t2.json").write_text(
        Task(id="t2", domain="test_domain", input="x").model_dump_json()
    )
    (domain / "_review.json").write_text(
        BenchmarkAuditManifest(
            review_status="partial",
            reviewed_tasks=["t1"],
            draft_tasks=["t2"],
        ).model_dump_json()
    )
    monkeypatch.setattr("steadfast.cli._BENCHMARK_BASE", base)
    paths = resolve_benchmark("test_domain")
    assert [p.name for p in paths] == ["t1.json"]
    captured = capsys.readouterr()
    assert "filtered 1 draft task" in captured.err
    assert "t2" in captured.err


def test_audit_gate_excludes_manifest_from_task_globbing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_review.json` shouldn't be tried as a task even though it's a .json file."""
    base = tmp_path / "benchmarks"
    domain = base / "test_domain"
    domain.mkdir(parents=True)
    (domain / "t1.json").write_text(
        Task(id="t1", domain="test_domain", input="x").model_dump_json()
    )
    (domain / "_review.json").write_text(
        BenchmarkAuditManifest(
            review_status="complete",
            reviewed_tasks=["t1"],
        ).model_dump_json()
    )
    monkeypatch.setattr("steadfast.cli._BENCHMARK_BASE", base)
    paths = resolve_benchmark("test_domain")
    # _review.json is filtered out by the convention exclusion.
    assert all(not p.name.startswith("_") for p in paths)


@pytest.mark.parametrize(
    "domain_dir_name",
    ["customer_support", "code_repair", "multi_hop_research"],
)
def test_every_committed_task_loads_as_valid_task(domain_dir_name: str) -> None:
    """Schema-loading test: every JSON in every domain (reviewed or draft) is a valid Task."""
    base = Path(__file__).resolve().parents[1] / "benchmarks" / domain_dir_name
    # Includes draft tasks, since the audit-gate filter is a CLI-level concern,
    # not a schema concern.
    task_paths = [
        p
        for p in sorted(base.glob("*.json"))
        if not p.name.startswith("_")
        and not p.name.endswith(".draft.json")
        and not p.name.startswith("distractors_v")
        and not p.name.startswith("cases_v")
    ]
    assert task_paths, f"no task files found under {base}"
    for path in task_paths:
        task = Task.model_validate_json(path.read_text(encoding="utf-8"))
        assert task.id, f"empty task id in {path}"
        assert task.domain == domain_dir_name, (
            f"task {task.id} in {path}: domain={task.domain!r} but directory is "
            f"{domain_dir_name!r}"
        )
        # Judge dispatch contract: rubric/safety_harmful tasks must have a rubric ground_truth;
        # exact_match must have an exact ground_truth.
        if task.judge == "rubric":
            assert task.ground_truth is not None
            assert task.ground_truth.kind == "rubric"
        elif task.judge == "exact_match":
            assert task.ground_truth is not None
            assert task.ground_truth.kind == "exact"


@pytest.mark.parametrize(
    ("domain_dir_name", "expected_min_hard_fraction"),
    [
        ("customer_support", 0.10),  # METHODOLOGY §3.4 floor
        ("code_repair", 0.10),
        ("multi_hop_research", 0.10),
    ],
)
def test_difficulty_distribution_meets_floor(
    domain_dir_name: str, expected_min_hard_fraction: float
) -> None:
    """METHODOLOGY §3.4: each benchmark suite has ≥10% hard tasks."""
    base = Path(__file__).resolve().parents[1] / "benchmarks" / domain_dir_name
    task_paths = [
        p
        for p in sorted(base.glob("*.json"))
        if not p.name.startswith("_")
        and not p.name.endswith(".draft.json")
        and not p.name.startswith("distractors_v")
        and not p.name.startswith("cases_v")
    ]
    tasks = [Task.model_validate_json(p.read_text()) for p in task_paths]
    hard = sum(1 for t in tasks if t.difficulty == "hard")
    fraction = hard / len(tasks)
    assert fraction >= expected_min_hard_fraction, (
        f"{domain_dir_name}: {hard}/{len(tasks)} = {fraction:.2%} hard, "
        f"below METHODOLOGY §3.4 floor of {expected_min_hard_fraction:.0%}"
    )


def test_benchmark_all_resolves_across_three_domains(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--benchmark all` returns every reviewed task across the three task domains."""
    paths = resolve_benchmark("all")
    # 17 + 17 + 17 = 51 reviewed tasks at end-state.
    assert len(paths) == 51
    # All three task domains present in the path set.
    parents = {p.parent.name for p in paths}
    assert parents == {"customer_support", "code_repair", "multi_hop_research"}
    # Safety is excluded (has its own dispatch path).
    assert "safety" not in parents
    # Stderr surfaces the resolution summary.
    captured = capsys.readouterr()
    assert "resolved 51 task(s) across 3 domain(s)" in captured.err


def test_benchmark_all_skips_safety_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Safety directory is silently skipped because it uses a bank-file dispatch."""
    base = tmp_path / "benchmarks"
    # One regular task domain
    domain = base / "test_domain"
    domain.mkdir(parents=True)
    (domain / "t1.json").write_text(
        Task(id="t1", domain="test_domain", input="x").model_dump_json()
    )
    (domain / "_review.json").write_text(
        BenchmarkAuditManifest(
            review_status="complete", reviewed_tasks=["t1"]
        ).model_dump_json()
    )
    # Plus a `safety` directory with a bank file (would crash `_read_task_id`
    # if not skipped — `cases_v1.json` has no `id` field).
    safety = base / "safety"
    safety.mkdir()
    (safety / "cases_v1.json").write_text(
        '{"version":"v1","review_status":"reviewed","cases":[]}'
    )
    monkeypatch.setattr("steadfast.cli._BENCHMARK_BASE", base)
    paths = resolve_benchmark("all")
    # Only test_domain's task; safety skipped silently.
    assert [p.name for p in paths] == ["t1.json"]


def test_benchmark_all_raises_when_no_audited_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every domain drafted → `--benchmark all` fails loud rather than returning empty."""
    base = tmp_path / "benchmarks"
    domain = base / "test_domain"
    domain.mkdir(parents=True)
    (domain / "t1.json").write_text(
        Task(id="t1", domain="test_domain", input="x").model_dump_json()
    )
    (domain / "_review.json").write_text(
        BenchmarkAuditManifest(
            review_status="draft", reviewed_tasks=[], draft_tasks=["t1"]
        ).model_dump_json()
    )
    monkeypatch.setattr("steadfast.cli._BENCHMARK_BASE", base)
    import typer

    with pytest.raises(typer.BadParameter, match="resolved to zero tasks"):
        resolve_benchmark("all")


def test_audit_manifest_review_status_values() -> None:
    """The review_status literal enforces the three-state lifecycle."""
    # Valid values
    BenchmarkAuditManifest(review_status="draft")
    BenchmarkAuditManifest(review_status="partial")
    BenchmarkAuditManifest(review_status="complete")
    # Invalid value rejected at validation time
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BenchmarkAuditManifest.model_validate({"review_status": "nope"})
